"""Read the indexed archive and resolve its earlier-layer references."""
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from compression import zstd
import io
import struct
import numpy as np
from gguf_tools import GGUF
from paired_weights import decode_tensor
from ternary_codec import FIXED, INDEX, MAGIC as V1_MAGIC

HEADER = struct.Struct("<8sII")

def down_name(up):
    return up.removesuffix("ffn_up.weight") + "ffn_down.weight"

def decompress(frame, limit):
    decoder = zstd.ZstdDecompressor()
    result = decoder.decompress(frame, max_length=limit + 1)
    if len(result) > limit or not decoder.eof or decoder.unused_data:
        raise ValueError("Compressed record does not fit the declared layout")
    return result

class PairImage:
    """Expose canonical records on demand, caching a single transposed down projection."""

    def __init__(self, storage):
        self.storage = storage
        magic, count, size = HEADER.unpack_from(storage)
        start = HEADER.size + 4 * count
        if magic != b"QTile001" or count == 0 or start + size > len(storage):
            raise ValueError("Invalid paired-weight directory")
        lengths = np.frombuffer(storage, "<u4", count=count, offset=HEADER.size)
        self.physical = [start + size, *(start + size + lengths.cumsum(dtype=np.int64)).tolist()]
        if not np.all(lengths) or self.physical[-1] != len(storage):
            raise ValueError("Paired-weight records do not cover the archive")
        self.prefix = decompress(storage[start:start + size], 64 << 20)
        tag, self.original_bytes, header_size, n = FIXED.unpack_from(self.prefix)
        self.header_start = FIXED.size + INDEX.size * n
        if tag != V1_MAGIC or count != n or self.header_start + header_size != len(self.prefix):
            raise ValueError("Invalid canonical ternary header")
        self.g = GGUF(io.BytesIO(self.prefix[self.header_start:]))
        self.names = sorted(self.g.tensors, key=lambda k: self.g.tensors[k][2])
        if len(self.names) != count:
            raise ValueError("Tensor counts differ")
        self.by_name = {k: i for i, k in enumerate(self.names)}
        self.spans = {}
        self.virtual = [len(self.prefix)]
        for i, name in enumerate(self.names):
            off, length = INDEX.unpack_from(self.prefix, FIXED.size + i * INDEX.size)
            if off != self.virtual[-1] or length == 0:
                raise ValueError("Invalid canonical tensor extent")
            self.spans[name] = off, length
            self.virtual.append(off + length)
        self.reference_name = self.reference = None
        self.vocab_plan = None
        self.row_parents={}
        self.row_cache={}
        if magic == b"QTile001":
            from row_options import ROW_HEADER
            from predictor import CONFIG
            from paired_weights import HEADER as WEIGHT_HEADER
            for i,name in enumerate(self.names):
                pos=self.physical[i]
                if self.g.tensors[name][1]!=42 or storage[pos] not in range(128,132):
                    continue
                start=pos+WEIGHT_HEADER.size
                _,codec,nopts=CONFIG.unpack_from(storage,start)
                if codec not in (3,4): continue
                if nopts<ROW_HEADER.size: raise ValueError("Truncated row reference")
                distance=ROW_HEADER.unpack_from(storage,start+CONFIG.size)[0]
                prefix,layer,kind=name.split(".",2)
                prior=f"{prefix}.{int(layer)-distance}.{kind}"
                if not distance or prior not in self.by_name or self.g.tensors[prior][:2]!=self.g.tensors[name][:2]:
                    raise ValueError("Invalid earlier-layer row reference")
                self.row_parents[name]=prior
        self.row_sources=set(self.row_parents.values())

    def vocab_table(self, name, *, file=None, cache_pages=0, cache_rows=0):
        from vocab_rows import VocabPlan, VocabTable

        if name not in ("output.weight", "token_embd.weight"):
            raise ValueError("Expected a vocabulary tensor")
        if self.vocab_plan is None:
            self.vocab_plan = VocabPlan(self.g)
        i = self.by_name[name]
        start, end = self.physical[i:i + 2]
        return VocabTable(self.vocab_plan, self.g.tensors[name][0][0], memoryview(self.storage)[start:end],
                          file=file, file_offset=start if file is not None else 0,
                          cache_pages=cache_pages, cache_rows=cache_rows)

    def __len__(self):
        return self.virtual[-1]

    def canonical_directory(self):
        return self.g, self.spans, (self.header_start, len(self.prefix) - self.header_start), self.original_bytes

    def read_range(self, start, end):
        if not 0 <= start <= end <= len(self):
            raise ValueError("Read outside the virtual ternary image")
        if end <= len(self.prefix):
            return self.prefix[start:end]
        pieces = []
        if start < len(self.prefix):
            pieces.append(self.prefix[start:])
            start = len(self.prefix)
        while start < end:
            i = bisect_right(self.virtual, start) - 1
            name = self.names[i]
            stop = min(end, self.virtual[i + 1])
            value = self.read_record(name, *self.spans[name])
            pieces.append(memoryview(value)[start - self.virtual[i]:stop - self.virtual[i]])
            start = stop
        return b"".join(pieces)

    def __getitem__(self, key):
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("Model reads require contiguous slices")
        start, end, _ = key.indices(len(self))
        return self.read_range(start, max(start, end))

    def read_record(self, name, off, size):
        return self._record(name,off,size,None)

    def native_record(self, name, off, size, stride):
        return self._record(name,off,size,stride)

    def _record(self, name, off, size, stride):
        if self.spans[name] != (off, size):
            raise ValueError("Tensor directory mismatch")
        i = self.by_name[name]
        data = memoryview(self.storage)[self.physical[i]:self.physical[i + 1]]
        dims, kind, _ = self.g.tensors[name]
        if kind == 0:
            result = decompress(data, size)
        elif kind == 42 and data[0] == 64:
            table = self.vocab_table(name)
            result = table.canonical() if stride is None else table.native_blocks(stride)
        elif kind == 42 and len(dims) == 2:
            ref = None
            if data[0] in (1, 2, 129, 130):
                if not name.endswith(".ffn_up.weight"):
                    raise ValueError("Only an up projection can refer to a down projection")
                target = down_name(name)
                if self.reference_name != target:
                    self.read_record(target, *self.spans[target])
                ref = self.reference
                if ref is None or ref.shape != (dims[1], dims[0]):
                    raise ValueError("Missing matching transposed down projection")
            up = name.removesuffix("ffn_down.weight") + "ffn_up.weight"
            keep = (name.endswith(".ffn_down.weight") and up in self.by_name and
                    self.storage[self.physical[self.by_name[up]]] in (1, 2, 129, 130))
            previous=None
            parent=self.row_parents.get(name)
            if parent is not None:
                if parent not in self.row_cache:
                    self.read_record(parent,*self.spans[parent])
                previous=self.row_cache.pop(parent)
            captured=np.empty((dims[1],dims[0]),np.uint8) if name in self.row_sources else None
            with ThreadPoolExecutor(max_workers=4) as pool:
                result, transpose = decode_tensor(data, dims[1], dims[0], ref, keep, pool, stride,
                                                  previous_rows=previous,capture_rows=captured)
            if captured is not None:
                self.row_cache[name]=captured
            if keep:
                self.reference_name, self.reference = name, transpose
        else:
            raise ValueError("Unsupported tensor in the paired model")
        expected = size if stride is None else (dims[0]*dims[1]//128)*stride
        if (len(result) if stride is None else result.nbytes) != expected:
            raise ValueError("Incorrect reconstructed tensor size")
        return result if stride is None else result.reshape(-1)
