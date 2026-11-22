"""Run Bonsai 2 9B with its original PTQ1_0 matrices and file-backed input embeddings."""
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import ctypes as C
import math
import os
import struct

import numpy as np

from archive_io import checked_file
from bonsai9 import NineArchive
from versions9 import ARCHIVE_SIZE, archive_identity
from gguf_tools import metadata_array
from load_kernels import native
from model_stream import PackedStream
from native_engine import NativeEngine, NativeTokenizer, bind
from native_paths import backend_directory
from paired_archive import decompress
from paired_weights import CONTEXTS, HEADER, row_pages
from predictor import CONFIG
from residual_scales import restore_meta
from vocab_rows import scale_bits, scale_lut


class InputRows:
    """Index the existing rANS states once, then decode one requested row with pread."""
    def __init__(self, archive, file, cache_rows=128):
        name = "token_embd.weight"
        self.width, self.rows = archive.g.tensors[name][0]
        widths = metadata_array(archive.g, "prism.hadamard.sign_widths")
        offset = sum(widths[:widths.index(self.width)])
        signs = metadata_array(archive.g, "prism.hadamard.sign_values")
        self.signs = np.array(signs[offset:offset + self.width], "<u4").view("<i4").astype(np.float32)
        if archive.g.metadata["prism.hadamard.block_size"] != self.width or np.any(np.abs(self.signs) != 1):
            raise ValueError("Unsupported signed embedding transform")
        self.normalization = 1 / math.sqrt(self.width)
        index = archive.by_name[name]
        start, end = archive.physical[index:index + 2]
        encoded = archive.page_directory(index)
        if encoded[0] != 1:
            raise ValueError("Expected an entropy-coded embedding")
        record = encoded[1:]
        mode, length, count = HEADER.unpack_from(record)
        meta = record[HEADER.size:HEADER.size + length]
        groups = self.rows * self.width // 128
        fixed = 12 + 2 * groups + self.width + self.rows + 4 * CONTEXTS[0]
        if mode == 128:
            transformed, codec, options = CONFIG.unpack_from(meta)
            if codec or options or transformed not in (0, 1):
                raise ValueError("The embedding requires independent rows")
            frame = meta[CONFIG.size:]
            raw = restore_meta(frame, self.rows, self.width, 0) if transformed else decompress(frame, fixed + groups)
        elif mode == 0:
            raw = decompress(meta, fixed + groups)
        else:
            raise ValueError("Unknown input embedding format")
        escapes, = struct.unpack_from("<I", raw)
        if escapes > groups or len(raw) != fixed + escapes:
            raise ValueError("Invalid embedding scale extent")
        parts = np.split(np.frombuffer(raw, np.uint8, offset=4),
                         np.cumsum([8, groups, groups, escapes, self.width, self.rows]))
        self.bits = scale_bits(parts[:4], self.rows)
        self.lut = scale_lut(self.bits)
        self.col, self.row = parts[4].copy(), parts[5].copy()
        self.base = np.frombuffer(parts[6].tobytes(), "<u2").reshape(-1, 2)
        pages = row_pages(self.rows, self.width)
        if len(pages) != count:
            raise ValueError("Invalid embedding page count")
        self.rows_per_page = pages[0].stop
        lengths = np.frombuffer(record, "<u4", count, HEADER.size + length)
        first = start + 1 + HEADER.size + length + 4 * count
        self.edges = np.r_[first, first + lengths.cumsum(dtype=np.int64)]
        if self.edges[-1] != end:
            raise ValueError("Embedding pages do not cover their tensor")
        self.checks = np.empty((self.rows + count, 5), "<u4")

        def scan(item):
            page, scope = item
            begin, end = map(int, self.edges[page:page + 2])
            payload = np.frombuffer(archive.span(index, begin - start, end - start), np.uint8)
            keys = self.keys(scope)
            states = self.checks[scope.start + page:scope.stop + page + 1]
            if native().index_stream(payload.ctypes.data, payload.size, keys.ctypes.data, len(keys),
                                     self.width, self.base.ctypes.data, len(self.base), states.ctypes.data):
                raise ValueError("Could not index embedding restart states")

        with archive.page_pool(4) as pool:
            for _ in pool.map(scan, enumerate(pages), buffersize=archive.page_workers):
                pass
        self.hot = OrderedDict()
        self.hot_rows = np.empty((cache_rows, self.width), np.float16)
        self.file = os.fdopen(os.dup(file.fileno()), "rb")

    def keys(self, scope):
        classes = self.lut[self.bits[scope] & 0x7fff]
        groups = (self.row[scope, None].astype(np.uint16) * 4 + classes) * 16
        return np.repeat(groups, 128, axis=1) + self.col

    def owned_bytes(self):
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, np.ndarray))

    def reset_cache(self):
        self.hot.clear()

    def embedding_into(self, token, out):
        if not 0 <= token < self.rows:
            raise IndexError(token)
        if token in self.hot:
            self.hot.move_to_end(token)
            slot = self.hot[token]
        else:
            page = token // self.rows_per_page
            pair = self.checks[token + page:token + page + 2]
            start, end = map(int, pair[:, 0])
            payload = os.pread(self.file.fileno(), end - start, int(self.edges[page]) + start)
            raw = np.frombuffer(payload, np.uint8)
            keys = self.keys(slice(token, token + 1))
            trits = np.empty(self.width, np.uint8)
            if len(payload) != end - start or native().seek_row(
                    raw.ctypes.data, raw.size, pair.ctypes.data, keys.ctypes.data,
                    self.width, self.base.ctypes.data, len(self.base), trits.ctypes.data):
                raise ValueError("The embedding row did not reach its recorded end state")
            slot = len(self.hot) if len(self.hot) < len(self.hot_rows) else self.hot.popitem(last=False)[1]
            self.hot[token] = slot
            np.multiply(trits.reshape(-1, 128).astype(np.int8) - 1, self.bits[token].view("<f2")[:, None],
                        out=self.hot_rows[slot].reshape(-1, 128))
        out[:] = self.hot_rows[slot]
        if native().inverse_embedding(out.ctypes.data, self.width, self.signs.ctypes.data, self.normalization):
            raise ValueError("Invalid inverse Hadamard transform")

    def embedding_batch(self, ids):
        result = np.empty((len(ids), self.width), np.float32)
        for i, token in enumerate(ids):
            self.embedding_into(int(token), result[i])
        return result

    def close(self):
        self.file.close()


def tied_header(gguf):
    """Keep Prism's output rotation when serving the input rows outside the native model."""
    if (gguf.metadata.get("prism.hadamard.version") != 1
            or metadata_array(gguf, "prism.hadamard.inverse_weight_names") != ["token_embd.weight"]):
        raise ValueError("Unsupported publisher Hadamard layout")
    header = bytearray(gguf.kv)
    struct.pack_into("<I", header, gguf.positions["prism.hadamard.version"][1], 2)
    count, = struct.unpack_from("<Q", header, 16)
    struct.pack_into("<Q", header, 16, count + 1)
    names = metadata_array(gguf, "prism.hadamard.weight_names")
    if names.count("output.weight") != 1:
        raise ValueError("Expected a single rotated output projection")

    def strings(values):
        return struct.pack("<IQ", 8, len(values)) + b"".join(
            struct.pack("<Q", len(name)) + name.encode("ascii") for name in values)

    start = gguf.positions["prism.hadamard.weight_names"][1]
    old = strings(names)
    if header[start:start + len(old)] != old:
        raise ValueError("Unexpected Hadamard metadata encoding")
    header[start:start + len(old)] = strings([n for n in names if n != "output.weight"])
    key = b"prism.hadamard.tied_output"
    header.extend(struct.pack("<Q", len(key)) + key + struct.pack("<IB", 7, 1))
    return header


class PagedStream:
    """Expose the original PTQ1_0 compute matrices to the pinned Metal backend."""
    repeated = frozenset(("ssm_alpha.weight", "ssm_norm.weight", "ssm_out.weight"))

    def __init__(self, archive, fast):
        self.archive = archive
        self.strides = {}
        hot = {f"blk.{layer}.ffn_{kind}.weight"
               for kind, count in (("up", 32), ("down", 32), ("gate", 22))
               for layer in range(count)} if fast else set()
        if not hot.issubset(archive.by_name):
            raise ValueError("Missing projection in the 9B execution layout")
        self.indices = [i for i, name in enumerate(archive.names) if name != "token_embd.weight"]
        header = tied_header(archive.g)
        struct.pack_into("<Q", header, 8, len(self.indices))
        offsets = [0]
        for i in self.indices:
            name = archive.names[i]
            dims, kind, _ = archive.g.tensors[name]
            extent = archive.logical[i + 1] - archive.logical[i]
            if name in hot:
                if kind != 143 or len(dims) != 2 or dims[0] % 128:
                    raise ValueError("Unexpected type or shape for a 9B ternary projection")
                kind = 142
                extent += math.prod(dims) // 128 * 6
                self.strides[i] = 34
            label = ("token_embd.weight" if name == "output.weight" else name).encode("ascii")
            header.extend(struct.pack("<Q", len(label)) + label + struct.pack("<I", len(dims)))
            header.extend(struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", kind, offsets[-1]))
            offsets.append(offsets[-1] + extent)
        header.extend(bytes(-len(header) % archive.g.alignment))
        self.header = np.frombuffer(header, np.uint8)
        self.edges = [len(header) + offset for offset in offsets]
        self.length = self.edges[-1]
        self.position = 0
        self.index = self.array = None
        self.layer = None
        self.pending = {}

    @contextmanager
    def open_c_file(self):
        try:
            with PackedStream.open_c_file(self) as stream:
                yield stream
        finally:
            self.index = self.array = None
            self.pending.clear()
            self.archive.cache.clear()

    def read_into(self, pointer, requested):
        count = PackedStream.read_into(self, pointer, requested)
        if self.index is not None and self.position == self.edges[self.index + 1]:
            self.index = self.array = None
        return count

    def buffer_at(self, pos):
        if not 0 <= pos < self.length:
            raise ValueError("Position outside the paged GGUF")
        if pos < self.edges[0]:
            return self.header, pos
        index = bisect_right(self.edges, pos) - 1
        if index != self.index:
            self.index = self.array = None
            tensor_index = self.indices[index]
            name = self.archive.names[tensor_index]
            layer, _, family = name.removeprefix("blk.").partition(".")
            if layer != self.layer:
                self.pending.clear()
                self.layer = layer
            self.array = self.pending.pop(tensor_index, None)
            if self.array is None:
                self.array = self.archive.record(tensor_index, self.strides.get(tensor_index, 28))
                if family in self.repeated and name not in self.archive.sources and name not in self.archive.parents:
                    self.pending[tensor_index] = self.array
            if self.array.nbytes != self.edges[index + 1] - self.edges[index]:
                raise ValueError("Incorrect native tensor extent")
            self.index = index
        return self.array, pos - self.edges[index]


class DirectStream(PagedStream):
    """Expose only GGUF metadata through stdio. The native loader owns the weight destinations."""
    def __init__(self, archive, fast, observe):
        super().__init__(archive, fast)
        self.observe = observe

    def read_into(self, pointer, requested):
        count = max(0, min(requested, len(self.header) - self.position))
        if count:
            C.memmove(pointer, self.header.ctypes.data + self.position, count)
            self.position += count
        return count

    def load_native(self, llama, stream, settings):
        from direct_weights9 import load_direct
        return load_direct(self, llama, stream, settings)

def active_metal_device():
    lib = C.CDLL(str(backend_directory() / "libggml.0.dylib"))
    bind(lib, "ggml_backend_load_all", None)()
    device = bind(lib, "ggml_backend_dev_by_name", C.c_void_p, C.c_char_p)(b"MTL0")
    if not device:
        raise RuntimeError("Apple Metal is unavailable to this process")
    return lib, device


def hadamard_bytes(gguf, llama, fast, lib, device):
    """Ask the active Metal allocator for each rotation/sign buffer, including alignment."""
    ptr, size = C.c_void_p, C.c_size_t

    class Init(C.Structure):
        _fields_ = [("mem_size", size), ("mem_buffer", ptr), ("no_alloc", C.c_bool)]

    buffer_type = bind(lib, "ggml_backend_dev_buffer_type", ptr, ptr)(device)
    overhead = bind(lib, "ggml_tensor_overhead", size)()
    initialize = bind(lib, "ggml_init", ptr, Init)
    tensor = bind(lib, "ggml_new_tensor_2d", ptr, ptr, C.c_int, C.c_int64, C.c_int64)
    allocation = bind(lib, "ggml_backend_alloc_ctx_tensors_from_buft_size", size, ptr, ptr)
    release = bind(lib, "ggml_free", None, ptr)
    widths = metadata_array(gguf, "prism.hadamard.sign_widths")
    block = gguf.metadata["prism.hadamard.block_size"]
    rotation_type = bind(llama, "ternzip9_rotation_format", C.c_int, C.c_uint32)(block) if fast else 0
    storage_bytes = bind(lib, "ggml_nbytes", size, ptr)
    total = 0
    rotation_bytes = 0
    for x, y in [(block, block), *((w, 1) for w in widths)]:
        ctx = initialize(Init(overhead, None, True))
        if not ctx:
            raise RuntimeError("Cannot size auxiliary Metal tensors")
        try:
            descriptor = tensor(ctx, rotation_type if y == block else 0, x, y)
            if not descriptor:
                raise RuntimeError("Cannot size a Metal weight buffer")
            measured = allocation(ctx, buffer_type)
            if y == block:
                rotation_bytes = measured
            if measured < storage_bytes(descriptor):
                raise ValueError("Unexpected Metal weight allocation")
            total += measured
        finally:
            release(ctx)
    return total, (1, len(widths)), (block * block, rotation_bytes)


def load(path, context, *, max_outputs=1, observe=None, page_workers=None):
    from chat_model import DECODER_RESERVE, WEIGHT_BUDGET
    size, sha = archive_identity(path)
    fast = size == ARCHIVE_SIZE
    C.CDLL(str(backend_directory() / "libggml-metal.0.dylib"))
    llama = C.CDLL(str(backend_directory() / "libllama.0.dylib"))
    if fast:
        if not hasattr(llama, "ternzip9_reuse_config"):
            raise ValueError("This 9B archive requires the Ternzip Metal backend with compact rotation support.")
        bind(llama, "ternzip9_reuse_config", None, C.c_uint)(8)
    with checked_file(path, size, sha) as file, ExitStack() as resources:
        ggml, device = active_metal_device()
        if page_workers is None:
            name = bind(ggml, "ggml_backend_dev_description", C.c_char_p, C.c_void_p)(device)
            direct = fast and hasattr(llama, "ternzip9_load_with_writer")
            page_workers = 14 if direct and name == b"Apple M4 Pro" else 10
        pool = resources.enter_context(ThreadPoolExecutor(max_workers=page_workers))
        archive = NineArchive.from_file(file, pool)
        archive.page_workers = page_workers
        rotation_bytes, counts, rotation = hadamard_bytes(archive.g, llama, fast, ggml, device)
        table = InputRows(archive, file)
        try:
            image = (DirectStream(archive, fast, observe) if fast and hasattr(llama, "ternzip9_load_with_writer")
                     else PagedStream(archive, fast))
            engine = NativeEngine(image, context, max_outputs=max_outputs, prefill_chunk=256,
                                  backend_greedy=max_outputs == 1, input_table=table, text_position_axes=4)
            engine.used_direct_writer = isinstance(image, DirectStream)
            engine.page_workers = page_workers
        except BaseException:
            table.close()
            raise
        try:
            if engine.hadamard_buffer_counts != counts or not engine.weight_buffers_mib:
                raise ValueError("The backend did not account for the expected native weights")
            if fast and engine.rotation_readback != rotation:
                raise ValueError("The packed Metal rotation failed readback verification")
            aligned = sum(math.ceil((mib + 0.005) * 1048576) for _, mib in engine.weight_buffers_mib)
            weight_bytes = max(engine.weight_bytes, aligned) + rotation_bytes + table.owned_bytes() + DECODER_RESERVE
            if weight_bytes > WEIGHT_BUDGET:
                raise ValueError("9B weight buffers exceeded the configured memory budget")
            tokenizer = NativeTokenizer(engine, archive.g.metadata)
        except BaseException:
            engine.close()
            raise
    return engine, tokenizer, weight_bytes
