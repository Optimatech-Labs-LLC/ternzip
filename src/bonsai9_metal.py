"""Run Bonsai 2 9B with its original PTQ1_0 matrices and file-backed input embeddings."""
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import math
import os
import struct

import numpy as np

from archive_io import checked_input
from bonsai9 import ARCHIVE_SHA, ARCHIVE_SIZE, NineArchive
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
        if archive.storage[start] != 1:
            raise ValueError("Expected an entropy-coded embedding")
        record = memoryview(archive.storage)[start + 1:end]
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
            payload = np.frombuffer(archive.storage, np.uint8, int(lengths[page]), int(self.edges[page]))
            keys = self.keys(scope)
            states = self.checks[scope.start + page:scope.stop + page + 1]
            if native().index_stream(payload.ctypes.data, payload.size, keys.ctypes.data, len(keys),
                                     self.width, self.base.ctypes.data, len(self.base), states.ctypes.data):
                raise ValueError("Could not index embedding restart states")

        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in pool.map(scan, enumerate(pages), buffersize=4):
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
    open_c_file = PackedStream.open_c_file
    read_into = PackedStream.read_into

    def __init__(self, archive):
        self.archive = archive
        self.indices = [i for i, name in enumerate(archive.names) if name != "token_embd.weight"]
        header = tied_header(archive.g)
        struct.pack_into("<Q", header, 8, len(self.indices))
        offsets = [0]
        for i in self.indices:
            name = archive.names[i]
            dims, kind, _ = archive.g.tensors[name]
            label = ("token_embd.weight" if name == "output.weight" else name).encode("ascii")
            header.extend(struct.pack("<Q", len(label)) + label + struct.pack("<I", len(dims)))
            header.extend(struct.pack("<" + "Q" * len(dims), *dims) + struct.pack("<IQ", kind, offsets[-1]))
            offsets.append(offsets[-1] + archive.logical[i + 1] - archive.logical[i])
        header.extend(bytes(-len(header) % archive.g.alignment))
        self.header = np.frombuffer(header, np.uint8)
        self.edges = [len(header) + offset for offset in offsets]
        self.length = self.edges[-1]
        self.position = 0
        self.index = self.array = None

    def buffer_at(self, pos):
        if not 0 <= pos < self.length:
            raise ValueError("Position outside the paged GGUF")
        if pos < self.edges[0]:
            return self.header, pos
        index = bisect_right(self.edges, pos) - 1
        if index != self.index:
            self.array = self.archive.record(self.indices[index])
            if self.array.nbytes != self.edges[index + 1] - self.edges[index]:
                raise ValueError("Incorrect native tensor extent")
            self.index = index
        return self.array, pos - self.edges[index]


def hadamard_bytes(gguf):
    """Ask the active Metal allocator for each rotation/sign buffer, including alignment."""
    lib = C.CDLL(str(backend_directory() / "libggml.0.dylib"))
    ptr, size = C.c_void_p, C.c_size_t

    class Init(C.Structure):
        _fields_ = [("mem_size", size), ("mem_buffer", ptr), ("no_alloc", C.c_bool)]

    bind(lib, "ggml_backend_load_all", None)()
    device = bind(lib, "ggml_backend_dev_by_name", ptr, C.c_char_p)(b"MTL0")
    if not device:
        raise RuntimeError("Apple Metal is unavailable to this process")
    buffer_type = bind(lib, "ggml_backend_dev_buffer_type", ptr, ptr)(device)
    overhead = bind(lib, "ggml_tensor_overhead", size)()
    initialize = bind(lib, "ggml_init", ptr, Init)
    tensor = bind(lib, "ggml_new_tensor_2d", ptr, ptr, C.c_int, C.c_int64, C.c_int64)
    allocation = bind(lib, "ggml_backend_alloc_ctx_tensors_from_buft_size", size, ptr, ptr)
    release = bind(lib, "ggml_free", None, ptr)
    widths = metadata_array(gguf, "prism.hadamard.sign_widths")
    block = gguf.metadata["prism.hadamard.block_size"]
    total = 0
    for x, y in [(block, block), *((w, 1) for w in widths)]:
        ctx = initialize(Init(overhead, None, True))
        if not ctx:
            raise RuntimeError("Cannot size auxiliary Metal tensors")
        try:
            if not tensor(ctx, 0, x, y):
                raise RuntimeError("Cannot size a Metal weight buffer")
            measured = allocation(ctx, buffer_type)
            if measured < x * y * 4:
                raise ValueError("Unexpected Metal weight allocation")
            total += measured
        finally:
            release(ctx)
    return total, (1, len(widths))


def load(path, context, *, max_outputs=1):
    from chat_model import DECODER_RESERVE, WEIGHT_BUDGET
    with checked_input(path, ARCHIVE_SIZE, ARCHIVE_SHA) as (file, data):
        archive = NineArchive(data)
        rotation_bytes, counts = hadamard_bytes(archive.g)
        table = InputRows(archive, file)
        try:
            engine = NativeEngine(PagedStream(archive), context, max_outputs=max_outputs, prefill_chunk=256,
                                  backend_greedy=max_outputs == 1, input_table=table, text_position_axes=4)
        except BaseException:
            table.close()
            raise
        try:
            if engine.hadamard_buffer_counts != counts or not engine.weight_buffers_mib:
                raise ValueError("The backend did not account for the expected native weights")
            aligned = sum(math.ceil((mib + 0.005) * 1048576) for _, mib in engine.weight_buffers_mib)
            weight_bytes = max(engine.weight_bytes, aligned) + rotation_bytes + table.owned_bytes() + DECODER_RESERVE
            if weight_bytes > WEIGHT_BUDGET:
                raise ValueError("9B weight buffers exceeded the configured memory budget")
            tokenizer = NativeTokenizer(engine, archive.g.metadata)
        except BaseException:
            engine.close()
            raise
    return engine, tokenizer, weight_bytes
