"""Decode count-conditioned ternary pages into the original scale-and-weight blocks."""
import ctypes as C
from functools import lru_cache
import struct

import numpy as np

from load_kernels import pack_native
from native_paths import library
from paired_archive import decompress
from paired_weights import CONTEXTS, HEADER as WH, row_pages
from predictor import CONFIG
from residual_scales import restore_meta
from vocab_rows import keys, scale_bits, scale_lut


@lru_cache(maxsize=1)
def kernel():
    lib = C.CDLL(str(library("quota-rans")))
    ptr, size = C.c_void_p, C.c_size_t
    lib.quota_decode.argtypes = [ptr, size, ptr, size, ptr, size, ptr, ptr, ptr]
    lib.quota_decode.restype = C.c_int
    return lib


@lru_cache(maxsize=1)
def grid():
    # Adjust the remaining fraction of zeros by the context's learned odds.
    # The reference odds are 42 zeros to 86 nonzeros in a 128-weight block.
    bucket, slots, zeros = np.ogrid[:65, :129, :129]
    numerator = zeros * bucket * 86
    denominator = numerator + np.maximum(0, slots - zeros) * (64 - bucket) * 42
    z = np.clip((4096 * numerator + denominator // 2) // np.maximum(1, denominator), 1, 4094)
    return np.ascontiguousarray(np.where(zeros == 0, 0, np.where(zeros == slots, 4096, z)), "<u2")


def model_table(bounds):
    zero, neg = np.asarray(bounds, dtype=np.int64).T
    if np.any((zero < 1) | (neg <= zero) | (neg >= 4096)):
        raise ValueError("Invalid base intervals")
    bucket = np.clip((zero * 64 + 2048) // 4096, 1, 63)
    sign = ((neg - zero) * 65536 + (4096 - zero) // 2) // (4096 - zero)
    return np.ascontiguousarray(np.column_stack((bucket * 129 * 129, sign)), "<u4")


def decode_page(page, sites, model):
    size, = struct.unpack_from("<I", page)
    if size + 20 > len(page):
        raise ValueError("Invalid count frame extent")
    counts = np.frombuffer(decompress(page[4:4+size], sites.size // 128), np.uint8)
    if len(counts) * 128 != sites.size:
        raise ValueError("Wrong block count")
    payload = np.frombuffer(page, np.uint8, offset=4+size)
    out = np.empty(sites.shape, np.uint8)
    if kernel().quota_decode(payload.ctypes.data, payload.size, sites.ctypes.data, sites.size, model.ctypes.data,
                             len(model), grid().ctypes.data, counts.ctypes.data, out.ctypes.data):
        raise ValueError("Invalid quota decoder state or unused bytes")
    return out


class Matrix:
    def __init__(self, stream, i, storage, record_size=None):
        self.name = stream.names[i]
        (self.width, self.rows), kind, _ = stream.g.tensors[self.name]
        end = len(storage) if record_size is None else record_size
        self.groups_per_row = self.width // 128
        groups = self.groups_per_row * self.rows
        self.pad = stream.logical[i+1] - stream.logical[i] - groups * 28
        if kind != 143 or self.width % 128 or not 0 <= self.pad < stream.g.alignment:
            raise ValueError("Unexpected native weight extent")
        mode, frame_size, page_count = WH.unpack_from(storage, 1)
        frame_at = 1 + WH.size
        transformed, codec, option_bytes = CONFIG.unpack_from(storage, frame_at)
        if mode != 128 or (codec, option_bytes) != (0,0) or transformed not in (0,1):
            raise ValueError("Expected independent ternary sites")
        frame = storage[frame_at+CONFIG.size:frame_at+frame_size]
        fixed = 12 + 2 * groups + self.width + self.rows + 4 * CONTEXTS[0]
        raw = restore_meta(frame, self.rows, self.width, 0) if transformed else decompress(frame, fixed+groups)
        escapes, = struct.unpack_from("<I", raw)
        if escapes > groups or len(raw) != fixed + escapes:
            raise ValueError("Scale metadata length")
        pal, rank, low, over, self.col, self.row, bounds = np.split(
            np.frombuffer(raw, np.uint8, offset=4), np.cumsum([8,groups,groups,escapes,self.width,self.rows]))
        if self.col.max() > 15 or self.row.max() > 3:
            raise ValueError("Invalid axis classes")
        self.bits = scale_bits((pal, rank, low, over), self.rows)
        self.lut = scale_lut(self.bits)
        self.model = model_table(bounds.view("<u2").reshape(-1,2))
        self.pages = row_pages(self.rows, self.width)
        if len(self.pages) != page_count:
            raise ValueError("Trit page count")
        lengths_at = frame_at + frame_size
        lengths = np.frombuffer(storage, "<u4", page_count, lengths_at)
        first = lengths_at + 4 * page_count
        self.edges = [first, *(first + lengths.cumsum(dtype=np.int64)).tolist()]
        if self.edges[-1] + self.pad != end:
            raise ValueError("Pages exceed tensor record")
        self.tail = storage[end-self.pad:end] if record_size is None else None

    def sites(self, part):
        return keys(part, self.row, self.col, self.bits, self.lut)

    def blocks(self, part):
        return slice(part.start*self.groups_per_row, part.stop*self.groups_per_row)



def record(archive, index, stride, data, *, destination=None, read_page=None, record_size=None):
    matrix = Matrix(archive, index, data, record_size)
    # lru_cache can execute a first call on multiple threads. Build the shared table once.
    grid()
    shape = (matrix.rows * matrix.groups_per_row, stride)
    dest = np.empty(shape, np.uint8) if destination is None else destination.reshape(shape)
    retained = (np.empty((matrix.rows, matrix.width), np.uint8)
                if destination is None and matrix.name in archive.sources else None)

    def decode(item):
        page_index, rows = item
        start, end = matrix.edges[page_index:page_index + 2]
        page = data[start:end] if read_page is None else read_page(start, end)
        trits = decode_page(page, matrix.sites(rows), matrix.model)
        pack_native(trits, matrix.bits[rows], stride, dest[matrix.blocks(rows)])
        if retained is not None:
            retained[rows] = trits

    with archive.page_pool(8) as pool:
        for _ in pool.map(decode, enumerate(matrix.pages), buffersize=archive.page_workers):
            pass
    if retained is not None:
        archive.cache[matrix.name] = retained
    if matrix.pad and destination is None:
        return np.concatenate((dest.ravel(), np.frombuffer(matrix.tail, np.uint8)))
    return dest.ravel()
