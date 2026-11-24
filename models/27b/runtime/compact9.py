import ctypes as C
from functools import lru_cache
from pathlib import Path
import struct
from compression import zstd
import numpy as np
from paired_archive import decompress
@lru_cache(maxsize=1)
def kernel():
    lib = C.CDLL(str(Path(__file__).parent / "lib/libquota-rans.dylib"))
    ptr, size = C.c_void_p, C.c_size_t
    lib.quota_encode.argtypes = [ptr, ptr, size, ptr, size, ptr, ptr, ptr, size]
    lib.quota_encode.restype = C.c_ssize_t
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


def encode_page(trits, sites, model):
    trits = np.ascontiguousarray(trits, np.uint8)
    if sites.dtype != np.uint16 or not sites.flags.c_contiguous or sites.size != trits.size or trits.size % 128:
        raise ValueError("Wrong encoder context extent")
    counts = np.empty(trits.size // 128, np.uint8)
    buffer = np.empty(trits.size * 2 + 32, np.uint8)
    start = kernel().quota_encode(trits.ctypes.data, sites.ctypes.data, trits.size, model.ctypes.data,
                                  len(model), grid().ctypes.data, counts.ctypes.data, buffer.ctypes.data, buffer.size)
    if start < 0:
        raise ValueError("Invalid quota encoder state")
    frame = zstd.compress(counts, level=9)
    return struct.pack("<I", len(frame)) + frame + buffer[start:].tobytes(), counts


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

