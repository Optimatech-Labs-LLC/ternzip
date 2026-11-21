"""Decode scalar rANS or pairs of four-trit words with per-weight probabilities."""
import ctypes as C
from functools import lru_cache
import numpy as np
from native_paths import library
from paired_weights import native


def pack_field(trits, bounds):
    values = np.ascontiguousarray(trits, np.uint8)
    field = np.ascontiguousarray(bounds, "<u2")
    if field.shape != (*values.shape, 2):
        raise ValueError("The target and probability field have different shapes")
    out = np.empty(values.size * 2 + 16, np.uint8)
    start = native().trit_encode(values.ctypes.data, None, values.size, field.ctypes.data,
                                 values.size, out.ctypes.data, out.size)
    if start < 0:
        raise ValueError("Invalid cumulative probability field")
    return out[start:].tobytes()


@lru_cache(maxsize=1)
def word_decoder():
    lib = C.CDLL(str(library("word-rans")))
    pointer, size = C.c_void_p, C.c_size_t
    lib.word_decode.argtypes = [pointer, size, pointer, pointer, pointer, size]
    lib.word_decode.restype = C.c_int
    return lib.word_decode


def unpack_field(payload, bounds, factors=None):
    data = np.frombuffer(payload, np.uint8)
    field = np.ascontiguousarray(bounds, "<u2")
    out = np.empty(field.shape[:-1], np.uint8)
    if factors is None:
        error = native().trit_decode(data.ctypes.data, data.size, None, out.size,
                                     field.ctypes.data, out.size, out.ctypes.data)
    else:
        odds = np.ascontiguousarray(factors, "<u2").reshape(5)
        error = word_decoder()(data.ctypes.data, data.size, field.ctypes.data,
                               odds.ctypes.data, out.ctypes.data, out.size)
    if error:
        raise ValueError("Invalid entropy stream")
    return out
