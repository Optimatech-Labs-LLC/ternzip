"""Use one exact pair of cumulative rANS bounds per weight."""
import numpy as np
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


def unpack_field(payload, bounds):
    data = np.frombuffer(payload, np.uint8)
    field = np.ascontiguousarray(bounds, "<u2")
    out = np.empty(field.shape[:-1], np.uint8)
    if native().trit_decode(data.ctypes.data, data.size, None, out.size,
                            field.ctypes.data, out.size, out.ctypes.data):
        raise ValueError("Invalid entropy stream")
    return out
