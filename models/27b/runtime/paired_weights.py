import ctypes as C
from functools import lru_cache
from pathlib import Path
import numpy as np
TRITS_PER_PAGE = 1 << 20


UNIT = 4096


@lru_cache(maxsize=1)
def native():
    lib = C.CDLL(str(Path(__file__).with_name("lib") / "libtrit-rans.dylib"))
    ptr, size = C.c_void_p, C.c_size_t
    lib.trit_encode.argtypes = [ptr, ptr, size, ptr, size, ptr, size]
    lib.trit_encode.restype = C.c_ssize_t
    lib.trit_decode.argtypes = [ptr, size, ptr, size, ptr, size, ptr]
    lib.trit_decode.restype = C.c_int
    return lib


def encode_page(trits, keys, bounds):
    out = np.empty(trits.size * 2 + 16, dtype=np.uint8)
    at = native().trit_encode(trits.ctypes.data, keys.ctypes.data, trits.size,
                              bounds.ctypes.data, len(bounds), out.ctypes.data, out.size)
    if at < 0:
        raise ValueError("Invalid trits, intervals, or page capacity")
    return out[at:].tobytes()


def decode_page(payload, keys, bounds):
    data = np.frombuffer(payload, np.uint8)
    out = np.empty(keys.shape, dtype=np.uint8)
    rc = native().trit_decode(data.ctypes.data, data.size, keys.ctypes.data, keys.size,
                              bounds.ctypes.data, len(bounds), out.ctypes.data)
    if rc:
        raise ValueError("The ternary page did not end at its recorded boundary and initial states")
    return out


def row_pages(rows, width):
    step = max(1, TRITS_PER_PAGE // width)
    return [slice(start, min(rows, start + step)) for start in range(0, rows, step)]


def classes(values, count):
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(order), np.uint8)
    labels[order] = np.arange(len(order)) * count // len(order)
    return labels


def axis_labels(weights, *, fine=False):
    rows, width = weights.shape
    c0 = np.zeros(width, np.int64)
    cp = np.zeros_like(c0)
    r0 = np.zeros(rows, np.int64)
    rp = np.zeros_like(r0)
    for part in row_pages(rows, width):
        block = weights[part]
        for symbol, col_total, row_total in ((1, c0, r0), (2, cp, rp)):
            mask = block == symbol
            col_total += mask.sum(axis=0)
            row_total[part] = mask.sum(axis=1)
    col_rank = classes(c0, 16)
    row_rank = classes(r0, 8)
    col = 4 * (col_rank // 4) + classes((cp + 0.5) / (rows - c0 + 1), 4)
    row = 2 * (row_rank // 4) + classes((rp + 0.5) / (width - r0 + 1), 2)
    if fine:
        col += (col_rank % 4) * 16
        row += (row_rank % 4) * 4
    return col, row


def context_keys(part, col, row, scales, reference, mode):
    """Select a probability table for every trit using only saved labels and decoded data."""
    label = row[part].astype(np.uint16)
    groups = (label[:, None] & 3) * 4 + scales[part]
    keys = (np.repeat(groups, 128, axis=1) << 4) + (col & 15)
    if mode == 3:
        keys += (((label[:, None] >> 2) * 4 + (col >> 4)) << 8)
    elif mode:
        ref = reference[part]
        if mode == 2:
            ref = np.where((label[:, None] & 16) != 0, 2 - ref, ref)
            condition = ((label[:, None] >> 2) & 3) * 3 + ref
        else:
            condition = ref.astype(np.uint16)
        keys += condition << 8
    return keys


def intervals(hist):
    neg, zero, pos = hist.T
    length = neg + zero + pos
    # These two interval boundaries encode P(zero) and P(positive | nonzero).
    z = np.clip(((2 * zero + 1) * UNIT + length) // (2 * length + 3), 1, UNIT - 2)
    nonzero = neg + pos
    p = np.clip(((2 * pos + 1) * (UNIT - z) + nonzero) // (2 * nonzero + 2), 1, UNIT - z - 1)
    return np.column_stack((z, UNIT - p)).astype("<u2")

