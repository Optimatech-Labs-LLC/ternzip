import math
import struct
import numpy as np


MAGIC = b"QTrits01"

FIXED = struct.Struct("<8sQQI")

INDEX = struct.Struct("<QQ")

RECORD = struct.Struct("<8sII")

POWERS = np.array([1, 3, 9, 27, 81], dtype=np.uint16)

NATIVE_TRITS = (
    (np.arange(256, dtype=np.uint16)[:, None] >> np.array([0, 2, 4, 6])) & 3
).astype(np.uint8)

FIVE_TRITS = ((np.arange(243, dtype=np.uint16)[:, None] // POWERS) % 3).astype(np.uint8)

THREE_TRITS = FIVE_TRITS[:27, :3]

def qblocks(buf, g, name):
    dims, kind, off = g.tensors[name]
    groups = math.prod(dims) // 128
    if kind != 42 or groups % 32:
        raise ValueError((name, dims, kind))
    return np.frombuffer(
        buf, dtype=np.uint8, count=groups * 34, offset=g.data_offset + off
    ).reshape(-1, 34)

def view(record):
    dictionary, groups, exceptions = RECORD.unpack_from(record)
    pos = RECORD.size

    def take(dtype, n):
        nonlocal pos
        a = np.frombuffer(record, dtype=dtype, count=n, offset=pos)
        pos += a.nbytes
        return a

    codes = take(np.uint8, groups * 25).reshape(groups, 25)
    tails = take(np.uint8, groups)
    low = take(np.uint8, groups)
    masks = take("<u4", groups // 32)
    prefix = take("<u4", groups // 32)
    high = take(np.uint8, exceptions)
    if pos != len(record):
        raise ValueError("Malformed tensor payload")
    return np.frombuffer(dictionary, np.uint8), codes, tails, low, masks, prefix, high

def unpack_range(arrays, start, end):
    palette, codes, tail, low, masks, prefix, overflow = arrays
    idx = np.arange(start, end, dtype=np.uint32)
    q, r = np.divmod(tail[start:end], np.uint8(27))
    if q.max(initial=0) > 8 or codes[start:end].max(initial=0) > 242:
        raise ValueError("Invalid ternary code")
    high = palette[np.minimum(q, 7)]
    selected = idx[q == 8]
    if len(selected):
        tile = selected // 32
        below = (np.uint32(1) << (selected % 32)) - np.uint32(1)
        prior = np.bitwise_count(masks[tile] & below)
        high[q == 8] = overflow[prefix[tile] + prior]
    trits = np.empty((end - start, 128), np.uint8)
    trits[:, :125] = FIVE_TRITS[codes[start:end]].reshape(-1, 125)
    trits[:, 125:] = THREE_TRITS[r]
    out = np.empty((end - start, 34), np.uint8)
    out[:, 0] = low[start:end]
    out[:, 1] = high
    t = trits.reshape(end - start, 32, 4)
    out[:, 2:] = t[:, :, 0] | (t[:, :, 1] << 2) | (t[:, :, 2] << 4) | (t[:, :, 3] << 6)
    return out

def original_chunks(buf):
    g, tensors, (p, header), _ = buf.canonical_directory()
    read = buf.read_record
    yield buf.read_range(p, p + header)
    for name, (off, sz) in tensors.items():
        data = read(name, off, sz)
        if g.tensors[name][1] == 0:
            yield data
        else:
            arrays = view(data)
            for start in range(0, len(arrays[1]), 16384):
                yield memoryview(
                    unpack_range(arrays, start, min(start + 16384, len(arrays[1])))
                ).cast("B")
