import numpy as np


def unpacked(data, offset, count, alphabet):
    bits = (int(alphabet) - 1).bit_length()
    if not bits:
        return np.zeros(count, np.uint32), offset
    n = (count * bits + 7) // 8
    stream = np.frombuffer(data[offset:offset+n], np.uint8)
    if len(stream) != n:
        raise ValueError("Truncated dictionary IDs")
    planes = np.unpackbits(stream, bitorder="little", count=count * bits).reshape(count, bits)
    ids = planes @ (1 << np.arange(bits, dtype=np.uint32))
    if np.any(ids >= alphabet):
        raise ValueError("ID outside dictionary")
    return ids, offset+n
