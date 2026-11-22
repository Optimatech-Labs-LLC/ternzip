"""Decode the two reversible representations of binary16 scale words in v0.3.0."""
from compression import zstd
import struct
import numpy as np
from paired_weights import CONTEXTS
from residual_scales import inverse, pack_planes

HEADER = struct.Struct("<BIII")
QUOTIENT = struct.Struct("<III")

def from_planes(data, shape, width):
    count = shape[0] * shape[1]
    packed = np.frombuffer(data, np.uint8).reshape(width, (count + 7) // 8)
    if count % 8 and np.any(packed[:, -1] >> (count % 8)):
        raise ValueError("Nonzero padding in a scale bit plane")
    words = np.zeros(count, np.uint16)
    for bit, plane in enumerate(packed):
        words |= np.unpackbits(plane, count=count, bitorder="little").astype(np.uint16) << bit
    return words.reshape(shape)


def expand(data, size):
    decoder = zstd.ZstdDecompressor()
    result = decoder.decompress(data, max_length=size + 1)
    if not decoder.eof or decoder.unused_data or len(result) != size:
        raise ValueError("Scale frame extent differs from the tensor directory")
    return result


def unpack_div4(payload, shape, width, seeded=True):
    seed_size, quotient_size, overflow_size = QUOTIENT.unpack_from(payload)
    seed_end = QUOTIENT.size + seed_size
    quotient_end = seed_end + quotient_size
    overflow_end = quotient_end + overflow_size
    if overflow_end > len(payload) or (not seeded and seed_size):
        raise ValueError("Invalid quotient/remainder frame sizes")
    rows, columns = shape
    residual_shape = (rows, columns - int(seeded))
    count = residual_shape[0] * residual_shape[1]
    result = np.empty(shape, np.uint16)
    if seeded:
        seeds = expand(payload[QUOTIENT.size:seed_end], width * ((rows + 7) // 8))
        result[:, :1] = from_planes(seeds, (rows, 1), width)
    coarse = expand(payload[seed_end:quotient_end], count)
    quotient = np.frombuffer(coarse, np.uint8).astype(np.uint16)
    escaped = quotient == 255
    num_escaped = int(np.count_nonzero(escaped))
    if num_escaped:
        overflow = expand(payload[quotient_end:overflow_end], 2 * num_escaped)
        extra = np.frombuffer(overflow, "<u2")
        if np.any(extra > (1 << (width - 2)) - 1 - 255):
            raise ValueError("Scale quotient exceeds the field width")
        quotient[escaped] += extra
    elif overflow_size:
        raise ValueError("Unexpected quotient exceptions")
    packed_low = expand(payload[overflow_end:], 2 * ((count + 7) // 8))
    remainder = from_planes(packed_low, residual_shape, 2)
    result[:, int(seeded):] = (quotient.reshape(residual_shape) << 2) | remainder
    return result


def unroll_quarter(residuals):
    """Recover each 16-bit word from the preceding rounded row average."""
    words = np.empty_like(residuals)
    words[:, 0] = residuals[:, 0]
    prediction = residuals[:, 0].astype(np.int32)
    for column in range(1, residuals.shape[1]):
        code = residuals[:, column].astype(np.int32)
        value = (prediction + ((code >> 1) ^ -(code & 1))) & 65535
        words[:, column] = value
        prediction = (3 * prediction + value + 2) // 4
    return words


def restore(frame, rows, width, mode):
    tag, side_n, table_n, extra_n = HEADER.unpack_from(frame)
    if tag not in (201, 204) or width % 128 or rows < 1 or width < 128 or not 0 <= mode < len(CONTEXTS):
        raise ValueError("Invalid scale-field header")
    pos = HEADER.size
    if extra_n or pos + side_n + table_n > len(frame):
        raise ValueError("Invalid scale-field extent")
    shape = (rows, width // 128)
    side = expand(frame[pos:pos + side_n], 8 + width + rows + 4 * CONTEXTS[mode])
    pos += side_n
    if tag == 204:
        if table_n < 4:
            raise ValueError("Missing rank table")
        length, = struct.unpack_from("<I", frame, pos)
        if not 6 + 2 * sum(shape) <= length <= 4 + 2 * 65536 + 2 * sum(shape):
            raise ValueError("Invalid rank table length")
        prefix = expand(frame[pos + 4:pos + table_n], length)
        occupied, = struct.unpack_from("<I", prefix)
        if length != 4 + 2 * occupied + 2 * sum(shape):
            raise ValueError("Rank table dimensions differ")
        residuals = unpack_div4(frame[pos + table_n:], shape, 16, seeded=False)
        bits = inverse(prefix + pack_planes(residuals, "bytes"), shape, "rank_center")
    else:
        if table_n:
            raise ValueError("Unexpected table for row-predicted scale words")
        bits = unroll_quarter(unpack_div4(frame[pos:], shape, 16))
    words = bits.ravel()
    palette = np.frombuffer(side, np.uint8, 8)
    high = (words >> 8).astype(np.uint8)
    lookup = np.full(256, 8, np.uint8)
    lookup[palette] = np.arange(8, dtype=np.uint8)
    ranks = lookup[high]
    overflow = high[ranks == 8]
    return b"".join((struct.pack("<I", len(overflow)), palette, ranks,
                     words.astype(np.uint8), overflow, side[8:]))
