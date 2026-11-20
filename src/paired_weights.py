"""Ternary entropy coding and reconstruction of a complete transformer matrix."""
import ctypes as C
from functools import lru_cache
import struct
from compression import zstd
import numpy as np
from native_paths import library
from ternary_codec import POWERS, RECORD

TRITS_PER_PAGE = 1 << 20

HEADER = struct.Struct("<BII")

CONTEXTS = (256, 768, 3072, 4096)

UNIT = 4096

@lru_cache(maxsize=1)
def native():
    lib = C.CDLL(str(library("trit-rans")))
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

def scale_classes(parts, rows):
    palette, ranks, low, overflow = parts
    if ranks.max() > 8 or np.count_nonzero(ranks == 8) != len(overflow):
        raise ValueError("Invalid high-byte scale selectors")
    high = palette[np.minimum(ranks, 7)]
    high[ranks == 8] = overflow
    magnitude = ((high.astype(np.uint16) << 8) | low) & 0x7fff
    hist = np.bincount(magnitude, minlength=32768)
    quartile = np.minimum(3, (2 * np.cumsum(hist) - hist) * 4 // (2 * len(ranks)))
    return quartile[magnitude].astype(np.uint8).reshape(rows, -1)

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

def partner_labels(weights, reference):
    """Learn the up/down sign orientation and its strength for each neuron."""
    if reference.shape != weights.shape:
        raise ValueError("The transpose does not align with the target matrix")
    rows, width = weights.shape
    agreement = np.empty(rows, np.int64)
    common = np.empty_like(agreement)
    heldout_correct = heldout_total = 0
    for part in row_pages(rows, width):
        a, b = weights[part], reference[part]
        both = (a != 1) & (b != 1)
        same = (a == b) & both
        common[part] = both.sum(axis=1)
        agreement[part] = 2 * same.sum(axis=1) - common[part]
        # Select the polarity on even columns and score it on odd columns.
        flip = 2 * same[:, ::2].sum(axis=1) < both[:, ::2].sum(axis=1)
        heldout_correct += int(((same[:, 1::2] ^ flip[:, None]) & both[:, 1::2]).sum())
        heldout_total += int(both[:, 1::2].sum())
    strength = classes(np.abs(agreement) / np.maximum(common, 1), 4)
    tags = strength * 4 + (agreement < 0).astype(np.uint8) * 16
    return tags, {"flipped_neurons": int((agreement < 0).sum()), "neurons": rows,
                  "heldout_signs_correct": heldout_correct, "heldout_nonzero_pairs": heldout_total}

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

def decode_tensor(record, rows, width, reference, keep_transpose, pool, native_stride=None,
                  *, previous_rows=None, capture_rows=None):
    mode, meta_size, npages = HEADER.unpack_from(record)
    extended=128<=mode<=131
    if extended:
        mode-=128
    pages = row_pages(rows, width)
    if mode >= len(CONTEXTS) or npages != len(pages) or (mode in (1, 2) and reference is None):
        raise ValueError("Invalid ternary tensor mode or reference")
    groups = rows * width // 128
    at = HEADER.size + meta_size
    data_start = at + 4 * npages
    if data_start > len(record):
        raise ValueError("Truncated ternary page directory")
    table_bytes = 4 * CONTEXTS[mode]
    fixed = 12 + 2 * groups + width + rows + table_bytes
    frame=record[HEADER.size:at]
    codec=0
    transformed=False
    option_data = b""
    if extended:
        from predictor import CONFIG
        transformed,codec,option_bytes=CONFIG.unpack_from(frame)
        if transformed not in (0,1) or codec>4 or option_bytes>len(frame)-CONFIG.size:
            raise ValueError("Invalid residual record options")
        option_data=frame[CONFIG.size:CONFIG.size+option_bytes]
        frame=frame[CONFIG.size+option_bytes:]
    if transformed:
        from residual_scales import restore_meta
        meta=restore_meta(frame,rows,width,mode)
    else:
        dec=zstd.ZstdDecompressor()
        meta=dec.decompress(frame,max_length=fixed+groups+1)
        if not dec.eof or dec.unused_data:
            raise ValueError("Invalid compressed probability metadata")
    if len(meta) < fixed:
        raise ValueError("Invalid scale and probability metadata")
    escapes, = struct.unpack_from("<I", meta)
    if escapes > groups or len(meta) != fixed + escapes:
        raise ValueError("Tensor scale escape count disagrees with its payload")
    fields = np.frombuffer(meta, np.uint8, offset=4)
    palette, ranks, low, overflow, col, row, table = np.split(
        fields, np.cumsum([8, groups, groups, escapes, width, rows]))
    if col.max() > (63 if mode == 3 else 15) or row.max() > (3, 3, 31, 15)[mode]:
        raise ValueError("Invalid row or column probability class")
    bounds = np.frombuffer(table.tobytes(), dtype="<u2").reshape(-1, 2)
    from predictor import Predictor
    prediction = Predictor(codec, option_data, rows, width, previous_rows, bounds)
    lengths = np.frombuffer(record, dtype="<u4", count=npages, offset=at)
    edges = [data_start, *(data_start + lengths.cumsum(dtype=np.int64)).tolist()]
    if edges[-1] != len(record):
        raise ValueError("Ternary page lengths do not cover the record")
    scales = scale_classes((palette, ranks, low, overflow), rows)
    if native_stride is None:
        codes = np.empty((groups, 25), np.uint8)
        tails = ranks * 27
    else:
        from load_kernels import pack_native
        from vocab_rows import scale_bits
        bits = scale_bits((palette,ranks,low,overflow),rows)
        native_data = np.empty((groups,native_stride),np.uint8)
    transpose = np.empty((width, rows), np.uint8) if keep_transpose else None

    def decode(item):
        i, part = item
        keys = context_keys(part, col, row, scales, reference, mode)
        payload=record[edges[i]:edges[i+1]]
        if prediction.active:
            from residual_words import unpack_field
            trits = unpack_field(payload, prediction.field(part, keys))
        else:
            trits = decode_page(payload, keys, bounds)
        if transpose is not None:
            transpose[:, part] = trits.T
        if capture_rows is not None:
            capture_rows[part]=trits
        flat = trits.reshape(-1, 128)
        target = slice(part.start * (width // 128), part.stop * (width // 128))
        if native_stride is not None:
            pack_native(flat,bits[part],native_stride,native_data[target])
            return
        codes[target] = np.einsum("ngk,k->ng", flat[:, :125].reshape(-1, 25, 5),
                                  POWERS, dtype=np.uint16)
        tails[target] += (flat[:, 125:].astype(np.uint16) @ POWERS[:3]).astype(np.uint8)

    for _ in pool.map(decode, enumerate(pages), buffersize=4):
        pass
    if native_stride is not None:
        return native_data,transpose
    escaped = (ranks == 8).reshape(-1, 32)
    masks = np.packbits(escaped, axis=1, bitorder="little")
    prefix = np.zeros(len(masks), dtype="<u4")
    np.cumsum(escaped.sum(axis=1, dtype=np.uint32)[:-1], out=prefix[1:])
    native_record = b"".join((RECORD.pack(palette.tobytes(), groups, escapes), codes, tails,
                              low, masks, prefix, overflow))
    return native_record, transpose
