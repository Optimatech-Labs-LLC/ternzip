"""Rebuild the published byte stream from the official group-128 GGUF and fitted encoding settings."""
from compression import zstd
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from pathlib import Path
import struct
import sys
import numpy as np

from archive_io import ARCHIVE_SHA, ARCHIVE_SIZE, SOURCE_SHA, SOURCE_SIZE, checked_input, output_file
from gguf_tools import GGUF
from paired_archive import HEADER as FILE_HEADER, down_name
from paired_weights import (CONTEXTS, HEADER, axis_labels, context_keys, encode_page, intervals,
                            partner_labels, row_pages, scale_classes)
from predictor import CONFIG, Predictor
from residual_scales import METHODS, transform
from residual_words import pack_field
from ternary_codec import FIXED, INDEX, MAGIC, NATIVE_TRITS, RECORD, qblocks
import vocab_rows as vocab

PROFILE_SHA = "1b7d0416ac41fa7d04d6bb84e069492a777f554829162e22d8e0181c4d000923"
ENTRY = struct.Struct("<BBBI")


def read_profile(names):
    data = (Path(__file__).resolve().parent.parent / "profiles" / "8b-v1.bin").read_bytes()
    if hashlib.sha256(data).hexdigest() != PROFILE_SHA or data[:8] != b"TPROFIL1":
        raise ValueError("Encoding profile checksum or version differs")
    pos = 8
    profile = {}
    for name in names:
        mode, scale_tag, codec, length = ENTRY.unpack_from(data, pos)
        pos += ENTRY.size
        opts = data[pos:pos + length]
        pos += length
        expected = data[pos:pos+32]
        pos += 32
        profile[name] = (mode, scale_tag, codec, opts, expected)
    if pos != len(data):
        raise ValueError("Invalid encoding profile length")
    return profile


def scales(blocks):
    high = blocks[:, 1]
    counts = np.bincount(high, minlength=256)
    palette = np.lexsort((np.arange(256), -counts))[:8].astype(np.uint8)
    lut = np.full(256, 8, np.uint8)
    lut[palette] = np.arange(8, dtype=np.uint8)
    codes = lut[high]
    return palette, codes, np.ascontiguousarray(blocks[:, 0]), high[codes == 8]


def scale_frame(tag, parts, rows, col, row, bounds):
    palette, ranks, low, overflow = parts
    fields = col.tobytes() + row.tobytes() + bounds.tobytes()
    if tag == 0:
        raw = struct.pack("<I", len(overflow)) + palette.tobytes() + ranks.tobytes() + low.tobytes() + overflow.tobytes() + fields
        return bytes([0]) + zstd.compress(raw, level=15)
    bits = vocab.scale_bits(parts, rows)
    raw = palette.tobytes() + fields + transform(bits, METHODS[tag-1])
    return bytes([tag]) + zstd.compress(raw, level=9)


def trits(blocks, dims):
    width, rows = dims
    return NATIVE_TRITS[blocks[:, 2:]].reshape(rows, width)


def vocabulary(weights, parts, plan, tag):
    rows, width = weights.shape
    bits = vocab.scale_bits(parts, rows)
    lut = vocab.scale_lut(bits)
    col, row = axis_labels(weights)
    hist = np.zeros((vocab.CONTEXTS, 3), np.int64)

    def sections(page):
        ids, parents, relation = plan.parts(page)
        roots = parents < 0
        base = vocab.keys(ids, row, col, bits, lut)
        child = vocab.context(base[~roots], relation[~roots], weights[ids[parents[~roots]]])
        return ((weights[ids[roots]], base[roots]), (weights[ids[~roots]], child))

    for page in range(len(plan.nrows)):
        for values, keys in sections(page):
            codes = (3 * keys.astype(np.int32) + values).ravel()
            hist += np.bincount(codes, minlength=3 * len(hist)).reshape(-1, 3)
    bounds = intervals(hist)
    frame = scale_frame(tag, parts, rows, col, row, bounds)
    stream = bytearray()
    sizes = []
    for page in range(len(plan.nrows)):
        for values, keys in sections(page):
            packed = encode_page(values, keys, bounds) if values.size else b""
            stream.extend(packed)
            sizes.append(len(packed))
    return b"".join((vocab.HEADER.pack(64, 2, len(frame), len(plan.nrows)), frame,
                      np.asarray(sizes, "<u4").tobytes(), stream))


def matrix(weights, parts, reference, previous, settings, pool):
    mode_tag, scale_tag, codec, options, _ = settings
    mode = mode_tag - 128
    rows, width = weights.shape
    groups = scale_classes(parts, rows)
    col, row = axis_labels(weights, fine=(mode == 3))
    if mode == 2:
        row = row + partner_labels(weights, reference)[0]
    hist = np.zeros((CONTEXTS[mode], 3), np.int64)
    pages = row_pages(rows, width)
    for part in pages:
        keys = context_keys(part, col, row, groups, reference, mode)
        codes = (3 * keys.astype(np.int32) + weights[part]).ravel()
        hist += np.bincount(codes, minlength=3 * len(hist)).reshape(-1, 3)
    bounds = intervals(hist)
    prediction = Predictor(codec, options, rows, width, previous, bounds)

    def page(part):
        keys = context_keys(part, col, row, groups, reference, mode)
        if prediction.active:
            return pack_field(weights[part], prediction.field(part, keys))
        return encode_page(weights[part], keys, bounds)

    streams = list(pool.map(page, pages, buffersize=4))
    frame = CONFIG.pack(1, codec, len(options)) + options + scale_frame(scale_tag, parts, rows, col, row, bounds)
    return b"".join((HEADER.pack(mode_tag, len(frame), len(streams)), frame,
                      np.asarray([len(s) for s in streams], "<u4").tobytes(), *streams))


def canonical_directory(raw, gguf, names):
    pos = FIXED.size + len(names) * INDEX.size + gguf.data_offset
    entries = bytearray()
    for name in names:
        dims, kind, _ = gguf.tensors[name]
        if kind == 0:
            n = 4 * math.prod(dims)
        else:
            blocks = qblocks(raw, gguf, name)
            high_counts = np.bincount(blocks[:, 1], minlength=256)
            escapes = len(blocks) - int(np.sort(high_counts)[-8:].sum())
            n = RECORD.size + 27 * len(blocks) + 8 * (len(blocks) // 32) + escapes
        entries.extend(INDEX.pack(pos, n))
        pos += n
    header = FIXED.pack(MAGIC, len(raw), gguf.data_offset, len(names))
    return zstd.compress(header + entries + raw[:gguf.data_offset], level=19)


def encode(source, destination):
    if zstd.zstd_version_info != (1, 5, 7):
        raise RuntimeError("Byte-identical encoding requires libzstd 1.5.7")
    with checked_input(source, SOURCE_SIZE, SOURCE_SHA) as (_, raw):
        gguf = GGUF(source)
        names = sorted(gguf.tensors, key=lambda name: gguf.tensors[name][2])
        profile = read_profile(n for n in names if gguf.tensors[n][1] == 42)
        prefix = canonical_directory(raw, gguf, names)
        plan = vocab.VocabPlan(gguf)

        def get_weights(name):
            return trits(qblocks(raw, gguf, name), gguf.tensors[name][0])

        def record(name, pool):
            dims, kind, off = gguf.tensors[name]
            if kind == 0:
                at = gguf.data_offset + off
                return zstd.compress(raw[at:at + 4*math.prod(dims)], level=9)
            settings = profile[name]
            mode, scale_tag, codec, options, expected = settings
            blocks = qblocks(raw, gguf, name)
            weights, parts = trits(blocks, dims), scales(blocks)
            if mode == 64:
                data = vocabulary(weights, parts, plan, scale_tag)
            else:
                ref = get_weights(down_name(name)).T.copy() if mode in (129, 130) else None
                previous = None
                if codec in (3, 4):
                    distance, = struct.unpack_from("<H", options)
                    _, layer, kind_name = name.split(".", 2)
                    previous = get_weights(f"blk.{int(layer)-distance}.{kind_name}")
                data = matrix(weights, parts, ref, previous, settings, pool)
            if hashlib.sha256(data).digest() != expected:
                raise ValueError(f"Encoded record differs from the release for {name}")
            return data

        with output_file(destination, ARCHIVE_SIZE, ARCHIVE_SHA) as out, ThreadPoolExecutor(4) as pool:
            out.write(FILE_HEADER.pack(b"QTile001", len(names), len(prefix)))
            out.write(bytes(4 * len(names)))
            out.write(prefix)
            sizes = []
            for i, name in enumerate(names, 1):
                data = record(name, pool)
                out.write(data)
                sizes.append(len(data))
                if i % 40 == 0:
                    print(f"Encoded {i}/{len(names)} tensors", file=sys.stderr, flush=True)
            out.seek(FILE_HEADER.size)
            out.write(np.asarray(sizes, "<u4").tobytes())
    return {"bytes": ARCHIVE_SIZE, "sha256": ARCHIVE_SHA}
