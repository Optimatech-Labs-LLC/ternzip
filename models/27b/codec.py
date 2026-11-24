"""Experimental, standalone-on-disk Ternzip archive for Prism's pinned 27B PTQ GGUF."""
import argparse
from bisect import bisect_right
from collections import Counter, OrderedDict
from compression import zstd
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

import numpy as np

RUNTIME = Path(__file__).resolve().with_name("runtime")
if not RUNTIME.is_dir():
    RUNTIME = Path(__file__).resolve().parents[1] / "Qwen3-8B-Compression" / "ternary-runtime"
sys.path.insert(0, str(RUNTIME))
import compact9
from gguf_tools import GGUF
from load_kernels import pack_native, q2_half_scales
from model_stream import PackedStream
from paired_archive import decompress
import paired_weights as site
from row_pairing import unpack_ptq
from residual_scales import inverse, pack_planes, transform, unpack_planes
from vocab_rows import scale_lut


PRISM_REV = "839bd57023c1558945afea70a74fa83c054f35f9"
SOURCE_SHA = "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
SOURCE_SIZE = 5_946_648_928
GGUF_HEADER_SHA = "c2745c0473a980a45c99fe1d2fe85cbbf86f7eca41132d25beb55f249fe703fa"
MAGIC = b"TZ27PT01"
HEADER = struct.Struct("<8sIIQ")
MATRIX = struct.Struct("<BIII")
BAND_WEIGHTS = 1 << 25
MAX_HEADER = 16 << 20
COUNTED = 1 << 31
PAIR_GROUPS = {ord("P"): 1, ord("S"): 2, ord("F"): 4}


@dataclass(frozen=True)
class Band:
    name: str
    start: int
    length: int
    rows: int = 0
    width: int = 0

    @property
    def weight_bytes(self):
        return self.rows * self.width // 128 * 28


def layout(g):
    names = sorted(g.tensors, key=lambda n: g.tensors[n][2])
    offsets = [g.data_offset + g.tensors[n][2] for n in names] + [SOURCE_SIZE]
    if offsets[0] != g.data_offset or len(names) != 851:
        raise ValueError("Unexpected 27B GGUF directory")
    bands = []
    for i, name in enumerate(names):
        dims, kind, _ = g.tensors[name]
        start, end = offsets[i:i + 2]
        if end <= start:
            raise ValueError("Overlapping or empty tensor")
        if kind != 143:
            bands.append(Band(name, start, end - start))
            continue
        width, rows = dims
        if width % 128 or not 0 <= end - start - width * rows // 128 * 28 < g.alignment:
            raise ValueError(f"Unexpected ternary dimensions for {name}")
        row_bytes = width // 128 * 28
        step = max(1, BAND_WEIGHTS // width)
        for first in range(0, rows, step):
            count = min(step, rows - first)
            at = start + first * row_bytes
            size = end - at if first + count == rows else count * row_bytes
            bands.append(Band(name, at, size, count, width))
    return bands


def peer_bands(bands):
    gates, indices, result = {}, Counter(), {}
    for i, b in enumerate(bands):
        ordinal = indices[b.name]
        indices[b.name] += 1
        if b.name.endswith(".ffn_gate.weight"):
            gates[b.name, ordinal] = i
        elif b.name.endswith(".ffn_up.weight"):
            gate = gates.get((b.name.replace(".ffn_up.weight", ".ffn_gate.weight"), ordinal))
            if gate is not None and (bands[gate].rows, bands[gate].width) == (b.rows, b.width):
                result[i] = gate
    return result


def read_header(src):
    src.seek(0)
    initial = src.read(MAX_HEADER)
    g = GGUF(io.BytesIO(initial))
    if g.data_offset > len(initial):
        raise ValueError("Incomplete GGUF header")
    return g, initial[:g.data_offset]


@contextmanager
def new_binary(path):
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".partial",
                                     mode="w+b", delete=False) as out:
        tmp = Path(out.name)
        try:
            yield out
            out.flush()
            os.fsync(out.fileno())
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                fcntl.flock(directory, fcntl.LOCK_EX)
                if os.path.lexists(path):
                    raise FileExistsError(path)
                tmp.rename(path)
            finally:
                os.close(directory)
        finally:
            tmp.unlink(missing_ok=True)


def pack_scales(bits, *, basic=False):
    """Try raw bit planes and reversible horizontal or vertical uint16 differences."""
    def frame(values):
        planes = (values >> 8).astype(np.uint8).tobytes() + values.astype(np.uint8).tobytes()
        return zstd.compress(planes, level=5)

    choices = [frame(bits)]
    for axis in (1, 0):
        delta = bits.copy()
        if axis == 1:
            np.subtract(bits[:, 1:], bits[:, :-1], out=delta[:, 1:])
        else:
            np.subtract(bits[1:], bits[:-1], out=delta[1:])
        choices.append(frame(delta))
    method = min(range(len(choices)), key=lambda i: len(choices[i]))
    selected = method, choices[method]
    if not basic:
        selected, _ = improve_scales(bits, *selected)
    return *selected, len(choices[0]) - len(selected[1])


def improve_scales(bits, method, frame):
    candidates = {method: frame}
    for mode, tag in (("center", 3), ("rank_center", 4)):
        source = transform(bits, mode)
        tail = len(source) - 2 * bits.size
        prefix = bytes(source[:tail])
        residual = unpack_planes(memoryview(source)[tail:], bits.size, "bytes")
        candidates[tag] = zstd.compress(prefix + pack_planes(residual, "bits"), level=7)
        # Keep both parts. The value on restore is 4 * quotient + remainder.
        high = residual >> 2
        esc = high >= 255
        remainders = b"".join(np.packbits((residual >> b) & 1, bitorder="little").tobytes() for b in (0, 1))
        data = b"".join((prefix, np.minimum(high, 255).astype(np.uint8).tobytes(),
                         (high[esc] - 255).astype("<u2").tobytes(), remainders))
        candidates[tag + 2] = zstd.compress(data, level=7)
    without_divide = min(len(v) for k, v in candidates.items() if k <= 4)
    best = min(candidates, key=lambda k: (len(candidates[k]), k))
    return (best, candidates[best]), without_divide - len(candidates[best])


def unpack_scales(frame, rows, cols, method):
    n = rows * cols
    if 3 <= method <= 6:
        limit = 132000 + 2 * (rows + cols) + 4 * n
        raw = decompress(frame, limit)
        mode = "rank_center" if method in (4, 6) else "center"
        if method <= 4:
            return inverse(memoryview(raw), (rows, cols), mode + "_bits")
        prefix = 2 * (rows + cols)
        if mode == "rank_center":
            alphabet, = struct.unpack_from("<I", raw)
            if not 1 <= alphabet <= 65536:
                raise ValueError("Invalid scale alphabet")
            prefix += 4 + 2 * alphabet
        high = np.frombuffer(raw, np.uint8, n, prefix).astype(np.uint16)
        escaped = high == 255
        ne = int(escaped.sum())
        at = prefix + n
        extra = np.frombuffer(raw, "<u2", ne, at)
        if np.any(extra > (65535 >> 2) - 255):
            raise ValueError("Scale quotient overflow")
        high[escaped] += extra
        at += 2 * ne
        plane = (n + 7) // 8
        if len(raw) != at + 2 * plane:
            raise ValueError("Scale remainder length mismatch")
        residual = high << 2
        for b in (0, 1):
            field = np.frombuffer(raw, np.uint8, plane, at + b * plane)
            if n % 8 and field[-1] >> (n % 8):
                raise ValueError("Nonzero scale remainder padding")
            residual |= np.unpackbits(field, count=n, bitorder="little").astype(np.uint16) << b
        return inverse(memoryview(raw[:prefix] + pack_planes(residual, "bytes")), (rows, cols), mode)
    raw = decompress(frame, 2 * n)
    if len(raw) != 2 * n or method not in (0, 1, 2):
        raise ValueError("Invalid scale stream")
    planes = np.frombuffer(raw, np.uint8)
    bits = ((planes[:n].astype(np.uint16) << 8) | planes[n:]).reshape(rows, cols)
    if method:
        np.cumsum(bits, axis=1 if method == 1 else 0, dtype=np.uint16, out=bits)
    return bits


def encode(raw, band, pool, *, basic_scales=False):
    if len(raw) != band.length:
        raise ValueError(f"Truncated source at {band.start}")
    if not band.rows:
        return b"R" + zstd.compress(raw, level=7), {}
    rows, width = band.rows, band.width
    blocks = np.frombuffer(raw, np.uint8, count=band.weight_bytes).reshape(-1, 28)
    bits = blocks[:, 26:].copy().view("<u2").reshape(rows, -1)
    w = unpack_ptq(blocks).reshape(rows, width)
    if not np.array_equal(pack_native(w, bits, 28), blocks):
        return b"R" + zstd.compress(raw, level=7), {"noncanonical_bands": 1}
    method, sf, scale_saving = pack_scales(bits, basic=basic_scales)
    col, row = site.axis_labels(w)
    scale = scale_lut(bits)[bits & 32767]
    pages = site.row_pages(rows, width)
    hist = np.zeros((256, 3), np.int64)
    for page in pages:
        keys = site.context_keys(page, col, row, scale, None, 0)
        labels = keys.astype(np.int32) * 3 + w[page]
        hist += np.bincount(labels.ravel(), minlength=768).reshape(256, 3)
    bounds = site.intervals(hist)
    table = compact9.model_table(bounds)
    meta = zstd.compress(col.tobytes() + row.tobytes() + bounds.tobytes(), level=5)

    def encode_page(page):
        keys = site.context_keys(page, col, row, scale, None, 0)
        plain = site.encode_page(w[page], keys, bounds)
        counted, _ = compact9.encode_page(w[page], keys, table)
        if len(counted) < len(plain):
            return counted, COUNTED | len(counted), len(plain) - len(counted)
        return plain, len(plain), 0

    streams, sizes, savings = zip(*pool.map(encode_page, pages, buffersize=6))
    prefix = MATRIX.pack(method, len(sf), len(meta), len(pages))
    payload = b"".join((b"T", prefix, sf, meta, np.asarray(sizes, dtype="<u4").tobytes(),
                        *streams, raw[band.weight_bytes:]))
    return payload, {"ternary_bands": 1, "pages": len(pages),
                     "counted_pages": sum(bool(x & COUNTED) for x in sizes),
                     "counted_bytes_saved": sum(savings), "scale_delta_bytes_saved": scale_saving,
                     f"scale_method_{method}_bands": 1}


def decode(frame, band, pool, *, stride=28, padding=None, reference=None, capture=None):
    source_pad = band.length - band.weight_bytes if band.rows else 0
    pad = source_pad if padding is None else padding
    if stride not in (28, 34, 36) or not 0 <= pad < 32:
        raise ValueError("Invalid destination layout")
    if frame[0] == ord("R"):
        raw = decompress(frame[1:], band.length)
        if len(raw) != band.length:
            raise ValueError("Wrong raw band length")
        data = np.frombuffer(raw, np.uint8)
        if not band.rows or (stride == 28 and capture is None):
            return data
        native = data[:band.weight_bytes].reshape(-1, 28)
        trits = unpack_ptq(native).reshape(band.rows, band.width)
        if capture is not None:
            capture[:] = trits
        if stride == 28:
            return data
        wide = pack_native(trits, native[:, 26:].copy().view("<u2"), 34)
        wide = (q2_half_scales(wide) if stride == 36 else wide).ravel()
        return np.concatenate((wide, np.zeros(pad, np.uint8))) if pad else wide
    factor = PAIR_GROUPS.get(frame[0])
    if not band.rows or (frame[0] != ord("T") and not factor):
        raise ValueError("Invalid band tag")
    if factor and (reference is None or reference.shape != (band.rows, band.width)):
        raise ValueError("Missing matching gate matrix")
    method, ns, nm, npages = MATRIX.unpack_from(frame, 1)
    start = 1 + MATRIX.size
    rows, width = band.rows, band.width
    pages = site.row_pages(rows, width)
    payload_at = start + ns + nm + 4 * npages
    if npages != len(pages) or payload_at > len(frame):
        raise ValueError("Invalid page directory")
    bits = unpack_scales(frame[start:start + ns], rows, width // 128, method)
    scale = scale_lut(bits)[bits & 32767]
    contexts = 768 * factor if factor else 256
    expected_meta = width + rows + 4 * contexts
    meta = decompress(frame[start + ns:start + ns + nm], expected_meta)
    if len(meta) != expected_meta:
        raise ValueError("Incomplete probability metadata")
    labels = np.frombuffer(meta, np.uint8)
    col, row = labels[:width], labels[width:width + rows]
    bounds = np.frombuffer(meta, "<u2", offset=width + rows).reshape(contexts, 2)
    if col.max() > 15 or row.max() > (31 if factor else 3) or (factor and np.any(((row >> 2) & 3) >= factor)):
        raise ValueError("Invalid row or column class")
    table = compact9.model_table(bounds)
    entries = np.frombuffer(frame, "<u4", npages, start + ns + nm)
    edges = [payload_at, *(payload_at + (entries & (COUNTED - 1)).cumsum(dtype=np.int64)).tolist()]
    if edges[-1] + source_pad != len(frame):
        raise ValueError("Page lengths do not cover band")
    nbytes = rows * width // 128 * stride
    out = np.empty(nbytes + pad, np.uint8)
    blocks = out[:nbytes].reshape(-1, stride)
    if pad:
        out[nbytes:] = np.frombuffer(frame[edges[-1]:], np.uint8) if stride == 28 else 0

    def unpack_page(item):
        i, page = item
        keys = site.context_keys(page, col, row, scale, reference, 2 if factor else 0)
        data = frame[edges[i]:edges[i + 1]]
        if entries[i] & COUNTED:
            trits = compact9.decode_page(data, keys, table)
        else:
            trits = site.decode_page(data, keys, bounds)
        n = width // 128
        dest = blocks[page.start * n:page.stop * n]
        if stride == 36:
            dest[:] = q2_half_scales(pack_native(trits, bits[page], 34)).reshape(-1, 36)
        else:
            pack_native(trits, bits[page], stride, dest)
        if capture is not None:
            capture[page] = trits

    for _ in pool.map(unpack_page, enumerate(pages), buffersize=6):
        pass
    return out


class Archive:
    read_into = PackedStream.read_into
    open_c_file = PackedStream.open_c_file

    def __init__(self, src, pool):
        self.src, self.pool = src, pool
        magic, n, header_length, self.length = HEADER.unpack(src.read(HEADER.size))
        if magic != MAGIC or self.length != SOURCE_SIZE or not 1 <= n <= 16384 or header_length > MAX_HEADER:
            raise ValueError("Unexpected 27B archive header")
        sizes = np.frombuffer(src.read(8 * n), "<u8")
        self.header = decompress(src.read(header_length), MAX_HEADER)
        if hashlib.sha256(self.header).hexdigest() != GGUF_HEADER_SHA:
            raise ValueError("Archive header differs from Prism's 27B GGUF")
        self.g = GGUF(io.BytesIO(self.header))
        self.bands = layout(self.g)
        if len(self.bands) != n or len(self.header) != self.g.data_offset:
            raise ValueError("Archive and GGUF layout disagree")
        self.logical = [b.start for b in self.bands]
        base = src.tell()
        remaining = os.fstat(src.fileno()).st_size - base
        if any(not 1 <= int(x) <= max(32 << 20, b.length * 2) for x, b in zip(sizes, self.bands)) or sum(map(int, sizes)) != remaining:
            raise ValueError("Archive lengths do not cover the file")
        self.physical = [base, *(base + sizes.cumsum(dtype=np.int64)).tolist()]
        self.position, self.index, self.array = 0, None, None
        self.header_array = np.frombuffer(self.header, np.uint8)
        self.parents = peer_bands(self.bands)
        self.gates = set(self.parents.values())
        self.references = OrderedDict()

    def frame(self, i):
        start, end = self.physical[i:i + 2]
        self.src.seek(start)
        return memoryview(self.src.read(end - start))

    def record(self, i, *, stride=28, padding=None):
        frame = self.frame(i)
        band = self.bands[i]
        reference = None
        if frame[0] in PAIR_GROUPS:
            if i not in self.parents:
                raise ValueError("Invalid gate reference")
            parent = self.parents[i]
            if parent not in self.references:
                self.record(parent)
            reference = self.references[parent]
        capture = np.empty((band.rows, band.width), np.uint8) if i in self.gates else None
        out = decode(frame, band, self.pool, stride=stride, padding=padding, reference=reference, capture=capture)
        if capture is not None:
            self.references[i] = capture
            self.references.move_to_end(i)
            if len(self.references) > 3:
                self.references.popitem(last=False)
        return out

    def buffer_at(self, pos):
        if not 0 <= pos < self.length:
            raise ValueError("GGUF offset out of bounds")
        if pos < len(self.header):
            return self.header_array, pos
        i = bisect_right(self.logical, pos) - 1
        if self.index != i:
            self.array = self.record(i)
            self.index = i
        return self.array, pos - self.logical[i]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pack(source, archive, receipt, basic_scales=False, independent=False):
    started = time.perf_counter()
    if source.stat().st_size != SOURCE_SIZE:
        raise ValueError("Wrong official file size")
    stats = Counter()
    with source.open("rb") as src, new_binary(archive) as out, ThreadPoolExecutor(6) as pool:
        g, header = read_header(src)
        bands = layout(g)
        parents = {} if independent or basic_scales else peer_bands(bands)
        gates = set(parents.values())
        source_gates = {}
        frame = zstd.compress(header, level=12)
        out.write(HEADER.pack(MAGIC, len(bands), len(frame), SOURCE_SIZE))
        out.write(bytes(8 * len(bands)))
        out.write(frame)
        digest = hashlib.sha256(header)
        sizes = []
        src.seek(len(header))
        for i, band in enumerate(bands):
            if src.tell() != band.start:
                raise ValueError("Unaccounted source bytes")
            raw = src.read(band.length)
            digest.update(raw)
            encoded, delta = encode(raw, band, pool, basic_scales=basic_scales)
            if i in gates:
                source_gates[i] = raw
            if i in parents:
                from pairing import choose_pair
                prior = source_gates.pop(parents[i])
                encoded, gain = choose_pair(raw, prior, band, encoded, pool)
                delta["paired_bytes_saved"] = gain
                delta["paired_bands"] = int(gain > 0)
            out.write(encoded)
            sizes.append(len(encoded))
            stats.update(delta)
            if (i + 1) % 100 == 0 or i + 1 == len(bands):
                print(f"{i + 1}/{len(bands)} bands | {src.tell():,} original | {out.tell():,} packed", flush=True)
        if digest.hexdigest() != SOURCE_SHA or src.read(1):
            raise ValueError("Source does not match Prism's SHA-256")
        out.seek(HEADER.size)
        out.write(np.asarray(sizes, dtype="<u8").tobytes())
    result = {"source_revision": PRISM_REV, "source_sha256": SOURCE_SHA, "source_bytes": SOURCE_SIZE,
              "archive_bytes": archive.stat().st_size, "bands": len(bands), "tensors": len(g.tensors),
              "encode_wall_seconds": time.perf_counter() - started, **stats}
    result["archive_sha256"] = sha(archive)
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def restore(archive, restored, receipt=None):
    started = time.perf_counter()
    digest = hashlib.sha256()
    with archive.open("rb") as src, new_binary(restored) as out, ThreadPoolExecutor(6) as pool:
        reader = Archive(src, pool)
        pos = 0
        while pos < reader.length:
            arr, offset = reader.buffer_at(pos)
            data = memoryview(arr)[offset:]
            digest.update(data)
            out.write(data)
            pos += len(data)
        if pos != SOURCE_SIZE or digest.hexdigest() != SOURCE_SHA:
            raise ValueError("Restored file does not match Prism")
    result = {"restored_bytes": pos, "restored_sha256": digest.hexdigest(),
              "restore_wall_seconds": time.perf_counter() - started}
    if receipt:
        receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for cmd, names in (("pack", ("source", "archive", "receipt")),
                       ("restore", ("archive", "restored", "receipt"))):
        q = sub.add_parser(cmd)
        for name in names:
            q.add_argument(name, type=Path)
        if cmd == "pack":
            q.add_argument("--basic-scales", action="store_true")
            q.add_argument("--independent", action="store_true")
    args = vars(p.parse_args())
    fn = {"pack": pack, "restore": restore}[args.pop("action")]
    fn(**args)
