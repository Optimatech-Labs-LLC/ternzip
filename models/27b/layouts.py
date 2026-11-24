"""Expose the same 27B archive in any of Prism's published ternary execution layouts."""
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

from codec import Archive, PackedStream, SOURCE_SHA, SOURCE_SIZE, new_binary, restore
from load_kernels import pack_native, q2_half_scales
from row_pairing import unpack_ptq


# Official file type, tensor type, bytes per 128 weights, file bytes and LFS SHA-256.
FORMATS = {
    "PTQ1_0": (143, 143, 28, SOURCE_SIZE, SOURCE_SHA),
    "PQ2_0": (141, 142, 34, 7_206_168_928, "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"),
    "Q2_0": (41, 42, 36, 7_626_008_928, "4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e"),
}


class PublishedView:
    read_into = PackedStream.read_into
    open_c_file = PackedStream.open_c_file

    def __init__(self, archive, fmt):
        self.base, self.fmt, self.g = archive, fmt, archive.g
        file_type, kind, self.stride, self.length, _ = FORMATS[fmt]
        g = self.g
        meta_kind, at = g.positions["general.file_type"]
        if meta_kind != 4 or g.metadata["general.file_type"] != 143:
            raise ValueError("Unexpected source file type")
        header = bytearray(g.kv)
        struct.pack_into("<I", header, at, file_type)
        offsets = {}
        self.logical = []
        self.pads = []
        offset = 0
        for i, band in enumerate(archive.bands):
            offsets.setdefault(band.name, offset)
            self.logical.append(g.data_offset + offset)
            last = i + 1 == len(archive.bands) or archive.bands[i + 1].name != band.name
            if band.rows:
                offset += band.rows * band.width // 128 * self.stride
                pad = (-offset % g.alignment) if last else 0
                offset += pad
            else:
                offset += band.length
                pad = 0
            self.pads.append(pad)
        for name, (dims, old_kind, _) in g.tensors.items():
            label = name.encode("utf-8")
            header.extend(struct.pack("<Q", len(label)) + label + struct.pack("<I", len(dims)))
            header.extend(struct.pack(f"<{len(dims)}QIQ", *dims, kind if old_kind == 143 else old_kind, offsets[name]))
        header.extend(bytes(-len(header) % g.alignment))
        if len(header) != g.data_offset or offset + len(header) != self.length:
            raise ValueError(f"Incorrect {fmt} GGUF layout")
        self.header_array = np.frombuffer(header, np.uint8)
        self.position, self.index, self.array = 0, None, None

    def buffer_at(self, pos):
        if not 0 <= pos < self.length:
            raise ValueError("Published view offset out of range")
        if pos < len(self.header_array):
            return self.header_array, pos
        i = bisect_right(self.logical, pos) - 1
        if self.index != i:
            data = self.base.record(i, stride=self.stride, padding=self.pads[i])
            self.index, self.array = i, data
        return self.array, pos - self.logical[i]


def restore_layout(archive, restored, receipt=None, fmt="PTQ1_0"):
    if fmt == "PTQ1_0":
        return restore(archive, restored, receipt)
    start = time.perf_counter()
    digest = hashlib.sha256()
    with archive.open("rb") as src, new_binary(restored) as out, ThreadPoolExecutor(6) as pool:
        view = PublishedView(Archive(src, pool), fmt)
        pos = 0
        while pos < view.length:
            array, offset = view.buffer_at(pos)
            data = memoryview(array)[offset:]
            digest.update(data)
            out.write(data)
            pos += len(data)
        if (pos, digest.hexdigest()) != FORMATS[fmt][3:]:
            raise ValueError(f"Restored {fmt} differs from Prism's official checksum")
    result = {"format": fmt, "restored_bytes": pos, "restored_sha256": digest.hexdigest(),
              "restore_wall_seconds": time.perf_counter() - start}
    if receipt:
        receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def verify_all(path, restored_ptq, restored_pq, receipt, reference=None):
    """Restore with only the archive as input. Optionally compare PTQ directly as it is written."""
    start = time.perf_counter()
    hashes = {fmt: hashlib.sha256() for fmt in FORMATS}
    sizes = dict.fromkeys(FORMATS, 0)
    with ExitStack() as stack:
        f = stack.enter_context(path.open("rb"))
        pool = stack.enter_context(ThreadPoolExecutor(6))
        base = Archive(f, pool)
        views = {fmt: PublishedView(base, fmt) for fmt in FORMATS}
        files = {"PTQ1_0": stack.enter_context(new_binary(restored_ptq)),
                 "PQ2_0": stack.enter_context(new_binary(restored_pq))}
        ref = stack.enter_context(reference.open("rb")) if reference else None

        def emit(fmt, data):
            num = len(data)
            sizes[fmt] += num
            hashes[fmt].update(data)
            if fmt in files:
                files[fmt].write(data)
            if fmt == "PTQ1_0" and ref and ref.read(num) != memoryview(data):
                raise ValueError(f"Direct comparison failed by offset {sizes[fmt]}")

        for fmt, view in views.items():
            emit(fmt, view.header_array)
        for i, band in enumerate(base.bands):
            native, _ = base.buffer_at(band.start)
            emit("PTQ1_0", native)
            if band.rows:
                pq, _ = views["PQ2_0"].buffer_at(views["PQ2_0"].logical[i])
                emit("PQ2_0", pq)
                num = band.rows * band.width // 128 * 34
                emit("Q2_0", q2_half_scales(pq[:num].reshape(-1, 34)).ravel())
                if views["Q2_0"].pads[i]:
                    emit("Q2_0", bytes(views["Q2_0"].pads[i]))
            else:
                for fmt in ("PQ2_0", "Q2_0"):
                    emit(fmt, native)
            if (i + 1) % 200 == 0:
                print(f"Restored {i + 1}/{len(base.bands)} bands", flush=True)
        if ref and ref.read(1):
            raise ValueError("Unexpected trailing reference data")
        proof = {}
        for fmt, digest in hashes.items():
            size, value = sizes[fmt], digest.hexdigest()
            if (size, value) != FORMATS[fmt][3:]:
                raise ValueError(f"{fmt} differs from Prism: {size} bytes, SHA-256 {value}")
            proof[fmt] = {"bytes": size, "sha256": value, "matches_publisher": True}
    result = {"restores": proof, "PTQ1_0_directly_compared_to_download": bool(reference),
              "three_layouts_wall_seconds": time.perf_counter() - start}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("path", "restored_ptq", "restored_pq", "receipt"):
        p.add_argument(name, type=Path)
    p.add_argument("--reference", type=Path)
    verify_all(**vars(p.parse_args()))
