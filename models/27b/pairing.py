"""Encode an FFN up band against its gate band, retaining the smaller complete record."""
import numpy as np

from codec import (COUNTED, MATRIX, PAIR_GROUPS, compact9, decompress, scale_lut,
                   site, unpack_ptq, unpack_scales, zstd)


def signed_rows(w, ref):
    rows, width = w.shape
    agreement = np.empty(rows, np.int64)
    common = np.empty(rows, np.int64)
    for page in site.row_pages(rows, width):
        a, b = w[page], ref[page]
        present = (a != 1) & (b != 1)
        common[page] = present.sum(1)
        agreement[page] = 2 * ((a == b) & present).sum(1) - common[page]
    bucket = site.classes(np.abs(agreement) / np.maximum(common, 1), 4)
    return bucket, (agreement < 0).astype(np.uint8) * 16


def choose_pair(raw, gate, band, original, pool):
    if original[0] != ord("T"):
        return original, 0
    method, ns, nm, npages = MATRIX.unpack_from(original, 1)
    at = 1 + MATRIX.size
    sf = original[at:at + ns]
    bits = unpack_scales(sf, band.rows, band.width // 128, method)
    oldmeta = decompress(original[at + ns:at + ns + nm], band.width + band.rows + 1024)
    col = np.frombuffer(oldmeta, np.uint8, band.width)
    row = np.frombuffer(oldmeta, np.uint8, band.rows, band.width)

    def trits(buffer):
        blocks = np.frombuffer(buffer, np.uint8, band.weight_bytes).reshape(-1, 28)
        return unpack_ptq(blocks).reshape(band.rows, band.width)

    w, ref = trits(raw), trits(gate)
    strength, flips = signed_rows(w, ref)
    scale = scale_lut(bits)[bits & 32767]
    pages = site.row_pages(band.rows, band.width)
    full_rows = row | (strength * 4) | flips
    hist = np.zeros((3072, 3), np.int64)
    for page in pages:
        k = site.context_keys(page, col, full_rows, scale, ref, 2)
        hist += np.bincount((3 * k.astype(np.int32) + w[page]).ravel(), minlength=3072 * 3).reshape(-1, 3)
    best = original
    for tag, groups in PAIR_GROUPS.items():
        labels = row | ((strength // (4 // groups)) * 4) | flips
        counts = hist.reshape(groups, 4 // groups, 768, 3).sum(1).reshape(-1, 3)
        bounds = site.intervals(counts)
        model = compact9.model_table(bounds)
        meta = zstd.compress(col.tobytes() + labels.tobytes() + bounds.tobytes(), level=5)

        def emit(page):
            keys = site.context_keys(page, col, labels, scale, ref, 2)
            stream, _ = compact9.encode_page(w[page], keys, model)
            return stream

        streams = list(pool.map(emit, pages, buffersize=6))
        sizes = np.asarray([len(p) | COUNTED for p in streams], "<u4").tobytes()
        new = b"".join([bytes((tag,)), MATRIX.pack(method, ns, len(meta), npages), sf, meta, sizes,
                        *streams, raw[band.weight_bytes:]])
        if len(new) < len(best):
            best = new
    return best, len(original) - len(best)
