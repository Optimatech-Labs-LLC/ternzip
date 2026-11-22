"""Lossless, independently addressable pages of related vocabulary rows."""

import fcntl
import os
import struct
from collections import OrderedDict, defaultdict
from compression import zstd

import numpy as np

from gguf_tools import metadata_array
from paired_weights import decode_page
from ternary_codec import POWERS, RECORD


TAG = 64
PAGE_ROWS = 16
HEADER = struct.Struct("<BBII")
BASE = 256
CONTEXTS = BASE * 10
ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
RELATIONS = ("root", "spelling", "affix", "merge")


def spelling(token):
    # Qwen byte-level BPE uses Ġ for a space. Leave its non-ASCII byte alphabet intact.
    return token.lstrip("Ġ").translate(ASCII_LOWER)


def variants(text):
    stripped = text.strip("._-/()[]{}:'\"")
    if len(stripped) >= 3 and stripped != text:
        yield stripped
    if not text.isascii() or not text.isalpha():
        return
    for ending in ("ing", "ies", "ers", "ed", "es", "er", "ly", "s"):
        if text.endswith(ending) and len(text) >= len(ending) + 3:
            stem = text[:-len(ending)]
            if ending == "ies":
                yield stem + "y"
            yield stem
            if ending in ("ing", "ed"):
                yield stem + "e"
                if stem[-1] == stem[-2]:
                    yield stem[:-1]


class VocabPlan:
    """Place each anchor and its followers in the same disk page."""

    def __init__(self, gguf):
        tokens = metadata_array(gguf, "tokenizer.ggml.tokens")
        types = metadata_array(gguf, "tokenizer.ggml.token_type")
        merges = {}
        for pair in metadata_array(gguf, "tokenizer.ggml.merges"):
            left, right = pair.split(" ", 1)
            merges.setdefault(left + right, (left, right))
        groups = defaultdict(list)
        for i, (word, kind) in enumerate(zip(tokens, types)):
            key = spelling(word)
            if kind == 1 and key:
                groups[key].append(i)
        parent = np.arange(len(tokens), dtype=np.int32)
        relation = np.zeros(len(tokens), np.uint8)
        for ids in groups.values():
            parent[ids[1:]] = ids[0]
            relation[ids[1:]] = 1
        for key, ids in groups.items():
            if len(ids) != 1:
                continue
            i = ids[0]
            choices = [(v, 2) for v in variants(key)]
            parts = sorted(merges.get(tokens[i], ()), key=len, reverse=True)
            choices += [(spelling(s), 3) for s in parts
                        if len(spelling(s)) >= 3 and 3 * len(spelling(s)) >= 2 * len(key)]
            for v, kind in choices:
                other = groups.get(v)
                if other and other[0] < i and parent[other[0]] == other[0]:
                    parent[i], relation[i] = other[0], kind
                    break
        families = defaultdict(list)
        for i, anchor in enumerate(parent):
            families[int(anchor)].append(i)
        pages, parents, types_out = [], [], []
        def new_page():
            pages.append([])
            parents.append([])
            types_out.append([])
        new_page()
        for members in families.values():
            for start in range(0, len(members), PAGE_ROWS):
                block = members[start:start + PAGE_ROWS]
                if len(pages[-1]) + len(block) > PAGE_ROWS:
                    new_page()
                pos = len(pages[-1])
                pages[-1].extend(block)
                parents[-1].extend([-1] + [pos] * (len(block) - 1))
                types_out[-1].extend([0] + [int(relation[i]) for i in block[1:]])
        self.rows = len(tokens)
        self.order = np.full((len(pages), PAGE_ROWS), -1, dtype=np.int32)
        self.parents = np.full(self.order.shape, -1, dtype=np.int8)
        self.relations = np.zeros(self.order.shape, dtype=np.uint8)
        self.nrows = np.array([len(x) for x in pages], dtype=np.uint8)
        self.page_for = np.empty(self.rows, np.int32)
        self.slot_for = np.empty(self.rows, np.uint8)
        for p, ids in enumerate(pages):
            self.order[p, :len(ids)] = ids
            self.parents[p, :len(ids)] = parents[p]
            self.relations[p, :len(ids)] = types_out[p]
            self.page_for[ids] = p
            self.slot_for[ids] = np.arange(len(ids))

    def parts(self, p):
        n = self.nrows[p]
        ids = self.order[p, :n]
        parent = self.parents[p, :n]
        return ids, parent, self.relations[p, :n]

    def bytes(self):
        return sum(x.nbytes for x in vars(self).values() if isinstance(x, np.ndarray))


def scale_bits(parts, rows):
    palette, ranks, low, overflow = parts
    high = palette[np.minimum(ranks, 7)]
    high[ranks == 8] = overflow
    return ((high.astype(np.uint16) << 8) | low).reshape(rows, -1)


def scale_lut(bits):
    hist = np.bincount((bits.ravel() & 32767), minlength=32768)
    return np.minimum(3, (2 * np.cumsum(hist) - hist) * 4 // (2 * bits.size)).astype(np.uint8)


def keys(ids, row_labels, col_labels, bits, lut):
    groups = row_labels[ids, None].astype(np.uint16) * 4 + lut[bits[ids] & 32767]
    # Broadcast one group offset across its 128 columns while writing the final key array.
    keys = np.empty((*groups.shape, 128), np.uint16)
    np.add((groups << 4)[:, :, None], col_labels.reshape(1, -1, 128), out=keys)
    return keys.reshape(len(groups), len(col_labels))


def context(base, relation, parent_trits):
    return base + ((1 + (relation[:, None].astype(np.uint16) - 1) * 3 + parent_trits) << 8)


class VocabTable:
    def __init__(self, plan, width, record, *, file=None, file_offset=0, cache_pages=0, cache_rows=0):
        self.plan, self.width = plan, width
        tag, conditional, meta_bytes, count = HEADER.unpack_from(record)
        if tag!=TAG or conditional not in (1,2) or count!=len(plan.nrows) or width % 128:
            raise ValueError("Unsupported vocabulary record")
        self.file = None
        self.record = None if file is not None else record
        groups = plan.rows * (width // 128)
        fixed = 12 + 2 * groups + width + plan.rows + 4 * CONTEXTS
        frame=record[HEADER.size:HEADER.size+meta_bytes]
        if conditional==2:
            from residual_scales import restore_meta
            raw=restore_meta(frame,plan.rows,width,TAG)
        else:
            dec=zstd.ZstdDecompressor()
            raw=dec.decompress(frame,max_length=fixed+groups+1)
            if not dec.eof or dec.unused_data:
                raise ValueError("Incomplete vocabulary probability metadata")
        escapes, = struct.unpack_from("<I", raw)
        if escapes > groups or len(raw) != fixed + escapes:
            raise ValueError("Invalid vocabulary metadata")
        fields = np.frombuffer(raw, np.uint8, offset=4)
        palette, ranks, low, overflow, col, row, freq = np.split(
            fields, np.cumsum([8, groups, groups, escapes, width, plan.rows]))
        if ranks.max() > 8 or np.count_nonzero(ranks == 8) != escapes or col.max() > 15 or row.max() > 3:
            raise ValueError("Invalid vocabulary scale or axis label")
        self.palette, self.col, self.row = palette.copy(), col.copy(), row.copy()
        self.bits = scale_bits((palette, ranks, low, overflow), plan.rows)
        self.lut = scale_lut(self.bits)
        self.bounds = np.frombuffer(freq.tobytes(), "<u2").reshape(-1, 2)
        pos = HEADER.size + meta_bytes
        lengths = np.frombuffer(record, "<u4", 2 * count, pos)
        self.edges = np.empty(1 + len(lengths), np.int64)
        self.edges[0] = pos + lengths.nbytes
        self.edges[1:] = self.edges[0] + lengths.cumsum(dtype=np.int64)
        if self.edges[-1] != len(record):
            raise ValueError("Vocabulary pages do not span the tensor record")
        self.edges += file_offset
        self.cache = np.empty((cache_pages, PAGE_ROWS, width), np.uint8)
        self.cache_ids = np.full(cache_pages, -1, np.int32)
        self.hot = OrderedDict()
        self.hot_rows = np.empty((cache_rows, width), np.float16)
        self.row_hits = self.row_misses = self.disk_reads = self.disk_bytes = 0
        if file is not None:
            self.file = os.fdopen(os.dup(file.fileno()), "rb", buffering=0)
            try:
                fcntl.fcntl(self.file.fileno(), fcntl.F_NOCACHE, 1)
            except BaseException:
                self.close()
                raise

    def owned_bytes(self):
        return self.plan.bytes() + sum(a.nbytes for a in vars(self).values() if isinstance(a, np.ndarray))

    def reset_cache(self):
        self.hot.clear()
        self.cache_ids.fill(-1)

    def activity(self):
        return {"row_hits":self.row_hits, "row_misses":self.row_misses,
                "page_reads":self.disk_reads, "bytes_read":self.disk_bytes}

    def page(self, p):
        cache_size = len(self.cache_ids)
        slot = p % cache_size if cache_size else 0
        if cache_size and self.cache_ids[slot] == p:
            return self.cache[slot]
        lo, mid, hi = map(int, self.edges[2*p:2*p+3])
        if self.file is None:
            payload = self.record[lo:hi]
        else:
            payload = os.pread(self.file.fileno(), hi-lo, lo)
            self.disk_reads += 1
            self.disk_bytes += len(payload)
        if len(payload) != hi-lo:
            raise ValueError("Truncated vocabulary page")
        ids, parent, rel = self.plan.parts(p)
        root = parent < 0
        base = keys(ids, self.row, self.col, self.bits, self.lut)
        out = np.empty((PAGE_ROWS, self.width), np.uint8)
        used = out[:len(ids)]
        used[root] = decode_page(payload[:mid-lo], base[root], self.bounds)
        if np.any(~root):
            child_keys = context(base[~root], rel[~root], used[parent[~root]])
            used[~root] = decode_page(payload[mid-lo:], child_keys, self.bounds)
        elif mid != hi:
            raise ValueError("Unexpected dependent vocabulary payload")
        if cache_size:
            self.cache[slot, :len(ids)] = used
            self.cache_ids[slot] = p
        return out

    def embedding_into(self, token, out):
        slot = self.hot.get(token)
        if slot is not None:
            self.hot.move_to_end(token)
            self.row_hits += 1
            out[:] = self.hot_rows[slot]
            return
        self.row_misses += 1
        page = self.page(int(self.plan.page_for[token]))
        trits = page[self.plan.slot_for[token]].reshape(-1, 128)
        scale = self.bits[token].view("<f2")
        if not len(self.hot_rows):
            np.multiply(trits.astype(np.int8) - 1, scale[:, None].astype(np.float32),
                        out=out.reshape(-1, 128))
            return
        if len(self.hot) < len(self.hot_rows):
            slot = len(self.hot)
        else:
            _, slot = self.hot.popitem(last=False)
        self.hot[token] = slot
        # A ternary weight is an FP16 scale times -1, 0, or 1; each result fits FP16 exactly.
        np.multiply(trits.astype(np.int8) - 1, scale[:, None], out=self.hot_rows[slot].reshape(-1, 128))
        out[:] = self.hot_rows[slot]

    def embedding_batch(self, ids):
        result = np.empty((len(ids), self.width), np.float32)
        for i, token in enumerate(ids):
            self.embedding_into(int(token), result[i])
        return result

    def native_blocks(self, stride):
        from load_kernels import pack_native
        out = np.empty((self.plan.rows,self.width//128,stride),np.uint8)
        for p in range(len(self.plan.nrows)):
            ids,_,_ = self.plan.parts(p)
            weights = self.page(p)[:len(ids)]
            out[ids] = pack_native(weights,self.bits[ids],stride).reshape(len(ids),self.width//128,stride)
        return out.reshape(-1)

    def canonical(self):
        groups = self.bits.size
        rows, width = self.plan.rows, self.width
        high = (self.bits.ravel() >> 8).astype(np.uint8)
        rank_lut = np.full(256, 8, np.uint8)
        rank_lut[self.palette] = np.arange(8, dtype=np.uint8)
        ranks = rank_lut[high]
        codes = np.empty((rows, width // 128, 25), np.uint8)
        tails = (ranks * 27).reshape(rows, -1)
        for p in range(len(self.plan.nrows)):
            ids, _, _ = self.plan.parts(p)
            weights = self.page(p)[:len(ids)].reshape(len(ids), -1, 128)
            codes[ids] = np.einsum("ngtk,k->ngt", weights[:, :, :125].reshape(len(ids), -1, 25, 5),
                                   POWERS, dtype=np.uint16)
            tails[ids] += (weights[:, :, 125:].astype(np.uint16) @ POWERS[:3]).astype(np.uint8)
        escaped = ranks == 8
        masks = np.packbits(escaped.reshape(-1, 32), axis=1, bitorder="little")
        prefix = np.zeros(len(masks), dtype="<u4")
        np.cumsum(escaped.reshape(-1, 32).sum(1, dtype=np.uint32)[:-1], out=prefix[1:])
        return b"".join((RECORD.pack(self.palette.tobytes(), groups, int(escaped.sum())),
                         codes, tails, self.bits.astype(np.uint8), masks, prefix, high[escaped]))

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
