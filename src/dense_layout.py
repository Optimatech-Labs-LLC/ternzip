import math
import struct

PTQ_TYPE = 143

PQ_TYPE = 142

def select_layout(g, spans, byte_limit, order=None, *, prefer_gate_up=False,prefer_attention=False):
    """Upgrade some dot-product tensors while bounding the entire execution image."""
    kinds = {name: (0 if g.tensors[name][1] == 0 else PTQ_TYPE) for name in spans}
    groups = {
        name: math.prod(g.tensors[name][0]) // 128 for name in spans if kinds[name]
    }
    minimum = g.data_offset + sum(
        size for name, (_, size) in spans.items() if kinds[name] == 0
    )
    minimum += 28 * sum(groups.values())
    if byte_limit is None:
        return kinds
    if byte_limit < minimum:
        raise ValueError(f"The minimum native execution image is {minimum} bytes")
    remaining = byte_limit - minimum
    if order is None:
        priorities = (
            "output.weight",
            "ffn_down",
            "ffn_gate",
            "ffn_up",
            "attn_q",
            "attn_output",
            "attn_k",
            "attn_v",
        )

        def priority(name):
            if prefer_attention:
                if ".attn_" in name:
                    return 0,name
                if name=="output.weight":
                    return len(priorities)+2,name
            if prefer_gate_up:
                if "ffn_gate" in name or "ffn_up" in name:
                    return 1, name
                if "ffn_down" in name:
                    return 3, name
            family = next(
                (i for i, part in enumerate(priorities) if part in name),
                len(priorities),
            )
            return family, name

        order = sorted(groups, key=priority)
    for name in order:
        if name == "token_embd.weight" or name not in groups:
            continue
        cost = groups[name] * 6
        if cost <= remaining:
            kinds[name] = PQ_TYPE
            remaining -= cost
    return kinds

def fused_attention_spans(g, spans, kinds):
    """Lay Q, K and V rows end to end when all three have the same native block format."""
    groups = {}
    aliases = {}
    for query in spans:
        if not query.endswith(".attn_q.weight"):
            continue
        stem = query.removesuffix(".attn_q.weight")
        names = tuple(f"{stem}.attn_{part}.weight" for part in ("q", "k", "v"))
        if not all(name in spans for name in names) or len({kinds[name] for name in names}) != 1:
            continue
        dims = [g.tensors[name][0] for name in names]
        if any(len(shape) != 2 or shape[0] != dims[0][0] for shape in dims):
            raise ValueError(f"Incompatible attention matrices in {stem}")
        groups.update(dict.fromkeys(names, names))
        aliases[query] = (f"{stem}.attn_qkv.weight", (dims[0][0], sum(shape[1] for shape in dims)))
        aliases.update(dict.fromkeys(names[1:]))
    ordered = {}
    for name in spans:
        if name not in ordered:
            for part in groups.get(name, (name,)):
                ordered[part] = spans[part]
    return ordered, aliases

def execution_layout(compact, byte_limit=None, order=None, full_q2=False, *, prefer_gate_up=False,
                     external_input=False, fuse_qkv=False,prefer_attention=False):
    g, spans, _, _ = compact.canonical_directory()
    if external_input:
        if (byte_limit is None or g.metadata["general.architecture"] != "qwen3"
                or g.tensors["output.weight"][0] != g.tensors["token_embd.weight"][0]):
            raise ValueError("Paged input requires the matching untied Qwen vocabulary matrices")
        byte_limit += math.prod(g.tensors["token_embd.weight"][0]) // 128 * 28
    kinds = select_layout(g, spans, byte_limit, order, prefer_gate_up=prefer_gate_up,prefer_attention=prefer_attention)
    if full_q2:
        kinds = {name: PQ_TYPE if kind else 0 for name, kind in kinds.items()}
    header = bytearray(g.kv)
    if external_input:
        # The graph accepts explicit input embeddings. Its tied output fallback
        # points to the sole native vocabulary matrix, which holds the true output weights.
        spans = {name:extent for name,extent in spans.items() if name != "token_embd.weight"}
    aliases = {}
    if fuse_qkv:
        spans, aliases = fused_attention_spans(g, spans, kinds)
    struct.pack_into("<Q", header, 8, len(spans) - sum(value is None for value in aliases.values()))
    meta_type, location = g.positions["general.file_type"]
    if meta_type != 4:
        raise ValueError("GGUF file type is not a uint32")
    struct.pack_into("<I", header, location, PQ_TYPE if full_q2 else PTQ_TYPE)
    offsets = {}
    length = 0
    for name, (_, encoded_length) in spans.items():
        dims, kind, _ = g.tensors[name]
        if kind not in (0, 42):
            raise ValueError(f"Unsupported tensor type {kind} in {name}")
        offsets[name] = length
        block_bytes = 34 if kinds[name] == PQ_TYPE else 28
        length += encoded_length if kind == 0 else math.prod(dims) // 128 * block_bytes
        if length % g.alignment:
            raise ValueError(f"Unaligned tensor end for {name}")
    for name in (spans if fuse_qkv else g.tensors):
        if name not in offsets:
            continue
        dims = g.tensors[name][0]
        if name in aliases and aliases[name] is None:
            continue
        key, dims = aliases.get(name, (name, dims))
        key = b"token_embd.weight" if external_input and name == "output.weight" else key.encode()
        header.extend(struct.pack("<Q", len(key)) + key + struct.pack("<I", len(dims)))
        header.extend(struct.pack("<" + "Q" * len(dims), *dims))
        header.extend(struct.pack("<IQ", kinds[name], offsets[name]))
    header.extend(bytes(-len(header) % g.alignment))
    return g, spans, kinds, bytes(header), length
