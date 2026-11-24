import struct
from pathlib import Path
from contextlib import nullcontext
SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


def unpack(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def string(f):
    return f.read(unpack(f, "Q")).decode()


def value(f, kind, keep=False):
    if kind in SCALARS:
        return unpack(f, SCALARS[kind])
    if kind == 8:
        if keep:
            return string(f)
        count = unpack(f, "Q")
        f.seek(count, 1)
        return
    if kind == 9:
        subtype = unpack(f, "I")
        count = unpack(f, "Q")
        if subtype in SCALARS:
            f.seek(count * struct.calcsize("<" + SCALARS[subtype]), 1)
        else:
            for _ in range(count):
                value(f, subtype)
        return
    raise ValueError(f"Unknown GGUF value type {kind}")


class GGUF:
    def __init__(self, path):
        stream = hasattr(path, "read")
        self.path = None if stream else Path(path)
        with nullcontext(path) if stream else open(path, "rb") as f:
            f.seek(0)
            if f.read(4) != b"GGUF" or unpack(f, "I") != 3:
                raise ValueError("Expected GGUF v3")
            n, nkv = unpack(f, "Q"), unpack(f, "Q")
            self.metadata = {}
            self.positions = {}
            for _ in range(nkv):
                key = string(f)
                kind = unpack(f, "I")
                self.positions[key] = (kind, f.tell())
                self.metadata[key] = value(f, kind, keep=kind != 9)
            self.kv_end = f.tell()
            self.tensors = {}
            for _ in range(n):
                name = string(f)
                rank = unpack(f, "I")
                dims = tuple(unpack(f, "Q") for _ in range(rank))
                kind, offset = unpack(f, "I"), unpack(f, "Q")
                self.tensors[name] = (dims, kind, offset)
            self.alignment = self.metadata.get("general.alignment") or 32
            self.data_offset = self.align(f.tell())
            f.seek(0)
            self.kv = f.read(self.kv_end)

    def align(self, n):
        return (n + self.alignment - 1) // self.alignment * self.alignment

