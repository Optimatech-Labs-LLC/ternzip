"""Restore the pinned Bonsai 2 9B PTQ1_0 checkpoint from independently framed tensors."""
from concurrent.futures import ThreadPoolExecutor
import io
import struct

import numpy as np

from gguf_tools import GGUF
from paired_archive import decompress
from paired_weights import HEADER as MATRIX_HEADER, decode_tensor
from predictor import CONFIG
from row_options import ROW_HEADER

from versions9 import ARCHIVE_SHA, ARCHIVE_SIZE, MAGIC, SOURCE_NAME, SOURCE_SHA, SOURCE_SIZE, archive_identity

DIRECTORY = struct.Struct("<8sIIQ")

class NineArchive:
    def __init__(self, storage):
        self.storage = storage
        magic, count, header_size, size = DIRECTORY.unpack_from(storage)
        start = DIRECTORY.size + 8 * count
        if magic != MAGIC or count != 427 or size != SOURCE_SIZE or start + header_size > len(storage):
            raise ValueError("Invalid 9B archive directory")
        self.header = decompress(storage[start:start + header_size], 64 << 20)
        self.g = GGUF(io.BytesIO(self.header))
        self.names = sorted(self.g.tensors, key=lambda name: self.g.tensors[name][2])
        if len(self.names) != count or self.g.data_offset != len(self.header):
            raise ValueError("The archive directory differs from the embedded GGUF")
        lengths = struct.unpack_from(f"<{count}Q", storage, DIRECTORY.size)
        self.physical = [start + header_size]
        for length in lengths:
            if not length:
                raise ValueError("Empty 9B record")
            self.physical.append(self.physical[-1] + length)
        self.logical = [len(self.header) + self.g.tensors[name][2] for name in self.names] + [size]
        if (self.physical[-1] != len(storage) or self.logical[0] != len(self.header)
                or any(end <= begin for begin, end in zip(self.logical, self.logical[1:]))):
            raise ValueError("9B records do not cover the complete file")
        self.by_name = {name: i for i, name in enumerate(self.names)}
        self.parents = {}
        for i, name in enumerate(self.names):
            start = self.physical[i]
            if storage[start] != 1 or not 128 <= storage[start + 1] <= 131:
                continue
            options = start + 1 + MATRIX_HEADER.size
            _, codec, length = CONFIG.unpack_from(storage, options)
            if codec != 3:
                continue
            if length < ROW_HEADER.size:
                raise ValueError("Truncated signed-row reference")
            distance = ROW_HEADER.unpack_from(storage, options + CONFIG.size)[0]
            prefix, layer, kind = name.split(".", 2)
            parent = f"{prefix}.{int(layer) - distance}.{kind}"
            if not distance or self.g.tensors.get(parent, ())[:2] != self.g.tensors[name][:2]:
                raise ValueError(f"Invalid signed-row reference for {name}")
            self.parents[name] = parent
        self.sources = set(self.parents.values())
        self.cache = {}

    def record(self, index, stride=28):
        """Decode one tensor, retaining trits only when a later matrix needs them."""
        name = self.names[index]
        start, end = self.physical[index:index + 2]
        data = memoryview(self.storage)[start:end]
        size = self.logical[index + 1] - self.logical[index]
        if stride not in (28, 34):
            raise ValueError("Unsupported ternary block layout")
        if data[0] == 2:
            from counted_trits9 import record
            return record(self, index, stride)
        if data[0] == 0:
            raw = decompress(data[1:], size)
            if len(raw) != size or name in self.sources:
                raise ValueError(f"Invalid raw tensor {name}")
            return np.frombuffer(raw, np.uint8)
        dims, kind, _ = self.g.tensors[name]
        if data[0] != 1 or len(dims) != 2 or kind != 143:
            raise ValueError(f"Unknown 9B tensor encoding for {name}")
        width, rows = dims
        padding = size - rows * (width // 128) * 28
        if width % 128 or not 0 <= padding < self.g.alignment:
            raise ValueError(f"Invalid PTQ1_0 extent for {name}")
        reference = None
        if name in self.parents:
            parent = self.parents[name]
            if parent not in self.cache:
                self.record(self.by_name[parent])
            reference = self.cache.pop(parent)
        retained = np.empty((rows, width), np.uint8) if name in self.sources else None
        with ThreadPoolExecutor(max_workers=4) as pool:
            blocks, _ = decode_tensor(data[1:len(data) - padding], rows, width, None, False,
                                      pool, stride, previous_rows=reference, capture_rows=retained)
        if retained is not None:
            self.cache[name] = retained
        if padding:
            return np.concatenate((blocks.ravel(), np.frombuffer(data[-padding:], np.uint8)))
        return blocks.ravel()

    def original_chunks(self):
        yield self.header
        for index in range(len(self.names)):
            yield self.record(index)
