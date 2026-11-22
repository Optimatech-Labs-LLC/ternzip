"""Restore the pinned Bonsai 2 9B PTQ1_0 checkpoint from independently framed tensors."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import io
import mmap
import os
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
    def __init__(self, storage, *, pool=None):
        self.storage = storage
        self.file = None
        self.pool = pool
        self.page_workers = 8
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

    @classmethod
    def from_file(cls, file, pool):
        # The directory needs scattered tensor headers. Unmap them before loading weights.
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as directory:
            archive = cls(directory, pool=pool)
        archive.storage = None
        archive.file = file
        return archive

    def encoded_record(self, index):
        start, end = self.physical[index:index + 2]
        if self.file is None:
            return memoryview(self.storage)[start:end]
        data = os.pread(self.file.fileno(), end - start, start)
        if len(data) != end - start:
            raise OSError("Short read of a 9B tensor")
        return memoryview(data)

    def span(self, index, begin, end):
        origin, limit = self.physical[index:index + 2]
        if not 0 <= begin <= end <= limit - origin:
            raise ValueError("9B page extends outside its record")
        if self.file is None:
            return memoryview(self.storage)[origin + begin:origin + end]
        result = os.pread(self.file.fileno(), end - begin, origin + begin)
        if len(result) != end - begin:
            raise OSError("Short read of a 9B page")
        return memoryview(result)

    def page_directory(self, index):
        head = self.span(index, 0, 1 + MATRIX_HEADER.size)
        _, meta_size, pages = MATRIX_HEADER.unpack_from(head, 1)
        end = len(head) + meta_size + 4 * pages
        return self.span(index, 0, end)

    def write_native(self, index, stride, destination):
        """Fill only the tensor's live bytes. GGUF file alignment is handled by restore."""
        from direct_weights9 import PackedParentRows, raw_into
        name = self.names[index]
        dims, kind, _ = self.g.tensors[name]
        logical = self.logical[index + 1] - self.logical[index]
        physical = self.physical[index + 1] - self.physical[index]
        tag = self.span(index, 0, 1)[0]
        if destination.dtype != np.uint8 or not destination.flags.c_contiguous or not destination.flags.writeable:
            raise ValueError("A writable contiguous byte destination is required")
        if tag == 0 and kind == 0:
            expected = 4 * int(np.prod(dims))
            if destination.size != expected or logical != expected or name in self.sources:
                raise ValueError("Unexpected raw tensor extent in this 9B execution layout")
            raw_into(self.span(index, 1, physical), destination)
            return
        if tag not in (1, 2) or kind != 143 or len(dims) != 2 or stride not in (28, 34):
            raise ValueError(f"Unsupported direct matrix {name}")
        width, rows = dims
        groups = rows * (width // 128)
        padding = logical - groups * 28
        if width % 128 or destination.size != groups * stride or not 0 <= padding < self.g.alignment:
            raise ValueError(f"Incorrect native destination for {name}")
        directory = self.page_directory(index)
        if tag == 2:
            from counted_trits9 import record
            record(self, index, stride, directory, destination=destination,
                   read_page=lambda a, b: self.span(index, a, b), record_size=physical)
            if name in self.sources:
                self.cache[name] = PackedParentRows(destination, rows, width, stride)
            return
        parent = self.parents.get(name)
        previous = self.cache.pop(parent) if parent is not None else None
        with self.page_pool(8) as pool:
            decode_tensor(directory[1:], rows, width, None, False, pool, stride,
                          previous_rows=previous, parallel_pages=self.page_workers,
                          destination=destination, read_page=lambda a, b: self.span(index, 1 + a, 1 + b),
                          record_size=physical - 1 - padding)
        if name in self.sources:
            self.cache[name] = PackedParentRows(destination, rows, width, stride)

    def page_pool(self, workers):
        return nullcontext(self.pool) if self.pool is not None else ThreadPoolExecutor(max_workers=workers)

    def record(self, index, stride=28):
        """Decode one tensor, retaining trits only when a later matrix needs them."""
        name = self.names[index]
        data = self.encoded_record(index)
        size = self.logical[index + 1] - self.logical[index]
        if stride not in (28, 34):
            raise ValueError("Unsupported ternary block layout")
        if data[0] == 2:
            from counted_trits9 import record
            return record(self, index, stride, data)
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
        with self.page_pool(4) as pool:
            blocks, _ = decode_tensor(data[1:len(data) - padding], rows, width, None, False,
                                      pool, stride, previous_rows=reference, capture_rows=retained,
                                      parallel_pages=8 if self.pool is not None else 4)
        if retained is not None:
            self.cache[name] = retained
        if padding:
            return np.concatenate((blocks.ravel(), np.frombuffer(data[-padding:], np.uint8)))
        return blocks.ravel()

    def original_chunks(self):
        yield self.header
        for index in range(len(self.names)):
            yield self.record(index)
