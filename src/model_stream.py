"""A seekable virtual GGUF that converts one small page at a time during load."""

import bisect
from contextlib import contextmanager
import ctypes as C
import math
import struct

import numpy as np

from dense_layout import PQ_TYPE, execution_layout


class PackedStream:
    def __init__(self, model, limit):
        self.model = model
        g, spans, kinds, self.header, data_size = execution_layout(
            model, limit, external_input=True, fuse_qkv=True, prefer_attention=True)
        self.length = len(self.header) + data_size
        native_tensor_count = struct.unpack_from("<Q", self.header, 8)[0]
        self.fused_qkv_layers = (len(spans) - native_tensor_count) // 2
        self.segments = []
        self.names = list(spans)
        self.ends = [len(self.header)]
        pos = len(self.header)
        for name, (offset, stored) in spans.items():
            kind = kinds[name]
            if not kind:
                length = stored
            else:
                group_bytes = 34 if kind == PQ_TYPE else 28
                length = (math.prod(g.tensors[name][0]) // 128) * group_bytes
            self.segments.append((offset, stored, kind, pos))
            pos += length
            self.ends.append(pos)
        if pos != self.length:
            raise ValueError("Virtual execution length mismatch")
        self.header_array = np.frombuffer(self.header, np.uint8)
        self.active_segment = None
        self.page_index = None
        self.page = None
        self.position = 0
        self.saved_down_index = self.saved_down = None

    def set_segment(self, index):
        if self.active_segment == index:
            return
        offset, size, kind, _ = self.segments[index]
        self.active_segment = index
        self.page = self.page_index = None
        if kind:
            if self.saved_down_index == index:
                self.raw = self.saved_down
                return
            name = self.names[index]
            self.raw = self.model.native_record(name, offset, size, 34 if kind == PQ_TYPE else 28)
            if name.endswith(".ffn_down.weight"):
                self.saved_down_index, self.saved_down = index, self.raw
        else:
            self.raw = np.frombuffer(self.model.read_record(self.names[index], offset, size), np.uint8)

    def buffer_at(self, position):
        if not 0 <= position < self.length:
            raise ValueError("Position outside the virtual file")
        segment = bisect.bisect_right(self.ends, position) - 1
        if segment < 0:
            return self.header_array, position
        self.set_segment(segment)
        _, _, kind, start = self.segments[segment]
        relative = position - start
        return self.raw, relative

    def read_into(self, pointer, requested):
        read = 0
        count = min(requested, self.length - self.position)
        while read < count:
            array, offset = self.buffer_at(self.position)
            size = min(count - read, array.size - offset)
            if size <= 0:
                raise ValueError("Empty virtual page")
            C.memmove(pointer + read, array.ctypes.data + offset, size)
            read += size
            self.position += size
        return read

    @contextmanager
    def open_c_file(self):
        lib = C.CDLL(None)
        read_type = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_void_p, C.c_int)
        seek_type = C.CFUNCTYPE(C.c_int64, C.c_void_p, C.c_int64, C.c_int)
        problems = []

        @read_type
        def read(_, destination, count):
            try:
                return self.read_into(destination, count)
            except Exception as error:
                problems.append(error)
                return -1

        @seek_type
        def seek(_, offset, origin):
            if origin not in (0, 1, 2):
                return -1
            position = (0, self.position, self.length)[origin] + offset
            if position < 0 or position > self.length:
                return -1
            self.position = position
            return position

        lib.funopen.argtypes = [
            C.c_void_p,
            read_type,
            C.c_void_p,
            seek_type,
            C.c_void_p,
        ]
        lib.funopen.restype = C.c_void_p
        stream = lib.funopen(None, read, None, seek, None)
        if not stream:
            raise OSError("Could not open the model stream")
        lib.fclose.argtypes = [C.c_void_p]
        try:
            yield stream
            if problems:
                raise problems[0]
        finally:
            lib.fclose(stream)

    def close(self):
        self.__dict__.clear()
