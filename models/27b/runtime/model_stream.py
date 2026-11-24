import ctypes as C
from contextlib import contextmanager


class PackedStream:
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

