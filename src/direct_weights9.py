"""Direct destinations for 9B weights in the pinned Apple unified-memory backend."""
import ctypes as C
from functools import lru_cache

import numpy as np

from native_engine import ModelParams, bind


@lru_cache(maxsize=1)
def trit_digits():
    codes = np.arange(256, dtype=np.uint32)
    return np.stack([((codes * power // 256) % 3).astype(np.uint8) for power in (3, 9, 27, 81, 243)])


class PackedParentRows:
    """A signed-row predictor reads just its requested rows from the already filled Metal buffer."""
    def __init__(self, destination, rows, width, stride):
        self.shape = (rows, width)
        self.stride = stride
        self.blocks = destination.reshape(rows, width // 128, stride)

    def __getitem__(self, ids):
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= self.shape[0]):
            raise ValueError('Invalid row in an earlier Metal tensor')
        blocks = self.blocks[ids]
        output = np.empty((*blocks.shape[:2], 128), np.uint8)
        if self.stride == 34:
            quads = output.reshape(*blocks.shape[:2], 32, 4)
            for digit in range(4):
                quads[..., digit] = (blocks[..., 2:] >> (2 * digit)) & 3
        else:
            digits = trit_digits()
            for start, offset, lanes, count in ((0, 0, 16, 5), (80, 16, 8, 5), (120, 24, 2, 4)):
                code = blocks[..., offset:offset + lanes]
                dest = output[..., start:start + lanes*count].reshape(*blocks.shape[:2], count, lanes)
                for digit in range(count):
                    dest[..., digit, :] = digits[digit, code]
        return output.reshape(len(ids), self.shape[1])

@lru_cache(maxsize=1)
def zstd_functions():
    import _zstd
    path = getattr(_zstd, '__file__', None)
    if path:
        try:
            lib = C.CDLL(path)
            return (bind(lib, 'ZSTD_findFrameCompressedSize', C.c_size_t, C.c_void_p, C.c_size_t),
                    bind(lib, 'ZSTD_decompress', C.c_size_t, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t))
        except (OSError, AttributeError):
            pass
    return None


def raw_into(frame, destination):
    native = zstd_functions()
    if native is None:
        # Standalone CPython can build zstd into its interpreter and hide its C
        # symbols. Bound the temporary allocation while filling the same buffer.
        from compression import zstd
        decoder = zstd.ZstdDecompressor()
        out = memoryview(destination).cast('B')
        at = 0
        try:
            while not decoder.eof:
                part = decoder.decompress(frame, max_length=min(256 * 1024, len(out) - at + 1))
                frame = b''
                if len(part) > len(out) - at or (decoder.needs_input and not decoder.eof):
                    raise ValueError('Invalid or incomplete raw tensor frame')
                out[at:at + len(part)] = part
                at += len(part)
        except zstd.ZstdError as error:
            raise ValueError('Invalid raw tensor frame') from error
        if at != len(out) or decoder.unused_data:
            raise ValueError('Raw frame did not exactly fill its destination')
        return
    source = np.frombuffer(frame, np.uint8)
    frame_bytes, decompress = native
    if frame_bytes(source.ctypes.data, source.size) != source.size:
        raise ValueError('Invalid raw tensor frame')
    if decompress(destination.ctypes.data, destination.nbytes, source.ctypes.data, source.size) != destination.nbytes:
        raise ValueError('Raw tensor did not fill its destination')


def load_direct(image, llama, file, settings):
    ptr = C.c_void_p
    writer_type = C.CFUNCTYPE(C.c_int, ptr, C.c_char_p, ptr, C.c_size_t, C.c_int, C.c_char_p, ptr)
    errors = []
    written = set()

    @writer_type
    def fill(_, label, address, size, native_type, buffer_type, tensor):
        try:
            name = label.decode('ascii')
            if name == 'token_embd.weight':
                name = 'output.weight'
            i = image.archive.by_name[name]
            kind = 142 if i in image.strides else image.archive.g.tensors[name][1]
            if i in written or i not in image.indices or native_type != kind:
                raise ValueError(f'Unexpected native request for {name}, kind {native_type}')
            dest = np.ctypeslib.as_array((C.c_uint8 * size).from_address(address))
            image.archive.write_native(i, image.strides.get(i, 28), dest)
            if image.observe is not None:
                image.observe(name, size, buffer_type.decode('ascii'), tensor)
            written.add(i)
            return 1
        except BaseException as error:
            # A background page must never write to a buffer after the model is released.
            image.archive.pool.shutdown(wait=True, cancel_futures=True)
            errors.append(error)
            return 0

    load = bind(llama, 'ternzip9_load_with_writer', ptr, ptr, ModelParams, writer_type, ptr)
    model = load(file, settings, fill, None)
    if errors:
        raise errors[0]
    if model and (written != set(image.indices) or image.archive.cache):
        bind(llama, 'llama_model_free', None, ptr)(model)
        raise ValueError('Direct writer did not visit all native weights and signed-row dependencies')
    return model
