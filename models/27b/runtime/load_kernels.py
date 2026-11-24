import ctypes as C
from functools import lru_cache
from pathlib import Path
import numpy as np
@lru_cache(maxsize=1)
def native():
    lib = C.CDLL(str(Path(__file__).parent/"lib/libload-kernels.dylib"))
    ptr,size = C.c_void_p,C.c_size_t
    lib.pack_native.argtypes = [ptr,ptr,size,size,ptr]
    lib.corrected_intervals.argtypes = [ptr,size,size,ptr,size,ptr,ptr,ptr,ptr,ptr]
    lib.row_intervals.argtypes = [ptr,size,ptr,ptr,ptr]
    lib.block_intervals.argtypes = [ptr,size,size,size,size,size,size,C.c_uint8,C.c_uint8,
                                    ptr,ptr,ptr,ptr,ptr,ptr,ptr,size,ptr]
    lib.block_intervals.restype = C.c_int
    lib.index_stream.argtypes = [ptr,size,ptr,size,size,ptr,size,ptr]
    lib.seek_row.argtypes = [ptr,size,ptr,ptr,size,ptr,size,ptr]
    lib.inverse_embedding.argtypes = [ptr,size,ptr,C.c_float]
    for fn in (lib.pack_native,lib.corrected_intervals,lib.row_intervals,lib.index_stream,lib.seek_row):
        fn.restype = C.c_int
    lib.inverse_embedding.restype = C.c_int
    return lib


def pack_native(trits, bits, stride, dest=None):
    trits = np.ascontiguousarray(trits,np.uint8)
    bits = np.ascontiguousarray(bits,"<u2")
    if trits.size != bits.size*128 or stride not in (28,34):
        raise ValueError("Invalid native ternary extent")
    if dest is None:
        dest = np.empty((bits.size,stride),np.uint8)
    if dest.dtype!=np.uint8 or not dest.flags.c_contiguous or dest.nbytes!=bits.size*stride:
        raise ValueError("Mismatched native output buffer")
    if native().pack_native(trits.ctypes.data,bits.ctypes.data,bits.size,stride,dest.ctypes.data):
        raise ValueError("Invalid native ternary layout")
    return dest


def q2_half_scales(blocks):
    """Split a native 128-weight PQ2 block into the two official 64-weight Q2 blocks."""
    pairs = np.empty((len(blocks),2,18),np.uint8)
    pairs[:,:,:2] = blocks[:,None,:2]
    pairs[:,:,2:] = blocks[:,2:].reshape(-1,2,16)
    return pairs

