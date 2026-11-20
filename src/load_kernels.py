"""Bindings for the bounded, per-page native weight conversion."""
import ctypes as C
from functools import lru_cache
from native_paths import library
import numpy as np


@lru_cache(maxsize=1)
def native():
    lib = C.CDLL(str(library("load-kernels")))
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


def corrected_intervals(keys,base,lut,row,col):
    keys = np.ascontiguousarray(keys,"<u2")
    height,width = keys.shape
    if row.shape!=(height,2) or col.shape!=(width,2) or any(x.shape!=(len(base),129) for x in lut):
        raise ValueError("Invalid predictor table extent")
    arrays = [np.ascontiguousarray(a,t) for a,t in ((base,"<u2"),(lut[0],"<u2"),(lut[1],"<u2"),(row,np.int8),(col,np.int8))]
    b,z,s,r,c=(a.ctypes.data for a in arrays)
    out = np.empty((height,width,2),"<u2")
    if native().corrected_intervals(keys.ctypes.data,height,width,b,len(base),z,s,r,c,out.ctypes.data):
        raise ValueError("Invalid probability context")
    return out


def row_intervals(reference,bounds,factors):
    reference=np.ascontiguousarray(reference,np.uint8)
    bounds=np.ascontiguousarray(bounds,"<u2")
    factors=np.ascontiguousarray(factors,"<u2")
    if reference.shape!=bounds.shape[:-1] or bounds.shape[-1]!=2 or factors.shape!=(3,3):
        raise ValueError("Invalid cross-layer probability shape")
    out=np.empty_like(bounds)
    if native().row_intervals(reference.ctypes.data,reference.size,bounds.ctypes.data,factors.ctypes.data,out.ctypes.data):
        raise ValueError("Invalid trit or probability")
    return out


def interval_lookup(bounds, factors):
    sites=np.broadcast_to(bounds,(3,*bounds.shape))
    refs=np.broadcast_to(np.arange(3,dtype=np.uint8)[:,None],(3,len(bounds)))
    return row_intervals(refs,sites,factors)
