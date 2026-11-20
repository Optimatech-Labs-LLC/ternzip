from compression import zstd
import struct
import numpy as np
from paired_weights import CONTEXTS
from vocab_rows import CONTEXTS as VOCAB_CONTEXTS


def pack_planes(values, layout):
    flat=np.asarray(values,dtype="<u2").ravel()
    if layout=="bytes":
        return flat.view(np.uint8).reshape(-1,2).T.copy().tobytes()
    return b"".join(np.packbits(((flat>>i)&1).astype(np.uint8),bitorder="little").tobytes() for i in range(16))

def unpack_planes(payload,n,layout):
    if layout=="bytes":
        if len(payload)!=2*n:
            raise ValueError("Incorrect scale field length")
        return np.frombuffer(payload,np.uint8).reshape(2,n).T.copy().view("<u2").reshape(-1)
    planes=np.frombuffer(payload,np.uint8).reshape(16,(n+7)//8)
    if n%8 and np.any(planes[:,-1]>>(n%8)):
        raise ValueError("Nonzero padding in trailing scale bits")
    result=np.zeros(n,np.uint16)
    for i,plane in enumerate(planes):
        result|=np.unpackbits(plane,count=n,bitorder="little").astype(np.uint16)<<i
    return result

METHODS=("identity","center","center_bits","rank","rank_delta","rank_center",
         "rank_center_bits","row_delta","column_delta","bitplanes","rank_bits")

def zigzag(delta):
    value=delta.astype(np.uint16).view(np.int16).astype(np.int32)
    return ((value<<1)^(value>>15)).astype(np.uint16)

def unzigzag(value):
    return ((value.astype(np.int32)>>1)^-(value.astype(np.int32)&1)).astype(np.uint16)

def transform(bits,method):
    buf=bytearray()
    values=bits
    if method.startswith("rank"):
        # The FP16 bit patterns actually used by a tensor form an ordered alphabet.
        occupied=np.flatnonzero(np.bincount(bits.ravel(),minlength=65536)).astype("<u2")
        lut=np.zeros(65536,np.uint16)
        lut[occupied]=np.arange(len(occupied),dtype=np.uint16)
        values=lut[bits]
        buf.extend(struct.pack("<I",len(occupied)))
        buf.extend(np.diff(occupied,prepend=np.zeros(1,"<u2")).astype("<u2").tobytes())
    if "center" in method:
        rows,width=values.shape
        difference=values.astype(np.int32)
        if "column_only" not in method:
            row=(difference.sum(1,dtype=np.int64)//width).astype("<u2")
            buf.extend(pack_planes(row,"bytes"))
            difference-=row[:,None]
        if "row_only" not in method:
            col=(difference.sum(0,dtype=np.int64)//rows).astype("<i2")
            buf.extend(col.tobytes())
            difference-=col[None,:]
        values=zigzag(difference)
    elif "delta" in method:
        axis=0 if method=="column_delta" else 1
        seed=np.zeros((1,values.shape[1]) if axis==0 else (values.shape[0],1),np.uint16)
        values=zigzag(np.diff(values,axis=axis,prepend=seed))
    layout="bits" if "bits" in method or method=="bitplanes" else "bytes"
    buf.extend(pack_planes(values,layout))
    return buf

def inverse(payload,shape,method):
    rows,width=shape
    pos=0
    if method.startswith("rank"):
        n,=struct.unpack_from("<I",payload)
        if not 0<n<=65536:
            raise ValueError("Invalid scale alphabet")
        occupied=np.cumsum(np.frombuffer(payload,"<u2",n,4),dtype=np.uint16)
        if np.any(occupied[1:]<=occupied[:-1]):
            raise ValueError("Invalid ordered scale alphabet")
        pos=4+2*n
    if "center" in method:
        if "column_only" not in method:
            row=unpack_planes(payload[pos:pos+2*rows],rows,"bytes")
            pos+=2*rows
        if "row_only" not in method:
            col=np.frombuffer(payload,"<i2",width,pos)
            pos+=2*width
    layout="bits" if "bits" in method or method=="bitplanes" else "bytes"
    vals=unpack_planes(payload[pos:],rows*width,layout).reshape(shape)
    if "center" in method:
        vals=unzigzag(vals)
        if "column_only" not in method:
            vals+=row[:,None]
        if "row_only" not in method:
            vals+=col[None,:].view("<u2")
    elif "delta" in method:
        vals=np.cumsum(unzigzag(vals),axis=0 if method=="column_delta" else 1,dtype=np.uint16)
    if method.startswith("rank"):
        vals=occupied[vals]
    return vals

def restore_meta(frame,rows,width,mode):
    tag=frame[0]
    groups=rows*width//128
    contexts=VOCAB_CONTEXTS if mode==64 else CONTEXTS[mode]
    tail=rows+width+4*contexts
    dec=zstd.ZstdDecompressor()
    raw=dec.decompress(frame[1:],max_length=16+tail+3*groups+2*(rows+width)+131072)
    if not dec.eof or dec.unused_data:
        raise ValueError("Invalid transformed scale stream")
    if tag==0:
        return raw
    if tag>len(METHODS) or len(raw)<8+tail:
        raise ValueError("Unknown transformed scale layout")
    palette=np.frombuffer(raw,np.uint8,8)
    others=raw[8:8+tail]
    bits=inverse(memoryview(raw)[8+tail:],(rows,width//128),METHODS[tag-1]).ravel()
    high=(bits>>8).astype(np.uint8)
    rank=np.full(256,8,np.uint8)
    rank[palette]=np.arange(8,dtype=np.uint8)
    ranks=rank[high]
    overflow=high[ranks==8]
    return b"".join((struct.pack("<I",len(overflow)),palette,ranks,bits.astype(np.uint8),overflow,others))
