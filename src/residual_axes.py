from compression import zstd
import numpy as np

def frequencies(bounds):
    zero, end_negative = bounds.astype(np.int32).T
    return np.stack((end_negative-zero, zero, 4096-end_negative), axis=-1)

FRACTION=(1048576,1095000,1143480,1194106,1246974,1302182,1359835,1420040,
          1482910,1548564,1617125,1688722,1763488,1841564,1923097,2008240)

LIMIT=64

def odds_table(bounds):
    freq=frequencies(bounds).astype(np.int64)
    z,neg,pos=freq[:,1,None],freq[:,0,None],freq[:,2,None]
    power=np.arange(-LIMIT,LIMIT+1)
    whole,frac=np.divmod(power,16)
    gain=np.array(FRACTION,dtype=np.int64)[frac] << (whole+4)
    unit=1<<24
    def rate(yes,no,den):
        top=yes*gain
        bottom=top+no*unit
        return (top*den+bottom//2)//bottom
    zero=np.clip(rate(z,4096-z,4096),1,4094).astype(np.uint16)
    sign=np.clip(rate(pos,neg,65536),1,65535).astype(np.uint16)
    return zero,sign

def decode_options(payload,nrows,width):
    if len(payload)<2:
        raise ValueError("Missing residual predictor options")
    mask,level=payload[:2]
    if mask>15 or level>2:
        raise ValueError("Invalid residual predictor flags")
    factors=np.full(5,256,"<u2")
    count=(0,4,5)[level]
    factors[:count]=np.frombuffer(payload,"<u2",count,2)
    if np.any((factors<16)|(factors>4096)):
        raise ValueError("Invalid phrase multiplier")
    rest=payload[2+2*count:]
    rows,cols=np.zeros((nrows,2),np.int8),np.zeros((width,2),np.int8)
    need=sum(len(axis) for flag,axis in ((1,rows),(2,cols),(4,rows),(8,cols)) if mask&flag)
    if need:
        dec=zstd.ZstdDecompressor()
        fields=dec.decompress(rest,max_length=need+1)
        if len(fields)!=need or not dec.eof or dec.unused_data:
            raise ValueError("Invalid channel correction extent")
        p=0
        for flag,axis,column in ((1,rows,0),(2,cols,0),(4,rows,1),(8,cols,1)):
            if mask&flag:
                axis[:,column]=np.frombuffer(fields,np.int8,len(axis),p)
                p+=len(axis)
    elif rest:
        raise ValueError("Unexpected correction data")
    return rows,cols,factors
