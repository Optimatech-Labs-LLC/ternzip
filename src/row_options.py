from compression import zstd
import struct
import numpy as np

def decode_map(payload,count,total_rows):
    bits=(total_rows-1).bit_length()
    p=np.unpackbits(np.frombuffer(payload,np.uint8),count=count*(bits+1),bitorder="little").reshape(count,bits+1)
    return (p[:,:bits].astype(np.uint32)@(1<<np.arange(bits,dtype=np.uint32))).astype(np.int32),p[:,bits].astype(bool)

ROW_HEADER=struct.Struct("<HBBII")

def unpack_options(data,rows):
    if len(data)<ROW_HEADER.size+18: raise ValueError("Truncated row model")
    distance,method,compressed,nbase,nmap=ROW_HEADER.unpack_from(data)
    if not distance or method>3 or compressed>1: raise ValueError("Invalid row model")
    at=ROW_HEADER.size
    base=data[at:at+nbase];at+=nbase
    table=np.frombuffer(data,"<u2",9,at).reshape(3,3);at+=18
    if np.any((table<16)|(table>4096)): raise ValueError("Invalid row odds")
    words=None
    if method:
        words=np.full(5,256,"<u2")
        n=0 if method==3 else (4 if method==1 else 5)
        words[:n]=np.frombuffer(data,"<u2",n,at);at+=2*n
        if np.any((words<16)|(words>4096)): raise ValueError("Invalid word odds")
    if at+nmap!=len(data): raise ValueError("Incorrect row map length")
    payload=data[at:]
    needed=(rows*((rows-1).bit_length()+1)+7)//8
    if compressed:
        dec=zstd.ZstdDecompressor()
        payload=dec.decompress(payload,max_length=needed+1)
        if not dec.eof or dec.unused_data: raise ValueError("Incomplete row map")
    if len(payload)!=needed: raise ValueError("Incorrect number of row IDs")
    indices,flip=decode_map(payload,rows,rows)
    if np.any(indices>=rows): raise ValueError("Row index outside reference")
    return distance,base,table,words,indices,flip
