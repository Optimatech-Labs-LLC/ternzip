from compression import zstd
import struct
import numpy as np
from load_kernels import native
from packed_ids import unpacked

BLOCK_HEADER = struct.Struct("<HBBHII")

class Overrides:
    def __init__(self, data, rows, width):
        if len(data) < BLOCK_HEADER.size + 18:
            raise ValueError("Truncated small-block options")
        self.distance, self.base_codec, flags, self.width, nbase, active = BLOCK_HEADER.unpack_from(data)
        if not self.distance or self.base_codec > 4 or flags > 31 or not self.width:
            raise ValueError("Invalid small-block options or shape")
        start=BLOCK_HEADER.size
        self.row_word=0
        if flags & 8:
            self.row_word,=struct.unpack_from("<H",data,start)
            start+=2
            if not self.row_word or rows % self.width or width % self.row_word:
                raise ValueError("Vertical word groups do not tile the tensor")
            grid_rows, stripes=rows//self.width, width//self.row_word
        else:
            if width % self.width: raise ValueError("Horizontal blocks do not tile the tensor")
            grid_rows, stripes=rows, width//self.width
        count = grid_rows * stripes
        offset = start + nbase
        if offset+18 > len(data) or active > count:
            raise ValueError("Invalid block address count")
        self.base_options = data[start:offset]
        self.factors = np.frombuffer(data, "<u2", 9, offset).reshape(3,3)
        if np.any((self.factors < 16) | (self.factors > 4096)):
            raise ValueError("Invalid block odds")
        self.global_ids = bool(flags & 4)
        self.scalar = bool(flags & 16)
        choices = 2*(stripes if self.row_word else rows) if self.global_ids else 2
        mask_bytes = (count+7)//8
        expected = mask_bytes + (active*(choices-1).bit_length()+7)//8
        side = data[offset+18:]
        if flags & 2:
            decoder = zstd.ZstdDecompressor()
            side = decoder.decompress(side, max_length=expected+1)
            if not decoder.eof or decoder.unused_data:
                raise ValueError("Incomplete block address stream")
        if len(side) != expected:
            raise ValueError("Block address stream has a different extent")
        mask = np.unpackbits(np.frombuffer(side[:mask_bytes], np.uint8), bitorder="little", count=count).astype(bool)
        if np.count_nonzero(mask) != active:
            raise ValueError("Block switch count differs")
        ids, end = unpacked(side, mask_bytes, active, choices)
        if end != len(side):
            raise ValueError("Unused block addresses")
        flat = np.full(count, -1, np.int32)
        flat[mask] = ids
        codes = flat.reshape(stripes, grid_rows).T if flags & 1 else flat.reshape(grid_rows, stripes)
        self.codes = np.ascontiguousarray(codes)
        self.lookup = None

    def apply(self, previous, part, site, background, row_ids, flipped, keys=None):
        rows,cols=previous.shape
        n=part.stop-part.start
        if site.shape!=(n,cols,2) or background.shape!=site.shape or not previous.flags.c_contiguous:
            raise ValueError("Invalid block field geometry")
        source=np.ascontiguousarray(site,"<u2")
        result=np.ascontiguousarray(background,"<u2")
        if np.shares_memory(source,result):
            result=result.copy()
        if (row_ids is None)!=(flipped is None) or row_ids is not None and (
                len(row_ids)!=rows or len(flipped)!=rows or row_ids.dtype!=np.int32
                or not row_ids.flags.c_contiguous or not flipped.flags.c_contiguous):
            raise ValueError("Invalid underlying row alignment")
        table=self.lookup if keys is not None else None
        if table is not None:
            keys=np.ascontiguousarray(keys,"<u2")
            if keys.shape!=(n,cols): raise ValueError("Invalid block context keys")
        status=native().block_intervals(previous.ctypes.data,rows,cols,part.start,n,
            self.width if self.row_word else 1,self.row_word or self.width,bool(self.row_word),self.global_ids,
            self.codes.ctypes.data,None if row_ids is None else row_ids.ctypes.data,
            None if flipped is None else flipped.ctypes.data,source.ctypes.data,self.factors.ctypes.data,
            None if table is None else keys.ctypes.data,None if table is None else table.ctypes.data,
            0 if table is None else table.shape[1],result.ctypes.data)
        if status: raise ValueError("Invalid native block reference")
        return result

    def next_rows(self, rows, columns, row_ids, flipped):
        if self.row_word or self.width != columns:
            return row_ids,flipped
        ids=np.arange(rows,dtype=np.int32) if row_ids is None else row_ids.copy()
        signs=np.zeros(rows,bool) if flipped is None else flipped.copy()
        codes=self.codes[:,0]
        chosen=codes>=0
        if self.global_ids:
            ids[chosen]=codes[chosen]>>1
            signs[chosen]=(codes[chosen]&1)!=0
        else:
            signs[chosen]^=(codes[chosen]&1)!=0
        return ids,signs
