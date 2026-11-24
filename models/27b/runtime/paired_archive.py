from compression import zstd
def decompress(frame, limit):
    decoder = zstd.ZstdDecompressor()
    result = decoder.decompress(frame, max_length=limit + 1)
    if len(result) > limit or not decoder.eof or decoder.unused_data:
        raise ValueError("Compressed record does not fit the declared layout")
    return result

