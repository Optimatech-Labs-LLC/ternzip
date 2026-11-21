"""Run the packed model with file-backed input embeddings and greedy Metal decoding."""
import ctypes as C
import gc
import math
import time

from archive_io import ARCHIVE_SHA, ARCHIVE_SIZE, checked_input, is_bonsai9
from model_stream import PackedStream
from native_engine import NativeEngine, NativeTokenizer
from paired_archive import PairImage

WEIGHT_BUDGET = 2_000_000_000
DECODER_RESERVE = 8 * 1024 * 1024


def load_model(path, context):
    with checked_input(path, ARCHIVE_SIZE, ARCHIVE_SHA) as (file, data):
        image = PairImage(data)
        table = image.vocab_table("token_embd.weight", file=file, cache_pages=4, cache_rows=512)
        try:
            auxiliary = table.owned_bytes() + DECODER_RESERVE
            limit = WEIGHT_BUDGET - auxiliary + image.g.data_offset - 64 * 1024
            stream = PackedStream(image, limit)
            try:
                engine = NativeEngine(stream, context, max_outputs=1, prefill_chunk=256,
                                      backend_greedy=True, input_table=table)
            finally:
                stream.close()
        except BaseException:
            table.close()
            raise
        try:
            tokenizer = NativeTokenizer(engine, image.g.metadata)
            if not engine.weight_buffers_mib:
                raise RuntimeError("The backend did not report its weight allocations.")
            aligned = sum(math.ceil((mib + 0.005) * 1048576) for _, mib in engine.weight_buffers_mib)
            weight_bytes = max(engine.weight_bytes, aligned) + auxiliary
            if weight_bytes > WEIGHT_BUDGET:
                raise RuntimeError(f"The weight allocation exceeded its budget: {weight_bytes:,} bytes")
        except BaseException:
            engine.close()
            raise
    return engine, tokenizer, weight_bytes


def chat(path, prompt, context, maximum):
    if not 1 <= maximum < context:
        raise ValueError("The output limit must be positive and fit inside the context.")
    start = time.perf_counter()
    if is_bonsai9(path):
        from bonsai9 import ARCHIVE_SHA as archive_sha
        from bonsai9_metal import load
    else:
        load, archive_sha = load_model, ARCHIVE_SHA
    engine, tokenizer, weight_bytes = load(path, context)
    try:
        gc.collect()
        relief = getattr(C.CDLL(None), "malloc_zone_pressure_relief", None)
        if relief is not None:
            relief.argtypes = [C.c_void_p, C.c_size_t]
            relief.restype = C.c_size_t
            relief(None, 0)
        load_seconds = time.perf_counter() - start
        ids = tokenizer.chat_ids(prompt)
        if len(ids) + maximum > context:
            raise ValueError(f"The prompt and output limit require {len(ids) + maximum} context positions.")
        token = engine.prefix(ids)
        output = []
        started = time.perf_counter()
        while len(output) < maximum and token not in tokenizer.stop:
            output.append(token)
            token = engine.next(token)
        elapsed = time.perf_counter() - started
        return {"text": tokenizer.decode(output, skip_special_tokens=False),
                "input_tokens": len(ids), "output_tokens": len(output), "output_ids": output,
                "stop": "eos" if token in tokenizer.stop else "length",
                "load_seconds": load_seconds, "generation_tps": len(output) / elapsed if output else None,
                "weight_bytes_including_reserve": weight_bytes, "devices": engine.devices,
                "archive_sha256": archive_sha}
    finally:
        engine.close()
