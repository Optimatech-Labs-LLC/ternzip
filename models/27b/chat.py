"""Compare a local GGUF with chat directly from the packed 27B archive on Metal."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from codec import Archive, MAGIC, read_header, sha
from layouts import PublishedView
import native_engine
from native_engine import NativeEngine, NativeTokenizer


PROMPTS = {
    "story": "Write a short story about a lighthouse keeper who finds a handwritten map.",
    "explanation": "Explain why the sky changes color at sunset to a curious ten-year-old.",
    "instructions": "I have a sunny windowsill and a packet of basil seeds. How do I get started?",
    "editing": "Rewrite this as a warm email to a neighbor: Your package was left on my porch. Come by after six.",
    "coding": "Write a short Python function that counts repeated words, and explain how it works.",
    "planning": "Plan a rainy Saturday morning at home with two kids who like science.",
    "comparison": "How does an induction stove heat a pan compared with an electric coil?",
    "reasoning": "There are three boxes labeled apples, pears, and mixed. Every label is wrong. How can one draw identify all three?",
}


class LocalGGUF:
    def __init__(self, path):
        self.path = path

    @contextmanager
    def open_c_file(self):
        lib = ctypes.CDLL(None)
        lib.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.fopen.restype = ctypes.c_void_p
        lib.fclose.argtypes = [ctypes.c_void_p]
        ptr = lib.fopen(os.fsencode(self.path), b"rb")
        if not ptr:
            raise OSError(f"Cannot open {self.path}")
        try:
            yield ptr
        finally:
            lib.fclose(ptr)


def chat(engine, tokenizer, text, limit):
    ids = tokenizer.chat_ids(text)
    start = time.perf_counter()
    token = engine.prefix(ids)
    prefill = time.perf_counter() - start
    tokens = []
    start = time.perf_counter()
    while len(tokens) < limit and token not in tokenizer.stop:
        tokens.append(int(token))
        token = engine.next(token)
    elapsed = time.perf_counter() - start
    return {"input_tokens": len(ids), "tokens": tokens, "text": tokenizer.decode(tokens),
            "following_id": int(token), "prefill_seconds": prefill, "decode_seconds": elapsed,
            "decode_tps": len(tokens) / elapsed if tokens else None}


def run(path, fmt, receipt, rounds, limit, prompt, verbose):
    started = time.perf_counter()
    with ExitStack() as stack:
        f = stack.enter_context(path.open("rb"))
        is_archive = f.read(8) == MAGIC
        f.seek(0)
        if is_archive:
            pool = stack.enter_context(ThreadPoolExecutor(6))
            base = Archive(f, pool)
            image = base if fmt == "PTQ1_0" else PublishedView(base, fmt)
            g = base.g
        else:
            image = LocalGGUF(path)
            g, _ = read_header(f)
        engine = NativeEngine(image, 1024, log=verbose, max_outputs=1, prefill_chunk=128,
                              backend_greedy=True, text_position_axes=4)
        stack.callback(engine.close)
        tokenizer = NativeTokenizer(engine, g.metadata)
        load_seconds = time.perf_counter() - started
        if is_archive:
            base.array = None
            base.index = None
            base.references.clear()
            if image is not base:
                image.array = None
                image.index = None
        prompts = {"user": prompt} if prompt else PROMPTS
        # The first invocation warms any lazy Metal pipelines outside the reported token rates.
        arithmetic = chat(engine, tokenizer, "What is 17 times 6? Reply with only the number.", 16)
        results = []
        for i in range(rounds):
            items = list(prompts.items())
            for name, text in items[i:] + items[:i]:
                result = chat(engine, tokenizer, text, limit)
                results.append(dict(round=i + 1, name=name, **result))
                if prompt:
                    print(result["text"], flush=True)
                else:
                    print(f"{i + 1}/{rounds} {name}: {len(result['tokens'])} tokens, "
                          f"{result['decode_tps']:.2f} tokens/s", flush=True)
        detail = {"source": "packed archive" if is_archive else "GGUF",
                  "requested_archive_view": fmt if is_archive else None,
                  "gguf_file_type": g.metadata.get("general.file_type"),
                  "devices": engine.devices, "reported_model_tensor_bytes": engine.weight_bytes,
                  "native_model_buffers_mib": engine.weight_buffers_mib,
                  "load_seconds": load_seconds, "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "context": 1024, "thinking": False, "sampling": "greedy", "max_new_tokens": limit,
                  "arithmetic_check": arithmetic, "prompt_text": prompts, "chats": results}
    detail["backend_sha256"] = sha(native_engine.LIB_DIR / "libllama.0.dylib")
    if receipt:
        receipt.write_text(json.dumps(detail, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--format", dest="fmt", choices=("PQ2_0", "PTQ1_0", "Q2_0"), default="PQ2_0")
    p.add_argument("--receipt", type=Path)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--limit", type=int, default=128)
    p.add_argument("--prompt")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.rounds < 1 or not 1 <= args.limit <= 768:
        p.error("Positive rounds and 1 to 768 output tokens are required")
    run(**vars(args))
