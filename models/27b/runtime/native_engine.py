"""Small Python interface to the pinned macOS arm64 Prism llama.cpp build."""

import ctypes as C
from pathlib import Path
import re
import sys

import numpy as np

I, U, F, P, B = C.c_int32, C.c_uint32, C.c_float, C.c_void_p, C.c_bool
FP = C.POINTER(F)
IP = C.POINTER(I)
LIB_DIR = Path(__file__).with_name("lib")
_active_logger = None


class ModelParams(C.Structure):
    _fields_ = (
        [(k, P) for k in ("dspark_head_source", "devices", "tensor_buft_overrides")]
        + [(k, I) for k in ("n_gpu_layers", "split_mode", "load_mode", "main_gpu")]
        + [
            (k, P)
            for k in (
                "tensor_split",
                "progress_callback",
                "progress_data",
                "kv_overrides",
            )
        ]
        + [
            (k, B)
            for k in (
                "vocab_only",
                "check_tensors",
                "use_extra_bufts",
                "no_host",
                "no_alloc",
                "load_mtp",
            )
        ]
    )


class ContextParams(C.Structure):
    _fields_ = (
        [
            (k, U)
            for k in (
                "n_ctx",
                "n_batch",
                "n_ubatch",
                "n_seq_max",
                "n_rs_seq",
                "n_outputs_max",
                "n_outputs_max_per_seq",
            )
        ]
        + [
            (k, I)
            for k in (
                "n_threads",
                "n_threads_batch",
                "ctx_type",
                "rope_scaling_type",
                "pooling_type",
                "attention_type",
                "flash_attn_type",
            )
        ]
        + [
            (k, F)
            for k in (
                "rope_freq_base",
                "rope_freq_scale",
                "yarn_ext_factor",
                "yarn_attn_factor",
                "yarn_beta_fast",
                "yarn_beta_slow",
            )
        ]
        + [
            ("yarn_orig_ctx", U),
            ("defrag_thold", F),
            ("cb_eval", P),
            ("cb_eval_user_data", P),
            ("type_k", I),
            ("type_v", I),
            ("path_kv_mean_center", P),
            ("abort_callback", P),
            ("abort_callback_data", P),
        ]
        + [
            (k, B)
            for k in (
                "embeddings",
                "offload_kqv",
                "no_perf",
                "op_offload",
                "swa_full",
                "kv_unified",
            )
        ]
        + [("samplers", P), ("n_samplers", C.c_size_t), ("ctx_other", P)]
    )


class Batch(C.Structure):
    _fields_ = [
        ("n_tokens", I),
        ("token", IP),
        ("embd", FP),
        ("pos", IP),
        ("n_seq_id", IP),
        ("seq_id", C.POINTER(IP)),
        ("logits", C.POINTER(C.c_int8)),
    ]


class SamplerChainParams(C.Structure):
    _fields_ = [("no_perf", B)]


class SamplerSequence(C.Structure):
    _fields_ = [("sequence", I), ("sampler", P)]


class BufferOverride(C.Structure):
    _fields_ = [("pattern", C.c_char_p), ("buft", P)]


def bind(lib, name, result, *args):
    fn = getattr(lib, name)
    fn.restype = result
    fn.argtypes = list(args)
    return fn


class NativeEngine:
    def __init__(
        self, image, context_length=4352, log=False, max_outputs=1, prefill_chunk=256,
        backend_greedy=False, input_table=None, lookup_draft=0,text_position_axes=1,
        recurrent_lookback=0, batch_greedy=False,
    ):
        global _active_logger
        self.model = self.ctx = None
        self.sampler = self.sampled_token = None
        self.input_table = input_table
        if text_position_axes not in (1,4):
            raise ValueError("Unsupported text position layout")
        self.text_position_axes=text_position_axes
        self.lookup = None
        if lookup_draft and max_outputs == 1:
            from lookup_decode import LookupDecode
            if not 1 <= lookup_draft < prefill_chunk:
                raise ValueError("Lookup verification needs a batch large enough for its draft")
            self.lookup = LookupDecode(lookup_draft)
            max_outputs = 1 + lookup_draft
        self.weight_buffers_mib = []
        self.context_buffers_mib = []
        self.hadamard_buffer_counts = None
        self.rotation_readback = None
        self.position = 0
        self.context_length = context_length
        self.max_outputs = max_outputs
        self.prefill_chunk = prefill_chunk
        if not 1 <= max_outputs <= prefill_chunk <= 256:
            raise ValueError(
                "Output capacity and prefill chunk must satisfy 1 <= outputs <= chunk <= 256"
            )
        self.llama = C.CDLL(str(LIB_DIR / "libllama.0.dylib"))
        ggml = C.CDLL(str(LIB_DIR / "libggml.0.dylib"))
        ggml_base = C.CDLL(str(LIB_DIR / "libggml-base.0.dylib"))
        log_type = C.CFUNCTYPE(None, I, C.c_char_p, P)

        def logger(level, message, _):
            if b"buffer size" in message and any(x in message for x in (b"KV buffer size",b"compute buffer size",b"output buffer size")):
                buffer = re.search(rb"(\S+)\s+(KV|compute|output) buffer size\s*=\s*([\d.]+) MiB",message)
                if buffer:
                    self.context_buffers_mib.append((buffer[1].decode(),buffer[2].decode(),float(buffer[3])))
            if b"Hadamard-folded weight" in message:
                counts=re.search(rb"using (\d+) rotation\(s\) and (\d+) sign vector\(s\)",message)
                if counts:
                    self.hadamard_buffer_counts=tuple(map(int,counts.groups()))
            if b"ternzip rotation readback" in message:
                checked=re.search(rb"readback: (\d+) entries, (\d+) allocated bytes verified",message)
                if checked:self.rotation_readback=tuple(map(int,checked.groups()))
            if b"model buffer size" in message:
                match = re.search(rb"(\S+) model buffer size\s*=\s*([\d.]+) MiB", message)
                if match:
                    self.weight_buffers_mib.append((match[1].decode(), float(match[2])))
            if log or level == 4:
                sys.stderr.write(message.decode(errors="replace"))

        self.logger = log_type(logger)
        _active_logger = self.logger
        bind(self.llama, "llama_log_set", None, log_type, P)(self.logger, None)
        bind(ggml_base, "ggml_log_set", None, log_type, P)(self.logger, None)
        bind(ggml, "ggml_backend_load_all", None)()
        bind(self.llama, "llama_backend_init", None)()
        count = bind(ggml, "ggml_backend_dev_count", C.c_size_t)()
        device = bind(ggml, "ggml_backend_dev_get", P, C.c_size_t)
        device_type = bind(ggml, "ggml_backend_dev_type", I, P)
        description = bind(ggml, "ggml_backend_dev_description", C.c_char_p, P)
        self.devices = [
            description(device(i)).decode()
            for i in range(count)
            if device_type(device(i)) in (1, 2)
        ]
        if not self.devices:
            raise RuntimeError("Apple Metal is unavailable to this process")
        settings = bind(self.llama, "llama_model_default_params", ModelParams)()
        settings.n_gpu_layers = 99
        settings.load_mode = 0
        if input_table is not None:
            metal = next(device(i) for i in range(count) if device_type(device(i)) in (1, 2))
            buft = bind(ggml, "ggml_backend_dev_buffer_type", P, P)(metal)
            if not buft:
                raise RuntimeError("Could not select the shared Metal weight buffer")
            self.vocab_override = (BufferOverride * 2)(BufferOverride(b"^token_embd[.]weight$", buft), BufferOverride())
            settings.tensor_buft_overrides = C.cast(self.vocab_override, P)
        load = bind(self.llama, "llama_model_load_from_file_ptr", P, P, ModelParams)
        if hasattr(image, "open_c_file"):
            with image.open_c_file() as stream:
                self.model = load(stream, settings)
        else:
            lib_c = C.CDLL(None)
            stream = bind(lib_c, "fmemopen", P, P, C.c_size_t, C.c_char_p)(
                C.addressof(C.c_char.from_buffer(image)), len(image), b"rb"
            )
            if not stream:
                raise OSError("Could not open the in-memory packed GGUF")
            try:
                self.model = load(stream, settings)
            finally:
                bind(lib_c, "fclose", I, P)(stream)
        if not self.model:
            raise RuntimeError("Prism could not load the ternary execution image")
        try:
            self.weight_bytes = bind(self.llama, "llama_model_size", C.c_uint64, P)(
                self.model
            )
            self.parameters = bind(self.llama, "llama_model_n_params", C.c_uint64, P)(
                self.model
            )
            vocab = bind(self.llama, "llama_model_get_vocab", P, P)(self.model)
            self.vocab = vocab
            self.vocab_size = bind(self.llama, "llama_vocab_n_tokens", I, P)(vocab)
            max_context = bind(self.llama, "llama_model_n_ctx_train", I, P)(self.model)
            if not 1 <= context_length <= max_context:
                raise ValueError(f"Context length must be between 1 and {max_context}")
            cp = bind(self.llama, "llama_context_default_params", ContextParams)()
            cp.n_ctx = context_length
            cp.n_batch = cp.n_ubatch = prefill_chunk
            cp.n_seq_max = 1
            if recurrent_lookback:
                if not lookup_draft or recurrent_lookback < lookup_draft:
                    raise ValueError("Recurrent snapshots must cover the draft")
                cp.n_rs_seq = recurrent_lookback
            cp.n_outputs_max = cp.n_outputs_max_per_seq = max_outputs
            if backend_greedy and (max_outputs == 1 or self.lookup is not None or batch_greedy):
                default = bind(self.llama, "llama_sampler_chain_default_params", SamplerChainParams)()
                self.sampler = bind(self.llama, "llama_sampler_chain_init", P, SamplerChainParams)(default)
                if not self.sampler:
                    raise RuntimeError("Could not allocate the native sampler chain")
                greedy = bind(self.llama, "llama_sampler_init_greedy", P)()
                if not greedy:
                    raise RuntimeError("Could not allocate the native greedy sampler")
                bind(self.llama, "llama_sampler_chain_add", None, P, P)(self.sampler, greedy)
                self.sampler_config = SamplerSequence(0, self.sampler)
                cp.samplers = C.cast(C.pointer(self.sampler_config), P)
                cp.n_samplers = 1
                self.sampled_token = bind(self.llama, "llama_get_sampled_token_ith", I, P, I)
            self.ctx = bind(self.llama, "llama_init_from_model", P, P, ContextParams)(
                self.model, cp
            )
            if not self.ctx:
                raise RuntimeError("Could not allocate the inference context")
            self.memory = bind(self.llama, "llama_get_memory", P, P)(self.ctx)
            self.clear_memory = bind(self.llama, "llama_memory_clear", None, P, B)
            self.remove_sequence = bind(self.llama, "llama_memory_seq_rm", B, P, I, I, I)
            self.forward = bind(self.llama, "llama_decode", I, P, Batch)
            self.logits = bind(self.llama, "llama_get_logits", FP, P)
            self.last_logits = bind(self.llama, "llama_get_logits_ith", FP, P, I)
            if self.input_table is not None:
                self.input_row = np.empty(self.input_table.width, np.float32)
                self.single_position = (I*self.text_position_axes)()
                self.single_output = C.c_int8(1)
                self.single_batch = Batch(1, None, self.input_row.ctypes.data_as(FP),
                                          C.cast(self.single_position,IP), None, None,
                                          C.pointer(self.single_output))
        except BaseException:
            self.close()
            raise

    def clear(self):
        self.clear_memory(self.memory, True)
        self.position = 0
        if self.input_table is not None:
            self.input_table.reset_cache()

    def rewind(self, position):
        if not 0 <= position <= self.position:
            raise ValueError("Invalid target KV rollback position")
        if position < self.position and not self.remove_sequence(self.memory, 0, position, -1):
            raise RuntimeError("The target backend could not remove the rejected KV suffix")
        self.position = position

    def _submit(self, tokens, all_logits=False):
        ids = np.ascontiguousarray(tokens, dtype=np.int32)
        n = len(ids)
        if all_logits and n > self.max_outputs:
            raise ValueError(
                "This context has fewer output rows than the requested batch"
            )
        if (
            ids.ndim != 1
            or n < 1
            or n > self.prefill_chunk
            or self.position + n > self.context_length
        ):
            raise ValueError("Invalid decode batch length or context overflow")
        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError("Token ID outside model vocabulary")
        positions = np.arange(self.position, self.position + n, dtype=np.int32)
        if self.text_position_axes==4:
            # Qwen3.5 M-RoPE uses three identical text coordinates and a zero fourth plane.
            positions=np.tile(positions,4)
            positions[3*n:]=0
        flags = np.full(n, int(all_logits), dtype=np.int8)
        flags[-1] = 1
        embeddings = self.input_table.embedding_batch(ids) if self.input_table is not None else None
        batch = Batch(
            n,
            ids.ctypes.data_as(IP) if embeddings is None else None,
            None if embeddings is None else embeddings.ctypes.data_as(FP),
            positions.ctypes.data_as(IP),
            None,
            None,
            flags.ctypes.data_as(C.POINTER(C.c_int8)),
        )
        if self.forward(self.ctx, batch):
            raise RuntimeError("Native model forward pass failed")
        self.position += n
        return n

    def decode(self, tokens, all_logits=False):
        if self.sampler:
            raise ValueError("A logits query requires a context created without backend sampling")
        n = self._submit(tokens, all_logits)
        if all_logits:
            return np.ctypeslib.as_array(
                self.logits(self.ctx), shape=(n, self.vocab_size)
            )
        return np.ctypeslib.as_array(
            self.last_logits(self.ctx, -1), shape=(self.vocab_size,)
        )

    def prefix(self, ids):
        if not len(ids):
            raise ValueError("A prompt must have at least one token")
        self.clear()
        if self.lookup is not None:
            self.lookup.start(ids)
        if self.sampler:
            for start in range(0, len(ids), self.prefill_chunk):
                self._submit(ids[start:start + self.prefill_chunk])
            return self.backend_token()
        for start in range(0, len(ids), self.prefill_chunk):
            logits = self.decode(ids[start : start + self.prefill_chunk])
        return int(np.argmax(logits))

    def next(self, token):
        if self.lookup is not None:
            return self.lookup.advance(self, token)
        return self.next_once(token)

    def verify_greedy(self, inputs):
        if self.sampler:
            self._submit(inputs, all_logits=True)
            choices = [self.sampled_token(self.ctx,i) for i in range(len(inputs))]
            if any(token < 0 or token >= self.vocab_size for token in choices):
                raise ValueError("GPU greedy sampler returned an invalid draft decision")
            return choices
        return np.argmax(self.decode(inputs,all_logits=True),axis=1).tolist()

    def next_once(self, token):
        if self.input_table is not None:
            if not 0 <= token < self.vocab_size or self.position >= self.context_length:
                raise ValueError("Invalid input token or context overflow")
            self.input_table.embedding_into(token, self.input_row)
            self.single_position[0]=self.position
            if self.text_position_axes==4:
                self.single_position[1]=self.single_position[2]=self.position
            if self.forward(self.ctx, self.single_batch):
                raise RuntimeError("Native model forward pass failed")
            self.position += 1
            if self.sampler:
                return self.backend_token()
            logits = np.ctypeslib.as_array(self.last_logits(self.ctx, -1), shape=(self.vocab_size,))
            return int(np.argmax(logits))
        if self.sampler:
            self._submit([token])
            return self.backend_token()
        return int(np.argmax(self.decode([token])))

    def backend_token(self):
        token = self.sampled_token(self.ctx, -1)
        if not 0 <= token < self.vocab_size:
            raise RuntimeError("The backend did not produce a greedy token")
        return token

    def close(self):
        if self.ctx:
            bind(self.llama, "llama_free", None, P)(self.ctx)
            self.ctx = None
        if self.sampler:
            bind(self.llama, "llama_sampler_free", None, P)(self.sampler)
            self.sampler = None
        if self.model:
            bind(self.llama, "llama_model_free", None, P)(self.model)
            self.model = None
        if self.input_table is not None:
            self.input_table.close()
            self.input_table = None


class NativeTokenizer:
    """Reuse the tokenizer already resident in the native model."""

    def __init__(self, model, metadata):
        from chat_prompt import ChatPrompt

        self.model = model
        self.prompt = ChatPrompt(metadata.get("tokenizer.chat_template"))
        self._tokenize = bind(
            model.llama, "llama_tokenize", I, P, C.c_char_p, I, IP, I, B, B
        )
        self._detokenize = bind(
            model.llama, "llama_detokenize", I, P, IP, I, P, I, B, B
        )
        end_of_text = self.encode("<|endoftext|>")
        if len(end_of_text) != 1:
            raise ValueError("Expected the Qwen end-of-text token")
        self.stop = {metadata["tokenizer.ggml.eos_token_id"], end_of_text[0]}

    def encode(self, text):
        encoded = text.encode("utf-8")
        capacity = max(32, len(encoded) + 1)
        ids = np.empty(capacity, np.int32)
        count = self._tokenize(
            self.model.vocab,
            encoded,
            len(encoded),
            ids.ctypes.data_as(IP),
            capacity,
            False,
            True,
        )
        if count < 0:
            capacity = -count
            ids = np.empty(capacity, np.int32)
            count = self._tokenize(
                self.model.vocab,
                encoded,
                len(encoded),
                ids.ctypes.data_as(IP),
                capacity,
                False,
                True,
            )
        if count < 0:
            raise ValueError("Tokenization failed")
        return ids[:count].tolist()

    def chat_ids(self, prompt):
        return self.encode(self.prompt.render(prompt))

    def decode(self, tokens, skip_special_tokens=True):
        if not len(tokens):
            return ""
        ids = np.ascontiguousarray(tokens, np.int32)
        capacity = max(32, 8 * len(ids))
        for _ in range(2):
            output = C.create_string_buffer(capacity)
            count = self._detokenize(
                self.model.vocab,
                ids.ctypes.data_as(IP),
                len(ids),
                output,
                capacity,
                skip_special_tokens,
                not skip_special_tokens,
            )
            if count >= 0:
                return output.raw[:count].decode("utf-8", errors="replace")
            capacity = -count
        raise ValueError("Detokenization failed")
