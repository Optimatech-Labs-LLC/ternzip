# Ternzip

Ternzip compresses Prism ML’s ternary language models. One command restores the original GGUF, including its block scales, tokenizer, metadata and padding. On Apple Silicon, the custom loader can also generate text directly from the archive.

| Prism checkpoint | Original GGUF | Ternzip download | Bytes saved |
| --- | ---: | ---: | ---: |
| Ternary Bonsai 2 9B · PTQ1_0 | 1,997,486,976 | 1,834,157,871 | 163,329,105 (8.18%) |
| Ternary Bonsai 8B · Q2_0 | 2,182,184,672 | 1,623,736,952 | 558,447,720 (25.59%) |

The 9B download saving uses Prism’s dense PTQ1_0 as its baseline. The same Ternzip archive saves 24.11% against Prism’s PQ2_0 file and 28.26% against its Q2_0 file. Prism’s fork executes all three packings of the same ternary weights. Prism recommends PQ2_0 by default on Apple Silicon. Q2_0 also runs in upstream llama.cpp b11556 and later. [Prism’s packing guide](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf#choosing-a-packing) · [9B archive](https://huggingface.co/optimatechlabs/ternzip-9b) · [8B archive](https://huggingface.co/optimatechlabs/ternzip-8b)

## Restore Ternary Bonsai 2 9B

Download the 9B archive and the **v0.3.0 loader**. From the loader directory, with CPython 3.14 and stable Rust installed, run the following command. It writes `Ternary-Bonsai-2-9B-PTQ1_0.gguf` to the current directory and checks all 1,997,486,976 bytes against Prism’s SHA-256. Allow 2.00 GB for the output. To verify without writing a GGUF, use `./ternzip verify /path/to/ternzip-9b.trits`. [Versioned loader](https://github.com/Optimatech-Labs-LLC/ternzip/tree/v0.3.0) · [Prism’s SHA-256 manifest](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf/blob/82265ccc78ddaac4700286a35e9badcc32376dd9/SHA256SUMS)

```sh
./ternzip restore /path/to/ternzip-9b.trits
```

| File | SHA-256 |
| --- | --- |
| Ternzip archive | `47f52b0d8e0ec4d38f61d61e7100ec3b17fec6dbe2fe58a7904fa4a57c07a66b` |
| Restored PTQ1_0 GGUF | `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b` |

If an older loader prints `Expected 1,860,057,576`, it recognizes the v0.2.0 archive. Use the v0.3.0 source for this download. The matching launcher prints `ternzip v0.3.0` for `./ternzip --version`.

## Run Ternary Bonsai 2 9B on a Mac

Direct chat uses the custom loader on Apple Silicon. We tested a 48 GB M4 Pro and recommend 16 GB or more of unified memory for chat. An 8 GB Mac remains untested. The source launcher downloads the 2,552,087-byte Metal backend on first chat and verifies its SHA-256. The optional 29.65 MB Mac CLI preview includes that backend and works offline after download. [Mac preview and opening instructions](https://huggingface.co/optimatechlabs/ternzip-9b#optional-prebuilt-mac-cli-preview) · [Rebuild the Metal backend](backend/README.md#rebuild-on-apple-silicon)

The archive and loader can restore the model without an external checkpoint. Prism publishes the optional vision projector and speculative decoding files separately. [Prism’s packing guide](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf#choosing-a-packing)

```sh
./ternzip chat /path/to/ternzip-9b.trits 'What is 17 × 6? Answer with the number only.'
```

For the 9B archive, the coder uses each weight’s row, column and block scale to predict the frequencies of −1, 0 and +1. Selected matrices also use zero counts within blocks or signed row matches across layers. The scale encoder predicts original 16-bit scale words, then stores enough information to recover each bit. At load time the custom Metal runtime builds a mix of dense trits and two-bit slots, keeps input embedding rows file-backed and uses a lossless packed rotation. It runs the four-tap convolution across channels during ordinary single-token decoding.

The custom Metal runtime measured **60.12 tokens/s** and matched the Q2_0 control on **24/24** complete token streams and next or stopping decisions. On this workload its median rate was 9.0% above Prism’s default PQ2_0 and 9.1% above Prism Q2_0. The model uses the same ternary values. Its faster token generation comes from the execution layout and Metal kernels. File compression reduces the download and adds unpacking work at startup. A separate three-start check through the packaged loader matched all 24 streams and measured 59.94 tokens/s. Taken together, the measurements support about 60 tokens/s.

| Configuration | Download (bytes) | Median output tokens/s | Median warm load | Counted weight bytes | Matches vs. Q2_0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prism PTQ1_0 | 1,997,486,976 | 50.42 | 1.29 s | 2,053,638,719 | 24/24 |
| Prism PQ2_0 (Mac default) | 2,416,851,840 | 55.16 | 1.52 s | 2,473,006,204 | 24/24 |
| Prism Q2_0 | 2,556,640,128 | 55.09 | 1.53 s | 2,612,781,385 | reference |
| Ternzip v0.3.0 | 1,834,157,871 | 60.12 | 17.85 s | 1,999,911,846 | 24/24 |

One Apple M4 Pro, one greedy text stream, eight English prompts repeated across three fresh processes per configuration, up to 256 output tokens per chat. Thinking and speculation were disabled. Median output rate includes streamed detokenization and the next token or stop decision. Startup and prompt processing were timed separately. All files were on the same drive and were read once immediately before each timed load to warm the filesystem cache. Each startup median covers three starts with dependencies installed. The weight figures exclude context and other process memory. [Benchmark data](benchmark-9b.json) · [Metal source changes](backend)

The earlier three-start comparison measured 59.33 against 53.09 tokens/s for Q2_0 (11.75% higher). Median loads there were 21.00 s and 18.61 s, respectively, using different storage paths without the matched cache warm-up. The earlier 17.81 s observation was a single Ternzip start. Across prior sessions Ternzip’s 1.836 GB archive moved from 59.04 to 58.04 tokens/s and Prism Q2_0 from 52.16 to 53.09. Roughly 1 token/s of variation makes the old and new archive timings insufficient to rank those archives. Their execution weights and Metal runtime agree.

Created using Bonsai by Prism ML. Optimatech Labs modified the distribution by recompressing the official GGUF and adding a loader. This is an unofficial third-party release. Prism’s 9B checkpoint derives from Qwen3.5-9B, copyright 2026 Alibaba Cloud. [Apache License](LICENSE) · [Prism’s 9B notice](licenses/prism-bonsai-2-9b-NOTICE.txt) · [Qwen3.5 license](licenses/qwen3.5-9b-LICENSE.txt)

## Ternary Bonsai 8B

Ternzip packs Prism ML’s complete 2.18 GB ternary checkpoint into a 1.624 GB download. One command restores the official GGUF byte for byte. On Apple Silicon, the custom loader can also run the archive directly. [Download the archive](https://huggingface.co/optimatechlabs/ternzip-8b/resolve/main/ternzip-8b.trits) · [Official source file](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf/blob/c2aefbeb4b24469cd11579c3384b990404c17a30/Ternary-Bonsai-8B-Q2_0.gguf) · [Launch post on X](https://x.com/optimatechlabs/status/2123909382408679628)

| File | Bytes on disk |
| --- | ---: |
| Prism’s `Ternary-Bonsai-8B-Q2_0.gguf` | 2,182,184,672 |
| `ternzip-8b.trits` | 1,623,736,952 |
| **Saved** | **558,447,720 (558.45 MB, 25.59%)** |

The archive contains all 399 tensors, all 63,970,624 FP16 block scales, the tokenizer and GGUF metadata. The scales alone occupy 127,941,248 bytes in the original. Sizes use decimal GB and MB.

## How it works

Ternzip fits the probabilities of the three weight values (−1, 0 and +1) to this checkpoint using each weight’s row, column and block scale. For selected tensors, it also records which whole row or column in an earlier layer helps predict the current one, and whether to flip its signs. For example, column 121 of layer 1’s down projection uses sign-flipped column 46 of layer 0 (zero-based indices). A fast entropy coder called rANS uses those probabilities to encode the actual weights. Likely values cost fewer bits on average, and every mismatch still decodes exactly. [Ternzip predictor](src/predictor.py) · [rANS background](https://arxiv.org/abs/1311.2540)

For scales, 253 of the 254 ternary tensors use a sorted table of distinct FP16 bit patterns. Ternzip predicts each scale’s table index from a row average and a column correction, then stores the exact difference (the *residual*). Adding the residual back recovers the index and all 16 scale bits. The remaining tensor uses a table of common high bytes. [Scale encoder and decoder](src/residual_scales.py)

## Restore the official GGUF

Download the archive and open a terminal in the Ternzip source directory. On the tested Apple Silicon Mac, the prerequisites were CPython 3.14 and a stable Rust toolchain. Set aside 2.19 GB for the restored file. [Ternzip v0.1.0 source](https://github.com/Optimatech-Labs-LLC/ternzip/tree/v0.1.0)

```sh
./ternzip restore /path/to/ternzip-8b.trits
```

The command writes `Ternary-Bonsai-8B-Q2_0.gguf` in the current directory and verifies the written file against Prism’s published SHA-256. The full digest is `3c8d70470a5d97e5a2b9410ddd899cb740116591462626c60cb2fead6448f60b`. [Publisher’s file metadata](https://huggingface.co/api/models/prism-ml/Ternary-Bonsai-8B-gguf/tree/c2aefbeb4b24469cd11579c3384b990404c17a30?recursive=true&expand=true)

To check the same digest without writing the GGUF, use `verify`. To reproduce the archive from the official GGUF, use `pack`. Encoding uses the included fitted profile and libzstd 1.5.7. The decoder needs only the archive and the Ternzip software.

```sh
./ternzip verify /path/to/ternzip-8b.trits
./ternzip pack Ternary-Bonsai-8B-Q2_0.gguf ternzip-8b.trits
```

## Run on a Mac

Direct inference from `.trits` requires the **Ternzip custom loader**. It opens the archive, prepares Metal weight buffers and reads input embedding rows as needed. The launcher caches its Python dependencies and Rust libraries. Chat also fetches a checksum-pinned Prism runtime. Minarut’s 2.12 GB TQ2_0 GGUF runs in stock llama.cpp. [Prism Metal backend](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b10828-a153697) · [Minarut’s stock-compatible GGUF](https://huggingface.co/Minarut/Ternary-Bonsai-8B-GGUF-llamacpp-compatible)

```sh
./ternzip chat /path/to/ternzip-8b.trits 'What is 17 × 6? Answer with the number only.'
```

An Apple M4 Pro loaded the model in **about 19 seconds** (18.90 and 19.57 seconds with dependencies ready). Median generation speed was **61.38 output tokens/s** over ten greedy story and code runs with 4,096 input tokens and a 5,120-token context. Seven runs exceeded 60 tokens/s. The range was 58.64–62.02. Timing starts after prompt processing.

Model weights, the input embedding tables and cache, and the decoder reserve totaled 1,999,704,327 bytes. The whole process used 2.99–3.35 GB of physical memory after generation. Peak resident memory during startup was 5.29–5.30 GB. [Benchmark measurements](benchmark.json)

## Prior work

- TernaInfer studies compressed ternary weights for GPU inference. [Authors’ implementation](https://github.com/Gu-J/TernaInfer)
- Intel’s BITCOS stores a presence bitmap and the signs of the nonzero weights. [Georganas, Heinecke and Dubey](https://arxiv.org/abs/2609.16338)
- The llama.cpp TQ1_0 implementation packs five ternary values into a byte. [Ternary packing proposal and implementation](https://github.com/ggml-org/llama.cpp/pull/8151)
- James A. Storer and Harald Helfgott studied lossless block matching for images in 1997. [The Computer Journal](https://doi.org/10.1093/comjnl/40.2_and_3.137)

## License and attribution

Created using Bonsai by Prism ML. Optimatech Labs modified the distribution files by recompressing the official GGUF and supplying a separate loader. This is an unofficial, third-party recompression. Prism ML developed Ternary Bonsai 8B from Qwen3-8B, copyright 2024 Alibaba Cloud. The release uses Apache 2.0. The downloaded llama.cpp backend has its own MIT license. [Apache License](LICENSE) · [Prism NOTICE](NOTICE.txt) · [Qwen license and copyright](QWEN-LICENSE.txt) · [Backend license](licenses/llama.cpp-MIT.txt)
