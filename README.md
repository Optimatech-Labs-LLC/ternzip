# Ternzip

Ternzip compresses Prism ML’s ternary language models. One command restores the original GGUF, including its block scales, tokenizer, metadata and padding. On Apple Silicon, the custom loader can also generate text directly from the archive.

| Prism checkpoint | Original GGUF | Ternzip download | Bytes saved |
| --- | ---: | ---: | ---: |
| Ternary Bonsai 2 9B · PTQ1_0 | 1,997,486,976 | 1,834,157,871 | 163,329,105 (8.18%) |
| Ternary Bonsai 8B · Q2_0 | 2,182,184,672 | 1,623,736,952 | 558,447,720 (25.59%) |

The 9B download saving uses Prism’s dense PTQ1_0 as its baseline. The same Ternzip archive saves 24.11% against Prism’s PQ2_0 file and 28.26% against its Q2_0 file. Prism’s fork executes all three packings of the same ternary weights. Prism recommends PQ2_0 by default on Apple Silicon. Q2_0 also runs in upstream llama.cpp b11556 and later. [Prism’s packing guide](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf#choosing-a-packing) · [9B archive](https://huggingface.co/optimatechlabs/ternzip-9b) · [8B archive](https://huggingface.co/optimatechlabs/ternzip-8b)

## Ternzip v0.3.1 on Apple Silicon

The matched v0.3.0 → v0.3.1 comparison reduced median warm startup from **17.51 s to 11.27 s** (35.6% less time). Median **peak process memory during startup** fell from **5.372 GB to 2.876 GB** (46.5% lower). The loader unpacks bounded chunks directly into their final Metal buffers. Each version had three fresh starts with a warm filesystem cache. [Matched startup measurements](https://github.com/Optimatech-Labs-LLC/ternzip/blob/v0.3.1/9B-STARTUP.md)

| v0.3.1 chat mode | Median tokens/s | Chats at 60+ | CPU used by the chat thread | Token streams matching Prism |
| --- | ---: | ---: | ---: | ---: |
| Ordinary waiting (default) | 58.83 | 6/24 | 7.9% of one core | 24/24 |
| `--fast` | 61.08 | 23/24 | 98.9% of one core | 24/24 |

Fast mode gained **2.25 tokens/s** and used about **one full CPU core** during generation. Ordinary waiting uses less CPU on battery. Battery life was unmeasured. Each mode streamed 256 tokens per chat on eight English prompts across three fresh processes. Every generated ID, following-token and stopping decision, and streamed-text hash matched the Prism Q2_0 reference. [Per-chat timings and CPU usage](https://github.com/Optimatech-Labs-LLC/ternzip/blob/v0.3.1/benchmark-9b-wait.json)

**Memory and hardware.** The **2 GB budget** covers weights, embedding data, cache and decoder reserve. These totaled **1,999,911,846 bytes**, leaving **88,154 bytes**. The 2.876 GB startup process peak includes other allocations. Direct chat requires Apple Silicon and Metal. We tested one **48 GB M4 Pro** running macOS 26.5. **8 GB Macs are untested**. A minimum RAM configuration has yet to be established. Allow additional memory for macOS, other apps and longer contexts.

### Prism comparison by session

| Earlier v0.3.0 three-format session | Download size (bytes) | Median tokens/s |
| --- | ---: | ---: |
| Prism PTQ1_0 | 1,997,486,976 | 50.42 |
| Prism PQ2_0 (Prism’s Mac default) | 2,416,851,840 | 55.16 |
| Prism Q2_0 | 2,556,640,128 | 55.09 |
| Ternzip v0.3.0 | 1,834,157,871 | 60.12 |

The three-format session tested all four configurations on the same prompts. PQ2_0 had Prism’s highest median in that session. A separate, earlier v0.3.0 comparison measured **59.33** for Ternzip and **53.09** for Prism Q2_0. The v0.3.1 waiting-mode session compared ordinary waiting with `--fast` and used saved Prism reference outputs to check exactness. [v0.3.0 benchmark data](https://github.com/Optimatech-Labs-LLC/ternzip/blob/v0.3.0/benchmark-9b.json) · [Prism’s packing guide](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf#choosing-a-packing)

## Restore Ternary Bonsai 2 9B

Download `ternzip-9b.trits` and the v0.3.1 loader. The source launcher requires CPython 3.14 and stable Rust. The prebuilt Apple Silicon CLI bundles its interpreter, codecs and Metal backend. From either loader directory, run this command with the path to your archive. Allow 2.00 GB of free disk space for the output. [9B archive](https://huggingface.co/optimatechlabs/ternzip-9b/resolve/main/ternzip-9b.trits?download=true) · [v0.3.1 source](https://github.com/Optimatech-Labs-LLC/ternzip/tree/v0.3.1) · [Prebuilt Mac CLI](https://github.com/Optimatech-Labs-LLC/ternzip/releases/download/v0.3.1/ternzip-macos-arm64-v0.3.1.tar.gz) · [Asset checksums](https://github.com/Optimatech-Labs-LLC/ternzip/releases/download/v0.3.1/SHA256SUMS-v0.3.1.txt)

```sh
./ternzip restore /path/to/ternzip-9b.trits
```

The command writes `Ternary-Bonsai-2-9B-PTQ1_0.gguf` and verifies all 1,997,486,976 bytes against Prism’s SHA-256. To check the restore in memory, run `./ternzip verify /path/to/ternzip-9b.trits`. [Prism’s published SHA-256](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf/blob/82265ccc78ddaac4700286a35e9badcc32376dd9/SHA256SUMS)

| File | SHA-256 |
| --- | --- |
| Ternzip 9B archive | `47f52b0d8e0ec4d38f61d61e7100ec3b17fec6dbe2fe58a7904fa4a57c07a66b` |
| Restored PTQ1_0 GGUF | `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b` |

The v0.3.1 loader accepts the v0.3.0 and v0.2.0 9B archive formats. Its version command prints `ternzip v0.3.1`. The v0.3.1 release uses the same 1,834,157,871-byte archive as v0.3.0.

## Run Ternary Bonsai 2 9B on a Mac

The custom loader can chat directly from the archive. Run either command from the extracted Mac CLI or the source directory.

```sh
./ternzip chat /path/to/ternzip-9b.trits 'Describe a quiet morning.'
./ternzip chat /path/to/ternzip-9b.trits 'Describe a quiet morning.' --fast
```

The source launcher fetches the v0.3.1 Metal backend on first chat and checks the SHA-256 of the bundle and every library inside it. For offline use, supply the local bundle with `TERNZIP_9B_BACKEND_ARCHIVE` or use the prebuilt CLI. The Mac CLI uses ad hoc signing and has no Apple notarization. [Exact Metal download](https://github.com/Optimatech-Labs-LLC/ternzip/releases/download/v0.3.1/ternzip-metal-macos-arm64-v0.3.1.tar.gz) · [Backend source and build instructions](backend/README.md) · [Mac CLI opening instructions](https://huggingface.co/optimatechlabs/ternzip-9b#optional-prebuilt-mac-cli)

The 9B archive stores all 427 tensors. Ternzip uses row, column and scale information to encode the three weight values. Selected matrices also use block zero counts or signed row matches across layers. The decoder recovers every original weight and scale bit. The runtime reads input embedding rows from the archive and runs the compute matrices in dense trits or two-bit slots. The archive saves 163,329,105 bytes against PTQ1_0. The generation measurements come from the Metal runtime.

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
