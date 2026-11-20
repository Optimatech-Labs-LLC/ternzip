# Ternzip: lossless pack of Ternary Bonsai 8B

Ternzip packs Prism ML’s complete 2.18 GB ternary checkpoint into a 1.624 GB download. One command restores the official GGUF byte for byte. On Apple Silicon, the custom loader can also run the archive directly. [Download the archive](https://huggingface.co/optimatechlabs/ternzip-8b/resolve/main/ternzip-8b.trits) · [Official source file](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf/blob/c2aefbeb4b24469cd11579c3384b990404c17a30/Ternary-Bonsai-8B-Q2_0.gguf)

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
