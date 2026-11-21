# Ternzip

Ternzip compresses Prism ML’s ternary language models. One command restores the original GGUF, including its block scales, tokenizer, metadata and padding. On Apple Silicon, the custom loader can also generate text directly from the archive.

| Prism checkpoint | Original GGUF | Ternzip download | Bytes saved |
| --- | ---: | ---: | ---: |
| Ternary Bonsai 2 9B · PTQ1_0 | 1,997,486,976 | 1,860,057,576 | 137,429,400 (6.88%) |
| Ternary Bonsai 8B · Q2_0 | 2,182,184,672 | 1,623,736,952 | 558,447,720 (25.59%) |

The 9B source already packs five ternary values per byte across most of each weight block. Its 6.88% saving is measured against that dense file. Each row uses its own official source as the baseline. [9B archive](https://huggingface.co/optimatechlabs/ternzip-9b) · [8B archive](https://huggingface.co/optimatechlabs/ternzip-8b)

## Restore or run Ternary Bonsai 2 9B

Download `ternzip-9b.trits`. With CPython 3.14 and stable Rust installed, run this from the Ternzip source directory. It writes `Ternary-Bonsai-2-9B-PTQ1_0.gguf` in the current directory. Set aside 2.00 GB for the output. The launcher caches its Python dependencies and builds the small native codecs. [9B download](https://huggingface.co/optimatechlabs/ternzip-9b/resolve/main/ternzip-9b.trits)

```sh
./ternzip restore /path/to/ternzip-9b.trits
```

The command checks the input archive, hashes the written GGUF and compares it with Prism’s published SHA-256 before making the output available. It preserves an existing output file. To choose a different location, supply a second path. To check the restore without writing a GGUF, run `./ternzip verify /path/to/ternzip-9b.trits`. [Prism’s published checksums](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf/blob/82265ccc78ddaac4700286a35e9badcc32376dd9/SHA256SUMS)

| 9B file | SHA-256 |
| --- | --- |
| Ternzip archive | `a53379d2406c3be10498475e9722818986e534e3bef5cd4690b35ae74bdbd40f` |
| Restored PTQ1_0 GGUF | `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b` |

For direct text generation on Apple Silicon, the custom loader uses a checksum-pinned Prism Metal runtime. It reads input embeddings on demand and keeps the compute matrices in their original PTQ1_0 layout. You can also run the restored GGUF with Prism’s llama.cpp fork. The optional vision projector and speculative decoding files are separate downloads from Prism. The released encoder profile covers the 8B model. [Prism’s runtime and file guide](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf/blob/82265ccc78ddaac4700286a35e9badcc32376dd9/README.md)

```sh
./ternzip chat /path/to/ternzip-9b.trits 'What is 17 × 6? Answer with the number only.'
```

The 9B archive stores all 427 original tensors. The coder uses row, column and scale information to estimate how often each weight value occurs. For 12 matrices it also uses signed row matches in earlier layers. These predictions change how many bits each value costs. The decoder recovers every stored value, including the values the predictor got wrong. [9B decoder](src/bonsai9.py) · [9B loader](src/bonsai9_metal.py)

On an Apple M4 Pro, the 9B loader generated **48.48 tokens/s** on a story and **47.12 tokens/s** on code after processing 4,096 input tokens. Both generated sequences and stopping tokens matched Prism’s control. The top choice and reported target-token loss matched at all **13,371** scored positions. Counted weights, embedding tables, row cache and decoder reserve totaled **1,861,761,958 bytes**. An earlier run of this layout measured about **2.37 GB** for the whole process after generation. [9B benchmark measurements](benchmark-9b.json)

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
