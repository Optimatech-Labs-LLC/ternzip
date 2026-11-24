# Ternzip for Ternary Bonsai 2 27B

One **5.512 GB** archive recreates all three official language GGUFs for Prism ML’s Ternary Bonsai 2 27B. They hold the same weights in three layouts and total **20.779 GB** as separate files. Keeping one archive for all three saves **15.267 GB (73.47%)** before any storage deduplication. Each restore matched Prism’s published SHA-256.

On our 48 GB M4 Pro, warm startup took **58.74 s** from the archive and **1.50 s** from a restored PQ2_0 file on the same external drive.

| Official language GGUF | Original bytes | Bytes saved | Reduction |
| --- | ---: | ---: | ---: |
| PTQ1_0 | 5,946,648,928 | 434,362,915 | 7.30% |
| PQ2_0 | 7,206,168,928 | 1,693,882,915 | 23.51% |
| Q2_0 | 7,626,008,928 | 2,113,722,915 | 27.72% |

The archive is 5,512,286,013 bytes. With the 26,860,929-byte Mac tools, the two downloads total 5,539,146,942 bytes (73.34% below the three-file total).

## Running on a Mac

Both inputs generated about 20 tokens/s on the same Metal backend. All 3,021 generated token IDs matched across 24 paired chats, as did the next-token decision after every reply.

| Input on the M4 Pro | Median decode | Median warm startup | Peak process memory |
| --- | ---: | ---: | ---: |
| Ternzip direct chat | 20.13 tokens/s | 58.74 s | 8.52–8.54 GB |
| Prism PQ2_0 GGUF | 19.87 tokens/s | 1.50 s | 8.04 GB |

The 0.26 tokens/s difference is within the run-to-run variation we observed. Across the three starts, the median ranged from 19.65 to 20.71 tokens/s for Ternzip and from 19.75 to 20.73 for Prism’s PQ2_0.

Both inputs were on the same external drive with file caching enabled. Startup ends when the inference context and tokenizer are ready. A separate internal SSD check measured **57.66 s** (3 starts, median) after a full-file hash. Unpacking took 56.20 s and reading the archive took 0.40 s.

For this short-chat test, eight prompts ran in three fresh processes per input, with a 1,024-token context and a 128-token greedy output cap. Seven replies reached the cap and one ended at 111 on each pass. Decode excludes prompt processing. The 9B test used a 256-token cap.

Both paths loaded 7,195,047,936 bytes of weight tensors. Process peaks are in decimal GB and cover startup and chat. We tested one 48 GB M4 Pro and recommend **at least 16 GB of unified memory** for direct archive chat. A 16 GB Mac is untested. The measured process peak alone was 7.95 GiB, only 51 MiB below the entire capacity of an 8 GiB Mac. Prism supplies the optional vision projector separately.

## How it works

In 97.27% of this checkpoint’s 128-weight ternary blocks, exactly 42 weights are zero. Encoding the count and using it to predict the remaining weights saved 86,002,538 bytes in a full-file comparison with the same probability tables. Once all the zeros in a block appear, only the signs remain to encode. Ternzip also predicts scales from rows and columns, and uses gate weights to predict their paired up weights. It stores the corrections, tokenizer and metadata.

The 27B loader uses six workers to unpack row bands into temporary CPU buffers, then copies them to the Metal runtime. The 9B loader has a dedicated 14-worker direct writer on this Mac.

## Restore and chat offline

Use the archive and `SHA256SUMS` with the matching **Ternzip v0.4.0** tools for Apple Silicon. [v0.4.0 release](https://github.com/Optimatech-Labs-LLC/ternzip/releases/tag/v0.4.0) · [Mac tools (.tar.gz)](https://github.com/Optimatech-Labs-LLC/ternzip/releases/download/v0.4.0/ternzip-27b-tools-macos-arm64.tar.gz)

```sh
shasum -a 256 -c SHA256SUMS
tar -xzf ternzip-27b-tools-macos-arm64.tar.gz
./ternzip-27b-macos-arm64/ternzip-27b restore ternzip-27b-final.trits my-27b-PTQ1_0.gguf
```

The default restore writes PTQ1_0. Add `--format PQ2_0` for Prism’s Apple execution layout or `--format Q2_0` for stock llama.cpp b11556 or newer. The matching 27B tools also run chat directly from the archive or from a restored PQ2_0 file.

```sh
./ternzip-27b-macos-arm64/ternzip-27b chat ternzip-27b-final.trits "What is 17 times 6? Reply with only the number."
```

| File | SHA-256 |
| --- | --- |
| Archive (`ternzip-27b-final.trits`) | `3f78ad4a4c6389ab4908eb3b64be80faaf95a48376bfb9bca728d0220aa679a2` |
| Mac tools v0.4.0 (`ternzip-27b-tools-macos-arm64.tar.gz`) | `63983995aeaf3c160014d4346382456e09cdae6af6f9db89d47b8d649129bb8b` |

Created using Bonsai by Prism ML. Prism derived the model from `Qwen/Qwen3.8-27B` and identifies the GGUF architecture as `qwen35`. OptimaTech Labs independently modified the storage format. Apache 2.0 applies, with Prism and Qwen/Alibaba notices included. [Prism’s original release](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) [Model citation](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf#citation)
