# Ternzip v0.3.1 matched startup comparison

Ternzip loaded the 9B archive **6.24 seconds sooner** and used **2.50 GB less peak process memory at startup** on our 48 GB M4 Pro. The 1,834,157,871-byte archive restores Prism’s official PTQ1_0 file, including all metadata, scales and padding.

| Same Mac and archive | Released v0.3.0 | Released v0.3.1 |
| --- | ---: | ---: |
| Median warm startup | 17.51 s | 11.27 s |
| Median peak process memory at readiness | 5.372 GB | 2.876 GB |
| Chats matching Prism | 24/24 | 24/24 |
| Weights, embedding data, cache and decoder reserve | 1,999,911,846 bytes | 1,999,911,846 bytes |

The loader decodes bounded chunks directly into their final shared Metal buffers and reuses intermediate tables during unpacking. Both builds used the ordinary single-token generation loop. Eight English prompts were replayed across three fresh processes per build, with 256 generated tokens per chat and a 5,120-token context. All 6,144 token IDs, 24 following-token and stop decisions, and 24 streamed-text hashes matched Prism for each build.

We alternated the builds on macOS 26.5. Each process read the same archive from the same SSD to warm the filesystem cache, then the timer covered the loader’s own SHA-256 check, model and context creation, and allocator cleanup. The memory figure is process high-water RSS at readiness, in decimal GB. Cold startup and 8 GB Macs are untested.

Full restore produced 1,997,486,976 bytes. Its SHA-256 is `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b`. Archive SHA-256 is `47f52b0d8e0ec4d38f61d61e7100ec3b17fec6dbe2fe58a7904fa4a57c07a66b`.
