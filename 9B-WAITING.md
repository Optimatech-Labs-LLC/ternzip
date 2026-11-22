# Ternzip 9B waiting and output-table measurements

On a 48 GB Apple M4 Pro running macOS 26.5, the packaged ordinary wait delivered 58.83 tokens/s. The optional `--fast` mode delivered 61.08 and used 98.9% of one CPU core. Both matched all 24 Prism token streams. The 1,834,157,871-byte archive restores the official PTQ1_0 file byte for byte.

## Waiting on the GPU

Each row covers eight English prompts replayed in three fresh processes, with 256 streamed tokens per chat. Each method had to match all 6,144 IDs, the 24 following-token and stopping decisions, and the 24 text hashes from Prism's Q2_0 reference. The four experimental modes rotated within prompts in a balanced order. CPU is the median of the per-chat CPU-time-to-wall-time ratios during generation. Battery energy was unmeasured. [Per-chat measurements](benchmark-9b-wait.json)

| Experimental mode | Median tokens/s | Chats at 60+ | Caller CPU | Process CPU |
| --- | ---: | ---: | ---: | ---: |
| Ordinary wait | 58.54 | 10/24 | 9.0% | 12.6% |
| Poll when the caller is on an efficiency core | 59.49 | 7/24 | 12.2% | 15.4% |
| Sleep to 14 ms from submission, then poll | 57.22 | 0/24 | 17.9% | 22.6% |
| Poll every token | 61.50 | 24/24 | 98.7% | 101.3% |

Efficiency-core polling beat the ordinary wait in 14/24 matched chats. Its median paired differences were +0.62 tokens/s and +2.1 percentage points of caller CPU. It improved both measures in 9/24. The deadline mode slept for a mean 14.13 ms and polled for 0.89 ms per generated token.

| Packaged loader, separate paired trial | Median tokens/s | Chats at 60+ | Caller CPU | Process CPU |
| --- | ---: | ---: | ---: | ---: |
| Ordinary wait (default) | 58.83 | 6/24 | 7.9% | 11.2% |
| Fast wait (--fast) | 61.08 | 23/24 | 98.9% | 101.3% |

The experimental poll inspected the existing compute buffer. The packaged mode uses the registered device’s command queue and an ordered completion marker. Both use the ordinary sampled-token read. We verified that the registered device and the live scheduler’s device were identical on all three starts. The release-path trial alternated the two modes within each prompt. From the packaged fast median, 65 tokens/s requires another 0.988 ms per token.

## Can the output table skip work exactly?

We captured 32 actual projection inputs per prompt, one every eight tokens. A separate Metal graph on the official output table reproduced every FP32 vocabulary logit for all 256 inputs (SHA-256 per vector). The table has 248,320 rows of 4,096 weights in native PTQ1_0, 28 bytes per 128 weights including the scale. That is 1.75 bits per weight.

We grouped 128 vocabulary rows into each tile and stored upward-rounded norms for contiguous feature groups. The whole-row test bounds a dot product with the sum of the input and weight group-norm products. The prefix test computes 2,560 of 4,096 features on Metal and bounds the remaining 1,536. Both use views of the same native weight buffer. Their combined optional norm tables occupy 54,320 bytes. The production loader contains neither table.

Even when given the true winning score and omitting floating-point safety margins, the whole-row bound excluded 0 of 63,569,920 row-token pairs. The prefix rule excluded 28,622 (0.045%). Each of those rows saves at most the remaining 37.5% of its multiplies, amounting to 0.017% of the table's work before paying for the check. Across both tests 0/256 tokens could skip a third of the projection work.

For an executable check we scored the first 128 rows as a shortlist, used its winner as a lower bound, required a strict inequality and added a conservative FP32 rounding allowance. The prefix rule could certify individual rows on 12/256 tokens, totaling 1,418 row-token pairs. If the shortlist winner could not be certified against all outside rows, the native full projection chose the token. The whole-row and prefix guards avoided the full table on 0/256 and 0/256 tokens. Both returned the Prism winner on all 256. The control took 1.799 ms, the whole-row guard 1.978 ms and the prefix guard 3.294 ms. These operator timings include synchronized computation, readback and host argmax. They are separate from live chat and the earlier 1.72 ms isolated matvec.

## Size and memory

The 2 GB budget applies to model weights, owned embedding data and cache, and the decoder reserve. They totaled 1,999,911,846 bytes, leaving 88,154 bytes. In the matched startup trial, warm load went from 17.51 s to 11.27 s and median peak process RSS from 5.372 GB to 2.876 GB. The 48 GB M4 Pro is the only tested machine. An 8 GB Mac remains untested.

Archive SHA-256 is `47f52b0d8e0ec4d38f61d61e7100ec3b17fec6dbe2fe58a7904fa4a57c07a66b`. All 1,997,486,976 restored bytes matched the official GGUF. Restored SHA-256 is `3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b`. The packaged CLI also verified the archive and answered the arithmetic smoke prompt in both modes with network access disabled.
