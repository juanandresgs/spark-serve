# Performance, choices and comparison boundaries

Measured September 27–29, 2026 on DGX Spark GB10, 128 GB unified memory.
Production-selected September 30. Values below are observations, not hardware
capacity guarantees or proof of broad model quality. `results.json`, `quality.json`,
`history.json` and `measurements.json` retain sanitized numeric evidence.

## Latest matched comparison

Four slots, 262144 context, 8192 output allowance; thinking off for throughput and
context. Throughput is aggregate output tokens / whole group elapsed time,
including initial latency, with three repetitions of four clients and 512-token
caps. Quality uses thinking on, temperature 1 / top-p .95 / top-k 20. These compare whole
recipes: EXL3 / TensorFold 0.3.6.1 versus Affine4 / TensorFold 0.3.6.2.

| Metric | Cooperative EXL3 | Selected Affine4 |
|---|---:|---:|
| C4 code aggregate tok/s, median | 161.08 | 142.23 |
| C4 prose aggregate tok/s, median | 85.99 | 92.64 |
| Code repetition range | 159.42–163.86 | 139.99–151.82 |
| Prose repetition range | 85.64–87.65 | 89.39–93.23 |
| Cold 253843-token prompt, completed JSON | 534.79 s | 261.62 s |
| Cached appended turn, completed JSON | 1.710 s | 1.245 s |
| Two ~131K prompts +20 short arrivals, total | 376.60 s | 144.61 s |
| Mixed short-request latency p50/p95 | 2.440/2.721 s | 0.705/1.115 s |
| Mixed correctness | 22/22 | 22/22, one full JSON fence normalized |
| Distinct reasoning questions | 294/296 | 296/296 |
| Fresh distinct subset | 194/196 | 196/196 |
| Executable pure-function coding tasks | 20/20 | 20/20 |
| Fresh reasoning median/p95 | 7.09/34.31 s | 8.56/58.61 s |
| Coding median/p95 | 12.67/25.06 s | 14.80/243.56 s |

Cold context means a fresh prompt, not an OS-cold SSD/cache. Both arms produced
53 output tokens. Affine's explicit JSON validation buffers, so compare completed
answers rather than its first visible content against EXL3's first token.
Other workload timings are single matched runs. P95 uses nearest rank: with only
20 coding tasks it is the second slowest. Maximum latencies are not retained in
the public coding receipts and are not asserted here.
Quality tasks are arithmetic, Python semantics, FIFO state and shortest paths,
plus 20 independently authored executable functions with hidden tests. Not SWE-bench
or agentic repository work. Four repeated graph trials are excluded from unique
counts. The small two-question gap is not evidence of broad quality superiority.
The historical reasoning comparison uses the same recorded request settings,
but the effective template effort for those exact images is unverified; do not
read those scores or latencies as a matched-effort quality comparison.

## Why the selected policy is uncapped

The earlier greedy screen scored EXL3 98/100 and Affine4 94/100. All six affine
failures exhausted 8192 tokens without an answer and reproduced exactly with MTP
disabled. Recommended sampling resolved all six, with MTP/serial exact token parity.
Both packs then passed the original 100. This confound supersedes a blanket
interpretation that affine quantization caused the failures.

A greedy 2048 reasoning cap returned six answers, only four correct. With recommended
sampling, the optional cap preserved 296/296 and 20/20 in our screen. It reduced
coding p95 to96.65s but increased median to16.07s. Fresh reasoning median/p95 became
8.76/59.55s; maximum fell 118.28 → 94.76 s. Affine generated 2.36× the coding tokens of
EXL3 without a measured coding-quality gain. Longer reasoning is not automatically
better. Use the cap only as an explicit workload-tested latency policy.

## Previous local versions

These historical measurements precede the latest comparison and use different
packs, topology, runtimes and/or fixture details. Do not combine into a matched A/B.

| Version | C4 code / prose aggregate tok/s | Near-limit completed retrieval |
|---|---:|---:|
| Two-Spark TP2, NVFP4/b12x, MTP4 | 237.3 / 149.4 | 129.16 s at 253826 input tokens |
| Single-Spark EXL3, one slot | 98.7 / 50.4 | 534.37 s |
| Single-Spark EXL3, four slots, original scheduler | 192.5 / 84.5 | 534.97 s |
| Cooperative EXL3, latest control | 161.08 / 85.99 | 534.79 s at 253843 input tokens |
| Selected Affine4, one Spark/four slots | 142.23 / 92.64 | 261.62 s |

The original scheduler's mixed-load short-request p95 was 356.36 s; cooperative
scheduling reduced it to 2.73 s in the historical same-Spark crossover, with bulk
time 371.66 → 376.19 s. Affine then improved prefill alongside cooperative scheduling.
In those older recorded runs, the two-Spark TP2 recipe had higher C4 rates and a
faster near-limit completion; they are not matched against the later one-Spark
recipes. No unmeasured two-replica scaling claim is made.

## Public reference claims (not independently reproduced)

[MiaAI-Lab README, revision 856bb6b](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e/README.md),
read September 30, reports prose decode 62.4 tokens/s at C1, 106.7 at C4 and 119.3 at C5;
131,110-token prefill TTFT 59.60 s and roughly 195K-token TTFT 97 s with 4096-row chunks.
Its documented current default uses TensorFold 0.3.6.3, five slots and vision/video.
Our selected recipe uses 0.3.6.2, four slots, 2048-row cooperative chunks, text only.
Public prose decode metrics are not our end-to-end group metric, and the prompts,
caches and output lengths are unmatched. Our 254K completed retrieval is not their
195K TTFT test. These figures identify targets to reproduce, not a percentage
performance claim against their implementation. No Mia code/image was used.

## Verified and still open

The measured immutable image passed HTTP 11/11, typed-tool/API 36/36,
cancellation 6/6, exact final-image replay 9/9 and all six throughput repetitions.
Adapter tests 10/10 and repository tests 373/373 passed. Later quality/coding memory
samples saw 16.71 GiB available on affine versus 31.10 GiB on EXL3, no swap configured;
these are not entire-run minima. Broad coding quality, repeated-seed distributions,
endurance, cold reboot and vision remain unqualified. Source rebuild qualification
and production cutover receipts are described in BUILD-VALIDATION.md.
