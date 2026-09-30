# Performance, choices and comparison boundaries

Measured September 27–29, 2026 on DGX Spark GB10, 128 GB unified memory.
Production-selected September 30. Values below are observations, not hardware
capacity guarantees or proof of broad model quality. `results.json`, `quality.json`,
`history.json` and `measurements.json` retain sanitized numeric evidence.

## Latest matched comparison

Four slots, 262144 context, 8192 output allowance; thinking off for throughput and
context. Throughput is aggregate output tokens / whole group elapsed time,
including initial latency, with three repetitions of four clients and 512-token
caps. Quality uses thinking on, temperature1/top-p.95/top-k20. These compare whole
recipes: EXL3/TensorFold0.3.6.1 versus affine4/TensorFold0.3.6.2.

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
20 coding tasks it is the second slowest; EXL3's maximum was87.72s, affine256.97s.
Quality tasks are arithmetic, Python semantics, FIFO state and shortest paths,
plus20 independently authored executable functions with hidden tests. Not SWE-bench
or agentic repository work. Four repeated graph trials are excluded from unique
counts. The small two-question gap is not evidence of broad quality superiority.

## Why the selected policy is uncapped

The earlier greedy screen scored EXL398/100 and affine94/100. All six affine
failures exhausted8192 tokens without an answer and reproduced exactly with MTP
disabled. Recommended sampling resolved all six, with MTP/serial exact token parity.
Both packs then passed the original100. This confound supersedes a blanket
interpretation that affine quantization caused the failures.

A greedy2048 reasoning cap returned six answers, only four correct. With recommended
sampling, the optional cap preserved296/296 and20/20 in our screen. It reduced
coding p95 to96.65s but increased median to16.07s. Fresh reasoning median/p95 became
8.76/59.55s; maximum fell118.28→94.76s. Affine generated2.36× the coding tokens of
EXL3 without a measured coding-quality gain. Longer reasoning is not automatically
better. Use the cap only as an explicit workload-tested latency policy.

## Previous local versions

These historical measurements precede the latest comparison and use different
packs, topology, runtimes and/or fixture details. Do not combine into a matched A/B.

| Version | C4 code / prose aggregate tok/s | Near-limit completed retrieval |
|---|---:|---:|
| Two-Spark TP2, NVFP4/b12x, MTP4 | 237.3 /149.4 | 129.16 s at253826 input tokens |
| Single-Spark EXL3, one slot | 98.7 /50.4 | 534.37 s |
| Single-Spark EXL3, four slots, original scheduler | 192.5 /84.5 | 534.97 s |
| Cooperative EXL3, latest control | 161.08 /85.99 | 534.79 s at253843 input tokens |
| Selected Affine4, one Spark/four slots | 142.23 /92.64 | 261.62 s |

The original scheduler's mixed-load short-request p95 was356.36s; cooperative
scheduling reduced it to2.73s in the historical same-Spark crossover, with bulk
time371.66→376.19s. Affine then improved prefill alongside cooperative scheduling.
The two-Spark recipe remains faster on these throughput/context observations,
but consumes two Sparks. No unmeasured two-replica scaling claim is made.

## Public reference claims (not independently reproduced)

[MiaAI-Lab README, revision856bb6b](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e/README.md),
read September30, reports prose decode62.4tok/s atC1,106.7 atC4 and119.3 atC5;
131110-token prefill TTFT59.60s and roughly195K-token TTFT97s with4096-row chunks.
Its documented current default uses TensorFold0.3.6.3, five slots and vision/video.
Our selected recipe uses0.3.6.2, four slots,2048-row cooperative chunks, text only.
Public prose decode metrics are not our end-to-end group metric, and the prompts,
caches and output lengths are unmatched. Our254K completed retrieval is not their
195K TTFT test. These figures identify targets to reproduce, not a percentage
performance claim against their implementation. No Mia code/image was used.

## Verified and still open

The measured immutable image passed HTTP11/11, typed-tool/API36/36,
cancellation6/6, exact final-image replay9/9 and all six throughput repetitions.
Adapter tests10/10 and repository tests373/373 passed. Later quality/coding memory
samples saw16.71GiB available on affine versus31.10GiB on EXL3, no swap configured;
these are not entire-run minima. Broad coding quality, repeated-seed distributions,
endurance, cold reboot and vision remain unqualified. Source rebuild qualification
and production cutover receipts are described in BUILD-VALIDATION.md.
