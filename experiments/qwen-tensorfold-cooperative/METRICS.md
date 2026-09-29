# What improved, what did not

Observations from September 28, 2026 on DGX Spark/GB10. [Measurements](measurements.json)
and [historical comparison](history.json) contain numeric receipts without private
operational traces. Workloads, settings and cache state matter: these are bounded
experiments, not an assertion that this is the fastest public recipe.

## Change ledger

| Change | Observed benefit | Cost / what it does not prove |
|---|---|---|
| Two-Spark NVFP4/b12x TP2 → one-Spark EXL3/TensorFold | Frees one Spark; comparable short single-client code rate in the one-slot screen | Different pack, precision, engine and MTP; much slower cold long prompt and lower concurrent throughput. Not quality-equivalent by assumption |
| TensorFold one slot → four slots | Four-client code 98.7 → 192.5 end-to-end tok/s; prose 50.4 → 84.5 | Single-client code 97.9 → 86.3; prose 51.2 → 43.4. Four slots do not imply four maximum-context prompts fit |
| Checkpoint-derived EOS overlay | Correct stopping metadata; exact JSON fixtures pass after correction | Functional repair, no isolated speed benefit measured |
| Schema-aware tool arguments | 36/36 bounded API/function fixtures and native typed-tool/result loops passed after fixes | Converts schema-valid top-level values; malformed values remain for client rejection; not a universal tool-quality guarantee |
| Telemetry/discovery + preserved generator signature | Counters/model metadata available; explicit serial path works; real-client counter increments prove routing | No isolated speedup; metrics are compatibility counters, not fabricated engine telemetry |
| Cooperative 2048/burst4/head-skip bundle | Short p95 TTFT 356.36 → 2.73s; active-stream max pause 362.74 → 2.87s in separate test | Bulk time +1.22% / +0.67%; long-request first token later; still bursty; cold near-limit unchanged |
| 4096-row experiment | No material cold benefit | Cold reduction only ~0.33% vs control; short p95 3.88s; one recovered allocation warning; rejected |
| Intermediate-head skip | Avoids computing unused intermediate vocabulary logits | No independent ablation establishes a speed benefit |
| Initial2048/burst1/head-off → selected2048/burst4/head-on bundle | Complete short answers improved from roughly 9–13s to at most 5.82s in these runs | Multiple settings changed together. Do not assign the entire effect to burst4 |

## Historical topology/slot comparison

**End-to-end accounted output tokens/second**, including TTFT and queueing.
Not the isolated decode-rate metric used below. Output cap512; three sequential
requests per one-client cell, four/eight requests for concurrent cells.

| Workload | Two-Spark TP2 | One-Spark TF, 1 slot | One-Spark TF, 4 slots |
|---|---:|---:|---:|
| One code client | 96.5 | 97.9 | 86.3 |
| One prose client | 62.6 | 51.2 | 43.4 |
| Four code clients | 237.3 | 98.7 | 192.5 |
| Four prose clients | 149.4 | 50.4 | 84.5 |
| Eight code clients | 365.7 | 99.7 | 192.4 |
| Eight prose clients | 225.7 | 51.5 | 91.3 |

Eight-client median TTFT: TF one→four slots **18.43→5.75s code**, **34.90→10.64s
prose**; TP2 **0.50/0.41s**. The 253,826-input marker retrieval with 8,192 output
reservation generated only53 tokens: total latency **129.16s TP2**, **534.37s TF1**,
**534.97s TF4**. A strict appended turn reused253,826 input tokens and finished in
**2.221s TP2**, **1.379s TF1**, **1.318s TF4**. Exact replay on TF1 took533.795s;
that is not the same cache condition as a strict extension.

TP2 used NVFP4 experts, FP8 PLE/KV, MXFP8 head/HC and MTP4. TensorFold used EXL3
3.05bpw h5 ng5, mapped quantized ngram table and MTP6. Both allocated262,144 context.
These comparisons change several factors and precede the cooperative scheduler.
No extrapolation to two independent replicas is measured here.

## Same-Spark mixed-load crossover

Two fresh **131,096-input-token** retrievals, plus twenty **66-token** exact-JSON
requests arriving every6 seconds, starting2 seconds after launch. p95 is nearest
rank19 of20 (index18). Same checkpoint/settings/fixtures/seeds, kernels warmed;
22/22 prompt and generated-token fingerprints match, all22 correctness gates pass.

| Metric | Original scheduler | Selected bundle |
|---|---:|---:|
| Short TTFT median | 309.022s | 2.259s |
| Short TTFT p95 | 356.362s | 2.735s |
| Short TTFT maximum | 362.152s | 2.866s |
| Whole workload | 371.655s | 376.185s |
| Long completions | 365.903 / 365.906s | 375.637 / 376.177s |

**99.23%** short p95 reduction; **1.22%** longer total run. Slowest complete short
answer5.819s. The selected arm here used source-mounted adapters; the same source
and settings were then built into the separately tested final image. One paired
same-host session, not three alternating sessions or a statistical confidence interval.
The original long TTFTs were about182/364s, candidate about372/376s: scheduling
improves interference at the expense of the first long request's first token.

Earlier runs: clean original p95 **355.845s**; initial2048/burst1/head-off **2.727s**;
4096/burst4/head-on **3.883s**. The first original run (359.996s) overlapped artifact
staging; use the clean repeat and same-device crossover for the headline.

## Final-image stream interference

An already-streaming1,024-token code reply, then two fresh131,090-input retrievals.
One simultaneous comparison on separate Sparks; all3 prompt/output/token fingerprints
match. This is the packaged-image test, distinct from the22-request crossover.

| Metric | Original | Packaged candidate |
|---|---:|---:|
| Maximum output-chunk gap | 362.737s | 2.867s |
| p95 output-chunk gap | 0.0775s | 2.8195s |
| Code completion | 376.032s | 143.016s |
| Total3-request completion | 376.033s | 378.544s |

The giant baseline stall is hidden by p95. Candidate output remains bursty;
claim the reduction in worst observed pause, not improved p95 streaming smoothness.
Bulk completion increased **0.67%**.

## Cold prefill and decode

Prompt-cold means a new prefix, **not** a controlled OS reboot or cold-file-cache
condition. 253,841 actual input tokens, output cap/reserve8,192, context262,144.
All three-marker retrievals and appended continuations passed.

| Arm | Cold TTFT | Strict-extension TTFT |
|---|---:|---:|
| Original | 535.134s | 0.692s |
| Initial2048/burst1/head-off | 538.059s | 0.679s |
| 4096/burst4/head-on | 533.380s | 0.648s |
| Final image2048/burst4/head-on | 536.452s | 0.632s |

All follow-ups reused253,841 input tokens. There is **no material cold speedup**.
A pre-existing once-per-minute global cache-drop loop was unchanged in both arms;
its effect was not isolated. A separate cached-MTP/fresh-serial test at254,029 input
matched all128 generated token IDs (temperature0.7, seed4321); this is not an8192-token
near-limit generation test.

Median decode rates, three512-token-capped fixtures per kind: original **90.724 code /
42.425 prose tok/s**, selected **92.656 / 43.813**. Small sample, different accounting
from the historical end-to-end table; insufficient evidence for a general decode gain.

## Qualification boundaries

- Final-image36/36 API/tool/reasoning fixtures;16 token-ID cases matched original MTP,
  candidate MTP and candidate serial. No broad coding or quality-noninferiority study.
- Six cancellation/recovery cycles **0.575–0.710s** with one active prefill; full-slot
  cancellation and GPU-error recovery remain unqualified.
- Final-image available-memory minima about **30.86GiB candidate /31.10GiB control**,
  no swap and no new GPU/OOM fault lines in that final window. The earlier4096 arm
  had one recovered allocation warning; the whole trial was not fault-free.
- Original native Pi and Hermes typed-tool loops each increased candidate completed
  requests by exactly2, establishing actual candidate routing for those checks.
- No soak, reboot, four-full-context-slot qualification or universal5-second SLO.
  The public from-source image has not been GPU-built/tested as part of publication.

Raw operational logs are withheld; source-receipt hashes and public-safe structured
measurements support these observations. Hashes provide identity, not independent
replication. See the [Luna audit ledger](CLAIMS.md).
