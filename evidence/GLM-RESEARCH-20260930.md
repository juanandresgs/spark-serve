# GLM research evidence: September 30, 2026

These records retain a completed research window. **No candidate is a production
finalist, and the reviewed GLM recommendation is unchanged.** The research snapshots
are not portable build recipes. This import does not publish the original private
source, logs, deployment inventory, checkpoint overlays or response contents.

Six recipe snapshots and twenty run/cohort records use the `glm-window-20260930-`
prefix. Sources are explicit-field sanitized snapshots bound to original receipt
SHA-256 digests. The private importer retains the path/hash provenance separately.
The original forty records and reviewed recommendations are preserved unchanged.

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| Graph-enabled vLLM baseline | Near-limit request aborted below the 5 GiB available-memory floor | Successful long-response latency |
| Eager vLLM near-limit request | 841,804 input and 8,192 output tokens; retrieval passed; TTFT 640.344 s; request wall time 1,022.980 s | Complete requested answer: output hit its cap |
| Eager continuation | Expected eight-token answer with 841,820 input tokens in 5.518 s | An instrumented cache hit: cached-token accounting was absent |
| Adaptive/fixed policy screen | Twelve C1/C4/C8 cohorts across four arms; actual capped-output throughput and median TTFT; one typed-tool pass per arm | Completed-task goodput, quality noninferiority or a policy winner; all measured answers capped and every comparison output hash differed from A1 |
| TensorFold startup trials | Graph-enabled starts crossed the memory floor; final eager startup reached readiness | Generation qualification from readiness alone |
| TensorFold near-limit request | Same user-content fixture; tokenizer counted 841,797 input tokens; client timed out at 326.978 s before any output | A speed regression, retrieval failure, or long-context success: deadline was shorter than vLLM's observed TTFT and server usage was absent |
| Compact-tail GPU harness | Twenty single-GPU pooling/ring/rollback/resume/graph cases, with exact source hashes | Full-model logits, two-rank equivalence, or model-scale memory use |

The TensorFold configuration has one serialized slot, BF16 latent cache, MTP-only
drafting and text input. It is not equivalent to the eight-slot, FP8-target-KV
vLLM production contract with tools/reasoning/vision. Recipe snapshots preserve
these differences and known immutable identities; unspecified software and image
metadata remains unknown.

Each policy arm had explicit prefix warmups, but measured cache-hit counters were
not retained. The records therefore keep cache state unknown. Throughput is actual
completion tokens divided by the whole mixed code/prose cohort wall time, including
all capped answers. No percentage gains are imported. Existing documentation tables
and graphs remain tied to their previously reviewed evidence.

Further research requires a separately authorized window: one brief smoke check,
then the full 850K context contract with an 8,192-token output reserve, early C4/C8
mixed long/short traffic, client/tool loops, reasoning, vision and cancellation.
Reserve time for full cold prefill, completed output and restoration. Small capped
screens cannot select a production winner. That future research is separate from
this completed evidence-format reconciliation.
