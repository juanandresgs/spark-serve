# spark-serve

**Choose a one-Spark Qwen recipe or a two-Spark GLM recipe.**

Build a local, OpenAI-compatible model server with pinned sources and settings,
practical run guides, and measurements you can inspect. Start with the recipes
below, then choose the tradeoffs that suit your workload. Download model weights
separately from their publishers.

{{choices}}

These are our recommended starting points for each model. They are deployment
choices; the measurements below do not establish performance parity between
one- and two-Spark systems. Context and request
counts are configured limits, not a promise of simultaneous maximum-length
capacity. The results below help choose a **recipe**; they do not rank the two
models' quality.

## Choose your setup

```sh
git clone https://github.com/juanandresgs/spark-serve.git
cd spark-serve
```

- **One Spark → [patched TensorFold 0.6 + EXL3 source guide](experiments/qwen-tensorfold-native-0.6/README.md).**
  Follow its pinned build and run steps. It is a **standalone Docker recipe**; the broker's
  `recipes prepare` cannot install it.
- **Two Sparks → [GLM adaptive DFlash2 guide](recipes/glm53-flash-adaptive-2spark/README.md).**
  Follow the [deployment walkthrough](REPRODUCTION.md) to supply your host,
  storage and network bindings, build the runtime, and validate your pair.
  The included broker manages admission, start, stop and recovery for managed
  recipes. New deployments need their own acceptance checks.

Want to compare options first? Run this from the checkout; it works offline:

```sh
PYTHONPATH=src python3 -m spark_serve recipes options
# Narrow the list with --model qwen or --model glm; add --json before recipes.
```

## Qwen fixed-cap speed screen · October 1

TensorFold is the serving runtime; EXL3 and Affine4 are different weight formats. This compares complete one-Spark recipes, not isolated runtime or quantization effects. C1 has one active request; C4 shows the combined rate for four concurrent clients.

{{qwen-final-v2-throughput}}

{{qwen-final-v2-speed}}

Across these measured recipes, patched TensorFold 0.6 + EXL3 had the highest C4 aggregate throughput; Affine4 returned faster cold long-prompt responses in separate tests, with strict-format failures at some sizes. These are complete recipes with different runtime and weight settings, so choose based on throughput, long-prefill latency and format requirements.

<details>
<summary>Test method, recipe settings and observed ranges</summary>

The `cell-warmup-v2` schedule used four measured groups per cell, each containing
four requests, after 17 excluded warmups. C1 sent the four requests serially;
C4 sent them concurrently. Rates divide all output tokens by complete group
wall time, including prefill. The selected production run used HTTP admission 8
with four engine slots. No cache reset was performed. C4 is an aggregate
group rate, never a per-client rate. The four-slot engine setting describes
backend capacity; gateway HTTP admission is separate.

Patched TensorFold 0.6.0 + EXL3 used BF16 KV, native memory-managed prefix defaults,
burst4 and 1024-row prefill. The previous cooperative EXL3 recipe used TensorFold 0.3.6.1, BF16 KV, a
2 GiB prompt cache and snapshots disabled. Affine4 used TensorFold 0.3.6.2, a
different 4-bit weight pack and int8 KV. The production primary used gateway admission 8. Earlier maintenance-primary and second-host speed runs used admission 4; all are independent results and are never pooled. Run records below link the exact
medians, per-cell ranges and source identity boundary.

Run receipts: [patched TensorFold 0.6 production primary](evidence/runs/production-final1024-http8-speed-v2-20261001.json), [earlier maintenance primary](evidence/runs/native-primary-speed-v2-20261001.json), [independent second-host maintenance run](evidence/runs/native-replica-speed-v2-20261001.json), [cooperative EXL3](evidence/runs/qwen-cooperative-exl3-speed-v2-20261001.json), and [Affine4](evidence/runs/qwen-affine4-speed-v2-20261001.json). Each receipt links its public source projection with group totals and provenance.

</details>

<details>
<summary>Per-request decode proxies and the second native-host replication</summary>

{{qwen-final-v2-decode}}

{{qwen-final-v2-replication}}

The decode figures use a first-to-last-output-event timing proxy for individual requests; they exclude full-group prefill and queue time. The second native host is a separate four-group replication of the same source recipe, not pooled with the primary host.

</details>

## Bounded reasoning and executable coding checks

These samples used medium sampled thinking with four concurrent clients. They
are synthetic exact-answer checks, not a broad model-quality benchmark. Each
sampling-seed cohort contains 200 request rows over the same 195 distinct
prompts; the code suite has 20 small executable cases.

{{qwen-final-quality}}

The final production-admission-8 cohort passed all 200 reasoning request rows
over 195 distinct prompts, and all 20 executable coding cases. The earlier
admission-4 candidate cohort (199/200 request rows, one HTTP 429; 194/195
distinct prompts on the first sample) is retained separately in the linked
records. These are bounded synthetic checks, not a broad model-quality claim.

## Context and mixed-request trade-offs

The context ladder tested roughly 8K, 65K, 131K and 254K prompts, one request
at a time with thinking off and four engine slots configured. These formal
context/mixed checks used the final image before cutover under maintenance HTTP
admission 4; the production speed cohort used admission 8. Context capacity
was 262,144 tokens; the near-limit prompt contained 253,836 tokens. The table
keeps observed response time separate from the strict bare-JSON gate.

{{qwen-final-context}}

{{qwen-final-mixed}}

The mixed test paired 16 prompt rows per recipe across two repetitions. Native
and cooperative EXL3 passed the strict retrieval gate at both long-prompt
sizes. Affine4's 131,072-token long-retrieval responses returned the correct
value but failed the strict JSON format check; at 253,824 tokens they passed
both checks. The short code and prose cells are capped responsiveness probes,
not completed-task quality scores. Sample counts are small, so these tables
report no p95.

<details>
<summary>Context cache observations and Affine4 response_format option</summary>

The patched native image's exact 253,836-token resend reused 253,835 prompt
tokens and completed in about 1.03 seconds. The cooperative EXL3 and plain
Affine4 resends had no reported prompt-cache hits. In the mixed run, repeated
near-limit short requests reported two native cache hits in active code and
four in new prose; no cache reset was performed, so the mixed cells are not
described as uniformly cold.

Affine4 also passed two single-request, opt-in API `response_format` probes at
131,072 and 253,824 tokens. They used buffered fence normalization and schema
validation; this is not constrained generation and does not replace the plain
prompt format failures above.

API compatibility also differs by complete recipe. The final native production
image passed a 48-check API/tool capability matrix (12 basic and 36 typed-tool
checks), plus separate front-cancel recovery and Pi/Hermes adapter-client checks.
The matrix is capability evidence, not a latency benchmark; 48 counts checks,
not HTTP requests. Strict `json_schema` response formatting returns HTTP 400.
Affine4's opt-in `response_format` path buffers the response, normalizes a
trailing fence, then validates the schema; it is not constrained generation.

Core receipts: [patched native](evidence/runs/qwen-native-final-core-20261001.json),
[cooperative EXL3](evidence/runs/qwen-cooperative-final-core-20261001.json), and
[Affine4](evidence/runs/qwen-affine4-final-core-20261001.json). The final production
API and named-client acceptance is recorded separately in
[the acceptance run](evidence/runs/production-final1024-http8-acceptance-20261001.json). These records
link sanitized numeric source projections; they omit private routes, raw
outputs, and per-prompt hashes.

</details>

## Qwen: patched TensorFold 0.6 + EXL3 is recommended with caveats

Use the patched TensorFold 0.6 + EXL3 guide for the measured one-Spark recipe;
TensorFold is the serving software and EXL3 is the weight format. Affine4 remains
a useful alternative when cold long-prefill response time matters, with strict
JSON-format limitations described below. The cooperative EXL3 guide remains a
rollback path for existing deployments.

<details>
<summary>Earlier cooperative EXL3 and September comparison tests</summary>

### Fresh C1/C4 test on the restored EXL3 image · October 1

Both settings used the same identified EXL3 image and four configured backend
slots. C1 used one active client; C4 used four simultaneous clients.

{{qwen-restored-throughput}}

{{qwen-restored}}

<details>
<summary>Per-request decode-rate proxies, separate from group throughput</summary>

{{qwen-restored-decode}}

</details>

The October 1 test used the deployed EXL3 image, TensorFold 0.3.6.1 and the
pinned EXL3 model revision. It is retained as historical evidence; the current
recommendation uses the separately identified patched TensorFold 0.6 image. See the [recipe guide](experiments/qwen-tensorfold-cooperative/README.md)
and [structured run receipts](evidence/runs/).

### Historical September 29 complete-recipe comparison

{{qwen-throughput}}

TensorFold versions and model packs differ in this historical test. C4 rates
aggregate output over a whole four-request group, including prefill; they are
not per-client rates. Engine slots describe backend capacity, while client
concurrency describes requests offered by the benchmark. These results do not
isolate runtime-engine effects from quantization effects.

{{gains}}

{{qwen-waiting}}

The historical September 29 figures do not establish the exact image identity of
every run or effective reasoning effort across the runtime versions. Their
reasoning scores and latencies are not a matched-medium quality comparison. The
long-prompt figure measures a **completed, validated JSON answer**, not time
to first token. Latency is elapsed waiting time; lower is faster. The median is
the middle result; p95 describes the slow end. Single context/mixed runs and
small tail samples are observations, not guarantees.

This is a comparison of complete configurations: TensorFold versions and
scheduling differ, as do the EXL3 3.05 bpw and Affine4 4-bit model packs. It does
not isolate a runtime-engine effect from a quantization effect.

</details>

### Affine4 remains an alternative for long prompts and mixed traffic

Both recipes passed 20/20 executable coding tasks in a small synthetic suite;
this does not establish a coding-quality difference. Affine4 retains stronger
measured long-prompt completion and mixed-traffic short-request latency. Its
fresh public-source build qualification covers bounded API, tool and token-replay
checks, not its historical performance test.

{{qwen-tails}}

The small reasoning and coding samples do not establish broad model quality. See [EXL3 methodology](experiments/qwen-tensorfold-cooperative/METRICS.md)
and [Affine4 performance and methodology](recipes/qwen38-flash-affine4-1spark/PERFORMANCE.md).

<details>
<summary>Qwen numbers, correctness scores and median / p95 timings</summary>

{{qwen}}

Reasoning and coding correctness/latency use sampled thinking; concurrency for these historical quality rows is unknown. C4 throughput and context checks use thinking off. Repeated reasoning fixtures are excluded from the totals. Correctness is separate from speed.

</details>

## GLM: adaptive drafting on two Sparks

DFlash2 drafts tokens ahead, then verifies them against the target model.
The adaptive recipe varies the draft length while retaining BF16 dense layers
and the same configured serving capacity as our fixed-draft control.

{{glm-gains}}

{{glm-throughput}}

GLM and Qwen use different hardware, workloads and measurement histories.
Read each panel within its own model. GLM's figures are original-site
observations from September 11; they do not qualify a fresh portable deployment.

<details>
<summary>GLM numeric comparison and evidence scope</summary>

{{glm}}

[Recorded checks](recipes/glm53-flash-adaptive-2spark/results.json) include eight
long-context checks and five structured-tool/mixed checks. Configured context is
850K; the largest recorded fixture was about 801K tokens.

</details>

## What you can rely on—and what still needs testing

**Affine4:** the shared sources were independently rebuilt on a Spark. That build
passed 11 API checks, 36 tool/API checks, six exact sampled token replays,
10 adapter tests and a real file-reader check. The performance and quality
results above belong to the qualified production image; they were not rerun in
full on the rebuild. Verified weights were reused. A fresh-download clean-machine
test, long soak and reboot qualification remain open.
[Build evidence](recipes/qwen38-flash-affine4-1spark/BUILD-VALIDATION.md).

**GLM:** the portable source package and site rendering pass offline checks.
The measured original runtime's results do not establish rebuilt-image or
new-site behavior. Recoverable startup allocation warnings remain unresolved.
[Release qualification gates](MASTER_PLAN.md).

## More choices and comparison sources

{{alternatives}}
- [All recipes](recipes/): four-Spark GLM, mixed fleets, full GLM and historical experiments, each with its own status.

<details>
<summary>Public reference figures: useful context, unmatched measurements</summary>

The following are **historical figures reported by MiaAI-Lab**, preserved at
pinned source revisions. Public decode rates differ from our full-group Qwen
rates, and public prefill latency differs from our completed-response latency.
The public C1 reference is one active request on a four-stream backend; the
source does not report its repetition count or exact decode-rate denominator.
The local selected-image C1 decode proxy is 51.99 tokens/s; the local full-request
C1 group rate is a separate measurement. The public repetition count and exact
decode denominator are unstated, so the timing boundary remains approximate and
we do not infer a cross-source winner.

{{public}}

Public Qwen uses TensorFold 0.3.6.3; the local EXL3 test uses TensorFold 0.3.6.1. GLM's
pinned prose figure is from September 8 with FP8 dense layers and a 14 GiB KV
budget; our local recipe retains BF16 dense layers with 11 GiB KV. Its public
structured/code figures are a separate August 28 workload.
The [source audit and newer upstream reports](comparisons/SOURCES.md) explain
what changed. None of these figures supports a cross-source winner.

</details>

The tables and charts share [one evidence source](comparisons/README.md); checks
reject stale generated output. Our Qwen adapters were independently written;
the public implementation was not imported.

The broker is MIT licensed. Derived runtimes retain upstream licenses; model
access and licenses remain with their publishers. See [third-party notices](THIRD_PARTY.md),
[source pins](sources.json) and each recipe's notices.
