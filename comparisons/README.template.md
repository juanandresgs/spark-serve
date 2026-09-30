# spark-serve

**Put your DGX Spark to work: Qwen on one, GLM on two.**

Build a local, OpenAI-compatible model server with pinned sources and settings,
practical run guides, and measurements you can inspect. Start with the recipes
below, then choose the tradeoffs that suit your workload. Download model weights
separately from their publishers.

{{choices}}

These are our recommended starting points for each model. Context and request
counts are configured limits, not a promise of simultaneous maximum-length
capacity. The results below help choose a **recipe**; they do not rank the two
models' quality.

## Choose your setup

```sh
git clone https://github.com/juanandresgs/spark-serve.git
cd spark-serve
```

- **One Spark → [Qwen Affine4 quick start](recipes/qwen38-flash-affine4-1spark/README.md).**
  Download the pinned weights, build the Docker image, and run it on your Spark.
  Allow at least 180 GB of disk and 100 GiB of available host memory. Affine4 is
  a **standalone Docker recipe**; the broker's `recipes prepare` cannot install it.
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

## Qwen: choose Affine4 for long prompts and interactive traffic

Our one-Spark comparison gives Affine4 the edge on long prompts, mixed traffic,
and prose. **Cooperative EXL3 is faster on the measured code workload and has
shorter reasoning tails.** Both remain available.

{{qwen-throughput}}

These are complete recipes with different TensorFold versions, four request
slots and 262K configured context. Output speed includes prefill and the whole
request group's elapsed time. Higher is faster; C4 means four concurrent
requests. Each workload has its own axis.

{{gains}}

{{qwen-waiting}}

The long-prompt figure measures a **completed, validated JSON answer**, not time
to first token. Latency is elapsed waiting time; lower is faster. The median is
the middle result; p95 describes the slow end. Single context/mixed runs and
small tail samples are observations, not guarantees.

### The tradeoff: thinking can take longer

Both recipes passed 20/20 executable coding tasks in our small synthetic suite.
Affine4 took longer on the slowest part of that distribution, without a measured
coding-quality gain. This matters if you need predictable interactive latency.

{{qwen-tails}}

The recommendation keeps reasoning uncapped. An optional 2,048-token reasoning
cap improved coding p95 in a bounded follow-up but can truncate useful reasoning;
test that policy on your own workload. These fixtures do not establish broad
coding superiority. See [performance and methodology](recipes/qwen38-flash-affine4-1spark/PERFORMANCE.md).

<details>
<summary>Qwen numbers, correctness scores and median / p95 timings</summary>

{{qwen}}

The reasoning totals exclude repeated fixtures. Quality uses sampled thinking;
speed and context checks use thinking off. Correctness is separate from speed.

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

**Qwen:** the shared sources were independently rebuilt on a Spark. That build
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

{{public}}

Public Qwen uses TensorFold 0.3.6.3; our selected recipe uses 0.3.6.2. GLM's
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
