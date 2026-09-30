# spark-serve

Run Qwen on **one NVIDIA DGX Spark**, or GLM-5.3-Flash on **two**.
Start with these two recipes: each includes pinned sources, settings, build
instructions and the evidence behind our choices. Model weights are downloaded
separately from their publishers.

{{choices}}

Qwen Affine4 is our recommended **Qwen** recipe. It handles long prompts and
mixed interactive traffic substantially better than our previous EXL3 recipe,
although EXL3 still wins on some coding and reasoning workloads. GLM remains
our two-Spark choice; these measurements do not establish a quality ranking
between the two models.

## Browse the recipe options

After cloning, run these commands without installing anything or connecting to a Spark:

```sh
PYTHONPATH=src python3 -m spark_serve recipes options
PYTHONPATH=src python3 -m spark_serve recipes options --model qwen
PYTHONPATH=src python3 -m spark_serve --json recipes options --model glm
```

The CLI lists the recommendation and retained alternatives with their guides.
It does not start a model or change an existing installation. With the updated
CLI installed, use `spark-serve recipes options` instead.
From another directory, add `recipes --source /path/to/spark-serve options`.

**Reading the numbers:** tokens/s measures output speed; higher is better.
Latency measures waiting time; lower is better. The median is the middle request;
p95 describes the slow end, with 95% of requests at or below it. C1, C4 and C8
mean one, four and eight concurrent requests. Small samples make tail estimates
less stable. Correctness is separate from speed.

## Get started

```sh
git clone https://github.com/juanandresgs/spark-serve.git
cd spark-serve
```

For **Qwen**, follow the [one-Spark quick start](recipes/qwen38-flash-affine4-1spark/README.md):
download the pinned weights, build the Docker image, then run it on your Spark.
Allow roughly 180 GB of disk and 100 GiB of available memory. This recipe is a
standalone Docker deployment; the broker's `recipes prepare` command does not
install it.

For **GLM**, follow the [two-Spark guide](recipes/glm53-flash-adaptive-2spark/README.md)
and [deployment walkthrough](REPRODUCTION.md). Supply your host, storage and
network bindings, build the pinned runtime, and validate it on your pair of
Sparks. The included broker owns admission, start, stop and recovery for managed
recipes.

## Why these settings?

### Qwen: our matched one-Spark comparison

September 29, 2026. Same workloads, four request slots and 262K context. Rates
below count output tokens over the entire request group's elapsed time; C4
means four concurrent requests. Speed rows are medians of three runs.

{{qwen}}

Affine4 cuts cold long-prompt completion time by about **51%** and mixed-load
short-request p95 by **59%**. The tradeoff is lower code throughput and longer
reasoning tails. The small synthetic correctness checks support this choice but
do not prove broad coding superiority or that more reasoning always helps.
We keep normal reasoning behavior by default; an optional 2,048-token reasoning
cap reduced coding p95 to 96.65 s in a bounded follow-up, but is not enabled globally.

These are comparisons of complete recipes, including different TensorFold
versions, rather than an isolated test of quantization. See [methodology, older
variants and all caveats](recipes/qwen38-flash-affine4-1spark/PERFORMANCE.md).

### GLM: our matched two-Spark comparison

September 11, 2026. Adaptive DFlash2 chooses how many draft tokens to verify,
while retaining BF16 dense layers and the same serving capacity.

{{glm}}

The main gain is prose: about **12%** for one request and **23%** across eight.
[Recorded qualification](recipes/glm53-flash-adaptive-2spark/results.json)
includes eight long-context checks and five structured-tool/mixed checks.

### How do the public numbers compare?

These are reference points reported by MiaAI-Lab, **not matched A/B tests**.
We preserve the metric definitions and configuration differences so the numbers
are useful without implying an unsupported win.

{{public}}

Public Qwen uses a newer runtime; public GLM's prose row also uses a different KV
budget. A fresh run with identical inputs and timing boundaries is needed for a
ranking. Our Qwen adapters were independently written; the public implementation
was not imported.

## What has been verified?

**Qwen:** an independent build from the shared sources compiled on a Spark and
passed 11 API checks, 36 tool/API checks and six exact-token replays against the
qualified production image. The full performance table was measured on that
production image, not rerun in full on the rebuilt image. The build reused
verified weights; a fresh-download clean-machine install, long soak and reboot
qualification remain outstanding. [Build receipt](recipes/qwen38-flash-affine4-1spark/build-validation.json).

**GLM:** the measurements above describe the qualified original runtime. The
portable source package and site rendering pass offline checks; rebuilt images
and a new pair of Sparks still require destination acceptance. Recoverable
startup allocation warnings remain unresolved. [Release gates](MASTER_PLAN.md).

## Previous recipes remain available

{{alternatives}}
- [Other recipes](recipes/): four-Spark GLM, mixed fleets, full GLM and historical experiments retain their own evidence and status.

This recommendation does not migrate existing installations. The broker is MIT
licensed; derived runtimes retain upstream licenses. See [third-party
notices](THIRD_PARTY.md), [source pins](sources.json) and the individual recipe
notices. Model access and licenses remain with their publishers.

## Updating the comparisons

The tables are generated from [structured recipe choices](comparisons/models.json)
and the linked benchmark receipts. See [the update workflow](comparisons/README.md).
GitHub checks reject stale tables or broken evidence references. Share this page
for the overview, or a recipe's direct link for build and test instructions.
