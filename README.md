# spark-serve

Pinned serving recipes and one model lifecycle broker for NVIDIA DGX Sparks.
This distribution contains source, patches, build procedures and model-source
revisions. It contains **no model weights**.

This is a deployment candidate. Portable launchers and a fresh-site installer
are included; clean GPU compilation and deployment acceptance are still pending.
Do not interpret historical benchmark results as a completed test of this package.

| Recipe | Configuration | Evidence |
|---|---|---|
| `glm53-flash-adaptive-2spark` | GLM Flash, TP2, E3 grouped EXL3, adaptive DFlash2 k2/4/7, 850K context, images, 8 sequences, 11 GiB FP8 KV | Current best verified GLM Flash settings |
| `glm53-flash-adaptive-4spark` | Two independent copies of that GLM configuration; native ready-only routing | Current four-Spark GLM arrangement |
| `agent-fleet-211-adaptive` | Adaptive GLM on two Sparks plus two independent Qwen TP1 workers | Current per-model settings; historical mixed-fleet rate is a different run |
| `qwen38-flash-1spark` | Qwen Flash Next on one Spark, 262K context, PLE offload, FP8 KV, BF16 SSM, MTP3 | Same verified settings as the fleet workers |
| `glm53-full-4spark` | Full GLM, TP4/DCP4, 307200 context, DFlash2 k7, 12 sequences, 6 billion KV bytes per rank | Bounded original-site qualification |

GLM Flash keeps BF16 dense paths, a 2048-token batch budget, mixed-prefill 512,
row cap 128 and the pinned donor transplant. Adaptive verification retains k7
for structured and unobserved requests and captures all eight-slot graph shapes.
In the matched prose checks, single-request throughput increased from 22.22 to
24.98 output tokens/s and eight-request throughput from 72.25 to 88.57. These
are workload-specific measurements, not universal model speed claims.

Qwen TP1 retains 262144 context, PLE offload, 20 GiB target KV, 26 GiB host
reserve, 5 GiB slack, FP8 KV, BF16 SSM, MTP3 and the V2 runner. Full GLM keeps
NVMe KV tiering off and balanced mixed-prefill behavior. The fixed-k GLM recipes
remain as historical references. Qwen TP2 and DeepSeek remain experimental
because their recorded startup/fabric failures have not been resolved.

Start with [deployment and validation](REPRODUCTION.md). The `deploy/` programs
stage the same runtime source and settings on the original operator's fleet or
a new site. The site file supplies machine identity, storage and network bindings.
`spark-serve` remains the sole owner of model admission, start, stop and recovery.

The broker is MIT licensed. Derived runtime components retain their upstream
licenses; see [third-party notices](THIRD_PARTY.md) and exact [source pins](sources.json).
Model access and licenses remain with each model publisher. Download from those
sources under your own account where required.
