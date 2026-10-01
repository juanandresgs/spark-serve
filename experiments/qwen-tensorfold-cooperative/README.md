> **Recommendation:** use cooperative EXL3 for the measured coding workload; choose [Affine4](../../recipes/qwen38-flash-affine4-1spark/README.md) when fresh long-prompt completion and mixed-traffic latency matter more. The EXL3 source kit is a standalone recipe and is not independently GPU rebuilt as the historically measured image. See the dated [comparison](../../comparisons/README.md) for scope and provenance.

# Qwen3.8-Flash-Next on one DGX Spark: cooperative TensorFold recipe

A source-pinned single-Spark recipe with schema-aware tool arguments, honest
Prometheus-compatible counters and cooperative prefill scheduling. The scheduler
lets decoding and short requests make progress while long prompts are processed.
It uses the pinned TensorFold math; these adapters contain no copied model kernels.

**Measured result:** with two 131K-token prompts and twenty short arrivals on the
same Spark, short-request p95 first-token latency fell **356.36 → 2.73 seconds**
while total workload time increased **1.22%**. In a separate packaged-image test,
the longest pause in an active stream fell **362.74 → 2.87 seconds**.
A fresh 253,841-token prompt still took **536.45 seconds** to its first token.
This is a responsiveness result, not a 130× decoding speedup or a SOTA claim.
See [metrics and causal limits](METRICS.md), [machine-readable receipts](measurements.json),
[claim audit](CLAIMS.md), and [next experiments](IMPROVEMENTS.md).

Historical EXL3 runs include different configurations and build identities.
This retained public source kit has CPU checks but no independent GPU build and
installation qualification against the historically measured image. Do not attribute every
historical result to this exact exported source. No weights or images are
distributed; a destination still needs its own acceptance checks.

## Fixed recipe

| Component | Setting |
|---|---|
| Hardware | One NVIDIA DGX Spark, GB10, nominal 128 GB unified RAM |
| TensorFold | 0.3.6.1, commit `34bae79ac97da6c3ab3fe10159cf49633ce8112a` |
| Checkpoint | `turboderp/Qwen3.8-Flash-Next-exl3`, `3.05bpw_h5_ng5`, revision `69e33439ae950f17bcbe95c98f117d80f759ab6d` |
| Context / output cap | 262,144 total / 8,192 maximum output tokens |
| Engine slots / drafting | Four / MTP6 |
| Prompt cache | 2 GiB; no disk snapshot directory |
| Scheduling | 2,048-row turns, four decode rounds per turn |
| Intermediate vocabulary head | Skipped; separate benefit unproven |
| API | Text; thinking off by default, offered tool schemas supported |

The 8,192-token cap is a **reservation**, not proof of an 8,192-token response at
maximum context. Model-pack differences prevent quality-equivalence claims against
NVFP4/vLLM. The original site retained about 31 GiB available memory in the final
trial; this is an observation, not a per-slot capacity guarantee.

API compatibility also has limits: strict response-format schema normalization
and the `spark_reliability` reasoning-budget opt-in are Affine4-specific and are
not available on this EXL3 route. Clients should use the common text and tool
schema behavior described above rather than assume every Affine4 API extension
is present.

## Build and stage on an idle Spark

Requires Linux ARM64, working NVIDIA Container Toolkit/Docker GPU access,
Python 3 and sufficient storage for the checkpoint, base image and build cache.
Use your existing lifecycle manager when integrating; this foreground script is
an isolated evaluation runner and does not install a second daemon or router.
The API binds loopback only and supplies no authentication layer.

```bash
cd experiments/qwen-tensorfold-cooperative
python3 -m venv .venv
. .venv/bin/activate
pip install huggingface_hub
export QWEN_DATA=/srv/qwen-tensorfold
python stage.py --directory "$QWEN_DATA"
docker build --build-arg BASE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c -t spark-qwen-cooperative:0.3.6.1 .
# Actual parser and telemetry boundaries, inside the built runtime:
docker run --rm --entrypoint bash -v "$PWD:/checks:ro" spark-qwen-cooperative:0.3.6.1 -lc 'cd /checks && python -m unittest test_adapter test_telemetry'
python -m unittest test_cooperative
bash run.sh
```

`stage.py` downloads the exact revision and verifies every file against published
blob/LFS digests. `overlay.py` derives root EOS IDs from the checkpoint generation
metadata and verifies tokenizer IDs. It leaves the original weights and config
unchanged; the derived config is a read-only container overlay. Nested text-config
EOS is preserved because it has a separate PLE/ngram role.

The Dockerfile fetches the exact TensorFold commit and builds our adapter layer
from source. The NGC base is digest-pinned and `jsonschema` is version-pinned;
transitive pip resolution is **not fully locked**, so bit-identical images are not
promised. The dependency freeze is retained inside `/opt/tensorfold-dependencies.txt`.
Record `docker image inspect spark-qwen-cooperative:0.3.6.1` and the model verification
receipt with every new result. First startup may compile kernels. Use a separate
terminal for health and inference after the server reports ready:

```bash
curl --fail http://127.0.0.1:8898/health
curl --fail http://127.0.0.1:8898/v1/models
curl --fail http://127.0.0.1:8898/metrics
curl --fail http://127.0.0.1:8898/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Return exactly READY"}],"max_tokens":32,"temperature":0}'
```

Stop with `docker stop spark-qwen-cooperative`. For an A/B control, stop the
candidate and restart the **same image** with `SPARK_COOPERATIVE=0 SPARK_FINAL_HEAD=0
bash run.sh`. Keep all other settings and warmed kernels fixed. This disables the
scheduler/head adapters, retaining the same tool adapter, EOS and telemetry.
The comparison control is an adapter-off control, not untouched upstream software.

## Reproduce the workloads

The harnesses are portable adaptations of the executed synthetic benchmark code;
new paths/CLI flags do not make these exact public files GPU-qualified. Generate
fixtures with the pinned runtime's own template/tokenizer inside the built container;
this loads tokenizer metadata, not model weights, and needs no GPU.
Keep one fixture file for both arms. A new prefix/`--rep` is required between
prompt-cold trials; a runtime restart also clears retained prompt state.

```bash
mkdir -p results
docker run --rm --entrypoint python -v "$PWD:/checks" -v "$QWEN_DATA/target:/target:ro" spark-qwen-cooperative:0.3.6.1 /checks/context.py --model-dir /target --model qwen3.8-flash-next --fixtures /checks/results/fixtures.json --prepare
python measure.py --base http://127.0.0.1:8898/v1 --label candidate --suite mixed --rep 1 --fixtures results/fixtures.json --out-directory results
python measure.py --base http://127.0.0.1:8898/v1 --label candidate --suite context --rep 2 --fixtures results/fixtures.json --out-directory results
python measure.py --base http://127.0.0.1:8898/v1 --label candidate --suite short --rep 3 --fixtures results/fixtures.json --out-directory results
python stream_gap.py --base http://127.0.0.1:8898/v1 --label candidate --fixtures results/fixtures.json --out-directory results
```

Expect the near-limit cold test to take about nine minutes on the observed stack.
Run alternating control/candidate trials on the same Spark with idle background
work, explicit cache conditions and at least three repeats before estimating a
stable improvement. Compare correctness, fingerprints, first-token time, maximum
stream gap, total completion and memory/faults. Outputs are saved locally and
never executed. The harness's code-presence checks do not establish code quality.

## Integration and limitations

Route the logical model to the single backend and verify actual completion-counter
changes through your real clients. Loading another replica does not route work to
it. Prefix continuity requires affinity to the process retaining its state. At this
pin a retained strict prompt extension can hit cache; an identical replay is not a
cache hit, and serial `draft:false` deliberately avoids reuse.

`/v1/models` reports the fixed 262K/four-slot backend. Advertise tools/text/reasoning
through the existing router only after your client checks. Do not advertise vision.
Metrics use vLLM-compatible names for dashboard compatibility; they are TensorFold
counters. Prompt totals include cached input on successful requests, output totals
count callback tokens, and no KV-utilization or latency-percentile metrics are
invented. Keep the four-slot/context launch values fixed: telemetry hardcodes them.

The scheduler patches private APIs at the pinned revision; upgrades require a new
audit. Four occupied slots still queue arrivals. Prefill handoffs synchronize CUDA;
a hung GPU turn can block the dispatcher. Full-queue cancellation, GPU-error
recovery, cold boots and long soak remain unqualified. Socket half-close is treated
as disconnect. The head-skip path has matching bounded token tests, not universal
numerical equivalence. Tool conversion covers declared top-level properties and
leaves schema-invalid values unchanged for client rejection; it never executes tools.

See [licenses and provenance](THIRD_PARTY.md). Use a maintenance lane and retain
your previous service/image for rollback. This recipe does not alter production
routing, install boot services or claim a tested restart policy.

## Later TensorFold work

TensorFold [0.6.0](https://github.com/ashhart/TensorFold/releases/tag/v0.6.0),
released September 30, adds CUDA prompt fill inside decode rounds, prompt-state
resumption, consolidated EXL3 tables and streamed typed tool arguments. These
are upstream capabilities to evaluate, not results for this pinned 0.3.6.1
recipe. The release's stated throughput improvements are for NVFP4, not this
EXL3 checkpoint. An isolated upgrade trial should preserve this runtime as its
control and separately check C1/C4 code and prose, reasoning quality, tools,
cached resends and mixed traffic before any performance claim.
