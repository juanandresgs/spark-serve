# Qwen3.8 Flash Next Affine4 on one DGX Spark

A reproducible text-serving recipe for GB10/128 GB: four slots, 262144 context,
MLX affine4 group32 weights, int8 KV, MTP6, SSD-backed n-gram tables and cooperative
prefill. Independently authored adapters over pinned TensorFold 0.3.6.2.
See [PERFORMANCE.md](PERFORMANCE.md) for local versions and attributed public claims.

## Build and run

Requirements: a DGX Spark with NVIDIA-enabled Docker, a C++ compiler in the image,
Python 3 and enough disk for the approximately 106 GiB checkpoint plus images/cache
(allow at least 180 GB). Leave the GPU free of other workloads; our lifecycle
requires 100 GiB available host memory before loading. Four configured slots do
not establish four simultaneous maximum-length prompt capacity. This kit is text-only.

From the cloned repository root on your Spark:

```sh
cd recipes/qwen38-flash-affine4-1spark
python3 -m venv .venv
. .venv/bin/activate
python -m pip install huggingface_hub
python stage.py --directory "$PWD/data"
docker build -t spark-qwen-affine4:20260930 .
QWEN_DATA="$PWD/data" bash run.sh
```

Run on the Spark. `stage.py` verifies all published file digests at the pinned
revision. `run.sh` checks them again, mounts weights read-only and keeps mutable
kernel/cache files separately. It binds port 8898 to loopback; use SSH forwarding
for remote tests. Do not expose it publicly without an authenticated gateway.
Initial kernel compilation/loading can take several minutes. Stop with Ctrl-C or
`docker stop spark-qwen-affine4`; no second model daemon is installed.

```sh
curl -fsS http://127.0.0.1:8898/health
curl -fsS http://127.0.0.1:8898/v1/chat/completions \
 -H 'Content-Type: application/json' -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Return exactly {\"answer\":7}"}],"max_tokens":128,"response_format":{"type":"json_object"}}'
```

## Selected choices

- Four slots and 262144 total context; 8192 maximum output allowance. Text only.
- MTP six drafts, confidence 0.6; int8 KV; 2 GiB prompt cache; no disk snapshots.
- Cooperative 2048-row prefill and four decode rounds; bounded positional SSD
  reads/read-ahead and tiled selector enabled. Intermediate-head skip is off.
- Non-thinking is the server default, matching our interactive contract. Explicit
  thinking requests use temperature 1, top-p .95, top-k 20; see `presets.json`.
  Those are existing runtime defaults when omitted; greedy callers override them.
- No forced reasoning cap by default. Optional `spark_reliability` cap2048 and
  answer_reserve1024 uses the remaining total allowance for a final answer.
  It can truncate useful reasoning and buffers responses. See measured tradeoffs.
- `response_format` opts into strict JSON/schema validation and whole-fence
  unwrapping. It buffers, never repairs values, and rejects invalid/truncated output.
  This is not grammar-constrained decoding or a guarantee of successful generation.
  Include desired fields in the prompt. No external schema references are fetched.
- New JSON/budget policies combined with tools are rejected as unqualified;
  ordinary typed tool calls retain their tested path.

## Repeat the tests

```sh
mkdir -p results
python bench.py --base http://127.0.0.1:8898/v1 --label affine --out results --quick --clients 4 --reps 3
python evaluate.py --base http://127.0.0.1:8898/v1 --out results/reasoning.json --temperature 1 --schema --fixture-seed 20260929
python api_checks.py --base http://127.0.0.1:8898/v1 --out results/api.json
```

Repeat reasoning with fixture seed20260930 and with `--budget 2048` for the optional
policy. `--draft off` enables serial controls. These programs store local synthetic
responses; review before sharing. Quality fixtures check the result, not merely
schema validity. Four duplicate graph questions across the original/fresh sets
must be excluded from pooled unique scores; see quality.json.

The exact source file hashes are in MANIFEST.json. The eight adapter source files
match bytes extracted from the measured immutable image; qualified-image.json
records that image. Public build qualification is recorded separately in
BUILD-VALIDATION.md. No private deployment files, hostnames, runtime captures,
credentials, weights or original repository history are in the export.
