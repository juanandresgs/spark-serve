# TensorFold 0.6.0 native source build (experimental)

TensorFold is the serving software; EXL3 is the model's compressed weight format. This is a public source-build scaffold for TensorFold 0.6.0 and the pinned Qwen EXL3 model. It remains experimental while final-image speed, quality, context and mixed-traffic checks complete. The retained cancellation-plus-burst4 parent is a separate baseline; see the dated qualification record for which checks apply to each image.

The base image, runtime commit, and model revision are pinned in `pins.json`. The base image's 216-package inventory and the tested primary runtime's 217-package inventory differ only by TensorFold 0.6.0. `runtime-constraints.txt` pins the base inventory; it is not an install requirements file. The primary Dockerfile installs the pinned TensorFold source under those constraints, checks that active dependencies satisfy their declared versions, then compares the complete result with `expected-primary-inventory.json`. A build fails for missing packages, extra packages, or version drift. Patched derivative Dockerfiles reinstall the local source with `--no-deps`; their complete input hashes and package inventory checks are recorded separately. Capture method and package deltas are recorded in [`inventory-provenance.json`](inventory-provenance.json).

## Build and inspect

On a Linux ARM64 host with Docker and access to the pinned NVIDIA base image:

```sh
cd experiments/qwen-tensorfold-native-0.6
python3 build.py --tag local/qwen-tensorfold-native:0.6.0 \
  --receipt ./artifacts/build-receipt.json
```

The build pulls the exact base digest, checks out the exact TensorFold commit, validates the runtime version and complete package inventory, then records the local image ID, repository digests when available, rootfs diff IDs, source hashes, and pinned model/runtime identity. Image IDs are daemon-store-specific: compare repository and rootfs provenance across Docker stores rather than requiring `.Id` equality. The local `.Id` is not treated as a config digest.

This command builds the primary runtime only. It does not download model weights, create a serving configuration, or deploy anything. Use a separately reviewed model staging and launch procedure. Do not treat a successful build or matching package inventory as GPU or end-to-end qualification.

## Stage the pinned model snapshot

The model weights are downloaded separately from Hugging Face. `stage_model.py`
pins the exact repository revision and checks every file against its published
LFS SHA-256 or Git blob ID. It does not write an absolute path or credential to
a receipt. Install `huggingface_hub` in a host Python environment, then run:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'huggingface_hub==1.24.0'
QWEN_DATA=/srv/qwen-tensorfold-native
mkdir -p "$QWEN_DATA/target"
python stage_model.py --directory "$QWEN_DATA/target"
python stage_model.py --directory "$QWEN_DATA/target" --verify-only
```

The model repository and commit are pinned in `pins.json`; the helper refuses a
different commit. For a gated repository, authenticate with Hugging Face's
standard token environment or credential store. Keep tokens outside shell
history and source files.

## Run a local experimental endpoint

After building the cancellation, burst-4 and 1024-row prefill derivatives below,
this command starts the final image in the foreground. That image sets
`TF_FLASH_DECODE_BURST=4` and `TF_FLASH_PREFILL_ROWS=1024`; its model files are
the pinned EXL3 weight pack. Image labels and settings are part of the recipe
identity.

```sh
QWEN_DATA=/srv/qwen-tensorfold-native
QWEN_CACHE="$QWEN_DATA/cache"
IMAGE=local/qwen-tensorfold-native:0.6.0-cancel-burst4-prefill1024
mkdir -p "$QWEN_CACHE"
docker run --rm --name qwen-tensorfold-native \
  --gpus all --network host --ipc host --cap-add IPC_LOCK --ulimit memlock=-1 \
  -v "$QWEN_DATA/target:/model:ro" -v "$QWEN_CACHE:/cache" \
  -e OMP_NUM_THREADS=4 -e TORCH_CUDA_ARCH_LIST=12.1 -e MAX_JOBS=2 \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/cache/hf \
  -e TRITON_CACHE_DIR=/cache/triton -e TORCH_EXTENSIONS_DIR=/cache/torch \
  -e PYTHONUNBUFFERED=1 \
  "$IMAGE" serve /model --backend cuda --name qwen3.8-flash-next \
  --host 127.0.0.1 --port 8080 --context 262144 --parallel 4 \
  --max-tokens 8192 --mtp-drafts 6 --mtp-confidence 0.7 \
  --kv-dtype bf16 --prompt-cache-gib 2 --snapshot-dir none \
  --no-thinking --no-update-check
```

The endpoint binds loopback only and has no authentication layer. In another
terminal, wait for startup and check health, model discovery, and a simple
completion:

```sh
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/models
curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Return exactly READY"}],"max_tokens":32,"temperature":0}'
```

Stop the foreground container with Ctrl-C. This manual run command is for
evaluation and acceptance; it does not install another service manager or
replace an existing lifecycle authority. A passing health check or sample
request is not a substitute for the bounded workload and recovery checks below.

## Reproduce the fixed-cap C1/C4 speed screen

The standalone `benchmark_speed.py` replays the portable `cell-warmup-v2`
protocol used for the October 2026 comparison. It sends one exact-answer
128-token canary, then four excluded warmup cells (C1 code, C1 prose, C4 code,
C4 prose; four requests each), then 64 measured requests: four repetitions of
those same four cells, with the cell order reversed on alternating repetitions.
C1 uses one active client for four serial requests; C4 uses four simultaneous
clients. All code and prose requests use the same two fixed prompts with unique
numbered fixture identifiers, a 512-token cap, temperature 0, thinking off,
medium effort fields, and the recorded deterministic seed formula. Warmups are
kept separately and are never included in the measured rates.

Create a local identity file containing only the pinned recipe identity and
public setting values. For example:

```json
{
  "schema_version": 1,
  "image_digest": "sha256:<64 lowercase hex digits>",
  "runtime_repository": "https://github.com/ashhart/TensorFold",
  "runtime_revision": "<40 lowercase hex digits>",
  "runtime_version": "0.6.0",
  "model_repository": "turboderp/Qwen3.8-Flash-Next-exl3",
  "model_revision": "<40 lowercase hex digits>",
  "model_branch": "3.05bpw_h5_ng5",
  "configuration_sha256": "<64 lowercase hex digits>",
  "settings": {
    "context_tokens": 262144,
    "parallel_slots": 4,
    "max_output_tokens": 8192,
    "kv_cache": "bfloat16",
    "mtp_tokens": 6,
    "reasoning_effort": "medium",
    "thinking": "off"
  }
}
```

Then run the same command for each arm, changing only the endpoint, served
model alias, label, and identity file to match the arm actually measured:

```sh
python3 benchmark_speed.py \
  --base "$SPARK_SERVE_BASE" \
  --model qwen3.8-flash-next \
  --label native-cancel \
  --identity ./identity.json \
  --out ./results/native-cancel.json \
  --timing-condition quiet
python3 curate_speed.py ./results/native-cancel.json \
  --out ./results/native-cancel-summary.json
```

The base URL must be user supplied and end in `/v1`. An optional bearer token
comes from `SPARK_SERVE_API_KEY`; neither its value nor the endpoint or identity
file path is written to the receipt. The identity schema rejects endpoint,
host, node, credential, and arbitrary setting fields. Receipts retain only the
allowlisted recipe identity, request and response hashes, token counts, finish
reasons, event timings, cache counters when exposed, and numeric group
summaries. Model response and reasoning text are discarded before any receipt
is written. Keep each arm's receipt separate and preserve failed or incomplete
results.

Rates are total completion tokens divided by the full four-request group wall
time, including prefill and queueing. A cell receives a numeric rate only when
all four repetitions pass valid token-usage, visible-output, finish-reason, and
no-reasoning gates. Failed or incomplete cells remain `null`; they are not
treated as zero or silently dropped. A `length` finish is accepted for this
throughput-only 512-token screen only when all 512 completion tokens are
reported. This does not measure task correctness or answer quality. No prompt
cache reset is performed; the receipt records any cache counters the endpoint
provides. Mark a run `quiet` only after the operators confirm no competing
benchmark or staging traffic.

The schedule is pinned in `speed_protocol.py`; the two prompt fixtures and
OpenAI-compatible streaming request logic are in `transport.py`. The test suite
checks the canonical fixture hash, request body, bearer-token handling, warmup
exclusion, privacy boundary, and missing-result curation without calling a
model endpoint.

## Bounded synthetic quality sample definitions

`quality_fixtures.py`, `coding_cases.py`, and `quality_contract.py` preserve the
exact generated arithmetic, Python-semantics, FIFO-state, graph-distance, and
20 small executable coding cases used by the bounded quality screen. The
matching seeds, request settings, prompt wrappers, completion rule, and grading
contract are summarized in `quality-protocol.json`. The two reasoning sets
contain 200 rows each but 195 distinct prompts; they are separate seed repeats,
not 390 unique questions. The coding cases are 20 synthetic tasks, not a broad
coding benchmark.

These files make prompts and grading inspectable, but this source kit does not
include a portable quality-evaluation runner or execution sandbox. They do not
reproduce a model result by themselves. The public result table identifies its
seed set and counts; do not infer general model quality from these bounded
fixtures.

## Separate prefill-cancellation variant

The optional `Dockerfile.cancellation` builds an experimental derivative from
the exact local image ID emitted by `build.py`. It preserves the stock source
build and applies a small patch to TensorFold's CUDA scheduler and Qwen Flash
Next prefill loop. The patch polls the existing cancellation callback between
prefill passes so a disconnected request can release its slot before first
token generation. It does not interrupt an already-running GPU kernel.

Build and test this derivative on the same Docker host as the base image; local
image IDs cannot be transferred between hosts:

```sh
PARENT_IMAGE_ID="$(python3 -c 'import json; print(json.load(open("./artifacts/build-receipt.json"))["local_image_id"])')"
python3 build_cancellation.py --parent-image-id "$PARENT_IMAGE_ID" \
  --tag local/qwen-tensorfold-native:0.6.0-cancel \
  --receipt ./artifacts/cancellation-build-receipt.json
```

The derivative verifies the original and patched source-file hashes, applies
the patch without network access, reinstalls TensorFold without dependency
resolution, then runs six CPU-only tests against the real `App.run` callback,
scheduler, and decoder round with GPU work mocked. They cover healthy streaming
and nonstreaming callbacks, empty SSE behavior, disconnect cancellation, slot
cleanup, and decoding-peer progress after a cancelled prefill. Passing them
does not qualify GPU cancellation latency, quality, speed, tools, model API
behavior beyond the tested callback, or production reliability. Those require
tests against the exact built image.

The complete patch and source hash manifest are under [`cancellation/`](cancellation/).
This derivative is not the production recommendation.

## Optional cancellation plus decode-burst-4 derivative

The separate `decode-burst/` patch applies only after the cancellation-only
derivative. It adds `TF_FLASH_DECODE_BURST`, whose default remains `1`; the
burst image sets it to `4`. The scheduler uses the same 2048-row EXL3 passes
and runs up to four completed decode rounds between ordinary prompt-prefill
passes while existing streams are active. A standalone fill and a newly
admitted prompt after decode-only work still receive immediate fill priority.
This is a prompt-latency versus active-stream responsiveness tradeoff, not a
claimed optimization. The cancellation-only image and default behavior remain
separate baselines.

Build on the same Docker host as the exact cancellation-only image. Use its local
image ID from the cancellation build receipt; the ID is not portable between
Docker stores or hosts:

```sh
PARENT_CANCEL_ID="$(python3 -c 'import json; print(json.load(open("./artifacts/cancellation-build-receipt.json"))["local_image_id"])')"
python3 build_burst.py --parent-cancellation-image-id "$PARENT_CANCEL_ID" \
  --tag local/qwen-tensorfold-native:0.6.0-cancel-burst4 \
  --receipt ./artifacts/burst4-build-receipt.json
```

The builder checks the parent variant and runtime revision, binds and rechecks a
throwaway local tag, verifies the cancellation-patched source hashes before
applying the burst patch, reinstalls without dependency resolution, verifies
the resulting source hash, and runs both test suites. The expected result is
eight scheduler and six cancellation CPU checks, with no skips. These mock-GPU
checks validate code paths and source composition only; they do not establish
GPU cancellation latency, prompt latency, decode rate, quality, or serving
behavior.

## Optional grammar image

Grammar support is a separate image and remains unqualified. Its pinned package delta is recorded in `grammar-requirements.txt`, with the complete tested 222-package inventory in `expected-grammar-inventory.json`. Build it only from the exact local image ID produced by the primary build:

```sh
PARENT_IMAGE_ID="$(python3 -c 'import json; print(json.load(open("./artifacts/build-receipt.json"))["local_image_id"])')"
python3 build_grammar.py --parent-image-id "$PARENT_IMAGE_ID" \
  --tag local/qwen-tensorfold-native:0.6.0-grammar \
  --receipt ./artifacts/grammar-build-receipt.json
```

The builder checks the requested local ID, binds a unique temporary Docker tag to that ID, verifies the binding before building, and records the parent and child rootfs identities. The grammar image's package inventory must exactly match its tested inventory. This build does not establish grammar-constrained generation correctness or tool compatibility.

## Qualification boundary

The cancellation-plus-burst4 derivative has a linked public build and activation receipt and passed 14 mocked-GPU CPU checks. Its startup and direct health check were observed, but the receipt records no named API roundtrip. No GPU cancellation-latency, speed, quality, full-context, restart/reboot recovery, endurance, or serving qualification is established by that build. Quality fixtures are included for inspection, but this kit has no portable quality-evaluation runner or model-result receipt. Grammar support is deliberately not included in the base image. This experimental recipe makes no performance or recommendation claim.

See the [recipe qualification record](../../recipes/qwen-tensorfold-native-exl3/qualification.json) and [third-party notices](THIRD_PARTY.md).

## Separate burst4 plus 1024-row prefill derivative

The optional `scheduling/chunk/` patch chains after the retained burst4 image.
It validates `TF_FLASH_PREFILL_ROWS` as `1024` or `2048` before GPU allocation;
when unset, the runtime's supplied constructor value remains unchanged. The
public experimental derivative selects 1024 rows and keeps
`TF_FLASH_DECODE_BURST=4`. The burst4-only 2048-row image remains a separate
baseline. The patch changes one TensorFold source file and does not update
dependencies.

Build it on each Docker host from the exact local ID of that host's burst4
image; local image IDs cannot be transferred between hosts:

```sh
PARENT_BURST_ID="$(python3 -c 'import json; print(json.load(open("./artifacts/burst4-build-receipt.json"))["local_image_id"])')"
python3 build_chunk.py --parent-burst-image-id "$PARENT_BURST_ID" \
  --tag local/qwen-tensorfold-native:0.6.0-cancel-burst4-prefill1024 \
  --receipt ./artifacts/chunk-build-receipt.json
```

The builder binds the exact parent image, verifies the runtime and burst patch
source hashes, applies the 1024-row patch without network access, checks the
installed source hash, and runs 19 cancellation/scheduler CPU tests (six
cancellation, eight burst, five prefill-row tests). CPU checks do not establish
GPU behavior, speed, quality, or serving reliability. This variant remains
experimental until separate GPU/API qualification.
