# TensorFold 0.6.0 native source build (experimental)

This is a public, portable source-build scaffold for TensorFold 0.6.0 and the pinned Qwen EXL3 model. It is retained for evaluation only. It is not the production recommendation, and this public source kit has not yet been rebuilt on a DGX Spark or qualified for GPU behavior, model quality, throughput, or serving reliability.

The base image, runtime commit, and model revision are pinned in `pins.json`. The base image's 216-package inventory and the tested primary runtime's 217-package inventory differ only by TensorFold 0.6.0. `runtime-constraints.txt` pins the base inventory; it is not an install requirements file. The Dockerfile installs the pinned TensorFold source without dependency resolution, checks that every active TensorFold dependency is present and satisfies its declared version, then compares the complete result with `expected-primary-inventory.json`. A build fails for missing packages, extra packages, or version drift. Capture method and package deltas are recorded in [`inventory-provenance.json`](inventory-provenance.json).

## Build and inspect

On a Linux ARM64 host with Docker and access to the pinned NVIDIA base image:

```sh
cd experiments/qwen-tensorfold-native-0.6
python3 build.py --tag local/qwen-tensorfold-native:0.6.0 \
  --receipt ./artifacts/build-receipt.json
```

The build pulls the exact base digest, checks out the exact TensorFold commit, validates the runtime version and complete package inventory, then records the local image ID, repository digests when available, rootfs diff IDs, source hashes, and pinned model/runtime identity. Image IDs are daemon-store-specific: compare repository and rootfs provenance across Docker stores rather than requiring `.Id` equality. The local `.Id` is not treated as a config digest.

This command builds the primary runtime only. It does not download model weights, create a serving configuration, or deploy anything. Use a separately reviewed model staging and launch procedure. Do not treat a successful build or matching package inventory as GPU or end-to-end qualification.

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

The scaffold still needs an actual build from this exported public kit on the target Spark, followed by model load and API/tool checks. Quality, speed, full-context behavior, restart/reboot recovery, endurance, and optional grammar support are unmeasured here. Grammar support is deliberately not included in this base image. No performance or recommendation claim is made by this experimental recipe.

See the [recipe qualification record](../../recipes/qwen-tensorfold-native-exl3/qualification.json) and [third-party notices](THIRD_PARTY.md).
