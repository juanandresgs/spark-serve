# Reproducible recipe and benchmark evidence

Use the linked JSON records here to identify **what ran**, **what was measured**,
and **why a recipe is recommended**. A source build, a passing API request and a
reboot test establish different things. No imported result qualifies a new machine.

The three record types have Draft 2020-12 JSON Schemas in `schemas/`:

| Record | Purpose | Identity |
| --- | --- | --- |
| `recipes/*.json` | Model/runtime pins, settings, hardware requirements and source hashes | Versioned ID plus SHA-256 content fingerprint |
| `runs/*.json` | Actual identity, environment, workload, request policy, measurements, accounting, failures and separate qualifications | Immutable ID plus fingerprint; links to a recipe fingerprint |
| `recommendations/*.json` | Reviewed choice for a model/hardware pair, rationale and retained alternatives | Versioned decision linking recipe and run fingerprints |

Fingerprints hash UTF-8 canonical JSON: sorted keys, compact separators, ASCII
escaping, finite numbers, excluding only the top-level `fingerprint` field.
They bind content, not authorship. Imported source receipts are retained under content-addressed `sources/` paths,
with their original public path recorded. Later source updates cannot erase earlier
evidence. Source files and receipts use SHA-256 over their
exact bytes. A source snapshot does not prove which image was actually running.
`historical_association` makes that distinction explicit; new runner records
require an operator attestation of the actual immutable server identity.

## Validate and inspect

From the repository root, using Python 3.11 or newer:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[evidence]'
python -m spark_serve.evidence
python -m spark_serve.comparisons --check
python -m unittest discover -s deploy -p 'test_*.py'
```

The serving CLI does not require JSON Schema dependencies. The optional `evidence`
extra installs the schema validator for contributors and benchmark emission. The
comparison reader always checks fingerprints, receipt values, links and decisions;
CI additionally checks every schema and generated-page freshness.

The importer `deploy/import_evidence.py` documents the initial migration from the
published receipts at public commit `2a4d956`. It is deterministic, refuses to
replace differing records, and preserves the original receipt files and table
specification (`legacy-comparisons-v1.json`). Do not rerun it to incorporate new
experiments: append new records instead. Imported cohorts are aggregate historical
evidence, not freshly executed individual runs. Missing metadata stays `null` or
is listed in `unknowns`. In particular, a receipt hash is not a fixture hash.

## Produce shareable benchmark records

The existing Qwen benchmark runner can emit one immutable record per request group,
in addition to its local raw results. First create a **sanitized server identity**
file with these exact keys. Replace the placeholders with observed server values;
never use your benchmark client's OS/driver as the server environment.

```json
{
  "recipe_id": "qwen-affine4-source-20260930-v1",
  "image_digest": "sha256:<64 lowercase hex characters from the running image>",
  "runtime_revision": "<40 lowercase hex characters>",
  "model_revision": "<40 lowercase hex characters>",
  "hardware": {
    "accelerator": "NVIDIA GB10",
    "nodes": 1,
    "memory_gb_per_node": 128,
    "topology": "single GPU"
  },
  "software": {
    "os": "<server OS version>",
    "kernel": "<server kernel version>",
    "driver": "<server NVIDIA driver version>",
    "cuda": "<server CUDA version>",
    "runtime_version": "<server TensorFold version>"
  }
}
```

With the intended recipe already running and permission to benchmark it:

```sh
PYTHONPATH=src python recipes/qwen38-flash-affine4-1spark/bench.py \
  --base http://127.0.0.1:8898/v1 --label affine \
  --out acceptance/raw --quick --clients 4 --reps 3 \
  --evidence-metadata acceptance/server-identity.json \
  --evidence-dir acceptance/shareable
```

This makes model requests; it never starts, stops or reconfigures a service. The
runner checks recipe source hashes and matching model/runtime revision pins before
execution. The actual image digest remains operator-attested, not remotely
verified by the benchmark client. Keep that limitation with the record.

Each group records fixture/message hashes, workload-code hash, concurrency,
sampling, caps, cache policy, actual prompt/output/cached-token counts when the API
supplies them, per-request measurements and group wall time. Missing accounting
makes throughput unavailable; it never becomes zero. Failures still produce records.

For context evidence, prepare the existing synthetic `fixtures.json` with
`context.py --prepare` in your raw output directory, then use the same command
with `--suite context` instead of `--quick --clients 4`. It emits separate
priming/appended-request records with strict three-marker retrieval correctness.
Cold/warm prefix-cache labels come only from API-reported cached prompt tokens;
otherwise the state remains unknown. It does not imply a cold OS cache or reboot.

Shareable output uses a field allowlist and excludes endpoint URLs, model text,
prompts, tool payloads, raw exceptions and environment dumps. Raw synthetic results
stay under `acceptance/raw`; do not commit that directory. Review your supplied
version strings and the shareable files before submission. The separate mixed-load,
quality and lifecycle programs still retain their original receipts; their future
adapters must use this same schema, not claim that throughput/context emission covers them.

## Metric contract

Every measurement carries its definition, unit, aggregation, sample count,
sample unit, included population and workload conditions. Do not infer one metric
from another or subtract measurements with different timing boundaries.

| Metric | Definition / unit |
| --- | --- |
| TTFT | Client request start to first nonempty content, reasoning or tool delta; seconds |
| Response latency | Client request start to complete stream or error; seconds |
| End-to-end throughput | Sum of completion tokens / whole group wall seconds, including prefill and queueing; tokens/s |
| Decode-rate proxy | Per request `(completion_tokens - 1) / (last nonempty delta time - first nonempty delta time)`; tokens/s. Chunks may contain multiple tokens, so this is not exact token timing. |
| Correct / cases | Explicit integer counts and denominators; bounded fixture correctness is not broad model quality |
| Reported throughput | Historical/external tokens/s with an incompletely retained timing definition; never silently relabeled as decode or end-to-end |

New speed groups report the median of available successful-transport request
measurements for TTFT/response/decode, with its actual sample count. Group throughput
includes failed-request time; if any output-token count is missing it is `null`.
The basic nonempty-response gate is not a correctness test. Historical p95 values
retain their source aggregation; unknown quantile methods cannot support a new
percentage claim. Cold, warm, mixed, uncontrolled and unknown cache states are
separate. The speed runner warms the engine but does not reset prefix cache, so it
records `uncontrolled`, not a fabricated cold-cache qualification.

`evidence.percent_change(left_run, left_metric, right_run, right_metric)` fails
closed unless metric definitions, units, aggregation, sample count/population, workload,
fixture hash, concurrency, sampling, request settings, cache state and server
environment match, actual identities are complete, values/sample counts exist,
and neither run has failures. Recipe settings may differ, but a runtime software-version change does not pass
this strict percentage gate. Side-by-side absolute complete-recipe results remain
useful; neither view isolates the causal effect of one setting. External reports and
incomplete historical cohorts cannot pass. A zero baseline is rejected. The return
value is signed relative change, not automatically an improvement.

Absolute historical values remain useful side by side with scope labels. The
existing table renderer consumes only structured run measurements, checks their
source receipts, and refuses free-form percentage cells. Public reports have
pinned source commits and `external_report` origin; they cannot qualify our builds.

## Corrections, reviews and integration

Write records with `evidence.write_immutable`: it validates, fsyncs a temporary file
and atomically links a new ID without overwriting an existing file. Git/CI checks
reject edits, renames and deletions of committed recipe/run/decision snapshots,
source receipts and versioned schemas. Schema evolution uses a new schema version.
Corrections require a new ID and an explanatory limitation; preserve the old record.
Use `--immutable-base <base-commit>` to run the same history check locally.

Recommendations are deliberate reviews, not fastest-number selectors. Add a new
reviewed decision, preserve alternatives, update the model's `recommendation_record`
and selection in `comparisons/models.json`, then regenerate the documentation.
The reader rejects a selection that disagrees with its reviewed decision. The v1
decisions import the already published choices; migration itself makes no new
production or publication decision.

Build, API/tool, correctness, clean-install, restart, reboot, performance and
endurance statuses each carry their own scope. For example, the independent Qwen
public build passed bounded API/tool/replay checks using existing verified weights;
its record does not inherit the earlier image's full performance run or qualify a
fresh-download install, reboot or soak.

The implementation and active evidence handoffs are tracked in `MASTER_PLAN.md`.
The future GLM bakeoff runner should retain raw receipts and feed this contract;
unknown metadata must not be reconstructed from today's defaults.
