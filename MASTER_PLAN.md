# Deployment release plan

The release contains serving source, build patches and exact model-source
revisions, with no model weights. Recipes preserve measured serving settings;
site bindings supply host identity, storage and network configuration.

## Completed candidate work

- Preserve eleven recipes, including five portable deployment paths and their
  original qualification limits.
- Prepare pinned public image and runtime sources, including the full-GLM base.
- Verify recipe integrity and render every adapter against two offline sites.
- Build and test the native ready router for Linux ARM64.
- Stage the package on ARM64 Linux; validate the installed CLI, derived gateway
  configuration and resource permissions without starting models.
- Pass streamed text, tool and image checks against the existing GLM runtime.

## Before a qualified stable release

- Compile the fresh GLM images on idle Sparks and record their actual identities.
- Start GLM Flash, Qwen TP1 and full GLM through the portable broker adapters.
- Repeat representative performance checks with identical prompts, concurrency
  and completed-token accounting.
- Verify all-rank stop, managed restart and restoration of the prior service.
- Record the final source, image, model and site receipts together.

Track these gates in the repository's release-qualification issue. Keep completed
history; do not treat the older runtime's measurements as acceptance of a fresh
image or site. Contributions should include a small reversible change and checks
appropriate to its behavior. `deploy/check.py` is the offline check; `deploy/smoke.py`
provides API and optional disruptive restart acceptance.

## Qwen cooperative source recipe — issue #2

- Publish source-pinned adapters, source-build/run steps, synthetic benchmark tools,
  aggregate receipts and a causal metric ledger under `experiments/qwen-tensorfold-cooperative`.
- Two Luna reviews verify metrics, source behavior, limits and primary-source claims;
  Sol supplies a ranked improvement analysis.
- Export/privacy/CPU checks support this source candidate. Fresh GPU build, cold
  boot, sustained load and full-slot cancellation remain future qualification.
- No existing deployment, model weights or binary image is changed/distributed.

## Recommended GLM and Qwen entry points — issue #4

- Recommend the two-Spark adaptive GLM recipe and the one-Spark Affine4 Qwen recipe.
- Publish the allowlisted Affine4 source kit, pinned build receipt, benchmark
  methodology and independent adapter code. Keep EXL3 and vLLM alternatives.
- Show matched local variants separately from attributed, unmatched public numbers.
- Register Affine4 as a standalone source recipe; reject broker preparation until
  a portable adapter exists. Preserve existing deployment configurations.
- Check source manifests, catalog behavior, CPU adapter tests and shell syntax.
  GPU build/API/replay evidence is included; full rebuild performance, clean-machine
  install, reboot and endurance checks remain explicitly outstanding.

## Data-driven model comparison page — issue #6

- Define focused Qwen/GLM choices with links to retained alternatives.
- Generate README tables from existing numeric benchmark receipts; keep public
  reported figures and unmatched comparison boundaries explicit.
- Expose the same choices through `recipes options`, model filtering and JSON.
- Verify generated-page freshness, missing evidence, recommendation consistency
  and actual CLI behavior. No new benchmark or deployment qualification is claimed.

## Structured recipe and evidence contract — v1

Implemented in isolated feature work: three linked JSON record types with content
fingerprints, JSON Schema and semantic validation; immutable historical Qwen/GLM
cohorts; separately pinned external reports; reviewed recommendation snapshots;
existing table generation redirected to records; fail-closed percentage comparison;
and sanitized, create-only Qwen speed/context-run output. Original receipts and failed
mitigation evidence remain available. See `evidence/README.md`.

Acceptance: schema/reference/receipt integrity, immutable-history checks, invalid
comparison cases, real runner emission against a local synthetic HTTP boundary,
existing comparison/CLI tests, offline source/recipe/site checks and adapter tests.
These are software checks, not new hardware qualifications.

Integration obligations (evidence maintainer owns reconciliation):

| Dependency / owner | Resumption trigger | Acceptance |
| --- | --- | --- |
| README/graph presentation / evidence maintainer | Completed locally through presentation commit `a02b9f7` | Combined 15 evidence/chart tests pass; README and four SVGs are byte-identical to reviewed mobile presentation; numeric refs use validated records and percentage bypass is removed |
| GLM bakeoff / GLM research maintainer | Final sanitized receipts and exact source/image IDs are handed off | Append v1 runs with actual fixture/request/cache/environment/sample metadata, preserve failures and separate qualifications; propose a new reviewed decision only with operator approval |
| Remaining mixed-load, quality and lifecycle runners / evidence maintainer | Next authorized change to the corresponding runner | Emit the same run schema with domain-specific accounting and failure scopes; retain old receipts, do not pretend throughput/context emission covers those suites |

Publication and production promotion remain explicit operator decisions. The
initial normalization changes neither serving state nor the existing choices.

## Approachable README and generated charts — issue #8

- Rewrite the landing page around one-Spark Qwen Affine4 and two-Spark GLM
  adaptive DFlash2, with clear setup paths and retained alternatives.
- Generate four accessible SVGs from the existing comparison references. Keep
  models/hardware/workloads separate, show code and reasoning tradeoffs, retain
  numeric tables, and check page/chart freshness and evidence drift.
- Independently audit local receipts, pinned/current public sources, and the final
  README/charts with three Luna reviewers. Remove unsupported coding maxima;
  explicitly date historical public GLM figures and link to newer reports.
- Complete local checks and desktop/mobile visual review. Hold publication for
  the user's review of the rendered preview; no infrastructure or new benchmark.
