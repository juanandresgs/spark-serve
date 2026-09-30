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
