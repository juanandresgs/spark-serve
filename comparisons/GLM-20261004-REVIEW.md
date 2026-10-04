# GLM October 4 evidence and claim ledger

This ledger covers the dated GLM production panel in the root README. The
public [numeric projection](glm-production-20261004.json) contains every row of
the retention sequence, four earlier cohort summaries, and SHA-256 digests of
the original-site receipts. Those raw receipts include prompts, responses and
deployment details and remain private. A digest identifies a receipt; it does
not make that receipt public or qualify a new installation.

| Public claim | Original-site receipt | Public projection | Boundary |
| --- | --- | --- | --- |
| Retention candidate was the installed GLM configuration after promotion | `promotion-decision.json`, `final-live-promoted-1791151798767695322.json` | `receipts` digests; `configuration` | Bounded state on October 4, not a portable build or future health guarantee. |
| 3,131.098 versus 5,446.847 seconds; 42.515% less sequence elapsed | `a-sequence.json`, `b-sequence.json`, `comparison-reviewed.json` | `arms.*.sequence_seconds`, `comparison.observed_elapsed_reduction_percent` | One ten-request, same-budget cross-pair screen with no hardware swap; unequal output and completed work. |
| Strict 8/10 versus 7/10; shared JSON-format failures and control code cap | `a-sequence.json`, `b-sequence.json`, `comparison-reviewed.json` | `arms.*.rows`, `arms.*.strict_passes`, `comparison.failures` | Synthetic checks, not general model quality. The control produced no usable answer for one code revisit. |
| Cold first output is similar; retained long follow-up completes in 5.047 versus 1,660.614 seconds | `a-sequence.json`, `b-sequence.json` | `arms.*.rows[long-prime]`, `arms.*.rows[long-continuation]` | Both cold inputs report zero cached tokens. The follow-up includes different actual assistant histories, and both strict JSON checks fail. No cold-prefill speedup. |
| Final API 10/10, client tool roundtrip, actual V18 rollback, sampled memory floor | `final-api.json`, `final-client.json`, `candidate-operational-reviewed.json`, `final-stability-before-retirement.json`, `promotion-decision.json` | `receipts` digests | Operational acceptance on the original site only; no broad endurance or pending-snapshot-budget qualification. |
| Earlier V18 versus r4 cold C8 and warm C16 measures | `a-mixed.json`, `c-mixed.json`, `a-warm.json`, `c-warm.json` | `prior_interleave_screen` | Sequential production-pair comparison before retention; r4 cohorts emitted more total output tokens. The full cold cohort and 841K task took longer with r4. The 841K code task failed the same sandbox behavior gate on both arms. |

The graph uses separate zero-based axes for full-sequence elapsed time, cold
time to first output, and long-follow-up completion time. Its title, description,
series labels and adjacent numeric table state the limits without relying on
color. The rendered page and chart are generated from the projection by
`spark_serve.comparisons` and `spark_serve.glm_production`.

Three independent Luna reviews checked raw arithmetic and methodology,
deployment/qualification language, and chart readability. They confirmed the
published source digests and all displayed values. The reviews caught three
wording gaps that were corrected: the SVG footer now says each arm failed both
long strict-JSON checks; the projection now defines sequence wall time and its
inter-request overhead; and the earlier C8 text identifies four medium requests
that reused prefixes inside the initially cold cohort. The deployment review
also requested final live-state provenance; the projection includes the SHA-256
of the passing post-promotion live receipt.

Validation: comparison generation and `--check` passed; ten comparison tests
passed; `python3.11 deploy/check.py` verified 392 manifested files, 14 recipes
and ten offline site bindings. The SVG was rendered and inspected, and every
local link in the generated README resolves. These are offline publication
checks, not a new GPU benchmark or destination deployment qualification.
