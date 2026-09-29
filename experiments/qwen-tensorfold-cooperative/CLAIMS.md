# Claim audit and verification boundaries

Before publication, two separate Luna subagents reviewed the metric ledger and
adapter/source claims. A Sol subagent produced the improvement analysis; Luna
also checked its external-source claims. These are audits of evidence and code,
not independent GPU reproductions or guarantees of correctness.

| Claim family | Evidence / review | Boundary |
|---|---|---|
| Historical TP2 and slot comparison | Aggregate rows, context and continuation in [history.json](history.json), checked against original receipts | Pack/runtime/topology/MTP differ; small synthetic samples |
| 99.23% short p95 reduction, +1.22% bulk time | Same-host control/selected summaries in [measurements.json](measurements.json); arithmetic and22 fingerprint matches checked | One crossover; selected source mounted, final image qualified separately |
| Stream pause362.737→2.867s | Packaged streaming rows and3 matching fingerprints | Separate Sparks, one run; p95 gap worsens, output still bursty |
| Cold535.134→536.452s, cached0.692→0.632s | Context receipts and actual token/cache counts | No material cold speedup; prompt-cold, not OS-cold |
| Correctness, token matching and cancellation | Qualification, token-ID and cancellation sections | Bounded fixtures; no general quality, full-slot, reboot or soak guarantee |
| Tool conversion | Source audit and installed-parser unit tests | Top-level schema-valid conversions; invalid values left for client rejection |
| Scheduling/head behavior | Source audit of original dispatcher/wrappers | Private pinned APIs; head-skip enabled in selected bundle but benefit not isolated |
| Further improvements | Sol analysis and primary-source review | Experiments and external leads, not achieved local gains |
| Public-source install | Syntax/CPU checks and export/privacy checks | Fresh container GPU build and clean-site launch remain unqualified |

The audit corrected an early reviewer assumption: the selected2048-row candidate
**does enable intermediate-head skipping**, alongside decode burst4. Its inclusion
is not evidence of a separate speed benefit. No marginal claim is presented as
statistical superiority, and no external benchmark is treated as a matched result.

The exact public file set is listed with SHA256 in [MANIFEST.json](MANIFEST.json).
The exporter uses explicit file names and rejects private machine/path patterns.
No private repository history, deployment reference, raw capture, credentials,
model weights or binary runtime image are part of this package.
