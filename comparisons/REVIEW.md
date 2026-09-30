# README and chart review — September 30, 2026

Scope: public landing-page presentation and evidence-generated SVG assets,
tracked in [issue #8](https://github.com/juanandresgs/spark-serve/issues/8).
No new GPU benchmark, deployment, restart, model download or qualification was
performed. Publication is held for the user's preview review.

## Independent reviews

Three read-only Luna reviewers audited:

1. Local receipt references, arithmetic, metric definitions, sample counts,
   synthetic correctness, reasoning/coding tails, settings and rebuild scope.
2. Pinned public primary sources and current upstream READMEs. The source-recheck
   identities and historical-versus-newer distinctions are in `SOURCES.md` and
   `public-references.json`.
3. The finished README, charts, source data, graph axes/units/labels and deployment
   limits. The reviewer rendered all four assets after the final spacing change.

Resolved findings: removed coding maxima absent from the public receipts; dated
historical public GLM figures and linked newer reports; corrected joined words
in the performance guide; enlarged chart-panel spacing to prevent unit/heading
collisions. The final audit found no remaining numeric, unit or sample mismatch.
For integration with the structured-evidence contract, headline percentage claims
were replaced with absolute receipt values. Historical metadata gaps are not
filled by arithmetic, and this presentation does not override comparability gates.

## Verification

- `PYTHONPATH=src python3 deploy/check.py`: 201 manifested files, 12 recipes and
  10 offline site bindings passed; hardware qualification remains false.
- Seven comparison/CLI/graph tests passed, including changed-evidence propagation,
  tampered SVG detection, zero-origin bars, accessibility descriptions, rejected
  unmatched evidence and path traversal.
- Five standalone-source contract checks and ten existing reliability adapter
  tests passed. The package built and installed in an isolated local environment;
  installed CLI help and Qwen option discovery passed. Launcher syntax passed.
- All 21 local README links/assets exist. Changed presentation files were checked
  for private site names, paths and topology identifiers; none was found.
- Gitleaks directory scan returned three generic-key findings in manifests. Each
  was independently verified as a SHA-256 of `api_checks.py`, not a credential.
  This is a documented scanner false positive, not a clean scanner exit.
- Browser review confirmed all four images loaded, expandable tables rendered,
  and the page had no horizontal overflow at a 390px viewport. Desktop and full
  SVG visual inspection passed. Responsive viewport override was reset afterward.

The local preview uses a Markdown renderer with GitHub-style layout; it is not
proof of rendering on the public GitHub page. CI and public rendered-page checks
remain pending the user's authorization to publish. Full rebuild performance,
fresh-download clean-machine installation, soak/reboot and new-site GLM acceptance
remain outside this update's evidence.
