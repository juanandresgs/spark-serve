# Maintaining the model comparison page

The repository's main README is the shareable page. It focuses on Qwen on one
Spark and GLM-5.3-Flash on two. The CLI presents the same recipe choices:

```sh
PYTHONPATH=src python3 -m spark_serve recipes options
PYTHONPATH=src python3 -m spark_serve recipes options --model qwen
PYTHONPATH=src python3 -m spark_serve --json recipes options --model glm
```

Run these commands from the source checkout. An installed CLI also needs a source
checkout; use `spark-serve recipes --source /path/to/spark-serve options` from
elsewhere. Discovery never starts, stops or migrates a model.

## Sources of truth

- `models.json` defines the two model choices, recommended option, alternatives,
  comparison columns and references to evidence. Each local number resolves to
  the original recipe's JSON receipt using `file` plus a `path` array of keys.
- `public-references.json` records the attributed public figures. These are
  reported observations with different configurations, not local A/B evidence.
- `README.template.md` contains the explanation and table placeholders. The
  renderer supplies choices, comparison scopes, numbers and public source links.
- `charts.json` selects existing table metrics and, for paired median/p95 cells,
  the reference index to plot. It contains labels and scope, never copied numeric
  measurements. `comparison_charts.py` generates accessible SVGs with each
  workload on a separate zero-based axis. Text labels distinguish both recipes;
  `<title>`/`<desc>`, Markdown alt text and nearby tables preserve numeric access.
- `SOURCES.md` records the dated public-source recheck and newer upstream reports.
- Recipe manifests retain their build settings and qualification status. The
  choices file refers to them; it does not replace or install those recipes.

A cell contains numeric `refs` and a Python-style numeric `format`, for example
`{0:.2f} tokens/s`. Two refs can produce a median/p95 pair or a correct/total
score. Display formatting never changes the underlying measurement.

## Change workflow

1. Add a dated benchmark receipt with its workload, settings, measurement
   definition, sample count and qualification limits. Preserve previous evidence.
2. Point a comparison cell at the receipt. Keep unmatched public comparisons in
   the public section, with explicit differences in each row. A lower latency or
   higher rate alone does not establish a quality win.
3. Update the prose when the evidence changes a recommendation or its caveats.
4. Generate and check the page:

   ```sh
   PYTHONPATH=src python3 -m spark_serve.comparisons
   PYTHONPATH=src python3 -m spark_serve.comparisons --check
   PYTHONPATH=src python3 -m unittest discover -s deploy -p test_comparisons.py
   ```

5. Refresh affected recipe asset hashes and the release `MANIFEST.json`, then run
   `PYTHONPATH=src python3 deploy/check.py`. Submit data, evidence, generated page
   and manifests in the same change.

CI verifies references, recommendation/guide consistency and exact generated-page
and chart content, including missing or obsolete charts. Headline percentages
resolve from the same table references as their charts. Review SVGs visually at
desktop and mobile widths after generation; each is a standalone shareable asset.
CI does not decide whether two experiments are scientifically comparable
or verify external claims anew. Review remains necessary for methodology and prose.

This is deliberately a small schema for the two supported model families. It
provides machine-readable choices and evidence references without introducing a
benchmark service, automatic model ranking or another deployment controller.
