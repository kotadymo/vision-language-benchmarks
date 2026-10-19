# Benchmark Results

No full-dataset benchmark measurements have been published yet. Real-model smoke tests validate
adapter inference only; they are not benchmark results. Track model, dataset, and protocol
progress in [implementation status](../docs/implementation-status.md).

After an authorized evaluation, raw schema-validated JSON belongs in
`results/v0.1.0/raw/`. Regenerate the CSV, Markdown, metadata, PNG, and SVG reports from that
JSON with:

```powershell
uv run vlbench report --input results/v0.1.0/raw --output results/v0.1.0
```

Validated, sanitized per-run metric JSON under `raw/` may be committed. The private
`run-manifest.json` remains git-ignored. Do not add datasets, checkpoints, embeddings, caches,
credentials, or machine-specific paths. Publish only reports derived from genuine completed
runs and retain unsuccessful runs separately; see
[reproducibility](../docs/reproducibility.md).
