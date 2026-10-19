# Benchmark Results

The [v0.1.0 report](v0.1.0/report.md) contains 31 schema-validated, full-dataset CUDA FP32
measurements: 24 CIFAR-10/100 classification runs and seven XM3600 retrieval runs. The
[CSV summary](v0.1.0/summary.csv), [metadata](v0.1.0/metadata.json),
[run-status manifest](v0.1.0/run-status.json), and [raw per-run JSON](v0.1.0/raw/) are
published alongside the report. Sanitized real-model CUDA smoke-test execution evidence is in
[model-smoke-tests.json](v0.1.0/validation/model-smoke-tests.json). The
[official CIFAR-10 parity record](v0.1.0/validation/official-cifar10-parity.json) compares
OpenAI CLIP P0 with the pinned CLIP Benchmark implementation.

Regenerate derived artifacts from the raw JSON with:

```powershell
uv run --locked vlbench report --input results/v0.1.0/raw --output results/v0.1.0
```

Only complete and authorized measurements contribute numeric results. Failed attempts and
unmeasured/access-blocked combinations are not converted into scores; their statuses and
blockers are recorded in the run-status manifest and implementation status. The private
`run-manifest.json` remains git-ignored. Dataset images, model weights, embeddings, caches,
credentials, and machine-specific paths must never be committed. See
[reproducibility](../docs/reproducibility.md) for exact commands.
