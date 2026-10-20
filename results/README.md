# Benchmark Results

The [v0.1.0 report](v0.1.0/report.md) contains 38 schema-validated, completed full-dataset
CUDA FP32 measurements: 24 CIFAR-10/100 classification runs, seven historical XM3600
`all_captions` runs, and seven new `official_first_caption` runs. Each official-compatible run
uses the complete 3,600-image split and 3,600 first-caption candidates for its language. The
[CSV summary](v0.1.0/summary.csv), [metadata](v0.1.0/metadata.json),
[run-status manifest](v0.1.0/run-status.json), and [raw per-run JSON](v0.1.0/raw/) are
published alongside the report. Sanitized real-model CUDA smoke-test execution evidence is in
[model-smoke-tests.json](v0.1.0/validation/model-smoke-tests.json). The
[official CIFAR-10 parity record](v0.1.0/validation/official-cifar10-parity.json) compares
OpenAI CLIP P0 with the pinned CLIP Benchmark implementation. The
[XM3600 end-to-end parity record](v0.1.0/validation/official-xm3600-parity.json) compares the
Japanese multilingual OpenCLIP first-caption run with the pinned official CLI; the
[official CLI output](v0.1.0/validation/official-xm3600-openclip-ja.json) is retained.
The [historical-results audit](v0.1.0/validation/historical-results-audit.json) records the
31 original raw-file hashes, provenance checks, and legacy XM3600 variant assignment without
modifying the measured JSON.

The seven historical XM3600 runs retain all captions (3,600 images; 7,200 English or 7,185
Japanese candidates). They are not directly comparable with the pinned official XM3600
selection, which uses the first caption per image, and their unequal language candidate counts
do not support a matched-candidate English/Japanese delta. These original raw files are
preserved unchanged and remain a separate `all_captions` result set.

The seven new official-compatible runs use P2, FP32, seed 0, requested batch size 16, and the
actual pretrained checkpoints. EmbeddingGemma 2, multilingual OpenCLIP ViT-B/32, and SigLIP 2
Base were evaluated in English and Japanese; rinna Japanese CLIP ViT-B/16 was evaluated in
Japanese. The report presents the two variants in separate tables and plots, and reports
matched-language deltas only for the three models with paired runs. Language differences are
observed performance under each caption condition, not causal estimates of language
understanding. Every result table row links to its raw JSON.

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
