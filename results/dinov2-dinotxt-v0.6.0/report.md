# Vision-Language Benchmark Report

This report is regenerated from schema-validated raw JSON. Only completed runs contribute numeric metrics; failed, skipped, and blocked runs are listed separately. No values are estimated or fabricated.

## Conditions

- Validated result files: 6
- Completed measurements: 0
- Skipped runs: 0
- Failed runs: 0
- Blocked runs: 6
- Source commit(s): 53eeb22790fc8f135c3a94d589e2e422f8b6566a
- Runtime profile(s): not recorded
- Protocols and precisions are kept in separate tables and chart files.
- Accuracy and recall values in JSON are fractions; report tables display percentages.
- Checkpoint sizes are pinned weight-file sizes; tokenizer and configuration files may add to the download. Peak memory is reported as allocated / reserved GiB.
- Each completed table row links to its schema-validated raw result JSON.

## English zero-shot classification

No completed measured runs are available.

## Japanese/English official-compatible XM3600 retrieval (`official_first_caption`)

No completed measured runs are available.

## Japanese zero-shot classification

No completed measured runs are available.

## XM3600 all-caption retrieval (`all_captions`)

No completed measured runs are available.

## Matched English/Japanese retrieval

No unique, condition-matched English/Japanese result pairs are available.

## Model throughput and VRAM

No throughput measurements are available.

## Non-successful runs

### Blocked

| Model | Dataset | Language | Protocol | Retrieval variant | Model weight license | Reason | Run ID |
|---|---|---|---|---|---|---|---|
| DINOv2 ViT-L/14 + dino.txt | CIFAR-10 | en | p0 |  | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | a68a143e2e19b0d5 |
| DINOv2 ViT-L/14 + dino.txt | CIFAR-10 | en | p1 |  | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | a84a8a06e04ef7ed |
| DINOv2 ViT-L/14 + dino.txt | CIFAR-100 | en | p0 |  | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | c84d9fc9038d5f29 |
| DINOv2 ViT-L/14 + dino.txt | CIFAR-100 | en | p1 |  | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | 984949e7d7414e90 |
| DINOv2 ViT-L/14 + dino.txt | Crossmodal-3600 (XM3600) | en | p2 | official_first_caption | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | 7a9cfe6313eb4e28 |
| DINOv2 ViT-L/14 + dino.txt | Crossmodal-3600 (XM3600) | ja | p2 | official_first_caption | blocked | OpenVLCLIP v0.6.0 does not state a license for the external dino.txt Vision Head and Text Encoder weights. The package and DINOv2 backbone licenses do not establish rights for those separately distributed weights. | 0cbffc061dfc6663 |

## Validation and reproducibility

Each raw result is schema-validated before aggregation. Per-run JSON records retain status-appropriate provenance. Checkpoint fingerprints, dataset checksums, candidate IDs, and metrics are available only when a run reaches those stages; runs blocked earlier intentionally leave them unset.
- Validation artifacts: [checkpoint-integrity.json](validation/checkpoint-integrity.json).

## Known limitations

- Official numerical parity applies only to the exact combinations documented in the available validation records. It does not establish parity for other models, languages, datasets, precisions, or retrieval variants.
- Training-data contamination cannot be ruled out unless a model's pretraining corpus is independently verified.
- Dataset and model access restrictions remain in force. See [model-and-dataset-licenses.md](../../docs/model-and-dataset-licenses.md).
- Missing metrics remain missing. Unsupported languages and unavailable datasets are not represented as zero scores.

## Reproduction

```powershell
uv sync --locked --group dev
uv run vlbench doctor --device cuda
uv run vlbench report --input results/v0.1.0/raw --output results/v0.1.0
```

To reproduce a measurement, use the exact model revision, dataset checksum, split, protocol, language, precision, batch size, seed, source commit, and lockfile hash stored in its raw JSON.

## Figures

No meaningful figures are available from completed measurements.
