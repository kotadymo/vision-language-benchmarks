# Methodology

This document describes the implemented evaluation paths and distinguishes full measurements
from adapter smoke tests. The broader acceptance requirements are in the
[benchmark specification](benchmark-specification.md).

## Reference and protocols

The CLIP Benchmark reference is pinned to
[`ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596`](https://github.com/LAION-AI/CLIP_benchmark/commit/ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596).
The implementation records this reference in
[`prompts.py`](../src/vision_language_benchmarks/prompts.py).

- **P0** uses the CLIP Benchmark CIFAR zero-shot templates and class names. Each prompt
  embedding is L2-normalized, prompt embeddings are averaged per class, and each resulting
  class prototype is L2-normalized. Image embeddings are L2-normalized; their dot products
  with the class prototypes determine the ranking. Top-1 and Top-5 are reported as fractions.
  Top-5 is `null` if fewer than five classes are available.
- **P1** is a separate model-native classification comparison. EmbeddingGemma uses its
  `Classification` task prefix; SigLIP 2 uses its model-card photo template; other CLIP-family
  adapters use a single photo template. P0 and P1 are not combined.
- **P2** compares retrieval in English and Japanese on the same dataset. Paired plots and
  tables are emitted only when dataset checksum, model checkpoint, protocol, precision, seed,
  effective device, source commit, and ordered image IDs match. An unindexed `cuda` device is
  normalized to `cuda:0`; the recorded GPU identity and runtime must also match.

`P2` is the language-comparison protocol. XM3600 caption selection is a separate
`retrieval_variant` dimension and is included in configuration hashes, run IDs, manifests,
resume identity, raw metadata, and report grouping:

- **`official_first_caption`** follows the pinned CLIP Benchmark dataset builder: retain the
  first caption (`captions[0]`) for each image, preserve source annotation order, and exclude
  only records whose image file is missing. The local English and Japanese validation found
  3,600 images in each language, zero missing-image exclusions, identical ordered image IDs,
  and exactly one caption per image (3,600 text candidates per language). No records were
  duplicated, dropped to reach a target count, or synthesized.
- **`all_captions`** keeps every valid caption and the existing multiple-positive mapping:
  3,600 images with 7,200 English captions or 7,185 Japanese captions. These scores remain
  separate from the first-caption variant. The seven original XM3600 JSON records predate the
  variant field; their candidate counts identify them as `all_captions`, and the
  [historical audit](../results/v0.1.0/validation/historical-results-audit.json) records
  their SHA-256 hashes. The measured JSON files were not rewritten or relabeled.

English and Japanese in `official_first_caption` use the same ordered 3,600-image set and
3,600 candidates per language. A pair is eligible only when variant, candidate count and
ordered image IDs, dataset version/checksum, checkpoint, source commit, precision, preprocessing,
tokenizer, prompt, and runtime conditions match. Equal candidate counts control only pool size;
English and Japanese captions are not semantically identical in content, translation quality,
difficulty, or dataset composition. Their score differences are observed performance under
the stated language conditions, not a causal estimate of language understanding.

The historical `all_captions` English/Japanese candidate counts differ (7,200 versus 7,185),
so those runs are not a matched-candidate language comparison and are not paired for deltas.
Within each language, the old runs share image ordering and candidate IDs across models.

The CLIP Benchmark-compatible CIFAR prompt templates are the 18 strings defined in the
pinned reference. The model-specific prompt policy, task prefix, template hash, class-name
hash, and language code are recorded in each raw result.

## Retrieval metrics

Both image and text features are L2-normalized, and cosine similarity is computed with a
matrix product. Every caption is mapped to its positive image. Recall@1/5/10 is a hit rate:
each query scores 1 if at least one positive item appears in its top-k candidates, otherwise
0; the reported value is the mean over queries. With multiple captions per image, image-to-text
retrieval treats every caption for that image as positive. This matches the hit-based
image/text retrieval definition used by the pinned CLIP Benchmark implementation.

The result keys follow the pinned reference direction convention (the names refer to the
retrieved modality, not the query modality):

- `image_retrieval_recall@k`: text-to-image (T→I); the query is a caption and candidates are
  images.
- `text_retrieval_recall@k`: image-to-text (I→T); the query is an image and candidates are
  captions.

For `official_first_caption`, each image has exactly one positive caption and each caption has
one positive image. For `all_captions`, image-to-text queries have all captions for their image
as positives; recall remains a hit rate (at least one positive in top-k), rather than the
fractional number of positives retrieved. The differential test passes the same deterministic
synthetic features and positive-pair matrix to both directions of the pinned CLIP Benchmark
metric and `vlbench`, then compares R@1/5/10.

The implementation processes query rows in blocks and uses a stable sort for deterministic
tie handling. It validates feature shapes, finite values, embedding dimensions, positive-pair
mappings, and query coverage before reporting metrics.

## Inference and timing

Adapters use each model family's preprocessing and text encoder. Inference uses evaluation
mode and inference contexts; BF16 autocast is a separate opt-in condition and is rejected for
CPU or models/devices not marked as supporting it. Feature normalization is performed in
FP32 for metric computation.

EmbeddingGemma preserves its aspect-ratio-aware dynamic image resolution. Images are grouped by
their resulting tensor shape and encoded in same-shaped sub-batches; no extra resize or padding
is introduced. Its reported image batch size is the largest such sub-batch in the successful run.

On CUDA out-of-memory, the runner retries with a halved batch size down to one. The final
batch sizes and OOM history are stored in the result. If a batch of one still fails, the run
is retained as a failure rather than being reported as a measurement.

The result records image and text throughput separately, median and interquartile batch
inference times, component timings, peak CUDA allocated/reserved memory when available, the
requested and actual device, GPU/driver/runtime, model and tokenizer identifiers/revisions,
checkpoint fingerprints, dataset checksum, prompts, run configuration, source commit, and
lockfile hash. These values are produced by an actual run; the framework does not infer
performance from checkpoint size or hardware specifications.

## Results and validation status

Raw per-run JSON is validated against
[`result.schema.json`](../src/vision_language_benchmarks/reporting/result.schema.json).
Reports contain numeric metrics only for completed runs and list blocked, skipped, and failed
runs separately. CSV, Markdown, metadata JSON, PNG, and SVG files are regenerated from the
raw JSON. The tests check that regenerating those files from an unchanged input produces
byte-identical outputs and that no empty placeholder plots are emitted.

The historical v0.1.0 measurements include full 10,000-image CIFAR-10 and CIFAR-100 test
evaluations for six English-compatible models under separate P0 and P1 protocols, plus seven
full 3,600-image XM3600 `all_captions` retrieval runs. The `official_first_caption` selection,
validation, result separation, and reporting paths are implemented and unit-tested; report
status and raw results determine whether full CUDA evaluations have subsequently completed.
Adapter smoke tests are reported separately and are not included in benchmark rankings.

The pinned CLIP Benchmark evaluator and `vlbench` produce identical CIFAR-10 P0 Top-1 and
Top-5 values for OpenAI CLIP ViT-B/32: 89.83% and 99.61%, respectively. The parity record
documents the matching QuickGELU architecture, checkpoint hash, dataset checksum, prompt
count, precision, and zero absolute difference. This parity claim applies only to that
model/dataset/protocol combination; it does not imply end-to-end XM3600 parity for other models
or datasets. The XM3600 loader and metric differential evidence are recorded separately from
any full-model CLIP Benchmark evaluation.
