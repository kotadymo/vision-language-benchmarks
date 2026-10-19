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

For the published XM3600 comparisons, all models share the same dataset checksum and ordered
image IDs. Within each language, caption candidate-ID order and positive-pair count also match
across models. English and Japanese use their own language-specific caption pools and are
reported separately.

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

The result keys follow the reference convention:

- `image_retrieval_recall@k`: text-to-image retrieval.
- `text_retrieval_recall@k`: image-to-text retrieval.

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

The published v0.1.0 measurements include full 10,000-image CIFAR-10 and CIFAR-100 test
evaluations for six English-compatible models under separate P0 and P1 protocols, plus
full 3,600-image XM3600 retrieval for English and Japanese. Three multilingual-capable models
have matched English/Japanese XM3600 runs; Japanese-only rinna CLIP is reported for Japanese
without an English comparison. All published measurements use CUDA FP32 on the RTX 4070 Ti
SUPER. Adapter smoke tests are reported separately and are not included in benchmark rankings.

The pinned CLIP Benchmark evaluator and `vlbench` produce identical CIFAR-10 P0 Top-1 and
Top-5 values for OpenAI CLIP ViT-B/32: 89.83% and 99.61%, respectively. The parity record
documents the matching QuickGELU architecture, checkpoint hash, dataset checksum, prompt
count, precision, and zero absolute difference. This parity claim applies only to that
model/dataset/protocol combination; it does not imply parity for other models or datasets.
