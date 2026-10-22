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
  adapters use a single photo template. DINOv2+dino.txt uses the explicit template
  `a photo of a {class}.` without a task prefix. P0 and P1 are not combined.
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

EmbeddingGemma Matryoshka runs pin `openclip-gemma==0.4.0` and pass the requested dimension
directly to both native `encode_image` and `encode_text` calls. The supported dimensions are
768, 512, 256, and 128. The library retains leading coordinates and L2-normalizes the truncated
vector; the adapter verifies the resulting width, finite values, and unit norm. Each dimension
is an independent full-dataset inference run, not a slice or reranking of another dimension's
features. See the [dimension workflow](embedding-dimensions.md) for strict execution and
validation commands, and the [verified 24-run report](../results/mrl-openclip-gemma-v0.4.0/embedding-dimension-analysis.md)
for the measured dimension/accuracy/storage trade-offs.

DINOv2+dino.txt is a separate fixed-width model family: both global modalities are 2048D, and
the adapter does not expose MRL truncation. Its image transform and CLIP tokenizer come from
OpenVLCLIP v0.6.0; no task prefix is added. P0 uses the standard 18 templates, while P1 uses
`a photo of a {class}.` Japanese evaluation is exploratory. Model loading is currently blocked
because the reviewed upstream materials do not state licenses for the external dino.txt Vision
Head and Text Encoder weights; there are no DINOv2+dino.txt accuracy or throughput results.
See the [integration status](dinotxt-integration.md).

Native API equivalence was validated separately on a deterministic sample of 2 CIFAR-10 images,
38 classification prompts, and 4 XM3600 first-caption texts across English and Japanese. All
20 dimension/category checks passed; maximum absolute difference from the mathematical
reference was `2.980e-08`, maximum relative difference on meaningful coordinates was
`1.589e-07`, minimum cosine similarity was `0.9999999999999996`, and maximum unit-norm deviation was
`9.541e-08`. Adapter NumPy FP32 outputs matched direct native outputs exactly. The validation
used `rtol=1e-5`, `atol=1e-6` on CUDA FP32 with `openclip-gemma==0.4.0`; it does not reproduce
the full-dataset accuracy scores. See the [validation report](../results/mrl-openclip-gemma-v0.4.0/validation/native-mrl-api-equivalence.md)
and [JSON](../results/mrl-openclip-gemma-v0.4.0/validation/native-mrl-api-equivalence.json).

On CUDA out-of-memory, the runner retries with a halved batch size down to one. The final
batch sizes and OOM history are stored in the result. If a batch of one still fails, the run
is retained as a failure rather than being reported as a measurement.

The result records image and text throughput separately, median and interquartile batch
inference times, component timings, peak CUDA allocated/reserved memory when available, the
requested and actual device, GPU/driver/runtime, model and tokenizer identifiers/revisions,
checkpoint fingerprints, dataset checksum, prompts, run configuration, source commit, and
lockfile hash. These values are produced by an actual run; the framework does not infer
performance from checkpoint size or hardware specifications.

## Clean-source execution and publication lifecycle

Publishable measurements use strict reproducibility mode. The preflight verifies that the
repository is a valid Git checkout, the expected full commit exists and is checked out, no
tracked or staged changes or unexpected untracked files are present, `uv.lock` is consistent,
the staging output is ignored, and CUDA plus the requested model/dataset configuration are
available. Each strict run repeats the source check before inference, requires CUDA without
CPU fallback, and verifies the same source revision and clean worktree after inference and
result persistence. `--force` does not bypass these checks.

Strict-run JSON, logs, manifests, and intermediate outputs belong in an ignored or external
staging directory, not the tracked publication directory. Publication uses a separate
lifecycle manifest to classify results as active, historical, superseded, invalid, failed,
incomplete, blocked, or skipped. Only validated active and historical results are leaderboard
eligible; dirty, superseded, invalid, and unsuccessful records are excluded or rejected. The
`vlbench lifecycle` command builds this manifest from raw results and optional run manifests,
without rewriting the raw measurements.

## Results and validation status

Raw per-run JSON is validated against
[`result.schema.json`](../src/vision_language_benchmarks/reporting/result.schema.json).
Reports contain numeric metrics only for completed runs and list blocked, skipped, and failed
runs separately. CSV, Markdown, metadata JSON, PNG, and SVG files are regenerated from the
raw JSON. The tests check that regenerating those files from an unchanged input produces
byte-identical outputs and that no empty placeholder plots are emitted.

The v0.1.0 raw archive contains 45 completed full-dataset CUDA FP32 measurements: 24
10,000-image CIFAR-10/CIFAR-100 classification runs for six English-compatible models under
separate P0 and P1 protocols; seven historical XM3600 `all_captions` runs; seven earlier
`official_first_caption` runs; and seven clean replacements. The primary report contains 38
eligible rows: the 24 CIFAR runs, seven all-caption runs, and seven active replacements. The
seven earlier first-caption records remain unchanged and are marked superseded in the
[lifecycle manifest](../results/v0.1.0/validation/result-lifecycle.json), not counted as
independent observations.

The audit found one clean worktree and six earlier first-caption records with
`working_tree_dirty: true`. Run-status evidence attributes the observed dirty state to
generated raw result files being untracked during later runs. The exact historical
uncommitted file set is not recoverable, so no contents are inferred and the historical raw
records are not rewritten. A separate preservation manifest records SHA-256 hashes for all
38 raw JSON files that existed before the replacements were published. Adapter smoke tests
are reported separately and are not benchmark measurements.

All seven replacements ran from frozen source commit
`9d170b6071eecd75d7100ca70092d44d86650f01`, with
`working_tree_dirty: false`, CUDA FP32, seed 0, requested batch size 16, 3,600 images, and
3,600 first-caption candidates per language. Each old/new pair has identical Recall@1/5/10
values in both retrieval directions; the maximum absolute difference is zero. Actual
batch sizes and OOM history are recorded per run. The clean-run comparison, checkpoint and
dataset identities, candidate ordering, and worktree states are documented in the
[reproducibility audit](../results/v0.1.0/validation/clean-xm3600-reproducibility-audit.json).

The pinned CLIP Benchmark evaluator and `vlbench` produce identical CIFAR-10 P0 Top-1 and
Top-5 values for OpenAI CLIP ViT-B/32: 89.83% and 99.61%, respectively. The parity record
documents the matching QuickGELU architecture, checkpoint hash, dataset checksum, prompt
count, precision, and zero absolute difference.

An end-to-end official XM3600 retrieval run was performed for multilingual OpenCLIP
ViT-B/32 on Japanese `official_first_caption` candidates, using the pinned reference commit
above. Both implementations used the same 3,600 ordered image IDs and caption IDs, checkpoint
file SHA-256, tokenizer revision and token IDs, preprocessing configuration, precision, and
candidate mappings. The official metric keys were verified as text-to-image for
`image_retrieval_recall@k` and image-to-text for `text_retrieval_recall@k`. All six metrics
have the same integer hit counts; their maximum absolute numeric difference is
`2.9404958135081927e-8`, within the recorded `5e-8` tolerance. The difference is due to
float32 mean aggregation in the official implementation versus higher-precision hit-count
fractions in `vlbench`. The per-run comparison is recorded in
[`official-xm3600-parity.json`](../results/v0.1.0/validation/official-xm3600-parity.json),
and the official CLI output is in
[`official-xm3600-openclip-ja.json`](../results/v0.1.0/validation/official-xm3600-openclip-ja.json).

These are two narrow parity results, not a universal parity claim. XM3600 end-to-end parity
was established only for this OpenCLIP checkpoint, Japanese language, FP32 precision,
preprocessing, tokenizer, and first-caption candidate set. The separate
[clean replacement parity record](../results/v0.1.0/validation/official-xm3600-parity-clean.json)
refers to the validated Japanese multilingual OpenCLIP run and frozen source commit; the
original historical parity record remains unchanged. The clean comparison has matching
integer hit counts and a maximum absolute numeric difference of
`2.9404958135081927e-8`, within the recorded `5e-8` tolerance. This does not establish parity
for other models, languages, datasets, or `all_captions`.
