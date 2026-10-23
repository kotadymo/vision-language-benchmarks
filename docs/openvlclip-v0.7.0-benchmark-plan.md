# Benchmark plan: OpenVLCLIP v0.7.0 new models

- **Status:** planned; no new adapter, measurements, or results are claimed by this document
- **Date:** 2026-10-23
- **Benchmark repository baseline:** [vision-language-benchmarks main](https://github.com/kotadymo/vision-language-benchmarks/tree/main), inspected at `d9be684fea9702fdfe722aa6beb137bae8ce2952`
- **Model-library release:** [openvlclip==0.7.0](https://github.com/kotadymo/openvlclip/releases/tag/v0.7.0) ([release notes](https://github.com/kotadymo/openvlclip/blob/v0.7.0/docs/release-notes-v0.7.0.md), [model provenance](https://github.com/kotadymo/openvlclip/blob/v0.7.0/docs/model-provenance-v0.7.0.md))
- **Scope of this change:** documentation only, pushed directly to `docs/` on `main`. Do not change benchmark code, `uv.lock`, CI, historical results, or model weights in this commit.

## 1. Goal and exact model scope

Add reproducible, full-dataset **image↔text** classification/retrieval measurements for the three checkpoints introduced in **OpenVLCLIP v0.7.0**, while preserving the existing evaluation protocols and historical results:

| Proposed benchmark registry key (not implemented) | OpenVLCLIP model name | Official pinned checkpoint | Global dimension | New evaluation path |
| --- | --- | --- | ---: | --- |
| `openvlclip_siglip2_base` | `siglip2-base` | `google/siglip2-base-patch16-224` @ `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | 768 | FixRes 224 image/text |
| `openvlclip_siglip2_naflex_base` | `siglip2-naflex-base` | `google/siglip2-base-patch16-naflex` @ `b53b807d3a2d5e2b3911292f2d69e5341cdc064c` | 768 | variable-resolution image/text |
| `openvlclip_qwen3_vl_embedding_2b` | `qwen3-vl-embedding-2b` | `Qwen/Qwen3-VL-Embedding-2B` @ `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | 2048 | processor-aware image/text and instruction-aware embeddings |

**Not in v0.7.0:** `Qwen3-VL-Embedding-8B`. Do not register, download, benchmark, or claim it as a released v0.7.0 model. Native video and mixed-modal evaluation are a *separate future campaign* (section 8), not a reason to delay image/text results.

The benchmark repository **already has** a direct Transformers adapter `siglip2_base_224` and published results for it. **Do not replace it or count it as a newly measured model.** Run an OpenVLCLIP FixRes adapter separately to check implementation parity and measure its overhead; avoid treating the two wrappers around the same checkpoint as independent model families. Existing `embeddinggemma2`, `dinov2_dinotxt`, CLIP/OpenCLIP, and Japanese-CLIP results remain immutable comparison baselines.

## 2. Existing framework constraints to preserve

The current architecture uses `ModelSpec` and `MODELS` in `src/vision_language_benchmarks/registry.py`, an `adapters.VisionLanguageAdapter` with `preprocess_images / encode_images / preprocess_texts / encode_texts / metadata`, and `create_adapter()` dispatch. The engine computes L2-normalized cosine scores; classification averages per-template normalized text features, and retrieval reports **T→I** versus **I→T** separately. Do not switch to SigLIP learned scale/bias logits for the shared ranking tables or interpret cosine scores as probabilities.

- `docs/methodology.md` and `docs/benchmark-specification.md` define P0/P1/P2; `docs/dinotxt-integration.md` is the model-integration reporting precedent.
- `docs/embedding-dimensions.md` and `embedding_dimensions.py` currently accept `--embedding-dimension` only for `embeddinggemma2`. Any Qwen MRL support must be explicit and backward-compatible; never slice a non-MRL SigLIP vector as though it were a supported embedding.
- `pyproject.toml` currently pins `openvlclip[dinotxt]==0.6.0`, `openclip-gemma==0.4.0`, and `transformers==5.20.1`. Resolve and lock a deliberate v0.7.0 dependency route before implementing adapters. Avoid silently breaking the pinned dino.txt/EmbeddingGemma historical paths or forcing conflicting old/new extras in one environment.
- Reuse existing CIFAR/XM3600 loaders, prompt generators, metrics, run lifecycle, reports, and comparison tools. No generalized new benchmarking framework, CI matrix, or broad test suite is needed.

## 3. Primary full-dataset experiment matrix

Run the **native embedding dimension** of each of the three new adapter paths under the same frozen benchmark source revision, after CUDA smoke/parity gates pass:

| Dataset / split | Protocol | Language | Candidates / samples | Metrics | Runs per model |
| --- | --- | --- | --- | --- | ---: |
| CIFAR-10 official test | P0, common 18-template CLIP Benchmark labels | en | 10,000 images; 10 classes | Top-1 / Top-5 | 1 |
| CIFAR-10 official test | P1, model-native policy | en | 10,000 images; 10 classes | Top-1 / Top-5 | 1 |
| CIFAR-100 official test | P0, common 18-template CLIP Benchmark labels | en | 10,000 images; 100 classes | Top-1 / Top-5 | 1 |
| CIFAR-100 official test | P1, model-native policy | en | 10,000 images; 100 classes | Top-1 / Top-5 | 1 |
| XM3600 official first caption | P2, `official_first_caption` | en | 3,600 images and 3,600 first captions | T→I and I→T R@1/5/10 | 1 |
| XM3600 official first caption | P2, `official_first_caption` | ja | same ordered 3,600 images and 3,600 Japanese first captions | T→I and I→T R@1/5/10 | 1 |

**Core target: 3 adapters × 6 conditions = 18 completed full-dataset runs.** Use a distinct result campaign such as `results/openvlclip-v0.7.0/`; there are no results there yet. Do not commit empty report rows or placeholder plots. P0 and P1 must be shown separately. Japanese XM3600 measures observed behavior, not certified Japanese-language competence; do not assert translation or language equivalence. Match IDs/candidate order/dataset checksums before any bilingual delta or baseline comparison.

For the direct Transformers `siglip2_base_224` baseline, **reuse existing validated raw scores**, not a redundant full rerun, unless a controlled same-environment throughput/parity comparison truly requires fresh inference. Show any historical hardware, source, tokenizer, or implementation differences beside the comparison.

### Score and prompt policies

- **P0:** preserve the existing 18 CLIP Benchmark class-name templates and class-prototype aggregation. Required model-side processing (Qwen instruction/chat template, for example) must be fixed and disclosed: P0 is **common candidate-text protocol**, *not* numerical equivalence to the official CLIP model pathway.
- **P1 FixRes/NaFlex:** use the model-card-style `This is a photo of a {c}.` single template; NaFlex must get its own documented policy rather than accidentally falling through to the generic `a photo of a {c}.` template. Check the exact text length and truncation behavior.
- **P1 Qwen:** choose and freeze an explicit classification instruction *and* class-label rendering policy, justified by the upstream implementation. Do not silently copy EmbeddingGemma `Classification` or the SigLIP template, and do not tune the instruction on CIFAR test labels.
- **P2:** use original EN/JA captions and official first-caption ordering, not machine translation. For Qwen, freeze and publish retrieval instruction(s) and query/document roles. If text-to-image and image-to-text need different text encodings because role instructions differ, encode and evaluate **each direction separately** and identify each policy in the run signature; do not reuse one text matrix with a mismatched semantic role. Any changed P2 protocol must be versioned and compared only under identical policy.
- Rank by **unit-normalized image/text embeddings and cosine similarity** for all three model paths. No SigLIP sigmoid probabilities, calibrated-probability claims, or unverified CLIP logit-scale equivalence.

## 4. Model-specific adapter work

**Shared minimal adapter:** use `openvlclip.create_model_and_transforms(model_name, pretrained="official", device=..., precision=..., logit_scale_mode="cosine")`; validate returned checkpoint/revision, native dimension, finite [N,D] features, and matching image/text space. Keep OpenVLCLIP and Transformers checkpoint loading pinned rather than accepting upstream `main`. Expand adapter dispatch and registry only for the three requested models.

1. **SigLIP 2 FixRes:** apply the OpenVLCLIP returned RGB→224 preprocessing, use its own model-specific tokenizer, output 768D image/text features. Separately compare paired embeddings and rankings against the existing direct `siglip2_base_224` adapter on identical inputs, preprocessing, prompts, precision and pinned weights. Distinguish numeric parity from end-to-end throughput.
2. **SigLIP 2 NaFlex:** keep processor-aware `pixel_values`, `pixel_attention_mask` and `spatial_shapes` together. Implement safe collation/microbatching for variable image aspect ratios without discarding masks or changing sample order. Verify square, portrait and landscape images; output 768D vectors. Record NaFlex patch budget (the v0.7.0 default allows up to 256 patches), processor/configuration revisions, valid patch counts and preprocessing identity.
3. **Qwen 2B:** preprocessing returns RGB PIL images for its processor; pass them to `encode_image()`. For text, call **`encode_texts()` with the actual instruction**; `get_tokenizer()` → token-ID `encode_text()` is intentionally unsupported. Preserve Qwen's chat template, attention mask and last-valid-token pooling; output 2048D at native dimension. Batch processor-aware inputs with bounded memory, and validate candidate ordering. Store *query* versus *document* instruction policy independently of candidate text and checkpoint identity.

Do not silently substitute the official Transformers SigLIP adapter as a fallback if the OpenVLCLIP path fails. Fail with a clear blocked/failed result and retain provenance. Adapter metadata must name the actual inference library, not merely the underlying architecture.

## 5. Resource, dependency and execution policy

- Primary publishable results: **CUDA FP32**, seed `0`, requested batch size `16` where feasible, the same GPU/device/runtime and clean frozen source commit. Measure and store *actual* image/text batch sizes and OOM fallback history; allow halving through the existing runner, never undocumented CPU fallback. For memory-heavy Qwen, preflight with batch size 1 or 2; an OOM is a recorded failure or documented retry, not a success.
- Optional separate campaign: BF16 for all three models *only after* FP32 validity and adequate CUDA support; do not combine BF16 and FP32 leaderboards or assert speedup without measurement.
- Keep PyPI `openvlclip==0.7.0` (and `[qwen3vl]` extra where required), checkpoint commits and all dependency versions fixed in a newly checked `uv.lock`. Document any compatibility or package-environment conflict before changing the existing dino.txt extra from v0.6.0; use isolated benchmark environments if simultaneous locks cannot coexist cleanly.
- Review model-card licenses and exact pinned sources, model access, CIFAR citation conditions and XM3600 attribution *before downloading or running*. Register a verified/blocked model-weight status when appropriate. Never redistribute weights, hidden tokens, datasets, or cached full embeddings.
- Record GPU model/VRAM/driver, CUDA/PyTorch/Transformers/OpenVLCLIP versions, actual numerical precision, checkpoint and tokenizer revision/hash, processor settings, preprocessing hash, prompt-template hash, instruction/role policy, dimension/truncation strategy, model load time, embedding throughput, peak VRAM/RAM, dataset checksums, run/source/lock SHA, run status and failure details.

## 6. Qwen MRL: follow-up, not part of the 18-run core

After the 2048D native baseline passes, consider **1024D, 512D and 256D** Qwen measurements under the same six conditions (18 more model-dimension runs). Optionally add 128D/64D later if useful. Each dimension must be its own inference run with the officially documented `truncate_dim` path and L2 normalization **after** truncation; validate the chosen dimension with OpenVLCLIP, record the strategy, and keep dimension+instruction+processor identity in the run hash. Do not treat the Qwen Sentence Transformers embedding route as interchangeable with the canonical Qwen/OpenVLCLIP processor/chat-template route; the upstream provenance documents a numerical discrepancy.

SigLIP 2 FixRes and NaFlex do **not** support MRL: do not offer the `--embedding-dimension` flag for them. Keep the older EmbeddingGemma 768/512/256/128 protocol and results intact. Report storage-byte projections as derived values and measured index size/VRAM only when observed.

## 7. Implementation phases and minimal release gates

| Phase | Focus | Completion evidence |
| --- | --- | --- |
| 0 — Freeze protocol | Verify release 0.7.0, model SHAs/licenses, dependency resolution, GPU availability, dataset permissions; freeze P0/P1/P2 prompt+Qwen instruction policies | Written protocol/registry values; no inference needed |
| 1 — Minimal adapters | Add three registry keys + model-specific adapters; NaFlex processor-aware batches and Qwen `encode_texts`; extend dimensions only if running MRL | Real CUDA smoke on representative image/text, finite normalized shape, ordering, pinned source |
| 2 — Targeted correctness | FixRes versus direct Transformers parity; NaFlex mask/shape/order; Qwen processor/chat-template/instruction and correct T→I/I→T metrics; run identity/lifecycle | Small fixed inputs and one or two focused tests for changed code; no broad CI matrix |
| 3 — Full measurements | Freeze clean benchmark commit, run **18 full conditions** to ignored/external staging using strict reproducibility; track blocked/OOM outcomes | Complete actual per-run JSON, requested/actual batch size, precision, source SHA, no dirty worktree |
| 4 — Publish results | Validate schema, checksums, positive pairs and source integrity; lifecycle classification; copy only verified active raw JSON into new campaign; generate report/CSV/plots | `results/openvlclip-v0.7.0/` contains actual metrics plus audit; prior results untouched |
| 5 — Optional extensions | Qwen MRL, BF16, or controlled same-hardware performance repeats only when scientifically useful | Separate result groups and explicit extra conditions |

Use existing per-change tests and lightweight CI once for final substantive code. For a **docs-only plan**, reviewing the Markdown/links is sufficient; do not trigger full GPUs, new GitHub Actions jobs, extensive mock suites, or historical reruns solely for this document.

## 8. Future native-video/mixed-modal evaluation (explicitly separate)

Qwen3-VL-Embedding-2B provides `encode_video()` and `encode_multimodal()`, but the benchmark repository currently focuses on still-image classification and image↔text retrieval. A native-video retrieval benchmark must first define and license a **video↔text** dataset, fixed FPS/max_frames/pixel budget, video-level ground-truth positives and independent scoring; it is **not equivalent** to frame sampling/search or XM3600. Mixed text+image queries similarly need new relevance labels. Document these as future proposals with no invented accuracy or video metrics and do not block the core 18 runs on video codec/FFmpeg availability.

## 9. Reproducibility and comparison rules

1. Use `vlbench doctor --reproducibility` followed by `vlbench run --strict-reproducibility --expected-source-commit <full SHA>` on a **clean, committed** checkout. Write in-progress output under ignored `.cache/` (or outside the repository); publish only after validation. The exact commands/model keys in this plan are **targets, not current runnable CLI commands**.
2. Match dataset split/checksum, class and candidate order, normalization, protocol, language, retrieval variant, dimension, instruction/role policy, precision, and device before comparing model results. Disclose differences in model version/preprocessing/prompt/hardware even when metrics are presented together.
3. Preserve all historical `results/v0.1.0/`, `results/mrl-openclip-gemma-v0.4.0/` and `results/dinov2-dinotxt-v0.6.0/` JSON and reports byte-for-byte; do not reuse an old run ID for an implementation-change run.
4. Publish **Top-1/Top-5** for classification, **R@1/R@5/R@10** in both retrieval directions, and **actual** inference/preprocessing timings, throughput and peak VRAM separately. Exclude incomplete/failed/blocked/superseded runs from leaderboards, but preserve status records.
5. Compare FixRes OpenVLCLIP against direct SigLIP 2 as an **implementation validation**, not a distinct model-quality challenger. NaFlex versus FixRes is a valid checkpoint/preprocessing comparison only with the difference identified; Qwen prompt/instruction choices must appear alongside its score.

## 10. Definition of done for the eventual benchmark campaign

- [ ] Registry offers exactly the three additional OpenVLCLIP model *paths* (while retaining all current keys).
- [ ] All checkpoints, library versions, tokenizer and processor policies, dataset permissions and licenses are pinned/audited.
- [ ] Real CUDA image/text smoke and targeted model-specific correctness checks pass.
- [ ] Primary 18 full-dataset native-dimension runs complete or each unsuccessful condition is transparently recorded (no fabricated completeness).
- [ ] P0/P1, EN/JA, T→I/I→T, and any instruction-dependent variants are not conflated.
- [ ] Clean-source, model-space, run-identity and raw-result lifecycle checks pass.
- [ ] Publication report + raw JSON + CSV + figures exist only for verified results, with model speed/memory evidence.
- [ ] Historical results and unrelated CI remain unchanged.

**Source documents:** [benchmark methodology](methodology.md), [benchmark specification](benchmark-specification.md), [embedding dimensions](embedding-dimensions.md), [DINO integration precedent](dinotxt-integration.md), [model/dataset licenses](model-and-dataset-licenses.md), [OpenVLCLIP v0.7.0 release notes](https://github.com/kotadymo/openvlclip/blob/v0.7.0/docs/release-notes-v0.7.0.md).
