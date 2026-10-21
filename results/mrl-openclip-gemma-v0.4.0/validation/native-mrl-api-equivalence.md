# Native MRL API equivalence validation

This is a small-sample numerical validation of native `truncate_dim` behavior. It is separate from the 24 full-dataset benchmark measurements and does not reproduce their classification or retrieval accuracy scores.

- Validation status: **COMPLETED**
- Validation source commit: `dfda41a311f7a6b86a89aec95b7dd2aeebfe3073`
- Execution time (UTC): `2026-10-21T10:23:47.254260Z`
- Package: `openclip-gemma==0.4.0`
- Model: `google/embeddinggemma-2` at revision `914f7f89142e33e77833254d9c9b90c3cef7303b`
- Checkpoint SHA-256: `1ff16e54c7410c6a1c8a8fddd076136fba78368eec798b45a0f9532e83b2f510`
- Device: NVIDIA GeForce RTX 4070 Ti SUPER · CUDA 13.0 · PyTorch 2.14.1+cu130 · Python 3.12.10
- Worktree clean during validation: `True`

For each dimension, native image/text outputs were produced by the released model API with `truncate_dim=N`. The reference independently normalizes the leading N coordinates of an output produced with `truncate_dim=768`. The manual prefix operation is used only for this equivalence check; it did not replace native inference in any published benchmark run.

Tolerance: `rtol=1e-05`, `atol=1e-06`. Relative error is reported only where the absolute reference coordinate exceeds the absolute tolerance.

## Dataset samples and prompt paths

- `cifar10`: CIFAR-10 test images at indices [0, 1]; classification prompts use the first two class names and the existing P0/P1 prompt generators. (checksum `744e4b130af776f7a1c7a19427963e1f6d1c63984a1bb2bc2ceb4da2ba54c470`).
  - 2 image samples.
  - `classification_p0_en`: 36 selected texts; text IDs hash `46d7090923186f14dfa23ba1c46da000e6eae9ba80fc4b2908a1ddf7b79aa118`, content hash `8b1c60b0f278751599d99d2874b01265453fe5715aa831c5a607f00d2c7f8bd1`.
  - `classification_p1_en`: 2 selected texts; text IDs hash `79213cf50bd5e59aac4ca5d552a29166c6922c68a9101582dc516fb8d003f3ef`, content hash `f57afb7fa3caf6c985b378dcbd1b64924ce4603b38c799c3c9134c8818eb603d`.
- `crossmodal3600`: The first two candidate records in canonical annotation order, using `official_first_caption` (caption index 0). (checksum `ed934a9782286a688ca79d4594aec679ca4accb8d738186f986c831711545fce`).
  - `en`: 2 selected texts; text IDs hash `773813644528b7923c17470683c147abc6a83ac1f6a529fb022da682a20ff7ea`, content hash `7a893a212362fb08a1139e9897d8801cd25011b3053427497f2ac51a60ca8c3c`.
  - `ja`: 2 selected texts; text IDs hash `773813644528b7923c17470683c147abc6a83ac1f6a529fb022da682a20ff7ea`, content hash `395cd9b8866d97d14172e79d1d6e4778b4ddd645458be917653012d3dc1249e7`.

Classification uses the existing CIFAR-10 P0 and P1 prompt-generation functions with the EmbeddingGemma `Classification` task prompt. Retrieval uses the existing XM3600 `official_first_caption` candidate construction and the `SearchQuery` task prompt for English and Japanese.

## Native output versus mathematical reference

| Dimension | Modality | Dataset / protocol / language | Embeddings | Max abs. | Mean abs. | Max relative* | Min cosine | Max norm deviation | Invalid | Result |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 128 | classification_text | cifar10, p0, en, Classification | 36 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.99999999999999978 | 8.373e-08 | 0 | PASS |
| 128 | classification_text | cifar10, p1, en, Classification | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1 | 5.947e-08 | 0 | PASS |
| 128 | image | cifar10 | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.99999999999999978 | 4.144e-08 | 0 | PASS |
| 128 | retrieval_text | crossmodal3600, p2, en, SearchQuery | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1 | 5.459e-08 | 0 | PASS |
| 128 | retrieval_text | crossmodal3600, p2, ja, SearchQuery | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1 | 4.380e-08 | 0 | PASS |
| 256 | classification_text | cifar10, p0, en, Classification | 36 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.99999999999999978 | 5.982e-08 | 0 | PASS |
| 256 | classification_text | cifar10, p1, en, Classification | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1 | 7.136e-09 | 0 | PASS |
| 256 | image | cifar10 | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.99999999999999989 | 4.667e-08 | 0 | PASS |
| 256 | retrieval_text | crossmodal3600, p2, en, SearchQuery | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.99999999999999978 | 2.370e-08 | 0 | PASS |
| 256 | retrieval_text | crossmodal3600, p2, ja, SearchQuery | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1 | 2.068e-08 | 0 | PASS |
| 512 | classification_text | cifar10, p0, en, Classification | 36 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.99999999999999978 | 7.857e-08 | 0 | PASS |
| 512 | classification_text | cifar10, p1, en, Classification | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1 | 4.231e-08 | 0 | PASS |
| 512 | image | cifar10 | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1 | 5.479e-09 | 0 | PASS |
| 512 | retrieval_text | crossmodal3600, p2, en, SearchQuery | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1 | 2.079e-08 | 0 | PASS |
| 512 | retrieval_text | crossmodal3600, p2, ja, SearchQuery | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1 | 2.907e-08 | 0 | PASS |
| 768 | classification_text | cifar10, p0, en, Classification | 36 | 2.980e-08 | 1.063e-09 | 1.589e-07 | 0.99999999999999956 | 9.541e-08 | 0 | PASS |
| 768 | classification_text | cifar10, p1, en, Classification | 2 | 2.980e-08 | 9.835e-10 | 1.190e-07 | 1 | 5.841e-08 | 0 | PASS |
| 768 | image | cifar10 | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.99999999999999978 | 1.872e-08 | 0 | PASS |
| 768 | retrieval_text | crossmodal3600, p2, en, SearchQuery | 2 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1 | 3.398e-09 | 0 | PASS |
| 768 | retrieval_text | crossmodal3600, p2, ja, SearchQuery | 2 | 2.980e-08 | 1.363e-09 | 1.589e-07 | 0.99999999999999967 | 6.850e-08 | 0 | PASS |

* Relative error omits reference coordinates whose magnitude is at most `atol`; the number of included coordinates is recorded in the JSON.

## Adapter NumPy conversion

The adapter's returned NumPy FP32 arrays are compared with the direct native model outputs for the same inputs and dimension.

| Dimension | Modality | Dataset / protocol / language | Dtype | Max abs. | Mean abs. | Max norm deviation | Invalid | Result |
|---:|---|---|---|---:|---:|---:|---:|---|
| 128 | classification_text | cifar10, p0, en, Classification | float32 | 0.000e+00 | 0.000e+00 | 8.373e-08 | 0 | PASS |
| 128 | classification_text | cifar10, p1, en, Classification | float32 | 0.000e+00 | 0.000e+00 | 5.947e-08 | 0 | PASS |
| 128 | image | cifar10 | float32 | 0.000e+00 | 0.000e+00 | 4.144e-08 | 0 | PASS |
| 128 | retrieval_text | crossmodal3600, p2, en, SearchQuery | float32 | 0.000e+00 | 0.000e+00 | 5.459e-08 | 0 | PASS |
| 128 | retrieval_text | crossmodal3600, p2, ja, SearchQuery | float32 | 0.000e+00 | 0.000e+00 | 4.380e-08 | 0 | PASS |
| 256 | classification_text | cifar10, p0, en, Classification | float32 | 0.000e+00 | 0.000e+00 | 5.982e-08 | 0 | PASS |
| 256 | classification_text | cifar10, p1, en, Classification | float32 | 0.000e+00 | 0.000e+00 | 7.136e-09 | 0 | PASS |
| 256 | image | cifar10 | float32 | 0.000e+00 | 0.000e+00 | 4.667e-08 | 0 | PASS |
| 256 | retrieval_text | crossmodal3600, p2, en, SearchQuery | float32 | 0.000e+00 | 0.000e+00 | 2.370e-08 | 0 | PASS |
| 256 | retrieval_text | crossmodal3600, p2, ja, SearchQuery | float32 | 0.000e+00 | 0.000e+00 | 2.068e-08 | 0 | PASS |
| 512 | classification_text | cifar10, p0, en, Classification | float32 | 0.000e+00 | 0.000e+00 | 7.857e-08 | 0 | PASS |
| 512 | classification_text | cifar10, p1, en, Classification | float32 | 0.000e+00 | 0.000e+00 | 4.231e-08 | 0 | PASS |
| 512 | image | cifar10 | float32 | 0.000e+00 | 0.000e+00 | 5.479e-09 | 0 | PASS |
| 512 | retrieval_text | crossmodal3600, p2, en, SearchQuery | float32 | 0.000e+00 | 0.000e+00 | 2.079e-08 | 0 | PASS |
| 512 | retrieval_text | crossmodal3600, p2, ja, SearchQuery | float32 | 0.000e+00 | 0.000e+00 | 2.907e-08 | 0 | PASS |
| 768 | classification_text | cifar10, p0, en, Classification | float32 | 0.000e+00 | 0.000e+00 | 9.541e-08 | 0 | PASS |
| 768 | classification_text | cifar10, p1, en, Classification | float32 | 0.000e+00 | 0.000e+00 | 3.800e-08 | 0 | PASS |
| 768 | image | cifar10 | float32 | 0.000e+00 | 0.000e+00 | 1.872e-08 | 0 | PASS |
| 768 | retrieval_text | crossmodal3600, p2, en, SearchQuery | float32 | 0.000e+00 | 0.000e+00 | 3.398e-09 | 0 | PASS |
| 768 | retrieval_text | crossmodal3600, p2, ja, SearchQuery | float32 | 0.000e+00 | 0.000e+00 | 4.434e-08 | 0 | PASS |

Ranking agreement is not reported because the published raw records preserve aggregate Recall@K and candidate ordering, not per-query rankings or embeddings. That analysis requires separate full-dataset native inference.
