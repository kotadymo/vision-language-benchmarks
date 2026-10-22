from __future__ import annotations

from typing import Final

EMBEDDINGGEMMA_MODEL_KEY: Final = "embeddinggemma2"
NATIVE_EMBEDDING_DIMENSION: Final = 768
SUPPORTED_EMBEDDING_DIMENSIONS: Final = (768, 512, 256, 128)
EMBEDDING_DIMENSION_STRATEGY: Final = "leading-dimensions-l2-v1"


def resolve_embedding_dimension(
    model_key: str,
    embedding_dimension: int | None,
) -> int | None:
    if model_key != EMBEDDINGGEMMA_MODEL_KEY:
        if embedding_dimension is not None:
            if model_key == "dinov2_dinotxt":
                raise ValueError(
                    "DINOv2+dino.txt returns fixed 2048D global embeddings and does not "
                    "support --embedding-dimension or MRL truncation."
                )
            raise ValueError(
                "--embedding-dimension is supported only for the embeddinggemma2 model."
            )
        return None

    selected_dimension = (
        NATIVE_EMBEDDING_DIMENSION
        if embedding_dimension is None
        else embedding_dimension
    )
    if (
        not isinstance(selected_dimension, int)
        or isinstance(selected_dimension, bool)
        or selected_dimension not in SUPPORTED_EMBEDDING_DIMENSIONS
    ):
        supported = ", ".join(str(value) for value in SUPPORTED_EMBEDDING_DIMENSIONS)
        raise ValueError(
            f"Unsupported EmbeddingGemma embedding dimension {selected_dimension!r}; "
            f"choose one of {supported}."
        )
    return selected_dimension
