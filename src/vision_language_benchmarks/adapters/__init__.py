"""Model-specific vision-language encoder adapters."""

from __future__ import annotations

import importlib
from typing import Any

from vision_language_benchmarks.adapters.base import VisionLanguageAdapter
from vision_language_benchmarks.embedding_dimensions import resolve_embedding_dimension
from vision_language_benchmarks.errors import DependencyUnavailableError
from vision_language_benchmarks.registry import get_model


def create_adapter(
    model_key: str,
    device: str,
    precision: str,
    embedding_dimension: int | None = None,
) -> VisionLanguageAdapter:
    spec = get_model(model_key)
    resolved_embedding_dimension = resolve_embedding_dimension(
        model_key,
        embedding_dimension,
    )
    modules = {
        "openclip": "vision_language_benchmarks.adapters.openclip",
        "embeddinggemma": "vision_language_benchmarks.adapters.openclip_gemma",
        "dinov2_dinotxt": "vision_language_benchmarks.adapters.dinov2_dinotxt",
        "openvlclip_v07": "vision_language_benchmarks.adapters.openvlclip_v07",
        "siglip2": "vision_language_benchmarks.adapters.siglip2",
        "japanese_clip": "vision_language_benchmarks.adapters.japanese_clip",
    }
    try:
        module_name = modules[spec.adapter]
    except KeyError as error:
        raise ValueError(f"No adapter is registered for model family {spec.adapter!r}.") from error
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise DependencyUnavailableError(
            f"Cannot import the {spec.adapter!r} adapter. Install the "
            f"'{spec.dependency_extra}' extra."
        ) from error
    adapter_type: Any = module.Adapter
    if spec.adapter == "embeddinggemma":
        return adapter_type(
            spec,
            device,
            precision,
            embedding_dimension=resolved_embedding_dimension,
        )
    return adapter_type(spec, device, precision)


__all__ = ["VisionLanguageAdapter", "create_adapter"]
