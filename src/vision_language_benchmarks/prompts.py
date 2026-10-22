from __future__ import annotations

from collections.abc import Sequence

from vision_language_benchmarks.atomic import content_hash
from vision_language_benchmarks.registry import ProtocolName

CLIP_BENCHMARK_COMMIT = "ee14a9a8a9bcde09d51f4a4b1dff5a75182a3596"

CLIP_BENCHMARK_CIFAR_TEMPLATES: tuple[str, ...] = (
    "a photo of a {c}.",
    "a blurry photo of a {c}.",
    "a black and white photo of a {c}.",
    "a low contrast photo of a {c}.",
    "a high contrast photo of a {c}.",
    "a bad photo of a {c}.",
    "a good photo of a {c}.",
    "a photo of a small {c}.",
    "a photo of a big {c}.",
    "a photo of the {c}.",
    "a blurry photo of the {c}.",
    "a black and white photo of the {c}.",
    "a low contrast photo of the {c}.",
    "a high contrast photo of the {c}.",
    "a bad photo of the {c}.",
    "a good photo of the {c}.",
    "a photo of the small {c}.",
    "a photo of the big {c}.",
)

CLIP_SINGLE_TEMPLATE = "a photo of a {c}."
SIGLIP2_SINGLE_TEMPLATE = "This is a photo of a {c}."
DINOTXT_SINGLE_TEMPLATE = "a photo of a {c}."


def classification_prompt_groups(
    classnames: Sequence[str],
    protocol: ProtocolName,
    model_key: str,
) -> tuple[list[list[str]], str, str]:
    if protocol == "p0":
        templates = CLIP_BENCHMARK_CIFAR_TEMPLATES
        prompt_policy = "clip_benchmark_zeroshot_templates"
    elif protocol == "p1":
        if model_key == "embeddinggemma2":
            templates = ("{c}",)
            prompt_policy = "model_native_classification_task"
        elif model_key == "siglip2_base_224":
            templates = (SIGLIP2_SINGLE_TEMPLATE,)
            prompt_policy = "siglip2_model_card_photo_prompt"
        elif model_key == "dinov2_dinotxt":
            templates = (DINOTXT_SINGLE_TEMPLATE,)
            prompt_policy = "dinotxt_p1_single_photo_template"
        else:
            templates = (CLIP_SINGLE_TEMPLATE,)
            prompt_policy = "model_native_single_photo_template"
    else:
        raise ValueError("P2 is a paired-language retrieval protocol, not classification.")

    groups = [[template.format(c=name) for template in templates] for name in classnames]
    return groups, prompt_policy, content_hash({"templates": templates, "classes": classnames})


def retrieval_prompt_name(model_key: str) -> str | None:
    return "SearchQuery" if model_key == "embeddinggemma2" else None


def classification_prompt_name(model_key: str) -> str | None:
    return "Classification" if model_key == "embeddinggemma2" else None
