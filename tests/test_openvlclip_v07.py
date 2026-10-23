from __future__ import annotations

import numpy as np
import pytest

from vision_language_benchmarks.evaluation.metrics import retrieval_metrics
from vision_language_benchmarks.prompts import (
    CLIP_BENCHMARK_CIFAR_TEMPLATES,
    QWEN3_VL_DEFAULT_INSTRUCTION,
    QWEN3_VL_RETRIEVAL_QUERY_INSTRUCTION,
    classification_prompt_groups,
    model_input_policy,
)
from vision_language_benchmarks.registry import get_model


@pytest.mark.parametrize(
    ("model_key", "revision", "dimension"),
    [
        (
            "openvlclip_siglip2_base",
            "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
            768,
        ),
        (
            "openvlclip_siglip2_naflex_base",
            "b53b807d3a2d5e2b3911292f2d69e5341cdc064c",
            768,
        ),
        (
            "openvlclip_qwen3_vl_embedding_2b",
            "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
            2048,
        ),
    ],
)
def test_openvlclip_v07_models_use_pinned_native_checkpoints(
    model_key: str,
    revision: str,
    dimension: int,
) -> None:
    spec = get_model(model_key)

    assert spec.adapter == "openvlclip_v07"
    assert spec.dependency_extra == "openvlclip07"
    assert spec.checkpoint_revision == revision
    assert spec.global_embedding_dimension == dimension
    assert spec.weight_license_status == "verified"


def test_siglip2_and_qwen_p1_prompt_policies_are_explicit() -> None:
    classes = ["aquatic_mammals", "bicycle"]
    fixres, fixres_policy, _ = classification_prompt_groups(
        classes,
        "p1",
        "openvlclip_siglip2_base",
    )
    naflex, naflex_policy, _ = classification_prompt_groups(
        classes,
        "p1",
        "openvlclip_siglip2_naflex_base",
    )
    qwen, qwen_policy, _ = classification_prompt_groups(
        classes,
        "p1",
        "openvlclip_qwen3_vl_embedding_2b",
    )

    assert fixres == [
        ["This is a photo of a aquatic_mammals."],
        ["This is a photo of a bicycle."],
    ]
    assert naflex == fixres
    assert fixres_policy != naflex_policy
    assert qwen == [["aquatic mammals"], ["bicycle"]]
    assert qwen_policy == "qwen3_vl_default_instruction_spaced_labels"
    assert CLIP_BENCHMARK_CIFAR_TEMPLATES[0] == "a photo of a {c}."


def test_qwen_input_policy_freezes_classification_and_retrieval_roles() -> None:
    classification = model_input_policy(
        "openvlclip_qwen3_vl_embedding_2b",
        "classification",
        "p1",
    )
    retrieval = model_input_policy(
        "openvlclip_qwen3_vl_embedding_2b",
        "retrieval",
        "p2",
    )

    assert classification is not None
    assert classification["instruction"] == QWEN3_VL_DEFAULT_INSTRUCTION
    assert classification["label_rendering"] == "replace underscores with spaces"
    assert retrieval is not None
    assert retrieval["text_to_image"] == {
        "role": "query",
        "instruction": QWEN3_VL_RETRIEVAL_QUERY_INSTRUCTION,
    }
    assert retrieval["image_to_text"] == {
        "role": "document",
        "instruction": QWEN3_VL_DEFAULT_INSTRUCTION,
    }
    assert model_input_policy("siglip2_base_224", "classification", "p0") is None


def test_retrieval_metrics_use_separate_query_and_document_features() -> None:
    images = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query_texts = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    document_texts = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    metrics = retrieval_metrics(
        images,
        document_texts,
        [0, 1],
        ks=(1,),
        text_query_features=query_texts,
    )

    assert metrics["image_retrieval_recall@1"] == pytest.approx(1.0)
    assert metrics["text_retrieval_recall@1"] == pytest.approx(0.5)
