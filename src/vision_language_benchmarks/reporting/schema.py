from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema


@lru_cache(maxsize=1)
def result_schema() -> dict[str, Any]:
    schema_path = Path(__file__).with_name("result.schema.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_result(result: dict[str, Any]) -> None:
    jsonschema.validate(instance=result, schema=result_schema())
    metadata = result["benchmark_metadata"]
    if result["status"] != metadata["status"]:
        raise ValueError("Result and benchmark metadata statuses must match.")
    if result["status"] == "completed":
        metrics = result.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("Completed results must contain measured metrics.")
        required_metrics = (
            {"acc1", "acc5"}
            if result["task"] == "classification"
            else {
                "image_retrieval_recall@1",
                "image_retrieval_recall@5",
                "image_retrieval_recall@10",
                "text_retrieval_recall@1",
                "text_retrieval_recall@5",
                "text_retrieval_recall@10",
            }
        )
        if missing := required_metrics.difference(metrics):
            raise ValueError(
                f"Completed {result['task']} results are missing metrics: "
                f"{', '.join(sorted(missing))}."
            )
    elif "metrics" in result:
        raise ValueError("Unsuccessful results must not contain measured metrics.")
