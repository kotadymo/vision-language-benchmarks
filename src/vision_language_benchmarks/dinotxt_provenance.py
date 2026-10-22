from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from vision_language_benchmarks.atomic import content_hash, sha256_file
from vision_language_benchmarks.errors import DataIntegrityError

DINOTXT_OPENVLCLIP_VERSION = "0.6.0"
DINOTXT_OPENVLCLIP_MODEL_NAME = "dinov2-dinotxt"
DINOTXT_MODEL_ID = "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l"
DINOV2_SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


@dataclass(frozen=True, slots=True)
class DinoTxtArtifact:
    component: str
    kind: Literal["weight", "tokenizer"]
    filename: str
    sha256: str
    size_bytes: int
    license: str
    license_status: Literal["verified", "unverified"]


DINOTXT_COMPONENTS: tuple[DinoTxtArtifact, ...] = (
    DinoTxtArtifact(
        component="dinov2_backbone",
        kind="weight",
        filename="dinov2_vitl14_reg4_pretrain.pth",
        sha256="36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
        size_bytes=1_217_607_321,
        license="Apache-2.0",
        license_status="verified",
    ),
    DinoTxtArtifact(
        component="dinotxt_vision_head",
        kind="weight",
        filename="dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
        sha256="706fba5bcb7019d80e127f0ff123872a75e0e6099527cf1911db85bd61c88ef5",
        size_bytes=101_317_814,
        license="Not stated in the reviewed upstream materials",
        license_status="unverified",
    ),
    DinoTxtArtifact(
        component="dinotxt_text_encoder",
        kind="weight",
        filename="dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
        sha256="13378274c79f67af8b47b11dd57827d659e7b1ee01e4dde58d2266422ea7846a",
        size_bytes=2_152_639_923,
        license="Not stated in the reviewed upstream materials",
        license_status="unverified",
    ),
    DinoTxtArtifact(
        component="clip_bpe_vocabulary",
        kind="tokenizer",
        filename="bpe_simple_vocab_16e6.txt.gz",
        sha256="924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
        size_bytes=1_356_917,
        license="MIT",
        license_status="verified",
    ),
)

DINOTXT_CHECKPOINT_REVISION = (
    f"{DINOV2_SOURCE_REVISION}"
    f"+backbone-{DINOTXT_COMPONENTS[0].sha256}"
    f"+vision-{DINOTXT_COMPONENTS[1].sha256}"
    f"+text-{DINOTXT_COMPONENTS[2].sha256}"
    f"+bpe-{DINOTXT_COMPONENTS[3].sha256}"
)


def expected_component_records() -> list[dict[str, object]]:
    return [asdict(component) for component in DINOTXT_COMPONENTS]


def verify_cached_components(
    checkpoint_directory: Path,
    components: tuple[DinoTxtArtifact, ...] | None = None,
) -> list[dict[str, object]]:
    selected_components = components if components is not None else DINOTXT_COMPONENTS
    if not checkpoint_directory.is_dir():
        raise DataIntegrityError(
            f"OpenVLCLIP checkpoint directory does not exist: {checkpoint_directory}."
        )

    records: list[dict[str, object]] = []
    for component in selected_components:
        path = checkpoint_directory / component.filename
        if not path.is_file():
            raise DataIntegrityError(
                f"Expected dino.txt component is missing: {component.filename}."
            )
        observed_size = path.stat().st_size
        if observed_size != component.size_bytes:
            raise DataIntegrityError(
                f"Unexpected size for {component.filename}: {observed_size}; "
                f"expected {component.size_bytes} bytes."
            )
        observed_sha256 = sha256_file(path)
        if observed_sha256 != component.sha256:
            raise DataIntegrityError(
                f"SHA-256 mismatch for {component.filename}: {observed_sha256}."
            )
        records.append(
            {
                **asdict(component),
                "verified": True,
            }
        )
    return records


def checkpoint_fingerprint(
    component_records: list[dict[str, object]],
) -> dict[str, object]:
    weight_records = [
        record for record in component_records if record.get("kind") == "weight"
    ]
    if not weight_records:
        raise DataIntegrityError("No dino.txt weight components were verified.")
    checkpoint_files = [
        {
            "path": str(record["filename"]),
            "sha256": str(record["sha256"]),
        }
        for record in weight_records
    ]
    return {
        "checkpoint_files": checkpoint_files,
        "checkpoint_sha256": content_hash(
            [(record["path"], record["sha256"]) for record in checkpoint_files]
        ),
    }
