from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ProtocolName = Literal["p0", "p1", "p2"]
RunStatus = Literal["pending", "running", "completed", "skipped", "failed", "blocked"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    name: str
    adapter: str
    model_id: str
    checkpoint_revision: str | None
    pretrained_tag: str | None
    license: str
    languages: tuple[str, ...]
    supported_tasks: tuple[str, ...]
    dependency_extra: str
    architecture: str | None = None
    hf_repository: str | None = None
    repository_revision: str | None = None
    tokenizer_repository: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_license: str | None = None
    compatibility_note: str = ""
    supports_bf16: bool = True
    checkpoint_filename: str | None = None
    checkpoint_size_bytes: int | None = None
    uses_openclip_builtin_config: bool = False


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    key: str
    name: str
    task: str
    split: str
    expected_images: int | None
    expected_texts: int | None
    expected_classes: int | None
    languages: tuple[str, ...]
    language_codes: tuple[tuple[str, str], ...]
    license: str
    source: str
    access_status: str
    license_status: str = "unverified"
    access_reason: str = ""
    duplicate_of: str | None = None
    expected_text_counts_by_language: tuple[tuple[str, int], ...] = ()

    def source_language_code(self, language: str) -> str:
        codes = dict(self.language_codes)
        try:
            return codes[language]
        except KeyError as error:
            raise ValueError(
                f"Dataset {self.key!r} does not provide language {language!r}; "
                f"available languages: {', '.join(self.languages)}."
            ) from error

    def expected_text_count(self, language: str) -> int | None:
        counts = dict(self.expected_text_counts_by_language)
        return counts.get(canonical_language(language), self.expected_texts)


MODELS: dict[str, ModelSpec] = {
    "embeddinggemma2": ModelSpec(
        key="embeddinggemma2",
        name="EmbeddingGemma 2",
        adapter="embeddinggemma",
        model_id="google/embeddinggemma-2",
        checkpoint_revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
        pretrained_tag="official",
        license="Apache-2.0",
        languages=("en", "ja"),
        supported_tasks=("classification", "retrieval"),
        dependency_extra="gemma",
        hf_repository="google/embeddinggemma-2",
        checkpoint_filename="model.safetensors",
        checkpoint_size_bytes=1_488_915_288,
        compatibility_note=(
            "Uses the official Classification or SearchQuery task prompt; "
            "P0 scores are not numerically interchangeable with CLIP logits."
        ),
    ),
    "openai_clip_b32": ModelSpec(
        key="openai_clip_b32",
        name="OpenAI CLIP ViT-B/32",
        adapter="openclip",
        model_id="ViT-B-32",
        checkpoint_revision=None,
        pretrained_tag="openai",
        license="MIT",
        languages=("en",),
        supported_tasks=("classification", "retrieval"),
        dependency_extra="openclip",
        architecture="ViT-B-32",
        checkpoint_filename="ViT-B-32.pt",
        checkpoint_size_bytes=353_976_522,
        compatibility_note=(
            "The OpenAI checkpoint is selected by the open_clip 'openai' tag; "
            "record its downloaded checkpoint hash because the upstream weight "
            "is not a revision-addressable Hugging Face repository."
        ),
    ),
    "openclip_b32_laion2b": ModelSpec(
        key="openclip_b32_laion2b",
        name="OpenCLIP ViT-B/32 LAION-2B",
        adapter="openclip",
        model_id="ViT-B-32",
        checkpoint_revision="1a25a446712ba5ee05982a381eed697ef9b435cf",
        pretrained_tag="laion2b_s34b_b79k",
        license="MIT",
        languages=("en",),
        supported_tasks=("classification", "retrieval"),
        dependency_extra="openclip",
        architecture="ViT-B-32",
        hf_repository="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        checkpoint_filename="open_clip_model.safetensors",
        checkpoint_size_bytes=605_143_316,
        compatibility_note="English evaluation only; no Japanese claim is made.",
    ),
    "openclip_l14_laion2b": ModelSpec(
        key="openclip_l14_laion2b",
        name="OpenCLIP ViT-L/14 LAION-2B",
        adapter="openclip",
        model_id="ViT-L-14",
        checkpoint_revision="1627032197142fbe2a7cfec626f4ced3ae60d07a",
        pretrained_tag="laion2b_s32b_b82k",
        license="MIT",
        languages=("en",),
        supported_tasks=("classification", "retrieval"),
        dependency_extra="openclip",
        architecture="ViT-L-14",
        hf_repository="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        checkpoint_filename="open_clip_pytorch_model.safetensors",
        checkpoint_size_bytes=1_710_517_748,
        compatibility_note="English evaluation only; no Japanese claim is made.",
    ),
    "siglip2_base_224": ModelSpec(
        key="siglip2_base_224",
        name="SigLIP 2 Base/16 224",
        adapter="siglip2",
        model_id="google/siglip2-base-patch16-224",
        checkpoint_revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        pretrained_tag=None,
        license="Apache-2.0",
        languages=("en", "ja"),
        supported_tasks=("classification", "retrieval"),
        dependency_extra="siglip2",
        hf_repository="google/siglip2-base-patch16-224",
        checkpoint_filename="model.safetensors",
        checkpoint_size_bytes=1_500_800_904,
        compatibility_note=(
            "Uses the Transformers SigLIP 2 processor and feature methods; "
            "CLIP logit-scale assumptions are not applied."
        ),
    ),
    "rinna_ja_clip_b16": ModelSpec(
        key="rinna_ja_clip_b16",
        name="rinna Japanese CLIP ViT-B/16",
        adapter="japanese_clip",
        model_id="rinna/japanese-clip-vit-b-16",
        checkpoint_revision="ce22bc39e6f5dcb449e082954211acfb00c2227b",
        pretrained_tag=None,
        license="Apache-2.0",
        languages=("ja",),
        supported_tasks=("classification", "retrieval"),
        dependency_extra="japanese-clip",
        hf_repository="rinna/japanese-clip-vit-b-16",
        repository_revision="374e08d48b9ba72fcaf7459af4f8c93caffd6fb3",
        tokenizer_repository="rinna/japanese-roberta-base",
        tokenizer_revision="f93982a6f5a8bb9fd483b9deedae693091114b33",
        tokenizer_license="MIT",
        checkpoint_filename="model.safetensors",
        checkpoint_size_bytes=786_501_124,
        compatibility_note=(
            "Japanese-only. The separately pinned Japanese RoBERTa tokenizer is MIT-licensed."
        ),
    ),
    "openclip_multilingual_b32": ModelSpec(
        key="openclip_multilingual_b32",
        name="OpenCLIP Multilingual ViT-B/32",
        adapter="openclip",
        model_id="xlm-roberta-base-ViT-B-32",
        checkpoint_revision="506d40eb551f4801a1c27fc20a31c7b8f590deda",
        pretrained_tag="laion5b_s13b_b90k",
        license="MIT",
        languages=("en", "ja"),
        supported_tasks=("classification", "retrieval"),
        dependency_extra="openclip",
        architecture="xlm-roberta-base-ViT-B-32",
        hf_repository="laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k",
        tokenizer_repository="FacebookAI/xlm-roberta-base",
        tokenizer_revision="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        tokenizer_license="MIT",
        checkpoint_filename="open_clip_pytorch_model.bin",
        checkpoint_size_bytes=1_464_635_263,
        uses_openclip_builtin_config=True,
        compatibility_note=(
            "Multilingual text encoder; use the pinned XLM-R tokenizer and the built-in "
            "OpenCLIP architecture because this checkpoint repository has no "
            "open_clip_config.json."
        ),
    ),
}


DATASETS: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        key="cifar10",
        name="CIFAR-10",
        task="classification",
        split="test",
        expected_images=10_000,
        expected_texts=None,
        expected_classes=10,
        languages=("en",),
        language_codes=(("en", "en"),),
        license=(
            "The official University of Toronto page publishes the Python archive and "
            "requests citation of the dataset technical report; no separate acceptance "
            "agreement is displayed. This benchmark performs evaluation only and does "
            "not redistribute the archive."
        ),
        source="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        access_status="available",
        license_status="public_download_citation_required",
    ),
    "cifar100": DatasetSpec(
        key="cifar100",
        name="CIFAR-100",
        task="classification",
        split="test",
        expected_images=10_000,
        expected_texts=None,
        expected_classes=100,
        languages=("en",),
        language_codes=(("en", "en"),),
        license=(
            "The official University of Toronto page publishes the Python archive and "
            "requests citation of the dataset technical report; no separate acceptance "
            "agreement is displayed. This benchmark performs evaluation only and does "
            "not redistribute the archive."
        ),
        source="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        access_status="available",
        license_status="public_download_citation_required",
    ),
    "flickr30k": DatasetSpec(
        key="flickr30k",
        name="Flickr30k",
        task="retrieval",
        split="Karpathy test",
        expected_images=1_000,
        expected_texts=5_000,
        expected_classes=None,
        languages=("en",),
        language_codes=(("en", "en"),),
        license=(
            "Images are from Flickr and subject to Flickr Terms of Use. The caption "
            "documentation's stated license conflicts with its linked CC BY 3.0 license."
        ),
        source="https://shannon.cs.illinois.edu/DenotationGraph/",
        access_status="blocked",
        license_status="blocked",
        access_reason=(
            "The official project requires an access request and states that its Flickr "
            "images are subject to Flickr Terms of Use. Local archives do not establish "
            "that access was approved. Obtain and verify the official access grant and "
            "review the applicable terms before evaluation."
        ),
    ),
    "mscoco_captions": DatasetSpec(
        key="mscoco_captions",
        name="MS COCO Captions",
        task="retrieval",
        split="Karpathy 5K test",
        expected_images=5_000,
        expected_texts=25_010,
        expected_classes=None,
        languages=("en",),
        language_codes=(("en", "en"),),
        license=(
            "COCO annotations are CC BY 4.0. Images retain their original rights and are "
            "subject to Flickr Terms; downloading the dataset requires agreeing to the "
            "official COCO terms."
        ),
        source="https://cocodataset.org/#termsofuse",
        access_status="blocked",
        license_status="blocked",
        access_reason=(
            "The official COCO terms state that downloading the dataset signifies agreement; "
            "no user acceptance was provided. The current Karpathy-split preparer also does "
            "not preserve or validate the selected images' per-image rights. Do not download "
            "or evaluate until the user reviews and accepts the terms and image rights can be "
            "validated."
        ),
    ),
    "imagenet1k": DatasetSpec(
        key="imagenet1k",
        name="ImageNet-1K",
        task="classification",
        split="validation",
        expected_images=50_000,
        expected_texts=None,
        expected_classes=1_000,
        languages=("en",),
        language_codes=(("en", "en"),),
        license="ImageNet access terms and registration required.",
        source="https://image-net.org/",
        access_status="blocked",
        license_status="blocked",
        access_reason=(
            "The validation archive requires ImageNet authorization. No authorized "
            "local copy was found; do not bypass registration or access controls."
        ),
    ),
    "xtd10": DatasetSpec(
        key="xtd10",
        name="XTD10",
        task="retrieval",
        split="1K test",
        expected_images=1_000,
        expected_texts=1_000,
        expected_classes=None,
        languages=("en", "ja"),
        language_codes=(("en", "en"), ("ja", "jp")),
        license="Adobe Research License; noncommercial research only.",
        source="https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10",
        access_status="blocked",
        license_status="blocked",
        access_reason=(
            "The Adobe Research License states that exercising its rights accepts "
            "the terms. No user acceptance was provided; do not download or use."
        ),
    ),
    "crossmodal3600": DatasetSpec(
        key="crossmodal3600",
        name="Crossmodal-3600 (XM3600)",
        task="retrieval",
        split="all 3,600 images",
        expected_images=3_600,
        expected_texts=None,
        expected_classes=None,
        languages=("en", "ja"),
        language_codes=(("en", "en"), ("ja", "ja")),
        license=(
            "Google LLC licenses the XM3600 annotations under CC BY 4.0. The official "
            "image-attribution CSV lists all 3,600 images as CC BY 2.0; preserve "
            "attribution and do not redistribute the images."
        ),
        source="https://google.github.io/crossmodal-3600/",
        access_status="available",
        license_status="verified_attribution_required",
        expected_text_counts_by_language=(("en", 7_200), ("ja", 7_185)),
    ),
    "multilingual_mscoco_captions": DatasetSpec(
        key="multilingual_mscoco_captions",
        name="Multilingual MS COCO Captions",
        task="retrieval",
        split="XTD10 1K test",
        expected_images=1_000,
        expected_texts=1_000,
        expected_classes=None,
        languages=("en", "ja"),
        language_codes=(("en", "en"), ("ja", "jp")),
        license="Adobe Research License; same access constraint as XTD10.",
        source="https://github.com/adobe-research/Cross-lingual-Test-Dataset-XTD10",
        access_status="blocked",
        license_status="blocked",
        access_reason=(
            "This CLIP Benchmark dataset uses XTD10 assets and is not an independent "
            "evaluation. Its Adobe Research License requires user acceptance."
        ),
        duplicate_of="xtd10",
    ),
    "babel_imagenet": DatasetSpec(
        key="babel_imagenet",
        name="Babel-ImageNet",
        task="classification",
        split="ImageNet-1K validation, Japanese-labeled classes only",
        expected_images=None,
        expected_texts=None,
        expected_classes=None,
        languages=("en", "ja"),
        language_codes=(("en", "en"), ("ja", "ja")),
        license="Babel-ImageNet labels and ImageNet image terms must be verified.",
        source="https://github.com/gregor-ge/Babel-ImageNet",
        access_status="blocked",
        license_status="blocked",
        access_reason=(
            "Requires authorized ImageNet validation images and verified Japanese "
            "label coverage; neither is available locally."
        ),
    ),
}


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError as error:
        raise ValueError(
            f"Unknown model {key!r}; choose from {', '.join(sorted(MODELS))}."
        ) from error


def get_dataset(key: str) -> DatasetSpec:
    try:
        return DATASETS[key]
    except KeyError as error:
        raise ValueError(
            f"Unknown dataset {key!r}; choose from {', '.join(sorted(DATASETS))}."
        ) from error


def model_registry_dict() -> list[dict[str, object]]:
    return [asdict(MODELS[key]) for key in sorted(MODELS)]


def dataset_registry_dict() -> list[dict[str, object]]:
    return [asdict(DATASETS[key]) for key in sorted(DATASETS)]


def canonical_language(language: str) -> str:
    normalized = language.lower().strip()
    return "ja" if normalized in {"ja", "jp"} else normalized


def dataset_language_code(dataset_key: str, language: str) -> str:
    dataset = get_dataset(dataset_key)
    return dataset.source_language_code(canonical_language(language))


def language_is_supported(model: ModelSpec, language: str) -> bool:
    return canonical_language(language) in model.languages
