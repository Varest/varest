"""One-stage projector Efficient Tuning for LLaVA-NeXT VisionZip++.

The 779K LLaVA-NeXT dataset is read directly from its existing Parquet shards.
No Arrow cache, extracted image tree, or converted annotations are created.
Per-device batch size is intentionally one: the implemented inference contract
is one image/prompt and AnyRes view counts are ragged across source images.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from io import BytesIO
from json import dumps
from os import environ
from pathlib import Path
from random import Random
from typing import Any

import pyarrow.parquet as pq
import torch
from mle.vars import ExpConfig
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, get_scheduler, set_seed

from llmoe.backbone import VLM
from llmoe.config import Config
from llmoe.compressor.vision_zip.vision_zip import (
    VisionZipPlusPlusTrainingCompressor,
)
from llmoe.compressor.vision_zip.vision_zip_llava_next_pp import (
    FLASH_ATTN_VERSION,
    FINAL_TRAINING_RECIPE,
    TARGET_MODEL_ID,
    TARGET_MODEL_REVISION,
    TRAINING_ATTN_IMPLEMENTATION,
    checkpoint_metadata,
    load_projector_checkpoint,
    retention_target,
)


IGNORE_INDEX = -100
DATASET_DIR_NAME = "LLaVA-NeXT-Data"
DEFAULT_GLOBAL_BATCH_SIZE = 128
CHECKPOINT_FILENAME = "vision_zip_projector.pt"


@dataclass(frozen=True)
class VisionZipNextTrainingArgs:
    model_name_or_path: str
    model_revision: str
    data_path: str
    output_dir: str
    checkpoint_name: str
    initial_checkpoint_path: str | None
    retain_ratio: float
    num_train_epochs: float
    global_batch_size: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    warmup_ratio: float
    lr_scheduler_type: str
    model_max_length: int
    subset_frac: float
    dataset_limit: int | None
    dataset_total_rows: int
    dataset_image_rows: int
    dataset_shards: int
    dataloader_num_workers: int
    bf16: bool
    fp16: bool
    tf32: bool
    attn_implementation: str
    flash_attn_version: str
    flash_attn_wheel: str | None
    gradient_checkpointing: bool
    deepspeed_stage: int
    max_steps: int
    stop_after_steps: int | None
    save_steps: int
    save_total_limit: int
    logging_steps: int
    seed: int
    resume_from_checkpoint: str | None
    skip_final_save: bool


@dataclass(frozen=True)
class _RowLocation:
    shard: int
    row_group: int
    row: int


@dataclass(frozen=True)
class _ImageRowGroup:
    shard: int
    row_group: int
    row_count: int
    # ``None`` means every row in the group is image-bearing. A tuple is only
    # materialized for the single mixed row group in the supplied corpus.
    row_indices: tuple[int, ...] | None

    @property
    def image_count(self) -> int:
        return self.row_count if self.row_indices is None else len(self.row_indices)


def _image_rows_in_group(parquet: Any, group_index: int) -> tuple[int, ...] | None:
    """Return image-bearing row indices without reading embedded image bytes."""

    metadata = parquet.metadata.row_group(group_index)
    path_column = next(
        (
            metadata.column(index)
            for index in range(metadata.num_columns)
            if metadata.column(index).path_in_schema == "image.path"
        ),
        None,
    )
    if path_column is None:
        raise ValueError("LLaVA-NeXT Parquet schema is missing image.path")
    statistics = path_column.statistics
    if statistics is not None and statistics.null_count is not None:
        non_null = metadata.num_rows - int(statistics.null_count)
        if non_null == metadata.num_rows:
            return None
        if non_null == 0:
            return ()

    # Reading the nested path leaf is tiny (about 1 KiB for the supplied mixed
    # group) and preserves parent-struct validity; it never touches image.bytes.
    table = parquet.read_row_group(group_index, columns=["image.path"])
    values = table["image"].to_pylist()
    rows = tuple(index for index, value in enumerate(values) if value is not None)
    if statistics is not None and statistics.null_count is not None:
        expected = metadata.num_rows - int(statistics.null_count)
        if len(rows) != expected:
            raise ValueError(
                f"image.path metadata reports {expected} image rows but decoded {len(rows)}"
            )
    return rows


def _scan_image_groups(
    shards: Sequence[Path],
) -> tuple[list[_ImageRowGroup], int, int]:
    groups: list[_ImageRowGroup] = []
    total_rows = 0
    image_rows = 0
    for shard_index, shard in enumerate(shards):
        parquet = pq.ParquetFile(shard)
        for group_index in range(parquet.metadata.num_row_groups):
            metadata = parquet.metadata.row_group(group_index)
            total_rows += int(metadata.num_rows)
            row_indices = _image_rows_in_group(parquet, group_index)
            group = _ImageRowGroup(
                shard=shard_index,
                row_group=group_index,
                row_count=int(metadata.num_rows),
                row_indices=row_indices,
            )
            if group.image_count:
                groups.append(group)
                image_rows += group.image_count
    return groups, total_rows, image_rows


def _resolve_dataset_dir(input_dir: str | Path) -> Path | None:
    root = Path(input_dir)
    candidates = (root, root / DATASET_DIR_NAME)
    for candidate in candidates:
        if candidate.is_dir() and any((candidate / "data").glob("*.parquet")):
            return candidate
    return None


def vision_zip_pp_ft_fn_cd(config: ExpConfig) -> str:
    root = _resolve_dataset_dir(config.input_dir)
    if root is None:
        return f"Missing {DATASET_DIR_NAME}/data Parquet shards under {config.input_dir}"
    shards = sorted((root / "data").glob("*.parquet"))
    _, total_rows, image_rows = _scan_image_groups(shards)
    return (
        f"OK with {len(shards)} in-place LLaVA-NeXT Parquet shards, "
        f"{image_rows}/{total_rows} image-bearing rows"
    )


def _token_ids(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool = False,
) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    values = getattr(encoded, "input_ids", None)
    if values is None and isinstance(encoded, Mapping):
        values = encoded.get("input_ids")
    if values is None:
        raise TypeError("Llama-3 tokenizer did not return input_ids")
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("Expected one tokenized prompt")
        values = values[0]
    return [int(value) for value in values]


def llava_next_messages(conversations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert human/gpt records to the checkpoint's multimodal message form."""

    messages: list[dict[str, Any]] = []
    image_inserted = False
    for turn in conversations:
        source_role = str(turn.get("from", "")).lower()
        if source_role in {"human", "user"}:
            role = "user"
        elif source_role in {"gpt", "assistant"}:
            role = "assistant"
        else:
            raise ValueError(f"Unsupported conversation role {source_role!r}")
        text = str(turn.get("value", "")).replace("<image>", "").strip()
        content: list[dict[str, str]] = []
        if role == "user" and not image_inserted:
            content.append({"type": "image"})
            image_inserted = True
        content.append({"type": "text", "text": text})
        messages.append({"role": role, "content": content})
    if not messages or messages[0]["role"] != "user" or not image_inserted:
        raise ValueError("Training conversations must begin with one image-bearing user turn")
    if not any(message["role"] == "assistant" for message in messages):
        raise ValueError("Training conversation contains no assistant answer")
    return messages


def assistant_only_labels(input_ids: Sequence[int], tokenizer: Any) -> list[int]:
    """Mask all rows except Llama-3 assistant contents and their EOT token."""

    prefix = _token_ids(
        tokenizer,
        "<|start_header_id|>assistant<|end_header_id|>",
    )
    eot = int(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    if not prefix or eot < 0:
        raise ValueError("Tokenizer is missing the Llama-3 assistant/EOT controls")
    ids = [int(value) for value in input_ids]
    labels = [IGNORE_INDEX] * len(ids)
    found = 0
    cursor = 0
    while cursor <= len(ids) - len(prefix):
        if ids[cursor : cursor + len(prefix)] != prefix:
            cursor += 1
            continue
        start = cursor + len(prefix)
        # The multimodal template currently renders three newlines here while
        # a text-only message renders two. Treat either whitespace encoding as
        # header formatting rather than an answer target.
        while start < len(ids) and tokenizer.decode([ids[start]]).isspace():
            start += 1
        try:
            end = ids.index(eot, start)
        except ValueError as error:
            raise ValueError("Assistant turn is missing its <|eot_id|> terminator") from error
        labels[start : end + 1] = ids[start : end + 1]
        found += 1
        cursor = end + 1
    if found == 0:
        raise ValueError("Tokenized prompt contains no assistant span")
    return labels


def processor_native_token_counts(
    processor: Any,
    image_size: torch.Tensor | Sequence[int],
) -> tuple[int, int]:
    """Return native content and structural counts using HF processor geometry.

    This mirrors ``LlavaNextProcessor._get_number_of_features`` and its
    ``default``-strategy CLS subtraction. The count is therefore captured at
    the native assembled/unpadded boundary without running the vision tower.
    """

    from transformers.image_processing_utils import select_best_resolution

    values = image_size.tolist() if isinstance(image_size, torch.Tensor) else image_size
    original_height, original_width = (int(values[0]), int(values[1]))
    crop_size = processor.image_processor.crop_size
    processed_height = int(crop_size["height"])
    processed_width = int(crop_size["width"])
    patch_size = int(processor.patch_size)
    best_height, best_width = select_best_resolution(
        [original_height, original_width],
        processor.image_processor.image_grid_pinpoints,
    )
    scale_height = best_height // processed_height
    scale_width = best_width // processed_width
    _, structural = processor._get_unpadded_features(
        original_height,
        original_width,
        processed_height // patch_size,
        processed_width // patch_size,
        scale_height,
        scale_width,
    )
    native_with_cls = processor._get_number_of_features(
        original_height,
        original_width,
        processed_height,
        processed_width,
    )
    content = int(native_with_cls) - 1 - int(structural)
    if content < 1 or structural < 1:
        raise ValueError(
            f"Invalid native LLaVA-NeXT geometry: content={content}, structural={structural}"
        )
    return content, int(structural)


def build_llava_next_training_example(
    sample: Mapping[str, Any],
    processor: Any,
    *,
    original_content_tokens: int,
    structural_tokens: int,
    retain_ratio: float,
    model_max_length: int,
) -> tuple[list[int], list[int]]:
    """Build compact IDs and labels before the native Llama-3 prefill."""

    messages = llava_next_messages(sample["conversations"])
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # Match LlavaNextProcessor.__call__: its tokenizer default prepends the
    # Llama-3 BOS token to this rendered template.
    input_ids = _token_ids(
        processor.tokenizer,
        prompt,
        add_special_tokens=True,
    )
    labels = assistant_only_labels(input_ids, processor.tokenizer)
    image_token_id = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
    positions = [index for index, token in enumerate(input_ids) if token == image_token_id]
    if len(positions) != 1:
        raise ValueError(f"Expected one <image> token before expansion, got {len(positions)}")

    target_content_tokens = retention_target(original_content_tokens, retain_ratio)
    visual_count = target_content_tokens + int(structural_tokens)
    position = positions[0]
    input_ids = input_ids[:position] + [image_token_id] * visual_count + input_ids[position + 1 :]
    labels = labels[:position] + [IGNORE_INDEX] * visual_count + labels[position + 1 :]
    input_ids = input_ids[: int(model_max_length)]
    labels = labels[: int(model_max_length)]
    if input_ids.count(image_token_id) != visual_count:
        raise ValueError(
            f"model_max_length={model_max_length} truncated the {visual_count}-token visual span"
        )
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("model_max_length removed every assistant training target")
    return input_ids, labels


class LlavaNextParquetDataset(Dataset):
    """Row-group-local, zero-conversion reader for the supplied dataset."""

    columns = ("id", "conversations", "data_source", "image")

    def __init__(
        self,
        root: Path,
        *,
        processor: Any,
        retain_ratio: float,
        model_max_length: int,
        subset_frac: float,
        seed: int,
        limit: int | None,
    ) -> None:
        self.root = Path(root)
        self.processor = processor
        self.retain_ratio = float(retain_ratio)
        self.model_max_length = int(model_max_length)
        self.shards = sorted((self.root / "data").glob("*.parquet"))
        if not self.shards:
            raise FileNotFoundError(f"No Parquet shards under {self.root / 'data'}")

        groups, self.total_rows, self.image_rows = _scan_image_groups(self.shards)
        self._image_group_offsets = [0]
        for group in groups:
            self._image_group_offsets.append(self._image_group_offsets[-1] + group.image_count)
        if self._image_group_offsets[-1] != self.image_rows:
            raise RuntimeError("LLaVA-NeXT image-row index is internally inconsistent")
        if not 0.0 < float(subset_frac) <= 1.0:
            raise ValueError("subset_frac must be in (0, 1]")

        rng = Random(int(seed))
        count = self.image_rows
        if float(subset_frac) < 1.0:
            count = max(1, int(round(self.image_rows * float(subset_frac))))
            selected = rng.sample(range(self.image_rows), count)
        else:
            selected = list(range(self.image_rows))
        if limit is not None:
            if int(limit) < 1:
                raise ValueError("dataset limit must be positive")
            selected = selected[: int(limit)]

        # Shuffle row-group order, but keep selected rows from a group adjacent.
        # This retains seeded sampling while avoiding ~74K random row-group reads.
        grouped: dict[tuple[int, int], list[_RowLocation]] = defaultdict(list)
        for image_row in selected:
            group_position = bisect_right(self._image_group_offsets, image_row) - 1
            group = groups[group_position]
            row_in_group = image_row - self._image_group_offsets[group_position]
            row_index = (
                row_in_group if group.row_indices is None else group.row_indices[row_in_group]
            )
            grouped[(group.shard, group.row_group)].append(
                _RowLocation(group.shard, group.row_group, row_index)
            )
        keys = list(grouped)
        rng.shuffle(keys)
        self.locations: list[_RowLocation] = []
        for key in keys:
            rows = grouped[key]
            rng.shuffle(rows)
            self.locations.extend(rows)
        if not self.locations:
            raise ValueError("No LLaVA-NeXT training rows were selected")

        self._cached_key: tuple[int, int] | None = None
        self._cached_table: Any | None = None
        self._cached_parquet_shard: int | None = None
        self._cached_parquet: Any | None = None

    def __len__(self) -> int:
        return len(self.locations)

    def _row(self, location: _RowLocation) -> dict[str, Any]:
        key = (location.shard, location.row_group)
        if self._cached_key != key:
            if self._cached_parquet_shard != location.shard:
                self._cached_parquet = pq.ParquetFile(self.shards[location.shard])
                self._cached_parquet_shard = location.shard
            self._cached_table = self._cached_parquet.read_row_group(
                location.row_group,
                columns=list(self.columns),
            )
            self._cached_key = key
        return {name: self._cached_table[name][location.row].as_py() for name in self.columns}

    def _image(self, value: Any) -> Image.Image:
        if not isinstance(value, Mapping):
            raise ValueError("LLaVA-NeXT Parquet image column is not a mapping")
        raw = value.get("bytes")
        if raw:
            with Image.open(BytesIO(raw)) as image:
                return image.convert("RGB")
        path = value.get("path")
        if path:
            candidate = Path(str(path))
            for resolved in (candidate, self.root / candidate):
                if resolved.is_file():
                    with Image.open(resolved) as image:
                        return image.convert("RGB")
        raise FileNotFoundError("LLaVA-NeXT row has neither embedded image bytes nor a valid path")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self._row(self.locations[index])
        image_inputs = self.processor.image_processor(
            images=[self._image(sample["image"])],
            return_tensors="pt",
        )
        pixel_values = image_inputs.get("pixel_values")
        image_sizes = image_inputs.get("image_sizes")
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 5:
            raise ValueError(
                f"LLaVA-NeXT image processor must return [1, views, C, H, W], got "
                f"{getattr(pixel_values, 'shape', None)}"
            )
        if not isinstance(image_sizes, torch.Tensor) or image_sizes.shape != (1, 2):
            raise ValueError("LLaVA-NeXT image processor must return image_sizes [1, 2]")
        view_count = int(pixel_values.shape[1])
        original_content_tokens, structural_tokens = processor_native_token_counts(
            self.processor, image_sizes[0]
        )
        target_content_tokens = retention_target(original_content_tokens, self.retain_ratio)
        input_ids, labels = build_llava_next_training_example(
            sample,
            self.processor,
            original_content_tokens=original_content_tokens,
            structural_tokens=structural_tokens,
            retain_ratio=self.retain_ratio,
            model_max_length=self.model_max_length,
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values[0],
            "image_sizes": image_sizes[0],
            "view_count": torch.tensor(view_count, dtype=torch.long),
            "original_content_tokens": torch.tensor(original_content_tokens, dtype=torch.long),
            "target_content_tokens": torch.tensor(target_content_tokens, dtype=torch.long),
            "structural_tokens": torch.tensor(structural_tokens, dtype=torch.long),
        }


class LlavaNextVisionZipCollator:
    """Fail-closed single-row collator with compact Llama-3 positions."""

    def __init__(self, *, image_token_id: int, retain_ratio: float) -> None:
        self.image_token_id = int(image_token_id)
        self.retain_ratio = float(retain_ratio)

    def __call__(self, examples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if len(examples) != 1:
            raise NotImplementedError(
                "VisionZip++/LLaVA-NeXT requires per_device_train_batch_size=1"
            )
        example = examples[0]
        ids = example["input_ids"]
        labels = example["labels"]
        view_count = int(example["view_count"].item())
        original = int(example["original_content_tokens"].item())
        target = int(example["target_content_tokens"].item())
        structural = int(example["structural_tokens"].item())
        expected_target = retention_target(original, self.retain_ratio)
        if target != expected_target:
            raise ValueError(
                f"Training row target {target} disagrees with ratio-derived {expected_target}"
            )
        expected = target + structural
        actual = int((ids == self.image_token_id).sum())
        if actual != expected:
            raise ValueError(
                f"Training row has {actual} placeholders; expected {target} content + "
                f"{structural} structural tokens"
            )
        if int(example["pixel_values"].shape[0]) != view_count:
            raise ValueError("AnyRes view metadata and pixel tensor disagree")
        attention_mask = torch.ones_like(ids, dtype=torch.long)
        position_ids = torch.arange(ids.shape[0], dtype=torch.long)
        return {
            "input_ids": ids.unsqueeze(0),
            "labels": labels.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "position_ids": position_ids.unsqueeze(0),
            "pixel_values": example["pixel_values"].unsqueeze(0),
            "image_sizes": example["image_sizes"].unsqueeze(0),
        }


def freeze_for_projector_tuning(model: nn.Module) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projector = model.model.multi_modal_projector
    projector.requires_grad_(True)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    projector_count = sum(parameter.numel() for parameter in projector.parameters())
    if trainable != projector_count or trainable == 0:
        raise RuntimeError(
            f"Trainable parameter count {trainable} != projector count {projector_count}"
        )
    return projector


def _training_options(config: Config) -> dict[str, Any]:
    options = config.custom_args.get("training_args", {})
    if not isinstance(options, Mapping):
        raise TypeError("training_args must be a mapping")
    return dict(options)


def require_training_attention_backend(value: Any) -> str:
    """Return the sole supported training backend or fail before model loading."""

    backend = str(value)
    if backend != TRAINING_ATTN_IMPLEMENTATION:
        raise ValueError(
            "LLaVA-NeXT VisionZip++ training is FlashAttention-2 only; "
            f"expected attn_implementation={TRAINING_ATTN_IMPLEMENTATION!r}, "
            f"got {backend!r}. Inference remains on SDPA."
        )
    return backend


def require_flash_attention_runtime() -> None:
    """Validate the exact compiled FA2 package used by the Conrad recipe."""

    try:
        installed = package_version("flash-attn")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "VisionZip++ training requires the compiled flash-attn wheel "
            f"at version {FLASH_ATTN_VERSION}."
        ) from error
    if installed != FLASH_ATTN_VERSION:
        raise RuntimeError(
            f"VisionZip++ training requires flash-attn {FLASH_ATTN_VERSION}, got {installed}."
        )
    try:
        __import__("flash_attn_2_cuda")
    except ImportError as error:
        raise RuntimeError(
            "flash-attn is installed but its compiled FA2 CUDA extension cannot be imported"
        ) from error


def require_effective_flash_attention(model: nn.Module) -> None:
    """Fail if Transformers did not select FA2 for both train-time towers."""

    inner = getattr(model, "model", None)
    modules = {
        "language model": getattr(inner, "language_model", None),
        "vision tower": getattr(inner, "vision_tower", None),
    }
    mismatches: list[str] = []
    for label, module in modules.items():
        config = getattr(module, "config", None)
        effective = getattr(config, "_attn_implementation", None)
        if effective != TRAINING_ATTN_IMPLEMENTATION:
            mismatches.append(f"{label}={effective!r}")
    if mismatches:
        raise RuntimeError(
            "VisionZip++ training requested FlashAttention-2, but Transformers "
            "did not activate it for every tower: " + ", ".join(mismatches)
        )


def _dataset_manifest(root: Path) -> tuple[int, int, int]:
    shards = sorted((root / "data").glob("*.parquet"))
    _, total_rows, image_rows = _scan_image_groups(shards)
    return len(shards), total_rows, image_rows


def build_training_args(
    backbone: VLM,
    config: Config,
    compressor: VisionZipPlusPlusTrainingCompressor,
    dataset_dir: Path,
) -> VisionZipNextTrainingArgs:
    options = _training_options(config)
    if options.get("stage", "efficient_tuning") != "efficient_tuning":
        raise ValueError("VisionZip++ has one projector-only efficient_tuning stage")
    if str(backbone._model_id) != TARGET_MODEL_ID:
        raise ValueError(
            f"VisionZip++ training requires {TARGET_MODEL_ID}, got {backbone._model_id!r}"
        )
    revision = str(getattr(backbone, "_model_revision", TARGET_MODEL_REVISION))
    if revision != TARGET_MODEL_REVISION:
        raise ValueError(
            f"VisionZip++ training is pinned to revision {TARGET_MODEL_REVISION}, got {revision}"
        )

    world_size = max(1, int(environ.get("WORLD_SIZE", "1")))
    global_batch = int(
        environ.get(
            "GLOBAL_BATCH_SIZE",
            options.get("global_batch_size", DEFAULT_GLOBAL_BATCH_SIZE),
        )
    )
    per_device = int(
        environ.get(
            "PER_DEVICE_TRAIN_BATCH_SIZE",
            options.get("per_device_train_batch_size", options.get("batch_size", 1)),
        )
    )
    if per_device != 1:
        raise ValueError("LLaVA-NeXT VisionZip++ requires per_device_train_batch_size=1")
    if global_batch < 1:
        raise ValueError("global_batch_size must be positive")
    accumulation_value = environ.get(
        "GRADIENT_ACCUMULATION_STEPS",
        options.get("gradient_accumulation_steps"),
    )
    if accumulation_value is None:
        if global_batch % (world_size * per_device):
            raise ValueError("global_batch_size is not divisible by WORLD_SIZE x per-device batch")
        accumulation = global_batch // (world_size * per_device)
    else:
        accumulation = int(accumulation_value)
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if world_size * per_device * accumulation != global_batch:
        raise ValueError("Configured accumulation does not realize global_batch_size")

    subset_frac = float(options.get("subset_frac", 0.1))
    warmup_ratio = float(options.get("warmup_ratio", 0.03))
    if not 0.0 < subset_frac <= 1.0:
        raise ValueError("subset_frac must be in (0, 1]")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    bf16, fp16 = bool(options.get("bf16", True)), bool(options.get("fp16", False))
    if bf16 and fp16:
        raise ValueError("Choose at most one of bf16 and fp16")
    deepspeed_stage = int(options.get("deepspeed_stage", 0))
    if deepspeed_stage != 0:
        raise ValueError("VisionZip++ supports replicated DDP, not DeepSpeed/FSDP")

    dataset_limit = config.dataset_limit
    configured_limit = options.get("dataset_limit", options.get("max_samples"))
    if configured_limit is not None:
        dataset_limit = (
            int(configured_limit)
            if dataset_limit is None
            else min(int(dataset_limit), int(configured_limit))
        )
    if dataset_limit is not None and int(dataset_limit) < 1:
        raise ValueError("dataset_limit must be positive")
    shards, total_rows, image_rows = _dataset_manifest(dataset_dir)
    output_dir = Path(
        options.get("output_dir") or Path(config.output_dir) / "vision_zip_pp_llava_next"
    )
    checkpoint_name = str(
        options.get(
            "checkpoint_name",
            f"vision_zip_pp_llava_next_r{float(compressor._retain_ratio):g}.pt",
        )
    )
    if Path(checkpoint_name).name != checkpoint_name or not checkpoint_name.endswith(".pt"):
        raise ValueError("checkpoint_name must be a .pt filename")
    stop_after = environ.get("STOP_AFTER_STEPS", options.get("stop_after_steps"))
    if stop_after is not None and int(stop_after) < 1:
        raise ValueError("stop_after_steps must be positive")
    resume = options.get("resume_from_checkpoint") or environ.get("RESUME_FROM_CHECKPOINT")
    attention_backend = require_training_attention_backend(
        options.get("attn_implementation", TRAINING_ATTN_IMPLEMENTATION)
    )
    flash_attn_version = str(
        environ.get(
            "FLASH_ATTN_VERSION",
            options.get("flash_attn_version", FLASH_ATTN_VERSION),
        )
    )
    if flash_attn_version != FLASH_ATTN_VERSION:
        raise ValueError(
            f"VisionZip++ training requires flash-attn {FLASH_ATTN_VERSION}, "
            f"got configured version {flash_attn_version}"
        )
    flash_attn_wheel = environ.get("FLASH_ATTN_WHEEL", options.get("flash_attn_wheel"))

    resolved = VisionZipNextTrainingArgs(
        model_name_or_path=str(backbone._model_id),
        model_revision=revision,
        data_path=str(dataset_dir),
        output_dir=str(output_dir),
        checkpoint_name=checkpoint_name,
        initial_checkpoint_path=(
            None if compressor._checkpoint_arg is None else str(compressor._checkpoint_arg)
        ),
        retain_ratio=float(compressor._retain_ratio),
        num_train_epochs=float(options.get("num_train_epochs", options.get("num_epochs", 1))),
        global_batch_size=global_batch,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accumulation,
        learning_rate=float(options.get("learning_rate", 2e-5)),
        weight_decay=float(options.get("weight_decay", 0.0)),
        max_grad_norm=float(options.get("max_grad_norm", 1.0)),
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=str(options.get("lr_scheduler_type", "cosine")),
        model_max_length=int(options.get("model_max_length", options.get("max_length", 2048))),
        subset_frac=subset_frac,
        dataset_limit=None if dataset_limit is None else int(dataset_limit),
        dataset_total_rows=total_rows,
        dataset_image_rows=image_rows,
        dataset_shards=shards,
        dataloader_num_workers=int(
            environ.get(
                "DATALOADER_NUM_WORKERS",
                options.get("dataloader_num_workers", options.get("num_workers", 2)),
            )
        ),
        bf16=bf16,
        fp16=fp16,
        tf32=bool(options.get("tf32", True)),
        attn_implementation=attention_backend,
        flash_attn_version=flash_attn_version,
        flash_attn_wheel=flash_attn_wheel,
        gradient_checkpointing=bool(options.get("gradient_checkpointing", True)),
        deepspeed_stage=deepspeed_stage,
        max_steps=int(options.get("max_steps", -1)),
        stop_after_steps=None if stop_after is None else int(stop_after),
        save_steps=int(environ.get("SAVE_STEPS", options.get("save_steps", 100))),
        save_total_limit=int(environ.get("SAVE_TOTAL_LIMIT", options.get("save_total_limit", 2))),
        logging_steps=int(options.get("logging_steps", 10)),
        seed=int(options.get("seed", 42)),
        resume_from_checkpoint=None if not resume else str(resume),
        skip_final_save=bool(options.get("skip_final_save", False)),
    )
    if resolved.num_train_epochs <= 0:
        raise ValueError("num_train_epochs must be positive")
    if resolved.learning_rate <= 0 or resolved.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if resolved.max_grad_norm < 0:
        raise ValueError("max_grad_norm must be non-negative")
    if not 1 <= resolved.model_max_length <= 8192:
        raise ValueError("model_max_length must be in [1, 8192] for the pinned Llama-3 model")
    # The largest configured native span is 2,880 content + 48 structural
    # tokens. The exact sample-specific check remains in the dataset builder.
    if resolved.model_max_length <= retention_target(2880, resolved.retain_ratio) + 48:
        raise ValueError(
            "model_max_length must leave text capacity after the largest five-view visual span"
        )
    if resolved.dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be non-negative")
    if resolved.max_steps == 0 or resolved.max_steps < -1:
        raise ValueError("max_steps must be -1 or positive")
    if resolved.save_steps < 0 or resolved.save_total_limit < 1:
        raise ValueError("save_steps must be non-negative and save_total_limit positive")
    if resolved.logging_steps < 1:
        raise ValueError("logging_steps must be positive")
    return resolved


def build_hf_training_arguments(args: VisionZipNextTrainingArgs) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        train_sampling_strategy="sequential",
        max_steps=args.max_steps,
        save_strategy="no" if args.save_steps == 0 else "steps",
        save_steps=max(1, args.save_steps),
        save_total_limit=args.save_total_limit,
        save_only_model=False,
        logging_steps=args.logging_steps,
        report_to="none",
        average_tokens_across_devices=False,
        ddp_find_unused_parameters=False,
        deepspeed=None,
    )


def projectneura_warmup_steps(warmup_ratio: float, training_steps: int) -> int:
    if training_steps < 0:
        raise ValueError("training_steps must be non-negative")
    return int(float(warmup_ratio) * int(training_steps))


def _schedule(args: VisionZipNextTrainingArgs) -> dict[str, Any]:
    return {
        "stage": "efficient_tuning",
        "dataset": "LLaVA-NeXT-Data image-bearing train split",
        "dataset_path": args.data_path,
        "dataset_total_rows": args.dataset_total_rows,
        "dataset_image_rows": args.dataset_image_rows,
        "dataset_shards": args.dataset_shards,
        "dataset_access": "in_place_parquet_row_groups_no_conversion",
        "subset_frac": args.subset_frac,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "warmup_rounding": "floor_int_like_projectneura",
        "lr_scheduler_type": args.lr_scheduler_type,
        "global_batch_size": args.global_batch_size,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "model_max_length": args.model_max_length,
        "dataset_limit": args.dataset_limit,
        "max_steps": args.max_steps,
        "stop_after_steps": args.stop_after_steps,
        "precision": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
        "attn_implementation": args.attn_implementation,
        "flash_attn_version": args.flash_attn_version,
        "flash_attn_wheel": args.flash_attn_wheel,
        "gradient_checkpointing": args.gradient_checkpointing,
        "deepspeed_stage": args.deepspeed_stage,
        "initial_checkpoint_path": args.initial_checkpoint_path,
        "seed": args.seed,
    }


def _is_full_reference_recipe(args: VisionZipNextTrainingArgs) -> bool:
    schedule = _schedule(args)
    return all(schedule.get(key) == value for key, value in FINAL_TRAINING_RECIPE.items())


def _save_projector_artifact(
    path: Path,
    state_dict: Mapping[str, torch.Tensor],
    *,
    model: nn.Module,
    args: VisionZipNextTrainingArgs,
    compressor: VisionZipPlusPlusTrainingCompressor,
    artifact_role: str,
    training_complete: bool,
    global_step: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = checkpoint_metadata(
        model_id=args.model_name_or_path,
        model_revision=args.model_revision,
        retain_ratio=float(compressor._retain_ratio),
        image_grid_pinpoints=model.config.image_grid_pinpoints,
        training_schedule=_schedule(args),
        artifact_role=artifact_role,
        training_complete=training_complete,
        global_step=global_step,
    )
    payload = {
        "format_version": metadata["format_version"],
        "compressor": "VisionZipPlusPlusCompressor",
        "config": metadata,
        "training_args": asdict(args),
        "state_dict": {
            key: value.detach().to(torch.float32).cpu().contiguous()
            for key, value in state_dict.items()
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    sidecar_temporary = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_temporary.write_text(
        dumps({"config": metadata, "training_args": asdict(args)}, indent=2) + "\n",
        encoding="utf-8",
    )
    sidecar_temporary.replace(sidecar)


class _StopAfterStepCallback(TrainerCallback):
    def __init__(self, stop_after_steps: int | None) -> None:
        self.stop_after_steps = stop_after_steps

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if self.stop_after_steps is not None and int(state.global_step) >= self.stop_after_steps:
            control.should_training_stop = True
        return control


class VisionZipNextTrainer(Trainer):
    """Save projector-only artifacts while Trainer keeps optimizer/RNG state."""

    def __init__(
        self,
        *args: Any,
        resolved_args: VisionZipNextTrainingArgs,
        compressor: VisionZipPlusPlusTrainingCompressor,
        **kwargs: Any,
    ) -> None:
        self.resolved_args = resolved_args
        self.compressor = compressor
        super().__init__(*args, **kwargs)
        if self.is_deepspeed_enabled or self.is_fsdp_enabled or self.is_fsdp_xla_enabled:
            raise ValueError("VisionZipNextTrainer supports replicated single-GPU/DDP only")
        self.model_accepts_loss_kwargs = False

    def _base_model(self) -> nn.Module:
        return self.accelerator.unwrap_model(self.model, keep_torch_compile=False)

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        if self.lr_scheduler is None:
            optimizer = self.optimizer if optimizer is None else optimizer
            self.lr_scheduler = get_scheduler(
                self.resolved_args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=projectneura_warmup_steps(
                    self.resolved_args.warmup_ratio,
                    num_training_steps,
                ),
                num_training_steps=int(num_training_steps),
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def gather_projector_state(self) -> dict[str, torch.Tensor]:
        if not self.args.should_save:
            return {}
        projector = self._base_model().model.multi_modal_projector
        return {
            key: value.detach().to(torch.float32).cpu().contiguous()
            for key, value in projector.state_dict().items()
        }

    def _save(
        self,
        output_dir: str | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        del state_dict
        destination = Path(output_dir or self.args.output_dir)
        state = self.gather_projector_state()
        if self.args.should_save:
            _save_projector_artifact(
                destination / CHECKPOINT_FILENAME,
                state,
                model=self._base_model(),
                args=self.resolved_args,
                compressor=self.compressor,
                artifact_role="periodic_resume",
                training_complete=False,
                global_step=int(self.state.global_step),
            )
            torch.save(self.args, destination / "training_args.bin")

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        del _internal_call
        self._save(output_dir)

    def _load_from_checkpoint(
        self,
        resume_from_checkpoint: str,
        model: nn.Module | None = None,
    ) -> None:
        base = (
            self._base_model()
            if model is None
            else self.accelerator.unwrap_model(
                model,
                keep_torch_compile=False,
            )
        )
        load_projector_checkpoint(
            Path(resume_from_checkpoint) / CHECKPOINT_FILENAME,
            base.model.multi_modal_projector,
            retain_ratio=float(self.compressor._retain_ratio),
            model_id=self.resolved_args.model_name_or_path,
            model_revision=self.resolved_args.model_revision,
            image_grid_pinpoints=base.config.image_grid_pinpoints,
            allow_incomplete=True,
            expected_training_schedule=_schedule(self.resolved_args),
        )


def finetune_vision_zip_llava_next(backbone: VLM, config: Config) -> None:
    """Run the complete projector-only VisionZip++ recipe."""

    dataset_dir = _resolve_dataset_dir(config.input_dir)
    if dataset_dir is None:
        raise FileNotFoundError(f"Missing {DATASET_DIR_NAME} under {config.input_dir}")
    compressor_args = config.custom_args.get("compressor_args", {})
    if not isinstance(compressor_args, Mapping):
        raise TypeError("compressor_args must be a mapping")
    compressor = VisionZipPlusPlusTrainingCompressor(**dict(compressor_args))
    args = build_training_args(backbone, config, compressor, dataset_dir)

    local_rank = int(environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        backbone._device = f"cuda:{local_rank}"
    backbone._dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    backbone._attn_implementation = args.attn_implementation
    backbone._model_revision = args.model_revision
    set_seed(args.seed)
    require_flash_attention_runtime()

    model = backbone.model
    require_effective_flash_attention(model)
    processor = backbone.processor
    tokenizer = processor.tokenizer
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_padding_side = tokenizer.padding_side
    original_model_max_length = tokenizer.model_max_length
    original_training = model.training
    original_use_cache = getattr(model.config, "use_cache", None)
    original_text_use_cache = getattr(model.config.text_config, "use_cache", None)
    original_gradient_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    input_grads_already_enabled = bool(getattr(model, "_require_grads_hooks", ()))
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]

    compressor.to(device=backbone._device, dtype=backbone._dtype)
    try:
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = args.model_max_length
        compressor.install(backbone)
        projector = freeze_for_projector_tuning(model)
        projector.to(torch.float32)
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        model.train()
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            vision_tower = model.model.vision_tower
            if hasattr(vision_tower, "gradient_checkpointing_disable"):
                vision_tower.gradient_checkpointing_disable()
            elif hasattr(vision_tower, "gradient_checkpointing"):
                vision_tower.gradient_checkpointing = False
            if not input_grads_already_enabled and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        dataset = LlavaNextParquetDataset(
            dataset_dir,
            processor=processor,
            retain_ratio=args.retain_ratio,
            model_max_length=args.model_max_length,
            subset_frac=args.subset_frac,
            seed=args.seed,
            limit=args.dataset_limit,
        )
        collator = LlavaNextVisionZipCollator(
            image_token_id=int(model.config.image_token_id),
            retain_ratio=args.retain_ratio,
        )
        trainer = VisionZipNextTrainer(
            model=model,
            args=build_hf_training_arguments(args),
            train_dataset=dataset,
            data_collator=collator,
            callbacks=[_StopAfterStepCallback(args.stop_after_steps)],
            resolved_args=args,
            compressor=compressor,
        )
        if int(environ.get("RANK", "0")) == 0:
            trainable = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            print(
                "[VisionZip++/LLaVA-NeXT] projector-only Efficient Tuning "
                f"trainable={trainable:,}, samples={len(dataset):,}, "
                f"retain_ratio={args.retain_ratio:g}, global_batch={args.global_batch_size}",
                flush=True,
            )
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        if not args.skip_final_save:
            state = trainer.gather_projector_state()
            if trainer.is_world_process_zero():
                completed_epoch = float(trainer.state.epoch or 0.0)
                training_complete = (
                    _is_full_reference_recipe(args)
                    and completed_epoch + 1e-6 >= args.num_train_epochs
                )
                destination = Path(args.output_dir) / args.checkpoint_name
                _save_projector_artifact(
                    destination,
                    state,
                    model=model,
                    args=args,
                    compressor=compressor,
                    artifact_role="final_evaluation" if training_complete else "incomplete_run",
                    training_complete=training_complete,
                    global_step=int(trainer.state.global_step),
                )
                suffix = (
                    "ready for compressor_args.checkpoint_path"
                    if training_complete
                    else "resume/debug only; evaluation loading will reject it"
                )
                print(f"Saved {destination} ({suffix})", flush=True)
    finally:
        compressor.uninstall()
        if args.gradient_checkpointing:
            if not input_grads_already_enabled and hasattr(model, "disable_input_require_grads"):
                model.disable_input_require_grads()
            if original_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            elif not original_gradient_checkpointing and hasattr(
                model, "gradient_checkpointing_disable"
            ):
                model.gradient_checkpointing_disable()
        for parameter, requires_grad in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(requires_grad)
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
        if original_text_use_cache is not None:
            model.config.text_config.use_cache = original_text_use_cache
        tokenizer.padding_side = original_padding_side
        tokenizer.model_max_length = original_model_max_length
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        model.train(original_training)
