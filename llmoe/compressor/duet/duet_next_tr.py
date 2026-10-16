"""Compression-aware DUET training for the pinned LLaVA-NeXT-8B target.

The released DUET checkpoint was trained with LLaVA-1.5; Appendix Table 13's
LLaVA-NeXT results are training-free.  This module is therefore an explicit
architecture adaptation, not a claim that AMD released NeXT training weights.
It preserves the trained ProjectNeura recipe where it is architecture-neutral:
projector alignment on 558K, followed by language-model + projector instruction
tuning under the active DUET schedule, with the CLIP tower frozen.  Finetuning
reads ``LLaVA-NeXT-Data`` Parquet shards directly and never unpacks the embedded
images.
"""

from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from io import BytesIO
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    TrainingArguments,
    set_seed,
)

from llmoe.backbone.llava_next import (
    LLAVA_NEXT_8B_MODEL_ID,
    LLAVA_NEXT_8B_REVISION,
)
from llmoe.backbone.prototype import VLM
from llmoe.compressor.duet.duet import DUETCompressor, average_image_tokens
from llmoe.compressor.duet.duet_llava_next import (
    DUET_NEXT_BUDGET_SEMANTICS,
    DUET_NEXT_CONTENT_DEFINITION,
    DUET_NEXT_FLASH_ATTN_VERSION,
    DUET_NEXT_FEATURE_ORDER,
    DUET_NEXT_IMPLEMENTATION_VERSION,
    DUET_NEXT_INFERENCE_ATTENTION_BACKEND,
    DUET_NEXT_ROUNDING_RULE,
    DUET_NEXT_SOURCE_COMMITS,
    DUET_NEXT_STRUCTURAL_POLICY,
    DUET_NEXT_TRAINING_ATTENTION_BACKEND,
    DUET_NEXT_VIEW_POLICY,
    next_budget_for_geometry,
    retention_target,
)
from llmoe.compressor.duet.duet_tr import (
    CROSS_DEVICE_TOKEN_AVERAGING,
    IGNORE_INDEX,
    DuetTrainer,
    DuetTrainingLlavaForConditionalGeneration,
    LlavaJsonDataset,
    ResourceReportCallback,
    SaveControlCallback,
    StopAfterStepCallback,
    _TrainingBackbone,
    _deepspeed_config,
    _gather_final_state_dict,
    _mapping_option,
    _report_trainable_parameters,
    load_pretrained_projector,
    write_duet_metadata,
)
from llmoe.compressor.llava_next import (
    geometry_for_image,
    trim_image_placeholders_to_lengths,
)
from llmoe.config import Config



LLAVA_NEXT_8B_IMAGE_GRID_PINPOINTS = (
    (336, 672),
    (672, 336),
    (672, 672),
    (1008, 336),
    (336, 1008),
)
LLAVA_NEXT_8B_VISION_GEOMETRY = {
    "image_size": 336,
    "patch_size": 14,
    "patches_per_view": 576,
}


NEXT_MODEL_MAX_LENGTH = 4096
NEXT_STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "pretrain": {
        "learning_rate": 1e-3,
        "global_batch_size": 256,
        "deepspeed_stage": 2,
        "save_steps": 24000,
    },
    "finetune": {
        "learning_rate": 2e-5,
        "global_batch_size": 128,
        "deepspeed_stage": 3,
        "save_steps": 1000,
    },
}


def require_flash_attention_2_training_runtime() -> str:
    """Fail before model loading unless the pinned compiled FA2 runtime is usable.

    Inference deliberately stays on Transformers SDPA and never calls this
    function. Training is a separate loader and has no SDPA/eager fallback:
    accepting a fallback would change both the memory profile and the attention
    implementation used to produce the checkpoint.
    """

    try:
        installed_version = distribution_version("flash-attn")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "DUET/LLaVA-NeXT training requires flash-attn=="
            f"{DUET_NEXT_FLASH_ATTN_VERSION}; install the pinned Conrad wheel before "
            "starting training. Inference does not require this package because it uses SDPA."
        ) from error
    if installed_version != DUET_NEXT_FLASH_ATTN_VERSION:
        raise RuntimeError(
            "DUET/LLaVA-NeXT training requires flash-attn=="
            f"{DUET_NEXT_FLASH_ATTN_VERSION}, found {installed_version!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "DUET/LLaVA-NeXT FlashAttention-2 training requires CUDA; CPU training and "
            "automatic SDPA fallback are not supported."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("DUET/LLaVA-NeXT FA2 training requires BF16-capable CUDA GPUs.")
    try:
        flash_attn = __import__("flash_attn")
        __import__("flash_attn_2_cuda")
        from transformers.utils import is_flash_attn_2_available
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "flash-attn metadata is installed, but its compiled FA2 CUDA extension cannot "
            "be imported in the active Varest environment."
        ) from error
    module_version = str(getattr(flash_attn, "__version__", installed_version))
    if module_version != DUET_NEXT_FLASH_ATTN_VERSION:
        raise RuntimeError(
            "flash-attn package/module versions disagree: distribution="
            f"{installed_version!r}, module={module_version!r}."
        )
    if not is_flash_attn_2_available():
        raise RuntimeError(
            "Transformers reports FlashAttention 2 unavailable; DUET training will not "
            "fall back to SDPA or eager attention."
        )
    return installed_version


def find_next_parquet(root: str | Path) -> tuple[Path, ...]:
    root = Path(root)
    files = tuple(sorted(root.rglob("*.parquet"))) if root.is_dir() else (root,)
    files = tuple(path for path in files if path.is_file())
    if not files:
        raise FileNotFoundError(f"No LLaVA-NeXT Parquet shards found under {root}")
    return files


def _image_rows_in_group(parquet_file: Any, group_index: int) -> tuple[int, ...] | None:
    """In-group indices of image-bearing rows, or None when every row has one.

    Same discovery VisionZip++ uses on this corpus: read `image.path`'s null
    statistics, and only for a genuinely mixed row group read that tiny nested
    leaf (about 1 KiB) to find which rows. `image.bytes` is never touched.
    """
    metadata = parquet_file.metadata.row_group(group_index)
    path_column = next(
        (
            metadata.column(index)
            for index in range(metadata.num_columns)
            if metadata.column(index).path_in_schema == "image.path"
        ),
        None,
    )
    if path_column is None:
        return None  # no such column: treat every row as usable, as before
    statistics = path_column.statistics
    if statistics is not None and statistics.null_count is not None:
        non_null = metadata.num_rows - int(statistics.null_count)
        if non_null == metadata.num_rows:
            return None
        if non_null == 0:
            return ()
    values = parquet_file.read_row_group(group_index, columns=["image.path"])["image"].to_pylist()
    return tuple(index for index, value in enumerate(values) if value is not None)


class LlavaNextParquetDataset(Dataset):
    """Lazy, worker-safe access to embedded-image LLaVA-NeXT Parquet shards."""

    def __init__(self, root: str | Path, dataset_limit: int | None = None) -> None:
        import pyarrow.parquet as parquet

        self.root = Path(root)
        self.files = find_next_parquet(self.root)
        # Text-only rows are dropped. DUET prunes visual tokens, so a text-only
        # micro-batch skips the vision modules entirely; under ZeRO-3 that rank
        # never issues their parameter all-gather and the others block until
        # NCCL's watchdog kills the job. All three budgets hung this way at step
        # 25 on the same _ALLGATHER_BASE. HiDrop and VisionZip++ already train
        # image-bearing only on this corpus, so filtering also makes the NeXT
        # baselines comparable. DUET keeps text-only data on LLaVA-1.5, where
        # same-modality micro-batches make it safe; the NeXT trainer sets
        # group_by_modality_length=False, so that guarantee is absent here.
        self._groups: list[tuple[Path, int, int, int, tuple[int, ...] | None]] = []
        cursor = 0
        total_rows = 0
        for path in self.files:
            parquet_file = parquet.ParquetFile(path)
            for group in range(parquet_file.num_row_groups):
                rows = int(parquet_file.metadata.row_group(group).num_rows)
                total_rows += rows
                image_rows = _image_rows_in_group(parquet_file, group)
                usable = rows if image_rows is None else len(image_rows)
                if usable == 0:
                    continue
                self._groups.append((path, group, cursor, cursor + usable, image_rows))
                cursor += usable
        if cursor < total_rows:
            print(
                f"[DUET][DATA] {total_rows - cursor} text-only row(s) of {total_rows} excluded; "
                f"{cursor} image-bearing rows remain",
                flush=True,
            )
        self._ends = [entry[3] for entry in self._groups]
        self._length = cursor if dataset_limit is None else min(cursor, int(dataset_limit))
        if self._length < 1:
            raise ValueError("LLaVA-NeXT dataset_limit selected no rows")
        self._cache_key: tuple[str, int] | None = None
        self._cache_rows: list[dict[str, Any]] | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache_key"] = None
        state["_cache_rows"] = None
        return state

    def __len__(self) -> int:
        return self._length

    @property
    def row_groups(self) -> tuple[tuple[int, ...], ...]:
        groups: list[tuple[int, ...]] = []
        for _path, _group, start, end, _rows in self._groups:
            stop = min(end, self._length)
            if start >= stop:
                break
            groups.append(tuple(range(start, stop)))
        return tuple(groups)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import pyarrow.parquet as parquet

        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        group_index = bisect_right(self._ends, index)
        path, row_group, start, _end, image_rows = self._groups[group_index]
        key = (str(path), row_group)
        if self._cache_key != key or self._cache_rows is None:
            self._cache_rows = parquet.ParquetFile(path).read_row_group(row_group).to_pylist()
            self._cache_key = key
        offset = index - start
        row = self._cache_rows[offset if image_rows is None else image_rows[offset]]
        conversations = LlavaJsonDataset._normalize_conversations(
            row.get("conversations") or []
        )
        image_record = row.get("image")
        image: Image.Image | None = None
        if image_record is not None:
            payload = image_record.get("bytes")
            if payload:
                with Image.open(BytesIO(payload)) as opened:
                    image = opened.convert("RGB")
                    image.load()
            elif image_record.get("path"):
                candidate = Path(str(image_record["path"]))
                if not candidate.is_absolute():
                    candidate = self.root / candidate
                with Image.open(candidate) as opened:
                    image = opened.convert("RGB")
                    image.load()
        return {
            "id": row.get("id"),
            "data_source": row.get("data_source"),
            "image": image,
            "has_image": image is not None,
            "conversations": conversations,
        }


class ParquetRowGroupSampler(Sampler[int]):
    """Shuffle row groups while keeping each worker's expensive reads local."""

    def __init__(self, dataset: LlavaNextParquetDataset, *, seed: int = 42) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        # Trainer/Accelerate shards the returned batch sampler across ranks. Do
        # not shard here as well or a four-GPU job would see only 1/16 of data.
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        groups = list(self.dataset.row_groups)
        order = torch.randperm(len(groups), generator=generator).tolist()
        indices: list[int] = []
        for position in order:
            group = groups[position]
            local = torch.randperm(len(group), generator=generator).tolist()
            indices.extend(group[position] for position in local)
        return iter(indices)


class LlavaNextDuetTrainer(DuetTrainer):
    def _get_train_sampler(self, train_dataset: Any | None = None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if isinstance(dataset, LlavaNextParquetDataset):
            return ParquetRowGroupSampler(dataset, seed=int(self.args.seed))
        return super()._get_train_sampler(dataset)


def _as_chat(conversations: Sequence[Mapping[str, str]], has_image: bool,
             image_token: str) -> list[dict[str, Any]]:
    chat: list[dict[str, Any]] = []
    inserted_image = False
    for turn in conversations:
        role = str(turn["role"])
        value = str(turn["value"])
        marker_count = value.count(image_token)
        if marker_count > 1:
            raise NotImplementedError("DUET/LLaVA-NeXT training supports one source image")
        if marker_count and not has_image:
            raise ValueError("Conversation contains <image> but the Parquet row has no image")
        value = value.replace(image_token, "").strip()
        content: list[dict[str, Any]] = []
        if role == "user" and has_image and (marker_count or not inserted_image):
            content.append({"type": "image"})
            inserted_image = True
        content.append({"type": "text", "text": value})
        chat.append({"role": role, "content": content})
    if has_image != inserted_image:
        raise ValueError("Image data and the rendered conversation disagree")
    return chat


class LlavaNextDuetCollator:
    """Model-native Llama-3 chat tokenization with geometry-exact placeholders."""

    def __init__(self, processor: Any, model_config: Any, compressor: DUETCompressor,
                 *, model_max_length: int = NEXT_MODEL_MAX_LENGTH,
                 stage: str = "finetune") -> None:
        self.processor = processor
        self.model_config = model_config
        self.compressor = compressor
        self.model_max_length = int(model_max_length)
        self.stage = str(stage)
        if self.stage not in NEXT_STAGE_DEFAULTS:
            raise ValueError(f"Unknown DUET/LLaVA-NeXT training stage {stage!r}")
        self.image_token = str(getattr(processor, "image_token", "<image>"))
        self.image_token_id = int(
            processor.tokenizer.convert_tokens_to_ids(self.image_token)
        )
        if self.image_token_id != int(model_config.image_token_index):
            raise ValueError("LLaVA-NeXT processor/model image-token ids disagree")
        self.assistant_header = tuple(
            int(value) for value in processor.tokenizer(
                "<|start_header_id|>assistant<|end_header_id|>",
                add_special_tokens=False,
            )["input_ids"]
        )
        self.eot_id = int(processor.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        self.compressor._llava_next_decoder_layers = int(
            model_config.text_config.num_hidden_layers
        )

    def _assistant_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = [int(value) for value in input_ids.tolist()]
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        cursor = 0
        found = 0
        width = len(self.assistant_header)
        while cursor <= len(ids) - width:
            if tuple(ids[cursor:cursor + width]) != self.assistant_header:
                cursor += 1
                continue
            start = cursor + width
            try:
                stop = ids.index(self.eot_id, start) + 1
            except ValueError:
                stop = len(ids)
            if stop > start:
                labels[start:stop] = input_ids[start:stop]
                found += 1
            cursor = stop
        if not found:
            raise ValueError("No supervised assistant response survived tokenization")
        return labels

    def _plain_pretrain_text(
        self, conversations: Sequence[Mapping[str, str]], has_image: bool
    ) -> str:
        """Released LLaVA projector alignment: ``<image>caption<eot>``."""

        if not has_image:
            raise ValueError("DUET projector pretraining requires an image on every row")
        if len(conversations) != 2 \
                or conversations[0]["role"] != "user" \
                or conversations[1]["role"] != "assistant":
            raise ValueError(
                "DUET projector pretraining expects one user/image turn and one caption"
            )
        if str(conversations[0]["value"]).count(self.image_token) > 1:
            raise NotImplementedError("DUET/LLaVA-NeXT training supports one source image")
        caption = str(conversations[1]["value"]).strip()
        if not caption:
            raise ValueError("DUET projector pretraining caption is empty")
        eos = str(self.processor.tokenizer.eos_token or "")
        return self.image_token + caption + eos

    def _plain_pretrain_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        image_positions = (input_ids == self.image_token_id).nonzero(as_tuple=True)[0]
        if image_positions.numel() == 0:
            raise ValueError("Plain projector sample has no expanded image placeholders")
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        start = int(image_positions[-1]) + 1
        labels[start:] = input_ids[start:]
        if not bool((labels != IGNORE_INDEX).any()):
            raise ValueError("Plain projector sample has no caption tokens")
        return labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(batch) != 1:
            raise NotImplementedError(
                "DUET/LLaVA-NeXT training requires per_device_train_batch_size=1"
            )
        item = batch[0]
        has_image = bool(item["has_image"])
        if self.stage == "pretrain":
            rendered = self._plain_pretrain_text(item["conversations"], has_image)
        else:
            chat = _as_chat(item["conversations"], has_image, self.image_token)
            rendered = self.processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            )
        kwargs: dict[str, Any] = {
            "text": [rendered],
            "return_tensors": "pt",
            "padding": True,
            # Native AnyRes placeholders can be thousands of rows. Truncating
            # before DUET removes them would discard assistant text that fits in
            # the actual compressed sequence, so enforce model_max_length only
            # after the geometry-exact placeholder trim below.
            "truncation": False,
        }
        if has_image:
            kwargs["images"] = [item["image"]]
        inputs = self.processor(**kwargs)
        label_row = (
            self._plain_pretrain_labels(inputs["input_ids"][0])
            if self.stage == "pretrain"
            else self._assistant_labels(inputs["input_ids"][0])
        )
        inputs["labels"] = label_row.unsqueeze(0)
        if has_image:
            image_sizes = inputs.get("image_sizes")
            if not isinstance(image_sizes, torch.Tensor) or tuple(image_sizes.shape) != (1, 2):
                raise RuntimeError("LLaVA-NeXT processor did not return one image size")
            geometry = geometry_for_image(
                SimpleNamespace(config=self.model_config), image_sizes[0]
            )
            budget = next_budget_for_geometry(self.compressor, geometry)
            inputs = trim_image_placeholders_to_lengths(
                inputs,
                token_id=self.image_token_id,
                lengths=(budget.decoder_stage1_tokens,),
                pad_token_id=int(self.processor.tokenizer.pad_token_id),
            )
        sequence_width = int(inputs["input_ids"].shape[1])
        if sequence_width > self.model_max_length:
            for key, value in list(inputs.items()):
                if isinstance(value, torch.Tensor) and value.ndim == 2 \
                        and tuple(value.shape) == (1, sequence_width):
                    inputs[key] = value[:, : self.model_max_length]
            if has_image:
                expected = budget.decoder_stage1_tokens
                actual = int((inputs["input_ids"] == self.image_token_id).sum())
                if actual != expected:
                    raise ValueError(
                        f"model_max_length={self.model_max_length} truncated the compressed "
                        f"image span from {expected} to {actual} rows"
                    )
        if not bool((inputs["labels"] != IGNORE_INDEX).any()):
            raise ValueError("No supervised labels remain after AnyRes placeholder trimming")
        return {key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)}


def load_hf_llava_next_training_base(
    model_name_or_path: str,
    *,
    revision: str,
    dtype: torch.dtype | None,
    attn_implementation: str = DUET_NEXT_TRAINING_ATTENTION_BACKEND,
    model_max_length: int = NEXT_MODEL_MAX_LENGTH,
    local_files_only: bool = True,
) -> tuple[Any, LlavaNextForConditionalGeneration]:
    if attn_implementation != DUET_NEXT_TRAINING_ATTENTION_BACKEND:
        raise ValueError(
            "DUET/LLaVA-NeXT training is FlashAttention-2 only; got "
            f"attn_implementation={attn_implementation!r}. Inference uses the separate "
            "SDPA backbone loader."
        )
    if dtype is not torch.bfloat16:
        raise ValueError(
            "DUET/LLaVA-NeXT FlashAttention-2 training requires dtype=torch.bfloat16."
        )
    require_flash_attention_2_training_runtime()
    common = {
        "revision": revision,
        "local_files_only": bool(local_files_only),
    }
    processor = AutoProcessor.from_pretrained(model_name_or_path, **common)
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.model_max_length = int(model_max_length)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        **common,
    )
    if model.config.model_type != "llava_next" \
            or model.config.text_config.model_type != "llama":
        raise TypeError("Training target must be LLaVA-NeXT with a Llama-3 decoder")
    if int(model.config.image_token_index) != int(
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ):
        raise ValueError("Cached LLaVA-NeXT tokenizer/config image-token mismatch")
    text_backend = getattr(model.model.language_model.config, "_attn_implementation", None)
    vision_backend = getattr(model.model.vision_tower.config, "_attn_implementation", None)
    if text_backend != attn_implementation or vision_backend != attn_implementation:
        raise RuntimeError(
            f"Requested {attn_implementation!r}, but Transformers selected "
            f"text={text_backend!r}, vision={vision_backend!r}"
        )
    model.config.pad_token_id = int(processor.tokenizer.pad_token_id)
    model.config.text_config.pad_token_id = int(processor.tokenizer.pad_token_id)
    return processor, model


@dataclass
class DuetNextTrainingArgs:
    model_name_or_path: str
    model_revision: str
    data_path: str
    image_folder: str | None
    output_dir: str
    stage: str
    budget: int
    pretrain_mm_mlp_adapter: str | None
    num_train_epochs: float
    global_batch_size: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    model_max_length: int
    deepspeed_stage: int
    bf16: bool
    attn_implementation: str
    tf32: bool
    gradient_checkpointing: bool
    max_steps: int
    stop_after_steps: int | None
    save_steps: int
    save_total_limit: int
    logging_steps: int
    seed: int
    resume_from_checkpoint: str | None
    skip_final_save: bool
    dataset_limit: int | None
    dataloader_num_workers: int
    resource_report_path: str
    log_memory_every_n_steps: int


def _next_compressor(config: Config) -> DUETCompressor:
    options = _mapping_option(config, "compressor_args")
    options.pop("track_source", None)
    compressor = DUETCompressor(**options)
    compressor.validate_llava_next()
    compressor._metadata_backbone = "llava_next"
    return compressor


def _next_output_dir(config: Config, stage: str, budget: int) -> Path:
    name = f"llama3-llava-next-8b-duet-{budget}"
    return Path(config.output_dir) / "duet" / (
        f"{name}-pretrain" if stage == "pretrain" else name
    )


def _validate_projector_metadata(path: Path, compressor: DUETCompressor) -> None:
    metadata_path = path.parent / "duet_config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Projector {path} has no duet_config.json compatibility metadata"
        )
    import json

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    checks = {
        "schema_version": 5,
        "implementation_version": DUET_NEXT_IMPLEMENTATION_VERSION,
        "backend": "llava-next-8b-llama3",
        "base_model_id": LLAVA_NEXT_8B_MODEL_ID,
        "base_model_revision": LLAVA_NEXT_8B_REVISION,
        "transformers_version": "5.5.4",
        "training_attention_backend": DUET_NEXT_TRAINING_ATTENTION_BACKEND,
        "training_flash_attn_version": DUET_NEXT_FLASH_ATTN_VERSION,
        "inference_attention_backend": DUET_NEXT_INFERENCE_ATTENTION_BACKEND,
        "paper_profile": int(compressor._llava_next_profile),
        "requested_retention_ratio": float(compressor._retain_ratio),
        "budget_semantics": DUET_NEXT_BUDGET_SEMANTICS,
        "rounding_rule": DUET_NEXT_ROUNDING_RULE,
        "content_definition": DUET_NEXT_CONTENT_DEFINITION,
        "structural_token_policy": DUET_NEXT_STRUCTURAL_POLICY,
        "reference_views": 5,
        "reference_content_tokens": 2880,
        "profile_dominant_reference_total": compressor._dominant,
        "profile_contextual_reference_total": compressor._contextual,
        "cluster_width": compressor._cluster_width,
        "pdrop_layers": list(compressor._pdrop_layers),
        "pdrop_keep_ratios": list(compressor._pdrop_keep_ratios),
        "text_rater": compressor._text_rater,
        "score_reduction": compressor._score_reduction,
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default",
        "include_cls_in_decoder": False,
        "feature_order": DUET_NEXT_FEATURE_ORDER,
        "view_policy": DUET_NEXT_VIEW_POLICY,
        "image_grid_pinpoints": [list(point) for point in LLAVA_NEXT_8B_IMAGE_GRID_PINPOINTS],
        "vision_geometry": dict(LLAVA_NEXT_8B_VISION_GEOMETRY),
        "training_stage": "pretrain",
        "source_commits": dict(DUET_NEXT_SOURCE_COMMITS),
    }
    wrong = [key for key, value in checks.items() if payload.get(key) != value]
    if wrong:
        raise ValueError(
            "Finetune projector metadata is incompatible in fields: " + ", ".join(wrong)
        )


def build_next_training_args(backbone: VLM, config: Config, stage: str,
                             compressor: DUETCompressor) -> DuetNextTrainingArgs:
    options = _mapping_option(config, "training_args")
    defaults = NEXT_STAGE_DEFAULTS[stage]
    attn_implementation = str(
        options.get("attn_implementation", DUET_NEXT_TRAINING_ATTENTION_BACKEND)
    )
    if attn_implementation != DUET_NEXT_TRAINING_ATTENTION_BACKEND:
        raise ValueError(
            "DUET/LLaVA-NeXT training is FlashAttention-2 only; got "
            f"attn_implementation={attn_implementation!r}."
        )
    bf16 = bool(options.get("bf16", True))
    if not bf16:
        raise ValueError("DUET/LLaVA-NeXT FlashAttention-2 training requires bf16=true.")
    per_device = int(options.get("per_device_train_batch_size", 1))
    if per_device != 1:
        raise NotImplementedError(
            "DUET/LLaVA-NeXT physically prunes decoder positions; use per-device batch 1"
        )
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    global_batch = int(options.get("global_batch_size", defaults["global_batch_size"]))
    accumulation = int(options.get(
        "gradient_accumulation_steps",
        math.ceil(global_batch / world_size),
    ))
    if per_device * world_size * accumulation != global_batch:
        raise ValueError(
            "per_device_train_batch_size * WORLD_SIZE * gradient_accumulation_steps "
            f"must equal global_batch_size={global_batch}"
        )
    model_max_length = int(options.get("model_max_length", NEXT_MODEL_MAX_LENGTH))
    if model_max_length != NEXT_MODEL_MAX_LENGTH:
        raise ValueError(f"LLaVA-NeXT training pins model_max_length={NEXT_MODEL_MAX_LENGTH}")
    deepspeed_stage = int(options.get("deepspeed_stage", defaults["deepspeed_stage"]))
    if deepspeed_stage != defaults["deepspeed_stage"]:
        raise ValueError(
            f"DUET {stage} requires ZeRO-{defaults['deepspeed_stage']}"
        )
    if stage == "pretrain":
        data_path = Path(options.get(
            "data_path",
            Path(config.input_dir) / "LLaVA-Pretrain" / "blip_laion_cc_sbu_558k.json",
        ))
        image_folder: Path | None = Path(options.get(
            "image_folder", Path(config.input_dir) / "LLaVA-Pretrain" / "images"
        ))
    else:
        data_path = Path(options.get(
            "data_path", Path(config.input_dir) / "LLaVA-NeXT-Data"
        ))
        image_folder = None
    if not data_path.exists() or (image_folder is not None and not image_folder.is_dir()):
        raise FileNotFoundError(
            f"DUET/LLaVA-NeXT {stage} data is unavailable: {data_path} / {image_folder}"
        )
    output = Path(options.get("output_dir", _next_output_dir(
        config, stage, int(compressor._llava_next_profile)
    )))
    adapter: Path | None = None
    if stage == "finetune":
        candidate = options.get("pretrain_mm_mlp_adapter")
        adapter = Path(candidate) if candidate else (
            _next_output_dir(
                config, "pretrain", int(compressor._llava_next_profile)
            ) / "mm_projector.bin"
        )
        if adapter.is_dir():
            adapter = adapter / "mm_projector.bin"
        if not adapter.is_file():
            raise FileNotFoundError(
                f"Run DUET-{compressor.budget} NeXT pretraining first; missing {adapter}"
            )
        _validate_projector_metadata(adapter, compressor)
    dataset_limit = options.get("dataset_limit", config.dataset_limit)
    return DuetNextTrainingArgs(
        model_name_or_path=str(backbone._model_id),
        model_revision=str(getattr(backbone, "_model_revision", LLAVA_NEXT_8B_REVISION)),
        data_path=str(data_path),
        image_folder=None if image_folder is None else str(image_folder),
        output_dir=str(output),
        stage=stage,
        budget=int(compressor._llava_next_profile),
        pretrain_mm_mlp_adapter=None if adapter is None else str(adapter),
        num_train_epochs=float(options.get("num_train_epochs", 1.0)),
        global_batch_size=global_batch,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accumulation,
        learning_rate=float(options.get("learning_rate", defaults["learning_rate"])),
        weight_decay=float(options.get("weight_decay", 0.0)),
        warmup_ratio=float(options.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(options.get("lr_scheduler_type", "cosine")),
        model_max_length=model_max_length,
        deepspeed_stage=deepspeed_stage,
        bf16=bf16,
        attn_implementation=attn_implementation,
        tf32=bool(options.get("tf32", True)),
        gradient_checkpointing=bool(options.get("gradient_checkpointing", True)),
        max_steps=int(options.get("max_steps", -1)),
        stop_after_steps=(None if options.get("stop_after_steps") is None
                          else int(options["stop_after_steps"])),
        save_steps=int(options.get("save_steps", defaults["save_steps"])),
        save_total_limit=int(options.get("save_total_limit", 2)),
        logging_steps=int(options.get("logging_steps", 10)),
        seed=int(options.get("seed", 42)),
        resume_from_checkpoint=(None if not options.get("resume_from_checkpoint")
                                else str(options["resume_from_checkpoint"])),
        skip_final_save=bool(options.get("skip_final_save", False)),
        dataset_limit=None if dataset_limit is None else int(dataset_limit),
        dataloader_num_workers=int(options.get("dataloader_num_workers", 2)),
        resource_report_path=str(options.get(
            "resource_report_path", output / f"resource_{stage}_{compressor.budget}.json"
        )),
        log_memory_every_n_steps=int(options.get("log_memory_every_n_steps", 5)),
    )


def duet_next_metadata(compressor: DUETCompressor) -> dict[str, Any]:
    reference_content = 5 * 576
    reference_stage1 = retention_target(reference_content, compressor._retain_ratio)
    decoder_layers = int(compressor._llava_next_decoder_layers or 32)
    return {
        "schema_version": 5,
        "implementation_version": DUET_NEXT_IMPLEMENTATION_VERSION,
        "backend": "llava-next-8b-llama3",
        "base_model_id": LLAVA_NEXT_8B_MODEL_ID,
        "base_model_revision": LLAVA_NEXT_8B_REVISION,
        "transformers_version": "5.5.4",
        "training_attention_backend": DUET_NEXT_TRAINING_ATTENTION_BACKEND,
        "training_flash_attn_version": DUET_NEXT_FLASH_ATTN_VERSION,
        "inference_attention_backend": DUET_NEXT_INFERENCE_ATTENTION_BACKEND,
        "paper_profile": int(compressor._llava_next_profile),
        "requested_retention_ratio": float(compressor._retain_ratio),
        "budget_semantics": DUET_NEXT_BUDGET_SEMANTICS,
        "rounding_rule": DUET_NEXT_ROUNDING_RULE,
        "content_definition": DUET_NEXT_CONTENT_DEFINITION,
        "structural_token_policy": DUET_NEXT_STRUCTURAL_POLICY,
        "reference_views": compressor._llava_next_reference_views,
        "reference_content_tokens": reference_content,
        "profile_dominant_reference_total": compressor._dominant,
        "profile_contextual_reference_total": compressor._contextual,
        "reference_target_content_tokens": reference_stage1,
        "reference_content_layer_average": average_image_tokens(
            reference_stage1,
            compressor._pdrop_layers,
            compressor._pdrop_keep_ratios,
            decoder_layers,
        ),
        "runtime_realized_budget": (
            "records original_content_tokens, requested_retention_ratio, "
            "target_content_tokens, realized_content_tokens, realized_retention_ratio, "
            "and structural_tokens per image; Stage-2 content keeps use floor"
        ),
        "cluster_width": compressor._cluster_width,
        "pdrop_layers": list(compressor._pdrop_layers),
        "pdrop_keep_ratios": list(compressor._pdrop_keep_ratios),
        "text_rater": compressor._text_rater,
        "score_reduction": compressor._score_reduction,
        "use_stage1": compressor._use_stage1,
        "use_stage2": compressor._use_stage2,
        "vision_feature_layer": int(compressor._vision_feature_layer or -2),
        "vision_feature_select_strategy": "default",
        "include_cls_in_decoder": False,
        "image_grid_pinpoints": getattr(compressor, "_llava_next_image_grid_pinpoints", None),
        "vision_geometry": getattr(compressor, "_llava_next_vision_geometry", None),
        "feature_order": DUET_NEXT_FEATURE_ORDER,
        "view_policy": DUET_NEXT_VIEW_POLICY,
        "supported_modes": {
            "source_images_per_forward": 1,
            "batch_size": 1,
            "num_return_sequences": 1,
            "video": False,
            "beam_search": False,
            "greedy_or_sampling_dynamic_cache": True,
        },
        "training_stage": getattr(compressor, "_training_stage", None),
        "training_recipe": (
            "architecture adaptation: 558K projector alignment, then LLaVA-NeXT "
            "Parquet instruction tuning of projector+Llama with frozen CLIP"
        ),
        "released_llava_next_training_recipe": False,
        "training_dataset": getattr(compressor, "_training_dataset", None),
        "source_commits": dict(DUET_NEXT_SOURCE_COMMITS),
    }


def run_next_training_stage(backbone: VLM, config: Config, stage: str) -> None:
    compressor = _next_compressor(config)
    args = build_next_training_args(backbone, config, stage, compressor)
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    rank = int(os.environ.get("RANK", "0"))
    processor, base_model = load_hf_llava_next_training_base(
        args.model_name_or_path,
        revision=args.model_revision,
        dtype=torch.bfloat16 if args.bf16 else None,
        attn_implementation=args.attn_implementation,
        model_max_length=args.model_max_length,
        local_files_only=bool(getattr(backbone, "_local_files_only", True)),
    )
    base_model.config.use_cache = False
    base_model.config.text_config.use_cache = False
    compressor._llava_next_decoder_layers = len(base_model.model.language_model.layers)
    compressor._llava_next_image_grid_pinpoints = [
        list(map(int, point)) for point in base_model.config.image_grid_pinpoints
    ]
    vision = base_model.config.vision_config
    compressor._llava_next_vision_geometry = {
        "image_size": int(vision.image_size),
        "patch_size": int(vision.patch_size),
        "patches_per_view": int(vision.image_size // vision.patch_size) ** 2,
    }
    compressor._training_stage = stage
    compressor._training_dataset = args.data_path
    compressor.to(device=base_model.device, dtype=base_model.dtype)
    training_backbone = _TrainingBackbone(base_model, processor)
    training_backbone._model_id = args.model_name_or_path
    training_backbone.compressor_family = "llava_next"
    compressor.install(training_backbone)
    model = DuetTrainingLlavaForConditionalGeneration(base_model, compressor)

    for parameter in model.parameters():
        parameter.requires_grad_(stage == "finetune")
    # The trained ProjectNeura path freezes CLIP in both stages.  Stage 1 trains
    # only the projector; Stage 2 trains projector + Llama/lm_head.
    base_model.model.vision_tower.requires_grad_(False)
    if stage == "pretrain":
        base_model.model.language_model.requires_grad_(False)
        base_model.lm_head.requires_grad_(False)
        base_model.model.multi_modal_projector.requires_grad_(True)
    elif args.pretrain_mm_mlp_adapter:
        load_pretrained_projector(
            args.pretrain_mm_mlp_adapter, base_model.model.multi_modal_projector
        )
    _report_trainable_parameters(model)

    gradient_checkpointing_kwargs = None
    if args.gradient_checkpointing:
        gradient_checkpointing_kwargs = {"use_reentrant": False}
        if args.deepspeed_stage == 3:
            gradient_checkpointing_kwargs["determinism_check"] = "none"
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    if stage == "pretrain":
        dataset: Dataset = LlavaJsonDataset(
            args.data_path, args.image_folder or "", dataset_limit=args.dataset_limit
        )
    else:
        dataset = LlavaNextParquetDataset(args.data_path, dataset_limit=args.dataset_limit)
    collator = LlavaNextDuetCollator(
        processor,
        base_model.config,
        compressor,
        model_max_length=args.model_max_length,
        stage=stage,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        report_to="none",
        average_tokens_across_devices=CROSS_DEVICE_TOKEN_AVERAGING,
        deepspeed=_deepspeed_config(
            args.deepspeed_stage,
            args.global_batch_size,
            1,
            args.gradient_accumulation_steps,
        ),
    )
    reference_content = 5 * 576
    reference_stage1 = retention_target(reference_content, compressor._retain_ratio)
    report = ResourceReportCallback(
        stage=stage,
        budget=args.budget,
        stage1_tokens=reference_stage1,
        layer_average=average_image_tokens(
            reference_stage1,
            compressor._pdrop_layers,
            compressor._pdrop_keep_ratios,
            compressor._llava_next_decoder_layers,
        ),
        global_batch_size=args.global_batch_size,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        model_max_length=args.model_max_length,
        dataset_length=len(dataset),
        max_steps=args.max_steps,
        stop_after_steps=args.stop_after_steps,
        resource_report_path=args.resource_report_path,
        log_memory_every_n_steps=args.log_memory_every_n_steps,
    )
    if rank == 0:
        write_duet_metadata(args.output_dir, compressor, asdict(args))
    trainer = LlavaNextDuetTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[SaveControlCallback(), StopAfterStepCallback(args.stop_after_steps), report],
        group_by_modality_length=False,
        stage=stage,
    )
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        if args.skip_final_save:
            return
        if stage == "pretrain":
            unwrapped = trainer.accelerator.unwrap_model(model)
            if trainer.is_world_process_zero():
                output = Path(args.output_dir)
                output.mkdir(parents=True, exist_ok=True)
                processor.save_pretrained(output)
                unwrapped.base_model.config.save_pretrained(output)
                torch.save(
                    {key: value.detach().cpu() for key, value in
                     unwrapped.base_model.model.multi_modal_projector.state_dict().items()},
                    output / "mm_projector.bin",
                )
                write_duet_metadata(output, compressor, asdict(args))
                report.write_report()
        else:
            state_dict = _gather_final_state_dict(trainer)
            if trainer.is_world_process_zero():
                model.save_pretrained(
                    args.output_dir, state_dict=state_dict, safe_serialization=True
                )
                processor.save_pretrained(args.output_dir)
                write_duet_metadata(args.output_dir, compressor, asdict(args))
                report.write_report()
    finally:
        compressor.uninstall()


__all__ = [
    "DUET_NEXT_FLASH_ATTN_VERSION",
    "DUET_NEXT_IMPLEMENTATION_VERSION",
    "DuetNextTrainingArgs",
    "LlavaNextDuetCollator",
    "LlavaNextDuetTrainer",
    "LlavaNextParquetDataset",
    "NEXT_MODEL_MAX_LENGTH",
    "ParquetRowGroupSampler",
    "build_next_training_args",
    "duet_next_metadata",
    "find_next_parquet",
    "load_hf_llava_next_training_base",
    "require_flash_attention_2_training_runtime",
    "run_next_training_stage",
]
