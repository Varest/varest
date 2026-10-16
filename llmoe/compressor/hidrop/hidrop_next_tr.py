"""Compression-aware HiDrop training for Llama-3-LLaVA-NeXT-8B.

The supplied ``LLaVA-NeXT-Data`` is the 779,289-row mixed-modality instruction
corpus, not the original BLIP-558K alignment corpus. Consequently ``tr`` is
explicitly a *compression-adaptation warm-up*: it starts from the checkpoint's trained
projector (never reinitialises it), freezes the language and vision models, and
adapts the projector to HiDrop's late-injected/pruned stream.  ``ft`` loads that
matching projector and trains the Llama-3 language model plus projector while
keeping CLIP frozen.  This preserves HiDrop's two optimization phases without
misrepresenting the available data as original LLaVA pretraining.
"""

from bisect import bisect_right
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
import json
import math
import os

from PIL import Image
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    TrainingArguments,
    set_seed,
)
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
)

from llmoe.backbone.llava_next import LLAVA_NEXT_8B_REVISION
from llmoe.backbone.prototype import VLM
from llmoe.compressor.hidrop.config import CONFIG_METADATA_KEYS, HiDropConfig
from llmoe.compressor.hidrop.hidrop import HiDropCompressor
from llmoe.compressor.hidrop.hidrop_llama import IGNORE_INDEX, HiDropLlamaModel
from llmoe.compressor.hidrop.provenance import (
    LLAVA_NEXT_TRAINING_SEMANTICS,
    RELEASED_LOSS_SEMANTICS,
    RELEASED_TRAINING_OPTIMIZER,
    build_run_manifest,
    is_pinned_llava_next_snapshot,
    record_output_file,
    validate_artifact_provenance,
    validate_fixed_training_environment,
    write_run_manifest,
)
from llmoe.compressor.hidrop.hidrop_tr import (
    ATTN_IMPLEMENTATION,
    CROSS_DEVICE_TOKEN_AVERAGING,
    HiDropSaveControlCallback,
    HiDropTrainer,
    ResourceReportCallback,
    StopAfterStepCallback,
    _deepspeed_config,
    _gather_final_state_dict,
    _mapping_option,
    _released_causal_lm_loss,
    _report_trainable_parameters,
    _resolve_adapter_path,
    validate_hidrop_artifact,
)
from llmoe.config import Config


NEXT_MODEL_MAX_LENGTH = 8192
NEXT_STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "pretrain": {
        "learning_rate": 1e-3,
        "global_batch_size": 256,
        "deepspeed_stage": 2,
    },
    "finetune": {
        "learning_rate": 2e-5,
        "global_batch_size": 128,
        "deepspeed_stage": 3,
    },
}


def _require_flash_attention_2(attn_implementation: Any) -> str:
    """Return the only supported NeXT training backend or fail before loading weights."""
    resolved = str(attn_implementation)
    if resolved != ATTN_IMPLEMENTATION:
        raise ValueError(
            "HiDrop/LLaVA-NeXT training is FlashAttention-2 only; "
            f"got attn_implementation={resolved!r}. Inference remains on SDPA."
        )
    return resolved


def resolve_llava_next_data_root(input_dir: str | Path) -> Path:
    root = Path(input_dir).expanduser()
    candidates = (root, root / "LLaVA-NeXT-Data")
    for candidate in candidates:
        if (candidate / "data").is_dir() and list((candidate / "data").glob("train-*.parquet")):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find LLaVA-NeXT-Data Parquet shards under {root} or {root / 'LLaVA-NeXT-Data'}."
    )


class LlavaNextParquetDataset(Dataset):
    """Lazy, non-duplicating reader for the local mixed-modality Parquet shards."""

    columns = ("id", "conversations", "data_source", "image")

    def __init__(self, root: str | Path, dataset_limit: int | None = None) -> None:
        self.root = resolve_llava_next_data_root(root)
        self.data_dir = self.root / "data"
        self.shards = tuple(sorted(self.data_dir.glob("train-*.parquet")))
        if len(self.shards) != 250:
            raise ValueError(
                f"Expected the released 250 LLaVA-NeXT Parquet shards, found {len(self.shards)}."
            )
        self._shard_starts = [0]
        self._row_group_starts: list[tuple[int, ...]] = []
        required = set(self.columns)
        for shard in self.shards:
            parquet = pq.ParquetFile(shard)
            missing = required - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{shard} is missing required columns {sorted(missing)}.")
            starts = [0]
            for group_index in range(parquet.metadata.num_row_groups):
                starts.append(starts[-1] + parquet.metadata.row_group(group_index).num_rows)
            if starts[-1] != parquet.metadata.num_rows:
                raise RuntimeError(f"Parquet row-group metadata is inconsistent in {shard}.")
            self._row_group_starts.append(tuple(starts))
            self._shard_starts.append(self._shard_starts[-1] + int(parquet.metadata.num_rows))
        total = self._shard_starts[-1]
        if total != 779_289:
            raise ValueError(f"Expected 779289 LLaVA-NeXT examples, found {total}.")
        if dataset_limit is None:
            self._length = total
        else:
            limit = int(dataset_limit)
            if limit < 1:
                raise ValueError(f"dataset_limit must be positive, got {dataset_limit}.")
            self._length = min(limit, total)
        self._cached_key: tuple[int, int] | None = None
        self._cached_rows: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return self._length

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cached_key"] = None
        state["_cached_rows"] = None
        return state

    def _coordinates(self, index: int) -> tuple[int, int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._shard_starts, index) - 1
        shard_row = index - self._shard_starts[shard_index]
        starts = self._row_group_starts[shard_index]
        row_group = bisect_right(starts, shard_row) - 1
        return shard_index, row_group, shard_row - starts[row_group]

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_index, row_group, offset = self._coordinates(int(index))
        key = (shard_index, row_group)
        if self._cached_key != key or self._cached_rows is None:
            table = pq.ParquetFile(self.shards[shard_index]).read_row_group(
                row_group,
                columns=list(self.columns),
            )
            self._cached_rows = table.to_pylist()
            self._cached_key = key
        row = self._cached_rows[offset]
        image_record = row.get("image")
        image: Image.Image | None = None
        if image_record is not None:
            image_bytes = image_record.get("bytes")
            if image_bytes is None:
                image_path = image_record.get("path")
                candidate = self.root / str(image_path) if image_path else None
                if candidate is None or not candidate.is_file():
                    raise FileNotFoundError(
                        f"LLaVA-NeXT row {row.get('id')!r} has no embedded image bytes and its path "
                        f"{image_path!r} is not extracted under {self.root}."
                    )
                image = Image.open(candidate)
            else:
                image = Image.open(BytesIO(bytes(image_bytes)))
            image.load()
            image = image.convert("RGB")
        return {
            "id": str(row.get("id") or index),
            "data_source": str(row.get("data_source") or ""),
            "conversations": self._normalize_conversations(row.get("conversations") or []),
            "image": image,
        }

    @staticmethod
    def _normalize_conversations(turns: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for turn in turns:
            role = turn.get("from") or turn.get("role")
            role = {"human": "user", "gpt": "assistant"}.get(str(role), str(role))
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"Unsupported LLaVA-NeXT conversation role {role!r}.")
            normalized.append(
                {"role": role, "value": str(turn.get("value") or turn.get("content") or "")}
            )
        non_system = [turn["role"] for turn in normalized if turn["role"] != "system"]
        if not non_system or non_system[0] != "user" or "assistant" not in non_system:
            raise ValueError(
                "LLaVA-NeXT rows must contain a user turn followed by an assistant turn."
            )
        for index, role in enumerate(non_system):
            expected = "user" if index % 2 == 0 else "assistant"
            if role != expected:
                raise ValueError("LLaVA-NeXT conversations must alternate user/assistant turns.")
        return normalized


def _assistant_labels(input_ids: torch.Tensor, tokenizer: Any, image_token_id: int) -> torch.Tensor:
    """Mask Llama-3 headers/users and supervise assistant content plus EOT."""
    if input_ids.ndim != 1:
        raise ValueError(f"Expected one token row, got {tuple(input_ids.shape)}.")
    start_id = int(tokenizer.convert_tokens_to_ids("<|start_header_id|>"))
    end_id = int(tokenizer.convert_tokens_to_ids("<|end_header_id|>"))
    eot_id = int(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    assistant_ids = [
        int(value) for value in tokenizer.encode("assistant", add_special_tokens=False)
    ]
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    values = [int(value) for value in input_ids.tolist()]
    cursor = 0
    assistant_regions = 0
    while cursor < len(values):
        try:
            header_start = values.index(start_id, cursor)
            header_end = values.index(end_id, header_start + 1)
        except ValueError:
            break
        role_ids = values[header_start + 1 : header_end]
        try:
            content_end = values.index(eot_id, header_end + 1)
            stop = content_end + 1
        except ValueError:
            stop = len(values)
        if role_ids == assistant_ids:
            labels[header_end + 1 : stop] = input_ids[header_end + 1 : stop]
            assistant_regions += 1
        cursor = max(stop, header_end + 1)
    labels[input_ids == int(image_token_id)] = IGNORE_INDEX
    if assistant_regions == 0 or not bool((labels != IGNORE_INDEX).any().item()):
        raise ValueError(
            "Could not locate supervised assistant content in the Llama-3 chat template."
        )
    return labels


@dataclass
class LlavaNextHiDropCollator:
    processor: Any
    model_max_length: int = NEXT_MODEL_MAX_LENGTH

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(batch) != 1:
            raise NotImplementedError(
                "HiDrop/LLaVA-NeXT training requires per-device batch size 1 for ragged AnyRes spans."
            )
        item = batch[0]
        image_token = getattr(self.processor, "image_token", "<image>")
        has_image = item.get("image") is not None
        marker_count = sum(turn["value"].count(image_token) for turn in item["conversations"])
        expected_markers = 1 if has_image else 0
        if marker_count != expected_markers:
            raise ValueError(
                "LLaVA-NeXT image/marker mismatch: "
                f"image={has_image}, expected {expected_markers} {image_token!r} markers, "
                f"got {marker_count}."
            )
        messages: list[dict[str, Any]] = []
        image_inserted = False
        image_message_index: int | None = None
        for turn in item["conversations"]:
            parts = turn["value"].split(image_token)
            content: list[dict[str, str]] = []
            for part_index, part in enumerate(parts):
                text = part.strip()
                if text:
                    content.append({"type": "text", "text": text})
                if part_index < len(parts) - 1:
                    if turn["role"] != "user":
                        raise ValueError("The LLaVA-NeXT image marker must occur in a user turn.")
                    content.append({"type": "image"})
                    image_inserted = True
                    image_message_index = len(messages)
            if not content:
                content.append({"type": "text", "text": ""})
            messages.append({"role": turn["role"], "content": content})
        if has_image and not image_inserted:
            raise RuntimeError("The validated LLaVA-NeXT image marker was not inserted.")
        if has_image and (
            image_message_index is None
            or not any(
                message["role"] == "assistant" for message in messages[image_message_index + 1 :]
            )
        ):
            raise ValueError(
                "LLaVA-NeXT training requires an assistant response after the image turn."
            )
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Every rank follows the same vision/projector graph. Text-only rows use
        # one deterministic base-size dummy image; the wrapper consumes an empty
        # projected-feature slice, yielding an exact zero contribution. This
        # prevents ZeRO collective mismatches when only the projector is trainable.
        processor_image = item["image"] if has_image else Image.new("RGB", (336, 336), "black")
        encoded = self.processor(
            images=processor_image,
            text=rendered,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.model_max_length),
            padding=False,
        )
        input_ids = encoded["input_ids"]
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise RuntimeError(f"LLaVA-NeXT processor returned input_ids {tuple(input_ids.shape)}.")
        image_token_id = int(self.processor.tokenizer.convert_tokens_to_ids(image_token))
        image_positions = torch.where(input_ids[0] == image_token_id)[0]
        if has_image and image_positions.numel() == 0:
            raise ValueError("LLaVA-NeXT truncation removed the image placeholder span.")
        if not has_image and image_positions.numel() != 0:
            raise ValueError("A text-only LLaVA-NeXT row unexpectedly produced image placeholders.")
        expected = (
            torch.arange(
                int(image_positions[0]),
                int(image_positions[-1]) + 1,
                device=image_positions.device,
            )
            if has_image
            else image_positions
        )
        if has_image and not torch.equal(image_positions, expected):
            raise ValueError(
                "LLaVA-NeXT training supports one contiguous image placeholder run only."
            )
        labels = _assistant_labels(
            input_ids[0], self.processor.tokenizer, image_token_id
        ).unsqueeze(0)
        result = {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}
        result["labels"] = labels
        if "attention_mask" not in result:
            result["attention_mask"] = torch.ones_like(input_ids)
        if "pixel_values" not in result or "image_sizes" not in result:
            raise RuntimeError("LLaVA-NeXT processor omitted pixel_values or image_sizes.")
        return result


class HiDropTrainingLlavaNextForConditionalGeneration(nn.Module):
    """Native NeXT outer model with loss aligned to HiDrop-mutated labels."""

    def __init__(
        self, base_model: LlavaNextForConditionalGeneration, hidrop_config: HiDropConfig
    ) -> None:
        super().__init__()
        if hidrop_config.backbone_family != "llava_next":
            raise ValueError("The LLaVA-NeXT training wrapper requires a llava_next HiDrop config.")
        self.base_model = base_model
        self.config = base_model.config
        self.generation_config = base_model.generation_config
        self.hidrop_config = hidrop_config
        self.compressor = HiDropCompressor(hidrop_config, pretrained=False)
        image_token_id = getattr(base_model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(base_model.config, "image_token_index", None)
        if image_token_id is None or int(image_token_id) < 0:
            raise ValueError("LLaVA-NeXT config has no valid image token id.")
        self.compressor._image_token_id = int(image_token_id)
        self.compressor._validate_llava_next_model(base_model)
        original_language_model = base_model.model.language_model
        object.__setattr__(self, "_original_language_model", original_language_model)
        base_model.model.language_model = HiDropLlamaModel(original_language_model, self.compressor)

    @property
    def model(self) -> nn.Module:
        return self.base_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.base_model.lm_head

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        if gradient_checkpointing_kwargs is None:
            self.base_model.gradient_checkpointing_enable()
        else:
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        self.base_model.model.language_model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.base_model.gradient_checkpointing_disable()
        self.base_model.model.language_model.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_sizes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> LlavaNextCausalLMOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        image_features = None
        if pixel_values is not None:
            if input_ids is None or image_sizes is None:
                raise ValueError(
                    "LLaVA-NeXT HiDrop image training requires input_ids and image_sizes."
                )
            self.compressor.configure_llava_next_layouts(self.model, image_sizes, pixel_values)
            image_output = self.model.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                return_dict=True,
            )
            feature_rows = list(image_output.pooler_output)
            if len(feature_rows) != 1 or int(inputs_embeds.shape[0]) != 1:
                raise NotImplementedError(
                    "HiDrop/LLaVA-NeXT training supports one image in batch size 1."
                )
            row_features = feature_rows[0].to(inputs_embeds.device, inputs_embeds.dtype)
            row_mask = input_ids[0] == int(self.compressor._image_token_id)
            placeholder_count = int(row_mask.sum())
            if placeholder_count and placeholder_count != int(row_features.shape[0]):
                raise ValueError(
                    "LLaVA-NeXT packed features/placeholders differ: "
                    f"features={row_features.shape[0]}, placeholders={int(row_mask.sum())}."
                )
            if placeholder_count:
                merged = inputs_embeds[0].masked_scatter(
                    row_mask.unsqueeze(-1).expand_as(inputs_embeds[0]),
                    row_features,
                )
            else:
                # Retain the dummy vision/projector graph without changing the
                # text embeddings. The empty reduction is exactly zero.
                merged = inputs_embeds[0] + row_features[0:0].sum()
            inputs_embeds = merged.unsqueeze(0)
            image_features = row_features
        self.compressor._latest_input_ids = (
            input_ids.detach() if torch.is_tensor(input_ids) else None
        )
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("use_cache", None)
        outputs = self.model.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        updated_labels = outputs.updated_labels if outputs.updated_labels is not None else labels
        if labels is not None:
            logits = self.lm_head(hidden_states).float()
        else:
            indices = (
                slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, indices, :])
        loss = None
        if updated_labels is not None:
            loss = _released_causal_lm_loss(
                logits,
                updated_labels,
                int(self.config.text_config.vocab_size),
            )
        return LlavaNextCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )

    def save_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        state_dict = kwargs.pop("state_dict", None)
        base_state_dict = None
        if state_dict is not None:
            base_state_dict = {
                key.removeprefix("base_model."): value
                for key, value in state_dict.items()
                if key.startswith("base_model.")
            }
        wrapped = self.base_model.model.language_model
        self.base_model.model.language_model = self._original_language_model
        try:
            self.base_model.save_pretrained(output, state_dict=base_state_dict, **kwargs)
        finally:
            self.base_model.model.language_model = wrapped
        (output / "hidrop_config.json").write_text(
            json.dumps(self.hidrop_config.hidrop_metadata(), indent=2),
            encoding="utf-8",
        )


@dataclass
class LlavaNextHiDropTrainingArgs:
    model_name_or_path: str
    model_revision: str | None
    data_path: str
    image_folder: str
    dataset_format: str
    output_dir: str
    stage: str
    training_stage_semantics: str
    hidrop_requested_retention_ratio: float
    hidrop_average_retained_tokens: int
    late_entry_layer: int
    early_exit_layer: int
    layer_list: tuple[int, ...]
    image_token_keep_list: tuple[int, ...]
    budget_convention: str
    newline_policy: str
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
    group_by_modality_length: bool
    max_steps: int
    stop_after_steps: int | None
    save_steps: int
    save_total_limit: int
    seed: int
    resume_from_checkpoint: str | None
    skip_final_save: bool
    dataset_limit: int | None
    dataloader_num_workers: int
    resource_report_path: str
    log_memory_every_n_steps: int
    requested_attn_implementation: str | None = None
    effective_text_attn_implementation: str | None = None
    effective_vision_attn_implementation: str | None = None


def _revision_kwargs(model_name_or_path: str, revision: str | None) -> dict[str, str]:
    if revision and not Path(model_name_or_path).expanduser().is_dir():
        return {"revision": revision}
    return {}


def load_hf_llava_next_training_base(
    model_name_or_path: str,
    *,
    model_revision: str | None,
    dtype: torch.dtype | None,
    attn_implementation: str,
    local_files_only: bool,
) -> tuple[Any, LlavaNextForConditionalGeneration]:
    attn_implementation = _require_flash_attention_2(attn_implementation)
    revision_kwargs = _revision_kwargs(model_name_or_path, model_revision)
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
        **revision_kwargs,
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        if processor.tokenizer.eos_token is None:
            raise ValueError("Llama-3 tokenizer needs an EOS token for right padding.")
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.model_max_length = NEXT_MODEL_MAX_LENGTH
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
        **revision_kwargs,
    )
    text_attention = getattr(base_model.model.language_model.config, "_attn_implementation", None)
    vision_attention = getattr(base_model.model.vision_tower.config, "_attn_implementation", None)
    if text_attention != attn_implementation or vision_attention != attn_implementation:
        raise RuntimeError(
            "Transformers did not apply the pinned attention backend to both LLaVA-NeXT towers: "
            f"requested={attn_implementation!r}, text={text_attention!r}, vision={vision_attention!r}."
        )
    config = base_model.config
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = int(config.vision_config.patch_size)
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = config.vision_feature_select_strategy
    if getattr(processor, "num_additional_image_tokens", None) is None:
        processor.num_additional_image_tokens = 1
    image_token_id = int(processor.tokenizer.convert_tokens_to_ids("<image>"))
    config_image_id = getattr(config, "image_token_id", getattr(config, "image_token_index", None))
    if config_image_id is None or int(config_image_id) != image_token_id:
        raise ValueError(
            f"LLaVA-NeXT tokenizer/config image-token mismatch: {image_token_id} != {config_image_id}."
        )
    pad_id = int(processor.tokenizer.pad_token_id)
    config.pad_token_id = pad_id
    config.text_config.pad_token_id = pad_id
    base_model.model.language_model.config.pad_token_id = pad_id
    return processor, base_model


def _effective_attention(
    base_model: LlavaNextForConditionalGeneration,
) -> tuple[str | None, str | None]:
    return (
        getattr(base_model.model.language_model.config, "_attn_implementation", None),
        getattr(base_model.model.vision_tower.config, "_attn_implementation", None),
    )


def configure_llava_next_trainable_parameters(model: nn.Module, stage: str) -> None:
    """Apply the complete stage policy explicitly, including CLIP freezing."""
    if stage not in NEXT_STAGE_DEFAULTS:
        raise ValueError(f"Unknown LLaVA-NeXT HiDrop stage {stage!r}.")
    for parameter in model.parameters():
        parameter.requires_grad_(stage == "finetune")
    model.base_model.model.vision_tower.requires_grad_(False)
    if stage == "pretrain":
        model.base_model.model.multi_modal_projector.requires_grad_(True)


def _stage_output_dir(config: Config, stage: str, budget: int) -> Path:
    suffix = "adapt-pretrain" if stage == "pretrain" else "finetune"
    return (
        Path(config.output_dir)
        / "hidrop"
        / (f"llava-next-8b-hidrop-{budget}-{ATTN_IMPLEMENTATION}-{suffix}")
    )


def _build_config(config: Config) -> HiDropConfig:
    compressor_args = _mapping_option(config, "compressor_args")
    for key in (
        *CONFIG_METADATA_KEYS,
        "checkpoint_path",
        "model_path",
        "pretrained",
        "sdpa_flash_only",
        "validate_provenance",
    ):
        compressor_args.pop(key, None)
    compressor_args["backbone_family"] = "llava_next"
    compressor_args["batch_size_supported"] = False
    return HiDropConfig(**compressor_args)


def _build_args(
    backbone: VLM,
    config: Config,
    stage: str,
    hidrop_config: HiDropConfig,
) -> LlavaNextHiDropTrainingArgs:
    options = _mapping_option(config, "training_args")
    attn_implementation = _require_flash_attention_2(
        options.get("attn_implementation", ATTN_IMPLEMENTATION)
    )
    defaults = NEXT_STAGE_DEFAULTS[stage]
    expected_stage_semantics = (
        "stock_projector_compression_adaptation"
        if stage == "pretrain"
        else "full_compression_aware_finetune"
    )
    declared_stage = options.get("stage")
    if declared_stage is not None and str(declared_stage) != stage:
        raise ValueError(
            f"LLaVA-NeXT config declares training stage {declared_stage!r}, but the CLI selected {stage!r}."
        )
    declared_semantics = options.get("training_stage_semantics")
    if declared_semantics is not None and str(declared_semantics) != expected_stage_semantics:
        raise ValueError(
            "LLaVA-NeXT training-stage semantics differ from the implemented recipe: "
            f"{declared_semantics!r} != {expected_stage_semantics!r}."
        )
    if int(options.get("per_device_train_batch_size", 1)) != 1:
        raise ValueError("HiDrop/LLaVA-NeXT requires per_device_train_batch_size=1.")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_batch = int(options.get("global_batch_size", defaults["global_batch_size"]))
    accumulation = int(
        options.get("gradient_accumulation_steps", math.ceil(global_batch / world_size))
    )
    if world_size * accumulation != global_batch:
        raise ValueError(
            "HiDrop/LLaVA-NeXT requires WORLD_SIZE * gradient_accumulation_steps to equal "
            f"global_batch_size; got {world_size} * {accumulation} != {global_batch}."
        )
    deepspeed_stage = int(options.get("deepspeed_stage", defaults["deepspeed_stage"]))
    if deepspeed_stage != int(defaults["deepspeed_stage"]):
        raise ValueError(
            f"LLaVA-NeXT {stage} is pinned to DeepSpeed ZeRO-{defaults['deepspeed_stage']}."
        )
    model_max_length = int(options.get("model_max_length", NEXT_MODEL_MAX_LENGTH))
    if model_max_length != NEXT_MODEL_MAX_LENGTH:
        raise ValueError(f"LLaVA-NeXT-8B training pins model_max_length={NEXT_MODEL_MAX_LENGTH}.")
    if bool(options.get("group_by_modality_length", False)):
        raise ValueError(
            "HiDrop/LLaVA-NeXT uses a synchronized dummy vision graph for text-only rows; "
            "modality grouping is disabled."
        )
    data_root = resolve_llava_next_data_root(options.get("data_path", config.input_dir))
    budget = int(hidrop_config.average_retained_tokens)
    output = Path(options.get("output_dir") or _stage_output_dir(config, stage, budget))
    adapter_value = options.get("pretrain_mm_mlp_adapter")
    if stage == "pretrain":
        if adapter_value:
            raise ValueError(
                "The NeXT compression-adaptation warm-up starts from the stock projector."
            )
        adapter = None
    else:
        default_adapter = _stage_output_dir(config, "pretrain", budget) / "mm_projector.bin"
        adapter = _resolve_adapter_path(adapter_value) if adapter_value else default_adapter
        if not adapter.is_file():
            raise FileNotFoundError(
                f"HiDrop/LLaVA-NeXT finetuning requires the matching warm-up projector {adapter}."
            )
    dataset_limit = config.dataset_limit
    if options.get("dataset_limit") is not None:
        requested = int(options["dataset_limit"])
        dataset_limit = requested if dataset_limit is None else min(int(dataset_limit), requested)
    stop_after = options.get("stop_after_steps")
    stop_after = None if stop_after is None else int(stop_after)
    revision = options.get(
        "model_revision", getattr(backbone, "_model_revision", LLAVA_NEXT_8B_REVISION)
    )
    model_name_or_path = str(backbone._model_id)
    model_path = Path(model_name_or_path).expanduser()
    if model_path.is_dir():
        if not is_pinned_llava_next_snapshot(model_path):
            raise ValueError(
                "HiDrop/LLaVA-NeXT training requires the pinned cached base snapshot; "
                f"got {model_path}."
            )
    elif model_name_or_path != "llava-hf/llama3-llava-next-8b-hf":
        raise ValueError(
            "HiDrop/LLaVA-NeXT training requires llava-hf/llama3-llava-next-8b-hf; "
            f"got {model_name_or_path!r}."
        )
    if str(revision) != LLAVA_NEXT_8B_REVISION:
        raise ValueError(
            "HiDrop/LLaVA-NeXT training requires model revision "
            f"{LLAVA_NEXT_8B_REVISION}; got {revision!r}."
        )
    return LlavaNextHiDropTrainingArgs(
        model_name_or_path=model_name_or_path,
        model_revision=None if revision is None else str(revision),
        data_path=str(data_root / "data"),
        image_folder=str(data_root),
        dataset_format="parquet_mixed_embedded_image_text_v1",
        output_dir=str(output),
        stage=stage,
        training_stage_semantics=expected_stage_semantics,
        hidrop_requested_retention_ratio=float(hidrop_config.retain_ratio),
        hidrop_average_retained_tokens=budget,
        late_entry_layer=int(hidrop_config.late_entry_layer),
        early_exit_layer=int(hidrop_config.early_exit_layer),
        layer_list=tuple(hidrop_config.layer_list),
        image_token_keep_list=tuple(hidrop_config.image_token_keep_list),
        budget_convention="per_image_native_content_decoder_layer_average",
        newline_policy="structural_passthrough_until_early_exit",
        pretrain_mm_mlp_adapter=None if adapter is None else str(adapter),
        num_train_epochs=float(options.get("num_train_epochs", 1.0)),
        global_batch_size=global_batch,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=accumulation,
        learning_rate=float(options.get("learning_rate", defaults["learning_rate"])),
        weight_decay=float(options.get("weight_decay", 0.0)),
        warmup_ratio=float(options.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(options.get("lr_scheduler_type", "cosine")),
        model_max_length=model_max_length,
        deepspeed_stage=deepspeed_stage,
        bf16=bool(options.get("bf16", True)),
        attn_implementation=attn_implementation,
        tf32=bool(options.get("tf32", True)),
        gradient_checkpointing=bool(options.get("gradient_checkpointing", True)),
        group_by_modality_length=False,
        max_steps=int(options.get("max_steps", -1)),
        stop_after_steps=stop_after,
        save_steps=int(options.get("save_steps", 1000)),
        save_total_limit=int(options.get("save_total_limit", 2)),
        seed=int(options.get("seed", 42)),
        resume_from_checkpoint=(
            str(options["resume_from_checkpoint"])
            if options.get("resume_from_checkpoint")
            else None
        ),
        skip_final_save=bool(options.get("skip_final_save", False)),
        dataset_limit=None if dataset_limit is None else int(dataset_limit),
        dataloader_num_workers=int(options.get("dataloader_num_workers", 2)),
        resource_report_path=str(
            options.get("resource_report_path") or output / f"resource_{stage}_{budget}.json"
        ),
        log_memory_every_n_steps=int(options.get("log_memory_every_n_steps", 5)),
    )


def run_hidrop_next_training(backbone: VLM, config: Config, stage: str) -> None:
    if stage not in NEXT_STAGE_DEFAULTS:
        raise ValueError(f"Unknown LLaVA-NeXT HiDrop stage {stage!r}.")
    if getattr(backbone, "compressor_family", None) != "llava_next":
        raise TypeError("The LLaVA-NeXT HiDrop trainer requires --backbone llava-next-8b.")
    hidrop_config = _build_config(config)
    args = _build_args(backbone, config, stage, hidrop_config)
    _require_flash_attention_2(args.attn_implementation)
    validate_fixed_training_environment(require_distributed_dependencies=True)
    if args.save_steps < 1 or args.save_total_limit < 1:
        raise ValueError("save_steps and save_total_limit must be positive.")
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        raise ValueError("stop_after_steps must be positive when set.")
    if args.pretrain_mm_mlp_adapter:
        validate_hidrop_artifact(
            args.pretrain_mm_mlp_adapter,
            hidrop_config,
            artifact_name="LLaVA-NeXT warm-up projector",
        )
    if args.resume_from_checkpoint:
        validate_hidrop_artifact(
            args.resume_from_checkpoint,
            hidrop_config,
            artifact_name="LLaVA-NeXT resume checkpoint",
        )

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    processor, base_model = load_hf_llava_next_training_base(
        args.model_name_or_path,
        model_revision=args.model_revision,
        dtype=torch.bfloat16 if args.bf16 else None,
        attn_implementation=args.attn_implementation,
        local_files_only=bool(getattr(backbone, "_local_files_only", True)),
    )
    args.requested_attn_implementation = args.attn_implementation
    args.effective_text_attn_implementation, args.effective_vision_attn_implementation = (
        _effective_attention(base_model)
    )
    model = HiDropTrainingLlavaNextForConditionalGeneration(base_model, hidrop_config)
    if stage == "finetune":
        adapter_state = torch.load(
            args.pretrain_mm_mlp_adapter, map_location="cpu", weights_only=True
        )
        model.base_model.model.multi_modal_projector.load_state_dict(adapter_state, strict=True)
    configure_llava_next_trainable_parameters(model, stage)
    _report_trainable_parameters(model)
    checkpoint_kwargs = {"use_reentrant": True} if args.gradient_checkpointing else None
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=RELEASED_TRAINING_OPTIMIZER,
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=checkpoint_kwargs,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_drop_last=True,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=10,
        average_tokens_across_devices=CROSS_DEVICE_TOKEN_AVERAGING,
        deepspeed=_deepspeed_config(
            args.deepspeed_stage,
            args.global_batch_size,
            1,
            args.gradient_accumulation_steps,
        ),
    )
    dataset = LlavaNextParquetDataset(args.image_folder, dataset_limit=args.dataset_limit)
    collator = LlavaNextHiDropCollator(processor, args.model_max_length)
    rank = int(os.environ.get("RANK", "0"))
    manifest = None
    if rank == 0:
        manifest = build_run_manifest(
            args=asdict(args),
            hidrop=hidrop_config.hidrop_metadata(),
            dataset_length=len(dataset),
            optimizer=RELEASED_TRAINING_OPTIMIZER,
            loss_semantics=RELEASED_LOSS_SEMANTICS,
            training_semantics=LLAVA_NEXT_TRAINING_SEMANTICS,
        )
        if args.pretrain_mm_mlp_adapter:
            validate_artifact_provenance(
                args.pretrain_mm_mlp_adapter,
                manifest,
                artifact_name="LLaVA-NeXT warm-up projector",
                resume=False,
            )
        if args.resume_from_checkpoint:
            validate_artifact_provenance(
                args.resume_from_checkpoint,
                manifest,
                artifact_name="LLaVA-NeXT resume checkpoint",
                resume=True,
            )
        write_run_manifest(args.output_dir, manifest)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "training_args_hidrop.json").write_text(
            json.dumps(asdict(args), indent=2, default=str),
            encoding="utf-8",
        )
    report = ResourceReportCallback(
        stage=stage,
        budget=args.hidrop_average_retained_tokens,
        requested_attn_implementation=args.attn_implementation,
        effective_text_attn_implementation=args.effective_text_attn_implementation,
        effective_vision_attn_implementation=args.effective_vision_attn_implementation,
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
    trainer = HiDropTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[
            HiDropSaveControlCallback(),
            StopAfterStepCallback(args.stop_after_steps),
            report,
        ],
        group_by_modality_length=False,
        run_manifest=manifest,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if stage == "pretrain":
        unwrapped = trainer.accelerator.unwrap_model(model)
        if trainer.is_world_process_zero():
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(output)
            unwrapped.base_model.config.save_pretrained(output)
            if getattr(unwrapped.base_model, "generation_config", None) is not None:
                unwrapped.base_model.generation_config.save_pretrained(output)
            projector_state = {
                key: value.detach().cpu()
                for key, value in unwrapped.base_model.model.multi_modal_projector.state_dict().items()
            }
            projector_path = output / "mm_projector.bin"
            torch.save(projector_state, projector_path)
            metadata = unwrapped.hidrop_config.hidrop_metadata()
            metadata["training_stage"] = "pretrain_compression_adaptation"
            (output / "hidrop_config.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            if manifest is not None:
                record_output_file(manifest, "mm_projector", projector_path)
                write_run_manifest(output, manifest)
            report.write_report()
        return
    if args.skip_final_save:
        if trainer.is_world_process_zero():
            report.write_report()
        return
    state_dict = _gather_final_state_dict(trainer)
    if trainer.is_world_process_zero():
        model.save_pretrained(args.output_dir, state_dict=state_dict, safe_serialization=True)
        processor.save_pretrained(args.output_dir)
        metadata_path = Path(args.output_dir) / "hidrop_config.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["training_stage"] = "finetune"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        report.write_report()
        print(f"Saved HiDrop/LLaVA-NeXT checkpoint to {args.output_dir}", flush=True)
