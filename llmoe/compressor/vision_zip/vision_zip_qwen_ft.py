"""Merger-only Qwen2.5-VL finetuning for VisionZip.

The released Qwen VisionZip model reduces decoder-side visual embeddings after
visual.merger, gathers the full-sequence M-RoPE positions through the same keep
mask, and then enters the language model. This wrapper preserves that forward
contract while applying each keep set independently per NaViT row, so a training
microbatch may contain images with different visual-token counts.

As in VisionZip Efficient Tuning on LLaVA, the vision encoder and language model
stay frozen. Qwen has no separate multi_modal_projector: visual.merger performs
the patch merge and the projection into decoder width, so it is the sole
trainable module and the only module written to the resulting checkpoint.
Selection itself remains non-differentiable; gradients flow through selected and
contextually merged visual embeddings into visual.merger.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import math
import os

import torch
from torch import nn
from torch.utils.data import Dataset, Subset
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from llmoe.backbone import VLM
from llmoe.compressor.hidrop.data import (
    LlavaJsonDataset,
    QwenHiDropCollator,
)
from llmoe.compressor.vision_zip.vision_zip import (
    VisionZipCompressor,
    contextual_merge,
)
from llmoe.config import Config
from llmoe.utils.qwen_training import (
    QwenVisionSignalCapture,
    pad_qwen_training_rows,
    qwen_training_row_views,
    qwen_visual_spans,
)


IGNORE_INDEX = -100
UPSTREAM_REPOSITORY = "https://github.com/JIA-Lab-research/VisionZip"
UPSTREAM_COMMIT = "8f86b55c6f000eb033e6912538af2dd7dcb30502"
SUPPORTED_ATTENTION_BACKENDS = ("flash_attention_2", "eager")
DEFAULT_GLOBAL_BATCH_SIZE = 128


class VisionZipTrainingQwenForConditionalGeneration(nn.Module):
    """Qwen wrapper with batched upstream-equivalent VisionZip surgery."""

    def __init__(
        self,
        base_model: nn.Module,
        compressor: VisionZipCompressor,
        processor: Any,
    ) -> None:
        super().__init__()
        if compressor._retain_ratio is None:
            raise ValueError(
                "Qwen VisionZip training requires compressor_args.retain_ratio."
            )
        self.base_model = base_model
        self.config = base_model.config
        self.generation_config = base_model.generation_config
        self.compressor = compressor
        self.processor = processor
        self.image_token_id = int(base_model.config.image_token_id)
        self.pad_token_id = int(
            getattr(processor.tokenizer, "pad_token_id", 0) or 0
        )
        self._vision_capture = QwenVisionSignalCapture(
            base_model.model.visual,
            attention_chunk=int(compressor._attention_chunk),
            label="VisionZip/Qwen training",
        )
        self.config.vision_zip = self.vision_zip_metadata()

    @property
    def model(self) -> nn.Module:
        return self.base_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.base_model.lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if gradient_checkpointing_kwargs is None:
            self.base_model.gradient_checkpointing_enable()
        else:
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        visual = self.model.visual
        if hasattr(visual, "gradient_checkpointing_disable"):
            visual.gradient_checkpointing_disable()
        elif hasattr(visual, "gradient_checkpointing"):
            visual.gradient_checkpointing = False

    def gradient_checkpointing_disable(self) -> None:
        self.base_model.gradient_checkpointing_disable()

    def enable_input_require_grads(self) -> None:
        if hasattr(self.base_model, "enable_input_require_grads"):
            self.base_model.enable_input_require_grads()

    def remove_training_hooks(self) -> None:
        self._vision_capture.remove()

    def vision_zip_metadata(self) -> dict[str, Any]:
        retain_ratio = float(self.compressor._retain_ratio)
        contextual_ratio = float(self.compressor._contextual_ratio)
        return {
            "format_version": 2,
            "method": "VisionZip",
            "backbone": "qwen2_5_vl",
            "retain_ratio": retain_ratio,
            "dominant_ratio": retain_ratio - contextual_ratio,
            "contextual_ratio": contextual_ratio,
            "training_scope": "visual_merger_only",
            "projector_equivalent": "model.visual.merger",
            "batch_size_supported": True,
            "released_qwen_forward": True,
            "published_training_recipe": False,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
        }

    def _position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_grid_thw: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if position_ids is not None:
            if position_ids.ndim == 2 and position_ids.shape == input_ids.shape:
                return position_ids[None].expand(3, *position_ids.shape)
            if (
                position_ids.ndim == 3
                and position_ids.shape[0] in (3, 4)
                and position_ids.shape[1:] == input_ids.shape
            ):
                return position_ids
            raise ValueError(
                "VisionZip Qwen position_ids must have shape [B, L], "
                f"[3, B, L], or [4, B, L]; got {tuple(position_ids.shape)}."
            )
        image_rows = (input_ids == self.image_token_id).any(dim=1)
        if image_grid_thw.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "VisionZip's Qwen collator must provide one real or dummy image "
                f"grid per row; got {image_grid_thw.shape[0]} grids for batch "
                f"{input_ids.shape[0]}."
            )
        real_grids = image_grid_thw[image_rows]
        token_types = (input_ids == self.image_token_id).to(torch.int)
        mrope_ids, _ = self.model.get_rope_index(
            input_ids,
            token_types,
            image_grid_thw=real_grids,
            attention_mask=attention_mask,
        )
        return mrope_ids

    def _token_counts(self, visual_count: int) -> tuple[int, int]:
        dominant_ratio = float(
            self.compressor._retain_ratio
            - self.compressor._contextual_ratio
        )
        dominant = min(
            int(dominant_ratio * int(visual_count)), int(visual_count)
        )
        contextual = max(
            int(self.compressor._contextual_ratio * int(visual_count)), 1
        )
        contextual = min(contextual, int(visual_count) - dominant)
        return max(0, dominant), max(0, contextual)

    def _apply_vision_zip(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        mrope_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        feature_rows: list[torch.Tensor],
        score_rows: list[torch.Tensor],
        metric_rows: list[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Apply the official dominant/contextual policy independently per row."""

        batch = int(input_ids.shape[0])
        if not (
            len(feature_rows)
            == len(score_rows)
            == len(metric_rows)
            == batch
        ):
            raise ValueError(
                "VisionZip Qwen training needs one feature, score, and metric "
                f"row per example; got features={len(feature_rows)}, "
                f"scores={len(score_rows)}, metrics={len(metric_rows)}, "
                f"batch={batch}."
            )
        spans = qwen_visual_spans(
            input_ids,
            self.image_token_id,
            attention_mask,
            label="VisionZip Qwen training",
        )
        hidden_rows: list[torch.Tensor] = []
        mrope_rows: list[torch.Tensor] = []
        id_rows: list[torch.Tensor] = []
        mask_rows = [] if attention_mask is not None else None
        label_rows = [] if labels is not None else None

        for row, span in enumerate(spans):
            hidden_row, mrope_row, id_row, mask_row, label_row = (
                qwen_training_row_views(
                    inputs_embeds,
                    mrope_ids,
                    input_ids,
                    attention_mask,
                    labels,
                    row,
                )
            )
            features = feature_rows[row].to(
                hidden_row.device, hidden_row.dtype
            )
            scores = score_rows[row].to(hidden_row.device)
            metric = metric_rows[row].to(hidden_row.device)
            if span is None:
                hidden_row = hidden_row + features[0:0].sum()
            else:
                visual_count = span.count
                if (
                    int(features.shape[0]) != visual_count
                    or int(scores.numel()) != visual_count
                    or int(metric.shape[0]) != visual_count
                ):
                    raise ValueError(
                        "VisionZip Qwen features, signals, and placeholders "
                        f"disagree for row {row}: placeholders={visual_count}, "
                        f"features={features.shape[0]}, scores={scores.numel()}, "
                        f"metric={metric.shape[0]}."
                    )
                dominant_num, contextual_num = self._token_counts(
                    visual_count
                )
                keep_local = torch.arange(
                    visual_count, device=features.device
                )
                selected_features = features
                if (
                    contextual_num >= 1
                    and dominant_num + contextual_num < visual_count
                ):
                    dominant_mask = torch.zeros(
                        visual_count,
                        dtype=torch.bool,
                        device=features.device,
                    )
                    if dominant_num:
                        dominant_mask[
                            scores.topk(dominant_num).indices
                        ] = True
                    leftover = torch.where(~dominant_mask)[0]
                    normalized_metric = metric.index_select(
                        0, leftover
                    ).float()
                    normalized_metric = normalized_metric / (
                        normalized_metric.norm(
                            dim=-1, keepdim=True
                        ).clamp_min(1e-6)
                    )
                    merged = contextual_merge(
                        features.index_select(0, leftover).unsqueeze(0),
                        normalized_metric.unsqueeze(0),
                        contextual_num,
                    )
                    target_positions = leftover.index_select(
                        0, merged.target_indices
                    )
                    keep_visual = dominant_mask.clone()
                    keep_visual[target_positions] = True
                    keep_local = torch.where(keep_visual)[0]
                    modified = features.index_copy(
                        0,
                        target_positions,
                        merged.tokens[0].to(features.dtype),
                    )
                    selected_features = modified.index_select(
                        0, keep_local
                    )

                hidden_row = torch.cat(
                    (
                        hidden_row[: span.start],
                        selected_features,
                        hidden_row[span.end :],
                    ),
                    dim=0,
                )
                selected_columns = torch.cat(
                    (
                        torch.arange(span.start, device=id_row.device),
                        span.start + keep_local.to(id_row.device),
                        torch.arange(
                            span.end,
                            id_row.shape[0],
                            device=id_row.device,
                        ),
                    )
                )
                mrope_row = mrope_row.index_select(
                    -1, selected_columns
                )
                id_row = id_row.index_select(0, selected_columns)
                if mask_row is not None:
                    mask_row = mask_row.index_select(
                        0, selected_columns
                    )
                if label_row is not None:
                    label_row = label_row.index_select(
                        0, selected_columns
                    )

            hidden_rows.append(hidden_row)
            mrope_rows.append(mrope_row)
            id_rows.append(id_row)
            if mask_rows is not None:
                mask_rows.append(mask_row)
            if label_rows is not None:
                label_rows.append(label_row)

        return pad_qwen_training_rows(
            hidden_rows,
            mrope_rows,
            id_rows,
            mask_rows,
            label_rows,
            pad_token_id=self.pad_token_id,
            ignore_index=IGNORE_INDEX,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise NotImplementedError(
                "VisionZip Qwen finetuning currently supports still images, "
                "matching the released implementation."
            )
        if input_ids is None or inputs_embeds is not None:
            raise ValueError(
                "VisionZip Qwen training requires input_ids and builds its own "
                "compression-aware embeddings."
            )
        if pixel_values is None or image_grid_thw is None:
            raise ValueError(
                "VisionZip Qwen training requires pixel_values and "
                "image_grid_thw from the collator."
            )
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("use_cache", None)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        position_ids = self._position_ids(
            input_ids, attention_mask, image_grid_thw, position_ids
        )
        image_output = self.model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        feature_rows = list(image_output.pooler_output)
        score_rows, metric_rows = self._vision_capture.consume(
            image_grid_thw
        )
        (
            inputs_embeds,
            position_ids,
            input_ids,
            attention_mask,
            labels,
        ) = self._apply_vision_zip(
            inputs_embeds,
            input_ids,
            position_ids,
            attention_mask,
            labels,
            feature_rows,
            score_rows,
            metric_rows,
        )
        outputs = self.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        if labels is not None:
            logits = self.lm_head(hidden_states).float()
        else:
            indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, indices, :])
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(
                shift_logits.device
            )
            loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )

        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLCausalLMOutputWithPast,
        )

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=None,
        )


@dataclass
class VisionZipQwenTrainingArgs:
    """Resolved settings for merger-only, compression-aware Qwen SFT."""

    model_name_or_path: str
    data_path: str
    image_folder: str
    output_dir: str
    checkpoint_name: str
    initial_checkpoint_path: str | None
    retain_ratio: float
    contextual_ratio: float
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
    deepspeed_stage: int
    bf16: bool
    fp16: bool
    tf32: bool
    attn_implementation: str
    gradient_checkpointing: bool
    subset_frac: float
    dataset_limit: int | None
    dataloader_num_workers: int
    max_steps: int
    stop_after_steps: int | None
    save_steps: int
    save_total_limit: int
    logging_steps: int
    seed: int
    skip_final_save: bool


def _mapping_option(config: Config, key: str) -> dict[str, Any]:
    value = config.custom_args.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return dict(value)


def build_qwen_training_args(
    backbone: VLM,
    config: Config,
    compressor: VisionZipCompressor,
    dataset_dir: Path,
) -> VisionZipQwenTrainingArgs:
    """Resolve merger-only Qwen tuning settings.

    VisionZip publishes its Qwen reduction forward but not a Qwen optimizer
    recipe. Defaults therefore mirror this repository's LLaVA Efficient Tuning
    path while retaining HiDrop-style global-batch and distributed controls.
    """

    if compressor._retain_ratio is None:
        raise ValueError(
            "VisionZip Qwen finetuning requires retain_ratio."
        )
    options = _mapping_option(config, "training_args")
    if options.get("resume_from_checkpoint"):
        raise ValueError(
            "VisionZip saves only visual.merger weights. To continue tuning, "
            "pass the earlier .pt file as compressor_args.checkpoint_path; "
            "optimizer-state resume is intentionally unsupported."
        )

    per_device = int(
        options.get(
            "per_device_train_batch_size",
            options.get("batch_size", 1),
        )
    )
    if per_device < 1:
        raise ValueError(
            "per_device_train_batch_size must be at least one."
        )
    global_batch = int(
        options.get("global_batch_size", DEFAULT_GLOBAL_BATCH_SIZE)
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    accumulation_value = options.get("gradient_accumulation_steps")
    if accumulation_value is None:
        accumulation = max(
            1, math.ceil(global_batch / (per_device * world_size))
        )
    else:
        accumulation = int(accumulation_value)
    if accumulation < 1:
        raise ValueError(
            "gradient_accumulation_steps must be at least one."
        )
    effective_global = per_device * world_size * accumulation
    if effective_global != global_batch:
        raise ValueError(
            "VisionZip requires an exact global batch product: "
            "per_device_train_batch_size * WORLD_SIZE * "
            f"gradient_accumulation_steps = {effective_global}, requested "
            f"global_batch_size={global_batch}."
        )

    attention_backend = str(
        options.get("attn_implementation", "flash_attention_2")
    )
    if attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError(
            "The released Qwen VisionZip path supports FlashAttention-2 and "
            f"eager attention; got {attention_backend!r}."
        )
    bf16 = bool(options.get("bf16", True))
    fp16 = bool(options.get("fp16", False))
    if bf16 and fp16:
        raise ValueError("Choose at most one of bf16 and fp16.")

    subset_frac = float(options.get("subset_frac", 0.1))
    if not 0.0 < subset_frac <= 1.0:
        raise ValueError(
            f"subset_frac must be in (0, 1], got {subset_frac}."
        )
    warmup_ratio = float(options.get("warmup_ratio", 0.03))
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1).")
    deepspeed_stage = int(options.get("deepspeed_stage", 0))
    if deepspeed_stage not in (0, 2, 3):
        raise ValueError(
            "deepspeed_stage must be 0 (disabled), 2, or 3."
        )

    save_steps = int(options.get("save_steps", 1000))
    save_total_limit = int(options.get("save_total_limit", 2))
    if save_steps < 0 or save_total_limit < 1:
        raise ValueError(
            "save_steps must be non-negative and save_total_limit at least one."
        )
    stop_after_steps = options.get("stop_after_steps")
    if stop_after_steps is not None:
        stop_after_steps = int(stop_after_steps)
        if stop_after_steps < 1:
            raise ValueError(
                "stop_after_steps must be at least one when provided."
            )

    data_path = Path(
        options.get("data_path") or _find_annotations(dataset_dir)
    )
    image_folder = Path(options.get("image_folder") or dataset_dir)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"VisionZip annotations not found: {data_path}"
        )
    if not image_folder.is_dir():
        raise FileNotFoundError(
            f"VisionZip image folder not found: {image_folder}"
        )

    output_dir = Path(
        options.get("output_dir")
        or (Path(config.output_dir) / "vision_zip")
    )
    checkpoint_name = str(
        options.get(
            "checkpoint_name",
            f"vision_zip_merger_r{float(compressor._retain_ratio):g}.pt",
        )
    )
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError(
            "checkpoint_name must be a filename, not a path."
        )

    dataset_limit = config.dataset_limit
    configured_limit = options.get(
        "dataset_limit", options.get("max_samples")
    )
    if configured_limit is not None:
        configured_limit = int(configured_limit)
        dataset_limit = (
            configured_limit
            if dataset_limit is None
            else min(int(dataset_limit), configured_limit)
        )

    return VisionZipQwenTrainingArgs(
        model_name_or_path=str(backbone._model_id),
        data_path=str(data_path),
        image_folder=str(image_folder),
        output_dir=str(output_dir),
        checkpoint_name=checkpoint_name,
        initial_checkpoint_path=(
            None
            if compressor._checkpoint_path is None
            else str(compressor._checkpoint_path)
        ),
        retain_ratio=float(compressor._retain_ratio),
        contextual_ratio=float(compressor._contextual_ratio),
        num_train_epochs=float(
            options.get(
                "num_train_epochs", options.get("num_epochs", 1)
            )
        ),
        global_batch_size=global_batch,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accumulation,
        learning_rate=float(options.get("learning_rate", 2e-5)),
        weight_decay=float(options.get("weight_decay", 0.0)),
        max_grad_norm=float(options.get("max_grad_norm", 1.0)),
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=str(
            options.get("lr_scheduler_type", "cosine")
        ),
        model_max_length=int(
            options.get(
                "model_max_length", options.get("max_length", 2048)
            )
        ),
        deepspeed_stage=deepspeed_stage,
        bf16=bf16,
        fp16=fp16,
        tf32=bool(options.get("tf32", True)),
        attn_implementation=attention_backend,
        gradient_checkpointing=bool(
            options.get("gradient_checkpointing", True)
        ),
        subset_frac=subset_frac,
        dataset_limit=(
            None if dataset_limit is None else int(dataset_limit)
        ),
        dataloader_num_workers=int(
            options.get(
                "dataloader_num_workers",
                options.get("num_workers", 2),
            )
        ),
        max_steps=int(options.get("max_steps", -1)),
        stop_after_steps=stop_after_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=int(options.get("logging_steps", 10)),
        seed=int(options.get("seed", 42)),
        skip_final_save=bool(options.get("skip_final_save", False)),
    )


def _find_annotations(dataset_dir: Path) -> Path:
    for name in (
        "llava_v1_5_mix665k.json",
        "llava_instruct_150k.json",
        "train.json",
    ):
        matches = sorted(dataset_dir.rglob(name))
        if matches:
            return matches[0]
    matches = sorted(dataset_dir.rglob("*.json"))
    if not matches:
        raise FileNotFoundError(
            f"No JSON annotations found under {dataset_dir}"
        )
    return matches[0]


def _load_training_base(
    backbone: VLM, args: VisionZipQwenTrainingArgs
) -> tuple[Any, nn.Module]:
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )

    processor_options: dict[str, Any] = {}
    min_pixels = getattr(backbone, "_min_pixels", None)
    max_pixels = getattr(backbone, "_max_pixels", None)
    if min_pixels is not None:
        processor_options["min_pixels"] = int(min_pixels)
    if max_pixels is not None:
        processor_options["max_pixels"] = int(max_pixels)
    local_files_only = bool(
        getattr(backbone, "_local_files_only", True)
    )
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        local_files_only=local_files_only,
        **processor_options,
    )
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = int(args.model_max_length)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if args.bf16
        else torch.float16
        if args.fp16
        else None
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    tokenizer_image_id = int(
        tokenizer.convert_tokens_to_ids(image_token)
    )
    if tokenizer_image_id != int(base_model.config.image_token_id):
        raise ValueError(
            "Qwen tokenizer/config image-token mismatch: "
            f"tokenizer={tokenizer_image_id}, "
            f"config={base_model.config.image_token_id}."
        )
    for model_config in (
        base_model.config,
        base_model.config.text_config,
        base_model.config.vision_config,
    ):
        model_config._attn_implementation = args.attn_implementation
        if hasattr(model_config, "_attn_implementation_internal"):
            model_config._attn_implementation_internal = (
                args.attn_implementation
            )
    base_model.config.pad_token_id = int(tokenizer.pad_token_id)
    base_model.config.text_config.pad_token_id = int(
        tokenizer.pad_token_id
    )
    base_model.config.use_cache = False
    base_model.config.text_config.use_cache = False
    return processor, base_model



def _load_merger_checkpoint(
    merger: nn.Module,
    checkpoint_path: str | Path,
    *,
    retain_ratio: float,
    contextual_ratio: float,
) -> None:
    """Load a prior merger-only artifact as the tuning initialization."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"VisionZip merger checkpoint not found: {path}"
        )
    checkpoint_config: Mapping[str, Any] = {}
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state: Mapping[str, torch.Tensor] = load_file(str(path))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, Mapping) and "state_dict" in payload:
            state = payload["state_dict"]
            candidate_config = payload.get("config", {})
            if isinstance(candidate_config, Mapping):
                checkpoint_config = candidate_config
        elif isinstance(payload, Mapping):
            state = payload
        else:
            raise TypeError(
                "Unsupported VisionZip merger checkpoint payload: "
                f"{type(payload).__name__}."
            )
    budget = checkpoint_config.get("retain_ratio")
    if budget is not None and float(budget) != float(retain_ratio):
        raise ValueError(
            f"Checkpoint was tuned for retain_ratio={budget}, "
            f"requested {retain_ratio}."
        )
    contextual = checkpoint_config.get("contextual_ratio")
    if (
        contextual is not None
        and float(contextual) != float(contextual_ratio)
    ):
        raise ValueError(
            f"Checkpoint was tuned for contextual_ratio={contextual}, "
            f"requested {contextual_ratio}."
        )
    missing, unexpected = merger.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"VisionZip checkpoint {str(path)!r} does not match "
            "model.visual.merger "
            f"(missing={list(missing)}, unexpected={list(unexpected)})."
        )


def _freeze_for_merger_tuning(base_model: nn.Module) -> nn.Module:
    """Freeze the Qwen ViT and LLM, then enable visual.merger alone."""

    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    merger = base_model.model.visual.merger
    merger.requires_grad_(True)
    trainable = sum(
        parameter.numel()
        for parameter in base_model.parameters()
        if parameter.requires_grad
    )
    merger_parameters = sum(
        parameter.numel() for parameter in merger.parameters()
    )
    if trainable != merger_parameters:
        raise RuntimeError(
            f"Trainable params ({trainable}) do not equal visual.merger "
            f"params ({merger_parameters}); freezing is incorrect."
        )
    return merger


def _cpu_merger_state(
    merger: nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(torch.float32).cpu().contiguous()
        for key, value in merger.state_dict().items()
    }


def _save_merger_artifact(
    path: Path,
    state_dict: Mapping[str, torch.Tensor],
    model: VisionZipTrainingQwenForConditionalGeneration,
    args: VisionZipQwenTrainingArgs,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = model.vision_zip_metadata()
    torch.save(
        {
            "format_version": 2,
            "compressor": VisionZipCompressor.__name__,
            "config": {
                **metadata,
                "base_model": args.model_name_or_path,
                "module_path": "model.visual.merger",
            },
            "training_args": asdict(args),
            "state_dict": {
                key: value.detach().to(torch.float32).cpu().contiguous()
                for key, value in state_dict.items()
            },
        },
        path,
    )


def _subset_dataset(
    dataset: Dataset, fraction: float, seed: int
) -> Dataset:
    if fraction >= 1.0:
        return dataset
    count = max(1, int(round(len(dataset) * fraction)))
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(
        len(dataset), generator=generator
    )[:count].tolist()
    return Subset(dataset, indices)


def _deepspeed_config(
    args: VisionZipQwenTrainingArgs,
) -> dict[str, Any] | None:
    if args.deepspeed_stage == 0:
        return None
    zero: dict[str, Any] = {
        "stage": int(args.deepspeed_stage),
        "overlap_comm": True,
        "contiguous_gradients": True,
    }
    if args.deepspeed_stage == 3:
        zero["stage3_gather_16bit_weights_on_model_save"] = False
    return {
        "train_batch_size": int(args.global_batch_size),
        "train_micro_batch_size_per_gpu": int(
            args.per_device_train_batch_size
        ),
        "gradient_accumulation_steps": int(
            args.gradient_accumulation_steps
        ),
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto"},
        "zero_optimization": zero,
    }


class _StopAfterStepCallback(TrainerCallback):
    def __init__(self, stop_after_steps: int | None) -> None:
        self.stop_after_steps = stop_after_steps

    def on_step_end(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        if (
            self.stop_after_steps is not None
            and int(state.global_step) >= int(self.stop_after_steps)
        ):
            control.should_training_stop = True
        return control


class VisionZipQwenTrainer(Trainer):
    """Trainer whose model checkpoints contain visual.merger and nothing else."""

    CHECKPOINT_FILENAME = "vision_zip_merger.pt"

    def __init__(
        self,
        *args: Any,
        resolved_args: VisionZipQwenTrainingArgs,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.resolved_args = resolved_args
        self.model_accepts_loss_kwargs = False

    def _merger(self) -> nn.Module:
        return self.model.base_model.model.visual.merger

    def gather_merger_state(self) -> dict[str, torch.Tensor]:
        """Gather only the tiny trainable module, including under ZeRO-3."""

        merger = self._merger()
        if self.is_deepspeed_enabled and self.resolved_args.deepspeed_stage == 3:
            import deepspeed

            parameters = list(merger.parameters())
            with deepspeed.zero.GatheredParameters(
                parameters, modifier_rank=None
            ):
                if self.args.should_save:
                    return _cpu_merger_state(merger)
                return {}
        if self.args.should_save:
            return _cpu_merger_state(merger)
        return {}

    def _save(
        self,
        output_dir: str | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        destination = Path(output_dir or self.args.output_dir)
        merger_state = (
            self.gather_merger_state()
            if state_dict is None
            else state_dict
        )
        _save_merger_artifact(
            destination / self.CHECKPOINT_FILENAME,
            merger_state,
            self.model,
            self.resolved_args,
        )

    def save_model(
        self,
        output_dir: str | None = None,
        _internal_call: bool = False,
    ) -> None:
        """Bypass Trainer's full-model state gathering."""

        destination = Path(output_dir or self.args.output_dir)
        state_dict = self.gather_merger_state()
        if self.args.should_save:
            self._save(str(destination), state_dict=state_dict)


def finetune_vision_zip_qwen(
    backbone: VLM,
    config: Config,
    compressor: VisionZipCompressor,
    dataset_dir: Path,
) -> None:
    """Tune only Qwen visual.merger with the released VisionZip forward."""

    args = build_qwen_training_args(
        backbone, config, compressor, dataset_dir
    )
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    processor, base_model = _load_training_base(backbone, args)
    merger = base_model.model.visual.merger
    if args.initial_checkpoint_path is not None:
        _load_merger_checkpoint(
            merger,
            args.initial_checkpoint_path,
            retain_ratio=args.retain_ratio,
            contextual_ratio=args.contextual_ratio,
        )
    model = VisionZipTrainingQwenForConditionalGeneration(
        base_model,
        compressor,
        processor,
    )
    _freeze_for_merger_tuning(base_model)

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[VisionZip/Qwen] merger-only Efficient Tuning "
            f"trainable={trainable:,}/{total:,} "
            f"({100.0 * trainable / max(1, total):.4f}%), "
            f"retain={args.retain_ratio:g}, "
            f"microbatch={args.per_device_train_batch_size}, "
            f"global_batch={args.global_batch_size}",
            flush=True,
        )

    source_dataset = LlavaJsonDataset(
        args.data_path,
        args.image_folder,
        dataset_limit=None,
    )
    image_indices = [
        index
        for index, has_image in enumerate(source_dataset.has_images)
        if has_image
    ]
    dataset: Dataset = Subset(source_dataset, image_indices)
    dataset = _subset_dataset(dataset, args.subset_frac, args.seed)
    if args.dataset_limit is not None:
        dataset = Subset(
            dataset,
            range(min(len(dataset), args.dataset_limit)),
        )
    if len(dataset) == 0:
        raise RuntimeError(
            "VisionZip Qwen finetuning has no training examples."
        )
    collator = QwenHiDropCollator(
        processor=processor,
        model_max_length=args.model_max_length,
        stage="finetune",
    )
    save_strategy = "no" if args.save_steps == 0 else "steps"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {
                "use_reentrant": (
                    args.deepspeed_stage == 3
                )
            }
            if args.gradient_checkpointing
            else None
        ),
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        max_steps=args.max_steps,
        save_strategy=save_strategy,
        save_steps=max(1, args.save_steps),
        save_total_limit=args.save_total_limit,
        save_only_model=True,
        logging_steps=args.logging_steps,
        report_to="none",
        average_tokens_across_devices=False,
        ddp_find_unused_parameters=False,
        deepspeed=_deepspeed_config(args),
    )
    trainer = VisionZipQwenTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[_StopAfterStepCallback(args.stop_after_steps)],
        resolved_args=args,
    )
    try:
        trainer.train()
    finally:
        model.remove_training_hooks()

    if args.skip_final_save:
        return

    merger_state = trainer.gather_merger_state()
    if trainer.is_world_process_zero():
        checkpoint_path = Path(args.output_dir) / args.checkpoint_name
        _save_merger_artifact(
            checkpoint_path,
            merger_state,
            model,
            args,
        )
        print(
            f"Saved Qwen VisionZip visual.merger checkpoint to "
            f"{checkpoint_path}. Pass this file as "
            "compressor_args.checkpoint_path.",
            flush=True,
        )
