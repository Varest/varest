"""Training entry points for :class:`HiDropCompressor` (``tr`` and ``ft``).

HiDrop is a **trained** method. The schedule is applied during training, not
bolted on afterwards, so the model learns to fuse from a visual stream that is
injected late, pruned in the middle, and gone before layer 25. Running the
schedule on a stock LLaVA checkpoint is an ablation, not the paper's setting.

Training is the released two-stage LLaVA-1.5 recipe, run under HiDrop:

* ``tr`` -- **projector pretraining** on the 558K BLIP/LAION/CC/SBU captions with
  ``plain`` conversations and square/crop images. Everything is frozen except
  ``multi_modal_projector``, which is re-initialised first so each budget gets its
  own projector. DeepSpeed ZeRO-2, global batch 256, lr 1e-3. Writes
  ``mm_projector.bin``.
* ``ft`` -- **full finetuning** on the 665K instruction mixture with ``vicuna_v1``
  conversations and padded images, starting from *that budget's* projector. The
  vision tower stays frozen; the projector and the language model train. DeepSpeed
  ZeRO-3, global batch 128, lr 2e-5. Writes a full HF LLaVA checkpoint, which is
  what ``compressor_args.checkpoint_path`` then points at for benchmarking.

The stages chain by budget: a 64-token finetune must start from the 64-token
projector. ``provenance.py`` enforces that with run manifests instead of trusting
the caller to keep directories straight.

Launching
---------
Training is FlashAttention-2 only and multi-GPU. Launch the harness under a
distributed launcher; the trainer picks up ``WORLD_SIZE``/``RANK`` from it::

    torchrun --nproc_per_node=8 -m llmoe --config local \\
      --compressor HiDrop --custom_args hidrop_pretrain.yaml \\
      --input_dir /datasets --output_dir /results tr

    torchrun --nproc_per_node=8 -m llmoe --config local \\
      --compressor HiDrop --custom_args hidrop_finetune.yaml \\
      --input_dir /datasets --output_dir /results ft

``hidrop_pretrain.yaml``::

    compressor_args:
      average_retained_tokens: 64
    training_args:
      per_device_train_batch_size: 8      # x world size x accumulation == 256

``hidrop_finetune.yaml``::

    compressor_args:
      average_retained_tokens: 64
    training_args:
      per_device_train_batch_size: 4      # x world size x accumulation == 128
      # defaults to the matching tr output; set it if the projector lives elsewhere
      # pretrain_mm_mlp_adapter: /results/hidrop/llava-v1.5-7b-hidrop-64-flash_attention_2-pretrain

Datasets are read from ``<input_dir>/LLaVA-Pretrain`` and
``<input_dir>/LLaVA-Tuning`` (see ``data.py``); ``training_args.data_path`` and
``training_args.image_folder`` override that.

What is fixed, and why
----------------------
Only ``per_device_train_batch_size`` and ``gradient_accumulation_steps`` are meant
to be tuned -- their product with the world size must equal the stage's global
batch, and the trainer refuses anything else. The rest of the recipe is pinned
because it changes what the weights mean: ``adamw_torch``; the Transformers 4.37
shifted-CE loss averaged per micro-batch (Transformers 5 token-count and
cross-device rescaling disabled); a 2048-token context; the seed set before the
projector is initialised; the slow Vicuna tokenizer right-padded with ``<unk>``;
and the released modality-length sampler, which uses each row exactly once.

Faithfulness notes / harness adaptations
----------------------------------------
* **No launcher scripts.** The sidecar drove this through Slurm/PBS job scripts
  that exported paths and hyperparameters as environment variables. Those are not
  ported: the same knobs are ``training_args``, and the launcher is whatever
  ``torchrun``/``deepspeed`` invocation the site uses. Erbium monitoring is left
  to the harness's own ``--smoke_test`` flag.

* **The trainer builds its own base model.** ``run_training`` hands over a
  :class:`~llmoe.backbone.prototype.VLM`, but its model is configured for
  *evaluation* -- fp16, sdpa, left-padded fast tokenizer, always-padded images.
  Training needs bf16 + FA2, a right-padded slow tokenizer with ``<unk>`` padding,
  and stage-specific image preprocessing. The backbone is lazy, so nothing has
  been loaded yet: the trainer takes its ``model_name_or_path`` and
  ``local_files_only`` and loads the training stack itself, leaving the eval
  backbone untouched.

* **Projector discovery.** ``ft`` defaults ``pretrain_mm_mlp_adapter`` to the
  matching budget's ``tr`` output under ``<output_dir>/hidrop/``, which is where
  ``tr`` writes it -- the sidecar's launcher did the same lookup in shell.

Ported from the sidecar's ``train/train_hidrop.py``.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import json
import math
import os
import time
import warnings

import torch
from mle.vars import ExpConfig
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    LlavaForConditionalGeneration,
    LlavaImageProcessor,
    LlavaProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

from llmoe.backbone import VLM
from llmoe.utils.depth import config_decoder_depth
from llmoe.compressor.hidrop.hidrop import (
    BACKENDS,
    HiDropBackend,
    HiDropCompressor,
    backend_for_training,
    detect_backend,
    training_compressor_args,
)
from llmoe.compressor.hidrop.config import CONFIG_METADATA_KEYS, HiDropConfig
from llmoe.compressor.hidrop.data import (
    LlavaHiDropCollator,
    LlavaJsonDataset,
    dataset_dir_for_stage,
    find_annotations,
    resolve_training_data,
)
from llmoe.compressor.hidrop.hidrop_llama import HiDropLlamaModel
from llmoe.compressor.hidrop.provenance import (
    RELEASED_LOSS_SEMANTICS,
    RELEASED_TRAINING_OPTIMIZER,
    build_run_manifest,
    record_output_file,
    validate_artifact_provenance,
    validate_fixed_training_environment,
    write_run_manifest,
)
from llmoe.config import Config


TRAINING_OPTIMIZER = RELEASED_TRAINING_OPTIMIZER
LOSS_SEMANTICS = RELEASED_LOSS_SEMANTICS
RELEASED_MODEL_MAX_LENGTH = 2048
#: Qwen's context for these adaptations. The released LLaVA recipe pins 2048 and
#: that pin stands; upstream publishes no Qwen recipe, and 2048 cannot hold both
#: a 2048-token image (DEFAULT_QWEN_IMAGE_TOKENS) and its text.
QWEN_MODEL_MAX_LENGTH = 4096
# Transformers 5 can rescale the loss by token counts gathered across devices.
# Released LLaVA does not; keeping it off preserves the recipe's effective LR.
CROSS_DEVICE_TOKEN_AVERAGING = False
ATTN_IMPLEMENTATION = "flash_attention_2"
# Released LLaVA-1.5 stage hyperparameters.
RELEASED_STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "pretrain": {
        "learning_rate": 1e-3,
        "global_batch_size": 256,
        "deepspeed_stage": 2,
        "group_by_modality_length": False,
    },
    "finetune": {
        "learning_rate": 2e-5,
        "global_batch_size": 128,
        "deepspeed_stage": 3,
        "group_by_modality_length": True,
    },
}


def _released_causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Transformers 4.37 LLaMA loss: shifted CE mean per micro-batch."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    return nn.CrossEntropyLoss()(shift_logits.view(-1, int(vocab_size)), shift_labels.view(-1))


def _projector_state_from_full_state_dict(state_dict: dict[str, torch.Tensor],
                                          module_path: str = "model.multi_modal_projector"
                                          ) -> dict[str, torch.Tensor]:
    """Pull one submodule's tensors out of a gathered full state dict.

    ``module_path`` is the backend's stage-1 module (LLaVA's ``model.multi_modal_projector``
    or Qwen's ``model.visual.merger``), under the training wrapper's ``base_model.`` prefix.
    """
    marker = f"base_model.{module_path}."
    return {
        key.split(marker, 1)[1]: value.detach().cpu()
        for key, value in state_dict.items()
        if marker in key
    }


@dataclass
class HiDropTrainingArgs:
    """Resolved training settings; serialized into the run manifest verbatim."""

    model_name_or_path: str
    data_path: str
    image_folder: str
    output_dir: str
    stage: str
    # Which backbone's recipe ran, and its budget as a filesystem-safe label. The budget
    # itself is spelled differently per backbone (see the optional fields below): LLaVA
    # publishes an absolute layer-average out of 576, Qwen a layer-averaged ratio.
    backbone: str
    budget_tag: str
    late_entry_layer: int
    early_exit_layer: int
    layer_list: tuple[int, ...]
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
    resource_report_path: str | None
    log_memory_every_n_steps: int
    # LLaVA-1.5 spelling of the budget: absolute counts out of 576 visual tokens.
    hidrop_average_retained_tokens: int | None = None
    image_token_keep_list: tuple[int, ...] = ()
    # Qwen2.5-VL spelling: fractions of each image's own visual-token span.
    hidrop_retain_ratio: float | None = None
    image_token_keep_ratio_list: tuple[float, ...] = ()
    requested_attn_implementation: str | None = None
    effective_text_attn_implementation: str | None = None
    effective_vision_attn_implementation: str | None = None


class HiDropTrainingModel(nn.Module):
    """VLM wrapper whose loss uses the labels that survived pruning.

    Token removal deletes label positions along with their tokens, so the causal loss
    cannot be computed against the labels that went in. The HiDrop decoder returns the
    updated labels and they are used here instead.

    Backbone-independent: everything that differs between LLaVA-1.5 and Qwen2.5-VL --
    how image features are produced, which decoder wrapper is installed, where the
    stage-1 module lives -- comes from ``backend`` (see `backend.py`). The forward is the
    same for both because both backbones expose per-image feature rows through
    ``get_image_features(...).pooler_output``.
    """

    def __init__(self, base_model: Any, hidrop_config: Any,
                 backend: HiDropBackend | None = None, processor: Any = None) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.generation_config = base_model.generation_config
        self.hidrop_config = hidrop_config
        self.backend: HiDropBackend = backend or detect_backend(base_model)
        self.compressor = HiDropCompressor.for_training(hidrop_config)
        self.compressor._image_token_id = self.backend.image_token_id(base_model, processor)
        original_language_model = base_model.model.language_model
        object.__setattr__(self, "_original_language_model", original_language_model)
        base_model.model.language_model = self.backend.build_decoder_wrapper(
            original_language_model, self.compressor
        )

    @property
    def model(self) -> nn.Module:
        return self.base_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.base_model.lm_head

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        if gradient_checkpointing_kwargs is None:
            self.base_model.gradient_checkpointing_enable()
        else:
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.base_model.model.language_model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.base_model.gradient_checkpointing_disable()
        self.base_model.model.language_model.gradient_checkpointing = False

    def forward(self, input_ids: torch.LongTensor | None = None, pixel_values: torch.FloatTensor | None = None,
                attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None,
                past_key_values: Any | None = None, inputs_embeds: torch.FloatTensor | None = None,
                vision_feature_layer: int | list[int] | None = None,
                vision_feature_select_strategy: str | None = None, labels: torch.LongTensor | None = None,
                logits_to_keep: int | torch.Tensor = 0, image_sizes: torch.Tensor | None = None,
                image_grid_thw: torch.LongTensor | None = None,
                **kwargs: Any) -> LlavaCausalLMOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        image_features = None
        if pixel_values is not None:
            feature_rows = self.backend.image_feature_rows(
                self.base_model,
                {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )
            if len(feature_rows) != inputs_embeds.shape[0]:
                raise ValueError(
                    "HiDrop training requires one real or dummy image feature row per text example; "
                    f"got {len(feature_rows)} feature rows for batch size {inputs_embeds.shape[0]}."
                )
            if input_ids is None:
                raise ValueError(
                    "HiDrop dummy-image graph handling requires input_ids when pixel_values are provided."
                )
            image_token_id = int(self.compressor._image_token_id)
            merged_rows = []
            normalized_features = []
            for batch_index, row_features in enumerate(feature_rows):
                row_features = row_features.to(inputs_embeds.device, inputs_embeds.dtype)
                normalized_features.append(row_features)
                row_embeds = inputs_embeds[batch_index]
                row_mask = input_ids[batch_index] == image_token_id
                image_token_count = int(row_mask.sum().item())
                if image_token_count:
                    if image_token_count != int(row_features.shape[0]):
                        raise ValueError(
                            "Image features and image placeholders do not match for HiDrop training row "
                            f"{batch_index}: tokens={image_token_count}, features={row_features.shape[0]}."
                        )
                    expanded_mask = row_mask.unsqueeze(-1).expand_as(row_embeds)
                    row_embeds = row_embeds.masked_scatter(expanded_mask, row_features)
                else:
                    # Match native LLaVA's cur_image_features[0:0] path. This
                    # contributes exactly zero while retaining the vision and
                    # projector modules in the distributed autograd graph.
                    row_embeds = row_embeds + row_features[0:0].sum()
                merged_rows.append(row_embeds)
            inputs_embeds = torch.stack(merged_rows, dim=0)
            image_features = torch.cat(normalized_features, dim=0)
        self.compressor._latest_input_ids = input_ids.detach() if torch.is_tensor(input_ids) else None
        # Transformers 5 passes this kwarg to models that accept **kwargs and
        # changes loss normalization across gradient accumulation. Released
        # LLaVA/Transformers 4.37 computes a mean CE loss per micro-batch.
        kwargs.pop("num_items_in_batch", None)
        requested_use_cache = kwargs.pop("use_cache", None)
        outputs = self.model.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False if self.training or labels is not None else requested_use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        updated_labels = outputs.updated_labels if getattr(outputs, "updated_labels", None) is not None else labels
        if labels is not None:
            logits = self.lm_head(hidden_states).float()
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if updated_labels is not None:
            loss = _released_causal_lm_loss(logits, updated_labels, int(self.config.text_config.vocab_size))
        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )

    def save_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        """Save the plain HF model plus its HiDrop schedule.

        The decoder wrapper is swapped out first so the checkpoint reloads with
        stock ``transformers``; the schedule travels beside it in
        ``hidrop_config.json``.
        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        state_dict = kwargs.pop("state_dict", None)
        base_state_dict = None
        if state_dict is not None:
            base_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("base_model."):
                    base_state_dict[key.removeprefix("base_model.")] = value
        wrapped = self.base_model.model.language_model
        self.base_model.model.language_model = self._original_language_model
        try:
            self.base_model.save_pretrained(output, state_dict=base_state_dict, **kwargs)
        finally:
            self.base_model.model.language_model = wrapped
        (output / "hidrop_config.json").write_text(
            json.dumps(self.backend.schedule_metadata(self.hidrop_config), indent=2),
            encoding="utf-8",
        )


class HiDropTrainingLlavaForConditionalGeneration(HiDropTrainingModel):
    """LLaVA-1.5 training wrapper.

    Retained as a name because it is part of this package's public surface; the behavior
    now lives in the backbone-independent :class:`HiDropTrainingModel`, with the LLaVA
    backend bound here.
    """

    def __init__(self, base_model: LlavaForConditionalGeneration, hidrop_config: HiDropConfig,
                 processor: Any = None) -> None:
        super().__init__(base_model, hidrop_config, backend=BACKENDS["llava"], processor=processor)


def _parameter_group_for_name(name: str) -> str:
    # Qwen's merger lives inside the vision tower, so it is matched before it.
    if ".visual.merger." in name or ".multi_modal_projector." in name or "mm_projector" in name:
        return "multi_modal_projector"
    if ".vision_tower." in name or name.startswith("base_model.model.vision_tower.") \
            or ".visual." in name:
        return "vision_tower"
    if ".language_model." in name or ".lm_head." in name:
        return "language_model"
    return "other"


def _report_trainable_parameters(model: nn.Module) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    total = 0
    trainable = 0
    groups: dict[str, dict[str, int]] = {}
    for name, parameter in model.named_parameters():
        numel = int(parameter.numel())
        group = _parameter_group_for_name(name)
        group_counts = groups.setdefault(group, {"total": 0, "trainable": 0})
        group_counts["total"] += numel
        total += numel
        if parameter.requires_grad:
            group_counts["trainable"] += numel
            trainable += numel
    ratio = 100.0 * float(trainable) / float(total) if total else 0.0
    print(f"[HIDROP][PARAMS] trainable={trainable:,} total={total:,} ratio={ratio:.4f}%")
    for group in sorted(groups):
        group_total = groups[group]["total"]
        group_trainable = groups[group]["trainable"]
        group_ratio = 100.0 * float(group_trainable) / float(group_total) if group_total else 0.0
        print(
            f"[HIDROP][PARAMS] {group}: "
            f"trainable={group_trainable:,} total={group_total:,} ratio={group_ratio:.4f}%"
        )


def _required_deepspeed_stage(training_stage: str) -> int:
    if training_stage == "pretrain":
        return 2
    if training_stage == "finetune":
        return 3
    raise ValueError(f"Unsupported training stage {training_stage!r}; choose pretrain or finetune.")


def _required_global_batch_size(training_stage: str) -> int:
    if training_stage == "pretrain":
        return 256
    if training_stage == "finetune":
        return 128
    raise ValueError(f"Unsupported training stage {training_stage!r}; choose pretrain or finetune.")


def _split_to_even_chunks(indices: list[int], lengths: Sequence[int], num_chunks: int) -> list[list[int]]:
    if len(indices) % num_chunks != 0:
        return [indices[idx::num_chunks] for idx in range(num_chunks)]
    num_indices_per_chunk = len(indices) // num_chunks
    chunks: list[list[int]] = [[] for _ in range(num_chunks)]
    chunk_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = int(chunk_lengths.index(min(chunk_lengths)))
        chunks[shortest_chunk].append(index)
        chunk_lengths[shortest_chunk] += int(lengths[index])
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunk_lengths[shortest_chunk] = int(1e12)
    return chunks


class ModalityLengthGroupedSampler(Sampler[int]):
    """Released LLaVA modality/length order without sample repetition."""

    def __init__(self, modality_lengths: Sequence[int], *, per_device_batch_size: int, world_size: int,
                 gradient_accumulation_steps: int, seed: int) -> None:
        self.modality_lengths = [int(value) for value in modality_lengths]
        if any(length == 0 for length in self.modality_lengths):
            raise ValueError(
                "modality_lengths entries must be non-zero; use positive lengths for image rows and "
                "negative lengths for text rows."
            )
        self.lengths = [abs(length) for length in self.modality_lengths]
        self.per_device_batch_size = max(1, int(per_device_batch_size))
        self.world_size = max(1, int(world_size))
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.length_group_world_size = self.world_size * self.gradient_accumulation_steps
        self.megabatch_size = self.per_device_batch_size * self.length_group_world_size
        self.seed = int(seed)
        self.epoch = 0
        self.image_indices = [idx for idx, length in enumerate(self.modality_lengths) if length > 0]
        self.text_indices = [idx for idx, length in enumerate(self.modality_lengths) if length < 0]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.modality_lengths)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if not self.modality_lengths:
            return iter(())
        if not self.image_indices or not self.text_indices:
            return iter(self._length_grouped_order(list(range(len(self.modality_lengths))), generator))

        image_order = self._length_grouped_order(self.image_indices, generator)
        text_order = self._length_grouped_order(self.text_indices, generator)
        image_megabatches = [
            image_order[index:index + self.megabatch_size]
            for index in range(0, len(image_order), self.megabatch_size)
        ]
        text_megabatches = [
            text_order[index:index + self.megabatch_size]
            for index in range(0, len(text_order), self.megabatch_size)
        ]
        final_megabatch = image_megabatches[-1] + text_megabatches[-1]
        megabatches = image_megabatches[:-1] + text_megabatches[:-1]
        if megabatches:
            order = torch.randperm(len(megabatches), generator=generator).tolist()
            megabatches = [megabatches[index] for index in order]
        if final_megabatch:
            # Match upstream LLaVA: combine both modality remainders once,
            # without padding or repeating examples.
            megabatches.append(sorted(final_megabatch))
        return iter(index for megabatch in megabatches for index in megabatch)

    def _length_grouped_order(self, indices: list[int], generator: torch.Generator) -> list[int]:
        if not indices:
            return []
        shuffled = [indices[pos] for pos in torch.randperm(len(indices), generator=generator).tolist()]
        grouped_order = []
        for idx in range(0, len(shuffled), self.megabatch_size):
            megabatch = shuffled[idx: idx + self.megabatch_size]
            megabatch = sorted(megabatch, key=lambda index: self.lengths[index], reverse=True)
            balanced_chunks = _split_to_even_chunks(megabatch, self.lengths, self.length_group_world_size)
            grouped_order.extend(index for chunk in balanced_chunks for index in chunk)
        return grouped_order


class HiDropTrainer(Trainer):
    def __init__(self, *args: Any, group_by_modality_length: bool = False,
                 run_manifest: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.group_by_modality_length = bool(group_by_modality_length)
        self.run_manifest = run_manifest
        # The wrapper accepts **kwargs for HF compatibility, but intentionally
        # uses the released per-microbatch mean CE loss. Prevent Transformers 5
        # from injecting num_items_in_batch and changing accumulation scaling.
        self.model_accepts_loss_kwargs = False

    def _training_backend(self) -> Any:
        """The backend describing this backbone's stage-1 module and schedule.

        Everything backbone-specific in `_save` comes from here. The trainer used
        to reach past it into LLaVA-shaped attributes, so each Qwen run died at
        its first checkpoint on whichever one it touched first.
        """
        backend = getattr(self.model, "backend", None)
        if backend is None:
            backend = getattr(self, "hidrop_backend", None)
        return backend

    def _schedule_metadata(self) -> dict[str, Any]:
        config = self.model.hidrop_config
        backend = self._training_backend()
        if backend is not None:
            return backend.schedule_metadata(config)
        # LLaVA's own config carries it; kept as the fallback for a bare trainer.
        return config.hidrop_metadata()

    def _stage1_artifact_name(self) -> str:
        backend = self._training_backend()
        return str(getattr(backend, "stage1_artifact_name", None)
                   or "mm_projector.bin")

    def _save(self, output_dir: str | None = None, state_dict: dict[str, torch.Tensor] | None = None) -> None:
        destination = Path(output_dir or self.args.output_dir)
        stage = self.run_manifest.get("run", {}).get("stage") if self.run_manifest is not None else None
        if stage == "pretrain":
            # Match released LLaVA adapter tuning: do not emit a redundant
            # consolidated 7B model at every step. Trainer saves the
            # DeepSpeed model/optimizer shards needed for exact resume after
            # this hook; this directory only needs the trained projector and
            # compatibility metadata.
            if state_dict is None:
                state_dict = self.model.state_dict()
            # The stage-1 module is named by the backend: LLaVA's
            # `model.multi_modal_projector`, Qwen's `model.visual.merger`. Taking
            # the default here looked for LLaVA's on every backbone, so a Qwen
            # pretrain run died at its first save.
            backend = self._training_backend()
            module_path = (getattr(self, "projector_module_path", None)
                           or getattr(backend, "stage1_module_path", None)
                           or "model.multi_modal_projector")
            projector_state = _projector_state_from_full_state_dict(state_dict, module_path)
            if not projector_state:
                raise RuntimeError(
                    f"Could not find stage-1 module {module_path!r} in the pretrain "
                    f"checkpoint state."
                )
            destination.mkdir(parents=True, exist_ok=True)
            # Qwen's stage-1 artifact is mm_merger.bin, LLaVA's mm_projector.bin;
            # writing the LLaVA name on Qwen would leave `ft` unable to find it.
            projector_path = destination / self._stage1_artifact_name()
            torch.save(projector_state, projector_path)
            metadata = self._schedule_metadata()
            if metadata.get("backbone_family") == "llava_next":
                metadata["training_stage"] = "pretrain_compression_adaptation"
            (destination / "hidrop_config.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            if self.run_manifest is not None:
                record_output_file(self.run_manifest, "mm_projector", projector_path)
        elif getattr(self.model.hidrop_config, "backbone_family", None) == "llava_next":
            # The NeXT wrapper owns a native HF model. Saving through it strips
            # wrapper prefixes so LlavaNextForConditionalGeneration can reload
            # every step checkpoint directly.
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(
                destination,
                state_dict=state_dict,
                safe_serialization=bool(getattr(self.args, "save_safetensors", True)),
            )
            processor = getattr(self.data_collator, "processor", None)
            if processor is None or not callable(getattr(processor, "save_pretrained", None)):
                raise RuntimeError("LLaVA-NeXT HiDrop checkpoints require the training processor.")
            processor.save_pretrained(destination)
            torch.save(self.args, destination / "training_args.bin")
            metadata = self._schedule_metadata()
            metadata["training_stage"] = "finetune"
            (destination / "hidrop_config.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
        else:
            super()._save(output_dir=str(destination), state_dict=state_dict)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "hidrop_config.json").write_text(
                json.dumps(self._schedule_metadata(), indent=2),
                encoding="utf-8",
            )
        if self.run_manifest is not None and self.is_world_process_zero():
            (destination / "training_args_hidrop.json").write_text(
                json.dumps(self.run_manifest["run"], indent=2, default=str),
                encoding="utf-8",
            )
            write_run_manifest(destination, self.run_manifest)

    def _init_training_state(self, max_steps: int, num_update_steps_per_epoch: int, num_train_epochs: int,
                             resume_from_checkpoint: str | None, trial: Any):
        result = super()._init_training_state(
            max_steps, num_update_steps_per_epoch, num_train_epochs, resume_from_checkpoint, trial)
        previous_save_steps = int(getattr(self.state, "save_steps", 0) or 0)
        self.state.compute_steps(self.args, max_steps)
        current_save_steps = int(getattr(self.state, "save_steps", 0) or 0)
        if (
            resume_from_checkpoint is not None
            and previous_save_steps != current_save_steps
            and int(os.environ.get("RANK", "0")) == 0
        ):
            print(
                f"[HIDROP][RESUME] overriding checkpoint save_steps={previous_save_steps} "
                f"with current save_steps={current_save_steps}"
            )
        return result

    def _get_train_sampler(self, train_dataset: Any | None = None):
        if not self.group_by_modality_length:
            if train_dataset is not None:
                try:
                    return super()._get_train_sampler(train_dataset)
                except TypeError:
                    return super()._get_train_sampler()
            return super()._get_train_sampler()
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        modality_lengths = getattr(dataset, "modality_lengths", None)
        if modality_lengths is None:
            rows = getattr(dataset, "rows", None)
            if rows is not None:
                modality_lengths = []
                for row in rows:
                    conversations = row.get("conversations") or []
                    length = max(1, sum(
                        len(str(turn.get("value") or turn.get("content") or "").split())
                        for turn in conversations
                    ))
                    modality_lengths.append(length if bool(row.get("image")) else -length)
        if modality_lengths is None:
            raise ValueError("group_by_modality_length requires a dataset with modality_lengths metadata.")
        world_size = int(os.environ.get("WORLD_SIZE") or getattr(self.args, "world_size", 1) or 1)
        sampler = ModalityLengthGroupedSampler(
            modality_lengths,
            per_device_batch_size=int(self.args.per_device_train_batch_size),
            world_size=world_size,
            gradient_accumulation_steps=int(self.args.gradient_accumulation_steps),
            seed=int(getattr(self.args, "seed", 42)),
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[HIDROP][SAMPLER] modality length grouping enabled "
                f"world_size={sampler.world_size} "
                f"gradient_accumulation_steps={sampler.gradient_accumulation_steps} "
                f"per_device_batch_size={sampler.per_device_batch_size} "
                f"megabatch_size={sampler.megabatch_size} "
                f"image_rows={len(sampler.image_indices)} "
                f"text_rows={len(sampler.text_indices)} "
                f"samples_without_repetition={len(sampler)}"
            )
        return sampler


class HiDropSaveControlCallback(TrainerCallback):
    """Keep step checkpoints on the configured cadence across a resume."""

    def _sync_save_steps(self, args: TrainingArguments, state: Any) -> int:
        save_steps = int(getattr(args, "save_steps", 0) or 0)
        if save_steps > 0 and int(getattr(state, "save_steps", 0) or 0) != save_steps:
            state.save_steps = save_steps
        return int(getattr(state, "save_steps", 0) or 0)

    def on_train_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        self._sync_save_steps(args, state)
        return control

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        save_steps = self._sync_save_steps(args, state)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        global_step = int(getattr(state, "global_step", 0) or 0)
        if (
            bool(getattr(control, "should_save", False))
            and max_steps > 0
            and global_step >= max_steps
            and save_steps > 0
            and global_step % save_steps != 0
        ):
            control.should_save = False
        return control


class StopAfterStepCallback(TrainerCallback):
    """Bounded smoke/capacity runs: stop early without disturbing the schedule."""

    def __init__(self, stop_after_steps: int | None) -> None:
        self.stop_after_steps = int(stop_after_steps) if stop_after_steps is not None else None

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.stop_after_steps is None:
            return control
        global_step = int(getattr(state, "global_step", 0) or 0)
        if global_step >= self.stop_after_steps:
            control.should_training_stop = True
            save_steps = int(getattr(args, "save_steps", 0) or 0)
            if save_steps > 0 and global_step % save_steps == 0:
                control.should_save = True
        return control


class ResourceReportCallback(TrainerCallback):
    """Step timing and peak memory, so a budget's cost is measured, not guessed."""

    def __init__(self, *, stage: str, budget: str, requested_attn_implementation: str,
                 effective_text_attn_implementation: str | None,
                 effective_vision_attn_implementation: str | None, global_batch_size: int,
                 per_device_train_batch_size: int, gradient_accumulation_steps: int,
                 dataloader_num_workers: int, model_max_length: int, dataset_length: int, max_steps: int,
                 stop_after_steps: int | None, resource_report_path: str | None,
                 log_memory_every_n_steps: int) -> None:
        self.stage = stage
        # A ratio-parameterised backbone reports a label like "r0.25", not an int;
        # this only ever lands in the resource report, so keep it as given.
        self.budget = str(budget)
        self.requested_attn_implementation = requested_attn_implementation
        self.effective_text_attn_implementation = effective_text_attn_implementation
        self.effective_vision_attn_implementation = effective_vision_attn_implementation
        self.global_batch_size = int(global_batch_size)
        self.per_device_train_batch_size = int(per_device_train_batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.dataloader_num_workers = int(dataloader_num_workers)
        self.model_max_length = int(model_max_length)
        self.dataset_length = int(dataset_length)
        self.max_steps = int(max_steps)
        self.stop_after_steps = int(stop_after_steps) if stop_after_steps is not None else None
        self.resource_report_path = resource_report_path
        self.log_memory_every_n_steps = int(log_memory_every_n_steps)
        self._step_start: float | None = None
        self.step_seconds: list[float] = []

    def on_train_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        if self._step_start is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._step_start
        self.step_seconds.append(float(elapsed))
        if (
            self.log_memory_every_n_steps > 0
            and state.global_step % self.log_memory_every_n_steps == 0
            and torch.cuda.is_available()
            and bool(getattr(state, "is_world_process_zero", True))
        ):
            allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(
                f"[HIDROP][RESOURCE] step={state.global_step} "
                f"max_allocated_gb={allocated:.2f} max_reserved_gb={reserved:.2f}"
            )

    def build_report(self) -> dict[str, Any]:
        warmup = min(2, len(self.step_seconds))
        measured = self.step_seconds[warmup:] or self.step_seconds
        avg_seconds = float(sum(measured) / len(measured)) if measured else 0.0
        steps_per_epoch = int(math.ceil(float(self.dataset_length) / float(self.global_batch_size))) \
            if self.global_batch_size > 0 else 0
        allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        return {
            "stage": self.stage,
            "budget": self.budget,
            "requested_attn_implementation": self.requested_attn_implementation,
            "effective_text_attn_implementation": self.effective_text_attn_implementation,
            "effective_vision_attn_implementation": self.effective_vision_attn_implementation,
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "global_batch_size": self.global_batch_size,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "dataloader_num_workers": self.dataloader_num_workers,
            "model_max_length": self.model_max_length,
            "max_steps": self.max_steps if self.max_steps > 0 else None,
            "stop_after_steps": self.stop_after_steps,
            "dataset_length": self.dataset_length,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "measured_seconds_per_optimizer_step": avg_seconds,
            "estimated_hours_per_epoch": float(steps_per_epoch * avg_seconds / 3600.0),
            "cuda_max_memory_allocated_gb": float(allocated),
            "cuda_max_memory_reserved_gb": float(reserved),
        }

    def write_report(self) -> None:
        if not self.resource_report_path:
            return
        path = Path(self.resource_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.build_report(), indent=2), encoding="utf-8")


def _deepspeed_config(stage: int, global_batch_size: int, per_device_train_batch_size: int,
                      gradient_accumulation_steps: int) -> dict[str, Any]:
    if int(stage) not in (2, 3):
        raise ValueError(f"HiDrop training only supports DeepSpeed ZeRO-2 or ZeRO-3, got ZeRO-{stage}.")
    zero_optimization: dict[str, Any] = {"stage": int(stage), "overlap_comm": True, "contiguous_gradients": True}
    if int(stage) == 3:
        zero_optimization["stage3_gather_16bit_weights_on_model_save"] = True
    return {
        "train_batch_size": int(global_batch_size),
        "train_micro_batch_size_per_gpu": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "bf16": {"enabled": "auto"},
        "zero_optimization": zero_optimization,
    }


def _gather_final_state_dict(trainer: Trainer) -> dict[str, torch.Tensor]:
    """Gather weights through the distributed wrapper prepared by Accelerate."""
    wrapped = getattr(trainer, "model_wrapped", None)
    if wrapped is None:
        raise RuntimeError("Trainer did not expose the prepared model wrapper for final serialization.")
    return trainer.accelerator.get_state_dict(wrapped)


def _effective_attn_implementations(base_model: Any, backend: HiDropBackend | None = None
                                    ) -> tuple[str | None, str | None]:
    """The attention backend each tower actually selected, for the run manifest."""
    backend = backend or BACKENDS["llava"]
    tower = backend.vision_tower(base_model)
    return (
        getattr(base_model.model.language_model.config, "_attn_implementation", None),
        getattr(getattr(tower, "config", None), "_attn_implementation", None),
    )


def _hidrop_semantic_signature(config: HiDropConfig) -> tuple[Any, ...]:
    return (
        config.implementation_version,
        config.average_retained_tokens,
        config.late_entry_layer,
        config.early_exit_layer,
        config.layer_list,
        config.image_token_keep_list,
        config.visual_tokens,
        config.num_decoder_layers,
        config.dtopk_lambda,
        config.dtopk_normalize,
        config.dtopk_mask_threshold,
        config.backbone_family,
        config.backbone_model_id,
        config.backbone_revision,
        config.image_grid_pinpoints,
        config.vision_image_size,
        config.vision_patch_size,
        config.vision_feature_select_strategy,
    )


def validate_hidrop_artifact(artifact_path: str | Path, expected_config: Any, *,
                             artifact_name: str, backend: HiDropBackend | None = None) -> Any:
    """Reject an adapter/checkpoint trained under a different HiDrop schedule.

    The stages chain by budget -- a finetune must start from *its own* budget's stage-1
    adapter -- and nothing in a bare ``state_dict`` records which schedule produced it, so
    the schedule travels alongside in ``hidrop_config.json`` and is compared here. LLaVA
    keeps the released implementation-version gate; Qwen compares the resolved schedule
    itself, since it has no released version line to check against.
    """
    backend = backend or BACKENDS["llava"]
    artifact = Path(artifact_path)
    metadata_dir = artifact if artifact.is_dir() else artifact.parent
    if backend.key == "llava":
        try:
            artifact_config = HiDropConfig.from_checkpoint(metadata_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"Cannot use {artifact_name} {artifact}: {exc}") from exc
        matches = _hidrop_semantic_signature(artifact_config) == _hidrop_semantic_signature(
            expected_config
        )
    else:
        metadata_path = metadata_dir / "hidrop_config.json"
        if not metadata_path.is_file():
            raise ValueError(
                f"Cannot use {artifact_name} {artifact}: no hidrop_config.json beside it, so the "
                "schedule it was trained under is unknown."
            )
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = backend.schedule_metadata(expected_config)
        keys = ("retain_ratio", "late_entry_layer", "early_exit_layer", "layer_list",
                "image_token_keep_ratio_list", "num_decoder_layers", "dtopk_lambda",
                "dtopk_normalize", "dtopk_mask_threshold")
        matches = all(recorded.get(key) == expected.get(key) for key in keys)
        artifact_config = recorded
    if not matches:
        raise ValueError(
            f"Cannot use {artifact_name} {artifact}: its HiDrop schedule/selector settings do not match "
            "the requested training run. Restart from a compatible artifact."
        )
    return artifact_config


def load_hf_llava_training_base(model_name_or_path: str, *, stage: str, dtype: torch.dtype | None,
                                attn_implementation: str, model_max_length: int = RELEASED_MODEL_MAX_LENGTH,
                                local_files_only: bool = True
                                ) -> tuple[LlavaProcessor, LlavaForConditionalGeneration]:
    """Load the pinned HF LLaVA base with the released *training* preprocessing."""
    if attn_implementation != ATTN_IMPLEMENTATION:
        raise ValueError(f"HiDrop training only supports flash_attention_2, got {attn_implementation!r}.")
    if stage not in {"pretrain", "finetune"}:
        raise ValueError(f"Unknown HiDrop training stage {stage!r}.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        model_max_length=int(model_max_length),
        padding_side="right",
        local_files_only=local_files_only,
    )
    if tokenizer.unk_token_id is None:
        raise ValueError("Faithful Vicuna-v1 training requires an <unk> token for right padding.")
    # Released LLaVA-1.5 v1 training pads with <unk>, not <eos>.
    tokenizer.pad_token = tokenizer.unk_token
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    if image_token_id is None or image_token_id < 0 or image_token_id == tokenizer.unk_token_id:
        raise ValueError("The Hugging Face LLaVA checkpoint must provide the canonical <image> token.")
    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    image_processor_config = image_processor.to_dict()
    image_processor_config.pop("do_pad", None)
    image_processor_config["image_processor_type"] = "LlavaImageProcessor"
    # The released recipe uses the default square/crop path for 558K
    # projector pretraining and --image_aspect_ratio pad for 665K finetuning.
    image_processor = LlavaImageProcessor(**image_processor_config, do_pad=stage == "finetune")
    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    if int(base_model.config.image_token_index) != int(image_token_id):
        raise ValueError(
            "Hugging Face LLaVA tokenizer/config image-token mismatch: "
            f"tokenizer={image_token_id}, config={base_model.config.image_token_index}."
        )
    base_model.config.pad_token_id = int(tokenizer.pad_token_id)
    base_model.config.text_config.pad_token_id = int(tokenizer.pad_token_id)
    base_model.model.language_model.config.pad_token_id = int(tokenizer.pad_token_id)
    vision_config = base_model.config.vision_config
    processor = LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=getattr(vision_config, "patch_size", 14),
        vision_feature_select_strategy="default",
        image_token="<image>",
    )
    base_model.config._attn_implementation = attn_implementation
    base_model.config.text_config._attn_implementation = attn_implementation
    base_model.config.vision_config._attn_implementation = attn_implementation
    if hasattr(base_model.config.text_config, "_attn_implementation_internal"):
        base_model.config.text_config._attn_implementation_internal = attn_implementation
    if hasattr(base_model.config.vision_config, "_attn_implementation_internal"):
        base_model.config.vision_config._attn_implementation_internal = attn_implementation
    base_model.model.language_model.config._attn_implementation = attn_implementation
    base_model.model.vision_tower.config._attn_implementation = attn_implementation
    if hasattr(base_model.model.language_model.config, "_attn_implementation_internal"):
        base_model.model.language_model.config._attn_implementation_internal = attn_implementation
    if hasattr(base_model.model.vision_tower.config, "_attn_implementation_internal"):
        base_model.model.vision_tower.config._attn_implementation_internal = attn_implementation
    text_attn_implementation, vision_attn_implementation = _effective_attn_implementations(base_model)
    if text_attn_implementation != attn_implementation:
        raise RuntimeError(
            f"Requested {attn_implementation!r} for the language model but Transformers selected "
            f"{text_attn_implementation!r}."
        )
    if vision_attn_implementation != attn_implementation:
        raise RuntimeError(
            f"Requested {attn_implementation!r} for the vision tower but Transformers selected "
            f"{vision_attn_implementation!r}."
        )
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Effective attention backends: text={text_attn_implementation!r} vision={vision_attn_implementation!r}")
    return processor, base_model


# --------------------------------------------------------------- harness wiring

def _dataset_check(config: ExpConfig, stage: str) -> str:
    custom_args = getattr(config, "custom_args", {}) or {}
    compressor_args = (
        custom_args.get("compressor_args", {}) if isinstance(custom_args, Mapping) else {}
    )
    family = (
        str(compressor_args.get("backbone_family", "llava")).replace("-", "_").lower()
        if isinstance(compressor_args, Mapping)
        else "llava"
    )
    if family == "llava_next":
        try:
            from llmoe.compressor.hidrop.hidrop_next_tr import resolve_llava_next_data_root

            training_args = (
                custom_args.get("training_args", {}) if isinstance(custom_args, Mapping) else {}
            )
            data_path = (
                training_args.get("data_path", config.input_dir)
                if isinstance(training_args, Mapping)
                else config.input_dir
            )
            root = resolve_llava_next_data_root(data_path)
        except (FileNotFoundError, ValueError) as exc:
            return str(exc)
        return f"OK with LLaVA-NeXT-Data Parquet shards under {root}"
    dataset_dir = dataset_dir_for_stage(config.input_dir, stage)
    name = dataset_dir.name
    if not dataset_dir.is_dir():
        return f"Missing {name} dataset under {config.input_dir}"
    try:
        find_annotations(dataset_dir, stage)
    except FileNotFoundError:
        return f"Missing {name} JSON annotations under {dataset_dir}"
    return f"OK with {name}"


def hidrop_tr_fn_cd(config: ExpConfig) -> str:
    return _dataset_check(config, "pretrain")


def hidrop_ft_fn_cd(config: ExpConfig) -> str:
    return _dataset_check(config, "finetune")


def _mapping_option(config: Config, key: str) -> dict[str, Any]:
    options = config.custom_args.get(key, {})
    if not isinstance(options, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return dict(options)


def _decoder_depth(backbone: VLM) -> int:
    """The decoder's layer count, read from the config without materialising weights.

    Both schedules are depth-relative -- Qwen's by construction, LLaVA's because a
    layer average divides by the depth -- and the schedule must be resolved once,
    *before* the training model loads, since the model is then built from it.
    Training LLaVA-1.5-13B against the 32-layer profile would otherwise produce a
    checkpoint whose recorded budget is not the one it was trained at.
    """
    return config_decoder_depth(backbone)


def _training_backend(backbone: VLM, config: Config) -> HiDropBackend:
    """Which backbone's recipe to run, decided before any weights load."""
    return backend_for_training(
        _mapping_option(config, "compressor_args"), str(backbone._model_id)
    )


def _hidrop_config_for_training(config: Config, backend: HiDropBackend | None = None,
                                num_decoder_layers: int | None = None) -> Any:
    """Build the schedule from ``compressor_args``.

    ``checkpoint_path`` is an inference-time knob (which trained model to load) and
    is ignored here: training *produces* checkpoints. Use
    ``training_args.pretrain_mm_mlp_adapter`` to chain ``tr`` into ``ft``.
    """
    per_device_train_batch_size = int(
        _mapping_option(config, "training_args").get("per_device_train_batch_size", 1)
    )
    backend = backend or BACKENDS["llava"]
    return backend.build_config(
        training_compressor_args(_mapping_option(config, "compressor_args")),
        batch_size_supported=per_device_train_batch_size > 1,
        num_decoder_layers=num_decoder_layers,
    )


def _stage_output_dir(config: Config, stage: str, hidrop_config: Any,
                      backend: HiDropBackend | None = None,
                      backbone: VLM | None = None) -> Path:
    backend = backend or BACKENDS["llava"]
    return backend.stage_output_dir(config.output_dir, stage, hidrop_config, backbone=backbone)


def _default_pretrain_adapter(config: Config, hidrop_config: Any,
                              backend: HiDropBackend | None = None,
                              backbone: VLM | None = None) -> Path:
    backend = backend or BACKENDS["llava"]
    return (_stage_output_dir(config, "pretrain", hidrop_config, backend, backbone)
            / backend.stage1_artifact_name)


def _resolve_adapter_path(value: str | Path, artifact_name: str = "mm_projector.bin") -> Path:
    candidate = Path(value)
    return candidate / artifact_name if candidate.is_dir() else candidate


def _build_training_args(backbone: VLM, config: Config, stage: str, hidrop_config: Any,
                         backend: HiDropBackend | None = None) -> HiDropTrainingArgs:
    """Resolve ``training_args`` into the released recipe, rejecting deviations."""
    options = _mapping_option(config, "training_args")
    defaults = RELEASED_STAGE_DEFAULTS[stage]
    backend = backend or BACKENDS["llava"]
    budget = backend.budget_tag(hidrop_config)

    per_device_train_batch_size = int(options.get("per_device_train_batch_size", 1))
    if per_device_train_batch_size < 1:
        raise ValueError(f"per_device_train_batch_size must be >= 1, got {per_device_train_batch_size}.")
    global_batch_size = int(options.get("global_batch_size", defaults["global_batch_size"]))
    expected_global_batch = _required_global_batch_size(stage)
    if global_batch_size != expected_global_batch:
        raise ValueError(
            f"HiDrop {stage} must preserve global_batch_size={expected_global_batch}; "
            f"got {global_batch_size}. Tune only micro-batch and gradient accumulation."
        )
    deepspeed_stage = int(options.get("deepspeed_stage", defaults["deepspeed_stage"]))
    expected_deepspeed_stage = _required_deepspeed_stage(stage)
    if deepspeed_stage != expected_deepspeed_stage:
        raise ValueError(
            f"HiDrop {stage} must use DeepSpeed ZeRO-{expected_deepspeed_stage}; got ZeRO-{deepspeed_stage}."
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    gradient_accumulation_steps = options.get("gradient_accumulation_steps")
    if gradient_accumulation_steps is None:
        micro_global = max(1, per_device_train_batch_size * world_size)
        gradient_accumulation_steps = max(1, math.ceil(global_batch_size / micro_global))
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    effective_global_batch = per_device_train_batch_size * world_size * gradient_accumulation_steps
    if effective_global_batch != global_batch_size:
        raise ValueError(
            "HiDrop requires an exact global batch product: "
            "per_device_train_batch_size * WORLD_SIZE * gradient_accumulation_steps "
            f"= {effective_global_batch}, requested global_batch_size={global_batch_size}."
        )

    released_length = (QWEN_MODEL_MAX_LENGTH if backend.key == "qwen"
                       else RELEASED_MODEL_MAX_LENGTH)
    model_max_length = int(options.get("model_max_length", released_length))
    if backend.key != "qwen" and model_max_length != RELEASED_MODEL_MAX_LENGTH:
        raise ValueError(
            "Faithful LLaVA-1.5/HiDrop training pins model_max_length="
            f"{RELEASED_MODEL_MAX_LENGTH}; got {model_max_length}."
        )
    if model_max_length < 1:
        raise ValueError(f"model_max_length must be >= 1, got {model_max_length}")
    attn_implementation = str(options.get("attn_implementation", ATTN_IMPLEMENTATION))
    if attn_implementation != ATTN_IMPLEMENTATION:
        raise ValueError(
            f"HiDrop training is FlashAttention-2 only; got attn_implementation={attn_implementation!r}."
        )
    save_steps = int(options.get("save_steps", 1000))
    if save_steps < 1:
        raise ValueError(f"save_steps must be >= 1, got {save_steps}.")
    save_total_limit = int(options.get("save_total_limit", 2))
    if save_total_limit < 1:
        raise ValueError(f"save_total_limit must be >= 1, got {save_total_limit}.")
    stop_after_steps = options.get("stop_after_steps")
    if stop_after_steps is not None:
        stop_after_steps = int(stop_after_steps)
        if stop_after_steps < 1:
            raise ValueError(f"stop_after_steps must be >= 1 when set, got {stop_after_steps}.")

    adapter = options.get("pretrain_mm_mlp_adapter")
    if stage == "pretrain":
        if adapter:
            raise ValueError("Stage-1 pretraining must start clean and cannot load pretrain_mm_mlp_adapter.")
        adapter_path = None
    else:
        adapter_path = (
            _resolve_adapter_path(adapter, backend.stage1_artifact_name) if adapter
            else _default_pretrain_adapter(config, hidrop_config, backend, backbone)
        )
        if not adapter_path.is_file():
            raise FileNotFoundError(
                f"Full finetuning requires the matching {budget} adapter from `tr`: {adapter_path} "
                "is missing. Run `tr` for this budget first, or set training_args.pretrain_mm_mlp_adapter."
            )

    data_path, image_folder = resolve_training_data(
        config.input_dir,
        stage,
        data_path=options.get("data_path"),
        image_folder=options.get("image_folder"),
    )
    output_dir = Path(options["output_dir"]) if options.get("output_dir") \
        else _stage_output_dir(config, stage, hidrop_config, backend, backbone)
    dataset_limit = config.dataset_limit
    if options.get("dataset_limit") is not None:
        option_limit = int(options["dataset_limit"])
        dataset_limit = option_limit if dataset_limit is None else min(dataset_limit, option_limit)
    resource_report_path = options.get("resource_report_path")
    if resource_report_path is None:
        resource_report_path = output_dir / f"resource_{stage}_{budget}-{attn_implementation}.json"

    return HiDropTrainingArgs(
        model_name_or_path=str(backbone._model_id),
        data_path=str(data_path),
        image_folder=str(image_folder),
        output_dir=str(output_dir),
        stage=stage,
        backbone=backend.key,
        budget_tag=budget,
        late_entry_layer=int(hidrop_config.late_entry_layer),
        early_exit_layer=int(hidrop_config.early_exit_layer),
        layer_list=tuple(hidrop_config.layer_list),
        hidrop_average_retained_tokens=(
            int(hidrop_config.average_retained_tokens)
            if hasattr(hidrop_config, "average_retained_tokens") else None
        ),
        image_token_keep_list=tuple(getattr(hidrop_config, "image_token_keep_list", ())),
        hidrop_retain_ratio=(
            float(hidrop_config.retain_ratio) if hasattr(hidrop_config, "retain_ratio") else None
        ),
        image_token_keep_ratio_list=tuple(
            getattr(hidrop_config, "image_token_keep_ratio_list", ())
        ),
        pretrain_mm_mlp_adapter=None if adapter_path is None else str(adapter_path),
        num_train_epochs=float(options.get("num_train_epochs", 1)),
        global_batch_size=global_batch_size,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
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
        group_by_modality_length=bool(
            options.get("group_by_modality_length", defaults["group_by_modality_length"])
        ),
        max_steps=int(options.get("max_steps", -1)),
        stop_after_steps=stop_after_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        seed=int(options.get("seed", 42)),
        resume_from_checkpoint=(
            str(options["resume_from_checkpoint"]) if options.get("resume_from_checkpoint") else None
        ),
        skip_final_save=bool(options.get("skip_final_save", False)),
        dataset_limit=None if dataset_limit is None else int(dataset_limit),
        dataloader_num_workers=int(options.get("dataloader_num_workers", 2)),
        resource_report_path=str(resource_report_path),
        log_memory_every_n_steps=int(options.get("log_memory_every_n_steps", 5)),
    )


def _run_training_stage(backbone: VLM, config: Config, stage: str) -> None:
    validate_fixed_training_environment(require_distributed_dependencies=True)
    backend = _training_backend(backbone, config)
    compressor_args = training_compressor_args(_mapping_option(config, "compressor_args"))
    reset_stage1 = bool(compressor_args.get("reset_stage1", backend.stage1_reset_by_default))
    # The Qwen schedule is depth-relative, so it cannot resolve until the decoder's own
    # layer count is known. Reading it off the config avoids loading the model twice.
    hidrop_config = _hidrop_config_for_training(
        config, backend, num_decoder_layers=_decoder_depth(backbone)
    )
    args = _build_training_args(backbone, config, stage, hidrop_config, backend)
    # Seed before creating any randomly initialized stage-1 module.
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 and not backend.released_recipe:
        warnings.warn(
            f"HiDrop training on the {backend.key} backbone is our adaptation of the released "
            "LLaVA-1.5 recipe, not a published one: upstream ships no HiDrop checkpoint or "
            "training recipe for it. Results are not comparable to the paper's numbers, and the "
            "run manifest records this.",
            stacklevel=3,
        )

    if args.pretrain_mm_mlp_adapter:
        validate_hidrop_artifact(args.pretrain_mm_mlp_adapter, hidrop_config,
                                 artifact_name="pretrained adapter", backend=backend)
    if args.resume_from_checkpoint:
        validate_hidrop_artifact(args.resume_from_checkpoint, hidrop_config,
                                 artifact_name="resume checkpoint", backend=backend)
    # The released LLaVA/HiDrop stack predates non-reentrant checkpointing and
    # uses PyTorch's reentrant path.  It is also the ZeRO-3-safe choice: the
    # non-reentrant metadata checker can observe temporarily partitioned
    # parameters as empty tensors during recomputation.
    gradient_checkpointing_kwargs = {"use_reentrant": True} if args.gradient_checkpointing else None
    processor, base_model = backend.load_training_base(
        args.model_name_or_path,
        stage=stage,
        dtype=torch.bfloat16 if args.bf16 else None,
        attn_implementation=args.attn_implementation,
        model_max_length=args.model_max_length,
        local_files_only=bool(getattr(backbone, "_local_files_only", True)),
    )
    args.requested_attn_implementation = args.attn_implementation
    args.effective_text_attn_implementation, args.effective_vision_attn_implementation = (
        _effective_attn_implementations(base_model, backend)
    )
    if stage == "pretrain" and not args.resume_from_checkpoint and reset_stage1:
        backend.reset_stage1_module(base_model)
    model = HiDropTrainingModel(base_model, hidrop_config, backend=backend, processor=processor)
    stage1_module = backend.stage1_module(model.base_model)
    if stage == "pretrain":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in stage1_module.parameters():
            parameter.requires_grad_(True)
    elif args.pretrain_mm_mlp_adapter:
        adapter = torch.load(args.pretrain_mm_mlp_adapter, map_location="cpu", weights_only=True)
        stage1_module.load_state_dict(adapter, strict=True)
    backend.freeze_vision_tower(model.base_model)
    if stage == "pretrain":
        # On Qwen the stage-1 merger sits *inside* the vision tower, so freezing the
        # tower above just froze it again. Re-enable it: it is what this stage trains.
        for parameter in stage1_module.parameters():
            parameter.requires_grad_(True)
    _report_trainable_parameters(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=TRAINING_OPTIMIZER,
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
        logging_steps=10,
        average_tokens_across_devices=CROSS_DEVICE_TOKEN_AVERAGING,
        deepspeed=_deepspeed_config(
            args.deepspeed_stage,
            args.global_batch_size,
            args.per_device_train_batch_size,
            args.gradient_accumulation_steps,
        ),
    )
    dataset = LlavaJsonDataset(args.data_path, args.image_folder, dataset_limit=args.dataset_limit)
    collator = backend.build_collator(
        processor, model_max_length=args.model_max_length, stage=stage
    )
    run_manifest = None
    if rank == 0:
        run_manifest = build_run_manifest(
            args=asdict(args),
            hidrop=backend.schedule_metadata(hidrop_config),
            dataset_length=len(dataset),
            optimizer=TRAINING_OPTIMIZER,
            loss_semantics=LOSS_SEMANTICS,
        )
        if args.pretrain_mm_mlp_adapter:
            validate_artifact_provenance(
                args.pretrain_mm_mlp_adapter,
                run_manifest,
                artifact_name="pretrained adapter",
                resume=False,
            )
        if args.resume_from_checkpoint:
            validate_artifact_provenance(
                args.resume_from_checkpoint,
                run_manifest,
                artifact_name="resume checkpoint",
                resume=True,
            )
        write_run_manifest(Path(args.output_dir), run_manifest)
        (Path(args.output_dir) / "training_args_hidrop.json").write_text(
            json.dumps(asdict(args), indent=2, default=str),
            encoding="utf-8",
        )
    report_callback = ResourceReportCallback(
        stage=stage,
        budget=args.budget_tag,
        requested_attn_implementation=args.attn_implementation,
        effective_text_attn_implementation=args.effective_text_attn_implementation,
        effective_vision_attn_implementation=args.effective_vision_attn_implementation,
        global_batch_size=args.global_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
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
        callbacks=[HiDropSaveControlCallback(), StopAfterStepCallback(args.stop_after_steps), report_callback],
        group_by_modality_length=args.group_by_modality_length,
        run_manifest=run_manifest,
    )
    trainer.projector_module_path = backend.stage1_module_path
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if stage == "pretrain":
        # Stage 1 saves only its adapter, not a redundant copy of the frozen 7B base;
        # `ft` consumes that file.
        unwrapped_model = trainer.accelerator.unwrap_model(model)
        if trainer.is_world_process_zero():
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(output)
            unwrapped_model.base_model.config.save_pretrained(output)
            if getattr(unwrapped_model.base_model, "generation_config", None) is not None:
                unwrapped_model.base_model.generation_config.save_pretrained(output)
            projector_state = {
                key: value.detach().cpu()
                for key, value in backend.stage1_module(
                    unwrapped_model.base_model
                ).state_dict().items()
            }
            projector_path = output / backend.stage1_artifact_name
            torch.save(projector_state, projector_path)
            (output / "hidrop_config.json").write_text(
                json.dumps(backend.schedule_metadata(unwrapped_model.hidrop_config), indent=2),
                encoding="utf-8",
            )
            (output / "training_args_hidrop.json").write_text(
                json.dumps(asdict(args), indent=2, default=str), encoding="utf-8"
            )
            if run_manifest is not None:
                record_output_file(run_manifest, backend.stage1_artifact_name, projector_path)
                # `mm_projector` is the ROLE the provenance check validates
                # (provenance.validate_artifact reads outputs["mm_projector"]),
                # not a filename -- Qwen's file is mm_merger.bin. Recording it
                # here, after training, is what makes the final adapter own the
                # key; otherwise the last periodic checkpoint owns it and stage 2
                # refuses the very adapter stage 1 just produced.
                record_output_file(run_manifest, "mm_projector", projector_path)
                write_run_manifest(output, run_manifest)
            report_callback.write_report()
            print(f"Saved HiDrop stage-1 adapter to {projector_path}", flush=True)
        return
    if args.skip_final_save:
        if trainer.is_world_process_zero():
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            (output / "training_args_hidrop.json").write_text(
                json.dumps(asdict(args), indent=2, default=str), encoding="utf-8"
            )
            report_callback.write_report()
        return
    state_dict = _gather_final_state_dict(trainer)
    if trainer.is_world_process_zero():
        model.save_pretrained(args.output_dir, state_dict=state_dict, safe_serialization=True)
        processor.save_pretrained(args.output_dir)
        (Path(args.output_dir) / "training_args_hidrop.json").write_text(
            json.dumps(asdict(args), indent=2, default=str), encoding="utf-8"
        )
        report_callback.write_report()
        print(
            f"Saved HiDrop-{args.budget_tag} checkpoint to {args.output_dir}. "
            "Benchmark it with compressor_args.checkpoint_path.",
            flush=True,
        )


def train_hidrop(backbone: VLM, config: Config) -> None:
    """`tr`: run the backbone's canonical HiDrop stage-1 adaptation."""
    if getattr(backbone, "compressor_family", None) == "llava_next":
        from llmoe.compressor.hidrop.hidrop_next_tr import run_hidrop_next_training

        run_hidrop_next_training(backbone, config, "pretrain")
        return
    _run_training_stage(backbone, config, "pretrain")


def finetune_hidrop(backbone: VLM, config: Config) -> None:
    """`ft`: run the backbone's canonical HiDrop full finetuning stage."""
    if getattr(backbone, "compressor_family", None) == "llava_next":
        from llmoe.compressor.hidrop.hidrop_next_tr import run_hidrop_next_training

        run_hidrop_next_training(backbone, config, "finetune")
        return
    _run_training_stage(backbone, config, "finetune")
