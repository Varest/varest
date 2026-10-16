"""Training entry points for :class:`DUETCompressor` (``tr`` and ``ft``).

DUET's headline numbers come from a backbone that was *trained* under the
schedule, not from bolting the schedule onto a stock checkpoint. Stage 1 hands
the projector ``S`` merged tokens instead of 576, and Stage 2 deletes most of
them partway down the decoder; adapting the projector and the language model to
that stream is what upstream's pretrain/finetune stack does.

Training is the released two-stage LLaVA-1.5 recipe, run with DUET installed:

* ``tr`` -- **projector pretraining** on the 558K BLIP/LAION/CC/SBU captions with
  ``plain`` conversations. Everything is frozen except ``multi_modal_projector``.
  Global batch 256, lr 1e-3, DeepSpeed ZeRO-2. Writes ``mm_projector.bin``.
* ``ft`` -- **full finetuning** on the 665K instruction mixture with ``vicuna_v1``
  conversations, starting from *that budget's* projector. The vision tower stays
  frozen; the projector and the language model train. Global batch 128, lr 2e-5,
  DeepSpeed ZeRO-3. Writes a full HF LLaVA checkpoint.

Qwen2.5-VL uses the same two stages as an explicit harness adaptation, because
upstream did not release a Qwen training recipe. Its dynamic visual grids use
ratio budgets, row-wise pruning, and right-padding after every cut; Stage 1
trains ``visual.merger`` and writes ``mm_merger.bin``, while Stage 2 trains that
merger plus the language model with the rest of the vision tower frozen.

The stages chain by budget: a 64-token finetune must start from the 64-token
projector, and ``ft`` looks for it under this budget's ``tr`` output rather than
trusting the caller to keep directories straight.

Benchmarking a trained checkpoint points the *backbone* at it -- the weights are
a plain HF backbone checkpoint, so there is no ``compressor_args.checkpoint_path``::

    llmoe --config local --compressor DUET \\
      --custom_args duet_trained.yaml bm mme

    # duet_trained.yaml
    backbone_args:
      model_name_or_path: /results/duet/llava-v1.5-7b-duet-64
    compressor_args:
      max_tokens: 64

Launching
---------
Training is FlashAttention-2 only and multi-GPU. Launch the harness under a
distributed launcher; the trainer picks up ``WORLD_SIZE``/``RANK`` from it::

    torchrun --nproc_per_node=8 -m llmoe --config local \\
      --compressor DUET --custom_args duet_pretrain.yaml \\
      --input_dir /datasets --output_dir /results tr

    torchrun --nproc_per_node=8 -m llmoe --config local \\
      --compressor DUET --custom_args duet_finetune.yaml \\
      --input_dir /datasets --output_dir /results ft

``duet_pretrain.yaml``::

    compressor_args:
      max_tokens: 64
    training_args:
      per_device_train_batch_size: 8      # x world size x accumulation == 256

``duet_finetune.yaml``::

    compressor_args:
      max_tokens: 64
    training_args:
      per_device_train_batch_size: 4      # x world size x accumulation == 128
      # defaults to the matching tr output; set it if the projector lives elsewhere
      # pretrain_mm_mlp_adapter: /results/duet/llava-v1.5-7b-duet-64-pretrain

Datasets are read from ``<input_dir>/LLaVA-Pretrain`` and
``<input_dir>/LLaVA-Tuning``; ``training_args.data_path`` and
``training_args.image_folder`` override that.

Faithfulness notes / harness adaptations
----------------------------------------
Ported from ``ProjectNeura/llmoe_baseline_duet_vlm`` (``train_duet_llava_hf.py``
and ``training/``), which is the DUET-VLM training stack this port follows.

* **No launcher script.** The sidecar drove this through an ``argparse`` CLI and
  Compute Canada job scripts. Those knobs are ``training_args`` here, and the
  launcher is whatever ``torchrun``/``deepspeed`` invocation the site uses.
  ``erbium`` monitoring is left to the harness's own profiling.

* **The trainer builds its own base model.** ``run_training`` hands over a
  :class:`~llmoe.backbone.prototype.VLM`, but its model is configured for
  *evaluation* -- fp16, SDPA, a left-padded processor. Training needs bf16 + FA2
  and right padding. The backbone is lazy, so nothing has been loaded yet: the
  trainer takes its ``model_name_or_path`` and ``local_files_only`` and loads the
  training stack itself, leaving the eval backbone untouched.

* **The compressor is the same object.** On LLaVA, the registered
  :class:`DUETCompressor` is installed on a thin adapter around the training
  model, so training and benchmarking run identical Stage-1/Stage-2 hooks. The
  Qwen inference adapter owns single-prompt cache surgery and cannot represent a
  right-padded training batch, so Qwen uses a batched decoder that calls the same
  Stage-1 and Stage-2 policy helpers row by row. ``prepare_inputs`` is never
  called during training; each collator emits the backbone's placeholders.

* **Pruned-label loss.** Stage 2 removes token positions, so the logits are
  shorter than the labels that went in. The wrapper below runs the base model with
  ``labels=None`` and takes the shifted CE against
  ``labels.gather(1, compressor.final_keep_index)`` -- the sidecar's
  ``duet_outer_forward`` does exactly this.

* **Same-modality micro-batches.** The sidecar's collator refuses a micro-batch
  mixing image and text-only rows rather than feeding LLaVA a dummy image; that is
  reproduced, and ``group_by_modality_length`` (on by default for both stages, as
  in the released scripts) is what keeps micro-batches uniform.

* **Gradient checkpointing.** The sidecar replaced ``LlamaModel.forward`` and
  never routed the decoder through ``_gradient_checkpointing_func``, so its
  ``--gradient_checkpointing`` was a no-op for the decoder stack. Hook-based
  pruning *is* recomputed here, which ``DUETCompressor._prune`` handles by
  replaying each boundary's stored cut.

* **The projector is adapted, not reset.** ``tr`` starts from the stock
  ``llava-hf/llava-1.5-7b-hf`` projector and compression-adapts it, which is what
  the released scripts do; it is not re-initialised per budget.
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
from PIL import Image
from mle.vars import ExpConfig
from torch import nn
from torch.utils.data import Dataset, Sampler
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

from llmoe.backbone.prototype import VLM
from llmoe.compressor.duet.duet import DUETCompressor, average_image_tokens
from llmoe.compressor.duet.duet_qwen_tr import (
    DuetTrainingQwenForConditionalGeneration,
)
from llmoe.compressor.hidrop.hidrop import BACKENDS as HIDROP_BACKENDS
from llmoe.compressor.hidrop.data import QwenHiDropCollator
from llmoe.config import Config
from llmoe.utils.depth import REFERENCE_DEPTH, config_decoder_depth
from llmoe.weights import BACKBONE_SLUGS, backbone_key


IGNORE_INDEX = -100
ATTN_IMPLEMENTATION = "flash_attention_2"
RELEASED_MODEL_MAX_LENGTH = 2048
#: Qwen's context for these adaptations. The released LLaVA recipe pins 2048 and
#: that pin stands; upstream publishes no Qwen recipe, and 2048 cannot hold both
#: a 2048-token image (DEFAULT_QWEN_IMAGE_TOKENS) and its text.
QWEN_MODEL_MAX_LENGTH = 4096
# Transformers 5 can rescale the loss by token counts gathered across devices.
# Released LLaVA does not; keeping it off preserves the recipe's effective LR.
CROSS_DEVICE_TOKEN_AVERAGING = False
# Released LLaVA-1.5 stage hyperparameters, as pinned by the sidecar's scripts.
RELEASED_STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "pretrain": {
        "learning_rate": 1e-3,
        "global_batch_size": 256,
        "deepspeed_stage": 2,
        "save_steps": 24000,
        "prompt_version": "plain",
    },
    "finetune": {
        "learning_rate": 2e-5,
        "global_batch_size": 128,
        "deepspeed_stage": 3,
        "save_steps": 1000,
        "prompt_version": "v1",
    },
}


# ---------------------------------------------------------------------------
# LLaVA conversation data for the two stages
# ---------------------------------------------------------------------------
#
# Both stages read the released LLaVA JSON format -- a list of rows with an
# optional `image` and a `conversations` list -- and differ only in which file:
#
# * `tr` (projector pretraining): `LLaVA-Pretrain/blip_laion_cc_sbu_558k.json`
#   with its `images` tree;
# * `ft` (full finetuning): `LLaVA-Tuning/llava_v1_5_mix665k.json` with the
#   dataset root as the image folder (its `image` paths are already prefixed).
#
# Directory names follow the harness convention (`LLaVA-Pretrain` is what
# `soft_topk_tr.py` and `hidrop_tr.py` use); `training_args.data_path` /
# `training_args.image_folder` override the lookup. Rows keep their modality
# length so the released modality-length-grouped sampler can order them.


DATASET_DIR_BY_STAGE = {"pretrain": "LLaVA-Pretrain", "finetune": "LLaVA-Tuning"}
ANNOTATIONS_BY_STAGE = {
    "pretrain": ("blip_laion_cc_sbu_558k.json",),
    "finetune": ("llava_v1_5_mix665k.json",),
}
# Visual tokens LLaVA-1.5 adds per image; used only for length bookkeeping.
VISUAL_TOKENS = 576
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
)


def dataset_dir_for_stage(input_dir: str | Path, stage: str) -> Path:
    if stage not in DATASET_DIR_BY_STAGE:
        raise ValueError(f"Unknown DUET training stage {stage!r}.")
    return Path(input_dir) / DATASET_DIR_BY_STAGE[stage]


def find_annotations(root: Path, stage: str) -> Path:
    """Locate the stage's released JSON, falling back to any JSON under ``root``."""
    for name in ANNOTATIONS_BY_STAGE[stage]:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    matches = sorted(root.rglob("*.json"))
    if not matches:
        raise FileNotFoundError(f"No JSON annotations found under {root}")
    return matches[0]


def resolve_training_data(
    input_dir: str | Path,
    stage: str,
    *,
    data_path: str | None = None,
    image_folder: str | None = None,
) -> tuple[Path, Path]:
    """Resolve ``(data_path, image_folder)`` for a stage, honouring explicit overrides."""
    dataset_dir = dataset_dir_for_stage(input_dir, stage)
    resolved_data = Path(data_path) if data_path else find_annotations(dataset_dir, stage)
    if not resolved_data.is_file():
        raise FileNotFoundError(f"DUET {stage} annotations not found: {resolved_data}")
    if image_folder:
        resolved_images = Path(image_folder)
    else:
        # 558K ships its images under `images/`; 665K paths are already prefixed
        # with their source dataset, so the dataset root is the image folder.
        candidate = dataset_dir / "images"
        resolved_images = candidate if stage == "pretrain" and candidate.is_dir() else dataset_dir
    if not resolved_images.is_dir():
        raise FileNotFoundError(f"DUET {stage} image folder not found: {resolved_images}")
    return resolved_data, resolved_images


class LlavaJsonDataset(Dataset):
    """The released LLaVA JSON rows, tagged with signed modality lengths."""

    def __init__(
        self, data_path: str | Path, image_folder: str | Path, dataset_limit: int | None = None
    ) -> None:
        self.data_path = Path(data_path)
        self.image_folder = Path(image_folder)
        rows = json.loads(self.data_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Expected {self.data_path} to contain a JSON list.")
        self.rows = rows[: int(dataset_limit)] if dataset_limit is not None else rows
        self.has_images = [bool(row.get("image")) for row in self.rows]
        conversation_lengths = [self._conversation_length(row) for row in self.rows]
        self.lengths = [
            length + (VISUAL_TOKENS if has_image else 0)
            for length, has_image in zip(conversation_lengths, self.has_images, strict=True)
        ]
        # Sign carries the modality: the released sampler groups image and
        # text-only rows into separate megabatches.
        self.modality_lengths = [
            length if has_image else -length
            for length, has_image in zip(conversation_lengths, self.has_images, strict=True)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        conversations = self._normalize_conversations(row.get("conversations") or [])
        image_name = row.get("image")
        image = None
        if image_name:
            image_path = Path(str(image_name))
            if not image_path.is_absolute():
                image_path = self.image_folder / image_path
            image = Image.open(image_path).convert("RGB")
        return {"image": image, "conversations": conversations, "has_image": image is not None}

    @staticmethod
    def _normalize_conversations(conversations: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for turn in conversations:
            role = turn.get("from") or turn.get("role")
            value = str(turn.get("value") or turn.get("content") or "")
            if role in {"human", "user"}:
                normalized.append({"role": "user", "value": value})
            elif role in {"gpt", "assistant"}:
                normalized.append({"role": "assistant", "value": value})
            elif role == "system":
                normalized.append({"role": "system", "value": value})
        if not any(turn["role"] == "assistant" for turn in normalized):
            raise ValueError("LLaVA training row must contain at least one assistant turn.")
        return normalized

    @staticmethod
    def _conversation_length(row: dict[str, Any]) -> int:
        length = 0
        for turn in row.get("conversations") or []:
            length += len(str(turn.get("value") or turn.get("content") or "").split())
        return max(1, length)


# ---------------------------------------------------------------------------
# The DUET collator
# ---------------------------------------------------------------------------
#
# The two stages use different templates:
#
# * `plain` (projector pretraining). The user instruction is *discarded*: the
#   prompt is `<image>` followed by the caption, and only the caption is
#   supervised.
# * `v1` (full finetuning). The Vicuna system prompt plus alternating
#   `USER:`/`ASSISTANT:` rounds, with everything but the assistant answers masked.
#
# `<image>` is expanded to `stage1_tokens` placeholders -- not 576 -- because
# Stage 1 reduces the vision tower's output before the projector, so that is how
# many features LLaVA will scatter in. Every placeholder is masked out of the
# labels.
#
# A micro-batch must be single-modality: DUET's Stage-2 gather assumes one
# contiguous visual span of the same width in every row, and upstream refuses the
# mixed case rather than padding it with a dummy image. `group_by_modality_length`
# is what keeps micro-batches uniform.


class LlavaDuetCollator:
    """Tokenize a stage's conversations and batch them with their images."""

    def __init__(
        self,
        processor: Any,
        *,
        model_max_length: int = RELEASED_MODEL_MAX_LENGTH,
        visual_tokens: int,
        prompt_version: str,
    ) -> None:
        self.processor = processor
        self.model_max_length = int(model_max_length)
        self.visual_tokens = int(visual_tokens)
        self.prompt_version = str(prompt_version)
        if self.prompt_version not in {"plain", "v1"}:
            raise ValueError("LLaVA DUET prompt_version must be 'plain' or 'v1'.")
        image_token = getattr(self.processor, "image_token", "<image>")
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id is None or int(image_token_id) < 0:
            raise ValueError(
                f"Tokenizer does not contain required LLaVA image token {image_token!r}."
            )
        # A tokenizer that merged or stripped repeated placeholders would silently
        # break LLaVA's feature/placeholder count check.
        encoded = self.processor.tokenizer(
            image_token * self.visual_tokens, add_special_tokens=False
        )
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        count = sum(1 for token_id in token_ids if int(token_id) == int(image_token_id))
        if count != self.visual_tokens:
            raise ValueError(
                f"Tokenizing {self.visual_tokens} repeated image tokens produced {count} "
                f"image token ids."
            )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("DUET collator received an empty batch.")
        has_image = bool(batch[0]["has_image"])
        if any(bool(item["has_image"]) != has_image for item in batch):
            raise ValueError(
                "DUET batched training requires same-modality micro-batches; enable "
                "training_args.group_by_modality_length for mixed data."
            )

        tokenizer = self.processor.tokenizer
        image_token = getattr(self.processor, "image_token", "<image>")
        image_token_id = int(tokenizer.convert_tokens_to_ids(image_token))
        rows: list[tuple[list[int], list[int]]] = []
        for item in batch:
            input_ids, labels = self._build_tokenized_conversation(
                item["conversations"], bool(item["has_image"])
            )
            if len(input_ids) > self.model_max_length:
                input_ids = input_ids[: self.model_max_length]
                labels = labels[: self.model_max_length]
            if has_image:
                count = sum(1 for token_id in input_ids if token_id == image_token_id)
                if count != self.visual_tokens:
                    raise ValueError(
                        f"DUET expected exactly {self.visual_tokens} image placeholder tokens "
                        f"after truncation, got {count}."
                    )
            if not any(label != IGNORE_INDEX for label in labels):
                raise ValueError(
                    "DUET collator produced no supervised assistant labels after truncation."
                )
            rows.append((input_ids, labels))

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", 0)
        pad_token_id = int(0 if pad_token_id is None else pad_token_id)
        max_len = max(len(input_ids) for input_ids, _ in rows)
        padded_ids: list[list[int]] = []
        padded_labels: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for input_ids, labels in rows:
            pad = max_len - len(input_ids)
            padded_ids.append(input_ids + [pad_token_id] * pad)
            padded_labels.append(labels + [IGNORE_INDEX] * pad)
            attention_masks.append([1] * len(input_ids) + [0] * pad)

        result: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        if has_image:
            image_inputs_by_key: dict[str, list[torch.Tensor]] = {}
            for item in batch:
                if item["image"] is None:
                    raise ValueError("DUET image batch contained a row without image data.")
                image_inputs = self.processor.image_processor(item["image"], return_tensors="pt")
                for key, value in image_inputs.items():
                    if torch.is_tensor(value):
                        image_inputs_by_key.setdefault(key, []).append(value)
            for key, values in image_inputs_by_key.items():
                result[key] = (
                    torch.cat(values, dim=0)
                    if values[0].shape[:1] == (1,)
                    else torch.stack(values, dim=0)
                )
        return result

    def _build_tokenized_conversation(
        self, conversations: list[dict[str, str]], has_image: bool
    ) -> tuple[list[int], list[int]]:
        tokenizer = self.processor.tokenizer
        eos = tokenizer.eos_token or ""
        image_token = getattr(self.processor, "image_token", "<image>")
        expanded_image_token = image_token * self.visual_tokens
        # (text, supervised, add_special_tokens) -- only the first segment gets BOS.
        segments: list[tuple[str, bool, bool]] = []
        if self.prompt_version == "plain":
            captions = [
                turn["value"].strip()
                for turn in conversations
                if turn["role"] == "assistant" and turn["value"].strip()
            ]
            if not captions:
                raise ValueError(
                    "LLaVA plain pretraining row must contain a non-empty assistant/caption turn."
                )
            segments.append((image_token if has_image else "", False, True))
            segments.append((f"{captions[0]}\n", True, False))
        else:
            segments.append((SYSTEM_PROMPT, False, True))
            saw_image = False
            for turn in conversations:
                role, value = turn["role"], turn["value"]
                if role == "system":
                    segments.append((value.strip() + " ", False, False))
                elif role == "user":
                    if has_image and not saw_image and image_token not in value:
                        value = f"{image_token}\n{value}"
                    saw_image = saw_image or image_token in value
                    segments.append((f"USER: {value.strip()} ASSISTANT: ", False, False))
                elif role == "assistant":
                    segments.append((f"{value.strip()}{eos}", True, False))

        input_ids: list[int] = []
        labels: list[int] = []
        image_placeholder_count = 0
        for text, supervised, add_special_tokens in segments:
            if has_image:
                image_placeholder_count += text.count(image_token)
            text = text.replace(image_token, expanded_image_token if has_image else "")
            encoded = tokenizer(text, add_special_tokens=add_special_tokens)
            token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            input_ids.extend(int(token_id) for token_id in token_ids)
            labels.extend((int(token_id) if supervised else IGNORE_INDEX) for token_id in token_ids)

        if has_image and image_placeholder_count != 1:
            raise ValueError(
                "DUET training expects exactly one image placeholder per image row before "
                f"expansion, got {image_placeholder_count}."
            )
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id is not None and int(image_token_id) >= 0:
            labels = [
                IGNORE_INDEX if token_id == int(image_token_id) else label
                for token_id, label in zip(input_ids, labels, strict=True)
            ]
        if not any(label != IGNORE_INDEX for label in labels):
            raise ValueError("DUET collator produced no supervised assistant labels.")
        return input_ids, labels


# ---------------------------------------------------------------------------
# The training model
# ---------------------------------------------------------------------------


class _TrainingBackbone:
    """The slice of :class:`VLM` that :meth:`DUETCompressor.install` touches.

    The compressor installs onto a ``VLM``, but training drives a bare
    ``LlavaForConditionalGeneration``. Rather than reimplement the hooks, wrap the
    training model in the attributes ``install()`` reads. ``prepare_inputs`` and
    ``generate`` exist only so the wrappers have something to wrap; the collator
    already emits ``stage1_tokens`` placeholders and training never generates, so
    neither is called.
    """

    def __init__(self, model: LlavaForConditionalGeneration, processor: LlavaProcessor) -> None:
        self.model = model
        self.processor = processor

    def prepare_inputs(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("DUET training builds its batches through LlavaDuetCollator.")

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        # The LLaVA-NeXT install() captures and wraps `generate` for inference. It
        # is only ever wrapped during training, never called, but it has to exist
        # or install() raises AttributeError before the first step.
        raise RuntimeError("DUET training does not generate; it optimises a training loss.")


class DuetTrainingLlavaForConditionalGeneration(nn.Module):
    """LLaVA wrapper whose loss uses the labels that survived Stage-2 pruning.

    Stage 2 deletes token positions, so the logits are shorter than the labels the
    collator produced. The base model therefore runs with ``labels=None`` and the
    causal loss is taken here against the labels gathered through
    :attr:`DUETCompressor.final_keep_index`.
    """

    def __init__(
        self, base_model: LlavaForConditionalGeneration, compressor: DUETCompressor
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.generation_config = base_model.generation_config
        self.compressor = compressor

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

    def gradient_checkpointing_disable(self) -> None:
        self.base_model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> Any:
        # Transformers 5 passes this to models accepting **kwargs and changes loss
        # normalization across gradient accumulation. Released LLaVA computes a
        # mean CE loss per micro-batch.
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("use_cache", None)
        outputs = self.base_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=None,
            use_cache=False,
            **kwargs,
        )
        if labels is None:
            return outputs
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError(
                "DUET could not compute the training loss: the model returned no logits."
            )
        keep_index = self.compressor.final_keep_index
        pruned_labels = (
            labels if keep_index is None else labels.gather(1, keep_index.to(labels.device))
        )
        if pruned_labels.shape[1] != logits.shape[1]:
            raise RuntimeError(
                f"DUET pruned {pruned_labels.shape[1]} labels but the decoder emitted "
                f"{logits.shape[1]} positions; the pruning schedule and the loss disagree."
            )
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = pruned_labels[..., 1:].contiguous().to(shift_logits.device)
        outputs.loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return outputs

    def save_pretrained(self, output_dir: str | Path, **kwargs: Any) -> None:
        """Save a plain HF LLaVA checkpoint plus the DUET schedule beside it.

        The weights carry no DUET-specific parameters -- the schedule is pure
        inference-time behaviour -- so the checkpoint reloads with stock
        ``transformers`` and is benchmarked through ``backbone_args``.
        """
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
        self.base_model.save_pretrained(output, state_dict=base_state_dict, **kwargs)
        write_duet_metadata(output, self.compressor)


def duet_metadata(compressor: DUETCompressor, *, backbone: str = "llava") -> dict[str, Any]:
    """The schedule a checkpoint was trained under."""
    if backbone == "llava_next" or getattr(compressor, "_metadata_backbone", None) == "llava_next":
        # Lazy to avoid importing the NeXT trainer while this shared module is
        # still defining wrapper classes reused by that trainer.
        from llmoe.compressor.duet.duet_next_tr import duet_next_metadata

        return duet_next_metadata(compressor)
    schedule_depth = int(getattr(compressor, "_schedule_depth", None) or REFERENCE_DEPTH)
    metadata: dict[str, Any] = {
        "backbone": str(backbone),
        "budget": compressor.budget,
        "dominant": compressor._dominant,
        "contextual": compressor._contextual,
        "retain_ratio": compressor._retain_ratio,
        "contextual_ratio": compressor._contextual_ratio,
        "cluster_width": compressor._cluster_width,
        "pdrop_layers": list(compressor._pdrop_layers),
        "pdrop_keep_ratios": list(compressor._pdrop_keep_ratios),
        "text_rater": compressor._text_rater,
        "score_reduction": compressor._score_reduction,
        "use_stage1": compressor._use_stage1,
        "use_stage2": compressor._use_stage2,
    }
    if backbone == "qwen":
        metadata.update(
            {
                "include_cls_token": False,
                "stage1_tokens": (
                    None
                    if compressor._dominant is None
                    else int(compressor._dominant) + int(compressor._contextual)
                ),
                "released_training_recipe": False,
            }
        )
    else:
        metadata.update(
            {
                "include_cls_token": compressor._include_cls_token,
                "stage1_tokens": compressor._stage1_tokens,
                # Against the decoder this schedule was resolved for, not the 32
                # the LLaVA default is stated at. Hardcoding 32 made a 40-layer
                # 13B run report 39.69 where its true average was 31.75 -- the
                # weights were right, the recorded number was not.
                "decoder_layers": schedule_depth,
                "average_image_tokens": average_image_tokens(
                    compressor._stage1_tokens,
                    compressor._pdrop_layers,
                    compressor._pdrop_keep_ratios,
                    schedule_depth,
                ),
                "released_training_recipe": True,
            }
        )
    return metadata


def write_duet_metadata(
    output: str | Path,
    compressor: DUETCompressor,
    training_args_payload: dict[str, Any] | None = None,
    *,
    backbone: str = "llava",
) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "duet_config.json").write_text(
        json.dumps(duet_metadata(compressor, backbone=backbone), indent=2),
        encoding="utf-8",
    )
    if training_args_payload is not None:
        (output / "training_args_duet.json").write_text(
            json.dumps(training_args_payload, indent=2, default=str),
            encoding="utf-8",
        )


def _effective_attn_implementations(
    base_model: LlavaForConditionalGeneration,
) -> tuple[str | None, str | None]:
    return (
        getattr(base_model.model.language_model.config, "_attn_implementation", None),
        getattr(base_model.model.vision_tower.config, "_attn_implementation", None),
    )


def load_hf_llava_training_base(
    model_name_or_path: str,
    *,
    stage: str,
    dtype: torch.dtype | None,
    attn_implementation: str,
    model_max_length: int = RELEASED_MODEL_MAX_LENGTH,
    local_files_only: bool = True,
) -> tuple[LlavaProcessor, LlavaForConditionalGeneration]:
    """Load the HF LLaVA base with the released *training* preprocessing."""
    if attn_implementation != ATTN_IMPLEMENTATION:
        raise ValueError(
            f"DUET training only supports flash_attention_2, got {attn_implementation!r}."
        )
    if stage not in RELEASED_STAGE_DEFAULTS:
        raise ValueError(f"Unknown DUET training stage {stage!r}.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        model_max_length=int(model_max_length),
        padding_side="right",
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    if (
        image_token_id is None
        or int(image_token_id) < 0
        or image_token_id == tokenizer.unk_token_id
    ):
        raise ValueError(
            "The Hugging Face LLaVA checkpoint must provide the canonical <image> token."
        )
    image_processor = AutoImageProcessor.from_pretrained(
        model_name_or_path, local_files_only=local_files_only
    )
    image_processor_config = image_processor.to_dict()
    image_processor_config.pop("do_pad", None)
    image_processor_config["image_processor_type"] = "LlavaImageProcessor"
    # The released recipe uses the default square/crop path for 558K projector
    # pretraining and --image_aspect_ratio pad for 665K finetuning.
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
    processor = LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=getattr(base_model.config.vision_config, "patch_size", 14),
        vision_feature_select_strategy=getattr(
            base_model.config, "vision_feature_select_strategy", "default"
        ),
        image_token="<image>",
    )
    for config in (
        base_model.config,
        base_model.config.text_config,
        base_model.config.vision_config,
        base_model.model.language_model.config,
        base_model.model.vision_tower.config,
    ):
        config._attn_implementation = attn_implementation
        if hasattr(config, "_attn_implementation_internal"):
            config._attn_implementation_internal = attn_implementation
    text_attn, vision_attn = _effective_attn_implementations(base_model)
    if text_attn != attn_implementation or vision_attn != attn_implementation:
        raise RuntimeError(
            f"Requested {attn_implementation!r} but Transformers selected text={text_attn!r} "
            f"vision={vision_attn!r}."
        )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Effective attention backends: text={text_attn!r} vision={vision_attn!r}", flush=True
        )
    return processor, base_model


def load_pretrained_projector(
    adapter_path: str | Path, projector: nn.Module
) -> dict[str, torch.Tensor]:
    """Load a ``tr`` ``mm_projector.bin`` into ``ft``'s projector.

    Released adapters have been written under several key layouts (LLaVA's
    ``mm_projector.0/2``, HF's ``linear_1/linear_2``, with or without a module
    prefix), so the names are normalised before loading rather than trusting one.
    """
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if not isinstance(adapter, dict):
        raise TypeError(f"Expected {adapter_path} to contain a state dict, got {type(adapter)}.")
    target_state = projector.state_dict()
    projector_state: dict[str, torch.Tensor] = {}
    for key, value in adapter.items():
        name = str(key)
        for prefix in (
            "model.multi_modal_projector.",
            "multi_modal_projector.",
            "model.mm_projector.",
            "mm_projector.",
        ):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        if name.startswith("0."):
            name = "linear_1." + name[2:]
        elif name.startswith("2."):
            name = "linear_2." + name[2:]
        if name in target_state:
            projector_state[name] = value
    if not projector_state:
        expected = ", ".join(sorted(target_state))
        sample = ", ".join(str(key) for key in list(adapter)[:8])
        raise ValueError(
            f"Could not map any projector weights from {adapter_path}. Expected keys like "
            f"{expected}; saw {sample}."
        )
    projector.load_state_dict(projector_state, strict=False)
    return projector_state


def _stage1_state_from_full_state_dict(
    state_dict: dict[str, torch.Tensor],
    module_path: str = "model.multi_modal_projector",
) -> dict[str, torch.Tensor]:
    marker = f"base_model.{module_path}."
    return {
        key.split(marker, 1)[1]: value.detach().cpu()
        for key, value in state_dict.items()
        if marker in key
    }


def _projector_state_from_full_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Backward-compatible name for the LLaVA stage-1 extractor."""
    return _stage1_state_from_full_state_dict(state_dict)


def _parameter_group_for_name(name: str) -> str:
    if ".visual.merger." in name or ".multi_modal_projector." in name:
        return "multi_modal_projector"
    if ".vision_tower." in name or ".visual." in name:
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
        counts = groups.setdefault(_parameter_group_for_name(name), {"total": 0, "trainable": 0})
        counts["total"] += numel
        total += numel
        if parameter.requires_grad:
            counts["trainable"] += numel
            trainable += numel
    ratio = 100.0 * float(trainable) / float(total) if total else 0.0
    print(f"[DUET][PARAMS] trainable={trainable:,} total={total:,} ratio={ratio:.4f}%", flush=True)
    for group in sorted(groups):
        group_total = groups[group]["total"]
        group_trainable = groups[group]["trainable"]
        group_ratio = 100.0 * float(group_trainable) / float(group_total) if group_total else 0.0
        print(
            f"[DUET][PARAMS] {group}: trainable={group_trainable:,} total={group_total:,} "
            f"ratio={group_ratio:.4f}%",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Sampler, trainer and callbacks
# ---------------------------------------------------------------------------


def _split_to_even_chunks(
    indices: list[int], lengths: Sequence[int], num_chunks: int
) -> list[list[int]]:
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
    """Released LLaVA modality/length order, without sample repetition.

    DUET additionally *needs* the modality grouping: the collator refuses a
    micro-batch that mixes image and text-only rows.
    """

    def __init__(
        self,
        modality_lengths: Sequence[int],
        *,
        per_device_batch_size: int,
        world_size: int,
        gradient_accumulation_steps: int,
        seed: int,
    ) -> None:
        self.modality_lengths = [int(value) for value in modality_lengths]
        if any(length == 0 for length in self.modality_lengths):
            raise ValueError(
                "modality_lengths entries must be non-zero; use positive lengths for image rows "
                "and negative lengths for text rows."
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

    @property
    def _kept(self) -> int:
        """Rows emitted per epoch when both modalities are present.

        Two alignments are needed, and only the first is about modality:

        * each modality's remainder is trimmed to the micro-batch stride, so no
          micro-batch straddles the image/text junction;
        * the total is then trimmed to a whole global batch, so every rank gets
          the same number of full micro-batches.

        Without the second, the epoch ends with a partial batch that the ranks
        divide unevenly; the collective then hangs until NCCL's watchdog aborts
        the job. Together they also mean the loader never has a short final batch
        to pad by cycling, which is what mixed the modalities to begin with.
        """
        if not self.image_indices or not self.text_indices:
            return len(self.modality_lengths)
        stride = self.per_device_batch_size * self.world_size

        def keep(count: int) -> int:
            tail = (count - 1) % self.megabatch_size + 1
            return count - tail % stride

        total = keep(len(self.image_indices)) + keep(len(self.text_indices))
        return total - total % self.megabatch_size

    def __len__(self) -> int:
        return self._kept

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if not self.modality_lengths:
            return iter(())
        if not self.image_indices or not self.text_indices:
            return iter(
                self._length_grouped_order(list(range(len(self.modality_lengths))), generator)
            )

        image_order = self._length_grouped_order(self.image_indices, generator)
        text_order = self._length_grouped_order(self.text_indices, generator)
        image_megabatches = [
            image_order[index : index + self.megabatch_size]
            for index in range(0, len(image_order), self.megabatch_size)
        ]
        text_megabatches = [
            text_order[index : index + self.megabatch_size]
            for index in range(0, len(text_order), self.megabatch_size)
        ]
        image_tail = image_megabatches[-1]
        text_tail = text_megabatches[-1]
        megabatches = image_megabatches[:-1] + text_megabatches[:-1]
        if megabatches:
            order = torch.randperm(len(megabatches), generator=generator).tolist()
            megabatches = [megabatches[index] for index in order]
        # Upstream LLaVA ends the epoch with `image_tail + text_tail` as one
        # megabatch. LLaVA's collator tolerates that; DUET's does not -- it requires
        # same-modality micro-batches and raises on the last step of the epoch.
        #
        # Emitting the two remainders as separate megabatches is necessary but not
        # sufficient: the flat index stream is cut into micro-batches at a fixed
        # stride, so a remainder whose length is not a multiple of that stride
        # leaves the following modality starting mid-micro-batch. Each remainder is
        # therefore trimmed to the stride, and `_kept` then trims the total to a
        # whole global batch so no short final batch survives at all -- see `_kept`
        # for why leaving one ragged hangs the job rather than fixing it.
        #
        # Together these give up under one global batch of rows per epoch (81 of
        # 665,297 at 4 GPUs x micro-batch 8 x accumulation 4, i.e. 0.012%), and say
        # so on stdout. The alternative is letting the loader pad by repeating rows,
        # which trains on duplicates and mixes the modalities anyway.
        stride = self.per_device_batch_size * self.world_size
        for tail in (image_tail, text_tail):
            if tail:
                megabatches.append(sorted(tail[: len(tail) // stride * stride]))
        order = [index for megabatch in megabatches for index in megabatch]
        # Trim to a whole global batch. Every block above is a multiple of the
        # micro-batch stride and single-modality, so cutting a stride-aligned
        # amount off the end keeps both properties.
        kept = self._kept
        dropped = len(self.modality_lengths) - kept
        if dropped:
            print(
                f"[DUET][SAMPLER] dropped {dropped} row(s) from the epoch tail: remainders "
                f"trimmed to the micro-batch stride ({stride}) so no micro-batch spans both "
                f"modalities, and the total trimmed to a whole global batch "
                f"({self.megabatch_size}) so the ranks stay in step",
                flush=True,
            )
        return iter(order[:kept])

    def _length_grouped_order(self, indices: list[int], generator: torch.Generator) -> list[int]:
        if not indices:
            return []
        shuffled = [
            indices[pos] for pos in torch.randperm(len(indices), generator=generator).tolist()
        ]
        grouped_order: list[int] = []
        for idx in range(0, len(shuffled), self.megabatch_size):
            megabatch = shuffled[idx : idx + self.megabatch_size]
            megabatch = sorted(megabatch, key=lambda index: self.lengths[index], reverse=True)
            balanced = _split_to_even_chunks(megabatch, self.lengths, self.length_group_world_size)
            grouped_order.extend(index for chunk in balanced for index in chunk)
        return grouped_order


class DuetTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        group_by_modality_length: bool = True,
        stage: str = "finetune",
        backbone: str = "llava",
        stage1_module_path: str = "model.multi_modal_projector",
        stage1_artifact_name: str = "mm_projector.bin",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.group_by_modality_length = bool(group_by_modality_length)
        self.stage = str(stage)
        self.backbone = str(backbone)
        self.stage1_module_path = str(stage1_module_path)
        self.stage1_artifact_name = str(stage1_artifact_name)
        # The wrapper accepts **kwargs for HF compatibility but intentionally uses
        # the released per-microbatch mean CE loss. Prevent Transformers 5 from
        # injecting num_items_in_batch and changing accumulation scaling.
        self.model_accepts_loss_kwargs = False

    def _get_train_sampler(self, train_dataset: Any | None = None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if dataset is None:
            return None
        if not self.group_by_modality_length or not hasattr(dataset, "modality_lengths"):
            try:
                return super()._get_train_sampler(dataset)
            except TypeError:  # transformers < 5 takes no argument
                return super()._get_train_sampler()
        sampler = ModalityLengthGroupedSampler(
            dataset.modality_lengths,
            per_device_batch_size=int(self.args.per_device_train_batch_size),
            world_size=max(1, int(getattr(self.args, "world_size", 1))),
            gradient_accumulation_steps=max(1, int(self.args.gradient_accumulation_steps)),
            seed=int(self.args.seed),
        )
        if self.is_world_process_zero():
            print(
                f"[DUET][SAMPLER] modality-length grouping: "
                f"image_rows={len(sampler.image_indices)} "
                f"text_rows={len(sampler.text_indices)} "
                f"samples_without_repetition={len(sampler)}",
                flush=True,
            )
        return sampler

    def _save(
        self, output_dir: str | None = None, state_dict: dict[str, torch.Tensor] | None = None
    ) -> None:
        destination = Path(output_dir or self.args.output_dir)
        if self.stage == "pretrain":
            # Match released LLaVA adapter tuning: do not emit a redundant
            # consolidated 7B model at every step. Trainer still saves the
            # DeepSpeed shards needed for an exact resume.
            if state_dict is None:
                state_dict = self.model.state_dict()
            stage1_state = _stage1_state_from_full_state_dict(state_dict, self.stage1_module_path)
            if not stage1_state:
                raise RuntimeError(
                    "Could not find DUET stage-1 module weights in the pretrain state."
                )
            destination.mkdir(parents=True, exist_ok=True)
            torch.save(stage1_state, destination / self.stage1_artifact_name)
        else:
            super()._save(output_dir=str(destination), state_dict=state_dict)
        write_duet_metadata(destination, self.model.compressor, backbone=self.backbone)


class SaveControlCallback(TrainerCallback):
    """Keep step checkpoints on the configured cadence across a resume."""

    @staticmethod
    def _sync_save_steps(args: TrainingArguments, state: Any) -> None:
        save_steps = int(getattr(args, "save_steps", 0) or 0)
        if save_steps > 0 and int(getattr(state, "save_steps", 0) or 0) != save_steps:
            state.save_steps = save_steps

    def on_train_begin(
        self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any
    ) -> Any:
        self._sync_save_steps(args, state)
        return control

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        self._sync_save_steps(args, state)
        return control


class StopAfterStepCallback(TrainerCallback):
    """Bounded smoke/capacity runs: stop early without disturbing the schedule."""

    def __init__(self, stop_after_steps: int | None) -> None:
        self.stop_after_steps = int(stop_after_steps) if stop_after_steps is not None else None

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.stop_after_steps is None:
            return control
        if int(getattr(state, "global_step", 0) or 0) >= self.stop_after_steps:
            control.should_training_stop = True
        return control


class ResourceReportCallback(TrainerCallback):
    """Step timing and peak memory, so a budget's cost is measured, not guessed."""

    def __init__(
        self,
        *,
        stage: str,
        budget: int | float | str,
        stage1_tokens: int | None,
        layer_average: float | None,
        global_batch_size: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        dataloader_num_workers: int,
        model_max_length: int,
        dataset_length: int,
        max_steps: int,
        stop_after_steps: int | None,
        resource_report_path: str | None,
        log_memory_every_n_steps: int,
    ) -> None:
        self.stage = stage
        self.budget = budget
        self.stage1_tokens = None if stage1_tokens is None else int(stage1_tokens)
        self.layer_average = None if layer_average is None else float(layer_average)
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

    def on_train_begin(
        self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_begin(
        self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any
    ) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def on_step_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs: Any) -> None:
        if self._step_start is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.step_seconds.append(float(time.perf_counter() - self._step_start))
        if (
            self.log_memory_every_n_steps > 0
            and state.global_step % self.log_memory_every_n_steps == 0
            and torch.cuda.is_available()
            and bool(getattr(state, "is_world_process_zero", True))
        ):
            allocated = torch.cuda.max_memory_allocated() / (1024**3)
            reserved = torch.cuda.max_memory_reserved() / (1024**3)
            print(
                f"[DUET][RESOURCE] step={state.global_step} "
                f"max_allocated_gb={allocated:.2f} max_reserved_gb={reserved:.2f}",
                flush=True,
            )

    def build_report(self) -> dict[str, Any]:
        warmup = min(2, len(self.step_seconds))
        measured = self.step_seconds[warmup:] or self.step_seconds
        avg_seconds = float(sum(measured) / len(measured)) if measured else 0.0
        steps_per_epoch = (
            int(math.ceil(float(self.dataset_length) / float(self.global_batch_size)))
            if self.global_batch_size > 0
            else 0
        )
        allocated = (
            torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        )
        reserved = (
            torch.cuda.max_memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
        )
        return {
            "stage": self.stage,
            "budget": self.budget,
            "stage1_tokens": self.stage1_tokens,
            "measured_layer_average_tokens": self.layer_average,
            "attn_implementation": ATTN_IMPLEMENTATION,
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


def _deepspeed_config(
    stage: int,
    global_batch_size: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    if int(stage) not in (2, 3):
        raise ValueError(
            f"DUET training only supports DeepSpeed ZeRO-2 or ZeRO-3, got ZeRO-{stage}."
        )
    zero_optimization: dict[str, Any] = {
        "stage": int(stage),
        "overlap_comm": True,
        "contiguous_gradients": True,
    }
    if int(stage) == 3:
        # The sidecar's ZeRO-3 settings: the pruned decoder allocates and frees
        # sequence-shaped buffers at every boundary, which the default overlapped
        # reduce path handles badly.
        zero_optimization.update(
            {
                "overlap_comm": False,
                "contiguous_gradients": False,
                "reduce_bucket_size": 10_000_000,
                "stage3_prefetch_bucket_size": 10_000_000,
                "stage3_param_persistence_threshold": 10_000,
                "stage3_max_live_parameters": 10_000_000,
                "stage3_max_reuse_distance": 10_000_000,
                "stage3_gather_16bit_weights_on_model_save": True,
            }
        )
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
        raise RuntimeError(
            "Trainer did not expose the prepared model wrapper for final serialization."
        )
    return trainer.accelerator.get_state_dict(wrapped)


# ---------------------------------------------------------------------------
# LLMOE tr / ft entry points
# ---------------------------------------------------------------------------


def _dataset_check(config: ExpConfig, stage: str) -> str:
    dataset_dir = dataset_dir_for_stage(config.input_dir, stage)
    name = dataset_dir.name
    if not dataset_dir.is_dir():
        return f"Missing {name} dataset under {config.input_dir}"
    try:
        find_annotations(dataset_dir, stage)
    except FileNotFoundError:
        return f"Missing {name} JSON annotations under {dataset_dir}"
    return f"OK with {name}"


def duet_tr_fn_cd(config: ExpConfig) -> str:
    return _dataset_check(config, "pretrain")


def duet_ft_fn_cd(config: ExpConfig) -> str:
    return _dataset_check(config, "finetune")


@dataclass
class DuetTrainingArgs:
    """Resolved training settings; serialized verbatim into ``training_args_duet.json``."""

    model_name_or_path: str
    data_path: str
    image_folder: str
    output_dir: str
    stage: str
    backbone: str
    budget: int | float | str
    budget_tag: str
    stage1_tokens: int | None
    stage1_module_path: str
    stage1_artifact_name: str
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
    logging_steps: int
    seed: int
    resume_from_checkpoint: str | None
    skip_final_save: bool
    dataset_limit: int | None
    dataloader_num_workers: int
    resource_report_path: str | None
    log_memory_every_n_steps: int


def _mapping_option(config: Config, key: str) -> dict[str, Any]:
    options = config.custom_args.get(key, {})
    if not isinstance(options, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return dict(options)


def _training_backbone(backbone: VLM) -> str:
    identity = (type(backbone).__name__ + " " + str(backbone._model_id)).lower()
    return "qwen" if "qwen" in identity else "llava"


def _compressor_for_training(
    config: Config, *, backbone: str = "llava", depth: int | None = None
) -> DUETCompressor:
    """Build and resolve the schedule for one training backbone.

    ``depth`` is the decoder's layer count. It has to be passed: Stage 2's
    boundaries are fractions of the decoder, and `install` resolves them the same
    way at evaluation time, so training without it would train under one schedule
    and evaluate under another.
    """
    compressor_args = _mapping_option(config, "compressor_args")
    compressor_args.pop("track_source", None)
    if backbone == "qwen" and compressor_args.get("pretrained") not in (
        None,
        False,
    ):
        raise ValueError(
            "DUET has no released Qwen training weights; remove compressor_args.pretrained."
        )
    if backbone == "qwen":
        compressor_args.pop("pretrained", None)
    compressor = DUETCompressor(**compressor_args)
    if backbone == "qwen":
        if compressor._retain_ratio is None:
            raise ValueError(
                "DUET Qwen training takes retain_ratio, the layer-averaged budget; "
                "max_tokens selects a LLaVA profile and absolute dominant/contextual "
                "counts are not comparable across NaViT's variable token counts."
            )
    elif compressor._retain_ratio is not None:
        raise ValueError(
            "retain_ratio selects DUET's Qwen training path; use max_tokens with "
            "the LLaVA backbone."
        )
    (
        compressor._pdrop_layers,
        compressor._pdrop_keep_ratios,
    ) = compressor._resolve_pdrop(backbone, depth)
    return compressor


def _budget_tag(compressor: DUETCompressor, backbone: str) -> str:
    if backbone == "qwen":
        return f"r{compressor._retain_ratio:g}"
    return str(compressor.budget)


def _stage_output_dir(
    config: Config,
    stage: str,
    budget_tag: str,
    *,
    backbone: str = "llava",
    weights_key: str | None = None,
) -> Path:
    """Where one training stage writes.

    ``backbone`` is the *recipe* (``llava``/``qwen``); ``weights_key`` is the model
    it is being run on. They differ for LLaVA-1.5-13B, which takes the LLaVA recipe
    on a different model, and it gets its own tag so its checkpoints cannot
    overwrite the 7B ones or be mistaken for them later.
    """
    root = Path(config.output_dir) / "duet"
    default_tag = "qwen2.5-vl-7b" if backbone == "qwen" else "llava-v1.5-7b"
    model_tag = (
        default_tag
        if weights_key in (None, "qwen-vl", "llava")
        else BACKBONE_SLUGS.get(weights_key, weights_key)
    )
    name = f"{model_tag}-duet-{budget_tag}"
    return root / (f"{name}-pretrain" if stage == "pretrain" else name)


def _resolve_adapter_path(value: str | Path, artifact_name: str = "mm_projector.bin") -> Path:
    candidate = Path(value)
    return candidate / artifact_name if candidate.is_dir() else candidate


def _build_training_args(
    backbone: VLM,
    config: Config,
    stage: str,
    compressor: DUETCompressor,
    *,
    backbone_kind: str = "llava",
) -> DuetTrainingArgs:
    """Resolve ``training_args`` into the released recipe, rejecting deviations."""
    options = _mapping_option(config, "training_args")
    defaults = RELEASED_STAGE_DEFAULTS[stage]
    budget = compressor.budget
    budget_tag = _budget_tag(compressor, backbone_kind)
    stage1_module_path = (
        "model.visual.merger" if backbone_kind == "qwen" else "model.multi_modal_projector"
    )
    stage1_artifact_name = "mm_merger.bin" if backbone_kind == "qwen" else "mm_projector.bin"

    per_device_train_batch_size = int(options.get("per_device_train_batch_size", 1))
    if per_device_train_batch_size < 1:
        raise ValueError(
            f"per_device_train_batch_size must be >= 1, got {per_device_train_batch_size}."
        )
    global_batch_size = int(options.get("global_batch_size", defaults["global_batch_size"]))
    if global_batch_size != defaults["global_batch_size"]:
        raise ValueError(
            f"DUET {stage} must preserve global_batch_size={defaults['global_batch_size']}; got "
            f"{global_batch_size}. Tune only micro-batch and gradient accumulation."
        )
    deepspeed_stage = int(options.get("deepspeed_stage", defaults["deepspeed_stage"]))
    if deepspeed_stage != defaults["deepspeed_stage"]:
        raise ValueError(
            f"DUET {stage} must use DeepSpeed ZeRO-{defaults['deepspeed_stage']}; got "
            f"ZeRO-{deepspeed_stage}."
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_global = max(1, per_device_train_batch_size * world_size)
    gradient_accumulation_steps = options.get("gradient_accumulation_steps")
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = max(1, math.ceil(global_batch_size / micro_global))
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    effective_global_batch = micro_global * gradient_accumulation_steps
    if effective_global_batch != global_batch_size:
        raise ValueError(
            "DUET requires an exact global batch product: per_device_train_batch_size * "
            f"WORLD_SIZE * gradient_accumulation_steps = {effective_global_batch}, requested "
            f"global_batch_size={global_batch_size}."
        )

    released_length = (
        QWEN_MODEL_MAX_LENGTH if backbone_kind == "qwen" else RELEASED_MODEL_MAX_LENGTH
    )
    model_max_length = int(options.get("model_max_length", released_length))
    if backbone_kind != "qwen" and model_max_length != RELEASED_MODEL_MAX_LENGTH:
        raise ValueError(
            f"DUET training pins model_max_length={RELEASED_MODEL_MAX_LENGTH} on "
            f"LLaVA, the length its released recipe was tuned at; got "
            f"{model_max_length}."
        )
    if model_max_length < 1:
        raise ValueError(f"model_max_length must be >= 1, got {model_max_length}")
    attn_implementation = str(options.get("attn_implementation", ATTN_IMPLEMENTATION))
    if attn_implementation != ATTN_IMPLEMENTATION:
        raise ValueError(
            "DUET training is FlashAttention-2 only; got "
            f"attn_implementation={attn_implementation!r}."
        )
    save_steps = int(options.get("save_steps", defaults["save_steps"]))
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
            raise ValueError(
                "DUET stage-1 pretraining starts from the base checkpoint and cannot "
                "load pretrain_mm_mlp_adapter."
            )
        adapter_path = None
    else:
        adapter_path = (
            _resolve_adapter_path(adapter, stage1_artifact_name)
            if adapter
            else _stage_output_dir(
                config,
                "pretrain",
                budget_tag,
                backbone=backbone_kind,
                weights_key=backbone_key(backbone),
            )
            / stage1_artifact_name
        )
        if not adapter_path.is_file():
            raise FileNotFoundError(
                f"Full finetuning requires the matching {budget_tag} stage-1 "
                f"artifact from tr: {adapter_path} is missing. Run tr for this "
                "schedule first, or set "
                "training_args.pretrain_mm_mlp_adapter."
            )

    data_path, image_folder = resolve_training_data(
        config.input_dir,
        stage,
        data_path=options.get("data_path"),
        image_folder=options.get("image_folder"),
    )
    output_dir = (
        Path(options["output_dir"])
        if options.get("output_dir")
        else _stage_output_dir(
            config,
            stage,
            budget_tag,
            backbone=backbone_kind,
            weights_key=backbone_key(backbone),
        )
    )
    dataset_limit = config.dataset_limit
    if options.get("dataset_limit") is not None:
        option_limit = int(options["dataset_limit"])
        dataset_limit = option_limit if dataset_limit is None else min(dataset_limit, option_limit)
    resource_report_path = options.get("resource_report_path")
    if resource_report_path is None:
        resource_report_path = output_dir / f"resource_{stage}_{budget_tag}.json"

    return DuetTrainingArgs(
        model_name_or_path=str(backbone._model_id),
        data_path=str(data_path),
        image_folder=str(image_folder),
        output_dir=str(output_dir),
        stage=stage,
        backbone=backbone_kind,
        budget=budget,
        budget_tag=budget_tag,
        stage1_tokens=(
            None if compressor._stage1_tokens is None else int(compressor._stage1_tokens)
        ),
        stage1_module_path=stage1_module_path,
        stage1_artifact_name=stage1_artifact_name,
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
        group_by_modality_length=bool(options.get("group_by_modality_length", True)),
        max_steps=int(options.get("max_steps", -1)),
        stop_after_steps=stop_after_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=int(options.get("logging_steps", 10)),
        seed=int(options.get("seed", 42)),
        resume_from_checkpoint=(
            str(options["resume_from_checkpoint"])
            if options.get("resume_from_checkpoint")
            else None
        ),
        skip_final_save=bool(options.get("skip_final_save", False)),
        dataset_limit=None if dataset_limit is None else int(dataset_limit),
        dataloader_num_workers=int(options.get("dataloader_num_workers", 2)),
        resource_report_path=str(resource_report_path),
        log_memory_every_n_steps=int(options.get("log_memory_every_n_steps", 5)),
    )


def _run_training_stage(backbone: VLM, config: Config, stage: str) -> None:
    backbone_kind = _training_backbone(backbone)
    compressor = _compressor_for_training(
        config, backbone=backbone_kind, depth=config_decoder_depth(backbone)
    )
    args = _build_training_args(
        backbone,
        config,
        stage,
        compressor,
        backbone_kind=backbone_kind,
    )
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    rank = int(os.environ.get("RANK", "0"))
    local_files_only = bool(getattr(backbone, "_local_files_only", True))
    if backbone_kind == "qwen":
        if rank == 0:
            warnings.warn(
                "DUET Qwen training is an adaptation of the released LLaVA "
                "recipe; upstream publishes Qwen inference but no Qwen "
                "training recipe or checkpoint.",
                stacklevel=3,
            )
        processor, base_model = HIDROP_BACKENDS["qwen"].load_training_base(
            args.model_name_or_path,
            stage=stage,
            dtype=torch.bfloat16 if args.bf16 else None,
            attn_implementation=args.attn_implementation,
            model_max_length=args.model_max_length,
            local_files_only=local_files_only,
        )
    else:
        processor, base_model = load_hf_llava_training_base(
            args.model_name_or_path,
            stage=stage,
            dtype=torch.bfloat16 if args.bf16 else None,
            attn_implementation=args.attn_implementation,
            model_max_length=args.model_max_length,
            local_files_only=local_files_only,
        )
        base_model.config.image_seq_length = args.stage1_tokens
        if hasattr(base_model.config, "num_image_tokens"):
            base_model.config.num_image_tokens = args.stage1_tokens
    base_model.config.use_cache = False
    base_model.config.text_config.use_cache = False
    compressor.to(device=base_model.device, dtype=base_model.dtype)
    if backbone_kind == "qwen":
        model = DuetTrainingQwenForConditionalGeneration(base_model, compressor, processor)
        stage1_module = base_model.model.visual.merger
    else:
        compressor.install(_TrainingBackbone(base_model, processor))
        model = DuetTrainingLlavaForConditionalGeneration(base_model, compressor)
        stage1_module = base_model.model.multi_modal_projector

    if stage == "pretrain":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in stage1_module.parameters():
            parameter.requires_grad_(True)
    elif args.pretrain_mm_mlp_adapter:
        if backbone_kind == "qwen":
            loaded_state = torch.load(
                args.pretrain_mm_mlp_adapter,
                map_location="cpu",
                weights_only=True,
            )
            stage1_module.load_state_dict(loaded_state, strict=True)
            loaded = loaded_state
        else:
            loaded = load_pretrained_projector(args.pretrain_mm_mlp_adapter, stage1_module)
        if rank == 0:
            print(
                f"Loaded {len(loaded)} stage-1 tensors from {args.pretrain_mm_mlp_adapter}",
                flush=True,
            )
    if backbone_kind == "qwen":
        base_model.model.visual.requires_grad_(False)
        for parameter in stage1_module.parameters():
            parameter.requires_grad_(True)
    elif getattr(base_model.model, "vision_tower", None) is not None:
        base_model.model.vision_tower.requires_grad_(False)
    _report_trainable_parameters(model)

    gradient_checkpointing_kwargs = None
    if args.gradient_checkpointing:
        if backbone_kind == "qwen":
            gradient_checkpointing_kwargs = {"use_reentrant": True}
        else:
            # LLaVA's pruning hooks change sequence length around recomputation.
            gradient_checkpointing_kwargs = {"use_reentrant": False}
            if args.deepspeed_stage == 3:
                gradient_checkpointing_kwargs["determinism_check"] = "none"
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    dataset = LlavaJsonDataset(args.data_path, args.image_folder, dataset_limit=args.dataset_limit)
    if backbone_kind == "qwen":
        collator = QwenHiDropCollator(
            processor=processor,
            model_max_length=args.model_max_length,
            stage=stage,
        )
    else:
        collator = LlavaDuetCollator(
            processor=processor,
            model_max_length=args.model_max_length,
            visual_tokens=args.stage1_tokens,
            prompt_version=RELEASED_STAGE_DEFAULTS[stage]["prompt_version"],
        )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
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
            args.per_device_train_batch_size,
            args.gradient_accumulation_steps,
        ),
    )
    report_callback = ResourceReportCallback(
        stage=stage,
        budget=args.budget,
        stage1_tokens=args.stage1_tokens,
        layer_average=(
            None
            if backbone_kind == "qwen"
            else average_image_tokens(
                compressor._stage1_tokens,
                compressor._pdrop_layers,
                compressor._pdrop_keep_ratios,
                len(base_model.model.language_model.layers),
            )
        ),
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
    if rank == 0:
        write_duet_metadata(
            Path(args.output_dir),
            compressor,
            asdict(args),
            backbone=backbone_kind,
        )
    trainer = DuetTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[
            SaveControlCallback(),
            StopAfterStepCallback(args.stop_after_steps),
            report_callback,
        ],
        group_by_modality_length=args.group_by_modality_length,
        stage=stage,
        backbone=backbone_kind,
        stage1_module_path=args.stage1_module_path,
        stage1_artifact_name=args.stage1_artifact_name,
    )
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    finally:
        if backbone_kind == "qwen":
            model.remove_training_hooks()
        else:
            compressor.uninstall()

    if args.skip_final_save:
        if trainer.is_world_process_zero():
            write_duet_metadata(
                Path(args.output_dir),
                compressor,
                asdict(args),
                backbone=backbone_kind,
            )
            report_callback.write_report()
        return

    if stage == "pretrain":
        # Stage 1 saves only the projector/merger, not a frozen 7B copy.
        unwrapped = trainer.accelerator.unwrap_model(model)
        if trainer.is_world_process_zero():
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(output)
            unwrapped.base_model.config.save_pretrained(output)
            if getattr(unwrapped.base_model, "generation_config", None) is not None:
                unwrapped.base_model.generation_config.save_pretrained(output)
            stage1_module = (
                unwrapped.base_model.model.visual.merger
                if backbone_kind == "qwen"
                else unwrapped.base_model.model.multi_modal_projector
            )
            artifact_path = output / args.stage1_artifact_name
            torch.save(
                {key: value.detach().cpu() for key, value in stage1_module.state_dict().items()},
                artifact_path,
            )
            write_duet_metadata(output, compressor, asdict(args), backbone=backbone_kind)
            report_callback.write_report()
            print(
                f"Saved DUET-{args.budget_tag} stage-1 adapter to {artifact_path}",
                flush=True,
            )
        return

    state_dict = _gather_final_state_dict(trainer)
    if trainer.is_world_process_zero():
        model.save_pretrained(args.output_dir, state_dict=state_dict, safe_serialization=True)
        processor.save_pretrained(args.output_dir)
        write_duet_metadata(Path(args.output_dir), compressor, asdict(args), backbone=backbone_kind)
        report_callback.write_report()
        print(
            f"Saved DUET-{args.budget_tag} checkpoint to {args.output_dir}. Benchmark it with "
            "backbone_args.model_name_or_path.",
            flush=True,
        )


def train_duet(backbone: VLM, config: Config) -> None:
    """`tr`: pretrain this budget's projector on 558K captions under the schedule."""
    if getattr(backbone, "compressor_family", None) == "llava_next":
        from llmoe.compressor.duet.duet_next_tr import run_next_training_stage

        run_next_training_stage(backbone, config, "pretrain")
        return
    _run_training_stage(backbone, config, "pretrain")


def finetune_duet(backbone: VLM, config: Config) -> None:
    """`ft`: full finetune on 665K instructions from this budget's projector."""
    if getattr(backbone, "compressor_family", None) == "llava_next":
        from llmoe.compressor.duet.duet_next_tr import run_next_training_stage

        run_next_training_stage(backbone, config, "finetune")
        return
    _run_training_stage(backbone, config, "finetune")
