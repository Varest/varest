"""HiDrop: hierarchical vision-token reduction via late injection, concave pyramid
pruning, and early exit.

Paper: Wu et al., "HiDrop: Hierarchical Vision Token Reduction in MLLMs via Late
Injection, Concave Pyramid Pruning, and Early Exit" (ICLR 2026, arXiv:2602.23699).
Reference code: https://github.com/EIT-NLP/HiDrop (upstream commit
``5dfc02b7de75aabe565c2040ad50498b70086185``), ported here from the
``llmoe_baseline_hidrop`` sidecar that ran it against this harness.

HiDrop is not a vision-encoder compressor: every one of LLaVA-1.5's 576 projected
visual tokens is built and handed to the decoder unchanged. What HiDrop changes is
*which layers see them*:

* **Late vision injection (§3.1).** Layers 1-8 run text-only. Layer 9 -- the
  injection point ``L_inj``, a local minimum of the visual layer-wise similarity
  curve -- receives the full 576 original projected visual embeddings, spliced
  back in at their original positions. The paper's evidence is that shallow layers
  act as passive conduits where cross-modal fusion has not yet started.
* **Concave pyramid pruning (§3.2).** After the filtering layers 10, 14, 16, and
  18 (local maxima of the paper's inter-layer visual attention similarity metric,
  ILVAS) the *next* decoder block's q/k projections score the surviving visual
  tokens -- causal softmax over the complete sequence, then the visual slice --
  and DTop-K keeps a scheduled count. The keep curve is concave: one aggressive
  cut at the start of fusion, then a long tail.
* **Early vision exit (§3.1).** Every remaining visual token is dropped before
  layer 25, where the paper's deep-to-shallow masking shows performance plateaus,
  so the deep layers reason language-only.
* **Persistent position encoding (§3.3).** Position IDs travel with their tokens
  and are never renumbered, so surviving text keeps its original RoPE phase and
  the KV cache stays valid across every removal.

Budget semantics
----------------
``average_retained_tokens`` is the **layer-averaged** visual-token count
(``sum(tokens held by each of the 32 decoder layers) / 32``), the same convention
``sparse_vlm.py``, ``must_drop.py`` and ``duet.py`` use -- not a final retained
count. A "64-token" HiDrop still gives layer 9 all 576 tokens and ends with 1.
The four named profiles are:

====== ================= ==================================== ========
budget filtering layers  visual tokens kept after each filter  schedule
====== ================= ==================================== ========
224    10, 14, 16, 18    474, 415, 412, 411                    derived
128    10, 14, 16, 18    318, 173, 168, 165                    derived
64     10, 14, 16, 18    215, 10, 5, 1                         released
32     9, 10, 14, 16, 18 132, 70, 10, 5, 1                     derived
====== ================= ==================================== ========

Only the 64-token schedule is the released one; the other three were derived by
the sidecar (``config.HIDROP_PROFILES`` records how each). The schedules are
immutable per budget: a different schedule with the same average is rejected,
because a HiDrop checkpoint is trained for one exact schedule.

Trained method
--------------
HiDrop is **not training-free**: the schedule is applied during LLaVA projector
pretraining and full finetuning, so the model learns to fuse from a late-injected,
progressively pruned visual stream. Running the schedule on a stock LLaVA
checkpoint is an ablation, not the paper's setting -- ``install()`` warns when no
trained checkpoint is given.

Produce one with the registered workers (see ``hidrop_tr.py``)::

    torchrun --nproc_per_node=8 -m llmoe --compressor HiDrop \\
      --custom_args hidrop_pretrain.yaml ... tr    # 558K projector, per budget
    torchrun --nproc_per_node=8 -m llmoe --compressor HiDrop \\
      --custom_args hidrop_finetune.yaml ... ft    # 665K full finetune

Then benchmark it. ``compressor_args.checkpoint_path`` takes the ``ft`` output
directory (or a Hugging Face repo id, resolved with ``snapshot_download``). The
backbone loads lazily, so ``install()`` simply points it at the checkpoint before
any weights are read -- ``transformers`` then loads the trained model itself::

    compressor_args:
      checkpoint_path: /results/hidrop/llava-v1.5-7b-hidrop-64-flash_attention_2
    batch_size: 1

The schedule is read from ``<checkpoint>/hidrop_config.json``, which fails closed
on a missing or older ``implementation_version``, and ``run_manifest.json`` is
checked so a checkpoint that never finished full finetuning cannot be reported as
one (``validate_provenance: false`` opts out for hand-assembled checkpoints). A
stage-1 directory holding only ``mm_projector.bin`` is also accepted -- the
projector is loaded over the stock backbone -- and flagged as the ablation it is.
Pointing ``backbone_args.model_name_or_path`` at the same directory is equivalent
and skips the second load; ``compressor_args: {average_retained_tokens: 64}``
alone selects a named profile with no weights of its own.

Faithfulness notes / harness adaptations
----------------------------------------
* **Where the work happens.** ``compress()`` is a deliberate no-op: HiDrop needs
  the full 576 tokens to reach the decoder. ``install()`` instead swaps
  ``vlm.model.model.language_model`` for :class:`HiDropLlamaModel`, which reuses
  the original embeddings, layers, norm, and rotary embedding and only changes the
  sequence. That is the released sidecar's design, unchanged.

* **LLaVA-1.5 only.** The profiles assume 576 CLIP tokens in one contiguous span
  and a 32-layer LLaMA decoder, so ``install()`` rejects a non-LLaVA backbone
  rather than mis-scheduling it.

* **No parallel decoupling.** The paper's third implementation trick (§3.3) runs
  the vision tower and projector *concurrently* with the text-only shallow layers
  and concatenates cached visual KV at the injection layer. This port keeps the
  sidecar's sequential path: the same layer sees the same inputs, so outputs are
  identical, but the prefill-latency half of the paper's speedup is not realized
  here. Token-count savings are.

* **``batch_size: 1`` is enforced, not preferred.** Batched generation left-pads,
  and the trained pruning state does not survive that; the sidecar's own preflight
  refused any other batch size. ``prepare_inputs`` and ``generate`` are wrapped to
  fail loudly instead of silently producing a differently-pruned sequence. Run
  independent benchmark workers to parallelize.

* **Prompt boundary.** The sidecar subclassed the backbone to restore the
  ``<image> \\n`` SentencePiece boundary LLaVA's native tokenizer helper produces,
  so evaluation tokenizes the way training did. The harness resolves backbones
  from a registry, so ``install()`` wraps ``vlm.format_prompts`` instead; the
  substitution is idempotent and a no-op on prompts without ``<image>\\n``.

* **Processor placeholder count.** Also from the sidecar's backbone subclass:
  ``num_additional_image_tokens`` is recomputed from the checkpoint's own vision
  config so ``<image>`` expands to exactly ``image_seq_length`` placeholders. It is
  a no-op on a stock ``llava-hf`` processor and only matters for checkpoints whose
  saved processor config drifted.

* **Attention backend.** The wrapper supports ``eager``, ``sdpa``, and
  ``flash_attention_2``; anything else is switched to ``sdpa`` at install and
  restored at uninstall. The sidecar additionally ran the whole evaluation inside
  ``sdpa_kernel(FLASH_ATTENTION)`` to match its FA2 training numerics, which here
  wraps ``generate`` instead (``compressor_args: {sdpa_flash_only: false}`` opts
  out). It is skipped off CUDA, where that backend does not exist.

* **Source history.** ``get_source_history()`` returns one ``[T_cur, T_orig]``
  matrix per pruning stage plus a leading identity, which is what
  ``llmoe.viz`` renders; the sidecar kept the whole stacked batch instead. Rows are
  one-hot -- HiDrop prunes, it never merges.

* **Training.** HiDrop has no learned parameters of its *own* -- the schedule
  trains the LLaVA model around it -- so ``tr`` and ``ft`` map onto the recipe's
  two stages (projector pretraining, then full finetuning) rather than onto a
  compressor optimizer. ``hidrop_tr.py`` keeps the released semantics; the
  sidecar's Slurm/PBS launcher scripts are not ported, since their knobs are
  ``training_args`` and the launcher is site-specific.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import nullcontext
from os import PathLike
from pathlib import Path
from typing import Any, ContextManager, override
import json
import os
import warnings

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from llmoe.backbone.prototype import VLM
from llmoe.compressor.hidrop.config import (
    CONFIG_METADATA_KEYS,
    HiDropConfig,
    HiDropDebugger,
    QwenHiDropConfig,
    llava_next_profile_for_ratio,
)
from llmoe.compressor.hidrop.hidrop_llama import (
    IGNORE_INDEX,
    HiDropForwardContext,
    HiDropLlamaModel,
    VisualTokenSpan,
)
from llmoe.compressor.hidrop.hidrop_llava_next import (
    LlavaNextVisualLayout,
    layout_from_model,
    llava_next_visualization_views,
    resolve_scaled_schedule,
)
from llmoe.compressor.hidrop.hidrop_qwen import HiDropQwenForwardContext, HiDropQwenModel
from llmoe.compressor.hidrop.provenance import (
    is_pinned_llava_next_snapshot,
    validate_inference_checkpoint_provenance,
)
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor, VisualizationView
from llmoe.weights import BACKBONE_SLUGS, backbone_key, resolve_weights


# ---------------------------------------------------------------------------
# Backbone-specific training and model integration
# ---------------------------------------------------------------------------


RELEASED_MODEL_MAX_LENGTH = 2048
ATTN_IMPLEMENTATION = "flash_attention_2"


class HiDropBackend(ABC):
    """Everything the two-stage recipe needs to know about one backbone."""

    key: str
    model_tag: str
    stage1_module_path: str
    stage1_artifact_name: str
    stage1_reset_by_default: bool
    #: Whether upstream publishes a HiDrop recipe/checkpoint for this backbone.
    released_recipe: bool

    # ------------------------------------------------------------------- schedule

    @abstractmethod
    def build_config(self, compressor_args: dict[str, Any], *, batch_size_supported: bool,
                     num_decoder_layers: int | None = None) -> Any:
        """Build the HiDrop schedule for this backbone from ``compressor_args``."""

    @abstractmethod
    def budget_tag(self, config: Any) -> str:
        """Short filesystem-safe budget label used in output directory names."""

    @abstractmethod
    def schedule_metadata(self, config: Any) -> dict[str, Any]:
        """The schedule fields written into manifests and ``hidrop_config.json``."""

    def semantic_signature(self, config: Any) -> tuple[Any, ...]:
        """Identity of a schedule, for refusing artifacts trained under another one."""
        metadata = self.schedule_metadata(config)
        return tuple(sorted((key, str(value)) for key, value in metadata.items()))

    # ---------------------------------------------------------------------- model

    @abstractmethod
    def load_training_base(self, model_name_or_path: str, *, stage: str, dtype: torch.dtype | None,
                           attn_implementation: str, model_max_length: int,
                           local_files_only: bool) -> tuple[Any, Any]:
        """Load ``(processor, base_model)`` with this backbone's *training* preprocessing."""

    @abstractmethod
    def build_decoder_wrapper(self, language_model: nn.Module, compressor: Any) -> nn.Module:
        """The HiDrop decoder that replaces ``model.model.language_model``."""

    @abstractmethod
    def image_feature_rows(self, base_model: Any, batch: dict[str, Any],
                           **kwargs: Any) -> list[torch.Tensor]:
        """Per-image feature rows, in batch order, ready to scatter into the embeddings."""

    @abstractmethod
    def build_collator(self, processor: Any, *, model_max_length: int, stage: str) -> Any:
        """The collator producing this backbone's tokenized, labelled training batches."""

    @abstractmethod
    def language_model_config(self, base_model: Any) -> Any:
        """The text config, for vocab size and attention-backend checks."""

    # --------------------------------------------------------------------- module

    def stage1_module(self, base_model: Any) -> nn.Module:
        module: Any = base_model
        for attribute in self.stage1_module_path.split("."):
            module = getattr(module, attribute)
        return module

    @abstractmethod
    def vision_tower(self, base_model: Any) -> nn.Module | None:
        """The frozen vision encoder (excluding the stage-1 module where they overlap)."""

    def freeze_vision_tower(self, base_model: Any) -> None:
        tower = self.vision_tower(base_model)
        if tower is not None:
            tower.requires_grad_(False)

    def reset_stage1_module(self, base_model: Any) -> None:
        """Re-initialise the stage-1 module deterministically."""
        module = self.stage1_module(base_model)
        initializer_range = float(
            getattr(self.language_model_config(base_model), "initializer_range", 0.02)
        )
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                nn.init.normal_(submodule.weight, mean=0.0, std=initializer_range)
                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)

    def image_token_id(self, base_model: Any, processor: Any) -> int:
        for attribute in ("image_token_id", "image_token_index"):
            token_id = getattr(base_model.config, attribute, None)
            if token_id is not None and int(token_id) >= 0:
                return int(token_id)
        raise ValueError(
            f"HiDrop could not resolve the image token id for the {self.key} backbone."
        )

    # -------------------------------------------------------------------- outputs

    def model_tag_for(self, backbone: Any | None = None) -> str:
        """The checkpoint tag for the model actually being trained.

        ``model_tag`` names the backbone the published recipe targets. Another
        backbone under the same recipe -- LLaVA-1.5-13B on the LLaVA path -- takes
        its own tag instead, so its checkpoints cannot overwrite the 7B ones or be
        mistaken for them later. The recipe's own backbone keeps the upstream
        spelling (``llava-v1.5-7b``) so existing run directories still resolve.
        """
        key = backbone_key(backbone) if backbone is not None else None
        if key is None or key == self.key:
            return self.model_tag
        return BACKBONE_SLUGS.get(key, key)

    def stage_output_dir(self, output_root: str | Path, stage: str, config: Any,
                         backbone: Any | None = None) -> Path:
        root = Path(output_root) / "hidrop"
        run_tag = f"{self.budget_tag(config)}-{ATTN_IMPLEMENTATION}"
        suffix = "-pretrain" if stage == "pretrain" else ""
        return root / f"{self.model_tag_for(backbone)}-hidrop-{run_tag}{suffix}"


# ---------------------------------------------------------------------------
# LLaVA-1.5 -- the released recipe, unchanged
# ---------------------------------------------------------------------------


class LlavaHiDropBackend(HiDropBackend):
    key = "llava"
    model_tag = "llava-v1.5-7b"
    stage1_module_path = "model.multi_modal_projector"
    stage1_artifact_name = "mm_projector.bin"
    stage1_reset_by_default = True
    released_recipe = True

    def build_config(self, compressor_args: dict[str, Any], *, batch_size_supported: bool,
                     num_decoder_layers: int | None = None) -> HiDropConfig:
        options = dict(compressor_args)
        options.pop("reset_stage1", None)
        options["batch_size_supported"] = bool(batch_size_supported)
        if num_decoder_layers is not None and not options.get("layer_list"):
            # The named profiles are stated at 32 layers; on a deeper Vicuna the
            # schedule is re-solved for that depth so the budget label holds.
            options["num_decoder_layers"] = int(num_decoder_layers)
        return HiDropConfig(**options)

    def budget_tag(self, config: HiDropConfig) -> str:
        return str(int(config.average_retained_tokens))

    def schedule_metadata(self, config: HiDropConfig) -> dict[str, Any]:
        return config.hidrop_metadata()

    def language_model_config(self, base_model: Any) -> Any:
        return base_model.config.text_config

    def vision_tower(self, base_model: Any) -> nn.Module | None:
        return getattr(base_model.model, "vision_tower", None)

    def build_decoder_wrapper(self, language_model: nn.Module, compressor: Any) -> nn.Module:
        from llmoe.compressor.hidrop.hidrop_llama import HiDropLlamaModel

        return HiDropLlamaModel(language_model, compressor)

    def build_collator(self, processor: Any, *, model_max_length: int, stage: str) -> Any:
        from llmoe.compressor.hidrop.data import LlavaHiDropCollator

        return LlavaHiDropCollator(
            processor=processor, model_max_length=model_max_length, visual_tokens=576, stage=stage
        )

    def image_feature_rows(self, base_model: Any, batch: dict[str, Any],
                           **kwargs: Any) -> list[torch.Tensor]:
        image_output = base_model.model.get_image_features(
            pixel_values=batch["pixel_values"],
            vision_feature_layer=kwargs.get("vision_feature_layer"),
            vision_feature_select_strategy=kwargs.get("vision_feature_select_strategy"),
            image_sizes=kwargs.get("image_sizes"),
            return_dict=True,
        )
        return list(image_output.pooler_output)

    def load_training_base(self, model_name_or_path: str, *, stage: str, dtype: torch.dtype | None,
                           attn_implementation: str, model_max_length: int,
                           local_files_only: bool) -> tuple[Any, Any]:
        from llmoe.compressor.hidrop.hidrop_tr import load_hf_llava_training_base

        return load_hf_llava_training_base(
            model_name_or_path,
            stage=stage,
            dtype=dtype,
            attn_implementation=attn_implementation,
            model_max_length=model_max_length,
            local_files_only=local_files_only,
        )


# ---------------------------------------------------------------------------
# Qwen2.5-VL -- our adaptation of the same recipe
# ---------------------------------------------------------------------------


class QwenHiDropBackend(HiDropBackend):
    key = "qwen"
    model_tag = "qwen2.5-vl-7b"
    stage1_module_path = "model.visual.merger"
    stage1_artifact_name = "mm_merger.bin"
    # Qwen's merger ships pretrained and aligned; see the module docstring.
    stage1_reset_by_default = False
    released_recipe = False

    def build_config(self, compressor_args: dict[str, Any], *, batch_size_supported: bool,
                     num_decoder_layers: int | None = None) -> QwenHiDropConfig:
        options = dict(compressor_args)
        options.pop("reset_stage1", None)
        retain_ratio = options.pop("retain_ratio", None)
        if retain_ratio is None:
            raise ValueError(
                "HiDrop training on Qwen2.5-VL is parameterised by `retain_ratio`, not "
                "`average_retained_tokens`: the NaViT encoder gives every image a different "
                "visual-token count, so an absolute budget is not comparable across samples. "
                "Pass e.g. `compressor_args: {retain_ratio: 0.25}`."
            )
        if num_decoder_layers is None:
            raise ValueError("The Qwen HiDrop schedule needs the decoder's depth to resolve.")
        options["batch_size_supported"] = bool(batch_size_supported)
        return QwenHiDropConfig.for_backbone(
            float(retain_ratio), num_decoder_layers=int(num_decoder_layers), **options
        )

    def budget_tag(self, config: QwenHiDropConfig) -> str:
        return f"r{config.retain_ratio:g}"

    def schedule_metadata(self, config: QwenHiDropConfig) -> dict[str, Any]:
        data = config.to_dict()
        # Drop the run-local diagnostics; only the schedule identity belongs in a manifest.
        for key in ("debug", "enabled", "batch_size_supported"):
            data.pop(key, None)
        data["released_recipe"] = False
        return data

    def language_model_config(self, base_model: Any) -> Any:
        return base_model.config.text_config

    def vision_tower(self, base_model: Any) -> nn.Module | None:
        return getattr(base_model.model, "visual", None)

    def freeze_vision_tower(self, base_model: Any) -> None:
        """Freeze the ViT but leave the merger's own grad flag to the caller.

        The merger lives *inside* ``visual`` on Qwen, unlike LLaVA where the projector is
        a sibling of the vision tower. Freezing the tower wholesale would freeze the
        stage-1 module too, so the merger is re-enabled here when it is what stage 1
        trains; the trainer sets the final requires_grad state right afterwards.
        """
        tower = self.vision_tower(base_model)
        if tower is None:
            return
        tower.requires_grad_(False)

    def build_decoder_wrapper(self, language_model: nn.Module, compressor: Any) -> nn.Module:
        from llmoe.compressor.hidrop.hidrop_qwen import HiDropQwenModel

        return HiDropQwenModel(language_model, compressor)

    def build_collator(self, processor: Any, *, model_max_length: int, stage: str) -> Any:
        from llmoe.compressor.hidrop.data import QwenHiDropCollator

        return QwenHiDropCollator(
            processor=processor, model_max_length=model_max_length, stage=stage
        )

    def image_feature_rows(self, base_model: Any, batch: dict[str, Any],
                           **kwargs: Any) -> list[torch.Tensor]:
        del kwargs
        image_grid_thw = batch.get("image_grid_thw")
        if image_grid_thw is None:
            raise ValueError(
                "HiDrop training on Qwen2.5-VL needs `image_grid_thw` alongside `pixel_values`; "
                "the collator must pass the processor's grid through."
            )
        image_output = base_model.model.get_image_features(
            pixel_values=batch["pixel_values"], image_grid_thw=image_grid_thw, return_dict=True
        )
        # Already split per image by `Qwen2_5_VLModel.get_image_features`.
        return list(image_output.pooler_output)

    def load_training_base(self, model_name_or_path: str, *, stage: str, dtype: torch.dtype | None,
                           attn_implementation: str, model_max_length: int,
                           local_files_only: bool) -> tuple[Any, Any]:
        """Load Qwen2.5-VL with right-padded training preprocessing.

        Mirrors :func:`load_hf_llava_training_base`: a right-padded tokenizer (the eval
        backbone left-pads for generation), the pinned attention backend on every
        submodule, and a consistency check between the tokenizer and the config's image
        token. Qwen's image processor needs no per-stage switch -- it is resolution-native,
        where LLaVA swaps square-crop for pad between the two stages.
        """
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if attn_implementation != ATTN_IMPLEMENTATION:
            raise ValueError(
                f"HiDrop training only supports {ATTN_IMPLEMENTATION}, got {attn_implementation!r}."
            )
        if stage not in {"pretrain", "finetune"}:
            raise ValueError(f"Unknown HiDrop training stage {stage!r}.")
        processor = AutoProcessor.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        tokenizer = processor.tokenizer
        tokenizer.model_max_length = int(model_max_length)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )
        image_token = getattr(processor, "image_token", "<|image_pad|>")
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        config_token_id = getattr(base_model.config, "image_token_id", None)
        if config_token_id is not None and int(config_token_id) != int(image_token_id):
            raise ValueError(
                "Qwen2.5-VL tokenizer/config image-token mismatch: "
                f"tokenizer={image_token_id}, config={config_token_id}."
            )
        base_model.config.pad_token_id = int(tokenizer.pad_token_id)
        base_model.config.text_config.pad_token_id = int(tokenizer.pad_token_id)
        for config in (base_model.config, base_model.config.text_config,
                       base_model.config.vision_config):
            config._attn_implementation = attn_implementation
            if hasattr(config, "_attn_implementation_internal"):
                config._attn_implementation_internal = attn_implementation
        base_model.model.language_model.config._attn_implementation = attn_implementation
        effective = getattr(base_model.model.language_model.config, "_attn_implementation", None)
        if effective != attn_implementation:
            raise RuntimeError(
                f"Requested {attn_implementation!r} for the language model but Transformers "
                f"selected {effective!r}."
            )
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Effective attention backend: text={effective!r}", flush=True)
        return processor, base_model


BACKENDS: dict[str, HiDropBackend] = {
    "llava": LlavaHiDropBackend(),
    "qwen": QwenHiDropBackend(),
}


def detect_backend(vlm_or_model: Any) -> HiDropBackend:
    """Pick the backend structurally, so forks and subclasses still resolve."""
    model = getattr(vlm_or_model, "model", vlm_or_model)
    inner = getattr(model, "model", None)
    if getattr(inner, "visual", None) is not None:
        return BACKENDS["qwen"]
    if getattr(inner, "vision_tower", None) is not None:
        return BACKENDS["llava"]
    raise TypeError(
        "HiDrop training supports the LLaVA-1.5 and Qwen2.5-VL backbones; this model exposes "
        "neither `model.model.vision_tower` nor `model.model.visual`."
    )


def backend_for_training(compressor_args: dict[str, Any], model_name_or_path: str) -> HiDropBackend:
    """Choose the backend before any weights load, from the parameterisation and the id.

    LLaVA-NeXT and Qwen both use per-image ratios, so an explicit
    ``backbone_family: llava_next`` takes precedence over the legacy parameter-name
    inference. The model id is the fallback for a config that names neither.
    """
    has_ratio = compressor_args.get("retain_ratio") is not None
    has_absolute = compressor_args.get("average_retained_tokens") is not None
    family = str(compressor_args.get("backbone_family", "")).replace("-", "_").lower()
    looks_qwen = "qwen" in str(model_name_or_path).lower()
    if family == "llava_next":
        if has_absolute:
            raise ValueError(
                "HiDrop/LLaVA-NeXT is parameterized by retain_ratio; remove "
                "average_retained_tokens from compressor_args."
            )
        if not has_ratio:
            raise ValueError("HiDrop/LLaVA-NeXT requires compressor_args.retain_ratio.")
        return BACKENDS["llava"]
    if has_ratio and has_absolute:
        raise ValueError(
            "`retain_ratio` (Qwen2.5-VL) and `average_retained_tokens` (LLaVA-1.5) select "
            "different backbones; pass exactly one."
        )
    if has_ratio:
        if not looks_qwen:
            warnings.warn(
                f"compressor_args.retain_ratio selects the Qwen2.5-VL HiDrop path, but "
                f"{model_name_or_path!r} does not look like a Qwen checkpoint.",
                stacklevel=3,
            )
        return BACKENDS["qwen"]
    if has_absolute:
        return BACKENDS["llava"]
    return BACKENDS["qwen"] if looks_qwen else BACKENDS["llava"]


def training_compressor_args(compressor_args: dict[str, Any]) -> dict[str, Any]:
    """Strip inference-only knobs so ``compressor_args`` can be reused verbatim.

    ``checkpoint_path`` says which trained model to *load*; training produces them. The
    stages chain through ``training_args.pretrain_mm_mlp_adapter`` instead.
    """
    options = dict(compressor_args)
    for key in (*CONFIG_METADATA_KEYS, "checkpoint_path", "model_path", "sdpa_flash_only",
                "validate_provenance"):
        options.pop(key, None)
    return options


_SUPPORTED_ATTN_IMPLEMENTATIONS = ("sdpa", "flash_attention_2", "eager")


def resolve_checkpoint_dir(checkpoint_path: str | Path) -> Path:
    """Local directory for a checkpoint given as a path or a Hugging Face repo id."""
    candidate = Path(checkpoint_path).expanduser()
    if candidate.is_dir():
        return candidate
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(checkpoint_path)))


def model_weight_files(directory: Path) -> list[Path]:
    """Full-model weight shards in `directory`, or `[]` for a projector-only artifact."""
    for index_name, single_name in (
        ("model.safetensors.index.json", "model.safetensors"),
        ("pytorch_model.bin.index.json", "pytorch_model.bin"),
    ):
        index = directory / index_name
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            names = sorted(set(weight_map.values()))
            if not names:
                raise ValueError(f"Checkpoint shard index {index} has no weight_map entries.")
            missing = [name for name in names if not (directory / name).is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete HiDrop checkpoint {directory}; missing shards {missing}.")
            return [directory / name for name in names]
        single = directory / single_name
        if single.is_file():
            return [single]
    return []


class HiDropCompressor(TokenCompressor):
    """Late injection, concave pyramid pruning, and early vision exit in the decoder.

    Parameters
    ----------
    retain_ratio:
        **LLaVA-NeXT and Qwen2.5-VL.** The budget as a *ratio*: the
        layer-averaged fraction of the image's visual tokens
        (``sum(fraction held by each decoder layer) / num_decoder_layers``). Qwen's
        NaViT encoder gives every image a different visual-token count, so the
        LLaVA path's absolute ``average_retained_tokens`` is not comparable across
        samples and each stage's budget is a fraction of that image's own span.
        Published budgets are ``0.375``, ``0.25``, ``0.125`` and ``0.0625``.
        LLaVA-NeXT resolves the ratio from its true unpadded AnyRes content span;
        ``backbone_family: llava_next`` distinguishes it from Qwen.
    average_retained_tokens:
        **LLaVA-1.5 only.** The same budget as an absolute layer-averaged count out
        of 576, which is how the released profiles are published (224/128/64/32).
    pretrained:
        **LLaVA-1.5 only.** ``True`` selects the released checkpoint for this
        budget, a path or repo id names a custom checkpoint, and ``False`` runs
        the untrained ablation. When omitted, the released checkpoint is used
        because HiDrop has no training-free form.
    checkpoint_path:
        The ``ft`` output directory whose schedule and weights are loaded. On
        Qwen, where there is no released checkpoint, use this for an in-repo
        trained checkpoint instead of ``pretrained``.
    """

    def __init__(self, config: HiDropConfig | None = None, *, retain_ratio: float | None = None,
                 sdpa_flash_only: bool = True, validate_provenance: bool = True,
                 pretrained: bool | str | PathLike[str] | None = None,
                 **config_kwargs: Any) -> None:
        for metadata_key in CONFIG_METADATA_KEYS:
            config_kwargs.pop(metadata_key, None)
        checkpoint_path = (
            config_kwargs.pop("checkpoint_path", None)
            or config_kwargs.pop("model_path", None)
            or (config.checkpoint_path if config is not None else None)
        )
        family_hint = (
            str(
                config.backbone_family
                if config is not None
                else config_kwargs.get("backbone_family", "llava")
            )
            .replace("-", "_")
            .lower()
        )
        next_ratio_mode = retain_ratio is not None and family_hint == "llava_next"
        # The weight request as asked for, before any resolution: replaying it in
        # `_point_backbone_at_checkpoint` once the backbone is known is what keeps
        # one backbone's released checkpoint off another.
        self._checkpoint_arg = checkpoint_path
        self._pretrained_arg = pretrained
        self._weight_budget: int = 0
        if retain_ratio is not None and not next_ratio_mode:
            if pretrained not in (None, False):
                raise ValueError(
                    "HiDrop's bundled `pretrained` weights are LLaVA-only; "
                    "for Qwen pass a Qwen `checkpoint_path`."
                )
        else:
            budget = (
                llava_next_profile_for_ratio(float(retain_ratio))
                if next_ratio_mode
                else int(
                    config.average_retained_tokens if config is not None
                    else config_kwargs.get(
                        "average_retained_tokens", HiDropConfig.average_retained_tokens
                    )
                )
            )
            # An explicit path is resolved here, because the schedule is read off
            # the checkpoint before anything else. A *released* artifact is not:
            # which one applies depends on the backbone, so resolving it here
            # would fetch (and read a schedule from) the only released backbone's
            # weights whatever backbone the run actually selected.
            # `_point_backbone_at_checkpoint` does that once the backbone is known.
            self._weight_budget = budget
            if checkpoint_path is not None:
                checkpoint_path = resolve_weights(
                    "HiDrop",
                    budget,
                    checkpoint_path=checkpoint_path,
                    pretrained=pretrained,
                )
        # Resolved once, here, because the schedule is read off the checkpoint
        # before anything else and a repo id has to become a local path first.
        self._checkpoint_dir: Path | None = (
            None if checkpoint_path is None else resolve_checkpoint_dir(checkpoint_path)
        )
        # "finetune" for a full model directory, "pretrain" for a bare stage-1 adapter.
        self._checkpoint_stage: str = "finetune"
        # The backbone is not known until install(), so the two parameterisations are
        # validated here and cross-checked against the backbone there. The Qwen
        # schedule additionally needs the decoder's depth, which only install() can
        # read, so the config itself is built there.
        self._retain_ratio: float | None = (
            float(retain_ratio) if retain_ratio is not None and not next_ratio_mode else None
        )
        self._qwen_overrides: dict[str, Any] = {}
        if self._retain_ratio is not None:
            if config is not None:
                raise ValueError(
                    "retain_ratio selects HiDrop's Qwen2.5-VL path, which resolves its own "
                    "schedule; pass either `retain_ratio` or a LLaVA `HiDropConfig`, not both."
                )
            if not 0.0 < self._retain_ratio < 1.0:
                raise ValueError(f"retain_ratio must be in (0, 1), got {self._retain_ratio}.")
            self._qwen_overrides = dict(config_kwargs)
            self.config: HiDropConfig | QwenHiDropConfig | None = None
            super().__init__(max_tokens=0)  # ratio mode: the retained count is per-image
            self.debugger: HiDropDebugger = HiDropDebugger(bool(config_kwargs.get("debug", False)))
        else:
            if next_ratio_mode:
                config_kwargs["retain_ratio"] = float(retain_ratio)
            if config is None and self._checkpoint_dir is not None:
                config = HiDropConfig.from_checkpoint(self._checkpoint_dir, **config_kwargs)
            self.config = config or HiDropConfig(**config_kwargs)
            super().__init__(max_tokens=int(self.config.average_retained_tokens))
            self.debugger = HiDropDebugger(self.config.debug)
        self._sdpa_flash_only: bool = bool(sdpa_flash_only)
        self._validate_provenance: bool = bool(validate_provenance)
        self._vlm: VLM | None = None
        self._hooks: list[Any] = []
        self._original_language_model: nn.Module | None = None
        self._original_prepare_inputs: Any | None = None
        self._original_generate: Any | None = None
        self._original_format_prompts: Any | None = None
        self._original_attn_implementation: str | None = None
        self._original_num_additional_image_tokens: int | None = None
        self._original_model_id: str | None = None
        self._use_flash_sdpa: bool = False
        self._latest_input_ids: torch.Tensor | None = None
        self._image_token_id: int | None = None
        self._pending_llava_next_layouts: tuple[LlavaNextVisualLayout, ...] = ()
        self.last_context: HiDropForwardContext | HiDropQwenForwardContext | None = None
        self.last_visual_token_counts_by_layer: list[int] = []
        self.last_patch_token_counts_by_layer: list[int] = []
        self.pruning_history: list[dict[str, Any]] = []

    @classmethod
    def for_training(cls, config: "HiDropConfig | QwenHiDropConfig") -> "HiDropCompressor":
        """A compressor that only carries the schedule, for the training wrappers.

        The trainers install the decoder wrapper themselves (they own the model, which
        `install()` expects to reach through a :class:`VLM`), so this skips the whole
        install path and just holds the config, debugger, and per-forward state the
        decoder reads. Accepts either backbone's schedule object.
        """
        compressor = cls.__new__(cls)
        TokenCompressor.__init__(
            compressor,
            max_tokens=int(getattr(config, "average_retained_tokens", 0) or 0),
        )
        compressor._retain_ratio = (
            float(config.retain_ratio) if isinstance(config, QwenHiDropConfig) else None
        )
        compressor._qwen_overrides = {}
        compressor._checkpoint_dir = None
        compressor._checkpoint_stage = "finetune"
        compressor.config = config
        compressor.debugger = HiDropDebugger(config.debug)
        compressor._sdpa_flash_only = False
        compressor._validate_provenance = False
        compressor._vlm = None
        compressor._hooks = []
        compressor._original_language_model = None
        compressor._original_prepare_inputs = None
        compressor._original_generate = None
        compressor._original_format_prompts = None
        compressor._original_attn_implementation = None
        compressor._original_num_additional_image_tokens = None
        compressor._original_model_id = None
        compressor._use_flash_sdpa = False
        compressor._latest_input_ids = None
        compressor._image_token_id = None
        compressor._pending_llava_next_layouts = ()
        compressor.last_context = None
        compressor.last_visual_token_counts_by_layer = []
        compressor.last_patch_token_counts_by_layer = []
        compressor.pruning_history = []
        return compressor

    @property
    def budget(self) -> int | float:
        """Layer-average count on LLaVA-1.5, or per-image ratio otherwise."""
        if self._retain_ratio is not None:
            return self._retain_ratio
        if self.config.backbone_family == "llava_next":
            return float(self.config.retain_ratio)
        return int(self.config.average_retained_tokens)

    @override
    def compress(
        self,
        features: torch.Tensor,
        *,
        attention: torch.Tensor | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> CompressedFeatures:
        """Pass the visual tokens through untouched.

        HiDrop does not compress at the vision/projector boundary -- late injection
        needs all 576 original projected tokens to reach layer 9 -- so every
        decision happens inside :class:`HiDropLlamaModel`. This stays a no-op for
        callers that reach for ``compress()`` directly.
        """
        del attention, layer_idx, kwargs
        return CompressedFeatures(tokens=features)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        """Per-stage ``[T_cur, T_orig]`` one-hot keep matrices, identity first.

        The pruning stages record themselves during the forward; snapshot 0 -- the
        pre-pruning identity the other compressors also record -- is built here so
        a benchmark run that never visualizes does not pay for it.
        """
        if not self._source_history:
            return []
        first = self._source_history[0]
        identity = torch.eye(int(first.shape[1]), dtype=first.dtype, device=first.device)
        return [identity, *self._source_history]

    @override
    def get_visualization_views(
        self,
        vlm: VLM,
        inputs: Mapping[str, Any],
        image_processor: Any,
    ) -> list[VisualizationView] | None:
        del vlm
        if self.config is None or self.config.backbone_family != "llava_next":
            return None
        context = self.last_context
        layout = getattr(context, "llava_next_layout", None)
        if layout is None:
            raise RuntimeError("HiDrop visualization did not capture an AnyRes layout.")
        pixel_values = inputs.get("pixel_values")
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError("HiDrop/LLaVA-NeXT visualization requires pixel_values.")
        return llava_next_visualization_views(pixel_values, image_processor, layout)

    def hidrop_config_dict(self) -> dict[str, Any]:
        """Schedule plus what the last forwards actually did (diagnostics)."""
        data = self.config.to_dict()
        data["pruning_history"] = self.pruning_history[-20:]
        data["last_visual_token_counts_by_layer"] = list(self.last_visual_token_counts_by_layer)
        data["last_patch_token_counts_by_layer"] = list(self.last_patch_token_counts_by_layer)
        layout = getattr(self.last_context, "llava_next_layout", None)
        if layout is not None:
            data["last_llava_next_layout"] = layout.to_metadata()
        resolved_schedule = getattr(self.last_context, "resolved_schedule", None)
        if resolved_schedule is not None:
            budget_metadata = resolved_schedule.to_metadata()
            data["last_resolved_budget"] = budget_metadata
            for key in (
                "original_content_tokens",
                "requested_retention_ratio",
                "target_content_tokens",
                "realized_content_tokens",
                "realized_retention_ratio",
                "structural_tokens",
            ):
                data[key] = budget_metadata[key]
        data["debug_warnings"] = list(self.debugger.warnings)
        return data

    # ------------------------------------------------------------------- install

    def validate_llava_next(self) -> None:
        """Backbone preflight used by :class:`LLaVANext8B`."""
        if self.config.backbone_family != "llava_next":
            raise TypeError(
                "HiDrop on llava-next-8b requires compressor_args.backbone_family: llava_next. "
                "This explicit contract prevents the released LLaVA-1.5 checkpoints from being "
                "loaded into the Llama-3/AnyRes backbone."
            )

    def validate_llava_next_backbone(self, backbone: VLM) -> None:
        """Reject an untracked model override before any large weights load."""
        model_id = str(getattr(backbone, "_model_id", ""))
        revision = getattr(backbone, "_model_revision", None)
        candidate = Path(model_id).expanduser()
        is_explicit_artifact = (
            self._checkpoint_dir is not None
            and candidate.is_dir()
            and os.path.realpath(candidate) == os.path.realpath(self._checkpoint_dir)
        )
        is_pinned_snapshot = is_pinned_llava_next_snapshot(candidate)
        if not (
            model_id == str(self.config.backbone_model_id)
            or is_pinned_snapshot
            or is_explicit_artifact
        ):
            raise ValueError(
                "HiDrop/LLaVA-NeXT is implemented for the pinned target model only; "
                f"backbone model_name_or_path={model_id!r}, expected "
                f"{self.config.backbone_model_id!r} or revision {self.config.backbone_revision!r}."
            )
        if (
            not is_explicit_artifact
            and not is_pinned_snapshot
            and revision != self.config.backbone_revision
        ):
            raise ValueError(
                "HiDrop/LLaVA-NeXT backbone revision differs from its compression config: "
                f"{revision!r} != {self.config.backbone_revision!r}."
            )

    def _validate_llava_next_model(self, model: Any) -> None:
        config = model.config
        if getattr(config, "model_type", None) != "llava_next":
            raise TypeError(
                "HiDrop/LLaVA-NeXT requires LlavaNextForConditionalGeneration; "
                f"got model_type={getattr(config, 'model_type', None)!r}."
            )
        text_config = config.text_config
        if getattr(text_config, "model_type", None) != "llama":
            raise TypeError(
                "HiDrop/LLaVA-NeXT-8B requires the checkpoint's Llama-3 decoder, got "
                f"text model_type={getattr(text_config, 'model_type', None)!r}."
            )
        if int(getattr(text_config, "num_hidden_layers", -1)) != self.config.num_decoder_layers:
            raise ValueError(
                "HiDrop/LLaVA-NeXT schedule/depth mismatch: "
                f"checkpoint has {getattr(text_config, 'num_hidden_layers', None)} layers, "
                f"config records {self.config.num_decoder_layers}."
            )
        rope_parameters = getattr(text_config, "rope_parameters", None) or {}
        decoder_signature = {
            "hidden_size": int(getattr(text_config, "hidden_size", -1)),
            "intermediate_size": int(getattr(text_config, "intermediate_size", -1)),
            "attention_heads": int(getattr(text_config, "num_attention_heads", -1)),
            "key_value_heads": int(getattr(text_config, "num_key_value_heads", -1)),
            "head_dim": int(getattr(text_config, "head_dim", -1)),
            "rms_norm_eps": float(getattr(text_config, "rms_norm_eps", float("nan"))),
            "rope_theta": float(
                rope_parameters.get("rope_theta", getattr(text_config, "rope_theta", float("nan")))
            ),
            "max_position_embeddings": int(getattr(text_config, "max_position_embeddings", -1)),
        }
        expected_decoder = {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "attention_heads": 32,
            "key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 8192,
        }
        if decoder_signature != expected_decoder:
            raise ValueError(
                "HiDrop/LLaVA-NeXT requires the pinned Llama-3-8B decoder signature: "
                f"model={decoder_signature}, expected={expected_decoder}."
            )
        vision_config = config.vision_config
        vision_signature = {
            "model_type": str(getattr(vision_config, "model_type", "")),
            "hidden_size": int(getattr(vision_config, "hidden_size", -1)),
            "intermediate_size": int(getattr(vision_config, "intermediate_size", -1)),
            "layers": int(getattr(vision_config, "num_hidden_layers", -1)),
            "attention_heads": int(getattr(vision_config, "num_attention_heads", -1)),
            "image_size": int(getattr(vision_config, "image_size", -1)),
            "patch_size": int(getattr(vision_config, "patch_size", -1)),
            "feature_layer": int(getattr(config, "vision_feature_layer", 0)),
            "feature_select_strategy": str(getattr(config, "vision_feature_select_strategy", "")),
        }
        expected_vision = {
            "model_type": "clip_vision_model",
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "layers": 24,
            "attention_heads": 16,
            "image_size": 336,
            "patch_size": 14,
            "feature_layer": -2,
            "feature_select_strategy": "default",
        }
        if vision_signature != expected_vision:
            raise ValueError(
                "HiDrop/LLaVA-NeXT requires the pinned CLIP-L/14-336 signature: "
                f"model={vision_signature}, expected={expected_vision}."
            )
        projector_signature = {
            "hidden_act": str(getattr(config, "projector_hidden_act", "")),
            "learned_image_newline": bool(getattr(config, "use_image_newline_parameter", False)),
            "text_vocab_size": int(getattr(text_config, "vocab_size", -1)),
        }
        expected_projector = {
            "hidden_act": "gelu",
            "learned_image_newline": True,
            "text_vocab_size": 128320,
        }
        if projector_signature != expected_projector:
            raise ValueError(
                "HiDrop/LLaVA-NeXT multimodal interface differs from the pinned checkpoint: "
                f"model={projector_signature}, expected={expected_projector}."
            )
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(config, "image_token_index", None)
        if int(image_token_id if image_token_id is not None else -1) != 128256:
            raise ValueError(
                "HiDrop/LLaVA-NeXT requires the pinned checkpoint image token 128256; "
                f"got {image_token_id}."
            )
        model_pinpoints = tuple(tuple(map(int, point)) for point in config.image_grid_pinpoints)
        expected_pinpoints = tuple(self.config.image_grid_pinpoints)
        geometry = (
            model_pinpoints,
            int(config.vision_config.image_size),
            int(config.vision_config.patch_size),
            str(config.vision_feature_select_strategy),
        )
        expected = (
            expected_pinpoints,
            int(self.config.vision_image_size),
            int(self.config.vision_patch_size),
            str(self.config.vision_feature_select_strategy),
        )
        if geometry != expected:
            raise ValueError(
                "HiDrop/LLaVA-NeXT checkpoint geometry differs from the training config: "
                f"model={geometry}, expected={expected}."
            )

    def configure_llava_next_layouts(
        self,
        inner_model: Any,
        image_sizes: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
    ) -> tuple[LlavaNextVisualLayout, ...]:
        """Capture native per-image geometry before the language-model call.

        The released path is intentionally restricted to one source image in a
        batch of one. Supporting ragged multi-image rows would require separate
        span state and per-row selector collectives, so it fails closed here.
        """
        self._pending_llava_next_layouts = ()
        if not isinstance(image_sizes, torch.Tensor) or image_sizes.ndim != 2:
            raise ValueError("HiDrop/LLaVA-NeXT requires image_sizes shaped [images, 2].")
        if tuple(image_sizes.shape) != (1, 2):
            raise NotImplementedError(
                "HiDrop/LLaVA-NeXT supports exactly one source image per example; "
                f"got image_sizes shape {tuple(image_sizes.shape)}."
            )
        observed_crops = None
        if pixel_values is not None:
            if not isinstance(pixel_values, torch.Tensor):
                raise TypeError("HiDrop/LLaVA-NeXT pixel_values must be a tensor.")
            if pixel_values.ndim == 5:
                if int(pixel_values.shape[0]) != 1:
                    raise NotImplementedError("HiDrop/LLaVA-NeXT does not support image batching.")
                observed_crops = int(pixel_values.shape[1])
            elif pixel_values.ndim == 4:
                observed_crops = int(pixel_values.shape[0])
            else:
                raise ValueError(
                    "HiDrop/LLaVA-NeXT expected pixel_values [1, crops, C, H, W] or "
                    f"[crops, C, H, W], got {tuple(pixel_values.shape)}."
                )
        layout = layout_from_model(
            inner_model,
            image_sizes[0],
            observed_crop_count=observed_crops,
        )
        self._pending_llava_next_layouts = (layout,)
        return self._pending_llava_next_layouts


    @staticmethod
    def _detect_backbone(vlm: VLM) -> str:
        """``"qwen"`` or ``"llava"``, checked structurally so forks still pass."""
        inner = getattr(getattr(vlm, "model", None), "model", None)
        model_type = getattr(getattr(vlm, "model", None), "config", None)
        if (
            getattr(vlm, "compressor_family", None) == "llava_next"
            or getattr(model_type, "model_type", None) == "llava_next"
        ):
            return "llava_next"
        if getattr(inner, "visual", None) is not None:
            return "qwen"
        if getattr(inner, "vision_tower", None) is not None:
            return "llava"
        raise TypeError(
            f"HiDrop supports the LLaVA-1.5 and Qwen2.5-VL backbones; "
            f"{type(getattr(vlm, 'model', None)).__name__} exposes neither "
            f"`model.model.vision_tower` nor `model.model.visual`. Run with `--backbone llava` "
            f"or `--backbone qwen-vl`."
        )

    @override
    def install(self, vlm: VLM) -> None:
        if self._vlm is not None:
            if self._vlm is vlm:
                return  # idempotent: repeated benchmarks must not stack wrappers
            raise RuntimeError("HiDropCompressor is already installed on another backbone")
        if self._retain_ratio is not None:
            # Qwen is selected by the parameterisation, before anything loads, so the
            # redirect runs here for the same reason it does on LLaVA: a trained
            # checkpoint has to be what `transformers` loads, not something copied over
            # a base model afterwards.
            self._point_backbone_at_checkpoint(vlm)
            _ = vlm.processor  # triggers the lazy load so model and processor both exist
            backbone = self._detect_backbone(vlm)
            if backbone != "qwen":
                raise ValueError(
                    "retain_ratio selects HiDrop's Qwen2.5-VL path, but the loaded backbone is "
                    f"{backbone}. On LLaVA-1.5 the budget is an absolute layer-average out of 576 "
                    "visual tokens, which is how the released profiles are published: pass e.g. "
                    "`compressor_args: {average_retained_tokens: 64}`."
                )
            self._vlm = vlm
            self._install_qwen(vlm, vlm.model)
            return
        if getattr(vlm, "compressor_family", None) == "llava_next":
            self.validate_llava_next()
            self.validate_llava_next_backbone(vlm)
        elif self.config.backbone_family == "llava_next":
            raise TypeError(
                "A llava_next HiDrop configuration can only be installed on llava-next-8b."
            )
        # Before the lazy load: a trained checkpoint replaces what gets loaded.
        self._point_backbone_at_checkpoint(vlm)
        _ = vlm.processor  # triggers the lazy load so model and processor both exist
        model = vlm.model
        model_type = getattr(model.config, "model_type", None)
        if model_type not in {"llava", "llava_next"}:
            raise TypeError(
                "HiDrop requires a Hugging Face LLaVA or LLaVA-NeXT conditional-generation model."
            )
        detected_family = "llava_next" if model_type == "llava_next" else "llava"
        if detected_family != self.config.backbone_family:
            raise TypeError(
                f"HiDrop config is for {self.config.backbone_family!r}, but the loaded model is "
                f"{detected_family!r}. Checkpoint weights and schedules are not interchangeable."
            )
        if detected_family == "llava_next":
            self._validate_llava_next_model(model)
            tokenizer = getattr(vlm.processor, "tokenizer", None)
            converter = getattr(tokenizer, "convert_tokens_to_ids", None)
            processor_signature = {
                "image_token_id": int(converter("<image>")) if callable(converter) else -1,
                "patch_size": int(getattr(vlm.processor, "patch_size", -1)),
                "feature_select_strategy": str(
                    getattr(vlm.processor, "vision_feature_select_strategy", "")
                ),
                "additional_image_tokens": int(
                    getattr(vlm.processor, "num_additional_image_tokens", -1)
                ),
            }
            expected_processor = {
                "image_token_id": 128256,
                "patch_size": 14,
                "feature_select_strategy": "default",
                "additional_image_tokens": 1,
            }
            if processor_signature != expected_processor:
                raise ValueError(
                    "HiDrop/LLaVA-NeXT processor differs from the pinned checkpoint contract: "
                    f"processor={processor_signature}, expected={expected_processor}."
                )
        # Installation is considered active only after the loaded architecture
        # passes every read-only contract above. A bad model can therefore be
        # corrected and retried with this compressor instance.
        self._vlm = vlm
        self._load_pretrained_stage1(model)
        if detected_family == "llava":
            self._install_prompt_boundary_shim(vlm)
            self._align_processor_placeholder_count(vlm)
        self._select_attn_implementation(model)
        self._install_batch_size_guards(vlm)
        self._image_token_id = self._resolve_image_token_id(vlm, model)
        if not self.config.enabled:
            self.debugger.log("HiDrop disabled; leaving VLM unmodified")
            return
        inner = model.model
        language_model = inner.language_model
        self.config = self.config.resolved_for_depth(len(language_model.layers))
        if len(language_model.layers) != self.config.num_decoder_layers:
            raise ValueError(
                f"HiDrop expected a {self.config.num_decoder_layers}-layer decoder, "
                f"got {len(language_model.layers)}."
            )
        self._original_language_model = language_model

        # `LlavaModel.forward` does not forward `input_ids` to the language model,
        # so the visual span has to be captured one level up.
        def outer_pre_hook(
            module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            self._latest_input_ids = input_ids.detach() if torch.is_tensor(input_ids) else None
            self._pending_llava_next_layouts = ()
            if detected_family == "llava_next":
                image_sizes = kwargs.get("image_sizes")
                pixel_values = kwargs.get("pixel_values")
                if pixel_values is None and len(args) > 1:
                    pixel_values = args[1]
                if image_sizes is None and len(args) > 2:
                    image_sizes = args[2]
                if image_sizes is not None or pixel_values is not None:
                    if image_sizes is None:
                        raise ValueError(
                            "HiDrop/LLaVA-NeXT received pixel_values without image_sizes."
                        )
                    self.configure_llava_next_layouts(module, image_sizes, pixel_values)
            return None

        self._hooks.append(inner.register_forward_pre_hook(outer_pre_hook, with_kwargs=True))
        inner.language_model = HiDropLlamaModel(language_model, self)
        self.debugger.log(
            f"installed budget={self.config.average_retained_tokens} "
            f"late_entry={self.config.late_entry_layer} layers={list(self.config.layer_list)} "
            f"keeps={list(self.config.image_token_keep_list)} early_exit={self.config.early_exit_layer}"
        )


    def _install_qwen(self, vlm: VLM, model: Any) -> None:
        """Swap in the Qwen2.5-VL decoder wrapper and resolve the schedule for its depth.

        The LLaVA-only steps are skipped rather than adapted: there is no
        ``mm_projector.bin`` to load, no SentencePiece ``<image> \\n`` boundary to
        restore (Qwen tokenizes through a chat template), and no
        ``num_additional_image_tokens`` to realign (Qwen's placeholder count is derived
        from the image grid, not a fixed square).
        """
        language_model = model.model.language_model
        depth = int(len(language_model.layers))
        self.config = QwenHiDropConfig.for_backbone(
            self._retain_ratio, num_decoder_layers=depth, **self._qwen_overrides
        )
        self.debugger = HiDropDebugger(self.config.debug)
        # Deferred from `_point_backbone_at_checkpoint`: the Qwen schedule is
        # depth-relative, so its identity is only known once the decoder is loaded.
        self._validate_checkpoint_provenance()
        self._load_pretrained_stage1(model)
        self._select_attn_implementation(model)
        self._install_batch_size_guards(vlm)
        self._image_token_id = self._resolve_image_token_id(vlm, model)
        warnings.warn(
            "HiDrop on Qwen2.5-VL is our adaptation of the released LLaVA-1.5 recipe: upstream "
            "publishes no Qwen checkpoint and no Qwen training recipe, so these numbers are not "
            "comparable to the paper's.",
            stacklevel=3,
        )
        if not self.config.enabled:
            self.debugger.log("HiDrop disabled; leaving VLM unmodified")
            return
        inner = model.model
        self._original_language_model = language_model

        # `Qwen2_5_VLModel.forward` calls the language model with `input_ids=None`, so
        # the visual span has to be captured one level up.
        def outer_pre_hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            self._latest_input_ids = input_ids.detach() if torch.is_tensor(input_ids) else None
            return None

        self._hooks.append(inner.register_forward_pre_hook(outer_pre_hook, with_kwargs=True))
        inner.language_model = HiDropQwenModel(language_model, self)
        print(
            f"[HiDrop/Qwen] install: retain_ratio={self.config.retain_ratio:g} over {depth} "
            f"layers, late_entry={self.config.late_entry_layer} "
            f"early_exit={self.config.early_exit_layer} "
            f"filters={list(self.config.layer_list)} "
            f"keeps={[round(ratio, 6) for ratio in self.config.image_token_keep_ratio_list]}, "
            f"image_token_id={self._image_token_id}",
            flush=True,
        )

    def uninstall(self) -> None:
        """Undo everything ``install()`` changed, in reverse order.

        The backbone's ``model_name_or_path`` is restored, so a later reload picks
        the base model up again; the weights already in memory stay the trained
        ones, since re-downloading a 7B base to undo a benchmark would be absurd.
        """
        if self._vlm is None:
            return
        vlm = self._vlm
        inner = vlm.model.model
        if self._original_language_model is not None:
            inner.language_model = self._original_language_model
        if self._original_prepare_inputs is not None:
            vlm.prepare_inputs = self._original_prepare_inputs
        if self._original_generate is not None:
            vlm.generate = self._original_generate
        if self._original_format_prompts is not None:
            vlm.format_prompts = self._original_format_prompts
        if self._original_num_additional_image_tokens is not None:
            vlm.processor.num_additional_image_tokens = self._original_num_additional_image_tokens
        if self._original_attn_implementation is not None:
            vlm.model.set_attn_implementation(self._original_attn_implementation)
        if self._original_model_id is not None:
            vlm._model_id = self._original_model_id
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._original_language_model = None
        self._original_prepare_inputs = None
        self._original_generate = None
        self._original_format_prompts = None
        self._original_num_additional_image_tokens = None
        self._original_attn_implementation = None
        self._original_model_id = None
        self._latest_input_ids = None
        self._image_token_id = None
        self._pending_llava_next_layouts = ()
        self._use_flash_sdpa = False
        self._vlm = None

    # ------------------------------------------------------- trained checkpoint

    @property
    def _backend(self) -> HiDropBackend:
        """The backbone whose recipe produced (or would produce) this checkpoint.

        `retain_ratio` selects Qwen and its absence selects LLaVA -- the same rule
        `backend_for_training` applies, so `tr`/`ft` and the benchmark agree on which
        stage-1 artifact and schedule identity a checkpoint carries.
        """
        return BACKENDS["qwen" if self._retain_ratio is not None else "llava"]

    def _schedule_metadata(self) -> dict[str, Any]:
        """This run's schedule identity, in the form the run manifest recorded it."""
        return self._backend.schedule_metadata(self.config)

    def _validate_checkpoint_provenance(self) -> None:
        """Refuse a checkpoint trained under another schedule, once one is resolved."""
        if not self._validate_provenance or self._checkpoint_dir is None or self.config is None:
            return
        validate_inference_checkpoint_provenance(
            self._checkpoint_dir, self._schedule_metadata(), expected_stage=self._checkpoint_stage
        )

    def _point_backbone_at_checkpoint(self, vlm: VLM) -> None:
        """Make the backbone load the weights HiDrop was trained with.

        Called before anything touches ``vlm.model``: the backbone is lazy, so
        redirecting its ``model_name_or_path`` at the checkpoint means the trained
        weights are loaded by ``transformers`` itself -- the checkpoint's own
        config, dtype handling, and key conversions included -- instead of being
        copied over a base model afterwards. A stage-1 directory has no model
        weights, so it is left for :meth:`_load_pretrained_projector`.
        """
        if self._retain_ratio is None and self._checkpoint_dir is not None:
            # Re-resolve now that the backbone is known. The released checkpoint is
            # a whole finetuned LLaVA-1.5-7B loaded *in place of* the selected
            # backbone, so on any other backbone this raises naming what exists
            # rather than quietly evaluating the 7B under another backbone's label.
            resolved = resolve_weights(
                "HiDrop",
                self._weight_budget,
                checkpoint_path=self._checkpoint_arg,
                pretrained=self._pretrained_arg,
                backbone=backbone_key(vlm),
            )
            self._checkpoint_dir = None if resolved is None else resolve_checkpoint_dir(resolved)
        elif self._retain_ratio is None and self._checkpoint_arg is None:
            resolved = resolve_weights(
                "HiDrop",
                self._weight_budget,
                pretrained=self._pretrained_arg,
                backbone=backbone_key(vlm),
            )
            if resolved is not None:
                # Deferred from `__init__`: adopt the schedule this checkpoint was
                # trained with, exactly as a caller-supplied path would have.
                self._checkpoint_dir = resolve_checkpoint_dir(resolved)
                self.config = HiDropConfig.from_checkpoint(self._checkpoint_dir)
                self.debugger = HiDropDebugger(self.config.debug)
        directory = self._checkpoint_dir
        if directory is None:
            warnings.warn(
                "HiDrop: no compressor_args.checkpoint_path given, so the schedule is running on an "
                "untrained base model. HiDrop is trained per schedule (`llmoe ... tr`, then `ft`); "
                "results without a HiDrop checkpoint are an ablation, not the paper's setting.",
                stacklevel=3,
            )
            return
        backend = self._backend
        weight_files = model_weight_files(directory)
        self._checkpoint_stage = "finetune" if weight_files else "pretrain"
        # On Qwen the schedule cannot resolve until the decoder's depth is known, so
        # `_install_qwen` calls this again once `self.config` exists.
        self._validate_checkpoint_provenance()
        if not weight_files:
            if not (directory / backend.stage1_artifact_name).is_file():
                raise FileNotFoundError(
                    f"HiDrop checkpoint {directory} contains neither model weights nor "
                    f"{backend.stage1_artifact_name}."
                )
            warnings.warn(
                f"HiDrop: {directory} holds only a pretrained stage-1 module, so the language model "
                "is the untrained base. This is the stage-1 ablation; run `ft` for the full recipe.",
                stacklevel=3,
            )
            return
        if self._already_loaded_from(vlm, directory):
            self.debugger.log(f"backbone already points at {directory}")
            return
        self._original_model_id = vlm._model_id
        vlm._model_id = str(directory)
        # Anything already materialized came from the untrained base; drop it so
        # the trained weights are what actually runs.
        if vlm._model is not None or vlm._processor is not None:
            vlm._model = None
            vlm._processor = None
        print(f"[HiDrop] loading trained weights from {directory}", flush=True)

    def _load_pretrained_stage1(self, model: Any) -> None:
        """Stage-1 artifacts: copy the trained adapter over the base module.

        ``mm_projector.bin`` -> LLaVA's ``multi_modal_projector``, ``mm_merger.bin`` ->
        Qwen's ``visual.merger``; the backend owns which is which, as it does in the
        trainer that wrote the file.
        """
        directory = self._checkpoint_dir
        if directory is None or model_weight_files(directory):
            return
        backend = self._backend
        artifact = directory / backend.stage1_artifact_name
        state = torch.load(artifact, map_location="cpu", weights_only=True)
        # strict=True: an adapter trained against another base fails here rather
        # than silently projecting into the wrong space.
        backend.stage1_module(model).load_state_dict(state, strict=True)
        self.debugger.log(f"loaded pretrained stage-1 module from {artifact}")

    @staticmethod
    def _already_loaded_from(vlm: VLM, directory: Path) -> bool:
        model_id = getattr(vlm, "_model_id", None)
        if not model_id:
            return False
        candidate = Path(str(model_id)).expanduser()
        if not candidate.is_dir():
            return False
        return os.path.realpath(candidate) == os.path.realpath(directory)

    def _install_batch_size_guards(self, vlm: VLM) -> None:
        """Refuse any batch but 1.

        Batched generation left-pads, which the trained pruning state does not
        survive, so both entry points fail loudly rather than quietly evaluating a
        differently-pruned sequence.
        """
        self._original_prepare_inputs = getattr(vlm, "prepare_inputs", None)
        self._original_generate = getattr(vlm, "generate", None)
        if callable(self._original_prepare_inputs):
            original_prepare_inputs = self._original_prepare_inputs

            def wrapped_prepare_inputs(images: Any, prompts: Any, *args: Any, **kwargs: Any) -> Any:
                image_batch = (
                    int(images.shape[0])
                    if torch.is_tensor(images) and images.ndim != 3
                    else (1 if torch.is_tensor(images) else len(images))
                )
                prompt_batch = 1 if isinstance(prompts, str) else len(prompts)
                if image_batch != 1 or prompt_batch != 1:
                    raise ValueError(
                        f"HiDrop inference is forced to batch_size=1; got {image_batch} images and "
                        f"{prompt_batch} prompts."
                    )
                return original_prepare_inputs(images, prompts, *args, **kwargs)

            vlm.prepare_inputs = wrapped_prepare_inputs
        if callable(self._original_generate):
            original_generate = self._original_generate

            def wrapped_generate(inputs: Any, *args: Any, **kwargs: Any) -> Any:
                input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
                if torch.is_tensor(input_ids) and int(input_ids.shape[0]) != 1:
                    raise ValueError(
                        f"HiDrop inference is forced to batch_size=1; got generated input batch "
                        f"{int(input_ids.shape[0])}."
                    )
                if self.config.backbone_family == "llava_next":
                    generation_config = getattr(vlm.model, "generation_config", None)
                    num_beams = int(
                        kwargs.get("num_beams", getattr(generation_config, "num_beams", 1))
                    )
                    if num_beams != 1:
                        raise NotImplementedError(
                            "HiDrop/LLaVA-NeXT does not support beam search because beam expansion would "
                            "duplicate one mutable AnyRes pruning context. Use num_beams=1."
                        )
                    use_cache = kwargs.get(
                        "use_cache", getattr(generation_config, "use_cache", True)
                    )
                    if use_cache is not True:
                        raise NotImplementedError(
                            "HiDrop/LLaVA-NeXT generation requires use_cache=True. Re-prefilling the "
                            "variable packed image on every decode step is not a supported mode."
                        )
                with self._attention_backend():
                    return original_generate(inputs, *args, **kwargs)

            vlm.generate = wrapped_generate

    def _install_prompt_boundary_shim(self, vlm: VLM) -> None:
        """Keep the post-image SentencePiece boundary upstream LLaVA tokenizes with."""
        original_format_prompts = getattr(vlm, "format_prompts", None)
        if not callable(original_format_prompts):
            return
        self._original_format_prompts = original_format_prompts

        def wrapped_format_prompts(prompts: Any) -> list[str]:
            return [
                make_hf_prompt_upstream_tokenization_compatible(prompt)
                for prompt in original_format_prompts(prompts)
            ]

        vlm.format_prompts = wrapped_format_prompts

    def _align_processor_placeholder_count(self, vlm: VLM) -> None:
        """Make ``<image>`` expand to exactly ``image_seq_length`` placeholders."""
        processor = vlm.processor
        config = vlm.model.config
        vision_config = getattr(config, "vision_config", None)
        image_seq_length = getattr(config, "image_seq_length", None)
        if vision_config is None or image_seq_length is None:
            return
        if not hasattr(processor, "num_additional_image_tokens"):
            return
        patch_tokens = (int(vision_config.image_size) // int(vision_config.patch_size)) ** 2
        removes_cls = getattr(processor, "vision_feature_select_strategy", None) == "default"
        additional = int(image_seq_length) - patch_tokens + int(removes_cls)
        if additional < 0:
            raise RuntimeError(
                "Checkpoint processor cannot produce the configured image sequence length: "
                f"image_seq_length={int(image_seq_length)}, patch_tokens={patch_tokens}."
            )
        if additional == int(processor.num_additional_image_tokens):
            return
        self._original_num_additional_image_tokens = int(processor.num_additional_image_tokens)
        processor.num_additional_image_tokens = additional
        self.debugger.log(
            f"num_additional_image_tokens {self._original_num_additional_image_tokens} -> {additional}"
        )

    def _select_attn_implementation(self, model: Any) -> None:
        """Ensure the decoder runs an attention backend the wrapper supports."""
        text_config = model.model.language_model.config
        current = getattr(text_config, "_attn_implementation", None)
        if current not in _SUPPORTED_ATTN_IMPLEMENTATIONS:
            set_attn_implementation = getattr(model, "set_attn_implementation", None)
            if not callable(set_attn_implementation):
                raise ValueError(
                    f"HiDrop needs one of {_SUPPORTED_ATTN_IMPLEMENTATIONS}, and the backbone reports "
                    f"{current!r} without a way to change it."
                )
            set_attn_implementation("sdpa")
            selected = getattr(text_config, "_attn_implementation", None)
            if selected not in _SUPPORTED_ATTN_IMPLEMENTATIONS:
                raise RuntimeError(
                    f"Failed to select sdpa attention for the LLaVA checkpoint: {selected!r}."
                )
            self._original_attn_implementation = current
            self.debugger.log(f"attn implementation {current!r} -> {selected!r}")
            current = selected
        # Upstream evaluates inside `sdpa_kernel(FLASH_ATTENTION)` so inference
        # matches its FA2-trained numerics. That backend is CUDA-only.
        self._use_flash_sdpa = (
            self._sdpa_flash_only
            and current == "sdpa"
            and str(getattr(model, "device", "cpu")).startswith("cuda")
        )

    def _attention_backend(self) -> ContextManager[Any]:
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION) if self._use_flash_sdpa else nullcontext()

    @staticmethod
    def _resolve_image_token_id(vlm: VLM, model: Any) -> int:
        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(model.config, "image_token_index", None)
        tokenizer = getattr(getattr(vlm, "processor", None), "tokenizer", None)
        if image_token_id is None and tokenizer is not None:
            converter = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(converter):
                image_token_id = converter("<image>")
        if image_token_id is None or int(image_token_id) < 0:
            raise ValueError(
                "HiDrop could not find image_token_id/image_token_index on the HF LLaVA config or tokenizer."
            )
        return int(image_token_id)

    # ------------------------------------------------------- per-forward context

    def _build_hidrop_context(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> HiDropForwardContext | None:
        """Locate this prefill's visual span and snapshot what late injection restores.

        Returns ``None`` when the forward has no usable visual span (text-only row,
        or ``input_ids`` that do not line up with the hidden states), which leaves
        the decoder running unmodified.
        """
        ids = input_ids if input_ids is not None else self._latest_input_ids
        if ids is None:
            return None
        if (
            ids.ndim != 2
            or ids.shape[0] != hidden_states.shape[0]
            or ids.shape[1] != hidden_states.shape[1]
        ):
            return None
        if hidden_states.shape[0] != 1 and not self.config.batch_size_supported:
            raise ValueError("HiDrop currently supports batch_size=1 only.")
        is_llava_next = self.config.backbone_family == "llava_next"
        if is_llava_next and int(hidden_states.shape[0]) != 1:
            raise NotImplementedError("HiDrop/LLaVA-NeXT supports batch_size=1 only.")
        spans: list[VisualTokenSpan | None] = []
        image_rows = 0
        for batch_idx in range(int(ids.shape[0])):
            positions = torch.where(ids[batch_idx] == int(self._image_token_id))[0]
            if positions.numel() == 0:
                spans.append(None)
                continue
            image_rows += 1
            start = int(positions[0].item())
            end = int(positions[-1].item()) + 1
            expected = torch.arange(start, end, device=positions.device)
            if positions.numel() != expected.numel() or not torch.equal(positions, expected):
                raise ValueError(
                    f"HiDrop expected a single contiguous visual-token span in batch row {batch_idx}, got "
                    f"{int(positions.numel())} image tokens from {start} to {end}."
                )
            if (
                not is_llava_next
                and int(positions.numel()) != self.config.visual_tokens
                and not self.config.allow_nonstandard_vision_length
            ):
                raise ValueError(
                    f"HiDrop expected {self.config.visual_tokens} visual tokens for LLaVA-1.5 in batch row "
                    f"{batch_idx}, got {int(positions.numel())}."
                )
            input_seq_len = (
                int(attention_mask[batch_idx].sum().item())
                if attention_mask is not None
                else int(ids.shape[1])
            )
            spans.append(VisualTokenSpan(start=start, end=end, input_seq_len=input_seq_len))
        if image_rows == 0:
            # A text-only prefill is a new request, not a decode continuation.
            # Discard the dummy AnyRes layout and any prior image diagnostics so
            # they cannot be mistaken for this sequence's state.
            if is_llava_next:
                self._pending_llava_next_layouts = ()
            self.last_context = None
            self.last_visual_token_counts_by_layer = []
            self.last_patch_token_counts_by_layer = []
            self._source_history.clear()
            return None
        llava_next_layout = None
        resolved_schedule = None
        if is_llava_next:
            layouts = self._pending_llava_next_layouts
            if len(layouts) != image_rows or len(layouts) != 1:
                raise ValueError(
                    "HiDrop/LLaVA-NeXT requires exactly one source image and one captured AnyRes layout; "
                    f"found {image_rows} visual span(s) and {len(layouts)} layout(s)."
                )
            llava_next_layout = layouts[0]
            span_count = next(item.count for item in spans if item is not None)
            if span_count != llava_next_layout.native_tokens:
                raise ValueError(
                    "LLaVA-NeXT image placeholder expansion differs from HF AnyRes packing: "
                    f"placeholder rows={span_count}, geometry rows={llava_next_layout.native_tokens}."
                )
            # This is the actual HF-assembled, spatially unpadded decoder span.
            # Capture its content count here—after packing and before HiDrop has
            # injected, selected, or removed any rows. Structural newlines are
            # subtracted rather than treated as compressible content.
            original_content_tokens = span_count - int(llava_next_layout.newline_tokens)
            resolved_schedule = resolve_scaled_schedule(
                self.config,
                llava_next_layout,
                observed_content_tokens=original_content_tokens,
            )
        if labels is not None and labels.shape != ids.shape:
            raise ValueError(
                f"HiDrop labels must match input_ids before pruning, got labels {tuple(labels.shape)} "
                f"and input_ids {tuple(ids.shape)}."
            )
        span = next(item for item in spans if item is not None)
        self.last_context = None
        self.last_visual_token_counts_by_layer = []
        self.last_patch_token_counts_by_layer = []
        self._source_history.clear()
        self.debugger.log(f"visual_token_start={span.start}")
        self.debugger.log(f"visual_token_end={span.end}")
        self.debugger.log(f"num_visual_tokens={span.count}")
        self.debugger.log(
            f"batch_visual_spans={[(item.start, item.end) if item is not None else None for item in spans]}"
        )
        return HiDropForwardContext(
            span=span,
            spans=tuple(spans),
            original_hidden_states=hidden_states.clone(),
            original_attention_mask=attention_mask.clone() if attention_mask is not None else None,
            original_position_ids=position_ids.clone(),
            original_labels=labels.clone() if labels is not None else None,
            active_visual_count=0,
            active_patch_count=0,
            llava_next_layout=llava_next_layout,
            resolved_schedule=resolved_schedule,
            preserved_visual_rows=set(),
            pruning_events=[],
        )



def make_hf_prompt_upstream_tokenization_compatible(prompt: str) -> str:
    """Preserve the post-image SentencePiece boundary used by LLaVA's native tokenizer helper."""
    return prompt.replace("<image>\n", "<image> \n")
