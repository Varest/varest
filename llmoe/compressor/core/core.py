"""CORE: Compact Object-centric REpresentations for token merging in LVLMs.

Reimplementation of arXiv:2511.14072 as a whole vision-language model.  The
default decoder is Vicuna-7B-v1.5; decoder-only ports (including Llama-3-8B)
retain the same ConvNeXt-L/OMG-Seg vision architecture. CORE is not a
training-free compressor bolted onto a CLIP-ViT tower, so this module carries
the model too::

    image -> frozen ConvNeXt-L -> F3 (1/32 stride, 1536-d) ---------------+
                 |                                                       |
                 +-> frozen OMG-Seg Mask2Former -> 300 query masks ------+-> merge
                                                                            |
                                                  sorted object-centric tokens
                                                                            |
                                              2-layer GELU MLP (1536 -> 4096)
                                                                            |
    text -> decoder embeddings -> insert at <image> -> selected decoder -> answer

Object masks compete per pixel (Sec. 3.1), merge the ``F_3`` grid by Eq. (1),
and sort by weighted centroid (Sec. 3.2). The fixed rate -- the default, and the
intended way to run CORE -- emits exactly ``K`` tokens per image via Supplement
Algorithms 1/2. On the Llama-3 target, ``K`` is resolved independently for each
image as ``max(1, round(retain_ratio * N_original))`` after its uncompressed F3
content sequence exists. The adaptive rate emits one token per valid object, a
per-image count with no requested budget; it is an ablation.

The pieces, in order:

* :class:`CoreConfig` -- paper hyper-parameters (Sec. 4.1).
* Vicuna tokenizer/prompt/loader helpers.
* :class:`ObjectCentricMerger` and friends -- the merge, pure PyTorch and
  CPU-testable with no pretrained weights.
* :class:`SegmentationVision` -- frozen ConvNeXt-L + OMG-Seg head.
* :class:`CoreModel` -- vision -> merge -> projector -> decoder.
* :class:`CoreVLM` -- the Vicuna LLMOE backbone, registered as ``core``.
* :class:`CoreLlama3VLM` -- the decoder-only Llama-3-8B-Instruct port,
  registered as ``core-llama3-8b``.
* :class:`CORECompressor` -- the LLMOE compressor carrying the merge strategy.

Training lives in :mod:`llmoe.compressor.core_tr`. The frozen vision stack runs
on ``timm`` + ``transformers`` rather than MMDetection, so CORE installs and runs
in the same environment as every other baseline; see :class:`SegmentationVision`.
Weight-bearing dependencies are imported lazily, so the merge stays importable and
CPU-testable on its own.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import re
import warnings
from typing import Any, Literal, Mapping, Sequence, override

import torch
import torch.nn.functional as F
from mipcandy import Device, auto_device
from torch import nn

from llmoe.backbone.prototype import (
    IdentityImagePreprocessor,
    ImagePreprocessor,
    PromptPreprocessor,
    VLM,
)
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.weights import resolve_weights


# ============================================================================
# Configuration (Sec. 4.1)
# ============================================================================


# The OpenCLIP ConvNeXt-L that OMG-Seg pairs with its head (`convnext_large_d_320`
# / `laion2b_s29b_b131k_ft_soup`), under the name timm gives the same weights.
DEFAULT_VISION_TIMM_MODEL = "convnext_large_mlp.clip_laion2b_ft_soup_320"

# The authors' release. Public and anonymous-readable; the file is the frozen
# `panoptic_head.*` weights only, with the ConvNeXt trunk coming from timm.
DEFAULT_OMGSEG_REPO = "LXT/OMG_Seg"
DEFAULT_OMGSEG_FILENAME = "omg_seg_convl.pth"


# Training and inference intentionally use different decoder kernels. Training
# is pinned to the compiled FlashAttention-2 wheel and validated in `core_tr`;
# inference keeps Varest's PyTorch SDPA path (which can itself select a fused
# CUDA kernel without importing the external `flash_attn` package).
TRAINING_ATTENTION_IMPLEMENTATION = "flash_attention_2"
INFERENCE_ATTENTION_IMPLEMENTATION = "sdpa"

# Backward-compatible public name used by older training integrations. New code
# should use the explicit training/inference constants above.
ATTENTION_IMPLEMENTATION = TRAINING_ATTENTION_IMPLEMENTATION


MergeMode = Literal["soft", "hard"]


Rate = Literal["adaptive", "fixed"]


def validate_retention_ratio(retain_ratio: float) -> float:
    """Validate CORE's per-image content-token retention ratio."""

    ratio = float(retain_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
    return ratio


def retention_target(original_content_tokens: int, retain_ratio: float) -> int:
    """Return ``max(1, round(r * N_original))`` for one uncompressed image.

    CORE's paper specifies absolute fixed-rate targets, not a separate rounding
    rule for ratio-driven evaluation, so this port uses the requested canonical
    ``round`` rule. ``N_original`` must be captured from the assembled content
    tensor before object merging; padded text/image-placeholder lengths are not
    valid inputs here.
    """

    original = int(original_content_tokens)
    ratio = validate_retention_ratio(retain_ratio)
    if original < 1:
        raise ValueError("CORE cannot budget an empty visual content sequence")
    return max(1, min(original, int(round(ratio * original))))


@dataclass(frozen=True)
class CoreTokenBudget:
    """Per-image evaluation accounting, excluding structural image rows."""

    original_content_tokens: int
    requested_retention_ratio: float
    target_content_tokens: int
    realized_content_tokens: int
    realized_retention_ratio: float
    structural_tokens: int


@dataclass
class CoreConfig:
    """Hyper-parameters for the full CORE model.

    Attributes mirror Sec. 4.1 of the paper unless noted:

    Vision / segmentation (all frozen):
        image_size: Input images are resized to ``1024 x 1024``.
        pyramid_channels: ConvNeXt-L feature-pyramid channel dims for
            ``F_0..F_3`` (1/4..1/32 strides): 192, 384, 768, 1536.
        feature_stride: Stride of the ``F_3`` map fed to the LLM (1/32), so a
            1024px image yields a ``32 x 32 = 1024`` token grid.
        num_queries: Mask2Former blank object queries (``N = 300``).
        decoder_layers: Mask2Former transformer-decoder depth (``L = 9``).
        num_thing_classes / num_stuff_classes: OMG-Seg pretraining recognises
            80 *thing* + 53 *stuff* categories; everything else collapses to a
            single *special* class (134 total treated equally).
        pyramid_strides: Strides of ``F_0..F_3``; the last must equal
            ``feature_stride``, and the first is the mask-feature resolution.
        mask_feat_channels / pixel_decoder_* / decoder_*: Mask2Former head
            geometry. Fixed by the released checkpoint's tensor shapes, not tuned.
        vision_timm_model: timm name for the OpenCLIP ConvNeXt-L trunk.
        vision_random_init: Skip the ConvNeXt download and start from random
            weights. Shapes only -- for tests, never for a reported number.
        omgseg_repo / omgseg_filename / omgseg_checkpoint: Where the frozen
            OMG-Seg head weights come from. The first two name the public
            Hugging Face release; ``omgseg_checkpoint`` overrides them with a
            local file.

    Merging (Sec. 3.2 / 4.2):
        merge_mode: ``"soft"`` (overlapping probability weights) or ``"hard"``
            (non-overlapping argmax assignment; the paper's stronger variant).
        rate: ``"fixed"`` (the default and the intended way to run CORE: merge
            to the per-image target via Supplement Algorithms 1/2) or
            ``"adaptive"`` (one token per valid object, a count that changes
            with every image). Only the fixed rate gives a requested budget;
            adaptive is an ablation.
        fixed_tokens: Token count for the fixed-rate strategy (e.g. 160 / 320 /
            640 in the paper's Table 1), emitted exactly. Left unset it is the
            full grid, which is the uncompressed CORE (vanilla) baseline and
            bypasses the segmentation decoder because no mask can change the
            identity token sequence. Retained for the original fixed-geometry
            Vicuna integration; mutually exclusive with ``retain_ratio``.
        retain_ratio: Per-image evaluation budget for the Llama-3 target. The
            target is computed from the actual, pre-merge F3 content length with
            Python's ``round`` and a one-token minimum. CORE has no structural
            image rows, so those counts contain content tokens only.
        mask_threshold: Probability floor below which a soft weight is treated
            as background (0 keeps every pixel; used only for soft masks).
        large_object_first: Fixed-rate object ordering. ``True`` merges the
            largest objects first (the paper's default, Sec. 4.2); ``False`` is
            the small-object-first ablation (Supplementary Sec. A).

    Projector / LLM:
        projector_hidden_act: GELU, per Sec. 4.1.
        llm_name_or_path: Decoder checkpoint. Decoder-only ports may point at a
            multimodal checkpoint whose language weights can be extracted.
        llm_revision: Optional immutable source revision.
        llm_hidden_size: Projector output and decoder embedding width.
        lora_*: LoRA config for stage-2 tuning (r=512, alpha=256, dropout=0.05,
            no bias).

    Runtime:
        attention_implementation: PyTorch ``sdpa`` for inference by default.
            CORE training rejects every value except ``flash_attention_2`` and
            additionally validates its pinned compiled package before loading.
    """

    # --- vision / segmentation -------------------------------------------------
    image_size: int = 1024
    pyramid_channels: tuple[int, int, int, int] = (192, 384, 768, 1536)
    feature_stride: int = 32
    num_queries: int = 300
    decoder_layers: int = 9
    num_thing_classes: int = 80
    num_stuff_classes: int = 53
    pyramid_strides: tuple[int, int, int, int] = (4, 8, 16, 32)
    mask_feat_channels: int = 256
    pixel_decoder_layers: int = 6
    pixel_decoder_ffn_dim: int = 1024
    decoder_ffn_dim: int = 2048
    decoder_heads: int = 8
    vision_timm_model: str = DEFAULT_VISION_TIMM_MODEL
    # Tests and dry runs that only need shapes; skips the ConvNeXt weight download.
    vision_random_init: bool = False
    omgseg_repo: str = DEFAULT_OMGSEG_REPO
    omgseg_filename: str = DEFAULT_OMGSEG_FILENAME
    # Explicit local `.pth`; unset reads `omgseg_filename` from the HF cache.
    omgseg_checkpoint: str | None = None
    # CLIP preprocessing used by the official OMG-LLaVA ConvNeXt-L recipe.
    vision_mean: tuple[float, float, float] = (0.4814, 0.4578, 0.4082)
    vision_std: tuple[float, float, float] = (0.2686, 0.2613, 0.2757)
    pad_image_to_square: bool = True

    # --- merging (Sec. 3.2 / 4.2) ---------------------------------------------
    merge_mode: MergeMode = "hard"
    rate: Rate = "fixed"
    fixed_tokens: int | None = None
    retain_ratio: float | None = None
    mask_threshold: float = 0.0
    large_object_first: bool = True

    # --- projector / LLM (Sec. 4.1) -------------------------------------------
    vision_hidden_size: int = 1536  # channels of F_3
    projector_hidden_act: str = "gelu"
    llm_name_or_path: str | None = None
    llm_revision: str | None = None
    llm_hidden_size: int = 4096
    image_token: str = "<image>"

    # --- LoRA (stage 2) --------------------------------------------------------
    use_lora: bool = True
    lora_r: int = 512
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: tuple[str, ...] | None = None

    # --- runtime ---------------------------------------------------------------
    attention_implementation: str = INFERENCE_ATTENTION_IMPLEMENTATION
    vision_dtype: str = "float32"  # training vision precision (Sec. 4.1)
    inference_vision_dtype: str = "float16"  # evaluation vision precision
    inference_dtype: str = "float16"  # projector / LLM evaluation precision

    def __post_init__(self) -> None:
        if self.llm_name_or_path is None:
            self.llm_name_or_path = DEFAULT_MODEL
        if (
            self.rate == "fixed"
            and self.fixed_tokens is None
            and self.retain_ratio is None
        ):
            # A backbone with no compressor is the uncompressed CORE (vanilla)
            # baseline, so the full grid is the right budget to fall back to.
            self.fixed_tokens = self.num_visual_tokens
        self.lora_target_modules = tuple(self.lora_target_modules or LORA_TARGET_MODULES)

    @property
    def llm_family(self) -> str:
        """Decoder family read from checkpoint metadata, never its path name."""
        from llmoe.compressor.core.core_qwen_lm import llm_family

        return llm_family(
            str(self.llm_name_or_path),
            local_files_only=True,
            revision=self.llm_revision,
        )

    @property
    def grid_size(self) -> int:
        """Side length of the F_3 token grid (image_size / feature_stride)."""
        return self.image_size // self.feature_stride

    @property
    def num_visual_tokens(self) -> int:
        """Full (uncompressed) visual-token count, e.g. 32*32 = 1024."""
        return self.grid_size * self.grid_size

    def validate(self) -> None:
        if not self.llm_name_or_path:
            raise ValueError("llm_name_or_path must be non-empty")
        if not self.lora_target_modules:
            raise ValueError("lora_target_modules must be non-empty")
        if self.merge_mode not in ("soft", "hard"):
            raise ValueError(f"merge_mode must be 'soft' or 'hard', got {self.merge_mode}")
        if self.rate not in ("adaptive", "fixed"):
            raise ValueError(f"rate must be 'adaptive' or 'fixed', got {self.rate}")
        if self.rate == "fixed":
            if self.fixed_tokens is not None and self.retain_ratio is not None:
                raise ValueError("fixed_tokens and retain_ratio are mutually exclusive")
            if self.retain_ratio is not None:
                validate_retention_ratio(self.retain_ratio)
            else:
                if self.fixed_tokens is None or self.fixed_tokens < 1:
                    raise ValueError(
                        "fixed rate requires retain_ratio or fixed_tokens >= 1"
                    )
                if self.fixed_tokens > self.num_visual_tokens:
                    raise ValueError(
                        f"fixed_tokens={self.fixed_tokens} exceeds the visual grid "
                        f"({self.num_visual_tokens})"
                    )
            if self.merge_mode != "hard":
                raise ValueError("fixed rate is defined only for hard masks")
        elif self.retain_ratio is not None:
            raise ValueError("retain_ratio applies only to CORE's fixed-rate strategy")
        if self.vision_hidden_size != self.pyramid_channels[-1]:
            raise ValueError(
                "vision_hidden_size must equal the last pyramid channel dim "
                f"({self.pyramid_channels[-1]})"
            )
        if self.attention_implementation not in ("flash_attention_2", "sdpa"):
            raise ValueError(
                "attention_implementation must be 'flash_attention_2' or 'sdpa', got "
                f"{self.attention_implementation!r}"
            )
        for name in ("vision_dtype", "inference_vision_dtype", "inference_dtype"):
            value = getattr(self, name)
            if value not in ("float16", "bfloat16", "float32"):
                raise ValueError(f"{name} must be float16, bfloat16, or float32, got {value}")
        if len(self.vision_mean) != 3 or len(self.vision_std) != 3:
            raise ValueError("vision_mean and vision_std must contain three RGB values")
        if any(std <= 0 for std in self.vision_std):
            raise ValueError("vision_std values must be positive")


# ============================================================================
# Vicuna-7B-v1.5 decoder: tokenizer, prompt, loading
# ============================================================================


# Prepared locally by the CORE asset step; point `llm_name_or_path` at that
# directory when it does not sit in the working directory.
DEFAULT_MODEL = "vicuna-7b-v1.5"


LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# Stage 1 uses the LLaVA plain caption recipe, stage 2 the Vicuna-v1 conversation
# recipe; `checkpoint.training_provenance` records which one produced a run.
TRAINING_RECIPES = ("llava_plain", "vicuna_v1")


VICUNA_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def load_tokenizer(config: Any, *, local_files_only: bool) -> Any:
    """Load and validate the tokenizer selected by the decoder family."""

    from transformers import AutoTokenizer

    from llmoe.compressor.core.core_qwen_lm import llm_family, load_qwen_tokenizer

    family = llm_family(
        str(config.llm_name_or_path),
        local_files_only=local_files_only,
        revision=getattr(config, "llm_revision", None),
    )
    if family == "qwen":
        return load_qwen_tokenizer(config, local_files_only=local_files_only)
    if family == "llama3":
        from llmoe.compressor.core.core_llama3_lm import load_llama3_tokenizer

        return load_llama3_tokenizer(config, local_files_only=local_files_only)

    tokenizer = AutoTokenizer.from_pretrained(
        config.llm_name_or_path,
        trust_remote_code=False,
        # Official LLaVA-v1 masking was defined against the slow Llama tokenizer.
        use_fast=False,
        local_files_only=local_files_only,
    )
    validate_tokenizer(tokenizer)
    return tokenizer


def validate_tokenizer(tokenizer: Any) -> None:
    """Fail closed when a tokenizer does not match the LLaVA-v1 recipe."""

    required = {
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Vicuna tokenizer is missing required fields: {missing}")
    if (getattr(tokenizer, "eos_token", None) or "</s>") != "</s>":
        raise ValueError("Vicuna-v1 requires </s> as its end-of-turn token")


def tokenize_image_prompt(
    tokenizer: Any,
    prompt: str,
    image_token: str,
    *,
    repetitions: int = 1,
) -> list[int]:
    """Tokenize one image prompt with official LLaVA boundary semantics.

    LLaVA tokenizes the text on each side of ``<image>`` independently and
    inserts its image sentinel between the chunks. Asking SentencePiece to
    tokenize a newly registered special token in-place is not equivalent: it
    can remove or add a word-boundary token adjacent to the marker. CORE uses a
    positive vocabulary ID for the sentinel, but otherwise follows the same
    split-and-insert algorithm. ``repetitions`` expands that one sentinel to
    the realized visual-token count for inference.
    """

    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    marker_count = prompt.count(image_token)
    if marker_count != 1:
        raise ValueError(
            f"CORE expects exactly one {image_token!r} marker, got {marker_count}"
        )

    chunks: list[list[int]] = []
    for chunk in prompt.split(image_token):
        encoded = tokenizer(chunk, add_special_tokens=True)
        token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise ValueError("CORE expected one tokenized prompt chunk")
            token_ids = token_ids[0]
        chunks.append([int(token_id) for token_id in token_ids])

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    offset = 0
    result: list[int] = []
    if chunks[0] and bos_token_id is not None and chunks[0][0] == int(bos_token_id):
        offset = 1
        result.append(chunks[0][0])

    image_id = int(tokenizer.convert_tokens_to_ids(image_token))
    separator = [image_id] * (offset + repetitions)
    for index, chunk in enumerate(chunks):
        result.extend(chunk[offset:])
        if index + 1 < len(chunks):
            result.extend(separator[offset:])
    return result


def format_inference_prompt(
    image_token: str,
    prompt: str,
    family: str = "llama",
    *,
    tokenizer: Any | None = None,
) -> str:
    """The single-image benchmark prompt for the decoder in use.

    ``family`` defaults to Vicuna's so every existing caller keeps its behaviour;
    a Qwen decoder was tuned under ChatML and answers poorly outside it.
    """

    if family == "qwen":
        from llmoe.compressor.core.core_qwen_lm import format_qwen_inference_prompt

        return format_qwen_inference_prompt(image_token, prompt)
    if family == "llama3":
        if tokenizer is None:
            raise ValueError("Llama-3 prompt formatting requires its pinned tokenizer")
        from llmoe.compressor.core.core_llama3_lm import format_llama3_inference_prompt

        return format_llama3_inference_prompt(tokenizer, image_token, prompt)
    return f"{VICUNA_SYSTEM_PROMPT} USER: {image_token}\n{prompt} ASSISTANT:"


def model_attention_implementation(model: Any) -> str | None:
    config = model.config
    return getattr(
        config,
        "attn_implementation",
        getattr(config, "_attn_implementation", None),
    )


def load_causal_lm(
    config: Any,
    *,
    dtype: torch.dtype | None,
    local_files_only: bool,
    attention_implementation: str,
) -> Any:
    """Load only the configured causal decoder with the requested kernel."""

    from transformers import AutoConfig, AutoModelForCausalLM

    from llmoe.compressor.core.core_qwen_lm import llm_family, load_qwen_causal_lm

    family = llm_family(
        str(config.llm_name_or_path),
        local_files_only=local_files_only,
        revision=getattr(config, "llm_revision", None),
    )
    if family == "qwen":
        return load_qwen_causal_lm(
            config, dtype=dtype, local_files_only=local_files_only,
            attention_implementation=attention_implementation)
    if family == "llama3":
        from llmoe.compressor.core.core_llama3_lm import load_llama3_causal_lm

        return load_llama3_causal_lm(
            config,
            dtype=dtype,
            local_files_only=local_files_only,
            attention_implementation=attention_implementation,
        )

    llm_config = AutoConfig.from_pretrained(
        config.llm_name_or_path,
        trust_remote_code=False,
        local_files_only=local_files_only,
    )
    if getattr(llm_config, "model_type", None) != "llama":
        raise TypeError(
            "CORE-Vicuna must resolve to a Llama causal LM; got "
            f"model_type={getattr(llm_config, 'model_type', None)!r}"
        )
    llm_config._attn_implementation = attention_implementation
    model = AutoModelForCausalLM.from_pretrained(
        config.llm_name_or_path,
        config=llm_config,
        trust_remote_code=False,
        dtype=dtype,
        attn_implementation=attention_implementation,
        local_files_only=local_files_only,
    )

    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size != int(config.llm_hidden_size):
        raise RuntimeError(
            f"Vicuna hidden size {hidden_size} does not match CORE projector output "
            f"{config.llm_hidden_size}"
        )
    actual = model_attention_implementation(model)
    if actual != attention_implementation:
        raise RuntimeError(
            "Vicuna did not activate the requested attention backend: "
            f"requested {attention_implementation!r}, got {actual!r}"
        )
    return model


# ============================================================================
# Object-centric merging (Sec. 3.1 / 3.2 / 4.2)
# ============================================================================


@dataclass
class MergedTokens:
    """Result of one image's object-centric merge.

    Attributes:
        tokens: ``[K, C]`` merged object tokens, sorted by ascending centroid.
        centroids: ``[K]`` raster-scan centroid position of each token.
        source: ``[K, HW]`` row-normalised weights mapping each merged token to
            the original ``F_3`` tokens it subsumes (for ``llmoe.viz`` and for
            inspecting the compression). Rows sum to 1.
    """

    tokens: torch.Tensor
    centroids: torch.Tensor
    source: torch.Tensor


def competitive_strategy(
    masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter ``N`` probability masks into valid soft/hard weights (Sec. 3.1).

    Args:
        masks: ``[N, H, W]`` per-query probabilities after sigmoid, already
            down-sampled to the ``F_3`` grid.

    Returns:
        ``(valid_idx, soft, hard)`` where

        * ``valid_idx`` is a ``[M]`` long tensor of the query indices that are
          the per-pixel maximum for at least one pixel (the valid objects);
        * ``soft`` is ``[M, HW]`` overlapping probability weights (the raw
          probabilities of the valid masks, raster-flattened);
        * ``hard`` is ``[M, HW]`` non-overlapping ``{0,1}`` weights, where pixel
          ``i`` is assigned to the single valid query with the highest
          probability at ``i``.

    ``M`` is data-dependent — this is exactly the adaptive token count.
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must be [N, H, W], got {tuple(masks.shape)}")
    num_queries, height, width = masks.shape
    flat = masks.reshape(num_queries, height * width)  # [N, HW]

    # Per-pixel winner across all N queries (the "competition").
    winner = flat.argmax(dim=0)  # [HW]
    valid_idx = torch.unique(winner)  # queries maximal for >= 1 pixel
    valid_idx, _ = torch.sort(valid_idx)

    soft = flat[valid_idx]  # [M, HW] overlapping probabilities

    # Hard assignment: 1 where this valid query is the pixel winner.
    # Map global winner ids onto positions within valid_idx.
    hard = (winner.unsqueeze(0) == valid_idx.unsqueeze(1)).to(masks.dtype)  # [M, HW]
    return valid_idx, soft, hard


def _weighted_merge(
    features: torch.Tensor, weights: torch.Tensor, *, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eq. (1): weighted token + centroid + row-normalised source, for all masks.

    Args:
        features: ``[HW, C]`` flattened ``F_3`` tokens (raster order).
        weights: ``[M, HW]`` per-mask weights ``omega_{n,i}``.

    Returns:
        ``(tokens[M, C], centroids[M], source[M, HW])``.
    """
    hw = features.shape[0]
    denom = weights.sum(dim=1, keepdim=True).clamp_min(eps)  # [M, 1]
    tokens = (weights @ features) / denom  # [M, C]
    positions = torch.arange(hw, device=features.device, dtype=features.dtype)  # [HW]
    centroids = (weights @ positions.unsqueeze(1)).squeeze(1) / denom.squeeze(1)  # [M]
    source = weights / denom  # row-normalised weights, rows sum to 1
    return tokens, centroids, source


def merge_by_masks(
    features: torch.Tensor, weights: torch.Tensor, *, sort: bool = True
) -> MergedTokens:
    """Merge ``F_3`` tokens by the given per-mask weights, then centroid-sort.

    Implements the adaptive path: one object token per (valid) mask, ordered by
    ascending centroid (Fig. 3, Sec. 3.2).

    Args:
        features: ``[HW, C]`` flattened ``F_3`` tokens (raster order).
        weights: ``[M, HW]`` weights (soft probabilities or hard ``{0,1}``).
        sort: When ``True`` (default) sort tokens by ascending centroid.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be [HW, C], got {tuple(features.shape)}")
    if weights.ndim != 2 or weights.shape[1] != features.shape[0]:
        raise ValueError(
            f"weights must be [M, HW={features.shape[0]}], got {tuple(weights.shape)}"
        )
    tokens, centroids, source = _weighted_merge(features, weights.to(features.dtype))
    if sort:
        order = torch.argsort(centroids)
        tokens, centroids, source = tokens[order], centroids[order], source[order]
    return MergedTokens(tokens=tokens, centroids=centroids, source=source)


def _raster_groups_from_hard(hard: torch.Tensor) -> list[torch.Tensor]:
    """Per-object token indices (raster order) from hard ``[M, HW]`` weights."""
    groups: list[torch.Tensor] = []
    for m in range(hard.shape[0]):
        idx = torch.nonzero(hard[m] > 0, as_tuple=False).flatten()  # already raster-sorted
        if idx.numel() > 0:
            groups.append(idx)
    return groups


def fixed_rate_merge(
    features: torch.Tensor,
    hard: torch.Tensor,
    target_tokens: int,
    *,
    large_object_first: bool = True,
    sort: bool = True,
) -> MergedTokens:
    """Fixed-rate object-order merging to exactly ``target_tokens`` (Sec. 4.2).

    The adaptive path yields a data-dependent number of object tokens; batched
    inference and fair-comparison ("fixed-rate") settings need a constant count.
    Following Sec. 4.2:

    * objects are ordered by area (token count) — largest first by default,
      because larger objects carry more intra-object redundancy;
    * each object is collapsed, in raster order, into a single token, and its
      original tokens are removed from the pool;
    * this repeats until the pool holds exactly ``target_tokens`` tokens; the
      boundary object is only *partially* merged (its leading raster tokens
      collapse into one token, the rest stay individual) so the count lands
      exactly on the target.
    * when the detected object count itself exceeds the strict target, retain
      the largest ``target_tokens - 1`` objects and average all smaller overflow
      regions into one catch-all token. This minimal exact-budget extension
      preserves coverage without introducing a learned pruning policy.

    The output is the union of merged-group tokens and still-individual tokens,
    centroid-sorted like the adaptive path. Every original ``F_3`` token maps to
    exactly one output token, so no information is dropped — only merged.

    This follows Supplement Algorithm 1 exactly: area priority with ascending
    mask-ID tie breaking, full/partial/no-merge budget cases, and a final sort
    by original or averaged centroid. ``large_object_first=False`` changes the
    area ordering as specified by Supplement Algorithm 2.

    Args:
        features: ``[HW, C]`` flattened ``F_3`` tokens (raster order).
        hard: ``[M, HW]`` hard ``{0,1}`` weights (non-overlapping, from
            :func:`competitive_strategy`).
        target_tokens: Desired output token count ``K`` (``1 <= K <= HW``).
    """
    hw = features.shape[0]
    if not 1 <= target_tokens <= hw:
        raise ValueError(f"target_tokens must be in [1, {hw}], got {target_tokens}")

    # Table 1's 1024-token CORE (vanilla) is the uncompressed raster-ordered
    # F_3 grid. Preserve it exactly and avoid constructing the same identity
    # mapping one row at a time.
    if target_tokens == hw:
        centroids = torch.arange(hw, device=features.device, dtype=features.dtype)
        source = torch.eye(hw, device=features.device, dtype=features.dtype)
        return MergedTokens(tokens=features, centroids=centroids, source=source)

    groups = _raster_groups_from_hard(hard)
    # Every token must belong to exactly one object; hard masks from the
    # competitive strategy already guarantee a full, non-overlapping partition.
    groups.sort(key=lambda g: g.numel(), reverse=large_object_first)

    # The paper permits the realized count to exceed the target when the object
    # count is larger. Our strict-budget extension keeps the largest objects and
    # combines every smaller overflow region into one area-weighted catch-all.
    # This uses the paper's existing area order and arithmetic averaging, adds
    # no learned policy, and ensures every source patch remains represented.
    num_objects = len(groups)
    if target_tokens < num_objects:
        overflow = torch.cat(groups[target_tokens - 1 :])
        strict_groups = groups[: target_tokens - 1] + [overflow]
        weights = torch.zeros(
            target_tokens,
            hw,
            dtype=features.dtype,
            device=features.device,
        )
        for row, group in enumerate(strict_groups):
            weights[row, group] = 1.0
        return merge_by_masks(features, weights, sort=sort)

    remaining = hw
    merged_groups: list[torch.Tensor] = []
    merged_members: set[int] = set()

    for group in groups:
        if remaining <= target_tokens:
            break
        size = group.numel()
        if remaining - (size - 1) >= target_tokens:
            # Collapse the whole object into one token.
            merged_groups.append(group)
            merged_members.update(group.tolist())
            remaining -= size - 1
        else:
            # Partial merge on the boundary object: collapse just enough leading
            # raster tokens to remove exactly (remaining - target_tokens) tokens.
            to_remove = remaining - target_tokens
            head = group[: to_remove + 1]
            merged_groups.append(head)
            merged_members.update(head.tolist())
            remaining = target_tokens
            break

    # Build the [K, HW] weight matrix: one row per merged group (uniform over its
    # members) plus one row per still-individual token.
    rows: list[torch.Tensor] = []
    for group in merged_groups:
        row = torch.zeros(hw, dtype=features.dtype, device=features.device)
        row[group] = 1.0
        rows.append(row)
    for i in range(hw):
        if i not in merged_members:
            row = torch.zeros(hw, dtype=features.dtype, device=features.device)
            row[i] = 1.0
            rows.append(row)

    weights = torch.stack(rows, dim=0)  # [K, HW]
    return merge_by_masks(features, weights, sort=sort)


class ObjectCentricMerger:
    """Stateless driver that turns (``F_3``, masks) into object-centric tokens.

    Wraps the competitive strategy + merging + (optional) fixed-rate strategy so
    the model and the LLMOE compressor share one code path.

    Args:
        merge_mode: ``"soft"`` or ``"hard"`` (Sec. 3.2).
        rate: ``"fixed"`` (merge to an explicit per-image target; the intended
            way to run CORE) or ``"adaptive"`` (one token per object, variable).
        fixed_tokens: Legacy absolute target for fixed-geometry CORE backbones.
        retain_ratio: Preferred Llama-3 evaluation budget. Resolved from the
            actual input sequence length independently in every ``merge`` call.
        mask_threshold: Zero soft weights below this probability (soft only).
        large_object_first: Fixed-rate ordering (Sec. 4.2 vs. Supp. A ablation).
    """

    def __init__(
        self,
        *,
        merge_mode: str = "hard",
        rate: str = "fixed",
        fixed_tokens: int | None = None,
        retain_ratio: float | None = None,
        mask_threshold: float = 0.0,
        large_object_first: bool = True,
    ) -> None:
        if merge_mode not in ("soft", "hard"):
            raise ValueError(f"merge_mode must be 'soft'/'hard', got {merge_mode}")
        if rate not in ("adaptive", "fixed"):
            raise ValueError(f"rate must be 'adaptive'/'fixed', got {rate}")
        if rate == "fixed" and fixed_tokens is not None and retain_ratio is not None:
            raise ValueError("fixed_tokens and retain_ratio are mutually exclusive")
        if rate == "fixed" and fixed_tokens is None and retain_ratio is None:
            raise ValueError("fixed rate requires retain_ratio or fixed_tokens >= 1")
        if rate == "fixed" and fixed_tokens is not None and fixed_tokens < 1:
            raise ValueError("fixed_tokens must be >= 1")
        if rate == "fixed" and retain_ratio is not None:
            retain_ratio = validate_retention_ratio(retain_ratio)
        if rate == "adaptive" and retain_ratio is not None:
            raise ValueError("retain_ratio applies only to CORE's fixed-rate strategy")
        if rate == "fixed" and merge_mode == "soft":
            # The paper's fixed-rate strategy is defined over the hard-mask
            # object partition (Sec. 4.2); soft masks overlap and cannot give a
            # clean token count.
            raise ValueError("fixed rate is only defined for merge_mode='hard'")
        self.merge_mode = merge_mode
        self.rate = rate
        self.fixed_tokens = fixed_tokens
        self.retain_ratio = retain_ratio
        self.mask_threshold = float(mask_threshold)
        self.large_object_first = bool(large_object_first)
        self.last_budget: CoreTokenBudget | None = None

    def target_for(self, original_content_tokens: int) -> int:
        """Resolve this image's target from its uncompressed content length."""

        original = int(original_content_tokens)
        if original < 1:
            raise ValueError("CORE cannot budget an empty visual content sequence")
        if self.rate != "fixed":
            raise RuntimeError("Adaptive CORE has no requested target token count")
        if self.retain_ratio is not None:
            return retention_target(original, self.retain_ratio)
        target = int(self.fixed_tokens)
        if target > original:
            raise ValueError(
                f"fixed_tokens={target} exceeds this image's original content "
                f"length ({original})"
            )
        return target

    def budget_record(
        self,
        original_content_tokens: int,
        realized_content_tokens: int,
        *,
        structural_tokens: int = 0,
    ) -> CoreTokenBudget:
        """Build the required content-only accounting for one image."""

        original = int(original_content_tokens)
        realized = int(realized_content_tokens)
        target = self.target_for(original)
        requested = (
            float(self.retain_ratio)
            if self.retain_ratio is not None
            else target / original
        )
        if not 1 <= realized <= original:
            raise ValueError(
                f"CORE realized {realized} content tokens from an original {original}"
            )
        return CoreTokenBudget(
            original_content_tokens=original,
            requested_retention_ratio=requested,
            target_content_tokens=target,
            realized_content_tokens=realized,
            realized_retention_ratio=realized / original,
            structural_tokens=int(structural_tokens),
        )

    def merge(self, features: torch.Tensor, masks: torch.Tensor) -> MergedTokens:
        """Compress one image's ``F_3`` grid using its object masks.

        Args:
            features: ``[HW, C]`` flattened ``F_3`` tokens (raster order), or a
                ``[H, W, C]`` grid which is flattened internally.
            masks: ``[N, H, W]`` per-query probabilities (sigmoid, at ``F_3``
                resolution).
        """
        if features.ndim == 3:
            height, width, channels = features.shape
            features = features.reshape(height * width, channels)
        # Capture N_original from the complete content tensor before even the
        # competitive mask selection runs. The target must never inherit a
        # count from valid objects or an already-merged representation.
        original_content_tokens = int(features.shape[0])
        target = (
            self.target_for(original_content_tokens)
            if self.rate == "fixed"
            else None
        )
        valid_idx, soft, hard = competitive_strategy(masks)
        del valid_idx

        if self.rate == "fixed":
            result = fixed_rate_merge(
                features,
                hard,
                int(target),
                large_object_first=self.large_object_first,
            )
            self.last_budget = self.budget_record(
                original_content_tokens, int(result.tokens.shape[0])
            )
            return result

        if self.merge_mode == "hard":
            weights = hard
        else:
            weights = soft
            if self.mask_threshold > 0.0:
                weights = torch.where(
                    weights >= self.mask_threshold, weights, torch.zeros_like(weights)
                )
        self.last_budget = None
        return merge_by_masks(features, weights)


# ============================================================================
# Frozen vision + segmentation stack (Sec. 3.1)
# ============================================================================


class SegmentationOutput:
    """Container for one forward pass of the frozen vision stack.

    Attributes:
        f3: ``[B, C, H, W]`` semantically-rich map fed to the LLM (1/32 stride).
        masks: ``[B, N, H, W]`` per-query probabilities (sigmoid), already
            down-sampled to the ``F_3`` grid, ready for the merge module.
        pyramid: The full ``(F_0, F_1, F_2, F_3)`` tuple (for debugging / viz).
    """

    def __init__(
        self,
        f3: torch.Tensor,
        masks: torch.Tensor | None,
        pyramid: tuple[torch.Tensor, ...],
    ) -> None:
        self.f3 = f3
        self.masks = masks
        self.pyramid = pyramid


def _resize_to(masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize ``[B, N, H, W]`` masks to ``size`` (the ``F_3`` grid)."""
    if masks.shape[-2:] == tuple(size):
        return masks
    return F.interpolate(masks, size=size, mode="bilinear", align_corners=False)



class SegmentationVision(nn.Module):
    """ConvNeXt-L + Mask2Former (OMG-Seg), frozen, shared across both heads.

    OMG-Seg is published as MMDetection weights, but nothing here runs MMDetection.
    The released checkpoint is head-only (``panoptic_head.*``, ~84MB), and its two
    halves both have exact counterparts in libraries this project already pins:

    * ConvNeXt-L feature pyramid -> ``timm``. The OpenCLIP model OMG-Seg names
      (``convnext_large_d_320`` / ``laion2b_s29b_b131k_ft_soup``) is the same
      network and the same weights as timm's ``DEFAULT_VISION_TIMM_MODEL``, which
      emits ``F_0..F_3`` at strides 4/8/16/32 with 192/384/768/1536 channels.
    * Mask2Former head -> ``transformers``. Its ``Mask2FormerPixelDecoder`` and
      ``Mask2FormerTransformerModule`` are the architecture MMDetection builds,
      in pure PyTorch: the multi-scale deformable attention is the ``grid_sample``
      reference implementation, so no CUDA kernel is compiled and no ``mmcv``
      build has to match the pinned torch.

    :func:`omgseg_state_dict_to_hf` translates the checkpoint's layer names; see
    it for the correspondence and the one tensor that needs more than a rename.

    Args:
        config: :class:`~llmoe.compressor.core.CoreConfig`.
    """

    def __init__(self, config: CoreConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone: nn.Module | None = None
        self.final_norm: nn.Module | None = None
        self.pixel_decoder: nn.Module | None = None
        self.transformer: nn.Module | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        local_files_only: bool = True,
    ) -> "SegmentationVision":
        """Build ConvNeXt-L + Mask2Former and load the frozen OMG-Seg weights.

        ``local_files_only`` follows LLMOE's backbone convention: the ConvNeXt and
        OMG-Seg weights are public checkpoints read from the Hugging Face cache, so
        it decides whether a missing copy is downloaded or is an error.
        """
        if self._loaded:
            return self
        cfg = self.config
        import timm  # deferred: unused until a real model is built

        if cfg.vision_random_init:
            # Nothing downstream fails on random features -- the merge runs, the
            # decoder answers, the benchmark scores -- so say so loudly here.
            warnings.warn(
                "CORE's ConvNeXt-L is randomly initialised (vision_random_init). "
                "Shapes are right and results are meaningless; this is a dry run.",
                stacklevel=2,
            )

        # --- ConvNeXt-L ----------------------------------------------------
        # `num_classes=0` keeps the CLIP head modules -- `head.norm` is needed
        # below -- while dropping the classifier. The stages are driven manually
        # rather than through `features_only=True`, which strips that norm off.
        backbone = timm.create_model(
            cfg.vision_timm_model,
            pretrained=not cfg.vision_random_init,
            num_classes=0,
        )
        # On a full timm model (as opposed to `features_only=True`) `feature_info`
        # is a plain list of per-stage dicts.
        channels = tuple(int(stage["num_chs"]) for stage in backbone.feature_info)
        strides = tuple(int(stage["reduction"]) for stage in backbone.feature_info)
        if channels != cfg.pyramid_channels or strides != cfg.pyramid_strides:
            raise ValueError(
                f"{cfg.vision_timm_model} has pyramid channels {channels} at strides "
                f"{strides}; CORE requires {cfg.pyramid_channels} at {cfg.pyramid_strides}"
            )
        self.backbone = backbone
        # ConvNeXt's pretrained final channel norm. `norm_pre` is Identity in this
        # OpenCLIP/timm build; the real LayerNorm2d lives on the head. Feeding the
        # raw stage-4 tensor to CORE's published projector produces embeddings
        # three orders of magnitude too large and can overflow the language decoder.
        self.final_norm = backbone.head.norm

        # --- Mask2Former head ----------------------------------------------
        # The 1024-token vanilla recipe consumes the F3 grid unchanged. No mask can
        # change an identity token sequence, so for that budget skip building the
        # six-layer pixel decoder and nine-layer transformer decoder entirely.
        if self._needs_masks():
            from transformers.models.mask2former.modeling_mask2former import (  # deferred
                Mask2FormerPixelDecoder,
                Mask2FormerTransformerModule,
            )

            hf_config = self.mask2former_config()
            self.pixel_decoder = Mask2FormerPixelDecoder(
                hf_config, feature_channels=list(cfg.pyramid_channels)
            )
            self.transformer = Mask2FormerTransformerModule(
                in_features=cfg.mask_feat_channels, config=hf_config
            )
            head = load_omgseg_checkpoint(cfg, local_files_only=local_files_only)
            self._load_head_weights(omgseg_state_dict_to_hf(head))

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)
        self.eval()
        for param in self.parameters():  # CORE never trains this stack
            param.requires_grad_(False)
        self._loaded = True
        return self

    def mask2former_config(self) -> Any:
        """The OMG-Seg head's geometry as a ``Mask2FormerConfig``.

        Every value is fixed by the released checkpoint's tensor shapes, so this is
        a transcription rather than a set of tunables. ``decoder_layers`` is one
        more than the built layer count: ``transformers`` reserves the first index
        for the prediction the decoder makes before layer 0.
        """
        from transformers import Mask2FormerConfig  # deferred

        cfg = self.config
        return Mask2FormerConfig(
            backbone_config=None,
            feature_size=cfg.mask_feat_channels,
            mask_feature_size=cfg.mask_feat_channels,
            hidden_dim=cfg.mask_feat_channels,
            num_queries=cfg.num_queries,
            encoder_layers=cfg.pixel_decoder_layers,
            encoder_feedforward_dim=cfg.pixel_decoder_ffn_dim,
            decoder_layers=cfg.decoder_layers + 1,
            dim_feedforward=cfg.decoder_ffn_dim,
            num_attention_heads=cfg.decoder_heads,
            feature_strides=list(cfg.pyramid_strides),
            common_stride=cfg.pyramid_strides[0],
            pre_norm=False,
            enforce_input_projection=False,
        )

    def _needs_masks(self) -> bool:
        """Whether any mask can still change this budget's token sequence."""
        cfg = self.config
        full_fixed_count = cfg.fixed_tokens == cfg.num_visual_tokens
        full_ratio = cfg.retain_ratio == 1.0
        return not (cfg.rate == "fixed" and (full_fixed_count or full_ratio))

    def _load_head_weights(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load the translated OMG-Seg weights, refusing anything inexact.

        A partial load is the failure mode that matters: a mistranslated name
        leaves a randomly initialised submodule that still emits plausibly shaped
        masks, so the run would produce quiet nonsense instead of an error.
        """
        for prefix, module in (("pixel_decoder", self.pixel_decoder),
                               ("transformer", self.transformer)):
            subset = {
                key[len(prefix) + 1:]: value
                for key, value in state_dict.items()
                if key.startswith(f"{prefix}.")
            }
            missing, unexpected = module.load_state_dict(subset, strict=False)
            # `position_embedder`/`position_embedding` are parameter-free sine
            # encodings, so they are legitimately absent from the checkpoint.
            missing = [
                key for key in missing
                if not key.startswith(("position_embedder.", "position_embedding."))
            ]
            if missing or unexpected:
                raise RuntimeError(
                    f"OMG-Seg weight translation is incomplete for {prefix}: "
                    f"missing={sorted(missing)[:8]} unexpected={sorted(unexpected)[:8]}"
                )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> SegmentationOutput:
        """Run the frozen stack.

        Args:
            pixel_values: ``[B, 3, image_size, image_size]`` preprocessed image
                batch (ConvNeXt/CLIP normalisation, resized to ``image_size``).

        Returns:
            A :class:`SegmentationOutput` with ``f3`` and ``F_3``-resolution
            per-query probability ``masks``.
        """
        if not self._loaded:
            self.load(device=pixel_values.device, dtype=pixel_values.dtype)

        pyramid = self._extract_pyramid(pixel_values)  # (F_0, F_1, F_2, F_3)
        f3 = self.final_norm(pyramid[-1])

        if not self._needs_masks():
            return SegmentationOutput(f3=f3, masks=None, pyramid=tuple(pyramid))

        mask_logits = self._predict_masks(pyramid)  # [B, N, h, w]
        masks = mask_logits.sigmoid()
        masks = _resize_to(masks, f3.shape[-2:])  # down-sample to the F_3 grid
        return SegmentationOutput(f3=f3, masks=masks, pyramid=tuple(pyramid))

    def _extract_pyramid(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """Drive ConvNeXt's stem and stages, collecting ``F_0..F_3``."""
        hidden = self.backbone.stem(pixel_values)
        pyramid: list[torch.Tensor] = []
        for stage in self.backbone.stages:
            hidden = stage(hidden)
            pyramid.append(hidden)
        return pyramid

    def _predict_masks(self, pyramid: Sequence[torch.Tensor]) -> torch.Tensor:
        """Per-query mask logits from the last decoder layer.

        CORE keeps only the masks. OMG-Seg's open-vocabulary classifier, IoU head
        and mask-pooling projection all score *what* an object is, which the merge
        never consults: Sec. 3.1 competes masks per pixel and Sec. 3.2 sorts by
        weighted centroid, neither by class. They are dropped rather than ported.
        """
        decoder_output = self.pixel_decoder(pyramid)
        # `multi_scale_features` is already coarsest-first (strides 32, 16, 8), the
        # order both stacks index `level_embed` by. Reversing it here would pair
        # every level with the wrong embedding without changing any shape.
        transformer_output = self.transformer(
            multi_scale_features=list(decoder_output.multi_scale_features),
            mask_features=decoder_output.mask_features,
        )
        # One prediction before layer 0 plus one per layer; the last is the head's
        # output, exactly as OMG-Seg takes `mask_pred_list[-1]`.
        return transformer_output.masks_queries_logits[-1]


def load_omgseg_checkpoint(
    cfg: CoreConfig, *, local_files_only: bool = True
) -> dict[str, torch.Tensor]:
    """The released OMG-Seg head weights, as a plain ``panoptic_head.*`` mapping.

    ``omgseg_checkpoint`` names a local file when set; otherwise the public
    checkpoint is read from the Hugging Face cache like any other base model, so
    ``local_files_only`` decides whether a missing copy is fetched or is an error.
    """
    if cfg.omgseg_checkpoint is not None:
        path = Path(cfg.omgseg_checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OMG-Seg checkpoint not found: {path}")
    else:
        from huggingface_hub import hf_hub_download  # deferred

        path = Path(hf_hub_download(
            cfg.omgseg_repo,
            cfg.omgseg_filename,
            local_files_only=local_files_only,
            token=False,  # the repo is public; a stale token turns that into a 401
        ))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = payload.get("state_dict", payload)
    head = {key: value for key, value in state_dict.items() if key.startswith("panoptic_head.")}
    if not head:
        raise ValueError(
            f"{path} holds no `panoptic_head.*` tensors; it is not an OMG-Seg checkpoint."
        )
    return head


# Tensors OMG-Seg ships but CORE never reads. `cls_proj`, `iou_embed` and
# `mask_pooling_proj` score object identity and confidence, which the merge
# ignores. `pb_embedding`, `mask_tokens` and `pos_linear` belong to the box-prompt
# branch, which OMG-Seg enters only for `data_tag in ('sam', 'sam_mul')`; CORE uses
# the 300 blank object queries (Sec. 3.1), so that branch is never taken.
_OMGSEG_UNUSED_PREFIXES = (
    "cls_proj.",
    "iou_embed.",
    "mask_pooling_proj.",
    "mask_tokens.",
    "pb_embedding.",
    "pos_linear.",
)

# MMDetection -> transformers, for the modules CORE actually runs. Both stacks
# implement the same Mask2Former and their layer bodies agree operation for
# operation, so this is almost entirely renaming.
_OMGSEG_RENAMES = (
    # -- pixel decoder: MSDeformAttnPixelDecoder -> Mask2FormerPixelDecoder ----
    ("pixel_decoder.level_encoding.weight", "pixel_decoder.level_embed"),
    ("pixel_decoder.mask_feature.", "pixel_decoder.mask_projection."),
    # The FPN branch for the stride-4 level; both stacks build it with bias=False.
    ("pixel_decoder.lateral_convs.0.conv.", "pixel_decoder.adapter_1.0."),
    ("pixel_decoder.lateral_convs.0.gn.", "pixel_decoder.adapter_1.1."),
    ("pixel_decoder.output_convs.0.conv.", "pixel_decoder.layer_1.0."),
    ("pixel_decoder.output_convs.0.gn.", "pixel_decoder.layer_1.1."),
    # -- transformer decoder --------------------------------------------------
    ("query_embed.weight", "transformer.queries_embedder.weight"),
    ("query_feat.weight", "transformer.queries_features.weight"),
    ("level_embed.weight", "transformer.level_embed.weight"),
    ("transformer_decoder.post_norm.", "transformer.decoder.layernorm."),
)

# `ffn.layers.0.0` is the first Linear of MMDetection's FFN (index 0 of a
# Sequential holding Linear + activation); `ffn.layers.1` is the second Linear.
_OMGSEG_FFN_RENAMES = (("ffn.layers.0.0.", "fc1."), ("ffn.layers.1.", "fc2."))

# MMDetection numbers a layer's norms in application order. Its pixel-decoder
# encoder layer runs self-attn -> norms.0 -> ffn -> norms.1, and its transformer
# decoder layer runs cross-attn -> norms.0 -> self-attn -> norms.1 -> ffn ->
# norms.2. `transformers` applies the same operations in the same order, so the
# positions line up -- which is what makes `norms.0` the *cross*-attention norm
# on the decoder side, not the self-attention one.
_PIXEL_ENCODER_NORMS = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
_DECODER_NORMS = {
    "0": "cross_attn_layer_norm",
    "1": "self_attn_layer_norm",
    "2": "final_layer_norm",
}


def omgseg_state_dict_to_hf(head: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate OMG-Seg's ``panoptic_head.*`` weights to ``transformers`` names.

    Returns a mapping keyed by ``pixel_decoder.*`` and ``transformer.*``, matching
    the submodules :class:`SegmentationVision` builds.

    Only one tensor needs more than a rename: MMDetection stores decoder
    self-attention as a single fused ``in_proj_weight``/``in_proj_bias``, which
    ``transformers`` splits into ``q_proj``/``k_proj``/``v_proj``. Cross-attention
    stays fused in both, so it is renamed like everything else.
    """
    out: dict[str, torch.Tensor] = {}
    for raw_key, tensor in head.items():
        key = raw_key[len("panoptic_head."):]
        if key.startswith(_OMGSEG_UNUSED_PREFIXES):
            continue

        # Fused self-attention -> three projections, in torch's q, k, v order.
        fused = re.fullmatch(
            r"transformer_decoder\.layers\.(\d+)\.self_attn\.attn\.in_proj_(weight|bias)", key
        )
        if fused:
            layer, suffix = fused.groups()
            parts = tensor.chunk(3, dim=0)
            for name, chunk in zip(("q_proj", "k_proj", "v_proj"), parts, strict=True):
                out[f"transformer.decoder.layers.{layer}.self_attn.{name}.{suffix}"] = chunk.clone()
            continue

        key = re.sub(r"^pixel_decoder\.input_convs\.(\d+)\.conv\.",
                     r"pixel_decoder.input_projections.\1.0.", key)
        key = re.sub(r"^pixel_decoder\.input_convs\.(\d+)\.gn\.",
                     r"pixel_decoder.input_projections.\1.1.", key)
        key = re.sub(
            r"^pixel_decoder\.encoder\.layers\.(\d+)\.norms\.([01])\.",
            lambda m: f"pixel_decoder.encoder.layers.{m[1]}.{_PIXEL_ENCODER_NORMS[m[2]]}.",
            key,
        )
        key = re.sub(
            r"^transformer_decoder\.layers\.(\d+)\.norms\.([012])\.",
            lambda m: f"transformer.decoder.layers.{m[1]}.{_DECODER_NORMS[m[2]]}.",
            key,
        )
        # `mask_embed` is a Linear/ReLU Sequential at indices 0, 2, 4; transformers
        # wraps each Linear in its own block, so the index halves.
        key = re.sub(
            r"^mask_embed\.([024])\.",
            lambda m: f"transformer.decoder.mask_predictor.mask_embedder.{int(m[1]) // 2}.0.",
            key,
        )
        for old, new in _OMGSEG_FFN_RENAMES:
            for scope in ("pixel_decoder.encoder.layers", "transformer_decoder.layers"):
                key = re.sub(rf"^({re.escape(scope)}\.\d+)\.{re.escape(old)}", rf"\1.{new}", key)
        key = key.replace("transformer_decoder.layers.", "transformer.decoder.layers.")
        # MMDetection nests both attentions one level deeper, under `.attn.`.
        # Self-attention's fused projections are handled above; this catches the
        # rest, `out_proj` on both sides and cross-attention's `in_proj_*`.
        key = re.sub(r"\.(self_attn|cross_attn)\.attn\.", r".\g<1>.", key)
        for old, new in _OMGSEG_RENAMES:
            if key.startswith(old):
                key = new + key[len(old):]
                break

        if not key.startswith(("pixel_decoder.", "transformer.")):
            raise KeyError(f"Untranslated OMG-Seg weight: {raw_key}")
        out[key] = tensor
    return out


class StubSegmentationVision(nn.Module):
    """Weight-free stand-in producing random masks/features at the right shapes.

    Only for tests and dry-runs (no OMG-Seg download). Shapes match the real
    stack so the merge module and model plumbing can be exercised end to end on
    CPU. It is deterministic given a manual seed.
    """

    def __init__(self, config: CoreConfig) -> None:
        super().__init__()
        self.config = config

    def load(self) -> "StubSegmentationVision":
        return self

    def forward(self, pixel_values: torch.Tensor) -> SegmentationOutput:
        cfg = self.config
        batch = pixel_values.shape[0]
        grid = cfg.grid_size
        channels = cfg.vision_hidden_size
        device = pixel_values.device
        dtype = pixel_values.dtype
        f3 = torch.randn(batch, channels, grid, grid, device=device, dtype=dtype)
        masks = torch.rand(batch, cfg.num_queries, grid, grid, device=device, dtype=dtype)
        return SegmentationOutput(f3=f3, masks=masks, pyramid=(f3,))


# ============================================================================
# The CORE model: vision -> merge -> projector -> decoder
# ============================================================================


class CoreProjector(nn.Module):
    """Two-layer MLP with GELU mapping ``F_3`` (1536) -> LLM hidden (4096).

    Matches Sec. 4.1: "The projector is a two-layer MLP with a GELU activation
    function, mapping the visual features from a dimension of 1536 to the LLM's
    hidden dimension of 4096."
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_act: str = "gelu") -> None:
        super().__init__()
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}.get(hidden_act.lower())
        if act is None:
            raise ValueError(f"Unsupported projector act: {hidden_act}")
        # LLaVA's standard mlp2x projector: Linear -> act -> Linear (out->out).
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = act()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CoreModel(nn.Module):
    """CORE end-to-end model.

    Args:
        config: :class:`~llmoe.compressor.core.CoreConfig`.
        segmentation: Frozen vision+seg stack. Defaults to
            :class:`~llmoe.compressor.core.SegmentationVision`; pass a
            ``StubSegmentationVision`` for weight-free dry runs.
        llm: Optional pre-built causal LM. When ``None`` it is loaded
            lazily from ``config.llm_name_or_path`` on first use / :meth:`load`.
    """

    def __init__(
        self,
        config: CoreConfig,
        *,
        segmentation: nn.Module | None = None,
        llm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.vision = segmentation if segmentation is not None else SegmentationVision(config)
        self.merger = ObjectCentricMerger(
            merge_mode=config.merge_mode,
            rate=config.rate,
            fixed_tokens=config.fixed_tokens,
            retain_ratio=config.retain_ratio,
            mask_threshold=config.mask_threshold,
            large_object_first=config.large_object_first,
        )
        self.projector = CoreProjector(
            config.vision_hidden_size, config.llm_hidden_size, config.projector_hidden_act
        )
        self.llm = llm
        self._image_token_id: int | None = None
        # Records the last batch's per-image source matrices (for llmoe.viz).
        self._source_history: list[torch.Tensor] = []
        # One content-only evaluation budget record per source image.
        self._last_budget: tuple[CoreTokenBudget, ...] = ()

    # ------------------------------------------------------------------
    # Loading / parameter freezing
    # ------------------------------------------------------------------
    def load_llm(self, *, dtype: torch.dtype | None = None, local_files_only: bool = True) -> None:
        """Lazily load the configured language backend and attention kernel."""
        if self.llm is not None:
            return
        self.llm = load_causal_lm(
            self.config,
            dtype=dtype,
            local_files_only=local_files_only,
            attention_implementation=self.config.attention_implementation,
        )

    def freeze_base(self) -> None:
        """Freeze ConvNeXt-L, Mask2Former and the base LLM (both stages)."""
        for param in self.vision.parameters():
            param.requires_grad_(False)
        if self.llm is not None:
            for param in self.llm.parameters():
                param.requires_grad_(False)

    def apply_lora(self) -> None:
        """Attach LoRA adapters to the LLM for stage-2 tuning (Sec. 4.1)."""
        if not self.config.use_lora:
            return
        if self.llm is None:
            raise RuntimeError("Load the LLM before applying LoRA.")
        from peft import LoraConfig, get_peft_model  # deferred

        targets = tuple(self.config.lora_target_modules or ())
        module_names = {name.rsplit(".", 1)[-1] for name, _ in self.llm.named_modules()}
        missing = sorted(set(targets) - module_names)
        if missing:
            raise RuntimeError(
                f"{self.config.llm_family} is missing configured LoRA modules: {missing}"
            )

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            target_modules=list(targets),
            task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        if not any(parameter.requires_grad for parameter in self.llm.parameters()):
            raise RuntimeError("PEFT attached no trainable LoRA parameters")

    # ------------------------------------------------------------------
    # Visual encoding: images -> projected object-centric tokens
    # ------------------------------------------------------------------
    def encode_images(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """Return one ``[K_b, llm_hidden]`` token sequence per image in the batch.

        Runs the frozen vision stack, merges the ``F_3`` grid per image with its
        object masks, records the source matrices, and projects the result.
        ``K_b`` is fixed under fixed-rate and variable under adaptive.
        """
        seg = self.vision(pixel_values)  # SegmentationOutput
        batch, channels, height, width = seg.f3.shape
        f3_tokens = seg.f3.permute(0, 2, 3, 1).reshape(batch, height * width, channels)

        projected: list[torch.Tensor] = []
        self._source_history = []
        self._last_budget = ()
        budget_records: list[CoreTokenBudget] = []
        for b in range(batch):
            original_content_tokens = int(f3_tokens[b].shape[0])
            target = (
                self.merger.target_for(original_content_tokens)
                if self.config.rate == "fixed"
                else None
            )
            full_token_path = target == original_content_tokens
            if full_token_path:
                visual_tokens = f3_tokens[b]
                source = None
                budget_records.append(
                    self.merger.budget_record(
                        original_content_tokens,
                        original_content_tokens,
                        structural_tokens=0,
                    )
                )
            else:
                if seg.masks is None:
                    raise RuntimeError("Object-centric merging requires segmentation masks")
                merged = self.merger.merge(f3_tokens[b], seg.masks[b])
                visual_tokens = merged.tokens
                source = merged.source
                if self.merger.last_budget is not None:
                    budget_records.append(self.merger.last_budget)
            tokens = self.projector(visual_tokens.to(self.projector.fc1.weight.dtype))
            projected.append(tokens)
            # Source matrices are useful for LLMOE visualisation but copying a
            # [K, 1024] tensor to the CPU for every training image needlessly
            # stalls the hot path. Evaluation keeps the previous behaviour.
            if not self.training:
                if source is None:
                    self._source_history.append(torch.eye(height * width, dtype=torch.float32))
                else:
                    self._source_history.append(source.detach().cpu().float())
        self._last_budget = tuple(budget_records)
        return projected

    @property
    def last_budget(self) -> tuple[CoreTokenBudget, ...]:
        """Per-image records from the most recent visual encoding pass."""

        return self._last_budget

    # ------------------------------------------------------------------
    # Language forward / generation
    # ------------------------------------------------------------------
    def _image_token(self) -> int:
        if self._image_token_id is None:
            raise RuntimeError("Set the image token id via set_image_token_id() first.")
        return self._image_token_id

    def set_image_token_id(self, token_id: int) -> None:
        self._image_token_id = int(token_id)

    @property
    def generation_config(self):  # noqa: ANN201 - proxies the LLM's config
        """Proxy to the LLM's generation config.

        LLMOE's ``VLM._configure_generation_tokenizer`` sets
        ``model.generation_config.pad_token_id``; CoreModel is a plain module,
        so forward that onto the wrapped decoder.
        """
        if self.llm is None:
            raise RuntimeError("Load the LLM before accessing generation_config.")
        return self.llm.generation_config

    @property
    def device(self) -> torch.device:
        return next(self.projector.parameters()).device

    def _merge_input_embeds(
        self,
        input_ids: torch.Tensor,
        image_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Scatter projected visual tokens into the text embedding sequence.

        Each ``<image>`` placeholder run in ``input_ids`` is replaced by that
        image's projected object-centric tokens. Assumes the placeholder count
        per row already matches ``K_b`` (see the data collator / processor).
        """
        embed_tokens = self.llm.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)
        image_token_id = self._image_token()
        for b in range(input_ids.shape[0]):
            positions = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
            feats = image_features[b].to(inputs_embeds.dtype)
            if positions.numel() != feats.shape[0]:
                raise ValueError(
                    f"Row {b}: {positions.numel()} <image> placeholders but "
                    f"{feats.shape[0]} visual tokens. Expand placeholders to K_b "
                    "(fixed-rate) or re-tokenise per image (adaptive)."
                )
            inputs_embeds[b, positions] = feats
        return inputs_embeds

    def _prepare_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        image_features: list[torch.Tensor],
        attention_mask: torch.Tensor | None,
        labels: torch.Tensor | None,
        max_length: int | None = None,
        has_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Replace one image marker per row with its variable-length features.

        Training data keeps a single ``<image>`` token in each conversation.
        Expanding here avoids running the frozen vision stack in DataLoader
        workers merely to discover ``K_b``. For backwards compatibility, rows
        that already contain exactly ``K_b`` image placeholders are accepted.
        """

        embed_tokens = self.llm.get_input_embeddings()
        base_embeds = embed_tokens(input_ids)
        image_token_id = self._image_token()
        rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        if len(image_features) != input_ids.shape[0]:
            raise ValueError(
                f"Received {len(image_features)} image feature rows for "
                f"a text batch of {input_ids.shape[0]}"
            )

        if has_image is None:
            has_image = torch.ones(
                input_ids.shape[0], dtype=torch.bool, device=input_ids.device
            )
        if has_image.ndim != 1 or has_image.shape[0] != input_ids.shape[0]:
            raise ValueError("has_image must contain one boolean per text row")

        for b, feats in enumerate(image_features):
            if attention_mask is None:
                valid = torch.arange(input_ids.shape[1], device=input_ids.device)
            else:
                valid = attention_mask[b].nonzero(as_tuple=True)[0]
            row_ids = input_ids[b, valid]
            row_embeds = base_embeds[b, valid]
            row_labels = labels[b, valid] if labels is not None else None
            positions = (row_ids == image_token_id).nonzero(as_tuple=True)[0]
            feats = feats.to(device=row_embeds.device, dtype=row_embeds.dtype)

            if not bool(has_image[b].item()):
                if positions.numel() != 0:
                    raise ValueError(f"Row {b}: text-only row contains an <image> marker")
                # LLaVA keeps a dummy image in text-only batches. This zero-valued
                # graph tether ensures the vision/projector surface remains part
                # of DDP even when an entire micro-batch is text-only.
                row_embeds = row_embeds + feats.sum() * 0.0
                if max_length is not None:
                    row_embeds = row_embeds[:max_length]
                    if row_labels is not None:
                        row_labels = row_labels[:max_length]
                rows.append(row_embeds)
                if row_labels is not None:
                    label_rows.append(row_labels)
                continue

            if positions.numel() == feats.shape[0]:
                # Legacy/pre-expanded representation.
                row_embeds = row_embeds.clone()
                row_embeds[positions] = feats
            elif positions.numel() == 1:
                pos = int(positions.item())
                row_embeds = torch.cat(
                    (row_embeds[:pos], feats, row_embeds[pos + 1 :]),
                    dim=0,
                )
                if row_labels is not None:
                    visual_labels = torch.full(
                        (feats.shape[0],),
                        -100,
                        dtype=row_labels.dtype,
                        device=row_labels.device,
                    )
                    row_labels = torch.cat(
                        (row_labels[:pos], visual_labels, row_labels[pos + 1 :]),
                        dim=0,
                    )
            else:
                raise ValueError(
                    f"Row {b}: expected one <image> marker or {feats.shape[0]} "
                    f"expanded markers, found {positions.numel()}."
                )
            if max_length is not None:
                row_embeds = row_embeds[:max_length]
                if row_labels is not None:
                    row_labels = row_labels[:max_length]
            rows.append(row_embeds)
            if row_labels is not None:
                label_rows.append(row_labels)

        max_len = max(row.shape[0] for row in rows)
        hidden = rows[0].shape[-1]
        inputs_embeds = rows[0].new_zeros((len(rows), max_len, hidden))
        expanded_attention = input_ids.new_zeros((len(rows), max_len))
        expanded_labels: torch.Tensor | None = None
        if labels is not None:
            expanded_labels = labels.new_full((len(rows), max_len), -100)

        for b, row in enumerate(rows):
            length = row.shape[0]
            inputs_embeds[b, :length] = row
            expanded_attention[b, :length] = 1
            if expanded_labels is not None:
                expanded_labels[b, :length] = label_rows[b]
        return inputs_embeds, expanded_attention, expanded_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        multimodal_max_length: int | None = None,
        has_image: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Training/eval forward: auto-regressive loss ``L_text`` (Sec. 3.3)."""
        image_features = self.encode_images(pixel_values)
        inputs_embeds, attention_mask, labels = self._prepare_multimodal_inputs(
            input_ids,
            image_features,
            attention_mask,
            labels,
            multimodal_max_length,
            has_image,
        )
        output = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        if output.loss is not None and not bool(torch.isfinite(output.loss.detach()).item()):
            def _stats(tensor: torch.Tensor) -> str:
                values = tensor.detach().float()
                finite = torch.isfinite(values)
                if not bool(finite.any().item()):
                    return f"shape={tuple(values.shape)} finite=0/{values.numel()}"
                selected = values[finite]
                return (
                    f"shape={tuple(values.shape)} finite={int(finite.sum())}/{values.numel()} "
                    f"min={float(selected.min()):.5g} max={float(selected.max()):.5g}"
                )

            feature_stats = "; ".join(_stats(features) for features in image_features)
            supervised = 0 if labels is None else int((labels != -100).sum().item())
            print(
                "[CORE non-finite] "
                f"loss={float(output.loss.detach())} supervised={supervised} "
                f"pixels=({_stats(pixel_values)}) visual=({feature_stats}) "
                f"embeds=({_stats(inputs_embeds)}) logits=({_stats(output.logits)})",
                flush=True,
            )
        return output

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Deterministic generation (Sec. 3.3 inference)."""
        image_features = self.encode_images(pixel_values)
        inputs_embeds = self._merge_input_embeds(input_ids, image_features)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def get_source_history(self) -> list[torch.Tensor]:
        """Per-image ``[K, HW]`` source matrices from the last encode."""
        return list(self._source_history)

    def trainable_parameters(self, *, include_lora: bool) -> list[nn.Parameter]:
        """Collect the parameters trained in a given stage.

        Stage 1: projector only. Stage 2: projector + LoRA adapters.
        """
        params = [p for p in self.projector.parameters()]
        if include_lora and self.llm is not None:
            params += [p for p in self.llm.parameters() if p.requires_grad]
        return params


# ============================================================================
# Trained-checkpoint provenance guards
# ============================================================================


def _normalise_model_path(value: Any) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.exists():
        return str(path.resolve())
    return str(value)


def asset_identity(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_dir():
        return None
    manifest_path = path / "core_asset_manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        "model": manifest.get("model"),
        "revision": manifest.get("revision"),
        "files": {
            name: {
                key: entry.get(key)
                for key in ("etag", "sha256", "size")
                if entry.get(key) is not None
            }
            for name, entry in sorted(manifest.get("files", {}).items())
        },
    }


def checkpoint_signature(config: Any, *, include_lora: bool) -> dict[str, Any]:
    signature = {
        "implementation": (
            "core-convnext-omgseg-llama3-8b-v3"
            if config.llm_family == "llama3"
            else "core-convnext-omgseg-v1"
        ),
        "llm_family": config.llm_family,
        "llm_name_or_path": _normalise_model_path(config.llm_name_or_path),
        "llm_revision": getattr(config, "llm_revision", None),
        "llm_asset_identity": asset_identity(config.llm_name_or_path),
        "llm_hidden_size": int(config.llm_hidden_size),
        "vision_hidden_size": int(config.vision_hidden_size),
        "projector_hidden_act": config.projector_hidden_act,
        "evaluation_budget_convention": (
            "per_sample_round_ratio_from_premerge_content_tokens"
        ),
        "evaluation_structural_tokens": 0,
        "vision": {
            "architecture": "OpenCLIP ConvNeXt-L + OMG-Seg Mask2Former",
            "image_size": int(config.image_size),
            "feature_stride": int(config.feature_stride),
            "native_tokens": int(config.num_visual_tokens),
            "vision_timm_model": config.vision_timm_model,
            "omgseg_repo": config.omgseg_repo,
            "omgseg_filename": config.omgseg_filename,
        },
    }
    if include_lora:
        signature["lora"] = {
            "r": int(config.lora_r),
            "alpha": int(config.lora_alpha),
            "dropout": float(config.lora_dropout),
            "bias": config.lora_bias,
            "target_modules": list(config.lora_target_modules or ()),
        }
    return signature


def _saved_signature(payload: dict[str, Any], *, include_lora: bool) -> dict[str, Any]:
    model_path = payload.get("llm_name_or_path", DEFAULT_MODEL)
    saved_family = payload.get("_llm_family", "llama")
    signature = {
        "implementation": payload.get(
            "_implementation",
            "core-convnext-omgseg-llama3-8b-v3"
            if saved_family == "llama3"
            else "core-convnext-omgseg-v1",
        ),
        "llm_family": saved_family,
        "llm_name_or_path": _normalise_model_path(model_path),
        "llm_revision": payload.get("llm_revision"),
        "llm_asset_identity": payload.get(
            "_llm_asset_identity", asset_identity(model_path)
        ),
        "llm_hidden_size": int(payload.get("llm_hidden_size", 4096)),
        "vision_hidden_size": int(payload.get("vision_hidden_size", 1536)),
        "projector_hidden_act": payload.get("projector_hidden_act", "gelu"),
        "evaluation_budget_convention": payload.get(
            "_evaluation_budget_convention",
            "per_sample_round_ratio_from_premerge_content_tokens",
        ),
        "evaluation_structural_tokens": int(
            payload.get("_evaluation_structural_tokens", 0)
        ),
        "vision": {
            "architecture": "OpenCLIP ConvNeXt-L + OMG-Seg Mask2Former",
            "image_size": int(payload.get("image_size", 1024)),
            "feature_stride": int(payload.get("feature_stride", 32)),
            "native_tokens": (
                int(payload.get("image_size", 1024))
                // int(payload.get("feature_stride", 32))
            )
            ** 2,
            "vision_timm_model": payload.get(
                "vision_timm_model", DEFAULT_VISION_TIMM_MODEL
            ),
            "omgseg_repo": payload.get("omgseg_repo", DEFAULT_OMGSEG_REPO),
            "omgseg_filename": payload.get(
                "omgseg_filename", DEFAULT_OMGSEG_FILENAME
            ),
        },
    }
    if include_lora:
        signature["lora"] = {
            "r": int(payload.get("lora_r", 512)),
            "alpha": int(payload.get("lora_alpha", 256)),
            "dropout": float(payload.get("lora_dropout", 0.05)),
            "bias": payload.get("lora_bias", "none"),
            "target_modules": list(
                payload.get("lora_target_modules") or LORA_TARGET_MODULES
            ),
        }
    return signature


def validate_checkpoint_compatibility(
    config: Any,
    checkpoint_dir: str | os.PathLike[str],
    *,
    include_lora: bool,
    purpose: str,
) -> None:
    """Reject shape-compatible checkpoints trained for another decoder."""

    directory = Path(checkpoint_dir).expanduser().resolve()
    config_path = directory / "core_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{purpose} is missing CORE provenance: {config_path}"
        )
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    actual = _saved_signature(payload, include_lora=include_lora)
    expected = checkpoint_signature(config, include_lora=include_lora)
    # `llm_name_or_path` is an absolute path on whichever machine trained the
    # checkpoint, so a released checkpoint can never match a fresh clone on it --
    # `_normalise_model_path` returns the recorded string verbatim once that path
    # stops existing. When both sides carry an asset identity (per-file sizes and
    # content hashes of the whole LLM) that is the substantive check and the path
    # adds nothing, so compare on identity alone. Without an identity on either
    # side the path is still the only signal there is, and stays in.
    if actual.get("llm_asset_identity") and expected.get("llm_asset_identity"):
        actual.pop("llm_name_or_path", None)
        expected.pop("llm_name_or_path", None)
    if actual != expected:
        differences = {
            key: (actual.get(key), expected.get(key))
            for key in sorted(set(actual) | set(expected))
            if actual.get(key) != expected.get(key)
        }
        raise ValueError(f"{purpose} is incompatible with this CORE model: {differences}")


# ============================================================================
# LLMOE backbone
# ============================================================================


class CorePromptPreprocessor(PromptPreprocessor):
    """Wrap prompts in the selected decoder's exact conversation template."""

    def __init__(self, image_token: str = "<image>") -> None:
        self._image_token = image_token

    def forward(self, vlm: VLM, prompts: Sequence[str]) -> list[str]:
        family = vlm.core_config.llm_family
        # Vicuna/Qwen formatting is string-only and must retain the old lazy
        # behavior. Llama-3 alone needs the pinned tokenizer's chat template.
        tokenizer = (
            getattr(vlm.processor, "tokenizer", None)
            if family == "llama3"
            else None
        )
        return [format_inference_prompt(
                    self._image_token, prompt, family, tokenizer=tokenizer)
                for prompt in prompts]


class CoreVLM(VLM):
    """The complete CORE vision model with a selectable causal decoder.

    Args:
        config: :class:`~llmoe.compressor.core.CoreConfig`, or a mapping of
            its fields (``backbone_args`` from a JSON/YAML custom-args file
            arrives as a mapping). Defaults are the paper's.
        adapter_dir: Directory with trained projector + LoRA weights (from the
            stage-2 finetune). A relative path is read against the
            ``llmoe_weights`` bundle. Left unset, the released adapters are used
            with a warning: CORE's projector is trained, so a randomly
            initialised one is a structure-only smoke run rather than a model.
        pretrained: ``False`` asks for that random initialisation explicitly;
            ``True`` (the default in effect) takes the released adapters,
            downloading them on demand. ``adapter_dir`` overrides both.
        segmentation: Optional pre-built vision stack (e.g. a stub for tests).
        core_args: Any remaining :class:`CoreConfig` fields, so
            ``backbone_args`` can name them at the top level instead of nesting
            them under ``config``.
    """

    weights_key = "core"
    short_answer_suffix = "\nAnswer the question using a single word or phrase."
    option_letter_suffix = "\nAnswer with the option's letter from the given choices directly."
    direct_answer_suffix = "\nAnswer the question directly."

    def __init__(
        self,
        *,
        config: CoreConfig | Mapping[str, Any] | None = None,
        adapter_dir: str | None = None,
        pretrained: bool | str | os.PathLike[str] | None = None,
        image_preprocessor: ImagePreprocessor | None = None,
        prompt_preprocessor: PromptPreprocessor | None = None,
        segmentation: Any | None = None,
        local_files_only: bool = True,
        device: Device = auto_device(),
        dtype: torch.dtype | None = None,
        **core_args: Any,
    ) -> None:
        self._core_config = _resolve_core_config(config, core_args)
        if dtype is None:
            device_type = torch.device(str(device)).type
            dtype = (
                getattr(torch, self._core_config.inference_dtype)
                if device_type == "cuda"
                else torch.float32
            )
        super().__init__(
            self._core_config.llm_name_or_path,
            image_preprocessor or IdentityImagePreprocessor(),
            prompt_preprocessor or CorePromptPreprocessor(self._core_config.image_token),
            local_files_only=local_files_only,
            device=device,
            dtype=dtype,
        )
        # Keyed on the full grid, not the compressor's budget: one stage-2
        # finetune serves every inference budget, and the release is named for it.
        # Artifacts are decoder-specific.  Passing the backbone key prevents a
        # Llama-3 CORE run from ever resolving the released Vicuna projector.
        resolved = resolve_weights(
            "CORE",
            self._core_config.num_visual_tokens,
            checkpoint_path=adapter_dir,
            pretrained=pretrained,
            backbone=type(self).weights_key,
        )
        self._adapter_dir = None if resolved is None else str(resolved)
        self._segmentation = segmentation
        self._tokenizer: Any | None = None
        self._image_processor: Any | None = None
        self._track_source: bool = False

    @property
    def core_config(self) -> CoreConfig:
        """The live CORE hyper-parameters.

        ``CORECompressor.install()`` mutates this before the model is built:
        ``SegmentationVision.load()`` skips building the Mask2Former head entirely
        on the uncompressed fixed-1024 path, so the merge strategy has to be
        settled before the first ``model`` access.
        """
        return self._core_config

    @property
    def loaded(self) -> bool:
        """Whether the weights have been materialised yet."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self) -> tuple[Any, Any]:
        cfg = self._core_config
        if (
            cfg.attention_implementation == "sdpa"
            and torch.device(str(self._device)).type == "cuda"
        ):
            # Force PyTorch SDPA to use Hopper's FlashAttention kernel during
            # real model execution. Disabling every fallback makes an
            # unsupported shape fail rather than silently selecting math,
            # memory-efficient, or cuDNN attention -- but that is only a useful
            # assertion where the flash kernel covers this model. It does on
            # Hopper (sm_90), which is what the released numbers were produced
            # on; elsewhere every attention call raises "Invalid backend"
            # instead, so the fallbacks stay enabled and PyTorch picks a kernel.
            torch.backends.cuda.enable_flash_sdp(True)
            flash_only = torch.cuda.get_device_capability(self._device)[0] == 9
            torch.backends.cuda.enable_math_sdp(not flash_only)
            torch.backends.cuda.enable_mem_efficient_sdp(not flash_only)
            torch.backends.cuda.enable_cudnn_sdp(not flash_only)
            if not flash_only:
                print(
                    "[CORE] sm_%d%d is not Hopper: leaving the SDPA fallback kernels enabled "
                    "so attention can run." % torch.cuda.get_device_capability(self._device),
                    flush=True,
                )
        tokenizer = load_tokenizer(cfg, local_files_only=self._local_files_only)
        if cfg.image_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [cfg.image_token]})
        self._tokenizer = tokenizer

        # Image processor: OpenCLIP ConvNeXt-L normalisation at image_size.
        self._image_processor = build_image_processor(cfg, self._local_files_only)

        segmentation = self._segmentation or SegmentationVision(cfg)
        if isinstance(segmentation, SegmentationVision):
            segmentation.load(
                device=self._device,
                dtype=self._model_vision_dtype(),
                local_files_only=self._local_files_only,
            )
        elif hasattr(segmentation, "load"):
            segmentation.load()
        model = CoreModel(cfg, segmentation=segmentation)
        model.load_llm(dtype=self._dtype, local_files_only=self._local_files_only)
        if len(tokenizer) > model.llm.get_input_embeddings().weight.shape[0]:
            model.llm.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        model.set_image_token_id(tokenizer.convert_tokens_to_ids(cfg.image_token))
        model.freeze_base()
        if self._adapter_dir is not None:
            _load_core_adapters(model, self._adapter_dir)
        model.to(self._device)
        model.projector.to(dtype=self._dtype)
        model.eval()

        processor = _CoreProcessor(tokenizer, self._image_processor)
        return processor, model

    def _configure_generation_tokenizer(self) -> None:
        """Apply LLMOE's generation-token setup to CORE's nested decoder."""

        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is None:
            return
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer needs a pad token or eos token for batched generation.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        if self._model is not None and self._model.llm is not None:
            self._model.llm.generation_config.pad_token_id = tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Full CORE inference (overrides the base VLM.forward pipeline)
    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor | Sequence[torch.Tensor],
        prompts: Sequence[str],
        *,
        max_new_tokens: int = 256,
        generation_profiler: Any | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if isinstance(images, torch.Tensor) and images.ndim != 4:
            raise ValueError(f"Expected a 4D image batch, got {images.ndim} dimensions")
        if len(images) != len(prompts):
            raise ValueError(
                f"Unmatched number of images and prompts: {len(images)} != {len(prompts)}"
            )
        self._lazy_load_model()
        model = self._model
        formatted = self.format_prompts(prompts)

        pixel_values = self._prepare_pixel_values(images)
        # One vision + merge pass; learn each image's token count K_b.
        image_features = self._encode_images(pixel_values)
        counts = [feat.shape[0] for feat in image_features]

        input_ids, attention_mask = self._tokenize_with_image_runs(formatted, counts)
        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)

        inputs_embeds = model._merge_input_embeds(input_ids, image_features)
        gen_kwargs = {"do_sample": False, **kwargs}
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id
        gen_kwargs.setdefault("pad_token_id", pad_id)
        # Each supported tokenizer exposes its trained end-of-turn token as EOS
        # (Vicuna </s>, Qwen <|im_end|>, or Llama-3 <|eot_id|>).
        gen_kwargs.setdefault("eos_token_id", self._tokenizer.eos_token_id)
        # The profiler instruments the module whose ``forward`` generation calls,
        # which here is the decoder itself rather than the CoreModel wrapper.
        measure_generate = getattr(generation_profiler, "measure_generate", None)
        generate_context = (
            measure_generate(model.llm) if callable(measure_generate) else nullcontext()
        )
        with torch.inference_mode():
            with generate_context:
                output_ids = model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs,
                )
        decoded = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [text.rstrip() for text in decoded]

    def prepare_inputs(
        self,
        images: torch.Tensor | Sequence[torch.Tensor],
        prompts: Sequence[str],
        *,
        pixel_dtype: torch.dtype | None = None,
    ) -> Any:
        """``input_ids``/``attention_mask``/``pixel_values`` for :class:`CoreModel`.

        The base implementation calls ``processor(images=..., text=...)``, which
        CORE has no equivalent of: the number of ``<image>`` placeholders depends
        on how many tokens the merge will emit, so the count has to be settled
        before tokenising. A legacy absolute ``fixed_tokens`` target is known
        without looking at the image. Ratio mode and the adaptive ablation run the
        vision stack here so ``N_original`` comes from the actual assembled F3
        content sequence rather than nominal geometry or text padding.

        ``generate`` is not driven from this: :meth:`forward` needs the encoded
        features themselves, so it runs one vision pass and shares the helpers
        below rather than encoding twice.
        """
        from transformers import BatchFeature

        if len(images) != len(prompts):
            raise ValueError(
                f"Unmatched number of images and prompts: {len(images)} != {len(prompts)}"
            )
        self._lazy_load_model()
        pixel_values = self._prepare_pixel_values(images)
        counts = self._visual_token_counts(pixel_values)
        input_ids, attention_mask = self._tokenize_with_image_runs(
            self.format_prompts(prompts), counts
        )
        if pixel_dtype is not None:
            pixel_values = pixel_values.to(pixel_dtype)
        return BatchFeature(
            data={
                "input_ids": input_ids.to(self._device),
                "attention_mask": attention_mask.to(self._device),
                "pixel_values": pixel_values,
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _prepare_pixel_values(
        self, images: torch.Tensor | Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """Images -> the normalised batch the ConvNeXt sees.

        Routed through ``format_images`` so an ``image_preprocessor`` named under
        ``backbone_args`` is honoured, as it is on every other backbone. The
        default is the identity, so this changes nothing unless one is supplied.
        """
        formatted = self.format_images(images)
        return self._image_processor(formatted).to(
            self._device, dtype=self._model_vision_dtype()
        )

    def _encode_images(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """Run the vision stack and the merge, returning per-image tokens."""
        device_type = torch.device(str(self._device)).type
        vision_dtype = self._model_vision_dtype()
        # OMG-Seg creates some positional/query tensors in FP32 even when its
        # evaluation weights are FP16. Autocast is the paper-compatible FP16
        # inference path and keeps those mixed inputs valid.
        with torch.autocast(
            device_type=device_type,
            dtype=vision_dtype,
            enabled=device_type == "cuda" and vision_dtype != torch.float32,
        ):
            return self._model.encode_images(pixel_values)

    def _visual_token_counts(self, pixel_values: torch.Tensor) -> list[int]:
        """How many ``<image>`` placeholders each row needs.

        Only the legacy absolute fixed-rate mode can answer without encoding.
        Ratio mode deliberately captures each actual pre-merge visual length.
        """
        cfg = self._core_config
        if cfg.rate == "fixed" and cfg.retain_ratio is None:
            return [int(cfg.fixed_tokens)] * int(pixel_values.shape[0])
        return [feat.shape[0] for feat in self._encode_images(pixel_values)]

    def _model_vision_dtype(self) -> torch.dtype:
        device_type = torch.device(str(self._device)).type
        if device_type == "cuda":
            return getattr(torch, self._core_config.inference_vision_dtype)
        # Half-precision CPU kernels are not generally supported.
        return torch.float32

    def _tokenize_with_image_runs(
        self, prompts: Sequence[str], counts: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenise, expanding the single ``<image>`` token into ``K_b`` copies."""
        tokenizer = self._tokenizer
        if len(prompts) != len(counts):
            raise ValueError(f"Prompt/count batch mismatch: {len(prompts)} != {len(counts)}")
        rows: list[list[int]] = []
        family = self._core_config.llm_family
        for prompt, count in zip(prompts, counts, strict=True):
            if family == "llama3":
                from llmoe.compressor.core.core_llama3_lm import (
                    tokenize_llama3_image_prompt,
                )

                row = tokenize_llama3_image_prompt(
                    tokenizer,
                    prompt,
                    self._core_config.image_token,
                    repetitions=int(count),
                )
            else:
                row = tokenize_image_prompt(
                    tokenizer,
                    prompt,
                    self._core_config.image_token,
                    repetitions=int(count),
                )
            rows.append(row)
        max_len = max(len(r) for r in rows)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for b, row in enumerate(rows):  # left-pad for generation
            input_ids[b, max_len - len(row):] = torch.tensor(row, dtype=torch.long)
            attention_mask[b, max_len - len(row):] = 1
        return input_ids, attention_mask


class CoreLlama3VLM(CoreVLM):
    """CORE's unchanged vision stack with Llama-3-8B-Instruct as decoder.

    The assigned Hugging Face checkpoint is used only as a pinned container for
    its Llama-3 language weights and tokenizer.  ``CoreModel`` still constructs
    the paper's OpenCLIP ConvNeXt-L, OMG-Seg Mask2Former head, object-centric
    merger, and 1536-to-4096 CORE projector.  In particular, this class never
    instantiates ``LlavaNextForConditionalGeneration`` and never consumes its
    CLIP-ViT, AnyRes crops, multimodal projector, or image-newline parameter.

    The stage-2 checkpoint for this port is released as
    ``methods/core/llama3-llava-next-8b/budget_1024`` and is registered under the
    :attr:`weights_key` below, so ``pretrained: true`` resolves it.  Provenance
    validation rejects Vicuna, Qwen, and native LLaVA-NeXT projectors even when
    their tensor shapes happen to agree.
    """

    weights_key = "core-llama3-8b"

    def __init__(
        self,
        *,
        config: CoreConfig | Mapping[str, Any] | None = None,
        pretrained: bool | str | os.PathLike[str] | None = False,
        **core_args: Any,
    ) -> None:
        from llmoe.compressor.core.core_llama3_lm import (
            DEFAULT_CORE_LLAMA3_MODEL,
            DEFAULT_CORE_LLAMA3_REVISION,
            LLAMA3_LLM_HIDDEN_SIZE,
            LLAMA3_LORA_TARGETS,
        )

        defaults: dict[str, Any] = {
            "llm_name_or_path": DEFAULT_CORE_LLAMA3_MODEL,
            "llm_revision": DEFAULT_CORE_LLAMA3_REVISION,
            "llm_hidden_size": LLAMA3_LLM_HIDDEN_SIZE,
            "lora_target_modules": LLAMA3_LORA_TARGETS,
        }
        if isinstance(config, CoreConfig):
            resolved_config: CoreConfig | Mapping[str, Any] = config
        else:
            values = dict(defaults)
            values.update(dict(config or {}))
            resolved_config = values
        super().__init__(
            config=resolved_config,
            pretrained=pretrained,
            **core_args,
        )
        if self.core_config.llm_family != "llama3":
            raise ValueError(
                "core-llama3-8b requires the pinned Llama-3-8B-Instruct decoder; "
                f"resolved family {self.core_config.llm_family!r}"
            )


class _CoreProcessor:
    """Minimal processor exposing ``tokenizer`` (LLMOE reads pad/eos from it)."""

    def __init__(self, tokenizer: Any, image_processor: Any) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor


class _CoreImageProcessor:
    """The published OMG-LLaVA ConvNeXt-L preprocessing, as a callable.

    ``image_mean``/``image_std`` are exposed under the names Hugging Face image
    processors use, because ``llmoe.viz`` de-normalises ``pixel_values`` through
    ``processor.image_processor.image_mean`` to recover a displayable canvas. A
    bare function would have satisfied CORE's own forward and broken only there.
    """

    def __init__(self, cfg: CoreConfig, process: Any) -> None:
        self.image_mean: tuple[float, float, float] = cfg.vision_mean
        self.image_std: tuple[float, float, float] = cfg.vision_std
        self.size: dict[str, int] = {"height": cfg.image_size, "width": cfg.image_size}
        self._process = process

    def __call__(self, images: Any) -> torch.Tensor:
        return self._process(images)


def build_image_processor(cfg: CoreConfig, local_files_only: bool) -> Any:
    """Apply the published OMG-LLaVA ConvNeXt-L image preprocessing."""
    import torchvision.transforms as T  # deferred

    normalize = T.Normalize(
        mean=cfg.vision_mean,
        std=cfg.vision_std,
    )

    def process(images: Any) -> torch.Tensor:
        from PIL import Image

        batch = []
        for img in images:
            if isinstance(img, Image.Image):
                image = img.convert("RGB")
            elif isinstance(img, torch.Tensor):
                tensor = img.float()
                if tensor.max() > 1.5:
                    tensor = tensor / 255.0
                image = T.functional.to_pil_image(tensor.clamp(0, 1)).convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            if cfg.pad_image_to_square and image.width != image.height:
                side = max(image.width, image.height)
                fill = tuple(int(channel * 255) for channel in cfg.vision_mean)
                square = Image.new("RGB", (side, side), fill)
                square.paste(
                    image,
                    ((side - image.width) // 2, (side - image.height) // 2),
                )
                image = square
            image = T.functional.resize(
                image,
                [cfg.image_size, cfg.image_size],
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            )
            tensor = T.functional.to_tensor(image)
            batch.append(normalize(tensor))
        return torch.stack(batch, dim=0)

    return _CoreImageProcessor(cfg, process)


def _load_core_adapters(model: CoreModel, adapter_dir: str) -> None:
    """Load trained projector (+ LoRA) weights from a stage-2 output dir."""
    validate_checkpoint_compatibility(
        model.config,
        adapter_dir,
        include_lora=os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")),
        purpose="CORE evaluation checkpoint",
    )
    projector_path = os.path.join(adapter_dir, "projector.pt")
    if not os.path.isfile(projector_path):
        raise FileNotFoundError(f"CORE projector checkpoint not found: {projector_path}")
    state = torch.load(projector_path, map_location="cpu", weights_only=True)
    model.projector.load_state_dict(state)

    adapter_config = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.isfile(adapter_config):
        from peft import PeftModel  # deferred

        # Load the trained adapter directly onto the frozen base model. Creating
        # a fresh default adapter first would collide with PEFT's default name.
        model.llm = PeftModel.from_pretrained(
            model.llm,
            adapter_dir,
            adapter_name="default",
            is_trainable=False,
        )
    elif model.config.use_lora:
        raise FileNotFoundError(
            f"CORE LoRA config not found: {adapter_config}. "
            "Use a complete stage-2 output directory, or set use_lora=false "
            "for a projector-only checkpoint."
        )


def _resolve_core_config(
    config: CoreConfig | Mapping[str, Any] | None, core_args: Mapping[str, Any]
) -> CoreConfig:
    """Build the evaluation config from ``backbone_args``.

    A ready-made :class:`CoreConfig` is used as-is (only validated). Otherwise a
    ``config`` mapping supplies a base and top-level fields override it, with the
    paper defaults underneath. ``custom_args`` is the only channel: CORE used to
    also read ``CORE_CONFIG_JSON`` and friends from the environment, which meant a
    run's real settings were not the ones its ``config.yaml`` recorded.
    """

    if isinstance(config, CoreConfig):
        if core_args:
            raise TypeError(
                "Pass either a CoreConfig or individual CoreConfig fields, not both: "
                f"{sorted(core_args)}"
            )
        config.validate()
        return config

    values: dict[str, Any] = {}
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError(
                f"backbone_args.config must be a CoreConfig or a mapping, got {type(config)}"
            )
        values.update(config)
    values.update(core_args)
    valid = {field.name for field in fields(CoreConfig)}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f"Unknown CoreConfig fields in backbone_args: {unknown}")
    resolved = CoreConfig(**values)
    resolved.validate()
    return resolved


# ============================================================================
# LLMOE compressor
# ============================================================================


class CORECompressor(TokenCompressor):
    """Object-centric token merging as an LLMOE compressor (Sec. 3.2 / 4.2).

    The Llama-3 target uses the fixed-rate CORE algorithm with an explicit
    ``retain_ratio``. Each image resolves its target from the true pre-merge F3
    content length; the original Vicuna/Qwen integrations may continue to use
    their legacy absolute ``max_tokens`` sweeps.

    :param max_tokens: Legacy absolute target for fixed-geometry CORE backbones.
        Mutually exclusive with ``retain_ratio``.
    :param retain_ratio: Requested fraction of each image's original content
        tokens. The Llama-3 target requires this form and resolves
        ``K=max(1, round(retain_ratio * N_original))`` per image.
    :param merge_mode: ``"soft"`` or ``"hard"`` (paper default: hard). Note this
        is orthogonal to ``rate``: it decides how pixels are assigned to objects,
        not how many tokens come out. The fixed rate requires ``"hard"``, because
        overlapping soft masks cannot give a clean count.
    :param rate: ``"fixed"`` (the default; exactly the resolved per-image target)
        or ``"adaptive"`` (one token per detected object, so the count changes
        with every image). Adaptive is an ablation with no requested budget.
    :param mask_threshold: Soft-weight floor (soft mode only).
    :param large_object_first: Fixed-rate ordering (Sec. 4.2 vs. Supp. A ablation).
    :param adapter_dir: Trained stage-2 output directory. Applied to the backbone
        at :meth:`install` when it does not already carry one, so a benchmark can
        name the trained weights next to the budget that uses them. A relative
        path is read against the ``llmoe_weights`` bundle.
    :param pretrained: ``True`` takes the released stage-2 adapters from that
        bundle, downloading them on demand; ``False`` leaves the backbone's own
        choice alone. ``adapter_dir`` overrides both.
    :param track_source: Keep per-image source matrices for ``llmoe.viz``.
    :param backbone: Which registered CORE release resolves an implicit pretrained
        artifact. It defaults to the paper's Vicuna-7B release; decoder-specific
        backbones select their own key.
    """

    def __init__(self, *, max_tokens: int | None = None,
                 retain_ratio: float | None = None,
                 merge_mode: str = "hard", rate: str = "fixed", mask_threshold: float = 0.0,
                 large_object_first: bool = True, adapter_dir: str | None = None,
                 checkpoint_path: str | None = None,
                 pretrained: bool | str | os.PathLike[str] | None = None,
                 track_source: bool = False, backbone: str = "core") -> None:
        if adapter_dir is not None and checkpoint_path is not None:
            if str(adapter_dir) != str(checkpoint_path):
                raise ValueError("adapter_dir and checkpoint_path name different CORE checkpoints")
        adapter_dir = checkpoint_path or adapter_dir
        if max_tokens is not None and retain_ratio is not None:
            raise ValueError("max_tokens and retain_ratio are mutually exclusive")
        if rate == "fixed" and max_tokens is None and retain_ratio is None:
            raise ValueError(
                "CORE's fixed rate needs a token budget: pass compressor_args."
                "retain_ratio for core-llama3-8b, or legacy max_tokens for the "
                "fixed-geometry Vicuna/Qwen backbones."
            )
        if max_tokens is not None and int(max_tokens) < 1:
            raise ValueError("max_tokens must be >= 1")
        if retain_ratio is not None:
            retain_ratio = validate_retention_ratio(retain_ratio)
        if rate == "adaptive" and (max_tokens is not None or retain_ratio is not None):
            raise ValueError(
                "CORE's adaptive rate emits one token per detected object and "
                "cannot honour max_tokens or retain_ratio."
            )
        # Adaptive has no count to declare, so `_max_tokens` stays 0 there and
        # `budget` reports the rate instead of a number CORE never produces.
        super().__init__(int(max_tokens) if max_tokens else 0)
        self._rate: str = rate
        self._retain_ratio: float | None = retain_ratio
        self._merger: ObjectCentricMerger = ObjectCentricMerger(
            merge_mode=merge_mode,
            rate=rate,
            fixed_tokens=max_tokens,
            retain_ratio=retain_ratio,
            mask_threshold=mask_threshold,
            large_object_first=large_object_first,
        )
        # `pretrained` alone is honoured here; the unset case is left to
        # `CoreVLM`, which is what actually loads the adapters.
        resolved = resolve_weights(
            "CORE",
            CoreConfig().num_visual_tokens,
            checkpoint_path=adapter_dir,
            pretrained=False if pretrained is None else pretrained,
            backbone=backbone,
        )
        self._adapter_dir: str | None = None if resolved is None else str(resolved)
        self._track_source: bool = bool(track_source)
        self._installed_vlm: VLM | None = None
        self._last_budget: tuple[CoreTokenBudget, ...] = ()

    @property
    @override
    def budget(self) -> int | float | str:
        """The run label: ratio, legacy absolute count, or ``"adaptive"``.

        Llama-3 reports the requested ratio because the realized count is
        per-image. Legacy fixed-geometry integrations retain their absolute
        count labels.

        The adaptive ablation emits one token per detected object, a count that
        changes with every image, so there is no number to report. Filing it under
        ``CORE-b0`` (what ``max_tokens`` would give) would both name a count CORE
        never produces and collapse every adaptive run into one directory.
        """
        if self._rate == "adaptive":
            return "adaptive"
        if self._retain_ratio is not None:
            return self._retain_ratio
        return self._max_tokens

    @property
    def last_budget(self) -> tuple[CoreTokenBudget, ...]:
        """Per-image content-token accounting from the latest evaluation batch."""

        model = getattr(self._installed_vlm, "_model", None)
        records = getattr(model, "last_budget", None)
        if records:
            return tuple(records)
        return self._last_budget

    @override
    def get_benchmark_record_metadata(self, batch_size: int) -> list[dict[str, Any]]:
        records = self.last_budget
        if not records:
            return super().get_benchmark_record_metadata(batch_size)
        if len(records) != int(batch_size):
            raise RuntimeError(
                f"CORE recorded {len(records)} budgets for an evaluation batch of {batch_size}."
            )
        return [{"compression_budget": asdict(record)} for record in records]

    @property
    def merger(self) -> ObjectCentricMerger:
        """The configured merge strategy, shared with the installed CoreModel."""
        return self._merger

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, masks: torch.Tensor | None = None,
                 **kwargs: Any) -> CompressedFeatures:
        """Merge each image's ``F_3`` grid under its object masks.

        :param features: ``[B, HW, C]`` or ``[HW, C]`` flattened ``F_3`` tokens in
            raster order.
        :param attention: unused -- CORE merges under mask guidance, not attention.
        :param masks: ``[B, N, H, W]`` or ``[N, H, W]`` per-query probabilities at
            the ``F_3`` resolution. Required: CORE cannot merge from features alone.
        :return: ``[B, K, C]`` or ``[K, C]``, same rank, dtype and device. ``K`` is
            the fixed-rate target resolved from each input's original content
            count, or the adaptive object count.
        """
        del attention, layer_idx, kwargs
        if masks is None:
            raise ValueError(
                "CORECompressor.compress requires object masks; CORE merges the "
                "F_3 grid under mask guidance, not from features alone."
            )
        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Expected [B, HW, C] or [HW, C], got {tuple(features.shape)}")
        if features.shape[1] == 0:
            raise ValueError("Cannot compress an empty sequence")
        batch = features.shape[0]
        if masks.ndim == 3:  # one mask set shared by the batch
            masks = masks.unsqueeze(0).expand(batch, *masks.shape)
        if masks.ndim != 4 or masks.shape[0] != batch:
            raise ValueError(f"Expected masks [B={batch}, N, H, W], got {tuple(masks.shape)}")

        merged: list[MergedTokens] = []
        budgets: list[CoreTokenBudget] = []
        for b in range(batch):
            result = self._merger.merge(features[b], masks[b])
            merged.append(result)
            if self._merger.last_budget is not None:
                budgets.append(self._merger.last_budget)
        self._last_budget = tuple(budgets)
        counts = {result.tokens.shape[0] for result in merged}
        if len(counts) > 1:
            # The adaptive count is data-dependent, so a batch can disagree on K.
            # CoreVLM handles that by expanding placeholders per image; a single
            # batched tensor cannot represent it.
            raise ValueError(
                f"Adaptive merging produced different token counts across the batch: "
                f"{sorted(counts)}. Use rate='fixed' for batched compression."
            )
        tokens = torch.stack([result.tokens for result in merged], dim=0)

        if self._track_source:
            identity = torch.eye(features.shape[1], dtype=torch.float32)
            self._source_history = [identity, merged[0].source.detach().cpu().float()]

        return CompressedFeatures(tokens=tokens.squeeze(0) if squeeze else tokens)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        """``[identity, merge]`` for the last forward, ``[T_cur, HW]`` each.

        Rows of the second entry are row-normalised merge weights, which the
        visualiser reads as merging rather than pruning. The installed model's own
        record wins, since real CoreVLM inference merges inside the vision stack.
        """
        model = getattr(self._installed_vlm, "_model", None)
        history = getattr(model, "get_source_history", None)
        if history is not None:
            recorded = history()
            if recorded:
                return [torch.eye(recorded[0].shape[1], dtype=torch.float32), recorded[0]]
        return list(self._source_history)

    # ------------------------------------------------------------------- install

    @override
    def install(self, vlm: VLM) -> None:
        """Hand the configured merge strategy to the CoreVLM.

        Nothing is wrapped or hooked: CORE merges inside its own vision stack, so
        installation is a configuration handshake with that stack rather than an
        interception of ``get_image_features``.
        """
        config = getattr(vlm, "core_config", None)
        if config is None:
            raise TypeError(
                "CORECompressor only installs on a CoreVLM (--backbone core or "
                "core-llama3-8b). "
                "CORE's merging is intrinsic to the ConvNeXt-L/Mask2Former stack "
                "and cannot be applied to a native CLIP-ViT LLaVA-NeXT backbone."
            )
        if self._installed_vlm is not None and self._installed_vlm is not vlm:
            raise RuntimeError("CORECompressor is already installed on another backbone")
        if (
            getattr(vlm, "weights_key", None) == "core-llama3-8b"
            and self._retain_ratio is None
        ):
            raise ValueError(
                "CORE on core-llama3-8b requires retain_ratio so every image's "
                "target is derived from its original visual content length; "
                "absolute max_tokens is not accepted."
            )
        self._installed_vlm = vlm

        # Settle the strategy on the shared CoreConfig *before* touching
        # `vlm.model`: encode_images() and SegmentationVision both branch on it,
        # and the full-grid budget skips loading the Mask2Former decoder entirely.
        for key in (
            "merge_mode",
            "rate",
            "fixed_tokens",
            "retain_ratio",
            "mask_threshold",
            "large_object_first",
        ):
            setattr(config, key, getattr(self._merger, key))
        config.validate()
        if self._adapter_dir is not None:
            self._apply_adapter_dir(vlm)

        # Resolve the lazy LLMOE property explicitly. Using getattr(..., None)
        # here masks an AttributeError raised during model setup and turns the
        # useful root cause into a misleading backbone-type error.
        _ = vlm.model
        model = getattr(vlm, "_model", None)
        if model is None or not hasattr(model, "merger"):
            raise TypeError(
                f"Backbone {type(vlm).__name__} did not build a CORE model to install onto."
            )
        model.merger = self._merger  # idempotent: repeated installs re-share one merger
        self._track_source = self._track_source or getattr(vlm, "_track_source", False)

    def _apply_adapter_dir(self, vlm: VLM) -> None:
        """Give the trained stage-2 directory to a backbone that has none."""
        installed = getattr(vlm, "_adapter_dir", None)
        if installed is not None:
            if str(installed) != str(self._adapter_dir):
                raise ValueError(
                    f"compressor_args.adapter_dir ({self._adapter_dir}) contradicts the "
                    f"backbone's adapter_dir ({installed})."
                )
            return
        if getattr(vlm, "loaded", False):
            raise RuntimeError(
                "The CORE backbone is already loaded; set adapter_dir under backbone_args "
                "so the trained weights load with the model."
            )
        vlm._adapter_dir = str(self._adapter_dir)

    def uninstall(self) -> None:
        self._installed_vlm = None
        self._source_history = []
        self._last_budget = ()


#: Vicuna-13B's hidden width. The released projector and decoder are both
#: 5120-wide, rather than the Vicuna-7B default of 4096.
VICUNA_13B_HIDDEN_SIZE: int = 5120


class CoreVLM13B(CoreVLM):
    """CORE on a Vicuna-13B decoder.

    CORE's vision stack (ConvNeXt-L + OMG-Seg) and its object-order merging are
    unchanged; only the language model differs. The stage-2 release is a
    projector plus LoRA adapters fitted to that decoder, so it is keyed
    separately: the 7B projector is 4096-wide against a 5120-wide decoder and
    the LoRA targets modules of a different depth.
    """

    weights_key = "core-13b"

    def __init__(self, *, llm_name_or_path: str | None = None, **kwargs: Any) -> None:
        if llm_name_or_path is not None:
            kwargs["llm_name_or_path"] = llm_name_or_path
        config = kwargs.get("config")
        if isinstance(config, Mapping):
            kwargs["config"] = {"llm_hidden_size": VICUNA_13B_HIDDEN_SIZE, **config}
        elif config is None:
            kwargs.setdefault("llm_hidden_size", VICUNA_13B_HIDDEN_SIZE)
        super().__init__(**kwargs)
