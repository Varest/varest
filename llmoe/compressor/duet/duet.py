"""DUET-VLM: VisionZip(C) in the ViT, then PyramidDrop text-to-vision pruning.

Reference code: https://github.com/AMD-AGI/DUET-VLM.

One compressor, both released backbones. ``install()`` detects whether it is
sitting on LLaVA-1.5 or Qwen2.5-VL and selects the matching implementation, so
``--compressor DUET`` works with either ``--backbone llava`` or
``--backbone qwen-vl``. Upstream ships them as two separate code drops (``llava/``
and ``qwen2_5_vl/``) because the backbones genuinely differ: LLaVA ranks patches
by ``[CLS]`` attention inside the CLIP tower and takes a fixed token budget, while
Qwen2.5-VL has no ``[CLS]``, emits a different token count per image, uses M-RoPE,
and needs the cut placed after the patch merger. ``QwenDuetAdapter`` holds the
divergent half, the way ``vision_zip.py`` and ``glimpse_prune.py`` split their two
releases; the dominant/contextual selection itself (``stage1.duet_stage1_plan``)
and the PyramidDrop schedule (``stage2``) are shared verbatim.

The two are parameterised differently and the parameters are **not**
interchangeable -- see ``max_tokens`` vs. ``dominant``/``contextual``/
``retain_ratio`` below.

LLaVA-1.5
---------
DUET combines two existing reductions into one schedule, and its published
budgets are only reproducible when both run together:

* **Stage 1 (vision tower).** VisionZip(C) on the penultimate CLIP layer's output
  -- the tensor LLaVA already projects. ``[CLS]`` plus the ``dominant`` most
  ``[CLS]``-attended patches are kept verbatim, and ``contextual`` merged tokens
  are built from the best-scoring leftovers. ``S = 1 + dominant + contextual``
  tokens enter the decoder instead of 576. See ``stage1.py``.

* **Stage 2 (decoder).** PyramidDrop: at each boundary in ``pdrop_layers`` the
  surviving visual tokens are re-ranked by the prompt's attention to them and cut
  to ``floor(S * keep_ratio)``. The default ``(16, 24)`` / ``(0.5, 0.0)`` means
  layers 0-15 see ``S``, layers 16-23 see ``S/2``, and layers 24-31 see none.
  See ``stage2.py``.

Budget semantics (LLaVA-1.5)
----------------------------
``max_tokens`` is **not** the final visual-token count. As in the paper -- and as
in ``sparse_vlm.py`` and ``must_drop.py`` here -- the reported budget is the token
count *averaged over the decoder's layers*::

    budget  dominant  contextual  Stage-1 S  avg over 32 layers  source
    224     349       8           358        223.75              extrapolated
    192     299       7           307        191.75              paper
    128     169       35          205        128.00              paper
    64      71        30          102         63.75              paper
    32      31        19           51         31.75              extrapolated

So ``max_tokens=128`` puts 205 tokens through layers 0-15, 102 through layers
16-23 and 0 through layers 24-31. Only these budgets are supported; pass
``dominant``/``contextual`` explicitly to explore another. Upstream counts
``[CLS]`` inside its ``dominant_tokens``, so every ``dominant`` here is the
paper's value minus one.

Faithfulness notes / harness adaptations (LLaVA-1.5)
----------------------------------------------------
The reference replaces ``LlamaModel.forward`` outright and patches the HF
processor class. This harness drives a stock ``LlavaForConditionalGeneration``, so
-- exactly like ``fastv.py``, ``sparse_vlm.py`` and ``nuwa.py`` -- every stage is
expressed through forward hooks:

* **Stage-1 capture.** The reference asks CLIP for ``output_attentions`` and
  reconstructs the map from ``q_proj``/``k_proj`` whenever the backend returns
  none. Here a pre-hook on the target layer's ``self_attn`` always recomputes the
  single ``[CLS]`` row plus the key metric, which is the same softmax, is
  backend-independent (no forcing eager), and is the only way to obtain the keys
  the contextual merge needs. The layer's ``forward`` is patched to emit the
  reduced tokens, so the projector only ever sees ``S`` of them -- the reference
  re-projects by hand instead, which is equivalent.

* **Placeholders.** The reference subclasses the processor to rewrite
  ``input_ids``. Here ``prepare_inputs`` is wrapped and each ``<image>`` run is
  trimmed to ``S``, matching ``vision_zip.py``; transformers 5.x scatters image
  features with a strict count check either way.

* **Stage-2 scoring.** The reference builds the boundary layer's attention from
  its own ``input_layernorm`` + ``q_proj``/``k_proj``; that is reproduced exactly,
  restricted to the rater rows, so the quadratic ``[L, L]`` map is never
  materialised and the backend stays on SDPA.

* **Positions.** Survivors keep their **original absolute** ``position_ids`` at
  every boundary (the reference's ``position_ids = current_to_original``), so RoPE
  offsets across the dropped span stay intact. While decoding, the reference then
  *subtracts* the number of dropped tokens from those ids, which contradicts the
  absolute ids it wrote at prefill; here decoding keeps the absolute ids the
  decoder computes and only the per-stage ``cache_position`` and mask width are
  narrowed to each stage's shorter cache. This is the same treatment ``fastv.py``
  gives the identical situation.

* **Batching.** The reference requires one contiguous visual span per row and the
  same visual count in every row; both are enforced here, and a ragged batch warns
  once and skips Stage 2 rather than corrupting the gather. Beam search expands
  the batch after prefill, which a per-stage pruned cache cannot follow, so it
  raises instead of silently producing garbage.

* **Training.** Upstream's compression-aware LLaVA pretrain/finetune stack
  (558K alignment + 665K mix) is registered here as ``tr``/``ft``. Stage 2
  physically removes positions, so ``final_keep_index`` maps surviving logits
  back to their labels. DUET can still run training-free as the paper's ablation.

Qwen2.5-VL
----------
Reference code: ``qwen2_5_vl/modeling_qwen2_5vl_duet.py``, a fork of the
transformers ~4.49 modeling file with both stages spliced into
``Qwen2_5_VLForConditionalGeneration.forward``. The published configuration is
``--dominant 540 --contextual 100 --layer-list 14 21 --ratio-list 0.5 0.25``
(``scripts/qwen/*.sh``), i.e. absolute Stage-1 counts and a two-boundary schedule
over the 28-layer decoder.

* **Stage 1.** There is no ``[CLS]`` here, so the LLaVA salience signal does not
  exist. Upstream replaces it and moves the reduction downstream of the patch
  merger: the last ViT block's self-attention is read off, each token scored by
  the attention it *receives* (``mean`` over heads, then ``sum`` over query rows),
  the per-patch scores and keys averaged within each ``2 x 2`` merger group, and
  both un-permuted by ``argsort(window_index)``. The same pooled dominant /
  contextual selection then runs on the *LLM-side* embeddings, and each merged
  token is written back into its anchor's own slot, so the survivors stay in
  raster order and keep their true M-RoPE coordinates.
* **Stage 2.** PyramidDrop again, at ``pdrop_layers`` over the decoder, scored
  from the boundary layer's own ``input_layernorm`` + ``q_proj``/``k_proj`` -- with
  M-RoPE rather than plain RoPE applied to the recomputed query/key.

Upstream prunes ``inputs_embeds`` *after* the image features have been scattered
into the prompt, and gathers the M-RoPE ``position_ids`` of the survivors from the
ones computed over the full sequence. That placement is not incidental -- it is
what makes the port possible at all:

* the strict placeholder/feature count check in ``get_placeholder_mask`` sees the
  full, unpruned counts and passes, so no ``<|image_pad|>`` trimming is needed;
* ``get_rope_index`` derives each visual token's ``(t, h, w)`` coordinates from
  ``image_grid_thw``, so it *must* run on the untouched sequence. Trimming
  placeholders first -- the route the LLaVA path above takes -- would desynchronise
  it from ``pixel_values`` and corrupt every position after the image.

So the Qwen path touches neither ``prepare_inputs`` nor ``get_image_features``. It
hangs a pre-hook on ``model.model.language_model``, which is exactly the call
upstream edits, and rewrites ``inputs_embeds`` / ``position_ids`` /
``attention_mask`` there; Stage 2 then runs on the decoder layers as on LLaVA. See
``QwenDuetAdapter`` for the faithfulness notes and the list of unsupported cases.
"""

from os import PathLike
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, Sequence, override

import torch
from torch.utils.hooks import RemovableHandle

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    decoder_attention_rows,
    decode_attention_mask,
    pruned_attention_mask,
)
from llmoe.compressor.duet.stage1 import duet_stage1
from llmoe.compressor.duet.stage2 import (
    DEFAULT_PDROP_KEEP_RATIOS,
    DEFAULT_PDROP_LAYERS,
    DUET_PROFILES,
    SCORE_REDUCTIONS,
    TEXT_RATERS,
    average_image_tokens,
    pdrop_average_factor,
    reduce_rater_scores,
    select_text_raters,
    stage_keep_counts,
)
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor, VisualizationView
from llmoe.utils.depth import REFERENCE_DEPTH, LayerSchedule, decoder_depth
from llmoe.weights import backbone_key, redirect_backbone, resolve_weights

try:  # RoPE helper for the boundary-attention recompute.
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as _APPLY_ROPE
except Exception:  # pragma: no cover - degrade to no-RoPE if the path moved
    _APPLY_ROPE = None

# The merge pool is `cluster_width * contextual` leftovers (upstream default).
DEFAULT_CLUSTER_WIDTH: int = 4
# Paper Appendix Table 13 profiles for LLaVA-NeXT. Values are the published
# five-view dominant/contextual mixtures; they select mixture proportions here,
# never fixed per-view runtime quotas.
DUET_LLAVA_NEXT_PROFILES: dict[int, tuple[int, int]] = {
    # 1120 is the released profile for the sweep's widest operating point; its
    # mixture is read from the checkpoint's own duet_config
    # (profile_dominant_reference_total 1750, profile_contextual_reference_total
    # 40), which the ratio formula below reproduces as 1790/2880 = 0.621528 --
    # exactly the requested_retention_ratio that checkpoint records.
    1120: (1750, 40),
    640: (850, 175),
    320: (360, 150),
    160: (225, 30),
}
DUET_LLAVA_NEXT_PAPER_BUDGETS: tuple[int, ...] = (1120, 640, 320, 160)
# Each profile's published Stage-1 content total divided by the five full
# 24x24 CLIP views (2,880 patches). Runtime NeXT targets apply these ratios to
# the exact post-unpadding content count; the profile still controls the
# dominant/contextual mixture.
DUET_LLAVA_NEXT_PROFILE_RATIOS: dict[int, float] = {
    profile: (dominant + contextual) / (5 * 576)
    for profile, (dominant, contextual) in DUET_LLAVA_NEXT_PROFILES.items()
}
# Qwen defaults: `scripts/qwen/*.sh` publishes `--layer-list 14 21 --ratio-list
# 0.5 0.25` over the 28-layer decoder, so the LLaVA (16, 24) schedule does not
# transfer. Selected at install() when the caller left `pdrop_layers` unset.
QWEN_PDROP_LAYERS: tuple[int, ...] = (14, 21)
QWEN_PDROP_KEEP_RATIOS: tuple[float, ...] = (0.5, 0.25)
# Decoder depth each published Stage-2 schedule is stated against. `_resolve_pdrop`
# moves the boundaries when the loaded decoder is deeper or shallower, which the
# ratios ride along unchanged: they are fractions of the Stage-1 count, so a
# proportionally scaled schedule spends exactly the same layer average.
QWEN_REFERENCE_DEPTH: int = 28
# Qwen ratio mode (upstream's `configure_duet` defaults): 0.65 dominant + 0.05
# contextual. `retain_ratio` is the sum, as in `vision_zip.py`.
DEFAULT_CONTEXTUAL_RATIO: float = 0.05
# Query rows scored per matmul in the Qwen last-block recompute. The full map
# would be [heads, L, L]; this bounds it to [heads, chunk, L].
DEFAULT_ATTENTION_CHUNK: int = 512


def resolve_image_token_id(config: Any, processor: Any) -> int:
    """Resolve the ``<image>`` placeholder id, mirroring LLaVA's own lookup order."""
    token_id = getattr(config, "image_token_id", None)
    if token_id is None:
        token_id = getattr(config, "image_token_index", None)
    if token_id is None:
        image_token = getattr(processor, "image_token", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if image_token is not None and tokenizer is not None:
            token_id = tokenizer.convert_tokens_to_ids(str(image_token))
    if token_id is None:
        raise RuntimeError("Could not resolve the image placeholder token id for DUET.")
    return int(token_id)


def validate_pdrop_schedule(pdrop_layers: Sequence[int], pdrop_keep_ratios: Sequence[float], *,
                            use_stage2: bool) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Normalise and check one Stage-2 schedule, whichever backbone supplied it."""
    layers = tuple(int(layer) for layer in pdrop_layers)
    ratios = tuple(float(ratio) for ratio in pdrop_keep_ratios)
    if len(layers) != len(ratios):
        raise ValueError(
            f"pdrop_layers and pdrop_keep_ratios must have the same length, got "
            f"{len(layers)} != {len(ratios)}"
        )
    if use_stage2 and not layers:
        raise ValueError("Stage 2 needs at least one pdrop boundary")
    if list(layers) != sorted(set(layers)) or any(layer < 1 for layer in layers):
        raise ValueError("pdrop_layers must be strictly increasing and >= 1")
    if any(not 0.0 <= ratio <= 1.0 for ratio in ratios):
        raise ValueError("pdrop_keep_ratios must be in [0, 1]")
    if list(ratios) != sorted(ratios, reverse=True):
        raise ValueError("pdrop_keep_ratios must be non-increasing")
    return layers, ratios


# ---------------------------------------------------------------------------
# Stage 2 tensor plumbing, shared by both backbones
# ---------------------------------------------------------------------------


def boundary_attention_scores(layer: Any, hidden: torch.Tensor,
                              position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
                              attention_mask: Any, raters: torch.Tensor, *,
                              apply_rope: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None,
                              ) -> torch.Tensor:
    """Head-averaged ``[B, R, L]`` attention of the rater rows at one boundary layer.

    Reproduces the reference's ``_boundary_attention_scores`` /
    ``pdrop_rank_drop``: the boundary layer's own ``input_layernorm`` +
    ``q_proj``/``k_proj`` on its *input* hidden states, softmaxed. Only the rater
    rows are built, so the quadratic ``[L, L]`` map is never materialised and the
    backend stays on SDPA. ``apply_rope`` is the backbone's positional embedding
    (plain RoPE on LLaVA, M-RoPE on Qwen); ``None`` skips it.
    """
    return decoder_attention_rows(
        layer,
        hidden,
        position_embeddings,
        attention_mask,
        raters,
        apply_rope=apply_rope,
        label="DUET Stage 2",
    )


# ---------------------------------------------------------------------------
# The compressor -- backbone detection, plus the LLaVA-1.5 path
# ---------------------------------------------------------------------------


class DUETCompressor(TokenCompressor):
    """VisionZip(C) in the vision tower plus PyramidDrop cuts in the decoder.

    Parameters
    ----------
    max_tokens:
        **LLaVA-1.5 only.** Published budget: the decoder-layer-averaged
        visual-token count, which selects the ``(dominant, contextual)`` profile
        (see the module docstring). Must be one of ``DUET_PROFILES`` unless both
        are given explicitly. Not consulted on Qwen2.5-VL, where the encoder emits
        a different token count per image.
    dominant:
        **LLaVA-1.5 only.** Tokens kept verbatim by Stage 1, ranked by ``[CLS]``
        attention. ``[CLS]`` itself is counted by ``include_cls_token``, not here.
        Normally left unset: ``max_tokens`` picks the published pair from
        ``DUET_PROFILES``. Rejected on Qwen, which takes ``retain_ratio``.
    contextual:
        **LLaVA-1.5 only.** Merged tokens Stage 1 builds from the leftovers; the
        ``max_tokens`` counterpart of ``dominant``.
    retain_ratio:
        **Qwen2.5-VL and LLaVA-NeXT.** On Qwen, the budget is in the convention
        the paper reports it in:
        visual tokens averaged over the decoder's layers, as a fraction of what the
        encoder emitted. Stage 2 thins the sequence further as it goes, so install()
        divides this back through the Stage-2 schedule to find the share Stage 1 must
        keep -- with Qwen's ``(14, 21) @ (0.5, 0.25)`` over 28 layers the schedule
        averages ``0.6875``, so ``retain_ratio=0.2222`` has Stage 1 keep ``0.3232``.
        That makes the ratio comparable with the other compressors' budgets rather
        than with upstream's Stage-1 knob, whose ``0.65 + 0.05 = 0.7`` default is
        ``0.48`` here. A target above the schedule's own average is unreachable and
        raises. Mutually exclusive with ``dominant``/``contextual``, and rejected on
        fixed-span LLaVA-1.5, where ``max_tokens`` selects published absolute
        counts. On LLaVA-NeXT it is a Stage-1 per-image content ratio:
        ``K=max(1, round_half_up(retain_ratio*N_original))`` after exact AnyRes
        unpadding. Structural newlines are excluded and preserved. Pair it with
        ``llava_next_profile`` to choose DUET's published mixture.
    llava_next_profile:
        **LLaVA-NeXT only.** One of ``640``, ``320``, or ``160``. This is a paper
        profile label for the dominant/contextual mixture, not a fixed budget;
        ``retain_ratio`` supplies the actual per-image target.
    contextual_ratio:
        Qwen-only share of the encoder's tokens spent on merged contextual tokens;
        Stage 1's remaining share goes to dominant. Upstream holds this at ``0.05``,
        which at ``retain_ratio=0.2222`` splits Stage 1 ``0.845``/``0.155`` -- the
        paper's Table 15 ratio (``870``/``153``) for its 640-token Qwen budget.
    cluster_width:
        Stage 1 merges only the ``cluster_width * contextual`` best-scoring
        leftovers; the rest are dropped.
    include_cls_token:
        LLaVA-only. Keep ``[CLS]`` as a visual token, as upstream does. Also forces
        LLaVA's feature-select strategy to ``"full"`` so it reaches the projector.
    pdrop_layers:
        Decoder boundaries; layer ``L`` is the first to run on the reduced set.
        ``None`` takes the backbone's published schedule -- ``DEFAULT_PDROP_LAYERS``
        on LLaVA's 32-layer decoder, ``QWEN_PDROP_LAYERS`` on Qwen's 28-layer one.
    pdrop_keep_ratios:
        Fraction of the Stage-1 count surviving each boundary, non-increasing.
        ``None`` pairs with whichever ``pdrop_layers`` default was selected.
    attention_chunk:
        Qwen-only. Query rows scored per matmul when recomputing the last vision
        block's attention. Lower it if the vision tower runs out of memory on very
        large images.
    text_rater:
        Which prompt tokens rank the visual span -- ``"last"``, ``"all"`` (the
        upstream default) or ``"salient"``.
    score_reduction:
        ``"mean"`` or ``"max"`` over the raters.
    vision_feature_layer:
        CLIP layer Stage 1 reduces. Taken from the model config when ``None``
        (-2 on LLaVA-1.5).
    use_stage1 / use_stage2:
        Run each stage. Disabling one is upstream's own ablation.
    track_source:
        Record ``[T_current, T_original]`` matrices for ``llmoe.viz``. Required by
        ``bm --visualize``; off by default so benchmark runs don't pay for it.
    pretrained:
        LLaVA-only released compression-aware backbone weights. ``True`` resolves
        this budget from the weights bundle; a path or repo id selects another.
        Qwen has no released DUET checkpoint and rejects this option.
    """

    def __init__(self, max_tokens: int = 128, *, dominant: int | None = None,
                 contextual: int | None = None, retain_ratio: float | None = None,
                 llava_next_profile: int | None = None,
                 contextual_ratio: float = DEFAULT_CONTEXTUAL_RATIO,
                 cluster_width: int = DEFAULT_CLUSTER_WIDTH,
                 include_cls_token: bool | None = None,
                 llava_next_reference_views: int = 5,
                 pdrop_layers: Sequence[int] | None = None,
                 pdrop_keep_ratios: Sequence[float] | None = None,
                 text_rater: str | None = None, score_reduction: str = "mean",
                 vision_feature_layer: int | None = None, use_stage1: bool = True,
                 use_stage2: bool = True, track_source: bool = False,
                 attention_chunk: int = DEFAULT_ATTENTION_CHUNK,
                 pretrained: bool | str | PathLike[str] | None = None) -> None:
        # The backbone is not known until install(), so the two parameterisations
        # are validated here and cross-checked against the backbone there.
        super().__init__(int(max_tokens))
        self._from_profile: bool = False
        self._from_llava_next_profile: bool = False
        self._llava_next_profile: int | None = (
            None if llava_next_profile is None else int(llava_next_profile)
        )
        self._llava_next_profile_counts: tuple[int, int] | None = None
        if self._llava_next_profile is not None:
            if self._llava_next_profile not in DUET_LLAVA_NEXT_PROFILES:
                raise ValueError(
                    "llava_next_profile must be one of "
                    f"{DUET_LLAVA_NEXT_PAPER_BUDGETS}, got {llava_next_profile}."
                )
            if retain_ratio is None:
                raise ValueError(
                    "llava_next_profile selects DUET's mixture only; LLaVA-NeXT also "
                    "requires retain_ratio for its per-image content target."
                )
            if dominant is not None or contextual is not None:
                raise ValueError(
                    "llava_next_profile supplies the dominant/contextual mixture; do not "
                    "also pass absolute dominant/contextual counts."
                )
            dominant, contextual = DUET_LLAVA_NEXT_PROFILES[self._llava_next_profile]
            self._llava_next_profile_counts = (dominant, contextual)
            self._from_profile = True
            self._from_llava_next_profile = True
            # The profile *is* this run's layer-average budget, which is what
            # `_max_tokens` means here (see the install log and the profile check
            # below) and what `install` resolves the released checkpoint with.
            # Leaving it at the constructor default silently resolves every
            # profile to the same checkpoint. `super().__init__` has already run,
            # so set the attribute rather than the local.
            self._max_tokens = int(self._llava_next_profile)
        if retain_ratio is not None:
            if (dominant is not None or contextual is not None) \
                    and self._llava_next_profile is None:
                raise ValueError(
                    "retain_ratio is upstream's ratio parameterisation and dominant/contextual "
                    "its absolute one; pass one or the other, not both."
                )
            retain_ratio = float(retain_ratio)
            contextual_ratio = float(contextual_ratio)
            if not 0.0 < retain_ratio <= 1.0:
                raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
            # Whether contextual_ratio leaves room for dominant tokens depends on
            # the Stage-2 schedule, which is only known once the backbone is in
            # hand; install() checks it there. Bound it loosely here.
            if not 0.0 <= contextual_ratio < 1.0:
                raise ValueError(
                    f"contextual_ratio must be in [0, 1), got {contextual_ratio}"
                )
        elif dominant is None or contextual is None:
            profiles = {**DUET_PROFILES, **DUET_LLAVA_NEXT_PROFILES}
            if int(max_tokens) not in profiles:
                supported = ", ".join(str(budget) for budget in sorted(profiles))
                raise ValueError(
                    f"Unsupported DUET budget: {max_tokens}. Only {{{supported}}} have a published "
                    f"profile; pass `dominant` and `contextual` to use another. On Qwen2.5-VL pass "
                    f"the counts (upstream publishes 540/100) or `retain_ratio` instead."
                )
            profile_dominant, profile_contextual = profiles[int(max_tokens)]
            dominant = profile_dominant if dominant is None else dominant
            contextual = profile_contextual if contextual is None else contextual
            self._from_profile = True
            self._from_llava_next_profile = int(max_tokens) in DUET_LLAVA_NEXT_PROFILES

        self._dominant: int | None = None if dominant is None else int(dominant)
        self._contextual: int | None = None if contextual is None else int(contextual)
        self._retain_ratio: float | None = retain_ratio
        self._contextual_ratio: float = float(contextual_ratio)
        self._cluster_width: int = int(cluster_width)
        self._attention_chunk: int = int(attention_chunk)
        if include_cls_token is None:
            include_cls_token = not self._from_llava_next_profile
        self._include_cls_token: bool = bool(include_cls_token)
        self._llava_next_reference_views = int(llava_next_reference_views)
        if self._llava_next_reference_views < 1:
            raise ValueError("llava_next_reference_views must be positive")
        if text_rater is None:
            text_rater = "last" if self._from_llava_next_profile else "all"
        # `None` means "the backbone's published schedule", resolved at install().
        # Set by `_resolve_pdrop`; see there. None until a schedule is resolved.
        self._schedule_depth: int | None = None
        self._pdrop_layers: tuple[int, ...] | None = (
            None if pdrop_layers is None else tuple(int(layer) for layer in pdrop_layers))
        self._pdrop_keep_ratios: tuple[float, ...] | None = (
            None if pdrop_keep_ratios is None
            else tuple(float(ratio) for ratio in pdrop_keep_ratios))
        if (self._pdrop_layers is None) != (self._pdrop_keep_ratios is None):
            raise ValueError(
                "pdrop_layers and pdrop_keep_ratios must be given together; the published "
                "schedules differ per backbone, so half a schedule cannot be completed."
            )
        if self._pdrop_layers is not None and self._pdrop_keep_ratios is not None:
            validate_pdrop_schedule(self._pdrop_layers, self._pdrop_keep_ratios,
                                    use_stage2=use_stage2)
        if self._dominant is not None and self._contextual is not None:
            if self._dominant < 0 or self._contextual < 0:
                raise ValueError("dominant and contextual must be non-negative")
            if self._dominant + self._contextual < 1:
                raise ValueError("DUET must keep at least one Stage-1 patch token")
        if self._cluster_width < 1:
            raise ValueError(f"cluster_width must be >= 1, got {cluster_width}")
        if text_rater not in TEXT_RATERS:
            raise ValueError(f"text_rater must be one of {TEXT_RATERS}, got {text_rater!r}")
        if score_reduction not in SCORE_REDUCTIONS:
            raise ValueError(
                f"score_reduction must be one of {SCORE_REDUCTIONS}, got {score_reduction!r}")

        self._text_rater: str = str(text_rater)
        self._score_reduction: str = str(score_reduction)
        self._vision_feature_layer: int | None = (
            None if vision_feature_layer is None else int(vision_feature_layer))
        self._use_stage1: bool = bool(use_stage1)
        self._use_stage2: bool = bool(use_stage2)
        self._track_source: bool = bool(track_source)
        # Tokens leaving Stage 1 and, therefore, the `<image>` placeholder count.
        # `None` in Qwen ratio mode, where it is only known per image.
        self._stage1_tokens: int | None = (
            None if self._dominant is None or self._contextual is None
            else self._dominant + self._contextual + (1 if self._include_cls_token else 0))
        # Layers `[pdrop_layers[s], pdrop_layers[s + 1])` run under stage `s`;
        # filled once install() has resolved the schedule.
        self._boundary_stage: dict[int, int] = {}
        self._qwen: Any | None = None
        self._llava_next_adapter: Any | None = None

        # Per-forward Stage-1 capture, aligned to patch rows (CLS stripped).
        self._cls_attention: torch.Tensor | None = None  # [B, N]
        self._metric: torch.Tensor | None = None  # [B, N, head_dim]
        self._stage1_source: torch.Tensor | None = None  # [S, N + 1], viz only
        # Per-forward Stage-2 state, rebuilt by the inner-model pre-hook.
        self._image_start: torch.Tensor | None = None  # [B] first visual column
        self._rater_index: torch.Tensor | None = None  # [B, R] offsets past the image
        self._rater_valid: torch.Tensor | None = None  # [B, R]
        self._visual_count: int = 0
        self._keep_counts: tuple[int, ...] = ()
        self._stage_state: dict[int, dict[str, Any]] = {}
        self._composed_source: torch.Tensor | None = None
        self._source_history: list[torch.Tensor] = []
        # Map from current columns to the original prefill, used to align labels
        # during the LLaVA compression-aware training recipe.
        self._final_keep_index: torch.Tensor | None = None
        self._forward_count: int = 0
        self._warned: set[str] = set()
        self._is_llava_next: bool = False
        self._llava_next_decoder_layers: int = 0
        self._llava_next_run_metadata: dict[str, Any] = {}
        self._llava_next_structural_positions: torch.Tensor | None = None
        self._llava_next_content_count: int = 0
        self._benchmark_indices: torch.Tensor | None = None
        self._metadata_backbone: str | None = None
        # Resolved in `install`, not here: the released checkpoint is a whole
        # finetuned LLaVA-1.5-7B that `redirect_backbone` loads *in place of* the
        # selected backbone, so resolving it before the backbone is known is how a
        # `--backbone llava-13b` run would silently end up evaluating the 7B.
        self._pretrained_arg = pretrained
        self._checkpoint_dir: Path | None = None

        # Installation bookkeeping, all undone by uninstall().
        self._original_model_id: str | None = None
        self._installed_vlm: VLM | None = None
        # These are borrowed from the backbone. TokenCompressor is an nn.Module,
        # so normal assignment would register them as owned submodules and emit
        # duplicate, storage-aliased keys under compressor.* at checkpoint time.
        object.__setattr__(self, "_language_model", None)
        self._hook_handles: list[RemovableHandle] = []
        self._image_token_id: int | None = None
        object.__setattr__(self, "_target_layer", None)
        self._original_layer_forward: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._original_select_strategy: Any = None

    def _set_borrowed_module(self, name: str, module: torch.nn.Module | None) -> None:
        """Keep a backbone-owned module reachable without taking nn.Module ownership."""

        if name not in {"_language_model", "_target_layer"}:
            raise ValueError(f"Unsupported borrowed DUET module slot {name!r}.")
        object.__setattr__(self, name, module)

    @property
    @override
    def budget(self) -> int | float | str:
        if self._retain_ratio is not None:
            return self._retain_ratio
        if self._from_profile:
            return self._max_tokens
        return f"d{self._dominant}c{self._contextual}"

    def validate_llava_next(self) -> None:
        """Require ratio budgeting plus one canonical DUET NeXT mixture."""

        if self._retain_ratio is None:
            raise ValueError(
                "DUET/LLaVA-NeXT requires retain_ratio so every image is budgeted from "
                "its exact post-unpadding content count."
            )
        if self._llava_next_profile is None:
            supported = ", ".join(str(value) for value in DUET_LLAVA_NEXT_PAPER_BUDGETS)
            raise ValueError(
                "DUET/LLaVA-NeXT requires llava_next_profile to select the published "
                f"dominant/contextual mixture; choose one of {{{supported}}}."
            )
        published = DUET_LLAVA_NEXT_PROFILES[self._llava_next_profile]
        if self._llava_next_profile_counts != published \
                or (self._dominant, self._contextual) != published:
            raise ValueError(
                f"DUET-{self._llava_next_profile}/LLaVA-NeXT requires the published "
                f"dominant/contextual profile {published}, got "
                f"{(self._dominant, self._contextual)}."
            )
        if self._include_cls_token:
            raise ValueError(
                "DUET/LLaVA-NeXT requires include_cls_token=false because CLS provides "
                "Stage-1 scores but is not a decoder token."
            )
        if self._llava_next_reference_views != 5:
            raise ValueError(
                "DUET/LLaVA-NeXT profiles are calibrated at exactly five reference views."
            )
        if not self._use_stage1 or not self._use_stage2:
            raise NotImplementedError(
                "DUET/LLaVA-NeXT implements the defining two-stage policy only."
            )
        if self._pdrop_layers is None and self._pdrop_keep_ratios is None:
            self._pdrop_layers = DEFAULT_PDROP_LAYERS
            self._pdrop_keep_ratios = DEFAULT_PDROP_KEEP_RATIOS
        layers = self._pdrop_layers
        ratios = self._pdrop_keep_ratios
        if layers != DEFAULT_PDROP_LAYERS or ratios != DEFAULT_PDROP_KEEP_RATIOS:
            raise ValueError(
                "DUET/LLaVA-NeXT requires pdrop_layers=[16,24] and "
                "pdrop_keep_ratios=[0.5,0.0]."
            )
        if self._cluster_width != DEFAULT_CLUSTER_WIDTH:
            raise ValueError("DUET/LLaVA-NeXT requires the paper's cluster_width=4")
        if self._text_rater != "last" or self._score_reduction != "mean":
            raise ValueError(
                "DUET/LLaVA-NeXT requires text_rater='last' and score_reduction='mean'."
            )
        if self._vision_feature_layer not in (None, -2):
            raise ValueError("DUET/LLaVA-NeXT requires vision_feature_layer=-2")

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[DUET] WARNING: {message}", flush=True)

    # ----------------------------------------------------------- TokenCompressor

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, keys: torch.Tensor | None = None,
                 **kwargs: Any) -> CompressedFeatures:
        """Run Stage 1 over ``[B, N, C]`` (or ``[N, C]``) patch features.

        This is the tensor-level entry point; the real path goes through
        ``install()``, where the vision-tower hook supplies the capture and the
        decoder finishes the reduction. ``[CLS]`` is *not* part of ``features``
        (the caller re-attaches it), so the output is ``dominant + contextual``
        tokens -- one fewer than ``stage1_tokens`` and, as the module docstring
        explains, unrelated to ``max_tokens``.

        ``attention`` is the ``[CLS]``->patch attention (``[N]`` or ``[B, N]``) and
        ``keys`` the per-patch key metric; both default to the encoder hook's
        capture. Without them Stage 1 falls back to L2 norms for salience and the
        features themselves as the merge metric, which is what upstream's
        ``visionzip_projected_features_fallback`` does when it cannot reach the
        pre-projector attention.

        LLaVA-only: the Qwen path reduces LLM-side embeddings against a per-image
        count and drives both stages from ``QwenDuetAdapter``, so it has no
        fixed-count tensor entry point.
        """
        del layer_idx, kwargs

        if self._dominant is None or self._contextual is None:
            raise RuntimeError(
                "DUETCompressor.compress() implements the fixed-count LLaVA Stage 1, but this "
                "compressor was built in ratio mode for Qwen2.5-VL, where the reduction runs on "
                "the LLM-side embeddings. See `QwenDuetAdapter` in `duet_qwen.py`."
            )

        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Expected [B, N, C] or [N, C], got {tuple(features.shape)}")
        batch, n_tokens, _ = features.shape
        if n_tokens == 0:
            raise ValueError("Cannot compress an empty sequence")

        if attention is None:
            attention = self._cls_attention
        if keys is None:
            keys = self._metric
        if attention is None:
            attention = features.float().norm(dim=-1)
        if keys is None:
            keys = features
        attention = attention.to(features.device).float()
        if attention.ndim == 1:
            attention = attention.unsqueeze(0)
        if attention.shape[0] == 1 and batch > 1:
            attention = attention.expand(batch, -1)
        keys = keys.to(features.device)
        if keys.ndim == 2:
            keys = keys.unsqueeze(0)
        if keys.shape[0] == 1 and batch > 1:
            keys = keys.expand(batch, -1, -1)

        tokens, source = duet_stage1(
            features, attention, keys, dominant=self._dominant, contextual=self._contextual,
            cluster_width=self._cluster_width, track_source=self._track_source,
        )
        if source is not None:
            self._stage1_source = self._with_cls_row(source[0].detach().float().cpu())
            self._source_history = [
                torch.eye(n_tokens + 1, dtype=torch.float32), self._stage1_source]
            self._composed_source = self._stage1_source
        return CompressedFeatures(tokens=tokens.squeeze(0) if squeeze else tokens)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        """``[identity, stage1, cut_1, ...]`` for the last forward, ``[T_cur, N + 1]``.

        Column 0 is ``[CLS]``, so patch ``p`` occupies column ``p + 1``. ``stage1``
        carries the real merge weights; each later entry is the same rows narrowed
        by one Stage-2 boundary. A boundary that leaves no visual token contributes
        no entry, since there is nothing left to render. Empty when ``track_source``
        is off or no forward has run. On Qwen the matrices cover the merged visual
        tokens alone, since there is no ``[CLS]``.
        """
        if self._qwen is not None:
            return self._qwen.get_source_history()
        return list(self._source_history)

    @override
    def get_benchmark_record_metadata(self, batch_size: int) -> list[dict[str, Any]]:
        if not self._llava_next_run_metadata:
            return super().get_benchmark_record_metadata(batch_size)
        if int(batch_size) != 1:
            raise RuntimeError("DUET/LLaVA-NeXT runtime metadata requires batch_size=1.")
        return [{"compression_runtime": dict(self._llava_next_run_metadata)}]

    @override
    def get_visualization_views(
        self,
        vlm: VLM,
        inputs: Mapping[str, Any],
        image_processor: Any,
    ) -> list[VisualizationView] | None:
        if not self._is_llava_next:
            return None
        from llmoe.compressor.duet.duet_llava_next import duet_visualization_view

        return [duet_visualization_view(vlm, inputs, image_processor)]

    @staticmethod
    def _with_cls_row(patch_source: torch.Tensor) -> torch.Tensor:
        """Widen a ``[out, N]`` patch attribution to ``[out + 1, N + 1]`` with CLS."""
        out_tokens, num_patches = patch_source.shape
        full = torch.zeros(out_tokens + 1, num_patches + 1, dtype=torch.float32)
        full[0, 0] = 1.0  # the CLS row stays one-hot on the CLS column
        full[1:, 1:] = patch_source
        return full

    # ------------------------------------------------------------------- install

    @staticmethod
    def _detect_backbone(vlm: VLM) -> str:
        """``"qwen"`` or ``"llava"``, checked structurally so forks still pass."""
        inner = getattr(getattr(vlm, "model", None), "model", None)
        if getattr(inner, "visual", None) is not None:
            return "qwen"
        if getattr(inner, "vision_tower", None) is not None:
            return "llava"
        raise TypeError(
            f"DUET supports the LLaVA-1.5 and Qwen2.5-VL backbones; "
            f"{type(getattr(vlm, 'model', None)).__name__} exposes neither "
            f"`model.model.vision_tower` nor `model.model.visual`. Run with `--backbone llava` "
            f"or `--backbone qwen-vl`."
        )

    def _resolve_pdrop(self, backbone: str, depth: int | None = None
                       ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """This backbone's Stage-2 schedule, published defaults included.

        A published boundary is a fraction of the decoder, not an absolute layer,
        so on a decoder that is not the depth the schedule was stated at -- 40
        layers on LLaVA-1.5-13B against the LLaVA default's 32 -- it moves to the
        same fraction. The ratios need no re-solving: they are fractions of the
        Stage-1 count, so the layer average is unchanged by construction. An
        explicit caller-supplied schedule is taken literally, as everywhere else.

        :param depth: the loaded decoder's depth. ``None`` -- the trainers, which
            resolve a schedule with no model in hand -- leaves the published
            boundaries where they are, which is the depth they train at.
        """
        reference = QWEN_REFERENCE_DEPTH if backbone == "qwen" else REFERENCE_DEPTH
        # The depth these boundaries describe. Recorded so a checkpoint's metadata
        # cannot state its layer average against a different decoder than the one
        # the schedule was resolved for -- which is how a 40-layer run came to
        # report its average over 32 layers.
        self._schedule_depth = int(depth) if depth is not None else reference
        if self._pdrop_layers is not None and self._pdrop_keep_ratios is not None:
            return validate_pdrop_schedule(self._pdrop_layers, self._pdrop_keep_ratios,
                                           use_stage2=self._use_stage2)
        if backbone == "qwen":
            layers, ratios = QWEN_PDROP_LAYERS, QWEN_PDROP_KEEP_RATIOS
        else:
            layers, ratios = DEFAULT_PDROP_LAYERS, DEFAULT_PDROP_KEEP_RATIOS
        if self._use_stage2 and depth is not None and int(depth) != reference:
            resolved = LayerSchedule(layers, ratios, reference, n_visual=1).rescaled_to(
                int(depth), min_keep=0, integral=False, label="DUET's Stage-2 schedule")
            layers, ratios = resolved.boundaries, tuple(float(r) for r in resolved.keeps)
        return validate_pdrop_schedule(layers, ratios, use_stage2=self._use_stage2)

    @override
    def install(self, vlm: VLM) -> None:
        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return  # idempotent: repeated benchmarks must not stack wrappers
            raise RuntimeError("DUETCompressor is already installed on another backbone")

        resolved = resolve_weights("DUET", int(self._max_tokens),
                                   pretrained=self._pretrained_arg,
                                   backbone=backbone_key(vlm))
        self._checkpoint_dir = None if resolved is None else Path(resolved)
        if self._checkpoint_dir is not None:
            identity = (
                type(vlm).__name__ + " " + str(getattr(vlm, "_model_id", ""))
            ).lower()
            if "qwen" in identity:
                raise ValueError("DUET's released `pretrained` weights are LLaVA-only")
            self._original_model_id = redirect_backbone(vlm, self._checkpoint_dir)

        backbone = self._detect_backbone(vlm)
        from llmoe.compressor.llava_next import is_llava_next

        self._is_llava_next = is_llava_next(vlm)
        if self._is_llava_next:
            if backbone != "llava":
                raise TypeError("DUET detected LLaVA-NeXT without a CLIP-style LLaVA layout")
            self.validate_llava_next()
        elif (
            int(self._max_tokens) in DUET_LLAVA_NEXT_PROFILES
            and (self._dominant, self._contextual)
            == DUET_LLAVA_NEXT_PROFILES[int(self._max_tokens)]
        ):
            raise ValueError(
                f"DUET budget {self._max_tokens} is a LLaVA-NeXT profile and cannot "
                "run on fixed-span LLaVA-1.5."
            )
        self._pdrop_layers, self._pdrop_keep_ratios = self._resolve_pdrop(
            backbone, None if self._is_llava_next else decoder_depth(vlm))
        self._boundary_stage = {
            layer: stage for stage, layer in enumerate(self._pdrop_layers)}
        if backbone == "qwen":
            self._install_qwen(vlm)
            return

        if self._retain_ratio is not None and not self._is_llava_next:
            raise ValueError(
                "retain_ratio selects DUET's Qwen2.5-VL path. On LLaVA-1.5 use `max_tokens` (or "
                "an explicit `dominant`/`contextual` split): the released budgets are absolute "
                "counts tuned per budget, so a ratio would not reproduce the paper's numbers."
            )
        inner = vlm.model.model
        language_model = getattr(inner, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if not layers:
            raise TypeError("DUET needs a decoder stack at model.model.language_model.layers")
        if self._use_stage2 and self._pdrop_layers[-1] >= len(layers):
            raise ValueError(
                f"pdrop layer {self._pdrop_layers[-1]} is too large for a {len(layers)}-layer "
                f"decoder"
            )
        if self._use_stage1 and getattr(inner, "multi_modal_projector", None) is None:
            raise TypeError("DUET Stage 1 needs a multi_modal_projector on model.model")

        self._installed_vlm = vlm
        self._set_borrowed_module("_language_model", language_model)
        self._llava_next_decoder_layers = len(layers)
        self._image_token_id = resolve_image_token_id(vlm.model.config, vlm.processor)

        try:
            if self._is_llava_next:
                from llmoe.compressor.duet.duet_llava_next import LlavaNextDuetAdapter

                self._metadata_backbone = "llava_next"
                self._llava_next_adapter = LlavaNextDuetAdapter(self, vlm)
                self._llava_next_adapter.install()
            elif self._use_stage1:
                self._install_stage1(vlm)
                self._wrap_prepare_inputs(vlm)
            if self._use_stage2:
                # Stage 2 is shared; Stage 1 has already rewritten NeXT's
                # placeholder run to its exact geometry-dependent length.
                self._hook_handles.append(
                    inner.register_forward_pre_hook(self._capture_span, with_kwargs=True))
                for layer_idx in range(self._pdrop_layers[0], len(layers)):
                    self._hook_handles.append(
                        layers[layer_idx].register_forward_pre_hook(
                            self._make_layer_hook(layer_idx), with_kwargs=True))
        except Exception:
            self.uninstall()
            raise

        if self._is_llava_next:
            print(
                f"[DUET/LLaVA-NeXT] install: profile={self._llava_next_profile}, "
                f"retention_ratio={self._retain_ratio:g}, target=round(r*N_original), "
                "structural_newlines=preserved",
                flush=True,
            )
            return

        average = average_image_tokens(
            self._stage1_tokens, self._pdrop_layers, self._pdrop_keep_ratios, len(layers))
        print(f"[DUET] install: budget={self._max_tokens} (layer-average), stage1={self._stage1_tokens} "
              f"(cls={int(self._include_cls_token)} + dominant={self._dominant} + "
              f"contextual={self._contextual}), pdrop={list(self._pdrop_layers)} @ratios"
              f"{list(self._pdrop_keep_ratios)} -> "
              f"{list(stage_keep_counts(self._stage1_tokens, self._pdrop_keep_ratios))}, "
              f"text_rater={self._text_rater}, stages={int(self._use_stage1)}{int(self._use_stage2)}, "
              f"measured layer-average={average:.2f} over {len(layers)} layers", flush=True)

    def qwen_stage1_ratios(self, num_layers: int) -> tuple[float, float]:
        """``(dominant, contextual)`` shares of the encoder's tokens Stage 1 keeps.

        ``retain_ratio`` is the layer-averaged budget, the convention the paper
        reports its Qwen token counts in. Stage 2 keeps thinning the sequence after
        Stage 1, so Stage 1 has to start above the target: divide the target by what
        Stage 2 averages over a ``num_layers`` decoder to recover its share. Shared by
        the inference adapter and the training decoder so both start at the same width.

        :param num_layers: decoder depth (28 for Qwen2.5-VL-7B).
        :raises ValueError: if the target is unreachable, or leaves no dominant tokens.
        """
        if self._retain_ratio is None:
            raise ValueError(
                "qwen_stage1_ratios needs retain_ratio; the dominant/contextual "
                "parameterisation gives Stage-1 counts directly."
            )
        layers, ratios = self._pdrop_layers, self._pdrop_keep_ratios
        if layers is None or ratios is None:  # not yet resolved against a backbone
            layers, ratios = self._resolve_pdrop("qwen")
        factor = (pdrop_average_factor(layers, ratios, num_layers)
                  if self._use_stage2 else 1.0)
        stage1_ratio = self._retain_ratio / factor
        if stage1_ratio > 1.0:
            raise ValueError(
                f"retain_ratio={self._retain_ratio:g} is a layer-averaged budget, but "
                f"Stage 2 alone (pdrop={list(layers)} @ratios{list(ratios)}) already "
                f"averages {factor:g} of the encoder's tokens over {num_layers} layers, "
                f"so Stage 1 cannot reach it even keeping every token. The highest "
                f"budget this schedule can express is {factor:g}."
            )
        dominant_ratio = stage1_ratio - self._contextual_ratio
        if dominant_ratio <= 0.0:
            raise ValueError(
                f"contextual_ratio={self._contextual_ratio:g} leaves no dominant tokens: "
                f"retain_ratio={self._retain_ratio:g} back-solves to a Stage-1 share of "
                f"{stage1_ratio:g}. Lower contextual_ratio or raise retain_ratio."
            )
        return dominant_ratio, self._contextual_ratio

    @staticmethod
    def _decoder_depth(vlm: VLM) -> int:
        """Qwen's decoder layer count, read structurally so forks still pass."""
        inner = getattr(getattr(vlm, "model", None), "model", None)
        layers = getattr(getattr(inner, "language_model", None), "layers", None)
        if not layers:
            raise TypeError(
                "DUET's Qwen path needs `model.model.language_model.layers` to turn a "
                "layer-averaged retain_ratio into a Stage-1 retention."
            )
        return len(layers)

    def _install_qwen(self, vlm: VLM) -> None:
        """Hand over to ``QwenDuetAdapter`` once parameters are checked."""
        if self._retain_ratio is None:
            raise ValueError(
                "DUET on Qwen2.5-VL takes `retain_ratio`, the layer-averaged budget the "
                "rest of the Qwen compressors take: the NaViT encoder gives every image a "
                "different visual-token count, so an absolute count is not comparable "
                "across samples -- upstream's is clamped per image and becomes a no-op on "
                "anything smaller than the count itself. Pass "
                "`compressor_args: {retain_ratio: 0.2222}`."
            )
        dominant_ratio, _ = self.qwen_stage1_ratios(self._decoder_depth(vlm))
        from llmoe.compressor.duet.duet_qwen import QwenDuetAdapter

        self._installed_vlm = vlm
        self._qwen = QwenDuetAdapter(
            dominant_ratio=dominant_ratio, contextual_ratio=self._contextual_ratio,
            cluster_width=self._cluster_width, pdrop_layers=self._pdrop_layers,
            pdrop_keep_ratios=self._pdrop_keep_ratios, text_rater=self._text_rater,
            score_reduction=self._score_reduction, use_stage1=self._use_stage1,
            use_stage2=self._use_stage2, track_source=self._track_source,
            attention_chunk=self._attention_chunk, retain_ratio=self._retain_ratio,
        )
        self._qwen.install(vlm, image_token_id=resolve_image_token_id(
            vlm.model.config, vlm.processor))

    def uninstall(self) -> None:
        if self._qwen is not None:
            self._qwen.uninstall()
            self._qwen = None
            self._installed_vlm = None
            return
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        if self._llava_next_adapter is not None:
            self._llava_next_adapter.uninstall()
        self._llava_next_adapter = None

        if self._target_layer is not None and self._original_layer_forward is not None:
            self._target_layer.forward = self._original_layer_forward
            self._target_layer._duet_patched = False
        if self._installed_vlm is not None and self._original_prepare_inputs is not None:
            self._installed_vlm.prepare_inputs = self._original_prepare_inputs
            self._installed_vlm._duet_prepare_wrapped = False
        config = getattr(getattr(self._installed_vlm, "model", None), "config", None)
        if config is not None and self._original_select_strategy is not None:
            config.vision_feature_select_strategy = self._original_select_strategy

        if self._original_model_id is not None and self._installed_vlm is not None:
            self._installed_vlm._model_id = self._original_model_id
            self._installed_vlm._model = None
            self._installed_vlm._processor = None
        self._original_model_id = None
        self._set_borrowed_module("_target_layer", None)
        self._original_layer_forward = None
        self._original_prepare_inputs = None
        self._original_select_strategy = None
        self._image_token_id = None
        self._installed_vlm = None
        self._set_borrowed_module("_language_model", None)
        self._is_llava_next = False
        self._llava_next_decoder_layers = 0
        self._llava_next_run_metadata = {}
        self._llava_next_structural_positions = None
        self._llava_next_content_count = 0
        self._benchmark_indices = None
        self._reset_forward_state()

    def _reset_forward_state(self) -> None:
        self._cls_attention = None
        self._metric = None
        self._stage1_source = None
        self._image_start = None
        self._rater_index = None
        self._rater_valid = None
        self._visual_count = 0
        self._keep_counts = ()
        self._stage_state = {}
        self._composed_source = None
        self._final_keep_index = None
        self._source_history = []
        self._llava_next_run_metadata = {}
        self._llava_next_structural_positions = None
        self._llava_next_content_count = 0
        self._benchmark_indices = None

    @property
    def final_keep_index(self) -> torch.Tensor | None:
        """Map surviving columns to the original prefill for training labels."""
        return self._final_keep_index

    # ------------------------------------------------------------------- Stage 1

    def _install_stage1(self, vlm: VLM) -> None:
        """Capture the CLS attention and keys, then reduce the target layer's output."""
        if self._include_cls_token:
            # transformers drops row 0 under the default strategy, but upstream
            # keeps [CLS] as a visual token, so it has to reach the projector.
            config = getattr(vlm.model, "config", None)
            if config is not None:
                self._original_select_strategy = getattr(
                    config, "vision_feature_select_strategy", None)
                config.vision_feature_select_strategy = "full"

        layers = self._vision_encoder_layers(vlm)
        layer = self._resolve_feature_layer(vlm)
        # `hidden_states[k]` is the output of layer `k - 1`, so a negative feature
        # layer maps to `layers[layer]` and a positive one to `layers[layer - 1]`.
        target = layers[layer] if layer < 0 else layers[layer - 1]
        if getattr(target, "_duet_patched", False):
            return

        self._hook_handles.append(
            target.self_attn.register_forward_pre_hook(self._capture_cls_attention,
                                                       with_kwargs=True))
        self._set_borrowed_module("_target_layer", target)
        self._original_layer_forward = target.forward
        original_layer_forward = target.forward

        def patched_layer_forward(*args: Any, **layer_kwargs: Any) -> torch.Tensor:
            return self._reduce_hidden_states(original_layer_forward(*args, **layer_kwargs))

        target.forward = patched_layer_forward
        target._duet_patched = True

    @staticmethod
    def _vision_encoder_layers(vlm: VLM) -> Any:
        """Locate the CLIP encoder layers.

        transformers 5.x hangs the encoder directly off ``CLIPVisionModel``;
        earlier layouts nest it under ``vision_model``.
        """
        vision_tower = getattr(getattr(getattr(vlm, "model", None), "model", None),
                               "vision_tower", None)
        vision_model = getattr(vision_tower, "vision_model", vision_tower)
        layers = getattr(getattr(vision_model, "encoder", None), "layers", None)
        if layers is None:
            raise RuntimeError(
                "DUET Stage 1 expects a CLIP-style vision tower exposing `encoder.layers`; "
                "the active backbone does not provide one."
            )
        return layers

    def _resolve_feature_layer(self, vlm: VLM) -> int:
        layer = self._vision_feature_layer
        if layer is None:
            config = getattr(getattr(vlm, "model", None), "config", None)
            layer = getattr(config, "vision_feature_layer", None)
            if layer is None:
                layer = getattr(config, "vision_feature_select_layer", -2)
        if not isinstance(layer, int):
            raise ValueError(
                f"DUET needs a single vision_feature_layer, got {layer!r}. Multiple selected "
                "layers would be read at different token counts."
            )
        return layer

    def _capture_cls_attention(self, module: Any, args: tuple, kwargs: dict) -> None:
        """Recompute this layer's ``[CLS]`` row and key metric from ``q_proj``/``k_proj``.

        The pre-hook sits on ``self_attn``, so ``hidden_states`` is already
        ``layer_norm1``-normalised -- the same tensor the reference feeds its own
        ``k_proj`` -- and this reproduces the layer's softmax exactly for row 0.
        """
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            return None
        batch, seq, _ = hidden_states.shape
        num_heads, head_dim, scale = module.num_heads, module.head_dim, module.scale
        queries = module.q_proj(hidden_states).view(batch, seq, num_heads, head_dim).transpose(1, 2)
        keys = module.k_proj(hidden_states).view(batch, seq, num_heads, head_dim).transpose(1, 2)
        logits = torch.matmul(queries[:, :, :1, :].float(), keys.float().transpose(-1, -2)) * scale
        # CLS attention over the patches only, summed over heads (drop CLS->CLS).
        self._cls_attention = logits.softmax(dim=-1)[:, :, 0, 1:].sum(dim=1)  # [B, N]
        self._metric = keys.mean(dim=1)[:, 1:, :].contiguous()  # [B, N, head_dim]
        return None

    def _reduce_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply Stage 1 to the target layer's ``[B, S, C]`` output.

        The encoder never reorders tokens, so the patch rows line up with the
        capture. ``[CLS]`` stays at index 0 and the patch rows are reduced to
        ``dominant + contextual``.
        """
        cls_token = hidden_states[:, :1, :]
        result = self.compress(hidden_states[:, 1:, :])
        # Clear the capture so a forward that somehow bypasses the pre-hook cannot
        # silently reuse stale attention/keys.
        self._cls_attention = None
        self._metric = None
        return torch.cat([cls_token, result.tokens.to(hidden_states.dtype)], dim=1)

    # ------------------------------------------------------------- placeholders

    def _wrap_prepare_inputs(self, vlm: VLM) -> None:
        if getattr(vlm, "_duet_prepare_wrapped", False):
            return
        self._original_prepare_inputs = vlm.prepare_inputs
        original_prepare_inputs = vlm.prepare_inputs

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            return self._trim_image_placeholders(vlm, original_prepare_inputs(*args, **kwargs))

        vlm.prepare_inputs = wrapped_prepare_inputs
        vlm._duet_prepare_wrapped = True

    def _trim_image_placeholders(self, vlm: VLM, inputs: Any) -> Any:
        """Trim each ``<image>`` run to the Stage-1 count LLaVA will be handed."""
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return inputs
        attention_mask = inputs.get("attention_mask")
        id_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        changed = False

        for row in range(input_ids.shape[0]):
            ids = input_ids[row]
            positions = (ids == self._image_token_id).nonzero(as_tuple=True)[0]
            if positions.numel() <= self._stage1_tokens:
                id_rows.append(ids)
                if attention_mask is not None:
                    mask_rows.append(attention_mask[row])
                continue
            changed = True
            keep = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
            keep[positions[self._stage1_tokens:]] = False
            id_rows.append(ids[keep])
            if attention_mask is not None:
                mask_rows.append(attention_mask[row][keep])

        if not changed:
            return inputs
        inputs["input_ids"] = self._left_pad_stack(id_rows, self._pad_token_id(vlm))
        if attention_mask is not None:
            inputs["attention_mask"] = self._left_pad_stack(mask_rows, 0)
        return inputs

    @staticmethod
    def _pad_token_id(vlm: VLM) -> int:
        tokenizer = getattr(vlm.processor, "tokenizer", None)
        pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        return int(pad) if pad is not None else 0

    @staticmethod
    def _left_pad_stack(rows: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        """Stack ragged rows with left padding, matching the tokenizer's side."""
        max_len = max(row.shape[0] for row in rows)
        padded: list[torch.Tensor] = []
        for row in rows:
            if row.shape[0] < max_len:
                pad = torch.full((max_len - row.shape[0],), pad_value, dtype=row.dtype,
                                 device=row.device)
                row = torch.cat([pad, row], dim=0)
            padded.append(row)
        return torch.stack(padded, dim=0)

    # -------------------------------------------------------- Stage 2: prefill

    def _capture_span(self, _module: Any, args: tuple, kwargs: dict) -> None:
        """Record the visual span and the raters from each prefill's ``input_ids``.

        Decode steps (width-1 ``input_ids``) are left alone so the state built by
        the prefill -- and the per-stage caches it produced -- stays valid.
        """
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or input_ids.shape[1] <= 1:
            return None
        past_key_values = kwargs.get("past_key_values")
        if self._is_llava_next and past_key_values is not None:
            get_seq_length = getattr(past_key_values, "get_seq_length", None)
            if not callable(get_seq_length) or int(get_seq_length()) != 0:
                raise NotImplementedError(
                    "DUET/LLaVA-NeXT cannot prune a prefill that starts from an "
                    "externally populated or unknown KV-cache layout."
                )

        # This hook fires before `get_image_features`, so Stage 1's capture and its
        # source attribution are (re)written later in this same forward.
        self._reset_forward_state()
        self._forward_count += 1

        rows = [(row == self._image_token_id).nonzero(as_tuple=True)[0] for row in input_ids]
        counts = {int(row.numel()) for row in rows}
        if counts == {0}:
            return None  # text-only prompt: nothing for Stage 2 to prune
        if self._is_llava_next and input_ids.shape[0] != 1:
            raise NotImplementedError(
                "DUET/LLaVA-NeXT supports physical Stage-2 pruning with batch_size=1 only"
            )
        if len(counts) != 1:
            if self._is_llava_next:
                raise NotImplementedError(
                    f"DUET/LLaVA-NeXT does not support ragged visual batches: {sorted(counts)}"
                )
            self._warn_once(
                "ragged_batch",
                f"rows carry different visual-token counts {sorted(counts)}; Stage 2 is SKIPPED "
                f"for such batches. Batch images of equal token count.",
            )
            return None
        count = counts.pop()
        starts = torch.stack([row[0] for row in rows], dim=0)
        if any(int(row[-1]) - int(row[0]) + 1 != count for row in rows):
            if self._is_llava_next:
                raise NotImplementedError(
                    "DUET/LLaVA-NeXT supports one contiguous visual span and one source image"
                )
            self._warn_once(
                "ragged_span",
                "the visual tokens are not one contiguous run per row; Stage 2 is SKIPPED.",
            )
            return None
        if self._use_stage1 and not self._is_llava_next and count != self._stage1_tokens:
            self._warn_once(
                "unexpected_count",
                f"expected {self._stage1_tokens} visual tokens per row but found {count}; the "
                f"pdrop ratios are applied to {count} instead.",
            )

        tokenizer = getattr(self._installed_vlm.processor, "tokenizer", None) \
            if self._installed_vlm is not None else None
        index, valid = select_text_raters(
            input_ids, kwargs.get("attention_mask"), starts, count,
            mode=self._text_rater, tokenizer=tokenizer,
        )
        if not bool(valid.any()):
            self._warn_once(
                "no_raters",
                "no prompt token is available to rate the visual span; Stage 2 is SKIPPED.",
            )
            return None

        self._image_start = starts
        self._visual_count = count
        self._keep_counts = stage_keep_counts(count, self._pdrop_keep_ratios)
        # Held relative to the end of the visual span, which is the only part of
        # the sequence that shrinks, so the offsets stay valid at every boundary.
        self._rater_index = index - (starts.unsqueeze(1) + count)
        self._rater_valid = valid
        if self._forward_count == 1:
            schedule = (
                "pending exact content geometry"
                if self._is_llava_next else str(list(self._keep_counts))
            )
            print(f"[DUET] first prefill: seq_len={input_ids.shape[1]}, visual={count} "
                  f"at [{int(starts[0])}, {int(starts[0]) + count}), "
                  f"raters={int(valid[0].sum())}, schedule={schedule}", flush=True)
        return None

    # -------------------------------------------------------- Stage 2: pruning

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
            return self._layer_hook(module, args, kwargs, layer_idx)

        return hook

    def _layer_hook(self, module: Any, args: tuple, kwargs: dict,
                    layer_idx: int) -> tuple[tuple, dict] | None:
        hidden = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            return None
        if self._is_llava_next and kwargs.get("past_key_values") is not None:
            from transformers.cache_utils import DynamicCache, DynamicLayer

            cache = kwargs["past_key_values"]
            layers = getattr(cache, "layers", ())
            if (
                not isinstance(cache, DynamicCache)
                or bool(getattr(cache, "offloading", False))
                or any(type(cache_layer) is not DynamicLayer for cache_layer in layers)
            ):
                raise NotImplementedError(
                    "DUET/LLaVA-NeXT supports Transformers DynamicCache only; static, "
                    "offloaded, quantized, and hybrid cache layouts cannot hold different "
                    "sequence lengths at decoder boundaries."
                )

        stage = self._boundary_stage.get(layer_idx)
        if stage is not None:
            if hidden.shape[1] > 1:
                if self._image_start is None:
                    return None  # no visual span this forward: nothing to prune
                hidden = self._prune(module, hidden, kwargs, stage)
            elif stage in self._stage_state:
                self._advance_decode_step(hidden, stage)

        state = self._active_state(layer_idx)
        if state is None:
            return None
        if hidden.shape[1] > 1:  # prefill: everything downstream sees the short sequence
            kwargs["position_ids"] = state["position_ids"]
            kwargs["position_embeddings"] = state["position_embeddings"]
            kwargs["attention_mask"] = state["mask"]
            if "cache_position" in kwargs:
                kwargs["cache_position"] = state["cache_position"]
        else:  # decode: only the key space is narrower than the unpruned layers'
            kwargs["attention_mask"] = decode_attention_mask(
                kwargs.get("attention_mask"), state["valid"])
            if "cache_position" in kwargs:
                kwargs["cache_position"] = state["cache_position"]
        return (hidden,), kwargs

    def _active_state(self, layer_idx: int) -> dict[str, Any] | None:
        """The most recent boundary state governing ``layer_idx``, if any cut ran."""
        for stage in range(len(self._pdrop_layers) - 1, -1, -1):
            if self._pdrop_layers[stage] <= layer_idx and stage in self._stage_state:
                return self._stage_state[stage]
        return None

    def _base_state(self, stage: int) -> dict[str, Any] | None:
        """The geometry stage ``stage`` starts from: the last earlier cut, if any."""
        for earlier in range(stage - 1, -1, -1):
            if earlier in self._stage_state:
                return self._stage_state[earlier]
        return None

    def _prune(self, layer: Any, hidden: torch.Tensor, kwargs: dict, stage: int) -> torch.Tensor:
        """Cut the visual span to this boundary's budget and size the rest of the stack.

        The decoder hands every layer the geometry it built once for the *full*
        sequence, so a later boundary must start from the previous boundary's state
        rather than from ``kwargs``.
        """
        if stage in self._stage_state:
            keep_index = self._stage_state[stage]["keep_index"]
            return hidden.gather(
                1, keep_index.unsqueeze(-1).expand(*keep_index.shape, hidden.shape[-1])
            )

        base = self._base_state(stage)
        batch, seq_len, hidden_dim = hidden.shape
        current_content = (
            self._llava_next_content_count if self._is_llava_next else self._visual_count
        )
        target = max(0, min(int(self._keep_counts[stage]), current_content))
        if target == current_content:
            return hidden  # this boundary's budget already fits; nothing to cut

        structural = self._llava_next_structural_positions
        if self._is_llava_next:
            if structural is None:
                raise RuntimeError(
                    "DUET/LLaVA-NeXT reached decoder pruning before structural newline "
                    "positions were captured from the packed AnyRes features."
                )
            if self._visual_count - int(structural.numel()) != current_content:
                raise RuntimeError(
                    "DUET/LLaVA-NeXT content/structural Stage-2 bookkeeping diverged"
                )

        start = self._image_start.to(hidden.device)
        columns = start.unsqueeze(1) + torch.arange(self._visual_count, device=hidden.device)
        if start.shape[0] != batch or int(columns.max()) >= seq_len:
            self._warn_once(
                "geometry_changed",
                f"the visual span {tuple(columns.shape)} does not fit the layer-"
                f"{self._pdrop_layers[stage]} input {tuple(hidden.shape)}; Stage 2 is SKIPPED "
                f"for this batch.",
            )
            return hidden

        base_ids = base["position_ids"] if base is not None else kwargs.get("position_ids")
        base_embeddings = base["position_embeddings"] if base is not None \
            else kwargs.get("position_embeddings")
        base_mask = base["mask"] if base is not None else kwargs.get("attention_mask")

        # Everything from here to `keep_index` builds *indices* -- topk, a scatter
        # into a bool mask, nonzero -- and the geometry stored beside them. None of
        # it carries a gradient, but `_boundary_scores` runs a real attention
        # forward, so with grad enabled this pass saves tensors that the cached
        # replay above does not. Under gradient checkpointing the backward re-runs
        # the segment, takes the replay branch, and torch refuses the mismatch:
        # "A different number of tensors was saved during the original forward and
        # recomputation." Deciding the cut under no_grad makes both branches the
        # single `gather` below, which is the only line that moves activations.
        with torch.no_grad():
            if self._is_llava_next:
                scores = self._boundary_scores(layer, hidden, base_embeddings, base_mask, start)
                structural = structural.to(hidden.device)
                content_mask = torch.ones(
                    self._visual_count, dtype=torch.bool, device=hidden.device
                )
                content_mask[structural] = False
                content_positions = content_mask.nonzero(as_tuple=True)[0]
                if target:
                    ranked = scores[:, content_positions].topk(target, dim=1).indices
                    selected_content = content_positions[ranked]
                else:
                    selected_content = torch.empty(
                        batch, 0, dtype=torch.long, device=hidden.device
                    )
                # Structural rows are never candidates: keep every newline and
                # restore original packed order around the selected content.
                local = torch.cat(
                    (selected_content, structural.unsqueeze(0).expand(batch, -1)), dim=1
                ).sort(dim=1).values
                kept_columns = columns.gather(1, local)
            elif target > 0:
                scores = self._boundary_scores(layer, hidden, base_embeddings, base_mask, start)
                local = scores.topk(target, dim=1).indices.sort(dim=1).values  # [B, target]
                kept_columns = columns.gather(1, local)
            else:
                local = torch.zeros(batch, 0, dtype=torch.long, device=hidden.device)
                kept_columns = local
            keep_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden.device)
            keep_mask.scatter_(1, columns, False)  # drop the whole visual span ...
            if local.shape[1] > 0:
                keep_mask.scatter_(1, kept_columns, True)  # ... then put the winners back
            keep_index = keep_mask.nonzero(as_tuple=True)[1].view(batch, -1)  # ascending per row
            new_len = keep_index.shape[1]

            # Survivors keep their ORIGINAL absolute positions, so RoPE offsets across
            # the dropped span -- and against the cached keys while decoding -- hold.
            if isinstance(base_ids, torch.Tensor) and base_ids.ndim == 2 \
                    and base_ids.shape[-1] == seq_len:
                position_ids = base_ids.expand(batch, seq_len).gather(1, keep_index)
            else:
                position_ids = keep_index
            # `rotary_emb` reads only dtype/device/shape off its first argument, so
            # the detached survivors serve; the differentiable gather is below.
            position_embeddings = self._language_model.rotary_emb(
                hidden.detach().gather(
                    1, keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim)),
                position_ids=position_ids,
            )
            mask, valid = pruned_attention_mask(base_mask, keep_index, seq_len)

        new_hidden = hidden.gather(1, keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim))

        self._stage_state[stage] = {
            "batch": batch,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "mask": mask,
            "valid": valid,
            # Layers at or past this boundary cache only the survivors, so their
            # cache positions are compact; the absolute ids above are RoPE's alone.
            "cache_position": torch.arange(new_len, device=hidden.device),
            "kv_len": new_len,
            "keep_index": keep_index,
        }
        self._final_keep_index = (
            keep_index
            if self._final_keep_index is None
            else self._final_keep_index.gather(1, keep_index)
        )
        if self._forward_count == 1:
            print(f"[DUET] stage {stage} @layer{self._pdrop_layers[stage]}: seq {seq_len} -> "
                  f"{new_len} (visual {self._visual_count} -> {local.shape[1]}, "
                  f"content {current_content} -> {target})", flush=True)
        if self._is_llava_next:
            structural_mask = torch.isin(local[0], structural)
            self._llava_next_structural_positions = structural_mask.nonzero(
                as_tuple=True
            )[0].detach()
            self._llava_next_content_count = target
            self._llava_next_run_metadata[f"stage_{stage}_content_tokens"] = target
            self._llava_next_run_metadata[f"stage_{stage}_structural_tokens"] = int(
                structural.numel()
            )
            self._llava_next_run_metadata[f"stage_{stage}_total_visual_tokens"] = int(
                local.shape[1]
            )
        self._visual_count = int(local.shape[1])
        self._record_cut(local)
        return new_hidden

    def _record_cut(self, local: torch.Tensor) -> None:
        """Narrow the composed source attribution to the survivors of one cut."""
        if not self._track_source or self._composed_source is None or local.shape[1] == 0:
            return
        self._composed_source = self._composed_source[local[0].detach().cpu()]
        self._source_history.append(self._composed_source)

    def _boundary_scores(self, layer: Any, hidden: torch.Tensor,
                         position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
                         attention_mask: Any, start: torch.Tensor) -> torch.Tensor:
        """Rate the visual tokens by the raters' attention at this boundary layer."""
        raters = self._rater_columns(hidden.device)  # [B, R] into the current sequence
        attention = boundary_attention_scores(
            layer, hidden, position_embeddings, attention_mask, raters, apply_rope=_APPLY_ROPE)
        return reduce_rater_scores(attention, self._rater_valid, start, self._visual_count,
                                   reduction=self._score_reduction)

    def _rater_columns(self, device: torch.device) -> torch.Tensor:
        """Current-sequence columns of the raters, following the shrinking span."""
        start = self._image_start.to(device)
        return self._rater_index.to(device) + (start.unsqueeze(1) + self._visual_count)

    # --------------------------------------------------------- Stage 2: decoding

    def _advance_decode_step(self, hidden: torch.Tensor, stage: int) -> None:
        """Extend one stage's pruned key space by the token generated this step."""
        state = self._stage_state[stage]
        if hidden.shape[0] != state["batch"]:
            raise RuntimeError(
                f"DUET cut the cache for batch {state['batch']} but decoding now runs batch "
                f"{hidden.shape[0]}. Beam search expands the batch after prefill and is not "
                f"supported; use greedy or sampling decoding."
            )
        column = torch.ones(state["batch"], 1, dtype=torch.bool, device=state["valid"].device)
        state["valid"] = torch.cat([state["valid"], column], dim=1)
        state["kv_len"] += 1
        state["cache_position"] = torch.tensor(
            [state["kv_len"] - 1], device=hidden.device, dtype=torch.long)
