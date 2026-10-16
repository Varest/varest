"""QuietPrune: query-guided early visual-token pruning inside the vision encoder.

Paper: Gao et al., "QuietPrune: Query-Guided Early Token Pruning for
Vision-Language Models" (CVPR 2026). Ported here from the
``llmoe_baseline_quietprune`` sidecar that reimplemented the paper against
``llava-hf/llava-1.5-7b-hf``.

**The paper does not evaluate LLaVA.** Its models are the Qwen3-VL and InternVL3
series (InternVL3-1B, Qwen3-VL-4B, InternVL3-8B), and the pruning geometry is
built *around* those backbones: Sec. 4.2 motivates the ``2x2`` group as the unit
precisely because "adjacent 2x2 tokens are merged into a single token for the LLM"
and unstructured pruning "breaks the local structural prior assumed by the merging
operation". So this LLaVA path is a port to a backbone the method was never
measured on. It is mechanically faithful -- pruning at ``1/4, 1/2, 3/4`` of ViT
depth, a fraction ``R`` of tokens dropped per layer, the adapter initialised by
inverting the projector in reverse module order, group relevance by averaging --
but it is not the paper's setting and its numbers are not the paper's numbers.

Every other pruning baseline in this package scores visual tokens *after* the
vision tower has finished -- either at the projector boundary or inside the
decoder. QuietPrune's claim is that the useful signal is available much earlier,
and that acting on it early is what actually saves compute: the ViT layers after
a cut never see the dropped tokens at all. Four pieces make that possible:

* **Text-to-vision adapter (Sec. 4.1).** The pretrained vision-to-text projector
  is approximately *inverted* -- ``W* = pinv(W)``, ``b* = -pinv(W) b`` per linear
  layer, ``gamma* = 1/gamma``, ``beta* = -beta/gamma`` per LayerNorm, activations
  kept as-is -- giving a map from LLM text embeddings back into CLIP's visual
  space that already starts near-correct before a single gradient step. This is
  the only trained part of QuietPrune (see :mod:`llmoe.compressor.quietprune_tr`).
* **``[Q-CLS]`` injection (Sec. 4.2).** The prompt's token embeddings are pushed
  through the adapter and mean-pooled into one token, prepended to the ViT
  sequence *after* patch embedding, so it carries no position embedding and never
  reaches the language model. The sequence the encoder sees is
  ``[Q-CLS], [CLIP-CLS], patch_1 ... patch_N, ([AGG])``.
* **Semi-structured 2x2 group pruning (Sec. 4.2).** At 1/4, 1/2, and 3/4 of ViT
  depth (layers 6, 12, 18 of LLaVA-1.5's 24-layer CLIP tower) the mean-over-heads
  ``[Q-CLS] -> patch`` attention row scores every surviving patch. Scores are
  averaged within 2x2 spatial groups and whole groups are dropped, which keeps
  local spatial structure intact for the 2x2 token merging that recent VLMs run
  before the LLM.
* **Redundant-token aggregation (Sec. 4.3).** The dropped tokens are collapsed
  into a single relevance-weighted token ``x_m = sum_i a_i x_i / sum_i a_i`` that
  rides along in the sequence and is handed to the LLM, so discarded context is
  attenuated rather than erased.

Budget semantics
----------------
QuietPrune's knob is ``R``, the *per-pruning-layer* group-drop rate, not a final
token count. The supplement tabulates the mapping from a target average LLM-stage
pruning rate ``R*`` to ``R`` (:data:`AVERAGE_TO_LAYER_PRUNING_RATE`); the paper
trains the adapter at ``R = 25%`` and varies ``R`` freely at evaluation, since
nothing about the adapter is budget-specific.

Because the group geometry is fixed and the drop count is ``floor(G * R)``, the
final token count is **deterministic** -- it depends on the patch grid and ``R``,
never on image content. That is what lets this compressor trim ``<image>``
placeholders ahead of the forward instead of resorting to an ``inputs_embeds``
splice. It also means a fixed budget can be requested directly: pass
``max_tokens`` and the gentlest ``R`` that fits it is solved for, which is how the
other fixed-budget baselines here are configured. For LLaVA-1.5's 24x24 grid::

    R*     R       patches after layers 6 / 12 / 18   tokens to the LLM
    0.20   0.585   240 / 100 / 44                     45
    0.30   0.669   192 / 64 / 24                      25
    0.40   0.737   152 / 40 / 12                      13
    0.50   0.794   120 / 28 / 8                       9
    0.60   0.843   92 / 16 / 4                        5
    0.70   0.888   68 / 8 / 4                         5
    0.80   0.928   44 / 4 / 4                         5

The trailing ``+1`` is the aggregate token. The paper's ``R*`` targets were
measured on Qwen3-VL/InternVL3, whose token flow differs from LLaVA-1.5's, so
these are the honest LLaVA counts rather than the tabulated ones -- prefer
``max_tokens`` when comparing against the fixed-budget baselines here.

Faithfulness notes / harness adaptations
----------------------------------------
* **Where the work happens.** ``install()`` replaces ``LlavaModel``'s
  ``get_image_features`` with a manual CLIP forward that injects ``[Q-CLS]``,
  prunes inside the encoder, and projects the survivors. ``compress()`` exposes
  one group-pruning step on plain tensors so the operation is testable in
  isolation, but it needs a relevance vector: without a query there is no query
  guidance, so it raises rather than inventing a score.

* **Attention without an eager backend.** The sidecar forced
  ``attn_implementation="eager"`` to read attention probabilities out of the CLIP
  layer. SDPA returns none, and transformers >= 5 drops ``output_attentions``
  from ``CLIPEncoderLayer`` entirely. Only the single ``[Q-CLS]`` row is needed,
  so this port recomputes it directly from the layer's own ``q_proj``/``k_proj``
  on the layer's own ``layer_norm1`` output -- numerically the same row, at three
  layers, with the fast backend left in place for everything else.

* **Batched, not one image at a time.** The sidecar's vision forward was batch-1.
  Since the per-layer surviving-group count is identical across a batch, this port
  keeps the pruning state as a dense ``[B, num_groups]`` mask and prunes the whole
  batch in one pass, which is what makes the harness's default ``batch_size: 16``
  usable.

* **Placeholders instead of an embedding splice.** The sidecar bypassed
  ``LlavaForConditionalGeneration`` and spliced pruned features into
  ``inputs_embeds`` itself, because it wanted content-dependent counts and
  multi-image rows. The harness drives ``generate()``, so this port trims each
  ``<image>`` placeholder run to the planned count in ``prepare_inputs`` and lets
  transformers do its own scatter -- the same result for the one-image-per-prompt
  rows every benchmark here produces, and it leaves the KV cache path untouched.

* **Backbone boundary.** The released Q-CLS path remains LLaVA-1.5-only: its
  injection point, CLS-then-patches layout, and invertible projector are CLIP-ViT
  assumptions. Qwen2.5-VL therefore uses a clearly labelled, training-free
  adaptation: VisionZip-style received attention from the final vision block
  ranks the same spatial groups, and reduction happens over the concatenated
  post-merger span. LLaVA adapter checkpoints are rejected on that path.

* **Source history.** ``get_source_history()`` returns a leading identity plus one
  ``[T_cur, T_orig]`` matrix per pruning layer. Retained rows are one-hot; the
  aggregate row carries the normalised relevance mass of everything it absorbed,
  so ``llmoe.viz`` renders QuietPrune as the prune-plus-merge that it is.

Configure and run::

    compressor_args:
      target_average_pruning_rate: 0.5     # or: pruning_rate: 0.794 / max_tokens: 64
      checkpoint_path: /results/quietprune/quietprune.pt

    llmoe --config local --compressor QuietPrune --custom_args quietprune.yaml bm mme

Running without ``checkpoint_path`` uses the inverse-initialised adapter, which is
the paper's *starting point*, not its result -- ``install()`` says so once.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from math import floor, isqrt
from os import PathLike
from typing import Any, override
import warnings

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from llmoe.backbone.prototype import VLM
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor, VisualizationView
from llmoe.weights import backbone_key, resolve_weights


# Supplement Table S1: target average LLM-stage pruning rate R* -> per-pruning-layer
# ViT group-drop rate R. Keys are the only R* values the paper tabulates.
AVERAGE_TO_LAYER_PRUNING_RATE: dict[float, float] = {
    0.20: 0.585,
    0.30: 0.669,
    0.40: 0.737,
    0.50: 0.794,
    0.60: 0.843,
    0.70: 0.888,
    0.80: 0.928,
}
# Sec. 4.1: the adapter is distilled at a mild rate and evaluated at any rate.
TRAINING_PRUNING_RATE: float = 0.25
# Activations the inverse projector reproduces verbatim (Sec. 4.1: GELU is close
# enough to the identity on the positive half that the paper keeps it un-inverted).
_ACTIVATION_NAMES = frozenset({"GELUActivation", "FastGELUActivation", "QuickGELUActivation"})


def infer_patch_grid(num_patches: int, grid: int | tuple[int, int] | None = None) -> tuple[int, int]:
    """Infer or validate the ``(height, width)`` patch grid behind ``num_patches``."""
    if num_patches < 1:
        raise ValueError("num_patches must be positive")
    if isinstance(grid, tuple):
        height, width = int(grid[0]), int(grid[1])
        if height * width != num_patches:
            raise ValueError(f"grid {grid} does not match {num_patches} patches")
        return height, width
    if isinstance(grid, int):
        if grid < 1 or num_patches % grid != 0:
            raise ValueError(f"grid side {grid} is invalid for {num_patches} patches")
        return grid, num_patches // grid
    side = isqrt(num_patches)
    if side * side != num_patches:
        raise ValueError(f"Cannot infer a square grid from {num_patches} patches")
    return side, side


# --------------------------------------------------------------------- adapter


def _invert_linear(linear: nn.Linear) -> nn.Linear:
    """``pinv``-initialise the layer that undoes ``y = x @ W.T + b`` (Sec. 4.1).

    PyTorch stores ``[out_features, in_features]`` and applies ``x @ W.T``, so the
    inverse layer maps ``out_features -> in_features`` and holds ``pinv(W)``
    directly. The pseudo-inverse is taken in fp32 whatever the projector's dtype:
    it is an SVD, and fp16 loses exactly the small singular values that decide how
    much of the text space survives the round trip.
    """
    inverse = nn.Linear(linear.out_features, linear.in_features, bias=linear.bias is not None)
    with torch.no_grad():
        weight = linear.weight.detach().float().cpu()
        inverse_weight = torch.linalg.pinv(weight)
        inverse.weight.copy_(inverse_weight)
        if linear.bias is not None and inverse.bias is not None:
            inverse.bias.copy_(-(inverse_weight @ linear.bias.detach().float().cpu()))
    return inverse


def _invert_layer_norm(layer_norm: nn.LayerNorm, *, eps: float = 1e-6) -> nn.LayerNorm:
    """``gamma* = 1/gamma``, ``beta* = -beta/gamma`` -- the paper's approximate inverse.

    Only the affine half is inverted; re-inflating the normalisation itself would
    need the per-token statistics, which are gone by then.
    """
    inverse = nn.LayerNorm(layer_norm.normalized_shape, eps=layer_norm.eps,
                           elementwise_affine=layer_norm.elementwise_affine)
    if layer_norm.elementwise_affine:
        with torch.no_grad():
            gamma = layer_norm.weight.detach().float().cpu()
            beta = layer_norm.bias.detach().float().cpu()
            # Keep the sign so a near-zero gamma is floored, never flipped.
            signed = gamma.sign().masked_fill(gamma == 0, 1.0) * gamma.abs().clamp_min(eps)
            inverse.weight.copy_(1.0 / signed)
            inverse.bias.copy_(-beta / signed)
    return inverse


def _leaf_modules(module: nn.Module) -> list[nn.Module]:
    children = list(module.children())
    if not children:
        return [module]
    leaves: list[nn.Module] = []
    for child in children:
        leaves.extend(_leaf_modules(child))
    return leaves


def _copy_activation(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.GELU):
        return nn.GELU(approximate=module.approximate)
    name = module.__class__.__name__
    if name == "FastGELUActivation":
        return nn.GELU(approximate="tanh")
    if name in _ACTIVATION_NAMES:
        return nn.GELU()
    if isinstance(module, nn.SiLU):
        return nn.SiLU(inplace=False)
    if isinstance(module, nn.ReLU):
        return nn.ReLU(inplace=False)
    return module.__class__()


def build_inverse_projector(projector: nn.Module) -> nn.Sequential:
    """Reverse a vision-to-text projector into a text-to-vision body (Sec. 4.1).

    Supported leaves are ``Linear``, ``LayerNorm``, the usual GELU/SiLU/ReLU
    activation wrappers, and pass-through ``Dropout``/``Identity``. LLaVA-1.5's
    ``Linear -> GELU -> Linear`` is the intended case; anything else raises, since
    a silently un-inverted leaf would leave the adapter initialised at noise and
    QuietPrune's whole training-efficiency claim rests on that initialisation.
    """
    modules: list[nn.Module] = []
    for module in reversed(_leaf_modules(projector)):
        if isinstance(module, nn.Linear):
            modules.append(_invert_linear(module))
        elif isinstance(module, nn.LayerNorm):
            modules.append(_invert_layer_norm(module))
        elif isinstance(module, (nn.GELU, nn.SiLU, nn.ReLU)) or module.__class__.__name__ in _ACTIVATION_NAMES:
            modules.append(_copy_activation(module))
        elif isinstance(module, (nn.Dropout, nn.Identity)):
            modules.append(nn.Identity())
        else:
            raise TypeError(
                f"Cannot invert projector leaf "
                f"{module.__class__.__module__}.{module.__class__.__name__}; QuietPrune's adapter "
                f"is the projector run backwards and has no initialisation for it."
            )
    if not modules:
        raise TypeError("Projector contains no invertible leaf modules")
    return nn.Sequential(*modules)


class QuietPruneAdapter(nn.Module):
    """The text-to-vision adapter and its ``[Q-CLS]`` pooling (Sec. 4.1 / 3.3).

    The body is the inverted projector; the forward maps prompt token embeddings
    into CLIP's hidden space and mean-pools them, masked so padding and the
    ``<image>`` placeholders themselves contribute nothing. This is QuietPrune's
    only trainable module.
    """

    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        self.body: nn.Module = body

    @classmethod
    def from_projector(cls, projector: nn.Module) -> "QuietPruneAdapter":
        return cls(build_inverse_projector(projector))

    @override
    def forward(self, query_embeds: torch.Tensor, query_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``[B, T, d_text] -> [B, 1, d_vision]``.

        :param query_embeds: Prompt token embeddings from the LLM embedding table.
        :param query_mask: ``[B, T]`` weights; zero out padding and image
            placeholders. ``None`` means every position counts.
        """
        if query_embeds.ndim != 3:
            raise ValueError(f"query_embeds must be [B, T, d], got {tuple(query_embeds.shape)}")
        adapted = self.body(query_embeds)
        if query_mask is None:
            return adapted.mean(dim=1, keepdim=True)
        if tuple(query_mask.shape) != tuple(query_embeds.shape[:2]):
            raise ValueError("query_mask must be [B, T] and match query_embeds")
        weights = query_mask.to(device=adapted.device, dtype=adapted.dtype).unsqueeze(-1)
        # clamp_min(1) rather than eps: an all-masked row is a degenerate prompt,
        # and a zero [Q-CLS] is a defensible neutral query, not a division blow-up.
        return (adapted * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)


# ------------------------------------------------------------------- grid plan


@dataclass
class QuietPrunePlan:
    """The deterministic token schedule for one patch grid and one rate.

    QuietPrune drops ``floor(G * R)`` whole groups per pruning layer, so both the
    geometry and the counts are settled before any image is seen. Everything the
    installed compressor needs to trim placeholders ahead of the forward, and
    everything ``name`` reports, comes from here.
    """

    grid: tuple[int, int]
    group_size: int
    pruning_layers: tuple[int, ...]
    pruning_rate: float
    keep_border_tokens: bool
    aggregate: bool
    include_cls: bool
    num_patches: int
    num_groups: int
    num_border: int
    group_members: torch.Tensor  # [num_groups, group_size ** 2] original patch indices
    group_of_patch: torch.Tensor  # [num_patches] -> group id; border patches -> num_groups
    groups_after_layer: tuple[int, ...] = ()
    patches_after_layer: tuple[int, ...] = ()
    final_tokens: int = 0

    def to(self, device: torch.device) -> "QuietPrunePlan":
        self.group_members = self.group_members.to(device)
        self.group_of_patch = self.group_of_patch.to(device)
        return self


def _groups_kept(num_groups: int, rate: float) -> int:
    """Groups surviving one pruning layer; at least one always does."""
    pruned = min(num_groups - 1, max(0, floor(num_groups * rate)))
    return num_groups - pruned


def build_plan(grid: tuple[int, int], *, group_size: int, pruning_layers: Sequence[int], pruning_rate: float,
               keep_border_tokens: bool, aggregate: bool, include_cls: bool) -> QuietPrunePlan:
    """Lay out the 2x2 groups over ``grid`` and walk the drop schedule."""
    height, width = grid
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    if not 0.0 <= pruning_rate < 1.0:
        raise ValueError("pruning_rate must be in [0, 1)")
    full_height, full_width = height - height % group_size, width - width % group_size
    members = [
        [(top + dy) * width + left + dx for dy in range(group_size) for dx in range(group_size)]
        for top in range(0, full_height, group_size)
        for left in range(0, full_width, group_size)
    ]
    if not members:
        raise ValueError(f"A {height}x{width} grid holds no complete {group_size}x{group_size} group")
    group_members = torch.tensor(members, dtype=torch.long)
    num_groups, num_patches = len(members), height * width
    # Border patches (a grid side not divisible by group_size) belong to no group.
    # Slot `num_groups` is the sentinel their lookup lands on: kept whenever
    # keep_border_tokens, and never a pruning candidate either way.
    group_of_patch = torch.full((num_patches,), num_groups, dtype=torch.long)
    group_of_patch[group_members.reshape(-1)] = torch.arange(num_groups).repeat_interleave(group_size ** 2)
    num_border = num_patches - num_groups * group_size ** 2

    border = num_border if keep_border_tokens else 0
    groups_after: list[int] = []
    patches_after: list[int] = []
    current_groups = num_groups
    for _ in pruning_layers:
        current_groups = _groups_kept(current_groups, pruning_rate)
        groups_after.append(current_groups)
        patches_after.append(current_groups * group_size ** 2 + border)
    retained = patches_after[-1] if patches_after else num_patches
    pruned_anything = retained != num_patches
    final = retained + (1 if aggregate and pruned_anything else 0) + (1 if include_cls else 0)
    return QuietPrunePlan(
        grid=(height, width), group_size=group_size, pruning_layers=tuple(pruning_layers),
        pruning_rate=float(pruning_rate), keep_border_tokens=keep_border_tokens, aggregate=aggregate,
        include_cls=include_cls, num_patches=num_patches, num_groups=num_groups, num_border=num_border,
        group_members=group_members, group_of_patch=group_of_patch, groups_after_layer=tuple(groups_after),
        patches_after_layer=tuple(patches_after), final_tokens=final,
    )


# ------------------------------------------------------------------ compressor


class QuietPruneCompressor(TokenCompressor):
    """Query-guided 2x2 group pruning inside supported LLaVA CLIP towers.

    :param max_tokens: Token budget handed to the LLM. When given, the gentlest
        per-layer rate whose schedule fits the budget is solved for, and
        ``pruning_rate``/``target_average_pruning_rate`` are ignored. This is how
        QuietPrune is compared against the fixed-budget baselines here.
    :param retain_ratio: Per-image fraction of the original compressible visual
        content span. Required by LLaVA-NeXT evaluation and also supported by
        the Qwen2.5-VL adaptation; structural separators are excluded.
    :param pruning_rate: Per-pruning-layer group-drop rate ``R``, used directly.
    :param target_average_pruning_rate: Paper-tabulated ``R*`` for the legacy
        LLaVA-1.5 path. NeXT evaluation accepts only ``retain_ratio``; its
        registered training path retains the canonical per-stage rate.
    :param pruning_layers: Zero-based CLIP encoder layers to prune after. Defaults
        to 1/4, 1/2, and 3/4 of the tower's depth -- ``(6, 12, 18)`` for LLaVA-1.5.
    :param group_size: Spatial group side; the paper's 2x2 is the default.
    :param keep_border_tokens: Protect patches outside the complete-group region of
        a grid whose side is not divisible by ``group_size``. A no-op on LLaVA-1.5's
        24x24 grid. When false those patches are dropped outright and, matching the
        reference, are not folded into the aggregate either.
    :param aggregate_pruned_tokens: Append the relevance-weighted aggregate of the
        dropped tokens (Sec. 4.3). Turning it off is the paper's ablation.
    :param aggregate_mode: ``"rolling"`` keeps the aggregate in the ViT sequence so
        later layers attend to it, folding each new prune into it; ``"final"``
        accumulates it outside the sequence and appends it once at the end.
    :param normalize_aggregation: Divide by the relevance mass. ``False`` -- the
        default -- is the paper's formula, ``x_m = sum_i a_i x_i`` (Sec. 4.3), where
        the aggregate's magnitude carries how much relevance mass was dropped.
        ``True`` makes it a convex combination instead, which is numerically
        steadier but is *not* the published method.

        .. note::
           The released adapter was *already* distilled with
           ``normalize_aggregation=False``, so this default aligns the code with
           the checkpoint rather than diverging from it -- the previous ``True``
           default was the mismatch. The bucket copy has since been replaced by
           one retrained under today's defaults, which also closes a second gap:
           the earlier adapter used ``aggregate_mode="final"`` while the default
           is now ``"rolling"``.
    :param adapter_dtype: dtype for the adapter's parameters. Defaults to the
        projector's; the trainer pins it to fp32.
    :param checkpoint_path: An adapter checkpoint from ``tr``/``ft``. A relative
        path is read against the ``llmoe_weights`` bundle. Left unset, the released
        adapter for the selected backbone is resolved -- one is registered for
        LLaVA-1.5-7B, 13B, Qwen-2.5-VL and LLaVA-NeXT-8B. With ``pretrained=false``
        the adapter stays at inverse initialization, an ablation.
    :param pretrained: ``False`` asks for that inverse initialisation explicitly;
        ``True`` (the default in effect) takes the released adapter, downloading
        it on demand. ``checkpoint_path`` overrides both.
    :param eps: Numerical floor for the aggregation weights.
    """

    def __init__(self, max_tokens: int | None = None, *, retain_ratio: float | None = None,
                 pruning_rate: float | None = None,
                 target_average_pruning_rate: float | None = None, pruning_layers: Sequence[int] | None = None,
                 group_size: int = 2, keep_border_tokens: bool = True, aggregate_pruned_tokens: bool = True,
                 aggregate_mode: str = "rolling", normalize_aggregation: bool = False,
                 query_guided: bool = True, track_source: bool = False,
                 adapter_dtype: str | torch.dtype | None = None,
                 checkpoint_path: str | PathLike[str] | None = None,
                 pretrained: bool | str | PathLike[str] | None = None, eps: float = 1e-6,
                 attention_chunk: int = 512) -> None:
        super().__init__(int(max_tokens) if max_tokens is not None else 0)
        if max_tokens is not None and int(max_tokens) < 1:
            raise ValueError("max_tokens must be >= 1")
        if pruning_rate is not None and not 0.0 <= float(pruning_rate) < 1.0:
            raise ValueError("pruning_rate must be in [0, 1)")
        if aggregate_mode not in ("rolling", "final"):
            raise ValueError(f"aggregate_mode must be 'rolling' or 'final', got {aggregate_mode!r}")
        if int(group_size) < 1:
            raise ValueError("group_size must be >= 1")
        if float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        if retain_ratio is not None and not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
        if retain_ratio is not None and any(
            value is not None
            for value in (
                max_tokens,
                pruning_rate,
                target_average_pruning_rate,
                pruning_layers,
            )
        ):
            raise ValueError(
                "retain_ratio cannot be combined with max_tokens, pruning_rate, "
                "target_average_pruning_rate, or custom pruning_layers."
            )
        if int(attention_chunk) < 1:
            raise ValueError("attention_chunk must be >= 1")

        self._retain_ratio: float | None = (
            None if retain_ratio is None else float(retain_ratio)
        )
        self._budget: int | None = int(max_tokens) if max_tokens is not None else None
        self._pruning_rate: float | None = None if pruning_rate is None else float(pruning_rate)
        self._target_average: float | None = (
            None
            if self._retain_ratio is not None
            else self._resolve_target_average(
                pruning_rate, target_average_pruning_rate, max_tokens
            )
        )
        self._pruning_layers: tuple[int, ...] | None = (
            None if pruning_layers is None else tuple(sorted({int(layer) for layer in pruning_layers}))
        )
        if self._pruning_layers is not None and any(layer < 0 for layer in self._pruning_layers):
            raise ValueError("pruning_layers must be non-negative")
        self._group_size: int = int(group_size)
        self._keep_border_tokens: bool = bool(keep_border_tokens)
        self._aggregate: bool = bool(aggregate_pruned_tokens)
        self._aggregate_mode: str = aggregate_mode
        self._track_source: bool = bool(track_source)
        self._normalize_aggregation: bool = bool(normalize_aggregation)
        self._adapter_dtype: torch.dtype | None = self._resolve_dtype(adapter_dtype)
        # Budget-independent: one released adapter covers every `R`, so the
        # budget passed here only has to be a number.
        self._checkpoint_explicit: bool = (
            checkpoint_path is not None or isinstance(pretrained, (str, PathLike)))
        # Resolved in `install()`, not here: QuietPrune's adapter differs per
        # backbone (each inverts a different projector), so the released artifact
        # cannot be chosen until the backbone is known -- and it is not, at
        # construction time.
        self._checkpoint_arg: Any = checkpoint_path
        self._pretrained_arg: Any = pretrained
        self._checkpoint_path: str | PathLike[str] | None = None
        self._eps: float = float(eps)
        self._attention_chunk: int = int(attention_chunk)
        # Set only by the registered tr/ft entry point. Public NeXT evaluation
        # must use retain_ratio so a direct stage rate cannot bypass the native
        # per-image content-budget convention.
        self._next_training_rate_mode: bool = False

        self.adapter: QuietPruneAdapter | None = None
        self._plan: QuietPrunePlan | None = None
        self._bypass: bool = False
        # Installation state, all restored by uninstall(). The backbone's modules
        # live in a plain dict on purpose: assigning them to attributes would make
        # `nn.Module.__setattr__` register the whole vision tower as a child of the
        # compressor, so `parameters()` -- what the trainer optimises and clips --
        # would sweep up the frozen VLM, and `train()` would flip its mode.
        self._backbone: dict[str, nn.Module] = {}
        self._vlm: VLM | None = None
        self._selected_layer_idx: int = -1
        self._select_strategy: str = "default"
        self._image_token_id: int | None = None
        self._pad_token_id: int | None = None
        self._latest_input_ids: torch.Tensor | None = None
        self._latest_attention_mask: torch.Tensor | None = None
        self._latest_labels: torch.Tensor | None = None
        self._pending_training_labels: torch.Tensor | None = None
        self._hooks: list[Any] = []
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._qwen: Any | None = None
        # Qwen only: run the paper's in-ViT [Q-CLS] schedule rather than the
        # training-free post-merger adaptation.
        self._query_guided: bool = bool(query_guided)
        self._qwen_install: Any | None = None
        self._backend: str | None = None
        self._next_adapter: Any | None = None
        self._last_budget_report: dict[str, Any] | None = None
        self._training_metadata: dict[str, Any] = {}

    # ------------------------------------------------------------- construction

    @staticmethod
    def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype | None:
        if dtype is None or isinstance(dtype, torch.dtype):
            return dtype
        resolved = getattr(torch, str(dtype), None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unsupported adapter_dtype {dtype!r}")
        return resolved

    @staticmethod
    def _resolve_target_average(pruning_rate: float | None, target_average: float | None,
                                max_tokens: int | None) -> float | None:
        """Validate ``R*``, and apply the 0.50 default only when it is the live knob."""
        if target_average is None:
            return None if (pruning_rate is not None or max_tokens is not None) else 0.50
        key = round(float(target_average), 2)
        if key not in AVERAGE_TO_LAYER_PRUNING_RATE:
            supported = ", ".join(f"{rate:.2f}" for rate in AVERAGE_TO_LAYER_PRUNING_RATE)
            raise ValueError(
                f"Unsupported target_average_pruning_rate={key:.2f}; the paper tabulates {supported}. "
                f"Pass pruning_rate to set R directly, or max_tokens to solve for it."
            )
        return key

    @property
    def pruning_rate(self) -> float:
        """The per-layer group-drop rate ``R`` in force, once resolved."""
        if self._plan is not None:
            return self._plan.pruning_rate
        if self._pruning_rate is not None:
            return self._pruning_rate
        if self._target_average is not None:
            return AVERAGE_TO_LAYER_PRUNING_RATE[self._target_average]
        raise RuntimeError("R is solved from max_tokens at install(); this compressor has none yet")

    @property
    def plan(self) -> QuietPrunePlan:
        if self._plan is None:
            raise RuntimeError("QuietPruneCompressor has no plan until install()")
        return self._plan

    @property
    def _vision_model(self) -> nn.Module:
        return self._backbone["vision_model"]

    @property
    def _projector(self) -> nn.Module:
        return self._backbone["projector"]

    @property
    @override
    def budget(self) -> int | float | str:
        """Final token count once ``install()`` has solved the plan; before that the
        requested count, or the group-drop rate ``R`` when the compressor is
        rate-driven and has no count at all.
        """
        if self._retain_ratio is not None:
            return self._retain_ratio
        if self._backend == "llava_next":
            return f"R{self.pruning_rate:.6g}"
        if self._plan is not None:
            return self._plan.final_tokens
        if self._budget is not None:
            return self._budget
        if self._pruning_rate is not None:
            return self._pruning_rate
        return self._target_average if self._target_average is not None else 0

    @staticmethod
    def _default_pruning_layers(num_layers: int) -> tuple[int, ...]:
        """1/4, 1/2, 3/4 of ViT depth -- ``(6, 12, 18)`` for a 24-layer tower."""
        if num_layers < 4:
            raise ValueError("QuietPrune needs at least 4 vision layers")
        return tuple(sorted({max(0, min(num_layers - 1, int(num_layers * f))) for f in (0.25, 0.50, 0.75)}))

    def _solve_rate_for_budget(self, budget: int, grid: tuple[int, int], pruning_layers: Sequence[int],
                               include_cls: bool) -> float:
        """Gentlest ``R`` whose schedule fits ``budget`` tokens.

        The final count is a non-increasing step function of ``R``, so sweeping a
        milli-rate ladder upwards and stopping at the first fit gives the mildest
        rate that still meets the budget -- every group kept is context kept.
        """
        for step in range(1000):
            rate = step / 1000.0
            plan = build_plan(grid, group_size=self._group_size, pruning_layers=pruning_layers,
                              pruning_rate=rate, keep_border_tokens=self._keep_border_tokens,
                              aggregate=self._aggregate, include_cls=include_cls)
            if plan.final_tokens <= budget:
                return rate
        raise ValueError(
            f"No per-layer rate over {len(tuple(pruning_layers))} pruning layers reaches "
            f"max_tokens={budget} on a {grid[0]}x{grid[1]} grid; the floor is one surviving group. "
            f"Add a pruning layer or raise the budget."
        )

    # ------------------------------------------------- tensor-level compression

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, relevance: torch.Tensor | None = None,
                 grid: int | tuple[int, int] | None = None, pruning_rate: float | None = None,
                 **kwargs: Any) -> CompressedFeatures:
        """Run one 2x2 group-pruning step on a plain patch-token tensor.

        This is the paper's core operation in isolation -- a single pruning layer
        on the full grid -- not the installed pipeline, which chains it three times
        inside the encoder. Real inference goes through :meth:`install`.

        :param features: ``[B, N, D]`` or ``[N, D]`` patch tokens in raster order.
        :param attention: ``[B, H, T, T]`` attention probabilities from the layer,
            whose ``[Q-CLS]`` row (index 0) is averaged over heads; the trailing
            ``N`` columns are read as the patches. Alternative to ``relevance``.
        :param relevance: ``[B, N]`` or ``[N]`` per-patch query relevance.
        :param grid: Patch grid, inferred as square when omitted.
        :param pruning_rate: Overrides the configured ``R`` for this call.
        :return: ``[B, K, D]`` or ``[K, D]``, same rank, dtype, and device, holding
            the retained patches in raster order followed by the aggregate token.
        """
        del layer_idx, kwargs
        if self._retain_ratio is not None:
            raise RuntimeError(
                "QuietPruneCompressor.compress() is the released CLIP tensor path. "
                "Qwen ratio pruning runs after concatenation through "
                "QwenQuietPruneAdapter."
            )
        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Expected [B, N, D] or [N, D], got {tuple(features.shape)}")
        if features.shape[1] == 0:
            raise ValueError("Cannot compress an empty sequence")
        batch, num_patches, _ = features.shape

        scores = self._relevance_from_args(relevance, attention, batch, num_patches)
        rate = self.pruning_rate if pruning_rate is None else float(pruning_rate)
        plan = build_plan(infer_patch_grid(num_patches, grid), group_size=self._group_size,
                          pruning_layers=(0,), pruning_rate=rate,
                          keep_border_tokens=self._keep_border_tokens, aggregate=self._aggregate,
                          include_cls=False).to(features.device)

        originals = torch.arange(num_patches, device=features.device).expand(batch, num_patches).contiguous()
        kept_groups = torch.ones((batch, plan.num_groups), dtype=torch.bool, device=features.device)
        tokens, originals, _, aggregate = self._prune_step(
            features, scores, originals, kept_groups, plan, plan.num_groups,
        )
        if aggregate is not None:
            tokens = torch.cat([tokens, aggregate[0].to(tokens.dtype).unsqueeze(1)], dim=1)
        identity = torch.eye(num_patches, device=features.device, dtype=features.dtype)
        self._source_history = [identity, self._source_map(originals, aggregate, plan, features.dtype)[0]]
        return CompressedFeatures(tokens=tokens.squeeze(0) if squeeze else tokens)

    def _relevance_from_args(self, relevance: torch.Tensor | None, attention: torch.Tensor | None,
                             batch: int, num_patches: int) -> torch.Tensor:
        if relevance is not None:
            scores = relevance.unsqueeze(0) if relevance.ndim == 1 else relevance
            if tuple(scores.shape) != (batch, num_patches):
                raise ValueError(f"relevance must be [{batch}, {num_patches}], got {tuple(scores.shape)}")
            return scores.float()
        if attention is not None:
            if attention.ndim != 4:
                raise ValueError(f"attention must be [B, H, T, T], got {tuple(attention.shape)}")
            return attention[:, :, 0, -num_patches:].float().mean(dim=1)
        raise ValueError(
            "QuietPruneCompressor.compress needs `relevance` or `attention`: its scores are "
            "[Q-CLS] -> patch attention, and without a query there is no query guidance. "
            "install() supplies them from inside the vision tower."
        )

    # ---------------------------------------------------------- pruning kernels

    def _prune_step(self, patches: torch.Tensor, relevance: torch.Tensor, originals: torch.Tensor,
                    kept_groups: torch.Tensor, plan: QuietPrunePlan, current_groups: int,
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """Drop the lowest-scoring whole groups and aggregate what was dropped (Sec. 4.2).

        The whole batch is pruned at once: the surviving-group count follows the
        schedule rather than the image, so ``kept_groups`` stays a rectangular
        ``[B, num_groups]`` mask and every gather below keeps its row width.

        :param patches: ``[B, K, D]`` surviving patch tokens in raster order.
        :param relevance: ``[B, K]`` query relevance, aligned with ``patches``.
        :param originals: ``[B, K]`` original patch index of each surviving token.
        :param kept_groups: ``[B, num_groups]`` groups still alive before this step.
        :param current_groups: How many that is (identical across the batch).
        :return: ``(patches, originals, kept_groups, aggregate)`` after the drop,
            where ``aggregate`` is ``(token, mass, source)`` or ``None``.
        """
        batch, _, hidden_size = patches.shape
        device = patches.device
        # Score every group by the mean relevance of its members. Scattering into a
        # dense [B, num_patches] row first means the group lookup is one indexing op
        # against a fixed [num_groups, group_size ** 2] table, whatever survived so far.
        dense = torch.zeros((batch, plan.num_patches), dtype=torch.float32, device=device)
        dense.scatter_(1, originals, relevance.float())
        group_scores = dense[:, plan.group_members].mean(dim=-1)
        group_scores = group_scores.masked_fill(~kept_groups, float("-inf"))

        num_kept = _groups_kept(current_groups, plan.pruning_rate)
        new_kept = torch.zeros_like(kept_groups)
        new_kept.scatter_(1, group_scores.topk(num_kept, dim=1).indices, True)

        # Sentinel column for border patches: kept iff keep_border_tokens, and never
        # a pruning candidate -- the reference drops them outright rather than
        # folding them into the aggregate.
        keep_lookup = torch.cat([new_kept, torch.full((batch, 1), plan.keep_border_tokens,
                                                      dtype=torch.bool, device=device)], dim=1)
        prune_lookup = torch.cat([kept_groups & ~new_kept,
                                  torch.zeros((batch, 1), dtype=torch.bool, device=device)], dim=1)
        keep_here = keep_lookup[:, plan.group_of_patch].gather(1, originals)
        prune_here = prune_lookup[:, plan.group_of_patch].gather(1, originals)
        kept_count, pruned_count = int(keep_here[0].sum()), int(prune_here[0].sum())

        keep_positions = keep_here.nonzero(as_tuple=True)[1].view(batch, kept_count)
        new_originals = originals.gather(1, keep_positions)
        new_patches = patches.gather(1, keep_positions.unsqueeze(-1).expand(batch, kept_count, hidden_size))

        aggregate = None
        if self._aggregate and pruned_count:
            prune_positions = prune_here.nonzero(as_tuple=True)[1].view(batch, pruned_count)
            aggregate = self._aggregate_tokens(
                patches.gather(1, prune_positions.unsqueeze(-1).expand(batch, pruned_count, hidden_size)),
                relevance.gather(1, prune_positions),
                originals.gather(1, prune_positions),
                plan.num_patches,
            )
        return new_patches, new_originals, new_kept, aggregate

    def _aggregate_tokens(self, tokens: torch.Tensor, relevance: torch.Tensor, originals: torch.Tensor,
                          num_patches: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x_m = sum_i a_i x_i / sum_i a_i`` over the dropped tokens (Sec. 4.3).

        :return: ``(token [B, D], mass [B], source [B, num_patches])``. The mass is
            what lets a later prune's aggregate be folded into this one in
            proportion to how much context each absorbed; the source row is always
            normalised, being a provenance distribution rather than a value.
        """
        weights = relevance.float().clamp_min(0.0)
        mass = weights.sum(dim=1)
        alive = mass > self._eps
        # An all-zero relevance row means the query had no opinion about anything it
        # dropped; a uniform average is the honest summary, and the mass is set to 1
        # so it still counts for something when folded into a later aggregate.
        uniform = torch.full_like(weights, 1.0 / max(1, weights.shape[1]))
        normalized = torch.where(alive.unsqueeze(1), weights / mass.clamp_min(self._eps).unsqueeze(1), uniform)
        mass = torch.where(alive, mass, torch.ones_like(mass))
        token_weights = normalized if self._normalize_aggregation else weights
        token = (tokens * token_weights.to(tokens.dtype).unsqueeze(-1)).sum(dim=1)
        source = torch.zeros((tokens.shape[0], num_patches), dtype=torch.float32, device=tokens.device)
        source.scatter_add_(1, originals, normalized)
        return token, mass, source

    def _combine_aggregates(self, previous: tuple[torch.Tensor, ...] | None,
                            current: tuple[torch.Tensor, ...] | None) -> tuple[torch.Tensor, ...] | None:
        """Fold a new aggregate into the rolling one, weighted by relevance mass."""
        if previous is None:
            return current
        if current is None:
            return previous
        previous_token, previous_mass, previous_source = previous
        current_token, current_mass, current_source = current
        total = (previous_mass + current_mass).clamp_min(self._eps)
        token = (previous_token * (previous_mass / total).to(previous_token.dtype).unsqueeze(-1)
                 + current_token * (current_mass / total).to(current_token.dtype).unsqueeze(-1))
        source = previous_source + current_source
        source = source / source.sum(dim=-1, keepdim=True).clamp_min(self._eps)
        return token, total, source

    @staticmethod
    def _source_map(originals: torch.Tensor, aggregate: tuple[torch.Tensor, ...] | None,
                    plan: QuietPrunePlan, dtype: torch.dtype, *, leading_cls: bool = False) -> torch.Tensor:
        """``[B, T_cur, num_patches]``: one-hot per retained patch, weights for the aggregate."""
        batch, kept = originals.shape
        rows = kept + (1 if aggregate is not None else 0) + (1 if leading_cls else 0)
        source = torch.zeros((batch, rows, plan.num_patches), dtype=dtype, device=originals.device)
        offset = 1 if leading_cls else 0
        source[:, offset:offset + kept].scatter_(2, originals.unsqueeze(-1), 1.0)
        if aggregate is not None:
            source[:, -1] = aggregate[2].to(dtype)
        return source

    # ------------------------------------------------------------ vision tower

    @staticmethod
    def _qcls_relevance(layer: nn.Module, hidden: torch.Tensor, patch_start: int,
                        num_patches: int) -> torch.Tensor:
        """Mean-over-heads ``[Q-CLS] -> patch`` attention for one encoder layer (Sec. 4.2).

        Recomputed from the layer's own ``q_proj``/``k_proj`` on the layer's own
        ``layer_norm1`` output rather than read back out of the attention module,
        which under SDPA returns no probabilities at all. The softmax runs over the
        full row -- ``[Q-CLS]``, CLIP CLS, and any aggregate included -- and only the
        patch columns are kept, matching the reference.
        """
        attention = layer.self_attn
        normed = layer.layer_norm1(hidden)
        batch, length, _ = normed.shape
        heads, head_dim = attention.num_heads, attention.head_dim
        queries = attention.q_proj(normed[:, :1]).view(batch, 1, heads, head_dim).transpose(1, 2)
        keys = attention.k_proj(normed).view(batch, length, heads, head_dim).transpose(1, 2)
        logits = torch.matmul(queries.float(), keys.float().transpose(-1, -2)) * attention.scale
        return logits.softmax(dim=-1)[:, :, 0, patch_start:patch_start + num_patches].mean(dim=1)

    @staticmethod
    def _run_encoder_layer(layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        outputs = layer(hidden, attention_mask=None)
        return outputs[0] if isinstance(outputs, tuple) else outputs

    def _vision_forward(self, pixel_values: torch.Tensor, q_cls: torch.Tensor) -> torch.Tensor:
        """Manual CLIP forward with ``[Q-CLS]`` injected and pruning inside (Sec. 4.2-3.5).

        :param pixel_values: ``[B, C, H, W]``.
        :param q_cls: ``[B, 1, D]`` pooled query token, already in the ViT's space.
        :return: ``[B, final_tokens, D]`` pre-projector visual tokens, taken from the
            same hidden state LLaVA's ``vision_feature_layer`` selects.
        """
        plan, vision_model = self.plan, self._vision_model
        hidden = vision_model.embeddings(pixel_values)
        batch = hidden.shape[0]
        has_cls = hidden.shape[1] == plan.num_patches + 1
        if not has_cls and hidden.shape[1] != plan.num_patches:
            raise ValueError(
                f"Vision embeddings produced {hidden.shape[1]} tokens; the plan expects "
                f"{plan.num_patches} patches (+1 CLS) for a {plan.grid[0]}x{plan.grid[1]} grid."
            )
        # [Q-CLS] is prepended *after* patch embedding, so it carries no position
        # embedding: it is a query probe, not a location in the image.
        hidden = torch.cat([q_cls.to(device=hidden.device, dtype=hidden.dtype), hidden], dim=1)
        if hasattr(vision_model, "pre_layrnorm"):
            hidden = vision_model.pre_layrnorm(hidden)
        patch_start = 2 if has_cls else 1

        originals = torch.arange(plan.num_patches, device=hidden.device).expand(batch, plan.num_patches)
        originals = originals.contiguous()
        kept_groups = torch.ones((batch, plan.num_groups), dtype=torch.bool, device=hidden.device)
        current_groups = plan.num_groups
        aggregate: tuple[torch.Tensor, ...] | None = None
        history: list[torch.Tensor] = []

        for layer_idx, layer in enumerate(vision_model.encoder.layers):
            # The selected feature layer is where the tower stops -- the layers past
            # it are exactly the compute LLaVA never uses. `-1` means the embeddings
            # were selected, so no encoder layer runs at all.
            if layer_idx > self._selected_layer_idx:
                break
            # Relevance comes from this layer's own attention, so it is read off the
            # layer's input; the drop then applies to the layer's output.
            relevance = (self._qcls_relevance(layer, hidden, patch_start, originals.shape[1])
                         if layer_idx in plan.pruning_layers else None)
            hidden = self._run_encoder_layer(layer, hidden)
            if relevance is not None:
                patch_end = patch_start + originals.shape[1]
                prefix = hidden[:, :patch_start]
                suffix = hidden[:, patch_end:]
                patches, originals, kept_groups, step = self._prune_step(
                    hidden[:, patch_start:patch_end], relevance, originals, kept_groups, plan, current_groups,
                )
                current_groups = plan.groups_after_layer[plan.pruning_layers.index(layer_idx)]
                if not self._aggregate:
                    suffix = hidden[:, :0]
                elif self._aggregate_mode == "rolling":
                    # The rolling aggregate rode through this layer's attention like
                    # any other token, so carry the *evolved* value forward, keeping
                    # the provenance and mass it already had.
                    if aggregate is not None and suffix.shape[1] > 0:
                        aggregate = (suffix[:, -1].float(), aggregate[1], aggregate[2])
                    aggregate = self._combine_aggregates(aggregate, step)
                    suffix = (aggregate[0].to(hidden.dtype).unsqueeze(1) if aggregate is not None
                              else hidden[:, :0])
                else:
                    aggregate = self._combine_aggregates(aggregate, step)
                    suffix = hidden[:, :0]
                hidden = torch.cat([prefix, patches, suffix], dim=1)
                history.append(self._source_map(originals, aggregate, plan, torch.float32)[0])

        patch_end = patch_start + originals.shape[1]
        tokens = hidden[:, patch_start:patch_end]
        if aggregate is not None:
            evolved = hidden[:, patch_end:]
            if evolved.shape[1] > 0:
                aggregate = (evolved[:, -1].float(), aggregate[1], aggregate[2])
                tokens = torch.cat([tokens, evolved[:, -1:].to(tokens.dtype)], dim=1)
            else:
                tokens = torch.cat([tokens, aggregate[0].to(tokens.dtype).unsqueeze(1)], dim=1)
        if plan.include_cls:
            # `vision_feature_select_strategy="full"` keeps CLIP's own CLS. It maps to
            # no original patch, so its source row is zero -- the convention
            # `llmoe.viz` already knows to strip.
            tokens = torch.cat([hidden[:, patch_start - 1:patch_start], tokens], dim=1)
        if tokens.shape[1] != plan.final_tokens:
            raise RuntimeError(
                f"QuietPrune produced {tokens.shape[1]} tokens but planned {plan.final_tokens}; the "
                f"trimmed <image> placeholders would not match."
            )
        identity = torch.eye(plan.num_patches, dtype=torch.float32, device=hidden.device)
        final = self._source_map(originals, aggregate, plan, torch.float32, leading_cls=plan.include_cls)[0]
        self._source_history = [identity, *history[:-1], final]
        return tokens

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        """Identity, then one ``[T_cur, num_patches]`` matrix per pruning layer.

        Retained rows are one-hot; the trailing aggregate row holds the normalised
        relevance mass of every patch it absorbed, so ``llmoe.viz`` reads that one
        token as a merge and the rest as pruning. Batch element 0 only, per the
        ``TokenCompressor`` contract.
        """
        if self._qwen is not None:
            return self._qwen.get_source_history()
        return list(self._source_history)

    # ------------------------------------------------------------- checkpoints

    def checkpoint_dict(self) -> dict[str, Any]:
        if self.adapter is None:
            raise RuntimeError("QuietPrune has no adapter to save until install() builds it")
        if self._backend == "llava_next":
            if self._next_adapter is None:
                raise RuntimeError("QuietPrune/LLaVA-NeXT lifecycle is incomplete")
            metadata = self._next_adapter.checkpoint_metadata()
            stage = str(self._training_metadata.get("stage", "untrained_ablation"))
            if stage not in {"tr", "ft", "untrained_ablation"}:
                raise ValueError(f"Unknown QuietPrune/LLaVA-NeXT checkpoint stage {stage!r}")
            if stage in {"tr", "ft"} and abs(self.pruning_rate - TRAINING_PRUNING_RATE) > 1e-8:
                raise ValueError(
                    f"Canonical QuietPrune adapters must be trained at R={TRAINING_PRUNING_RATE}; "
                    f"the installed schedule uses R={self.pruning_rate}."
                )
            recipe = (
                "adapter_only_self_distillation_kl_plus_answer_ce"
                if stage in {"tr", "ft"}
                else "inverse_projector_initialization_only"
            )
            metadata["training"] = {
                "stage": stage,
                "recipe": recipe,
                **self._training_metadata,
            }
            return {
                "metadata": metadata,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in self.adapter.state_dict().items()
                },
            }
        return {
            "format_version": 1,
            "compressor": self.__class__.__name__,
            "config": {
                "max_tokens": self._budget,
                "pruning_rate": self.pruning_rate,
                "target_average_pruning_rate": self._target_average,
                "pruning_layers": None if self._plan is None else list(self._plan.pruning_layers),
                "group_size": self._group_size,
                "keep_border_tokens": self._keep_border_tokens,
                "aggregate_pruned_tokens": self._aggregate,
                "aggregate_mode": self._aggregate_mode,
                "normalize_aggregation": self._normalize_aggregation,
            },
            "state_dict": {key: value.detach().cpu() for key, value in self.adapter.state_dict().items()},
        }

    def save_checkpoint(self, path: str | PathLike[str]) -> None:
        torch.save(self.checkpoint_dict(), path)

    def load_checkpoint(self, path: str | PathLike[str]) -> None:
        """Load an adapter checkpoint onto the built adapter.

        Only the adapter is stored: everything else about QuietPrune is geometry,
        and the paper's own claim is that one adapter serves every budget. So the
        rate recorded in a checkpoint is provenance, not a constraint -- a mismatch
        is reported, not refused. The adapter *shape* still has to match, which
        ``load_state_dict`` enforces.
        """
        if self.adapter is None:
            raise RuntimeError("Build the adapter with install() before loading a checkpoint")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Unsupported QuietPrune checkpoint payload: {type(payload).__name__}")
        if self._backend == "llava_next":
            if self._next_adapter is None:
                raise RuntimeError("QuietPrune/LLaVA-NeXT lifecycle is incomplete")
            metadata = payload.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError(
                    "LLaVA-NeXT QuietPrune checkpoint lacks required provenance metadata"
                )
            self._next_adapter.validate_checkpoint_metadata(metadata)
            schedule = metadata.get("schedule")
            if isinstance(schedule, Mapping):
                trained_layers = tuple(int(value) for value in schedule.get("pruning_layers", ()))
                if trained_layers and trained_layers != self.plan.pruning_layers:
                    raise ValueError(
                        f"Checkpoint pruning_layers={trained_layers}, compressor uses "
                        f"{self.plan.pruning_layers}"
                    )
            state = payload.get("state_dict")
            if not isinstance(state, Mapping):
                raise ValueError("LLaVA-NeXT QuietPrune checkpoint has no adapter state_dict")
            self.adapter.load_state_dict(state)
            return
        # The trainer writes a resumable checkpoint (`adapter_state_dict` beside the
        # optimizer state and RNG); a hand-saved adapter is a bare state dict under
        # `state_dict`, or the tensors themselves. All three are accepted.
        config = payload.get("quietprune_config") or payload.get("config") or {}
        if isinstance(config, Mapping):
            trained_rate = config.get("pruning_rate") or config.get("training_pruning_rate")
            if trained_rate is not None and abs(float(trained_rate) - self.pruning_rate) > 1e-6:
                print(
                    f"QuietPrune: adapter trained at R={float(trained_rate):.3f}, evaluating at "
                    f"R={self.pruning_rate:.3f}. The paper trains at R={TRAINING_PRUNING_RATE} and varies "
                    f"R at evaluation, so this is expected.",
                    flush=True,
                )
            trained_group = config.get("group_size")
            if trained_group is not None and int(trained_group) != self._group_size:
                raise ValueError(
                    f"Checkpoint uses group_size={trained_group}, compressor uses {self._group_size}"
                )
        state = payload.get("adapter_state_dict") or payload.get("state_dict") or payload
        self.adapter.load_state_dict(state)

    # --------------------------------------------------------------- lifecycle

    @contextmanager
    def bypass(self) -> Iterator[None]:
        """Run the installed backbone as plain full-token LLaVA.

        The self-distillation teacher is this same model without pruning, and the
        placeholder trim has to lift together with the pruning or the two disagree
        about sequence length. One flag governs both.
        """
        previous = self._bypass
        self._bypass = True
        try:
            yield
        finally:
            self._bypass = previous

    @override
    def install(self, vlm: VLM) -> None:
        if self._vlm is not None:
            if self._vlm is vlm:
                return  # idempotent: repeated benchmarks must not stack wrappers
            raise RuntimeError("QuietPruneCompressor is already installed on another backbone")

        _ = vlm.processor  # force the lazy load so model and processor both exist
        model = vlm.model
        if getattr(model.config, "model_type", None) == "llava_next":
            self.validate_llava_next()
            from llmoe.compressor.quietprune.quietprune_next import (
                LlavaNextQuietPruneAdapter,
            )

            adapter = LlavaNextQuietPruneAdapter(self, vlm)
            self._next_adapter = adapter
            try:
                adapter.install()
            except Exception:
                adapter.uninstall()
                self._next_adapter = None
                self._backend = None
                self._vlm = None
                self._backbone = {}
                self._plan = None
                self.adapter = None
                raise
            return
        inner = model.model
        if getattr(inner, "visual", None) is not None and self._query_guided:
            # The paper's mechanism: [Q-CLS] guidance and pruning *inside* the tower,
            # which is where its latency comes from. `R` is the paper's own knob and
            # makes the surviving count deterministic, so no retain_ratio is needed.
            from llmoe.compressor.quietprune.quietprune_qwen_install import (
                QwenQuietPruneInstaller,
            )

            installer = QwenQuietPruneInstaller(
                rate=self.pruning_rate,
                aggregate=self._aggregate,
                normalize_aggregation=self._normalize_aggregation,
                eps=self._eps,
            )
            installer.compressor = self
            installer.install(vlm)
            self.adapter = installer.adapter
            self._resolve_checkpoint("qwen-vl")
            self._load_qwen_adapter()
            self._vlm = vlm
            self._qwen_install = installer
            print(
                f"[QuietPrune/Qwen] install: R={self.pruning_rate:g} at ViT blocks "
                f"{list(installer._runner.pruning_layers)}, aggregate="
                f"{self._aggregate}, normalize={self._normalize_aggregation}. "
                f"Query-guided in-ViT pruning (Sec. 4.1-4.3).",
                flush=True,
            )
            return
        if getattr(inner, "visual", None) is not None:
            if self._retain_ratio is None:
                raise ValueError(
                    "QuietPrune on Qwen2.5-VL requires retain_ratio: NaViT "
                    "emits a different visual-token count for each image."
                )
            from llmoe.compressor.quietprune.quietprune_qwen import QwenQuietPruneAdapter
            from llmoe.utils.qwen import resolve_image_token_id

            adapter = QwenQuietPruneAdapter(
                compressor=self,
                retain_ratio=self._retain_ratio,
                attention_chunk=self._attention_chunk,
                track_source=True,
            )
            adapter.install(
                vlm,
                image_token_id=resolve_image_token_id(
                    model.config, vlm.processor, label="QuietPrune"
                ),
            )
            self._vlm = vlm
            self._qwen = adapter
            return
        if self._retain_ratio is not None:
            raise ValueError(
                "retain_ratio is QuietPrune's Qwen2.5-VL parameterisation; "
                "LLaVA-1.5 uses max_tokens or a pruning rate."
            )
        if getattr(model.config, "model_type", None) != "llava":
            raise TypeError(
                "QuietPrune requires the LLMOE LLaVA backbone (transformers "
                "LlavaForConditionalGeneration): [Q-CLS] injection, the CLS-then-patches "
                "layout, and the pinv-invertible projector are all CLIP-ViT assumptions."
            )
        vision_tower = inner.vision_tower
        self._backbone = {
            "vision_model": getattr(vision_tower, "vision_model", vision_tower),
            "projector": inner.multi_modal_projector,
        }
        self._vlm = vlm
        self._backend = "llava"

        num_layers = len(self._vision_model.encoder.layers)
        self._select_strategy = str(getattr(model.config, "vision_feature_select_strategy", "default"))
        self._selected_layer_idx = self._selected_encoder_layer(
            num_layers, getattr(model.config, "vision_feature_layer", -2)
        )
        # A pruning layer past the selected feature layer never runs, so it must not
        # appear in the plan either -- otherwise the placeholder trim would promise a
        # cut the forward never makes.
        pruning_layers = tuple(
            layer for layer in (self._pruning_layers or self._default_pruning_layers(num_layers))
            if layer <= self._selected_layer_idx
        )
        if not pruning_layers:
            raise ValueError(
                f"No configured pruning layer runs before this tower's selected feature layer "
                f"(encoder layer {self._selected_layer_idx} of {num_layers})"
            )

        vision_config = self._vision_model.config
        side = int(vision_config.image_size) // int(vision_config.patch_size)
        include_cls = self._select_strategy == "full"
        rate = (self._solve_rate_for_budget(self._budget, (side, side), pruning_layers, include_cls)
                if self._budget is not None else self.pruning_rate)
        self._plan = build_plan((side, side), group_size=self._group_size, pruning_layers=pruning_layers,
                                pruning_rate=rate, keep_border_tokens=self._keep_border_tokens,
                                aggregate=self._aggregate, include_cls=include_cls).to(model.device)
        self._max_tokens = self._plan.final_tokens

        self._build_adapter(model.device, backbone=backbone_key(vlm))
        self._install_hooks(vlm, inner, model)
        print(
            f"QuietPrune installed: R={self._plan.pruning_rate:.3f} at layers "
            f"{list(self._plan.pruning_layers)}, {self._plan.num_patches} -> "
            f"{'/'.join(str(count) for count in self._plan.patches_after_layer)} patches, "
            f"{self._plan.final_tokens} tokens to the LLM.",
            flush=True,
        )
        if self._checkpoint_path is None:
            print(
                "QuietPrune: no checkpoint_path, so the adapter is at its inverse initialisation. "
                "That is the paper's starting point, not its result -- train one with `tr`.",
                flush=True,
            )

    @staticmethod
    def _selected_encoder_layer(num_layers: int, vision_feature_layer: Any) -> int:
        """Zero-based encoder layer whose *output* ``vision_feature_layer`` selects.

        ``hidden_states`` has ``num_layers + 1`` entries (embeddings first), so the
        usual ``-2`` means "output of layer ``num_layers - 2``" and the tower can
        stop there -- part of where QuietPrune's savings come from.
        """
        if not isinstance(vision_feature_layer, int):
            raise TypeError(
                f"QuietPrune needs a single vision_feature_layer, got {vision_feature_layer!r}; a "
                f"multi-layer feature pool would need the tower run past the cut."
            )
        index = vision_feature_layer if vision_feature_layer >= 0 else num_layers + 1 + vision_feature_layer
        if not 0 <= index <= num_layers:
            raise ValueError(f"vision_feature_layer={vision_feature_layer} is invalid for {num_layers} layers")
        return index - 1

    def _resolve_checkpoint(self, backbone: str | None) -> None:
        """Pick the released artifact for this backbone, if one is wanted."""
        self._checkpoint_path = resolve_weights(
            "QuietPrune", self._budget or 0,
            checkpoint_path=self._checkpoint_arg,
            pretrained=self._pretrained_arg,
            backbone=backbone,
        )

    def _load_qwen_adapter(self) -> None:
        """Load a trained ``[Q-CLS]`` adapter onto the in-ViT Qwen path.

        The released artifact is a *LLaVA* adapter -- it inverts LLaVA's projector
        (4096 -> 1024), whereas Qwen's inverts ``visual.merger.mlp`` into the
        merger's input width. The two are not interchangeable, and
        :data:`~llmoe.weights.ARTIFACTS` is keyed by ``(method, budget)`` with no
        backbone in the key, so what gets auto-resolved is always the LLaVA one.

        A checkpoint is therefore loaded only when the caller named it, and its
        shapes are checked against the built adapter first, so a backbone mix-up
        reports itself instead of surfacing as a ``load_state_dict`` size error.
        """
        if self.adapter is None or self._checkpoint_path is None:
            return
        if not self._checkpoint_explicit:
            warnings.warn(
                "QuietPrune on Qwen ignores the released adapter: that one is "
                "LLaVA's (it inverts LLaVA's projector, not visual.merger). Pass "
                "`checkpoint_path` with a Qwen-trained adapter, otherwise the run "
                "uses the inverse initialisation, which is the paper's ablation.",
                stacklevel=3,
            )
            self._checkpoint_path = None
            return
        expected = {name: tuple(tensor.shape)
                    for name, tensor in self.adapter.state_dict().items()}
        payload = torch.load(self._checkpoint_path, map_location="cpu", weights_only=True)
        state = payload
        if isinstance(payload, Mapping):
            state = (payload.get("adapter_state_dict") or payload.get("state_dict")
                     or payload)
        mismatched = {
            name: (tuple(tensor.shape), expected.get(name))
            for name, tensor in state.items()
            if isinstance(tensor, torch.Tensor) and name in expected
            and tuple(tensor.shape) != expected[name]
        }
        if mismatched:
            raise ValueError(
                f"This adapter does not fit Qwen's [Q-CLS] adapter: {mismatched} "
                f"(got, expected). A LLaVA-trained QuietPrune adapter cannot be used "
                f"on Qwen -- the two invert different projectors."
            )
        self.load_checkpoint(self._checkpoint_path)

    def _build_adapter(self, device: torch.device, *, backbone: str | None) -> None:
        # The adapter inverts *this* backbone's projector, so the released weights
        # are per backbone: LLaVA-1.5-7B's is 4096 -> 1024 and the 13B's 5120 ->
        # 1024. Naming the backbone here is what makes an unreleased one fail by
        # name instead of loading another backbone's adapter.
        self._resolve_checkpoint(backbone)
        dtype = self._adapter_dtype or next(self._projector.parameters()).dtype
        self.adapter = QuietPruneAdapter.from_projector(self._projector)
        if self._checkpoint_path is not None:
            self.load_checkpoint(self._checkpoint_path)
        self.adapter.to(device=device, dtype=dtype)

    def _resolve_checkpoint_for_backend(self, backend: str,
                                        backend_backbone: str = "llava") -> None:
        """Select only artifacts compatible with the installed projector."""
        explicit = self._checkpoint_arg
        if explicit is None and isinstance(self._pretrained_arg, (str, PathLike)):
            explicit = self._pretrained_arg
        if backend == "llava_next":
            if explicit is not None:
                self._checkpoint_path = resolve_weights(
                    "QuietPrune",
                    self._budget or 0,
                    checkpoint_path=explicit,
                    pretrained=False,
                )
                return
            # The NeXT adapter is released and registered, so `pretrained: true`
            # resolves it by backbone rather than failing. Naming the backbone is
            # what keeps the LLaVA-1.5 adapter -- which inverts a different
            # projector -- from being loaded here.
            self._checkpoint_path = resolve_weights(
                "QuietPrune",
                self._budget or 0,
                checkpoint_path=self._checkpoint_arg,
                pretrained=self._pretrained_arg,
                backbone="llava-next-8b",
            )
            return
        self._checkpoint_path = resolve_weights(
            "QuietPrune",
            self._budget or 0,
            checkpoint_path=self._checkpoint_arg,
            pretrained=self._pretrained_arg,
            backbone=backend_backbone,
        )

    def _install_hooks(self, vlm: VLM, inner: nn.Module, model: nn.Module) -> None:
        """Capture the prompt, replace the vision path, and shorten the placeholders."""
        image_token_id = getattr(model.config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(model.config, "image_token_index", None)
        if image_token_id is None:
            raise ValueError("Could not resolve the LLaVA image token id from the model config")
        self._image_token_id = int(image_token_id)
        tokenizer = getattr(vlm.processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        self._pad_token_id = self._image_token_id if pad_token_id is None else int(pad_token_id)

        # `LlavaModel.forward` never passes input_ids down to get_image_features, but
        # [Q-CLS] is made of them, so grab them one level up.
        def capture_prompt(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            self._latest_input_ids = input_ids if torch.is_tensor(input_ids) else None
            attention_mask = kwargs.get("attention_mask")
            self._latest_attention_mask = attention_mask if torch.is_tensor(attention_mask) else None
            labels = kwargs.get("labels")
            if torch.is_tensor(labels):
                self._latest_labels = labels
            elif not self._bypass and self._pending_training_labels is not None:
                if not torch.is_tensor(input_ids) or self._pending_training_labels.shape != input_ids.shape:
                    raise ValueError("Pending QuietPrune training labels do not match input_ids")
                # The trainer computes CE itself and therefore does not pass
                # labels into the VLM. Consume this one-shot copy solely for the
                # Q-CLS query mask so answer tokens cannot leak into selection.
                self._latest_labels = self._pending_training_labels
                self._pending_training_labels = None
            else:
                self._latest_labels = None
            return None

        self._hooks.append(inner.register_forward_pre_hook(capture_prompt, with_kwargs=True))

        original_get_image_features = inner.get_image_features
        self._original_get_image_features = original_get_image_features

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            pixel_values = kwargs.get("pixel_values", args[0] if args else None)
            if self._bypass or pixel_values is None:
                return original_get_image_features(*args, **kwargs)
            tokens = self._vision_forward(pixel_values, self._query_to_qcls(pixel_values.shape[0]))
            features = self._projector(tokens)
            return BaseModelOutputWithPooling(last_hidden_state=features, pooler_output=list(features))

        inner.get_image_features = wrapped_get_image_features

        self._original_prepare_inputs = vlm.prepare_inputs
        original_prepare_inputs = self._original_prepare_inputs

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            inputs = original_prepare_inputs(*args, **kwargs)
            return inputs if self._bypass else self.trim_placeholders(inputs)

        vlm.prepare_inputs = wrapped_prepare_inputs

    def uninstall(self) -> None:
        """Undo everything ``install()`` changed, in reverse order."""
        if self._next_adapter is not None:
            self._next_adapter.uninstall()
            self._next_adapter = None
            self._latest_input_ids = None
            self._latest_attention_mask = None
            self._latest_labels = None
            self._pending_training_labels = None
            self._backbone = {}
            self._image_token_id = None
            self._pad_token_id = None
            self._vlm = None
            self._backend = None
            self._plan = None
            return
        if self._vlm is None:
            return
        if self._qwen is not None:
            self._qwen.uninstall()
            self._qwen = None
        if self._qwen_install is not None:
            self._qwen_install.uninstall()
            self._qwen_install = None
            self._vlm = None
            return
        vlm = self._vlm
        if self._original_get_image_features is not None:
            vlm.model.model.get_image_features = self._original_get_image_features
        if self._original_prepare_inputs is not None:
            vlm.prepare_inputs = self._original_prepare_inputs
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._original_get_image_features = None
        self._original_prepare_inputs = None
        self._latest_input_ids = None
        self._latest_attention_mask = None
        self._latest_labels = None
        self._pending_training_labels = None
        self._backbone = {}
        self._image_token_id = None
        self._pad_token_id = None
        self._vlm = None
        self._backend = None

    def validate_llava_next(self) -> None:
        """Fail closed before loading the 8B model for unsupported NeXT knobs."""
        if self._budget is not None:
            raise ValueError(
                "QuietPrune/LLaVA-NeXT uses a per-image retain_ratio, not a fixed "
                "max_tokens budget, because native AnyRes content length is image-specific."
            )
        if self._target_average is not None:
            raise ValueError(
                "QuietPrune/LLaVA-NeXT does not accept target_average_pruning_rate; "
                "use retain_ratio for evaluation or pruning_rate for canonical training."
            )
        if self._retain_ratio is not None and self._pruning_rate is not None:
            raise ValueError(
                "QuietPrune/LLaVA-NeXT requires exactly one budget control: "
                "retain_ratio for evaluation or pruning_rate for canonical training."
            )
        if self._retain_ratio is None and not (
            self._pruning_rate is not None and self._next_training_rate_mode
        ):
            raise ValueError(
                "QuietPrune/LLaVA-NeXT evaluation requires retain_ratio. Direct "
                "pruning_rate is reserved for the registered tr/ft training recipe."
            )
        if self._pruning_layers is not None:
            raise ValueError(
                "Canonical QuietPrune/LLaVA-NeXT fixes its pruning stages at "
                "one-quarter, one-half, and three-quarters of CLIP depth."
            )
        if not self._keep_border_tokens or not self._aggregate or self._aggregate_mode != "rolling":
            raise ValueError(
                "Canonical QuietPrune/LLaVA-NeXT keeps border tokens, aggregates "
                "pruned tokens, and uses rolling aggregation."
            )
        if self._normalize_aggregation:
            raise ValueError(
                "Canonical QuietPrune/LLaVA-NeXT uses the paper's unnormalized "
                "relevance-weighted aggregate."
            )
        if self._group_size != 2:
            raise ValueError("Canonical QuietPrune/LLaVA-NeXT fixes 2x2 spatial groups")

    def prepare_training_inputs(
        self, inputs: Mapping[str, Any], labels: torch.Tensor
    ) -> dict[str, Any]:
        """Architecture-aware student inputs for adapter self-distillation."""
        if self._backend != "llava_next" or self._next_adapter is None:
            prepared = dict(inputs)
            prepared["labels"] = labels
            prepared = self.trim_placeholders(prepared)
            self._pending_training_labels = prepared["labels"]
            return prepared
        return self._next_adapter.prepare_training_inputs(inputs, labels)

    @property
    def last_budget_report(self) -> dict[str, Any] | None:
        """Exact requested/realized budget metadata for the latest NeXT image."""
        return None if self._last_budget_report is None else dict(self._last_budget_report)

    @override
    def get_benchmark_record_metadata(self, batch_size: int) -> list[dict[str, Any]]:
        report = self.last_budget_report
        if report is None:
            return super().get_benchmark_record_metadata(batch_size)
        if int(batch_size) != 1:
            raise RuntimeError(
                "QuietPrune/LLaVA-NeXT budget reporting requires batch_size=1."
            )
        return [{"compression_budget": report}]

    @override
    def get_visualization_views(
        self,
        vlm: VLM,
        inputs: Mapping[str, Any],
        image_processor: Any,
    ) -> list[VisualizationView] | None:
        if self._backend != "llava_next":
            return None
        from llmoe.compressor.quietprune.quietprune_next import (
            llava_next_visualization_views,
        )

        return llava_next_visualization_views(vlm, inputs, image_processor)

    # ------------------------------------------------------------- query / LLM

    def _query_to_qcls(self, num_images: int) -> torch.Tensor:
        """Pool the captured prompt into one ``[Q-CLS]`` per image (Sec. 4.2).

        Image placeholders are masked out alongside padding: ``[Q-CLS]`` is meant to
        carry what the user asked, and the placeholder embedding says nothing about
        that.
        """
        if self.adapter is None or self._latest_input_ids is None:
            raise RuntimeError("QuietPrune saw no prompt for this forward; install() captures it")
        input_ids = self._latest_input_ids
        if input_ids.shape[0] != num_images:
            raise ValueError(
                f"QuietPrune expects one image per prompt on LLaVA-1.5, got {num_images} images for "
                f"{input_ids.shape[0]} prompts."
            )
        embeddings = self._vlm.model.model.get_input_embeddings()(input_ids)
        mask = (self._latest_attention_mask if self._latest_attention_mask is not None
                else torch.ones_like(input_ids)).long()
        mask = mask * (input_ids != self._image_token_id).long()
        if self._latest_labels is not None:
            if self._latest_labels.shape != input_ids.shape:
                raise ValueError("QuietPrune labels must match input_ids")
            mask = mask * (self._latest_labels == -100).long()
        return self.adapter(embeddings.to(dtype=next(self.adapter.parameters()).dtype), mask)

    def trim_placeholders(self, inputs: Any) -> Any:
        """Shorten every ``<image>`` placeholder run to the planned token count.

        Transformers checks that the placeholder count equals the feature count, so
        compression has to move both sides of that equality. Rows are re-padded on
        the left, matching the backbone's generation-time padding side.
        """
        if self._qwen_install is not None:
            # The Qwen count depends on the image grid, not on a fixed plan.
            return inputs if self._bypass else self._qwen_install._trim(inputs)
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return inputs
        target = self.plan.final_tokens
        attention_mask = inputs.get("attention_mask")
        has_attention = isinstance(attention_mask, torch.Tensor) and attention_mask.shape == input_ids.shape

        kept_ids: list[torch.Tensor] = []
        kept_attention: list[torch.Tensor] = []
        keep_masks: list[torch.Tensor] = []
        changed = False
        for row_index, row in enumerate(input_ids):
            keep = torch.ones(row.shape[0], dtype=torch.bool, device=row.device)
            positions = (row == self._image_token_id).nonzero(as_tuple=False).flatten()
            if positions.numel():
                # One run per image: split wherever the placeholder ids are not adjacent.
                splits = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=False).flatten() + 1
                for run in torch.tensor_split(positions, splits.tolist()):
                    if run.numel() < target:
                        raise ValueError(
                            f"QuietPrune plans {target} tokens but the prompt only has {run.numel()} "
                            f"<image> placeholders; the budget cannot exceed the original count."
                        )
                    if run.numel() > target:
                        keep[run[target:]] = False
                        changed = True
            kept_ids.append(row[keep])
            keep_masks.append(keep)
            if has_attention:
                kept_attention.append(attention_mask[row_index][keep])
        if not changed:
            return inputs

        width = max(row.numel() for row in kept_ids)
        new_ids = input_ids.new_full((len(kept_ids), width), self._pad_token_id)
        new_attention = attention_mask.new_zeros((len(kept_ids), width)) if has_attention else None
        for row_index, row in enumerate(kept_ids):
            new_ids[row_index, width - row.numel():] = row
            if new_attention is not None:
                new_attention[row_index, width - row.numel():] = kept_attention[row_index]
        inputs["input_ids"] = new_ids
        if new_attention is not None:
            inputs["attention_mask"] = new_attention
        for key in ("labels", "token_type_ids"):
            value = inputs.get(key)
            if not isinstance(value, torch.Tensor) or value.shape != input_ids.shape:
                continue
            fill = -100 if key == "labels" else 0
            trimmed = value.new_full((len(kept_ids), width), fill)
            for row_index, keep in enumerate(keep_masks):
                row = value[row_index][keep]
                trimmed[row_index, width - row.numel():] = row
            inputs[key] = trimmed
        if isinstance(inputs.get("position_ids"), torch.Tensor):
            active = inputs.get("attention_mask")
            if isinstance(active, torch.Tensor):
                position_ids = active.long().cumsum(-1) - 1
                position_ids.masked_fill_(active == 0, 0)
            else:
                position_ids = torch.arange(width, device=input_ids.device).expand(len(kept_ids), -1)
            inputs["position_ids"] = position_ids
        return inputs
