"""VisionZip visual-token compressor.

Paper: Yang et al., "VisionZip: Longer is Better than Necessary in Vision
Language Models" (arXiv:2412.04467).

One compressor supports LLaVA-1.5, native LLaVA-NeXT, Qwen2.5-VL, and InternVL.
``install()`` selects the matching implementation. Upstream ships its LLaVA and
Qwen variants as two separate code drops (``visionzip/`` and ``Qwen2_5_VL/``)
because they genuinely differ: LLaVA ranks patches by ``[CLS]`` attention
inside the CLIP tower and takes a fixed token
budget, while Qwen2.5-VL has no ``[CLS]``, emits a different token count per
image, and needs the cut placed after the patch merger. ``QwenVisionZipAdapter``
holds the divergent half, the way ``glimpse_prune.py`` splits its two releases;
``contextual_merge`` is the one step they share.

The two are parameterised differently and the parameters are **not**
interchangeable — see ``max_tokens`` vs. ``retain_ratio`` below.

LLaVA-1.5
---------
VisionZip keeps two kinds of tokens, both taken from the vision encoder's
penultimate layer — the layer LLaVA already reads its image features from:

* **Dominant** — ``[CLS]`` plus the ``dominant - 1`` patches receiving the most
  ``[CLS]`` attention (summed over heads). These carry the salient content.
* **Contextual** — the remaining patches merged down to ``contextual`` tokens.
  Evenly-spaced patches act as merge targets; every other remaining patch joins
  its most similar target (cosine similarity over the self-attention *keys*) and
  each target group is averaged, preserving background that selection drops.

The result is ``dominant + contextual`` tokens, with ``[CLS]`` counted inside
``dominant`` as in the official code (LLaVA-1.5 default ``54 + 10 = 64``).

Both stages run on pre-projector hidden states inside the ViT, so the projector
only ever sees compressed tokens and its nonlinearity is never applied before the
contextual averaging. ``install`` therefore:

1. pre-hooks the target layer's ``self_attn`` to recompute the CLS->patch
   attention and per-token key metric from ``q_proj``/``k_proj``, which keeps the
   capture independent of the attention backend (eager/sdpa/flash);
2. forces LLaVA's feature-select strategy to ``"full"`` so the preserved
   ``[CLS]`` is projected, matching the official integration;
3. patches that layer's ``forward`` so the reduction is applied to its output —
   the tensor transformers records as ``hidden_states[vision_feature_layer]``
   and hands to the projector;
4. trims ``<image>`` placeholders to the compressed count, since transformers
   5.x scatters image features with a strict count check.

The projector can be swapped for one fine-tuned at this token budget (the paper's
"Efficient Tuning" / VisionZip-double-dagger variant) by passing
``checkpoint_path``; see ``vision_zip_ft.py`` for how that checkpoint is produced.

LLaVA-NeXT
----------
The native NeXT adapter keeps the released LLaVA placement and signals: every
CLIP crop is reduced before projection using CLS attention and key similarity.
It then restores selected local patches in AnyRes mosaic order and appends the
local CLS/contextual summaries, matching the upstream LLaVA-1.6 packing path.
Only ``retain_ratio`` is public; exact per-image output lengths are derived from
the native packed span, crop budgets are balanced internally, and the published
54:10 dominant/contextual proportion is fixed. Architecture-specific tuned
projector weights are rejected on this training-free baseline.

Qwen2.5-VL
----------
Reference code: ``Qwen2_5_VL/qwen2_5vl_visionzip.py`` in
https://github.com/JIA-Lab-research/VisionZip (released 2025.05.26), a fork of the
transformers ~4.49 modeling file with the reduction spliced into
``Qwen2_5_VLForConditionalGeneration.forward``.

There is no ``[CLS]`` token here, so the LLaVA saliency signal does not exist.
Upstream replaces it, and moves the whole reduction downstream of the patch
merger:

* **Dominant.** The last ViT block's self-attention is read off and each token is
  scored by the attention it *receives*: ``mean`` over heads, then ``sum`` over
  query rows. The per-patch scores are averaged within each ``2 x 2`` merger
  group so they line up with the tokens the LLM actually sees, and the top
  ``dominant_ratio`` of them are kept verbatim.
* **Contextual.** The leftovers are merged into ``contextual_ratio`` tokens by
  cosine similarity over the last block's (post-RoPE) keys, likewise averaged per
  merger group — the same ``contextual_merge`` the LLaVA path uses.

Both signals are un-permuted by ``argsort(window_index)`` before use, since the
windowed ViT reorders tokens at merger-group granularity.

Upstream prunes ``inputs_embeds`` *after* the image features have been scattered
into the prompt, and gathers the M-RoPE ``position_ids`` of the survivors from
the ones computed over the full sequence. That placement is not incidental — it
is what makes the port possible at all:

* the strict placeholder/feature count check in ``get_placeholder_mask`` sees the
  full, unpruned counts and passes, so no ``<|image_pad|>`` trimming is needed;
* ``get_rope_index`` derives each visual token's ``(t, h, w)`` coordinates from
  ``image_grid_thw``, so it *must* run on the untouched sequence. Trimming
  placeholders first — the route the LLaVA path and every other compressor in
  this package takes — would desynchronise it from ``pixel_values`` and corrupt
  every position after the image.

So the Qwen path touches neither ``prepare_inputs`` nor ``get_image_features``.
It hangs a pre-hook on ``model.model.language_model``, which is exactly the call
upstream edits, and rewrites ``inputs_embeds`` / ``position_ids`` /
``attention_mask`` there. Layers ``0..N-1`` and the whole KV cache only ever see
the reduced sequence. See ``QwenVisionZipAdapter`` for the faithfulness notes and the
list of unsupported cases.
"""

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple, override

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor.internvl import (
    tokenizer_pad_token_id,
    trim_image_placeholders,
)
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.weights import backbone_key, resolve_weights

# Published (dominant, contextual) split per total budget. CLS counts as one
# dominant token, so `dominant - 1` patches are selected by CLS attention.
TOKEN_SPLITS: dict[int, tuple[int, int]] = {
    32: (27, 5),
    64: (54, 10),
    128: (108, 20),
    192: (162, 30),
    224: (189, 35),
}

# Published upstream splits: "70%" is 65% dominant + 5% contextual, "50%" is
# 45% + 5%. The contextual share is held fixed across both.
DEFAULT_CONTEXTUAL_RATIO: float = 0.05
# Query rows scored per matmul in the last-block recompute. The full map would be
# [heads, L, L]; this bounds it to [heads, chunk, L].
DEFAULT_ATTENTION_CHUNK: int = 512

def resolve_image_token_id(config: Any, processor: Any) -> int:
    """Resolve the ``<image>`` placeholder id, mirroring LLaVA's own lookup order.

    Shared with ``vision_zip_tr`` so the compressor's placeholder trimming and the
    training data pipeline agree on which token marks an image.
    """
    token_id = getattr(config, "image_token_id", None)
    if token_id is None:
        token_id = getattr(config, "image_token_index", None)
    if token_id is None:
        image_token = getattr(processor, "image_token", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if image_token is not None and tokenizer is not None:
            token_id = tokenizer.convert_tokens_to_ids(str(image_token))
    if token_id is None:
        raise RuntimeError("Could not resolve the image placeholder token id for VisionZip.")
    return int(token_id)


# ---------------------------------------------------------------------------
# Shared by both backbones
# ---------------------------------------------------------------------------


class ContextualMerge(NamedTuple):
    """Result of merging ``[B, R, C]`` leftovers into ``contextual_num`` tokens.

    Every index is local to the leftover block that was passed in, not to the
    original token sequence — the two callers hold different mappings back to it.
    """

    tokens: torch.Tensor  # [B, T, C] target features plus the mean of their group
    target_indices: torch.Tensor  # [T] leftover rows promoted to merge targets
    nontarget_mask: torch.Tensor  # [R] True for the leftovers that were merged away
    assign_one_hot: torch.Tensor  # [B, M, T] which target each merged token joined
    counts: torch.Tensor  # [B, T, 1] tokens merged into each target, floored at 1


def contextual_merge(hidden: torch.Tensor, metric_normalized: torch.Tensor,
                     contextual_num: int) -> ContextualMerge:
    """VisionZip's contextual merge.

    ``hidden`` and ``metric_normalized`` are the ``[B, R, *]`` leftovers — the
    tokens the dominant selection did not keep. Every ``step``-th leftover
    becomes a merge target; the rest join their most similar target by cosine
    similarity over the (already L2-normalised) key metric, and each target
    absorbs the *mean* of its group **on top of** its own features, which is
    upstream's ``target_hidden + aggregated_hidden`` rather than a plain average.

    ``metric_normalized`` is expected in float32 (the callers normalise there for
    stability); ``assign_one_hot`` follows ``hidden``'s dtype so the aggregation
    matmul stays in the model's precision, as upstream does.
    """
    if hidden.ndim != 3:
        raise ValueError(f"Expected [B, R, C] leftovers, got {tuple(hidden.shape)}")
    if metric_normalized.ndim != 3:
        raise ValueError(f"Expected [B, R, D] leftover metrics, got {tuple(metric_normalized.shape)}")
    if contextual_num < 1:
        raise ValueError("contextual_num must be >= 1")

    batch, remaining, _ = hidden.shape
    if remaining < contextual_num:
        raise ValueError(
            f"Cannot build {contextual_num} contextual tokens from {remaining} leftovers"
        )

    device = hidden.device
    # Evenly spaced targets: upstream strides the leftovers rather than clustering
    # them, so the merge keeps a roughly uniform spatial spread of the background.
    step = max(1, remaining // contextual_num)
    target_indices = torch.arange(0, remaining, step, device=device)[:contextual_num]
    num_targets = target_indices.shape[0]
    nontarget_mask = torch.ones(remaining, dtype=torch.bool, device=device)
    nontarget_mask[target_indices] = False

    target_metric = metric_normalized[:, target_indices, :]
    merge_metric = metric_normalized[:, nontarget_mask, :]
    assignment = torch.bmm(merge_metric, target_metric.transpose(1, 2)).argmax(dim=2)
    assign_one_hot = torch.zeros(
        batch, merge_metric.shape[1], num_targets, dtype=hidden.dtype, device=device
    )
    assign_one_hot.scatter_(2, assignment.unsqueeze(-1), 1)
    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)

    hidden_merge = hidden[:, nontarget_mask, :]
    aggregated = torch.bmm(assign_one_hot.transpose(1, 2), hidden_merge) / counts
    tokens = hidden[:, target_indices, :] + aggregated
    return ContextualMerge(tokens, target_indices, nontarget_mask, assign_one_hot, counts)


# ---------------------------------------------------------------------------
# The compressor — backbone detection, plus the LLaVA-1.5 path
# ---------------------------------------------------------------------------


class VisionZipCompressor(TokenCompressor):
    """Keep dominant tokens by ``[CLS]`` attention and merge the rest into context.

    Parameters
    ----------
    max_tokens:
        **LLaVA-1.5 only.** Total visual tokens (``dominant + contextual``). Must
        be one of ``TOKEN_SPLITS``; the split is undefined for other budgets.
    retain_ratio:
        **Qwen2.5-VL and native LLaVA-NeXT only.** Fraction of the image-dependent
        visual span to retain. Qwen publishes ``0.7`` and ``0.5`` and spends a
        fixed 0.05 share on contextual tokens. NeXT uses the same single public
        control but fixes the released LLaVA 54:10 split internally.
    contextual_ratio:
        Qwen-only share of ``retain_ratio`` spent on merged contextual tokens;
        the rest is dominant. Upstream holds this at ``0.05`` for every budget.
    attention_chunk:
        Qwen-only. Query rows scored per matmul when recomputing the last vision
        block's attention. Lower it if the vision tower runs out of memory on
        very large images.
    vision_feature_layer:
        LLaVA-only. CLIP layer to read attention/keys from and reduce. Taken from
        the model config when ``None`` (-2 on LLaVA-1.5).
    track_source:
        Record ``[T_current, T_original]`` matrices for ``llmoe.viz``. Required by
        ``bm --visualize``; off by default so benchmark runs don't pay for it.
    pretrained:
        LLaVA-only released efficient-tuning weights. ``True`` takes the
        projector for this budget from the ``llmoe_weights`` bundle, downloading
        it on demand; a path names your own. Left unset (or ``False``) VisionZip
        stays training-free on the base projector, which is its default form.
    checkpoint_path:
        Checkpoint from ``finetune_vision_zip`` (or a raw state dict), loaded
        into LLaVA's ``multi_modal_projector`` or Qwen's ``visual.merger``.
        On LLaVA it takes precedence over ``pretrained``; on Qwen it is the only
        supported tuned-weight input because the bundle has no Qwen checkpoint.
    """

    def __init__(self, max_tokens: int = 64, *, retain_ratio: float | None = None,
                 contextual_ratio: float = DEFAULT_CONTEXTUAL_RATIO,
                 attention_chunk: int = DEFAULT_ATTENTION_CHUNK,
                 vision_feature_layer: int | None = None, track_source: bool = False,
                 pretrained: bool | str | PathLike[str] | None = None,
                 checkpoint_path: str | PathLike[str] | None = None) -> None:
        # The backbone is not known until install(), so the two parameterisations
        # are validated here and cross-checked against the backbone there.
        self._configured_max_tokens: int = int(max_tokens)
        if retain_ratio is None:
            if int(max_tokens) not in TOKEN_SPLITS:
                supported = ", ".join(str(budget) for budget in sorted(TOKEN_SPLITS))
                raise ValueError(
                    f"Unsupported visual-token budget: {max_tokens}. Only {{{supported}}} are "
                    f"supported, because the dominant/contextual split is undefined elsewhere. "
                    f"On Qwen2.5-VL pass `retain_ratio` instead."
                )
            dominant, contextual = TOKEN_SPLITS[int(max_tokens)]
            super().__init__(dominant + contextual)
        else:
            retain_ratio = float(retain_ratio)
            contextual_ratio = float(contextual_ratio)
            if not 0.0 < retain_ratio <= 1.0:
                raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
            if not 0.0 <= contextual_ratio < 1.0:
                raise ValueError(
                    f"contextual_ratio must be in [0, 1), got {contextual_ratio}"
                )
            dominant = contextual = 0
            super().__init__(0)  # ratio mode: the retained count is per-image

        self._retain_ratio: float | None = retain_ratio
        self._contextual_ratio: float = contextual_ratio
        self._attention_chunk: int = int(attention_chunk)
        self._dominant: int = dominant
        self._dominant_patches: int = max(0, dominant - 1)
        self._contextual: int = contextual
        self._vision_feature_layer: int | None = (
            None if vision_feature_layer is None else int(vision_feature_layer)
        )
        self._track_source: bool = bool(track_source)
        if self._retain_ratio is not None:
            if pretrained not in (None, False):
                raise ValueError(
                    "VisionZip's bundled `pretrained` weights are LLaVA-only; "
                    "for Qwen pass a Qwen `checkpoint_path`."
                )
        # Resolved in `install`, not here: the released projector is per backbone
        # (LLaVA-1.5-7B's is 4096-wide, the 13B's 5120) and which backbone this is
        # only becomes known when one is handed over.
        self._checkpoint_arg = checkpoint_path
        self._pretrained_arg = pretrained
        self._checkpoint_path: Path | None = None
        self._qwen: Any | None = None

        # Per-forward capture from the encoder hook, aligned to patch tokens (the
        # leading CLS row is stripped) so indices match the layer output's rows.
        self._cls_attention_sum: torch.Tensor | None = None  # [B, N]
        self._metric_patches: torch.Tensor | None = None  # [B, N, head_dim]
        self._patch_source: torch.Tensor | None = None  # [B, out, N], pre-CLS
        self._image_token_id: int | None = None

        # Installation bookkeeping, all restored by uninstall().
        self._installed_vlm: VLM | None = None
        self._target_layer: Any | None = None
        self._original_layer_forward: Any | None = None
        self._attn_handle: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._original_select_strategy: Any = None
        self._original_projector_state: dict[str, torch.Tensor] | None = None
        self._internvl: Any | None = None
        self._llava_next: Any | None = None

    @property
    @override
    def budget(self) -> int | float:
        return self._max_tokens if self._retain_ratio is None else self._retain_ratio

    def validate_llava_next(self) -> None:
        """Accept one training-free NeXT configuration: retain_ratio."""

        from llmoe.compressor.llava_next import validate_retain_ratio

        validate_retain_ratio(self._retain_ratio, label="VisionZip")
        if self._configured_max_tokens != 64:
            raise ValueError(
                "VisionZip on LLaVA-NeXT only accepts retain_ratio; remove "
                "max_tokens (its constructor default is 64 and is ignored on NeXT)."
            )
        if self._contextual_ratio != DEFAULT_CONTEXTUAL_RATIO:
            raise ValueError(
                "VisionZip on LLaVA-NeXT fixes the released LLaVA dominant/contextual "
                "split internally; only retain_ratio is configurable."
            )
        if self._attention_chunk != DEFAULT_ATTENTION_CHUNK:
            raise ValueError(
                "VisionZip on LLaVA-NeXT fixes its attention implementation; "
                "attention_chunk is Qwen-only."
            )
        if self._vision_feature_layer is not None:
            raise ValueError(
                "VisionZip on LLaVA-NeXT uses the checkpoint's selected vision "
                "layer; vision_feature_layer is not configurable."
            )
        if (
            self._checkpoint_arg is not None
            or self._pretrained_arg not in (None, False)
        ):
            raise ValueError(
                "VisionZip on LLaVA-NeXT currently supports the training-free "
                "baseline only; projector/merger checkpoints are architecture-specific."
            )

    # ------------------------------------------------------------------ install

    @staticmethod
    def _detect_backbone(vlm: VLM) -> str:
        """``"qwen"`` or ``"llava"``, checked structurally so forks still pass."""
        inner = getattr(getattr(vlm, "model", None), "model", None)
        if getattr(inner, "visual", None) is not None:
            return "qwen"
        if getattr(inner, "vision_tower", None) is not None:
            return "llava"
        raise TypeError(
            f"VisionZip supports LLaVA-1.5, LLaVA-NeXT, Qwen2.5-VL, and InternVL; "
            f"{type(getattr(vlm, 'model', None)).__name__} exposes neither "
            f"`model.model.vision_tower` nor `model.model.visual`. Run with `--backbone llava` "
            f"or `--backbone qwen-vl`."
        )

    def _resolve_checkpoint(self, vlm: VLM) -> None:
        """Pick the projector for this backbone, if one is wanted.

        The Qwen path takes an explicit ``checkpoint_path`` only -- there is no
        released Qwen projector -- while the LLaVA path may also reach for the
        released artifact for this budget *and this backbone*.
        """
        if self._retain_ratio is not None:
            self._checkpoint_path = (
                None if self._checkpoint_arg is None else Path(self._checkpoint_arg)
            )
            return
        resolved = resolve_weights(
            "VisionZip",
            self._dominant + self._contextual,
            checkpoint_path=self._checkpoint_arg,
            pretrained=self._pretrained_arg,
            backbone=backbone_key(vlm),
        )
        self._checkpoint_path = None if resolved is None else Path(resolved)

    @override
    def install(self, vlm: VLM) -> None:
        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return
            raise RuntimeError("VisionZipCompressor is already installed on another backbone")

        from llmoe.compressor.llava_next import is_llava_next

        if is_llava_next(vlm):
            self.validate_llava_next()
            from llmoe.compressor.vision_zip.vision_zip_llava_next import (
                LlavaNextVisionZipAdapter,
            )

            self._installed_vlm = vlm
            self._llava_next = LlavaNextVisionZipAdapter(
                self,
                vlm,
                retain_ratio=float(self._retain_ratio),
                track_source=self._track_source,
            )
            try:
                self._llava_next.install()
            except Exception:
                self._llava_next = None
                self._installed_vlm = None
                raise
            print(
                f"[VisionZip/LLaVA-NeXT] install: retain_ratio="
                f"{self._retain_ratio:g}, pre-projector crop-local reduction, "
                "published dominant/contextual split",
                flush=True,
            )
            return

        backbone = self._detect_backbone(vlm)
        if backbone == "qwen":
            if self._retain_ratio is None:
                raise ValueError(
                    "VisionZip on Qwen2.5-VL is parameterised by `retain_ratio`, not `max_tokens`: "
                    "the NaViT encoder gives every image a different visual-token count, so a "
                    "fixed budget is not comparable across samples. Pass e.g. "
                    "`compressor_args: {retain_ratio: 0.5}` (upstream publishes 0.7 and 0.5)."
                )
            from llmoe.compressor.vision_zip.vision_zip_qwen import QwenVisionZipAdapter

            if self._contextual_ratio >= self._retain_ratio:
                raise ValueError(
                    "VisionZip on Qwen2.5-VL requires contextual_ratio < "
                    f"retain_ratio, got {self._contextual_ratio:g} and "
                    f"{self._retain_ratio:g}."
                )
            self._resolve_checkpoint(vlm)
            self._installed_vlm = vlm
            self._qwen = QwenVisionZipAdapter(
                dominant_ratio=self._retain_ratio - self._contextual_ratio,
                contextual_ratio=self._contextual_ratio,
                track_source=self._track_source,
                attention_chunk=self._attention_chunk,
            )
            self._qwen.install(vlm, image_token_id=self._resolve_image_token_id(vlm))
            if self._checkpoint_path is not None:
                self._load_projector(vlm)
            return

        if self._retain_ratio is not None:
            raise ValueError(
                "retain_ratio selects VisionZip's Qwen2.5-VL path. On "
                "LLaVA-1.5/InternVL use `max_tokens`: "
                "the released budgets use a different dominant/contextual split (54/10 at 64 "
                "tokens, not 0.65/0.05), so a ratio would not reproduce the paper's numbers."
            )
        from llmoe.compressor.vision_zip.vision_zip_internvl import (
            InternVLVisionZipAdapter,
        )

        self._resolve_checkpoint(vlm)
        internvl = InternVLVisionZipAdapter.supports(vlm)
        if internvl and self._checkpoint_path is not None:
            raise ValueError(
                "InternVL supports training-free VisionZip only; LLaVA projector "
                "checkpoints are architecture-specific and cannot be loaded."
            )

        self._installed_vlm = vlm
        if internvl:
            self._internvl = InternVLVisionZipAdapter(self, vlm)
            self._internvl.install()
        else:
            self._force_full_feature_select(vlm)
            self._install_vision_patch(vlm)
        self._wrap_prepare_inputs(vlm)
        if self._checkpoint_path is not None:
            self._load_projector(vlm)

    def uninstall(self) -> None:
        if self._installed_vlm is None:
            return
        vlm = self._installed_vlm
        if self._llava_next is not None:
            self._llava_next.uninstall()
            self._llava_next = None
            self._installed_vlm = None
            self._cls_attention_sum = None
            self._metric_patches = None
            return
        if self._qwen is not None:
            self._qwen.uninstall()
            self._qwen = None
            if self._original_projector_state is not None:
                self._tuned_module(vlm).load_state_dict(self._original_projector_state, strict=True)
                self._original_projector_state = None
            self._installed_vlm = None
            return
        if self._internvl is not None:
            self._internvl.uninstall()
        if self._attn_handle is not None:
            self._attn_handle.remove()
        if self._target_layer is not None and self._original_layer_forward is not None:
            self._target_layer.forward = self._original_layer_forward
            self._target_layer._visionzip_patched = False
        if self._original_prepare_inputs is not None:
            vlm.prepare_inputs = self._original_prepare_inputs
            vlm._visionzip_prepare_wrapped = False
        config = getattr(getattr(vlm, "model", None), "config", None)
        if config is not None and self._original_select_strategy is not None:
            config.vision_feature_select_strategy = self._original_select_strategy
        if self._original_projector_state is not None:
            # Put the stock projector back so a later run in the same process is
            # not silently evaluated with tuned weights.
            vlm.model.model.multi_modal_projector.load_state_dict(
                self._original_projector_state, strict=True
            )
        self._attn_handle = None
        self._target_layer = None
        self._original_layer_forward = None
        self._original_prepare_inputs = None
        self._original_select_strategy = None
        self._original_projector_state = None
        self._internvl = None
        self._installed_vlm = None
        self._cls_attention_sum = None
        self._metric_patches = None
        self._patch_source = None

    def _force_full_feature_select(self, vlm: VLM) -> None:
        """Make LLaVA project the preserved ``[CLS]`` token.

        The official integration inserts the compressed tower output including
        CLS, but transformers drops it under the default ``"default"`` strategy.
        """
        config = getattr(vlm.model, "config", None)
        if config is None:
            return
        self._original_select_strategy = getattr(config, "vision_feature_select_strategy", None)
        config.vision_feature_select_strategy = "full"

    def _tuned_module(self, vlm: VLM) -> Any:
        """The module ``finetune_vision_zip`` adapts to the reduced token count.

        LLaVA projects the tower's output through ``multi_modal_projector``; Qwen has
        no such module and does the equivalent job in ``visual.merger``, which is what
        the Qwen recipe tunes. One accessor so install, load and uninstall cannot
        disagree about which one it is.
        """
        inner = vlm.model.model
        return inner.visual.merger if self._retain_ratio is not None else inner.multi_modal_projector

    def _load_projector(self, vlm: VLM) -> None:
        """Load a tuned projector/merger in place.

        ``load_state_dict`` copies into the existing parameters, so the module
        keeps its device and dtype (checkpoints are stored fp32 and downcast on
        copy for a half-precision eval model).
        """
        path = self._checkpoint_path
        if not path.is_file():
            raise FileNotFoundError(f"VisionZip checkpoint not found: {path}")

        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state: Mapping[str, torch.Tensor] = load_file(str(path))
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(payload, Mapping) and "state_dict" in payload:
                checkpoint_config = payload.get("config", {})
                if not isinstance(checkpoint_config, Mapping):
                    checkpoint_config = {}
                # A checkpoint is only valid at the budget it was tuned for: the whole
                # point of tuning is adapting to that token count.
                if self._retain_ratio is not None:
                    budget = checkpoint_config.get("retain_ratio")
                    if budget is not None and float(budget) != self._retain_ratio:
                        raise ValueError(
                            f"Checkpoint was tuned for retain_ratio={budget}, compressor uses "
                            f"{self._retain_ratio}"
                        )
                    contextual = checkpoint_config.get("contextual_ratio")
                    if (
                        contextual is not None
                        and float(contextual) != self._contextual_ratio
                    ):
                        raise ValueError(
                            "Checkpoint was tuned for "
                            f"contextual_ratio={contextual}, compressor uses "
                            f"{self._contextual_ratio}"
                        )
                else:
                    budget = checkpoint_config.get("max_tokens")
                    if budget is not None and int(budget) != self._max_tokens:
                        raise ValueError(
                            f"Checkpoint was tuned for max_tokens={budget}, compressor "
                            f"uses {self._max_tokens}"
                        )
                state = payload["state_dict"]
            elif isinstance(payload, Mapping):
                state = payload
            else:
                raise TypeError(f"Unsupported projector checkpoint payload: {type(payload).__name__}")

        module = self._tuned_module(vlm)
        self._original_projector_state = {
            key: value.detach().clone() for key, value in module.state_dict().items()
        }
        missing, unexpected = module.load_state_dict(state, strict=False)
        if missing or unexpected:
            self._original_projector_state = None
            raise RuntimeError(
                f"VisionZip checkpoint {str(path)!r} does not match the model's "
                f"{type(module).__name__} (missing={list(missing)}, unexpected={list(unexpected)})."
            )

    @staticmethod
    def _vision_encoder_layers(vlm: VLM) -> Any:
        """Locate the CLIP encoder layers.

        transformers 5.x hangs the encoder directly off ``CLIPVisionModel``;
        earlier layouts nest it under ``vision_model``.
        """
        vision_tower = getattr(getattr(getattr(vlm, "model", None), "model", None), "vision_tower", None)
        vision_model = getattr(vision_tower, "vision_model", vision_tower)
        encoder = getattr(vision_model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None:
            layers = getattr(encoder, "layer", None)
        if layers is None:
            raise RuntimeError(
                "VisionZipCompressor expects a vision tower exposing "
                "`encoder.layers` or `encoder.layer`; the active backbone does not provide one."
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
                f"VisionZip needs a single vision_feature_layer, got {layer!r}. Multiple selected "
                "layers would be read at different token counts."
            )
        return layer

    def _install_vision_patch(self, vlm: VLM) -> None:
        layers = self._vision_encoder_layers(vlm)
        layer = self._resolve_feature_layer(vlm)
        # `hidden_states[k]` is the output of layer `k - 1`, so a negative feature
        # layer maps to `layers[layer]` and a positive one to `layers[layer - 1]`.
        target = layers[layer] if layer < 0 else layers[layer - 1]
        if getattr(target, "_visionzip_patched", False):
            return

        # Capture CLS->patch attention and the key metric from this layer's real
        # pre-projector self-attention.
        self._attn_handle = target.self_attn.register_forward_pre_hook(
            self._attn_pre_hook, with_kwargs=True
        )
        self._target_layer = target
        self._original_layer_forward = target.forward
        original_layer_forward = target.forward

        def patched_layer_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
            hidden_states = original_layer_forward(*args, **kwargs)
            return self._reduce_hidden_states(hidden_states)

        target.forward = patched_layer_forward
        target._visionzip_patched = True

    def _attn_pre_hook(self, module: Any, args: tuple, kwargs: dict) -> None:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if hidden_states is None:
            return
        batch, seq, _ = hidden_states.shape
        num_heads, head_dim, scale = module.num_heads, module.head_dim, module.scale
        queries = module.q_proj(hidden_states)
        keys = module.k_proj(hidden_states)
        q_norm, k_norm = getattr(module, "q_norm", None), getattr(module, "k_norm", None)
        if callable(q_norm):
            queries = q_norm(queries)
        if callable(k_norm):
            keys = k_norm(keys)
        queries = queries.view(batch, seq, num_heads, head_dim).transpose(1, 2)
        keys = keys.view(batch, seq, num_heads, head_dim).transpose(1, 2)
        # CLS (row 0) attention over every key, summed over heads, restricted to
        # patches (drop the CLS->CLS column).
        logits = torch.matmul(queries[:, :, :1, :].float(), keys.float().transpose(-1, -2)) * scale
        self._cls_attention_sum = logits.softmax(dim=-1)[:, :, 0, 1:].sum(dim=1)  # [B, S-1]
        self._metric_patches = keys.mean(dim=1)[:, 1:, :].contiguous()  # [B, S-1, head_dim]

    def _reduce_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the reduction to a layer's ``[B, S, C]`` output.

        The encoder layer never reorders tokens, so patch rows line up with the
        attention/keys captured by the pre-hook. CLS stays at index 0 (and is
        projected, since the strategy is forced to ``"full"``), leaving the patch
        rows reduced to ``dominant - 1 + contextual``.
        """
        cls_token = hidden_states[:, :1, :]
        result = self.compress(
            hidden_states[:, 1:, :], attention=self._cls_attention_sum, keys=self._metric_patches
        )
        # Clear the capture so a forward that somehow bypasses the pre-hook cannot
        # silently reuse stale attention/keys.
        self._cls_attention_sum = None
        self._metric_patches = None
        if self._track_source:
            self._record_source_history()
        return torch.cat([cls_token, result.tokens.to(hidden_states.dtype)], dim=1)

    def _record_source_history(self) -> None:
        """Snapshot the source attribution for ``llmoe.viz``.

        Records the pre-merge identity then the post-reduction matrix, both
        ``[T_current, T_original]`` for image 0. Column 0 is CLS, so patch ``p``
        occupies column ``p + 1`` and ``viz`` can drop the CLS column.
        """
        if self._patch_source is None:
            # Nothing recorded: leave the history empty so the viz driver fails
            # loudly rather than rendering a stale frame.
            self._source_history = []
            return
        patch_source = self._patch_source[0].detach().cpu().float()
        out_tokens, num_patches = patch_source.shape
        size = num_patches + 1
        full = torch.zeros(out_tokens + 1, size, dtype=torch.float32)
        full[0, 0] = 1.0  # CLS row stays one-hot on the CLS column
        full[1:, 1:] = patch_source
        self._source_history = [torch.eye(size, dtype=torch.float32), full]
        self._patch_source = None

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        """Source matrices from the most recent encoder forward.

        Index 0 is the pre-merge identity, index 1 is VisionZip's reduction.
        Empty when ``track_source`` is off or no forward has run. On LLaVA the
        matrices cover ``[CLS] + patches``; on Qwen they cover the merged visual
        tokens alone, since there is no ``[CLS]``.
        """
        if self._qwen is not None:
            return self._qwen.get_source_history()
        return list(self._source_history)

    def _wrap_prepare_inputs(self, vlm: VLM) -> None:
        if getattr(vlm, "_visionzip_prepare_wrapped", False):
            return
        self._original_prepare_inputs = vlm.prepare_inputs
        original_prepare_inputs = vlm.prepare_inputs

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            return self._trim_image_placeholders(vlm, original_prepare_inputs(*args, **kwargs))

        vlm.prepare_inputs = wrapped_prepare_inputs
        vlm._visionzip_prepare_wrapped = True

    # ---------------------------------------------------------------- compress

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, keys: torch.Tensor | None = None,
                 **kwargs: Any) -> CompressedFeatures:
        """Reduce ``[B, N, C]`` (or ``[N, C]``) patch features, CLS already stripped.

        Returns ``dominant - 1 + contextual`` tokens; ``_reduce_hidden_states``
        reattaches CLS so the projected sequence is exactly ``max_tokens``.
        ``attention`` and ``keys`` default to the encoder hook's capture.

        LLaVA-only: the Qwen path reduces LLM-side embeddings against per-image
        ratios and drives the whole reduction from ``QwenVisionZipAdapter``, so it has no
        fixed-count tensor entry point.
        """
        del layer_idx, kwargs

        if self._retain_ratio is not None:
            raise RuntimeError(
                "VisionZipCompressor.compress() implements the fixed-budget LLaVA reduction, but "
                "this compressor was built in ratio mode for Qwen2.5-VL, where the reduction runs "
                "on the LLM-side embeddings. See `QwenVisionZipAdapter` in `vision_zip_qwen.py`."
            )

        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Expected [B, N, C] or [N, C], got {tuple(features.shape)}")

        if attention is None:
            attention = self._cls_attention_sum
        if keys is None:
            keys = self._metric_patches
        if attention is None or keys is None:
            raise RuntimeError(
                "VisionZipCompressor.compress needs the vision-encoder CLS attention and key "
                "metric; call install() so the encoder hook captures them before inference."
            )
        if attention.ndim == 1:
            attention = attention.unsqueeze(0)
        if keys.ndim == 2:
            keys = keys.unsqueeze(0)

        batch, num_tokens, channels = features.shape
        attention = attention.to(features.device)
        keys = keys.to(features.device)
        dominant_target = self._dominant if self._internvl is not None else self._dominant_patches
        dominant_num = min(dominant_target, num_tokens)

        # --- Dominant: top-`dominant_num` patches by CLS attention. ------------
        if dominant_num > 0:
            top_indices = attention.topk(dominant_num, dim=1).indices
            top_indices, _ = top_indices.sort(dim=1)  # keep spatial order
            dominant_tokens = torch.gather(
                features, 1, top_indices.unsqueeze(-1).expand(-1, -1, channels)
            )
            keep_mask = torch.zeros(batch, num_tokens, dtype=torch.bool, device=features.device)
            keep_mask.scatter_(1, top_indices, True)
            remaining_mask = ~keep_mask
        else:
            dominant_tokens = features[:, :0, :]
            remaining_mask = torch.ones(batch, num_tokens, dtype=torch.bool, device=features.device)

        # Dominant rows are pure selection, so each is one-hot at the patch it
        # keeps (one-hot reads as pruning to the viz renderer).
        dominant_source = None
        if self._track_source:
            dominant_source = torch.zeros(
                batch, dominant_num, num_tokens, dtype=torch.float32, device=features.device
            )
            if dominant_num > 0:
                dominant_source.scatter_(2, top_indices.unsqueeze(-1), 1.0)

        remaining = num_tokens - dominant_num
        contextual_num = min(self._contextual, remaining)
        if contextual_num <= 0:
            if self._track_source:
                self._patch_source = dominant_source
            return CompressedFeatures(
                tokens=dominant_tokens.squeeze(0) if squeeze else dominant_tokens
            )

        hidden_remaining = features[remaining_mask].view(batch, remaining, channels)
        metric_remaining = keys[remaining_mask].view(batch, remaining, keys.shape[-1]).float()

        # --- Contextual: merge the rest into evenly spaced targets by key
        #     cosine similarity. ----------------------------------------------
        metric_norm = metric_remaining / metric_remaining.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        merged = contextual_merge(hidden_remaining, metric_norm, contextual_num)
        target_indices = merged.target_indices
        num_targets = int(target_indices.shape[0])
        nontarget_mask = merged.nontarget_mask
        assign_one_hot = merged.assign_one_hot
        counts = merged.counts  # [B, T, 1]
        contextual_tokens = merged.tokens

        if self._track_source:
            # Contextual token t is `target + mean(assigned merges)`, so its row
            # weights the target patch by 1 and each assigned patch by 1/count[t]
            # — fractional rows read as merging to the renderer. The masked gather
            # preserves order, so remaining patch k is original `remaining_global[k]`.
            remaining_global = (
                torch.arange(num_tokens, device=features.device)
                .unsqueeze(0)
                .expand(batch, num_tokens)[remaining_mask]
                .view(batch, remaining)
            )
            target_global = remaining_global.gather(1, target_indices.unsqueeze(0).expand(batch, -1))
            merge_global = remaining_global[:, nontarget_mask]
            contextual_source = torch.zeros(
                batch, num_targets, num_tokens, dtype=torch.float32, device=features.device
            )
            contextual_source.scatter_(2, target_global.unsqueeze(-1), 1.0)
            contextual_source.scatter_add_(
                2,
                merge_global.unsqueeze(1).expand(batch, num_targets, merge_global.shape[1]),
                assign_one_hot.transpose(1, 2).float() / counts.float(),
            )
            self._patch_source = torch.cat([dominant_source, contextual_source], dim=1)

        tokens = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(features.dtype)
        return CompressedFeatures(tokens=tokens.squeeze(0) if squeeze else tokens)

    # ------------------------------------------------------------- placeholders

    def _resolve_image_token_id(self, vlm: VLM) -> int:
        if self._image_token_id is None:
            self._image_token_id = resolve_image_token_id(vlm.model.config, vlm.processor)
        return self._image_token_id

    def _trim_image_placeholders(self, vlm: VLM, inputs: Any) -> Any:
        """Trim each source image's placeholder run to the fixed global budget."""

        return trim_image_placeholders(
            inputs,
            token_id=self._resolve_image_token_id(vlm),
            keep_for_run=lambda _source_tokens: self._max_tokens,
            pad_token_id=tokenizer_pad_token_id(vlm),
        )

    @staticmethod
    def _pad_token_id(vlm: VLM) -> int:
        tokenizer = getattr(vlm.processor, "tokenizer", None)
        pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        return int(pad) if pad is not None else 0

    @staticmethod
    def _left_pad_stack(rows: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        # Generation uses left padding; rows are uniform for single-image batches
        # but pad defensively in case a batch is ragged.
        max_len = max(row.shape[0] for row in rows)
        padded: list[torch.Tensor] = []
        for row in rows:
            if row.shape[0] < max_len:
                pad = torch.full((max_len - row.shape[0],), pad_value, dtype=row.dtype,
                                 device=row.device)
                row = torch.cat([pad, row], dim=0)
            padded.append(row)
        return torch.stack(padded, dim=0)


class VisionZipPlusPlusCompressor(VisionZipCompressor):
    """Checkpoint-required efficient-tuning baseline for LLaVA-NeXT-8B.

    ``retain_ratio`` is the sole compression control. It is applied per image
    to the true native, unpadded LLaVA-NeXT content count; learned newline rows
    are preserved outside that budget. The released 54:10 split is scaled over
    each crop's capacity after the image-wide target is allocated.
    """

    def __init__(
        self,
        retain_ratio: float,
        *,
        checkpoint_path: str | PathLike[str] | None = None,
        track_source: bool = False,
    ) -> None:
        if checkpoint_path is None:
            raise ValueError(
                "VisionZip++ requires compressor_args.checkpoint_path pointing to a "
                "completed LLaVA-NeXT projector artifact. Use VisionZip for the "
                "training-free baseline."
            )
        super().__init__(
            retain_ratio=retain_ratio,
            vision_feature_layer=None,
            track_source=track_source,
            pretrained=False,
            checkpoint_path=checkpoint_path,
        )
        self._require_checkpoint = True
        self._last_budget_report: Any | None = None

    @property
    @override
    def name(self) -> str:
        return f"VisionZip++-r{self.budget:g}"

    @property
    def budget_report(self) -> Any | None:
        return self._last_budget_report

    @override
    def validate_llava_next(self) -> None:
        from llmoe.compressor.llava_next import validate_retain_ratio

        validate_retain_ratio(self._retain_ratio, label="VisionZip++")
        if self._configured_max_tokens != 64:
            raise ValueError("VisionZip++/LLaVA-NeXT accepts retain_ratio, not max_tokens")
        if self._vision_feature_layer is not None:
            raise ValueError(
                "VisionZip++/LLaVA-NeXT uses the checkpoint's fixed feature layer"
            )
        if self._pretrained_arg not in (None, False):
            raise ValueError(
                "VisionZip++ requires its explicit NeXT checkpoint_path; pretrained is not accepted"
            )

    def _load_next_projector(self, vlm: VLM) -> None:
        from llmoe.compressor.vision_zip.vision_zip_llava_next_pp import (
            TARGET_MODEL_REVISION,
            load_projector_checkpoint,
        )

        path = self._checkpoint_path
        if path is None and not self._require_checkpoint:
            return
        if path is None or not path.is_file():
            raise FileNotFoundError(f"VisionZip++ checkpoint not found: {path}")
        projector = vlm.model.model.multi_modal_projector
        self._original_projector_state = {
            key: value.detach().clone() for key, value in projector.state_dict().items()
        }
        try:
            load_projector_checkpoint(
                path,
                projector,
                retain_ratio=float(self._retain_ratio),
                model_id=str(vlm._model_id),
                model_revision=str(
                    getattr(vlm, "_model_revision", TARGET_MODEL_REVISION)
                ),
                image_grid_pinpoints=vlm.model.config.image_grid_pinpoints,
            )
        except Exception:
            projector.load_state_dict(self._original_projector_state, strict=True)
            self._original_projector_state = None
            raise

    @override
    def install(self, vlm: VLM) -> None:
        from llmoe.compressor.vision_zip.vision_zip_llava_next_pp import (
            LlavaNextVisionZipAdapter,
            is_llava_next_backbone,
        )

        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return
            raise RuntimeError("VisionZip++ is already installed on another backbone")
        if not is_llava_next_backbone(vlm):
            raise TypeError(
                "VisionZip++ is pinned to llava-next-8b; use VisionZip on other backbones"
            )
        self.validate_llava_next()
        self._resolve_checkpoint(vlm)
        self._installed_vlm = vlm
        self._llava_next = LlavaNextVisionZipAdapter(self, vlm)
        try:
            self._llava_next.install()
            self._load_next_projector(vlm)
        except Exception:
            self._llava_next.uninstall()
            self._llava_next = None
            self._installed_vlm = None
            self._last_budget_report = None
            raise

    @override
    def uninstall(self) -> None:
        if (
            self._installed_vlm is not None
            and self._llava_next is not None
            and self._original_projector_state is not None
        ):
            self._installed_vlm.model.model.multi_modal_projector.load_state_dict(
                self._original_projector_state, strict=True
            )
            self._original_projector_state = None
        super().uninstall()
        self._last_budget_report = None


class VisionZipPlusPlusTrainingCompressor(VisionZipPlusPlusCompressor):
    """Internal checkpoint-optional policy used while producing a ++ artifact."""

    def __init__(
        self,
        retain_ratio: float,
        *,
        checkpoint_path: str | PathLike[str] | None = None,
        track_source: bool = False,
    ) -> None:
        VisionZipCompressor.__init__(
            self,
            retain_ratio=retain_ratio,
            vision_feature_layer=None,
            track_source=track_source,
            pretrained=False,
            checkpoint_path=checkpoint_path,
        )
        self._require_checkpoint = False
        self._last_budget_report: Any | None = None
