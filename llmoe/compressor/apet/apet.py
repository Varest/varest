"""Faithful reimplementation of ApET (Approximation-Error guided Token compression).

Paper: "ApET: Approximation-Error Guided Token Compression for Efficient VLMs".
Reference code: https://github.com/MaQianKun0/ApET. The compressor detects
whether it is sitting on LLaVA-1.5, native LLaVA-NeXT, Qwen2.5-VL, or InternVL at ``install()`` and selects
the matching implementation, so ``--compressor ApET`` works with both
``--backbone llava`` and ``--backbone qwen-vl``. Upstream ships them as two
separate code drops (``llava/`` and ``qwen/``) that are parameterised
differently (a token ``budget`` vs. token *ratios*, not interchangeable), but
share the training-free core prune:

1. sample a small basis set with FPS by default,
2. linearly reconstruct every visual token from those basis tokens,
3. rank tokens by reconstruction error,
4. keep high-error tokens, writing the basis tokens into the lowest-error kept
   slots (overwritten on LLaVA, blended 50/50 on Qwen — upstream's own split),
5. merge dropped tokens into the nearest retained non-basis token.

LLaVA-1.5
---------
Reproduces the paper's *two-stage* pipeline driven by a single ``budget``:
a post-encoder reduction (before the projector) and a middle-layer LLM prune at
``DEFAULT_COMPRESS_LAYER`` (16 for LLaVA-1.5). The reported budget is a
layer-aware average, so ``budget`` is mapped to the per-stage token counts via
``stage_budgets``. The middle-layer prune is wired through decoder forward
pre-hooks that re-supply the pruned attention mask / rotary positions to every
downstream layer (see ``_install_layer_hooks``).

* Like ``must_drop.py``, this runs against a stock transformers
  ``LlavaForConditionalGeneration``: every stage is expressed through wrappers
  and forward hooks rather than by editing model source. The post-encoder stage
  shrinks each image to ``visual_tokens`` features, so the surplus ``<image>``
  placeholders are trimmed in a ``prepare_inputs`` wrapper (transformers 5.x
  scatters image features with a strict count check).

Qwen2.5-VL
----------
Reference code: ``qwen/model/qwen2_5_vl_custom.py`` plus
``qwen/model/qwen2_5_vl_utils.py`` (``token_merging``/``fps``), a fork of the
transformers ~4.49 modeling file with both stages spliced into
``Qwen2_5_VLForConditionalGeneration.forward``. The published configuration is
``--image-token-ratio 0.300 --layer-list '[14]' --image-token-ratio-list
'[0.333]' --basis-token-num 10`` (``scripts/qwen2_5_vl/*.sh``).

* **Ratios, not a budget.** Qwen's NaViT encoder emits a different token count
  per image, so the LLaVA convention of a layer-averaged absolute budget is not
  comparable across samples. Upstream therefore parameterises by ratios and
  this port keeps that: stage A keeps ``int(retain_ratio * N)`` of each
  image's ``N`` tokens, and each boundary in ``layer_list`` keeps
  ``int(image_token_ratio_list[i] * current)`` of the *current* count.
* **Stage A** ranks the projected image embeddings — the exact values that get
  scattered into the prompt — so this port runs it as a pre-hook on
  ``model.model.language_model``, after the scatter. That placement is what
  makes the port possible on transformers 5.x: the strict placeholder/feature
  count check sees the full counts, and ``get_rope_index`` runs on the
  untouched sequence, whose M-RoPE ids the survivors then keep (the same route
  ``vision_zip.py`` and ``duet.py`` take, for the same reasons). Stage A blends
  each basis token 50/50 into a kept slot in ascending-index order (upstream's
  boolean-mask write) with ridge ``1e-5``.
* **Layer boundaries** re-run the same prune on the decoder's hidden states.
  Upstream's 1-indexed ``rank_layer`` cuts after layer ``rank_layer - 1``, so
  the same number is used directly as the 0-indexed pre-hook position: layer
  ``layer_list[i]`` is the first to run reduced (the ``fastv.py`` / ``duet.py``
  convention). The blend pairs slots in top-k order here and the ridge is
  ``1e-1``, exactly as in ``layer_prune``. The released
  ``Qwen2_5_VLModel_X.forward`` ships with the ``layer_prune`` *call* commented
  out even though the eval scripts pass ``--layer-list`` and the method is
  complete; this port honours the scripts. Pass ``layer_list: []`` to reproduce
  the stage-A-only behaviour of the release verbatim.
* **Decoding.** Upstream slices the mask/positions inside ``forward``, but
  ``generate`` keeps full-length copies, so this port carries the pruned
  prefill's last position forward and rebuilds the mask at each stage's real
  cache width on every decode step — the same treatment ``fastv.py``,
  ``vision_zip.py`` and ``duet.py`` give the identical situation. Survivors
  keep their original absolute M-RoPE coordinates throughout.
* **Batch size 1**, as upstream requires (`features[0]` indexing throughout):
  per-image token counts differ, so a padded batch cannot share one keep-set.
* Not supported: video, more than one image per sample, beam search,
  ``use_cache=False``, and sliding-window decoder configs. Each raises rather
  than silently producing garbage.
* The prune arithmetic runs in float32 and is cast back to the model dtype
  (upstream solves in float32 but blends/merges in bfloat16; float32 throughout
  is a strict precision upgrade of the same expressions).

Harness notes
-------------
* Source matrices for ``llmoe.viz`` are only recorded when ``track_source`` is
  set; pass ``track_source: true`` under ``compressor_args`` before running
  ``bm --visualize``.
"""

from collections.abc import Sequence
from math import sqrt
from typing import Any, Literal, override

import torch
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

from llmoe.backbone.prototype import VLM
from llmoe.compressor.internvl import trim_image_placeholders
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.utils.depth import REFERENCE_DEPTH, decoder_depth, scale_layer

SelectionMethod = Literal["fps", "dpc", "random"]


# Paper LLaVA-1.5 layer-aware budget configuration. The reported "token budget"
# is the average retained visual tokens per decoder layer, not the final count;
# see ``stage_budgets``. Compression runs at the post-encoder stage and at
# decoder layer ``DEFAULT_COMPRESS_LAYER``.
DEFAULT_COMPRESS_LAYER = 16  # mid-LLM pruning layer (Vicuna-7B has 32 layers)
DECODER_LAYERS = REFERENCE_DEPTH
FULL_VISUAL_TOKENS = 576  # LLaVA-1.5 visual tokens after the projector
STAGE_RATIO = 3  # visual_token_num : image_token_list (official scripts)

# Official Qwen2.5-VL configuration (``scripts/qwen2_5_vl/*.sh``): stage A keeps
# ``retain_ratio`` (published: 0.300) of the image tokens, then each
# boundary in ``layer_list`` keeps ``image_token_ratio_list[i]`` of the current
# count. Boundaries are upstream's 1-indexed ``rank_layer`` — the cut runs after
# layer ``rank_layer - 1``, so the same numbers serve directly as 0-indexed
# pre-hook positions over Qwen2.5-VL-7B's 28-layer decoder.
QWEN_LAYER_LIST: tuple[int, ...] = (14,)
QWEN_IMAGE_TOKEN_RATIO_LIST: tuple[float, ...] = (0.333,)
QWEN_LAYER_RIDGE: float = 1e-1  # `layer_prune`'s stabiliser; stage A uses `ridge`


def stage_budgets(budget: int, *, compress_layer: int | None = None,
                  num_decoder_layers: int = DECODER_LAYERS) -> tuple[int, int | None]:
    """Map a paper (layer-aware average) budget to ``(tokens, visual_tokens)``.

    The reported budget is the average retained visual tokens per decoder layer::

        B = (visual * L + image * (N - L)) / N

    with the official 3:1 stage ratio (``visual = STAGE_RATIO * image``), pruning
    at ``L = DEFAULT_COMPRESS_LAYER`` of ``N = DECODER_LAYERS``. Solving for the
    final mid-LLM count gives ``image = B * N / (STAGE_RATIO * L + (N - L))``; the
    post-encoder count is ``STAGE_RATIO * image``, returned as ``None`` once it
    reaches the full grid (so only the layer stage runs). For L=16, N=32 this
    reproduces the official 288/96 (B=192) and 96/32 (B=64) script configs.

    ``compress_layer`` / ``num_decoder_layers`` restate the same solve for another
    decoder. The split is unchanged whenever the cut stays at the midpoint --
    ``denom`` is then ``2N`` at any depth -- which is why moving the cut with the
    depth is all a deeper backbone needs.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    depth = int(num_decoder_layers)
    layer = DEFAULT_COMPRESS_LAYER if compress_layer is None else int(compress_layer)
    denom = STAGE_RATIO * layer + (depth - layer)
    image = round(budget * depth / denom)
    visual = STAGE_RATIO * image
    return image, (visual if visual < FULL_VISUAL_TOKENS else None)


class ApETCompressor(TokenCompressor):
    """Approximation-error guided visual-token compression.

    Args:
        budget: **LLaVA-1.5 only.** Paper *reported* token budget (a layer-aware
            average, e.g. 64/128/192). The compressor fixes the decoder pruning
            layer to ``DEFAULT_COMPRESS_LAYER`` (16) and derives the two-stage
            split (post-encoder ``visual_tokens`` and final mid-LLM ``tokens``)
            from the official 3:1 ratio via ``stage_budgets``. For B=192 this is
            the official ``visual_tokens=288 -> tokens=96`` config.
        retain_ratio: **Qwen2.5-VL and native LLaVA-NeXT only, and required
            there.** Fraction of
            each image's visual tokens stage A keeps, upstream's
            parameterisation — a fixed count is meaningless when the NaViT
            encoder gives every image a different one. The published
            configuration is ``0.300``. Setting it selects the Qwen path and is
            mutually exclusive with ``budget``.
        layer_list: Qwen-only decoder boundaries (upstream's 1-indexed
            ``rank_layer``; layer ``L`` is the first to run reduced). ``None``
            takes the published ``[14]``; ``[]`` disables the layer stage,
            reproducing the released code drop verbatim (its ``layer_prune``
            call is commented out).
        image_token_ratio_list: Qwen-only fraction of the *current* image tokens
            each boundary keeps, one entry per ``layer_list`` boundary. ``None``
            takes the published ``[0.333]`` when ``layer_list`` is defaulted.
        basis_tokens: Number of basis tokens used for linear reconstruction.
            The paper's default is 10 on both backbones (clamped per image on
            Qwen, where tiny images can keep fewer tokens than that).
        selection_method: Basis-token sampler. FPS is the paper default.
        ridge: Diagonal stabilizer for the small linear solve (stage A / the
            LLaVA stages). The Qwen layer boundaries use ``layer_ridge``.
        layer_ridge: Qwen-only stabiliser of the boundary solve; upstream's
            ``layer_prune`` hardcodes ``1e-1`` there against the wider-spread
            mid-decoder activations.
        track_source: Record source matrices for ``llmoe.viz``. Required by
            ``bm --visualize``; off by default because it materializes an
            ``[N, N]`` matrix per forward.
    """

    def __init__(self, *, budget: int | None = None, basis_tokens: int = 10,
                 selection_method: SelectionMethod = "fps", ridge: float = 1e-5,
                 track_source: bool = False, retain_ratio: float | None = None,
                 layer_list: Sequence[int] | None = None,
                 image_token_ratio_list: Sequence[float] | None = None,
                 layer_ridge: float = QWEN_LAYER_RIDGE) -> None:
        if basis_tokens < 1:
            raise ValueError("basis_tokens must be >= 1")
        if selection_method not in ("fps", "dpc", "random"):
            raise ValueError("selection_method must be one of: fps, dpc, random")
        if ridge <= 0:
            raise ValueError("ridge must be > 0")


        configured_layer_list = layer_list is not None
        configured_image_ratios = image_token_ratio_list is not None
        # The backbone is not known until install(), so the two parameterisations
        # are validated here and cross-checked against the backbone there.
        if retain_ratio is not None:
            if budget is not None:
                raise ValueError(
                    "budget is ApET's LLaVA-1.5 parameterisation and retain_ratio its "
                    "Qwen2.5-VL one; pass one or the other, not both."
                )
            retain_ratio = float(retain_ratio)
            if not 0.0 < retain_ratio <= 1.0:
                raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
            if layer_ridge <= 0:
                raise ValueError("layer_ridge must be > 0")
            if layer_list is None:
                layer_list = QWEN_LAYER_LIST
                if image_token_ratio_list is None:
                    image_token_ratio_list = QWEN_IMAGE_TOKEN_RATIO_LIST
            elif image_token_ratio_list is None and not list(layer_list):
                image_token_ratio_list = ()
            layers = tuple(int(layer) for layer in layer_list)
            ratios = tuple(float(ratio) for ratio in (image_token_ratio_list or ()))
            if len(layers) != len(ratios):
                raise ValueError(
                    f"layer_list and image_token_ratio_list must have the same length, got "
                    f"{len(layers)} != {len(ratios)}"
                )
            if list(layers) != sorted(set(layers)) or any(layer < 1 for layer in layers):
                raise ValueError("layer_list must be strictly increasing and >= 1")
            if any(not 0.0 < ratio <= 1.0 for ratio in ratios):
                raise ValueError("image_token_ratio_list entries must be in (0, 1]")
            super().__init__(max_tokens=0)  # ratio mode: the retained count is per-image
            budget = None
            tokens = visual_tokens = compress_layer = None
        else:
            if budget is None:
                raise ValueError(
                    "budget is required on LLaVA-1.5. On Qwen2.5-VL pass `retain_ratio` "
                    "instead (the official configuration is 0.300)."
                )
            # The decoder pruning layer is fixed (paper LLaVA-1.5 config); the
            # two-stage split is derived from the reported (layer-aware) budget.
            tokens, visual_tokens = stage_budgets(budget)
            compress_layer = DEFAULT_COMPRESS_LAYER
            if basis_tokens >= tokens:
                raise ValueError("basis_tokens must be smaller than tokens")
            if visual_tokens is not None and basis_tokens >= visual_tokens:
                raise ValueError("basis_tokens must be smaller than visual_tokens")
            layers = ()
            ratios = ()
            super().__init__(max_tokens=tokens)

        self._budget: int | None = None if budget is None else int(budget)
        self._basis_tokens = int(basis_tokens)
        self._selection_method: SelectionMethod = selection_method
        self._ridge = float(ridge)
        self._track_source = bool(track_source)
        self._compress_layer: int | None = (
            None if compress_layer is None else int(compress_layer)
        )
        self._configured_compress_layer: int | None = self._compress_layer
        self._visual_tokens: int | None = (
            int(visual_tokens) if visual_tokens is not None else None
        )
        # Budget for the vision-feature stage; set in install() once the mode is
        # known. Defaults to the final budget so compress() works standalone.
        self._vision_budget: int = self._max_tokens

        # Qwen2.5-VL ratio mode, delegated to QwenApETAdapter at install().
        self._retain_ratio: float | None = retain_ratio
        self._layer_list: tuple[int, ...] = layers
        self._image_token_ratio_list: tuple[float, ...] = ratios
        self._layer_ridge: float = float(layer_ridge)
        self._configured_layer_list = bool(configured_layer_list)
        self._configured_image_ratios = bool(configured_image_ratios)
        self._qwen: Any | None = None
        self._internvl: Any | None = None
        self._llava_next: Any | None = None

        self._inner_model: Any | None = None
        self._vlm: VLM | None = None
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._image_token_id: int | None = None

        # Layer-pruning state (only used when compress_layer is set).
        self._language_model: Any | None = None
        self._hook_handles: list[RemovableHandle] = []
        self._spans: list[tuple[int, int] | None] = []
        self._layer_state: dict[str, Any] = {}

    @property
    @override
    def budget(self) -> int | float:
        # `max_tokens` holds the derived stage-1 count (see `stage_budgets`), so
        # report LLaVA's layer-average label or Qwen's dynamic ratio.
        if self._retain_ratio is not None:
            return self._retain_ratio
        if self._budget is None:
            raise RuntimeError("ApET's LLaVA budget has not been configured")
        return self._budget

    @staticmethod
    def _detect_backbone(vlm: VLM) -> str:
        """``"qwen"`` or ``"llava"``, checked structurally so forks still pass."""
        inner = getattr(getattr(vlm, "model", None), "model", None)
        if getattr(inner, "visual", None) is not None:
            return "qwen"
        if getattr(inner, "vision_tower", None) is not None:
            return "llava"
        raise TypeError(
            f"ApET supports the LLaVA-1.5, Qwen2.5-VL, and InternVL backbones; "
            f"{type(getattr(vlm, 'model', None)).__name__} exposes neither "
            f"`model.model.vision_tower` nor `model.model.visual`. Run with `--backbone llava` "
            f"or `--backbone qwen-vl`."
        )

    def validate_llava_next(self) -> None:
        """Expose only ApET's canonical NeXT scale as ``retain_ratio``."""

        from llmoe.compressor.llava_next import validate_retain_ratio

        ratio = validate_retain_ratio(self._retain_ratio, label="ApET")
        if (2.0 / 3.0) < ratio < 1.0:
            raise ValueError(
                "ApET's released LLaVA-NeXT schedule keeps one third after "
                "layer 16, so layer-average retain_ratio must be <= 2/3 (or "
                "exactly 1 for an identity pass-through)."
            )
        if (
            self._basis_tokens != 10
            or self._selection_method != "fps"
            or self._ridge != 1e-5
            or self._layer_ridge != QWEN_LAYER_RIDGE
            or self._configured_layer_list
            or self._configured_image_ratios
        ):
            raise ValueError(
                "ApET on LLaVA-NeXT fixes the released FPS sampler, ten basis "
                "tokens, ridge, 3:1 stage split, and layer-16 decoder cut; only "
                "retain_ratio is configurable."
            )

    @override
    def install(self, vlm: VLM) -> None:
        from llmoe.compressor.llava_next import is_llava_next

        if is_llava_next(vlm):
            self.validate_llava_next()
            if self._llava_next is not None:
                return
            from llmoe.compressor.apet.apet_llava_next import LlavaNextApETAdapter

            adapter = LlavaNextApETAdapter(
                self,
                vlm,
                retain_ratio=float(self._retain_ratio),
                track_source=self._track_source,
            )
            self._llava_next = adapter
            try:
                adapter.install()
            except Exception:
                self._llava_next = None
                raise
            return

        backbone = self._detect_backbone(vlm)
        if backbone == "qwen":
            if self._retain_ratio is None:
                raise ValueError(
                    "ApET on Qwen2.5-VL is parameterised by token ratios, not `budget`: the "
                    "NaViT encoder gives every image a different visual-token count, so a fixed "
                    "budget is not comparable across samples. Pass e.g. "
                    "`compressor_args: {retain_ratio: 0.3}` (the official configuration is "
                    "0.300 with layer_list [14] keeping 0.333)."
                )
            if self._qwen is not None:
                return  # idempotent: running several benchmarks must not stack hooks
            from llmoe.compressor.apet.apet_qwen import QwenApETAdapter

            self._qwen = QwenApETAdapter(
                retain_ratio=self._retain_ratio,
                layer_list=self._layer_list,
                image_token_ratio_list=self._image_token_ratio_list,
                basis_tokens=self._basis_tokens,
                selection_method=self._selection_method,
                ridge=self._ridge,
                layer_ridge=self._layer_ridge,
                track_source=self._track_source,
            )
            self._qwen.install(vlm, image_token_id=self._resolve_image_token_id(vlm))
            return

        if self._retain_ratio is not None:
            raise ValueError(
                "retain_ratio selects ApET's Qwen2.5-VL path. On "
                "LLaVA-1.5/InternVL use `budget`: "
                "the released LLaVA configs are absolute layer-averaged budgets (64/128/192), so "
                "a ratio would not reproduce the paper's numbers."
            )

        from llmoe.compressor.apet.apet_internvl import InternVLApETAdapter

        if InternVLApETAdapter.supports(vlm):
            if self._internvl is not None:
                return
            self._internvl = InternVLApETAdapter(self)
        if self._compress_layer is not None:
            # The paper's cut is the middle of the decoder, not layer 16 as such,
            # so it moves with the depth -- 16 of Vicuna-7B's 32, 20 of the 13B's
            # 40. The two-stage split needs no re-solve: at the midpoint it is
            # `budget / 2` whatever the depth (see `stage_budgets`).
            self._compress_layer = scale_layer(self._compress_layer, decoder_depth(vlm))
        if self._compress_layer is None:
            # Vision-feature stage only: reduce the full visual tokens to tokens.
            self._vision_budget = self._max_tokens
            self._wrap_get_image_features(vlm)
            self._install_prepare_inputs(vlm, trim=True, record=False)
            return

        two_stage = self._visual_tokens is not None
        # Stage A (optional): vision stage reduces to visual_tokens; otherwise
        # the full visual tokens flow into the LLM and only the layer prunes.
        self._vision_budget = self._visual_tokens if two_stage else self._max_tokens
        if two_stage:
            self._wrap_get_image_features(vlm)
        # Stage B: middle-layer pruning down to tokens.
        self._install_layer_hooks(vlm)
        self._install_prepare_inputs(vlm, trim=two_stage, record=True)

    def _install_prepare_inputs(self, vlm: VLM, *, trim: bool, record: bool) -> None:
        """Wrap prepare_inputs to (optionally) trim image placeholders to the
        vision-stage budget and/or record per-row visual spans for the hooks."""
        if getattr(vlm, "_apet_prepare_wrapped", False):
            return
        self._vlm = vlm
        self._original_prepare_inputs = vlm.prepare_inputs

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            inputs = self._original_prepare_inputs(*args, **kwargs)
            if trim:
                inputs = self._trim_image_placeholders(vlm, inputs)
            if record:
                input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
                if input_ids is not None:
                    self._record_spans(vlm, input_ids)
            return inputs

        vlm.prepare_inputs = wrapped_prepare_inputs
        vlm._apet_prepare_wrapped = True

    def uninstall(self) -> None:
        if self._llava_next is not None:
            self._llava_next.uninstall()
            self._llava_next = None
        if self._qwen is not None:
            self._qwen.uninstall()
            self._qwen = None
        if self._internvl is not None:
            self._internvl.uninstall()
            self._internvl = None
        if self._inner_model is not None and self._original_get_image_features is not None:
            self._inner_model.get_image_features = self._original_get_image_features
            self._inner_model._apet_wrapped = False
        if self._vlm is not None and self._original_prepare_inputs is not None:
            self._vlm.prepare_inputs = self._original_prepare_inputs
            self._vlm._apet_prepare_wrapped = False
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._layer_state = {}
        self._spans = []
        self._inner_model = None
        self._vlm = None
        self._language_model = None
        self._original_get_image_features = None
        self._original_prepare_inputs = None

        self._compress_layer = self._configured_compress_layer

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        if self._qwen is not None:
            return self._qwen.get_source_history()
        return list(self._source_history)

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, **kwargs: Any) -> CompressedFeatures:
        del attention, layer_idx, kwargs
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.dim() != 3:
            raise ValueError(f"features must be [B, N, C] or [N, C], got {tuple(features.shape)}")

        batch, num_tokens, _ = features.shape
        if self._retain_ratio is not None:
            # Ratio mode (Qwen2.5-VL): stage-A semantics, sized per image. The
            # real vision path runs through QwenApETAdapter; this entry point keeps
            # tensor-level tests working.
            keep = max(1, int(self._retain_ratio * num_tokens))
        else:
            # The vision-feature stage reduces to _vision_budget (the final budget
            # when single-stage, or the first-stage budget when paired with a layer).
            keep = self._vision_budget
        if keep >= num_tokens:
            self._record_identity(num_tokens, features.device)
            return CompressedFeatures(tokens=features.squeeze(0) if squeeze else features)

        if self._retain_ratio is not None:
            basis_count = max(1, min(self._basis_tokens, keep, num_tokens))
        else:
            basis_count = min(self._basis_tokens, keep - 1, num_tokens)
        if basis_count < 1:
            raise ValueError("ApET needs at least one basis token")

        work = features.float()

        out_rows: list[torch.Tensor] = []
        source_rows: list[torch.Tensor] = []

        for b in range(batch):
            if self._retain_ratio is not None:
                kept, _, source = _qwen_prune_block(
                    work[b], keep, basis_count, selection_method=self._selection_method,
                    ridge=self._ridge, ascending_blend=True,
                )
            else:
                kept, _, source = self._prune_block(work[b], keep, basis_count)
            out_rows.append(kept)
            source_rows.append(source)

        out = torch.stack(out_rows, dim=0).to(features.dtype)
        if self._track_source:
            self._source_history = [
                torch.eye(num_tokens, dtype=torch.float32),
                source_rows[0].detach().cpu().float(),
            ]
        return CompressedFeatures(tokens=out.squeeze(0) if squeeze else out)

    def _prune_block(self, block: torch.Tensor, keep: int, basis_count: int
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Faithful ApET prune of one visual-token block ``[N, C]`` (float).

        Mirrors the official ``layer_prune`` + ``token_merging``:

        * select ``basis_count`` seed tokens (FPS by default) and rank every
          token by linear-reconstruction error,
        * keep the ``keep`` highest-error tokens,
        * overwrite the ``basis_count`` lowest-error *kept* slots with the seed
          tokens (these basis tokens are kept but are not merge targets),
        * merge every dropped token into the most similar of the remaining
          ``keep - basis_count`` retained tokens (average merging).

        Returns ``(kept[keep, C], kept_idx_sorted[keep], source[keep, N])``,
        where rows are in ascending original position and ``source`` maps each
        kept token to the original tokens it now subsumes (for ``llmoe.viz``).
        """
        num_tokens, _ = block.shape
        work = block.clone()

        basis, basis_source, _ = _select_basis(
            work.unsqueeze(0), basis_count, self._selection_method)
        basis = basis[0]
        basis_source = basis_source[0]

        errors = _approximation_errors(work.unsqueeze(0), basis.unsqueeze(0), self._ridge)[0]
        errors_indices = torch.topk(errors, k=keep).indices
        basis_slots = errors_indices[-basis_count:]
        merge_targets = errors_indices[:-basis_count]

        work[basis_slots] = basis

        eye = torch.eye(num_tokens, dtype=torch.float32, device=block.device)
        source = torch.zeros(num_tokens, num_tokens, dtype=torch.float32, device=block.device)
        source[merge_targets] = eye[merge_targets]
        source[basis_slots] = basis_source

        selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=block.device)
        selected_mask[errors_indices] = True
        dropped = (~selected_mask).nonzero(as_tuple=True)[0]

        if dropped.numel() > 0 and merge_targets.numel() > 0:
            target_values = work[merge_targets]
            similarity = F.cosine_similarity(
                work[dropped].unsqueeze(1),
                target_values.unsqueeze(0),
                dim=2,
            )
            nearest = similarity.argmax(dim=1)
            counts = torch.ones(merge_targets.numel(), 1, dtype=work.dtype, device=block.device)
            merged = target_values.clone()
            merged.index_add_(0, nearest, work[dropped])
            counts.index_add_(
                0,
                nearest,
                torch.ones(dropped.numel(), 1, dtype=work.dtype, device=block.device),
            )
            work[merge_targets] = merged / counts

            merged_source = source[merge_targets].clone()
            merged_source.index_add_(0, nearest, eye[dropped])
            source[merge_targets] = merged_source / counts.float()

        kept_order = errors_indices.sort().values
        return work[kept_order], kept_order, source[kept_order]

    # ------------------------------------------------------------------
    # Middle-layer LLM pruning (faithful to the paper's ``layer_list`` path)
    # ------------------------------------------------------------------
    def _install_layer_hooks(self, vlm: VLM) -> None:
        if self._hook_handles:  # already installed
            return
        self._vlm = vlm
        language_model = vlm.model.model.language_model
        self._language_model = language_model
        num_layers = len(language_model.layers)
        if self._compress_layer >= num_layers:
            raise ValueError(
                f"compress_layer={self._compress_layer} exceeds the decoder depth ({num_layers})"
            )

        # Pruning at layer L makes the sequence shorter for every layer >= L.
        # In modern transformers the rotary embeddings and attention mask are
        # computed once and broadcast to all layers, so each downstream layer
        # needs the pruned mask/positions re-supplied via its own pre-hook.
        self._layer_state = {}
        for layer_idx in range(self._compress_layer, num_layers):
            layer = language_model.layers[layer_idx]
            handle = layer.register_forward_pre_hook(
                self._make_layer_hook(layer_idx), with_kwargs=True)
            self._hook_handles.append(handle)

    def _record_spans(self, vlm: VLM, input_ids: torch.Tensor) -> None:
        image_token_id = self._resolve_image_token_id(vlm)
        spans: list[tuple[int, int] | None] = []
        for row in range(input_ids.shape[0]):
            positions = (input_ids[row] == image_token_id).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                spans.append(None)
            else:
                spans.append((int(positions[0]), int(positions.numel())))
        self._spans = spans

    def _make_layer_hook(self, layer_idx: int) -> Any:
        target = self._compress_layer

        def hook(module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
            hidden_states = args[0]
            if layer_idx == target:
                if hidden_states.shape[1] != 1:
                    hidden_states = self._prefill_prune(hidden_states, kwargs)
                else:
                    self._decode_update(hidden_states, kwargs)
            # Read state *after* the target layer has (re)built it for this pass.
            state = self._layer_state
            if not state:
                # No span info (e.g. text-only batch); leave the layer untouched.
                return (hidden_states,), kwargs
            kwargs["position_embeddings"] = state["pos_emb"]
            kwargs["attention_mask"] = state["mask"]
            kwargs["position_ids"] = state["pids"]
            return (hidden_states,), kwargs

        return hook

    def _prefill_prune(self, hidden_states: torch.Tensor, kwargs: dict) -> torch.Tensor:
        device = hidden_states.device
        dtype = hidden_states.dtype
        batch, seq_len, _ = hidden_states.shape
        budget = self._max_tokens

        if not self._spans or len(self._spans) != batch or any(s is None for s in self._spans):
            # Cannot locate a uniform image block for every row; skip pruning.
            self._layer_state = {}
            return hidden_states

        new_rows: list[torch.Tensor] = []
        keep_rows: list[torch.Tensor] = []
        for b in range(batch):
            start, count = self._spans[b]
            keep = max(1, min(budget, count))
            if count <= keep:
                kept_block = hidden_states[b, start: start + count]
                local = torch.arange(count, device=device)
            else:
                basis_count = min(self._basis_tokens, keep - 1, count)
                block = hidden_states[b, start: start + count].float()
                kept_block, local, _ = self._prune_block(block, keep, basis_count)
                kept_block = kept_block.to(dtype)
            global_keep = torch.cat(
                [
                    torch.arange(start, device=device),
                    start + local,
                    torch.arange(start + count, seq_len, device=device),
                ]
            )
            new_row = torch.cat(
                [hidden_states[b, :start], kept_block, hidden_states[b, start + count:]],
                dim=0,
            )
            new_rows.append(new_row)
            keep_rows.append(global_keep)

        lengths = {row.shape[0] for row in new_rows}
        if len(lengths) != 1:
            raise RuntimeError(
                "ApET layer pruning needs a uniform kept length across the batch; "
                f"got {sorted(lengths)}. This happens when images contribute different "
                "visual-token counts; pad the batch or run images of equal token count."
            )

        new_hidden = torch.stack(new_rows, dim=0)
        keep_index = torch.stack(keep_rows, dim=0)  # [B, L] original positions kept
        new_len = new_hidden.shape[1]

        mask, valid, kind = self._build_prefill_mask(
            kwargs.get("attention_mask"), keep_index, seq_len, dtype
        )
        position_ids = torch.arange(new_len, device=device).unsqueeze(0).expand(batch, new_len)
        cos, sin = self._language_model.rotary_emb(new_hidden, position_ids=position_ids)

        self._layer_state = {
            "pos_emb": (cos, sin),
            "mask": mask,
            "pids": position_ids,
            "valid": valid,
            "mask_kind": kind,
            "prefill_len": new_len,
            "decode_step": -1,
        }
        return new_hidden

    def _build_prefill_mask(self, attention_mask: torch.Tensor | None, keep_index: torch.Tensor,
                            seq_len: int, dtype: torch.dtype
                            ) -> tuple[torch.Tensor | None, torch.Tensor, str]:
        """Build the pruned attention mask, mirroring the incoming format.

        Different attention backends pass different mask formats (eager: additive
        float 4D; SDPA: boolean 4D; flash: 2D padding; or ``None``). To match the
        model exactly we keep the same kind rather than normalising, selecting the
        kept query rows / key columns from the model's own mask where possible.

        Returns ``(mask, valid, kind)`` where ``valid`` is a ``[B, L]`` boolean of
        attendable kept keys (reused during decode) and ``kind`` records the
        format so decode can rebuild a matching mask.
        """
        batch, new_len = keep_index.shape
        device = keep_index.device

        if attention_mask is not None and attention_mask.dim() == 4:
            am = torch.gather(
                attention_mask, 2, keep_index[:, None, :, None].expand(batch, 1, new_len, seq_len)
            )
            am = torch.gather(
                am, 3, keep_index[:, None, None, :].expand(batch, 1, new_len, new_len)
            )
            if am.dtype == torch.bool:
                valid = am[:, 0, -1, :]
                return am, valid, "bool4d"
            neg = torch.finfo(am.dtype).min
            valid = am[:, 0, -1, :] > (neg / 2)
            return am, valid, "float4d"

        if attention_mask is not None and attention_mask.dim() == 2:
            # 2D padding mask [B, S] (1 = keep). Select the kept keys.
            pad = attention_mask.gather(1, keep_index)
            return pad, pad.bool(), "pad2d"

        # No mask supplied: rely on the backend's causal default.
        valid = torch.ones(batch, new_len, dtype=torch.bool, device=device)
        return None, valid, "none"

    def _decode_update(self, hidden_states: torch.Tensor, kwargs: dict) -> None:
        state = self._layer_state
        if not state:
            return
        device = hidden_states.device
        dtype = hidden_states.dtype
        batch = hidden_states.shape[0]
        state["decode_step"] += 1
        position = state["prefill_len"] + state["decode_step"]
        position_ids = torch.full((batch, 1), position, dtype=torch.long, device=device)
        cos, sin = self._language_model.rotary_emb(hidden_states, position_ids=position_ids)

        new_col = torch.ones(batch, 1, dtype=torch.bool, device=device)
        valid = torch.cat([state["valid"], new_col], dim=1)  # [B, kv_len]
        kind = state.get("mask_kind", "float4d")
        if kind == "none":
            mask: torch.Tensor | None = None
        elif kind == "pad2d":
            mask = valid.to(dtype=torch.long)
        elif kind == "bool4d":
            mask = valid[:, None, None, :]
        else:  # float4d
            neg = torch.finfo(dtype).min
            mask = torch.where(
                valid[:, None, None, :],
                torch.zeros((), dtype=dtype, device=device),
                torch.full((), neg, dtype=dtype, device=device),
            )
        state["pos_emb"] = (cos, sin)
        state["mask"] = mask
        state["pids"] = position_ids
        state["valid"] = valid

    def _wrap_get_image_features(self, vlm: VLM) -> None:
        inner = vlm.model.model
        if getattr(inner, "_apet_wrapped", False):
            return

        if self._internvl is not None:
            self._internvl.install(vlm, wrap_prepare_inputs=False)
            return

        self._inner_model = inner
        self._original_get_image_features = inner.get_image_features
        if hasattr(inner, "vision_tower") and hasattr(inner, "multi_modal_projector"):
            inner.get_image_features = self._llava_get_image_features(inner)
        else:
            inner.get_image_features = self._generic_get_image_features(inner)
        inner._apet_wrapped = True

    def _llava_get_image_features(self, inner: Any) -> Any:
        def wrapped_get_image_features(pixel_values: torch.FloatTensor,
                                       vision_feature_layer: int | list[int] | None = None,
                                       vision_feature_select_strategy: str | None = None,
                                       output_hidden_states: bool | None = None,
                                       **kwargs: Any) -> Any:
            del vision_feature_select_strategy, output_hidden_states
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            kwargs.pop("return_dict", None)
            kwargs.pop("output_hidden_states", None)
            image_outputs = inner.vision_tower(
                pixel_values,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

            if vision_feature_layer is None:
                vision_feature_layer = getattr(inner.config, "vision_feature_layer", None)
            if vision_feature_layer is None:
                vision_feature_layer = getattr(inner.config, "vision_feature_select_layer", -2)

            if isinstance(vision_feature_layer, int):
                selected = image_outputs.hidden_states[vision_feature_layer][:, 1:, :]
            else:
                selected = torch.cat(
                    [image_outputs.hidden_states[idx][:, 1:, :] for idx in vision_feature_layer],
                    dim=-1,
                )

            compressed = self.compress(selected).tokens
            projected = inner.multi_modal_projector(compressed)
            image_outputs.pooler_output = list(projected)
            return image_outputs

        return wrapped_get_image_features

    def _generic_get_image_features(self, inner: Any) -> Any:
        original_get_image_features = self._original_get_image_features

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            image_outputs = original_get_image_features(*args, **kwargs)
            pooler_output = image_outputs.pooler_output
            compressed = [
                self.compress(tokens.unsqueeze(0)).tokens.squeeze(0)
                for tokens in list(pooler_output)
            ]
            image_outputs.pooler_output = compressed
            return image_outputs

        return wrapped_get_image_features

    def _trim_image_placeholders(self, vlm: VLM, inputs: Any) -> Any:
        """Trim every source-image run to ApET's one global vision-stage budget."""
        if self._internvl is not None:
            return self._internvl.trim_inputs(vlm, inputs)
        return trim_image_placeholders(
            inputs,
            token_id=self._resolve_image_token_id(vlm),
            keep_for_run=lambda run_length: min(self._vision_budget, run_length),
            pad_token_id=self._pad_token_id(vlm),
        )

    def _resolve_image_token_id(self, vlm: VLM) -> int:
        if self._image_token_id is not None:
            return self._image_token_id

        config = getattr(vlm.model, "config", None)
        token_id = getattr(config, "image_token_id", None)
        if token_id is None:
            token_id = getattr(config, "image_token_index", None)
        if token_id is None:
            processor = vlm.processor
            image_token = getattr(processor, "image_token", None)
            tokenizer = getattr(processor, "tokenizer", None)
            if image_token is not None and tokenizer is not None:
                token_id = tokenizer.convert_tokens_to_ids(str(image_token))
        if token_id is None:
            raise RuntimeError("Could not resolve the image placeholder token id for ApET.")

        self._image_token_id = int(token_id)
        return self._image_token_id

    @staticmethod
    def _pad_token_id(vlm: VLM) -> int:
        tokenizer = getattr(vlm.processor, "tokenizer", None)
        pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        return int(pad) if pad is not None else 0

    @staticmethod
    def _left_pad_stack(rows: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(row.shape[0] for row in rows)
        padded: list[torch.Tensor] = []
        for row in rows:
            if row.shape[0] < max_len:
                pad = torch.full(
                    (max_len - row.shape[0],),
                    pad_value,
                    dtype=row.dtype,
                    device=row.device,
                )
                row = torch.cat([pad, row], dim=0)
            padded.append(row)
        return torch.stack(padded, dim=0)

    def _record_identity(self, num_tokens: int, device: torch.device) -> None:
        if self._track_source:
            self._source_history = [torch.eye(num_tokens, dtype=torch.float32, device=device).cpu()]


def _gather_basis(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_idx = indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
    return features.gather(1, gather_idx)


def _one_hot_source(indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    source = torch.zeros(
        indices.shape[0],
        indices.shape[1],
        num_tokens,
        dtype=torch.float32,
        device=indices.device,
    )
    source.scatter_(2, indices.unsqueeze(-1), 1.0)
    return source


def _farthest_point_sampling(features: torch.Tensor, count: int) -> torch.Tensor:
    batch, num_tokens, _ = features.shape
    centroids = torch.zeros(batch, count, dtype=torch.long, device=features.device)
    distances = torch.full((batch, num_tokens), 1e10, dtype=features.dtype, device=features.device)
    farthest = torch.randint(0, num_tokens, (batch,), device=features.device)

    batch_indices = torch.arange(batch, device=features.device)
    for i in range(count):
        centroids[:, i] = farthest
        centroid = features[batch_indices, farthest].unsqueeze(1)
        current_dist = ((features - centroid) ** 2).sum(-1)
        distances = torch.minimum(distances, current_dist)
        farthest = distances.max(dim=1).indices
    return centroids


def _density_peak_basis(features: torch.Tensor, count: int
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, num_tokens, channels = features.shape
    dist_matrix = torch.cdist(features.float(), features.float()) / sqrt(channels)

    nearest = torch.topk(dist_matrix, k=count, dim=-1, largest=False).values
    density = (-(nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand_like(density) * 1e-6

    higher_density = density[:, None, :] > density[:, :, None]
    dist_max = dist_matrix.flatten(1).max(dim=-1).values[:, None, None]
    dist_to_higher = (
        dist_matrix * higher_density + dist_max * (~higher_density)
    ).min(dim=-1).values
    centers = torch.topk(dist_to_higher * density, k=count, dim=-1).indices

    center_dist = torch.gather(
        dist_matrix,
        1,
        centers.unsqueeze(-1).expand(-1, -1, num_tokens),
    )
    clusters = center_dist.argmin(dim=1)

    batch_idx = torch.arange(batch, device=features.device)[:, None].expand(batch, count)
    cluster_idx = torch.arange(count, device=features.device)[None, :].expand(batch, count)
    clusters[batch_idx.reshape(-1), centers.reshape(-1)] = cluster_idx.reshape(-1)

    basis = torch.zeros(batch, count, channels, dtype=features.dtype, device=features.device)
    source = torch.zeros(batch, count, num_tokens, dtype=torch.float32, device=features.device)
    for b in range(batch):
        for c in range(count):
            members = (clusters[b] == c).nonzero(as_tuple=True)[0]
            if members.numel() == 0:
                fallback = centers[b, c]
                basis[b, c] = features[b, fallback]
                source[b, c, fallback] = 1.0
                continue
            weight = 1.0 / float(members.numel())
            basis[b, c] = features[b, members].mean(dim=0)
            source[b, c, members] = weight
    # ``centers`` are real token positions (the density peaks) and serve as the
    # anchor positions for each synthesized cluster-average basis token.
    return basis, source, centers


def _approximation_errors(features: torch.Tensor, basis: torch.Tensor,
                          ridge: float) -> torch.Tensor:
    """Linear-reconstruction error of every token against ``basis`` (paper Eq. 2-4)."""
    gram = torch.matmul(basis, basis.transpose(-1, -2))
    diag = gram.diagonal(dim1=-2, dim2=-1)
    diag.add_(ridge)
    rhs = torch.matmul(basis, features.transpose(-1, -2))
    try:
        coeff = torch.linalg.solve(gram, rhs).transpose(-1, -2)
    except torch.linalg.LinAlgError:
        # Duplicate seed tokens (flat colour regions give identical embeddings)
        # make the Gram matrix singular beyond what the ridge can absorb. The
        # pseudo-inverse yields the minimum-norm coefficients the ridge is
        # meant to stabilise, so the error ranking is unchanged where the solve
        # succeeds and well-defined where it would have crashed.
        coeff = torch.matmul(torch.linalg.pinv(gram), rhs).transpose(-1, -2)
    reconstructed = torch.matmul(coeff, basis)
    return torch.norm(features - reconstructed, dim=-1)


def _select_basis(features: torch.Tensor, count: int, method: SelectionMethod
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``count`` seed tokens from ``[B, N, C]`` features by ``method``."""
    if method == "fps":
        num_tokens = features.shape[1]
        indices = _farthest_point_sampling(features, count)
        return _gather_basis(features, indices), _one_hot_source(indices, num_tokens), indices
    if method == "random":
        batch, num_tokens, _ = features.shape
        indices = torch.randint(0, num_tokens, (batch, count), device=features.device)
        return _gather_basis(features, indices), _one_hot_source(indices, num_tokens), indices
    return _density_peak_basis(features, count)


def _qwen_prune_block(block: torch.Tensor, keep: int, basis_count: int, *,
                      selection_method: SelectionMethod, ridge: float,
                      ascending_blend: bool
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Faithful ApET Qwen prune of one visual-token block ``[N, C]`` (float32).

    Mirrors the official Qwen release (``Qwen2_5_VLForConditionalGeneration_X``'s
    inline stage A and ``Qwen2_5_VLModel_X.layer_prune`` + ``token_merging``),
    which differs from the LLaVA drop's ``_prune_block`` in one step:

    * select ``basis_count`` seed tokens (FPS by default) and rank every token
      by linear-reconstruction error under ``ridge``,
    * keep the ``keep`` highest-error tokens,
    * **blend** each seed 50/50 into one of the ``basis_count`` lowest-error
      *kept* slots (the LLaVA drop overwrites them instead). Stage A pairs the
      slots in ascending index order (upstream writes through a boolean
      ``index_mask``); the layer stage pairs them in top-k order
      (``top_rank_index[-k:]``) — ``ascending_blend`` selects between the two.
      Blended slots are kept but are not merge targets,
    * merge every dropped token into the most similar of the remaining
      ``keep - basis_count`` retained tokens (count-average, upstream's
      ``token_merging`` with ``scaling=1``).

    Returns ``(kept[keep, C], kept_idx_sorted[keep], source[keep, N])``, rows in
    ascending original position; ``source`` maps each kept token to the original
    tokens it now subsumes (for ``llmoe.viz``).
    """
    num_tokens, _ = block.shape
    work = block.clone()

    basis, basis_source, _ = _select_basis(work.unsqueeze(0), basis_count, selection_method)
    basis = basis[0]
    basis_source = basis_source[0]

    errors = _approximation_errors(work.unsqueeze(0), basis.unsqueeze(0), ridge)[0]
    errors_indices = torch.topk(errors, k=keep).indices
    basis_slots = errors_indices[-basis_count:]
    merge_targets = errors_indices[:-basis_count]
    if ascending_blend:
        basis_slots = basis_slots.sort().values

    eye = torch.eye(num_tokens, dtype=torch.float32, device=block.device)
    source = eye.clone()
    work[basis_slots] = 0.5 * basis + 0.5 * work[basis_slots]
    source[basis_slots] = 0.5 * basis_source + 0.5 * eye[basis_slots]

    selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=block.device)
    selected_mask[errors_indices] = True
    dropped = (~selected_mask).nonzero(as_tuple=True)[0]

    if dropped.numel() > 0 and merge_targets.numel() > 0:
        target_values = work[merge_targets]
        similarity = F.cosine_similarity(
            work[dropped].unsqueeze(1),
            target_values.unsqueeze(0),
            dim=2,
        )
        nearest = similarity.argmax(dim=1)
        counts = torch.ones(merge_targets.numel(), 1, dtype=work.dtype, device=block.device)
        merged = target_values.clone()
        merged.index_add_(0, nearest, work[dropped])
        counts.index_add_(
            0,
            nearest,
            torch.ones(dropped.numel(), 1, dtype=work.dtype, device=block.device),
        )
        work[merge_targets] = merged / counts

        merged_source = source[merge_targets].clone()
        merged_source.index_add_(0, nearest, eye[dropped])
        source[merge_targets] = merged_source / counts.float()

    kept_order = errors_indices.sort().values
    return work[kept_order], kept_order, source[kept_order]


