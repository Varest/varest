"""Nuwa: training-free two-stage visual-token pruning that keeps the image intact.

Paper: Huang et al., "Nuwa: Mending the Spatial Integrity Torn by VLM Token
Pruning" (arXiv:2602.02951). Reference code:
https://github.com/Man-PaperRejected/Nuwa.

One compressor supports LLaVA-1.5, native LLaVA-NeXT, Qwen2.5-VL, and InternVL;
``install()`` selects the matching implementation. They are parameterised
differently: LLaVA-1.5 and InternVL retain their released ``max_tokens``
profiles, while Qwen and native NeXT use ``retain_ratio`` because their visual
span is image-dependent. NeXT exposes no other method knob: its released
distance, regional selection, pillar threshold, layer-16 cut, and 10% deep
retention are fixed internally. Their shared selection policy lives in
``policy.py`` and shared decoder geometry in ``_pruning.py``. The rest of this
docstring describes the LLaVA-1.5 path.

Attention-only pruners rank patches by how much the CLS token looks at them and
keep the winners, which reliably collapses the survivors onto one or two salient
blobs and throws away the layout of everything else. Nuwa fixes that in two
places:

* **Stage 1** (post-vision-encoder). The 24x24 patch grid is cut into 2x2
  regions and each region must contribute candidates, so coverage survives.
  The candidates are ranked globally by CLS attention down to the Stage-1 budget
  (~1.75x the reported budget), and the patches that did not make it are merged
  into the ones that did, weighted by ``ReLU(cosine similarity) * spatial
  proximity``. The highest-attention survivors ("Pillars") keep their own
  features untouched; the rest ("Collectors") absorb their neighbourhood.

* **Stage 2** (decoder layer 16). Stage 1 only ever saw the image, so the second
  cut is made once the prompt is in play: the last text token's attention over
  the visual positions, read off layer 15, ranks them and the top quarter of the
  reported budget survives.

* **RPME** (Relative Position Mapping Extension). Survivors are re-addressed with
  their *original* patch indices as ``position_ids``, so RoPE still sees the true
  spatial distance between two patches that used to be far apart, and the text
  after the image is shifted by ``num_patches - N_stage1`` to close the gap.

The reported budget is the layer-weighted average of the two stages,
``budget = (N_stage1 + N_stage2) / 2``, confirmed by the authors: a budget of 64
means 112 tokens through layers 0-15 and 16 tokens after that.

Documented deviations from the paper
------------------------------------
The official code differs from the paper in several places. This is a
reimplementation of the **official code**, for baseline comparability, and every
divergence is reproduced deliberately:

===  ==========================================  ==============================================
 #   Paper                                       Official code (= this implementation)
===  ==========================================  ==============================================
 1   ``S(t_i) = a_cls,i * ||k_i||_2`` (Eq. 1)    CLS attention summed over heads; no norm factor
 2   ``M x M`` grid partitioning (Sec 3.2)       ``M = 2``, hardcoded
 3   Pillars = 75th pctile of key norms (Eq. 4)  ``PENALTY_THRESHOLD_PERCENTILE = 0.55`` on CLS attention
 4   Weights normalised to sum 1 (Eq. 7)         self-weight overridden to 1.0 *after* normalising
 5   ``q`` = mean text embedding, cosine (3.4)   last text token's raw attention from layer 15
 6   ``A_ij = cos(f_i, f_j)`` (Eq. 5)            ``ReLU(cos)`` -- negatives zeroed
===  ==========================================  ==============================================

Why this needs more than ``compress(features)``
-----------------------------------------------
Stage 1 wants the vision tower's CLS attention and a stack of intermediate hidden
states, Stage 2 lives inside the decoder, and RPME rewrites ``position_ids`` --
none of which reach the default ``wrap_get_image_features`` path. ``install()``
therefore wires four things and ``uninstall()`` unwinds all of them:

1. a pre-hook asking the vision tower for attentions, plus eager attention there
   (SDPA returns no weights);
2. a ``get_image_features`` wrapper running Stage 1 and re-projecting;
3. a ``prepare_inputs`` wrapper trimming the ``<image>`` placeholders to the
   Stage-1 budget (LLaVA asserts placeholders == features);
4. a custom attention function on the decoder that runs eager at the capture
   layer only, an RPME pre-hook on ``rotary_emb``, and Stage-2 pre-hooks from the
   prune layer down.

Harness adaptations
-------------------
* **Stage 2 deletes, through hooks.** The official code vendors a patched
  ``modeling_llama.py`` to drop the rows; this harness drives a stock
  ``LlavaForConditionalGeneration``, so the cut is made in a pre-hook that
  rewrites ``hidden_states`` and re-supplies the ``position_ids``,
  ``position_embeddings``, ``attention_mask`` and ``cache_position`` that the
  decoder built once for the full sequence. Tokens really are gone from the prune
  layer down, so the deeper layers' FLOPs and KV cache shrink with the budget.
  The cache is ragged as a result -- full length above the cut, short below it --
  which is why the decode path rebuilds the mask at the narrower width each step.
* **Eager only where it is needed.** Capturing layer 15's weights needs eager
  attention; the registered attention function delegates every other layer back
  to SDPA rather than making the whole decoder eager.
* **Batching.** The reference assumes one image per forward. Here the LLM-side
  stages are per-row, so ``--batch-size > 1`` works as long as the rows carry the
  same number of visual tokens; a ragged batch warns once and falls back to
  Stage 1 alone. Beam search expands the batch after prefill, which a ragged cache
  cannot follow, so it raises rather than silently producing garbage.
"""

import math
from typing import Any, Sequence, override

import torch
from torch.utils.hooks import RemovableHandle

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    advance_decode_step,
    apply_prune_state,
    gather_position_ids,
    keep_columns,
    pruned_attention_mask,
    select_rope_tables,
)
from llmoe.compressor.nuwa.nuwa_internvl import (
    NEXT_PILLAR_PERCENTILE,
    InternVLNuwaAdapter,
    next_stage1_count,
    next_top_n,
)
from llmoe.compressor.nuwa.policy import (
    average_hidden_states,
    get_stage2_keep_count,
    nuwa_stage1,
    register_capture_attention,
    select_by_text_attention,
    set_attn_implementation,
    unregister_capture_attention,
)
from llmoe.compressor.nuwa.nuwa_qwen import (
    DEFAULT_ATTENTION_CHUNK,
    QWEN_DISTANCE_RATIO,
    QWEN_PILLAR_PERCENTILE,
    QWEN_PROFILES,
    QWEN_SIM_BLOCKS,
    QWEN_STAGE2_LAYER,
    QwenNuwaAdapter,
)
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.utils.depth import REFERENCE_DEPTH, scale_layer
from llmoe.utils.qwen import resolve_image_token_id as shared_resolve_image_token_id

# Official released Stage-1 budgets. Each is `round(budget * 7 / 4)`, which is the
# fallback for any budget not listed here (32 and 224 are author-recommended
# extensions of the released 64/128/192 configs).
STAGE1_CONFIGS: dict[int, int] = {
    32: 56,
    64: 112,
    128: 224,
    192: 336,
    224: 392,
}

# Candidates kept per 2x2 region, `ceil(stage1_count / num_regions)` on the 24x24
# grid (144 regions). Listed for the official budgets, computed otherwise.
TOPN_CONFIGS: dict[int, int] = {
    32: 1,
    64: 1,
    128: 2,
    192: 3,
    224: 3,
}

# Official default: the penalty matrix zeroes out patches more than sqrt(280)
# ~= 16.7 patch-widths apart.
DEFAULT_DISTANCE: float = 280.0
DEFAULT_STAGE2_LAYER: int = 16
# The official similarity feature is the mean of these vision-tower hidden states.
DEFAULT_SIM_LAYERS: tuple[int, int] = (16, 25)


def resolve_image_token_id(config: Any, processor: Any) -> int:
    """Resolve the ``<image>`` placeholder id, mirroring LLaVA's own lookup order."""
    return shared_resolve_image_token_id(config, processor, label="Nuwa")


class NuwaCompressor(TokenCompressor):
    """Two-stage spatial-integrity-preserving visual-token pruner.

    :param max_tokens: the reported budget, i.e. the layer-weighted average of the
        two stages. **LLaVA-1.5 and InternVL only.** 32/64/128/192/224 use the
        released LLaVA configs and 160/320/640 the released LLaVA-NeXT ones;
        anything else is derived from the same split. Not consulted on
        Qwen2.5-VL, whose encoder emits a different token count per image.
    :param retain_ratio: **Qwen2.5-VL and native LLaVA-NeXT only.** On Qwen this
        is one of the published ``0.25``/``0.5``/``0.75`` profiles. On NeXT it
        is the requested decoder-layer-average fraction; the released Stage-2
        policy makes values through ``0.55`` reachable, while ``1`` is identity.
    :param distance: squared spatial cutoff for the aggregation penalty. Patches
        further than ``sqrt(distance)`` patch-widths apart never merge. The Qwen
        path replaces it with ``distance_ratio``, since its grid varies per image.
    :param distance_ratio: **Qwen2.5-VL only.** The same cutoff as a fraction of
        the padded grid's diagonal (0.8 upstream).
    :param pillar_threshold_percentile: quantile of benchmark-token CLS attention
        splitting Pillars (kept verbatim) from Collectors (merge their
        neighbourhood). ``None`` takes each release's own value: 0.55 on
        LLaVA-1.5, 0.5 on the tiled path (``llava-next.sh`` exports ``THRE=0.5``)
        and 0.1 on Qwen2.5-VL.
    :param top_n_per_region: candidates per 2x2 region. ``None`` takes the
        published value for the selected budget.
    :param stage2_layer: decoder layer that prunes; its predecessor supplies the
        ranking attention. ``None`` takes the backbone's published layer -- 16 on
        the 32-layer CLIP decoders, 15 on Qwen's 28-layer one.
    :param vision_feature_layer: vision-tower hidden state used as the merged
        features. ``None`` on InternVL takes the model config's own value; the
        LLaVA default is -2, matching LLaVA's ``vision_feature_layer``.
    :param sim_layers: ``[start, end)`` vision-tower hidden states averaged into
        the similarity features. The Qwen release averages ViT *blocks* 7..23
        instead; see ``QWEN_SIM_BLOCKS``.
    :param grid_h: patch-grid height, 24 for LLaVA-1.5 at 336px. LLaVA-only: the
        tiled and NaViT backbones derive their grids per image.
    :param grid_w: patch-grid width, 24 for LLaVA-1.5 at 336px.
    :param attention_chunk: **Qwen2.5-VL only.** Query rows scored per matmul when
        recomputing the last vision block's attention. Lower it if the vision
        tower runs out of memory on very large images.
    :param track_source: record ``[T_current, T_original]`` matrices for
        ``llmoe.viz``. Needed by ``bm --visualize``; off by default so benchmark
        runs do not pay for it.
    """

    def __init__(self, max_tokens: int = 64, *, retain_ratio: float | None = None,
                 distance: float = DEFAULT_DISTANCE, distance_ratio: float = QWEN_DISTANCE_RATIO,
                 pillar_threshold_percentile: float | None = None,
                 top_n_per_region: int | None = None, stage2_layer: int | None = None,
                 vision_feature_layer: int | None = None,
                 sim_layers: Sequence[int] = DEFAULT_SIM_LAYERS,
                 grid_h: int = 24, grid_w: int = 24,
                 attention_chunk: int = DEFAULT_ATTENTION_CHUNK,
                 track_source: bool = False) -> None:
        super().__init__(int(max_tokens))
        self._grid_h: int = int(grid_h)
        self._grid_w: int = int(grid_w)
        if self._grid_h % 2 or self._grid_w % 2:
            raise ValueError(f"Nuwa partitions the grid into 2x2 regions; got {grid_h}x{grid_w}")
        self._num_patches: int = self._grid_h * self._grid_w
        # The upper bound is the backbone's patch count, which only the LLaVA path
        # knows here; install() checks it once the backbone is known.
        if self._max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if retain_ratio is not None and not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
        if pillar_threshold_percentile is not None \
                and not 0.0 <= float(pillar_threshold_percentile) <= 1.0:
            raise ValueError(f"pillar_threshold_percentile must be in [0, 1], got {pillar_threshold_percentile}")
        if float(distance) <= 0.0:
            raise ValueError(f"distance must be > 0, got {distance}")
        if not 0.0 < float(distance_ratio) <= 1.0:
            raise ValueError(f"distance_ratio must be in (0, 1], got {distance_ratio}")
        if stage2_layer is not None and int(stage2_layer) < 1:
            raise ValueError("stage2_layer must be >= 1; the ranking attention comes from the layer before it")

        self._retain_ratio: float | None = None if retain_ratio is None else float(retain_ratio)
        self._distance: float = float(distance)
        self._distance_ratio: float = float(distance_ratio)
        self._distance_threshold: float = math.sqrt(self._distance)
        # `None` means "each release's own constant", resolved per backbone below.
        self._configured_percentile: float | None = (
            None if pillar_threshold_percentile is None else float(pillar_threshold_percentile))
        self._pillar_threshold_percentile: float = (
            self._configured_percentile if self._configured_percentile is not None else 0.55)
        self._configured_top_n: int | None = (
            None if top_n_per_region is None else int(top_n_per_region))
        # Resolved here rather than at install() so `name` is stable before the
        # backbone is known; `retain_ratio` is what selects the Qwen path.
        self._configured_stage2_layer: int | None = (
            None if stage2_layer is None else int(stage2_layer)
        )
        self._stage2_layer: int = (
            self._configured_stage2_layer
            if self._configured_stage2_layer is not None
            else (
                QWEN_STAGE2_LAYER
                if self._retain_ratio is not None
                else DEFAULT_STAGE2_LAYER
            )
        )
        self._configured_feature_layer: int | None = (
            None if vision_feature_layer is None else int(vision_feature_layer))
        self._vision_feature_layer: int = (
            self._configured_feature_layer if self._configured_feature_layer is not None else -2)
        if len(tuple(sim_layers)) != 2:
            raise ValueError(f"sim_layers must be a [start, end) pair, got {tuple(sim_layers)}")
        self._sim_layers: tuple[int, int] = (int(sim_layers[0]), int(sim_layers[1]))
        self._attention_chunk: int = int(attention_chunk)
        self._track_source: bool = bool(track_source)
        self._qwen: QwenNuwaAdapter | None = None
        self._internvl: InternVLNuwaAdapter | None = None
        self._llava_next: Any | None = None

        num_regions = (self._grid_h // 2) * (self._grid_w // 2)
        self._stage1_count: int = min(
            self._num_patches,
            STAGE1_CONFIGS.get(self._max_tokens, max(1, round(self._max_tokens * 7 / 4))),
        )
        self._top_n_per_region: int = self._configured_top_n if self._configured_top_n is not None else min(
            4, max(1, TOPN_CONFIGS.get(self._max_tokens, math.ceil(self._stage1_count / num_regions)))
        )
        # The separation step cannot yield more than this many candidates, so the
        # Stage-1 budget is capped by it and the placeholder trim must agree.
        self._stage1_count = min(self._stage1_count, num_regions * self._top_n_per_region)
        self._stage2_keep: int = get_stage2_keep_count(self._max_tokens, self._stage1_count)

        # Per-forward state, rebuilt by prepare_inputs and get_image_features.
        self._vision_positions: torch.Tensor | None = None  # [B, N_stage1] absolute columns
        self._benchmark_indices: torch.Tensor | None = None  # [B, N_stage1] original patch indices
        self._stage2_attention: torch.Tensor | None = None  # [B, H, S, S] at the capture layer
        # Geometry of the post-cut sequence, handed to every layer below the cut.
        self._prune_state: dict[str, Any] = {}
        self._suspend_rpme: bool = False
        self._stage1_source: torch.Tensor | None = None  # [N_stage1, N] merge weights, viz only
        self._warned: set[str] = set()

        # Installation bookkeeping, all undone by uninstall().
        self._installed_vlm: VLM | None = None
        self._inner_model: Any | None = None
        self._language_model: Any | None = None
        self._hook_handles: list[RemovableHandle] = []
        self._image_token_id: int | None = None
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._attention_name: str | None = None
        self._previous_llm_attn: str | None = None
        self._previous_vision_attn: str | None = None
        self._vision_config: Any | None = None

    @property
    def budget(self) -> int | float:
        return self._max_tokens if self._retain_ratio is None else self._retain_ratio

    def validate_llava_next(self) -> None:
        """Enforce Nuwa's single canonical NeXT control: retain_ratio."""

        from llmoe.compressor.llava_next import validate_retain_ratio

        ratio = validate_retain_ratio(self._retain_ratio, label="Nuwa")
        if 0.55 < ratio < 1.0:
            raise ValueError(
                "Nuwa's released LLaVA-NeXT schedule keeps 10% after layer 16, "
                "so layer-average retain_ratio must be <= 0.55 (or exactly 1 for "
                "an identity pass-through)."
            )
        if self._max_tokens != 64:
            raise ValueError(
                "Nuwa on LLaVA-NeXT only accepts retain_ratio; remove max_tokens "
                "(its constructor default is 64 and is ignored on NeXT)."
            )
        noncanonical = (
            self._distance != DEFAULT_DISTANCE
            or self._distance_ratio != QWEN_DISTANCE_RATIO
            or self._configured_percentile is not None
            or self._configured_top_n is not None
            or self._configured_stage2_layer is not None
            or self._configured_feature_layer is not None
            or self._sim_layers != DEFAULT_SIM_LAYERS
            or self._grid_h != 24
            or self._grid_w != 24
            or self._attention_chunk != DEFAULT_ATTENTION_CHUNK
        )
        if noncanonical:
            raise ValueError(
                "Nuwa on LLaVA-NeXT fixes the released distance, pillar split, "
                "regional candidates, vision layers, and layer-16 decoder cut; "
                "only retain_ratio is configurable."
            )

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[Nuwa] WARNING: {message}", flush=True)

    # ----------------------------------------------------------- TokenCompressor

    @override
    @torch.no_grad()
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, **kwargs: Any) -> CompressedFeatures:
        """Run Stage 1 alone, straight down to ``max_tokens``.

        This is the tensor-level entry point -- the real path goes through
        ``install()``, where Stage 1 stops at the larger Stage-1 budget and the
        decoder finishes the job. Here there is no decoder, so the whole budget is
        spent in one pass.

        ``attention`` carries the CLS->patch attention Nuwa ranks with, as ``[N]``,
        ``[B, N]``, or raw vision-tower weights ``[B, H, S, S]`` (row 0, summed over
        heads, CLS column dropped). Without it there is no CLS row to read, so the
        salience falls back to each token's L2 norm -- the ``||k_i||_2`` factor of
        the paper's Eq. 1, which the official code drops. Separation, Alignment and
        Aggregation are otherwise unchanged.

        :param features: ``[B, N, D]`` or ``[N, D]`` visual tokens.
        :param attention: CLS attention over the patches, see above.
        :return: the merged tokens, same rank, dtype and device as ``features``.
        """
        del layer_idx, kwargs

        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Expected [B, N, D] or [N, D], got {tuple(features.shape)}")
        batch, n_tokens, _ = features.shape
        if n_tokens == 0:
            raise ValueError("Cannot compress an empty sequence")

        grid_h, grid_w = self._resolve_grid(n_tokens)
        num_regions = (grid_h // 2) * (grid_w // 2)
        keep = min(self._max_tokens, n_tokens)
        top_n = min(4, max(1, math.ceil(keep / num_regions)))
        cls_attention = self._resolve_cls_attention(features, attention, batch, n_tokens)

        merged, _, weights = nuwa_stage1(
            patch_features=features,
            sim_features=features,
            cls_attention=cls_attention,
            grid_h=grid_h,
            grid_w=grid_w,
            num_tokens=keep,
            top_n_per_region=top_n,
            distance_threshold=self._distance_threshold,
            pillar_threshold_percentile=self._pillar_threshold_percentile,
        )
        if self._track_source:
            self._source_history = [
                torch.eye(n_tokens, dtype=torch.float32),
                weights[0].detach().float().cpu(),
            ]
        return CompressedFeatures(tokens=merged.squeeze(0) if unbatched else merged)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        """``[identity, stage1, stage2]`` for the last forward, ``[T_cur, N]`` each.

        ``stage1`` holds the real merge weights, so a row shows every patch its
        token now stands for; ``stage2`` keeps only the rows that survived the
        decoder cut, still expressed over the original patches. Empty when
        ``track_source`` is off. On the Qwen and InternVL backbones the matrices
        cover the merged visual tokens of the whole prompt, which is the grid those
        backbones actually prune.
        """
        if self._qwen is not None:
            return self._qwen.get_source_history()
        if self._internvl is not None:
            return self._internvl.get_source_history()
        return list(self._source_history)

    # --------------------------------------------------------------- compress helpers

    def _resolve_grid(self, n_tokens: int) -> tuple[int, int]:
        """Patch grid for ``n_tokens``: the configured one, or a square if it fits."""
        if n_tokens == self._num_patches:
            return self._grid_h, self._grid_w
        side = int(round(math.sqrt(n_tokens)))
        if side * side != n_tokens or side % 2:
            raise ValueError(
                f"Nuwa needs an even square patch grid; {n_tokens} tokens do not match the configured "
                f"{self._grid_h}x{self._grid_w} and are not an even square. Pass grid_h/grid_w."
            )
        return side, side

    @staticmethod
    def _resolve_cls_attention(features: torch.Tensor, attention: torch.Tensor | None,
                               batch: int, n_tokens: int) -> torch.Tensor:
        """Normalise ``attention`` to ``[B, N]`` CLS scores, or fall back to L2 norms."""
        if attention is None:
            return features.float().square().sum(dim=-1).sqrt()

        scores = attention.to(features.device).float()
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        elif scores.ndim == 4:
            # Raw vision-tower weights: the CLS row, summed over heads, patches only.
            scores = scores.sum(dim=1)[:, 0, :]
        if scores.ndim != 2:
            raise ValueError(f"Unsupported attention shape {tuple(attention.shape)}")
        if scores.shape[1] == n_tokens + 1:
            scores = scores[:, 1:]  # drop the CLS self-column
        if scores.shape[0] == 1 and batch > 1:
            scores = scores.expand(batch, -1)
        if tuple(scores.shape) != (batch, n_tokens):
            raise ValueError(
                f"attention scores {tuple(scores.shape)} do not match the {batch} x {n_tokens} "
                f"tokens being compressed"
            )
        return scores

    # ------------------------------------------------------------------- install

    @staticmethod
    def _detect_backbone(vlm: VLM) -> str:
        """``"qwen"``, ``"internvl"`` or ``"llava"``, checked structurally.

        InternVL and LLaVA both expose a ``vision_tower``; only InternVL folds its
        encoder patches with ``pixel_shuffle``, which is exactly the difference
        Stage 1 has to know about, so that is what the two are told apart by.
        """
        inner = getattr(getattr(vlm, "model", None), "model", None)
        if getattr(inner, "visual", None) is not None:
            return "qwen"
        if getattr(inner, "vision_tower", None) is not None:
            return "internvl" if callable(getattr(inner, "pixel_shuffle", None)) else "llava"
        raise TypeError(
            f"Nuwa supports LLaVA-1.5, LLaVA-NeXT, Qwen2.5-VL, and InternVL; "
            f"{type(getattr(vlm, 'model', None)).__name__} exposes neither "
            f"`model.model.vision_tower` nor `model.model.visual`. Run with `--backbone llava`, "
            f"`--backbone qwen-vl` or `--backbone internvl`."
        )

    @override
    def install(self, vlm: VLM) -> None:
        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return  # idempotent: repeated benchmarks must not stack wrappers
            raise RuntimeError("NuwaCompressor is already installed on another backbone")

        from llmoe.compressor.llava_next import is_llava_next

        if is_llava_next(vlm):
            self.validate_llava_next()
            from llmoe.compressor.nuwa.nuwa_llava_next import LlavaNextNuwaAdapter

            self._stage2_layer = DEFAULT_STAGE2_LAYER
            self._pillar_threshold_percentile = NEXT_PILLAR_PERCENTILE
            self._installed_vlm = vlm
            self._llava_next = LlavaNextNuwaAdapter(
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
                f"[Nuwa/LLaVA-NeXT] install: retain_ratio={self._retain_ratio:g}, "
                f"stage2=layer{self._stage2_layer}, deep_ratio=0.1, "
                "top_n_per_region=3, pillars>=p0.5",
                flush=True,
            )
            return

        backbone = self._detect_backbone(vlm)
        if backbone != "qwen" and self._retain_ratio is not None:
            raise ValueError(
                "retain_ratio selects Nuwa's Qwen2.5-VL path, whose encoder emits a different token "
                "count per image. On the CLIP-style backbones use `max_tokens`: their released "
                "budgets are absolute counts tuned per budget, so a ratio would not reproduce the "
                "paper's numbers."
            )
        if backbone == "qwen":
            if self._retain_ratio is None:
                supported = ", ".join(f"{ratio:g}" for ratio in sorted(QWEN_PROFILES))
                raise ValueError(
                    f"Nuwa on Qwen2.5-VL is parameterised by `retain_ratio` ({{{supported}}}), not by "
                    f"`max_tokens`: its NaViT encoder emits a different token count per image, so a "
                    f"layer-averaged token budget is not comparable across samples."
                )
            if self._retain_ratio not in QWEN_PROFILES:
                supported = ", ".join(
                    f"{ratio:g}" for ratio in sorted(QWEN_PROFILES)
                )
                raise ValueError(
                    f"Unsupported Nuwa retain_ratio: {self._retain_ratio}. "
                    f"The Qwen2.5-VL release publishes {{{supported}}}."
                )
            self._stage2_layer = (
                self._configured_stage2_layer
                if self._configured_stage2_layer is not None
                else QWEN_STAGE2_LAYER
            )
            self._install_qwen(vlm)
            return
        if backbone == "internvl":
            self._install_internvl(vlm)
            return
        if not 1 <= self._max_tokens <= self._num_patches:
            raise ValueError(
                f"max_tokens must be in [1, {self._num_patches}] on LLaVA-1.5, got {self._max_tokens}")

        inner = vlm.model.model
        language_model = getattr(inner, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if not layers:
            raise TypeError("Nuwa needs a decoder stack at model.model.language_model.layers")
        if self._configured_stage2_layer is None and len(layers) != REFERENCE_DEPTH:
            # The published cut is the middle of the decoder -- layer 16 of
            # Vicuna-7B's 32 -- which is what makes the reported budget
            # `(N_stage1 + N_stage2) / 2` a layer average. Move it with the depth
            # so it stays the midpoint (20 of LLaVA-1.5-13B's 40) and the budget
            # keeps meaning what it says. An explicit `stage2_layer` is honoured.
            self._stage2_layer = scale_layer(DEFAULT_STAGE2_LAYER, len(layers))
        if self._stage2_layer >= len(layers):
            raise ValueError(f"stage2_layer={self._stage2_layer} is too large for a {len(layers)}-layer decoder")
        if getattr(inner, "vision_tower", None) is None or getattr(inner, "multi_modal_projector", None) is None:
            raise TypeError("Nuwa needs a vision_tower and a multi_modal_projector on model.model")

        self._installed_vlm = vlm
        self._inner_model = inner
        self._language_model = language_model
        self._image_token_id = resolve_image_token_id(vlm.model.config, vlm.processor)

        self._force_eager_vision(inner.vision_tower)
        self._install_vision_capture(inner.vision_tower)
        self._install_stage1(inner)
        self._install_prepare_inputs(vlm)
        self._install_attention_capture(language_model)
        self._install_rpme(language_model)
        self._install_stage2(layers)

        print(f"[Nuwa] install: budget={self._max_tokens} (stage1={self._stage1_count}, "
              f"stage2={self._stage2_keep}), top_n_per_region={self._top_n_per_region}, "
              f"prune @layer{self._stage2_layer} of {len(layers)} (score @layer{self._stage2_layer - 1}), "
              f"grid={self._grid_h}x{self._grid_w}", flush=True)

    def _install_qwen(self, vlm: VLM) -> None:
        """Hand the forward to ``nuwa_qwen.py``; see its docstring for what differs."""
        stage1_ratio, stage2_ratio, top_n = QWEN_PROFILES[self._retain_ratio]
        self._qwen = QwenNuwaAdapter(
            retain_ratio=self._retain_ratio,
            stage1_ratio=stage1_ratio,
            stage2_ratio=stage2_ratio,
            top_n_per_region=self._configured_top_n if self._configured_top_n is not None else top_n,
            pillar_threshold_percentile=(
                self._configured_percentile if self._configured_percentile is not None
                else QWEN_PILLAR_PERCENTILE),
            distance_ratio=self._distance_ratio,
            stage2_layer=self._stage2_layer,
            sim_layers=QWEN_SIM_BLOCKS,
            attention_chunk=self._attention_chunk,
            track_source=self._track_source,
        )
        self._installed_vlm = vlm
        self._qwen.install(vlm, image_token_id=resolve_image_token_id(vlm.model.config, vlm.processor))

    def _install_internvl(self, vlm: VLM) -> None:
        """Hand the forward to ``nuwa_internvl.py``; see its docstring for what differs."""
        self._internvl = InternVLNuwaAdapter(
            stage1_count=next_stage1_count(self._max_tokens),
            top_n_per_region=(self._configured_top_n if self._configured_top_n is not None
                              else next_top_n(self._max_tokens)),
            distance=self._distance,
            pillar_threshold_percentile=(
                self._configured_percentile if self._configured_percentile is not None
                else NEXT_PILLAR_PERCENTILE),
            stage2_layer=self._stage2_layer,
            # InternVL's own default is the last hidden state, not LLaVA's -2, so an
            # untouched `vision_feature_layer` defers to the model config.
            vision_feature_layer=self._configured_feature_layer,
            sim_layers=self._sim_layers,
            track_source=self._track_source,
        )
        self._installed_vlm = vlm
        self._internvl.install(
            vlm, image_token_id=resolve_image_token_id(vlm.model.config, vlm.processor))

    def uninstall(self) -> None:
        if self._llava_next is not None:
            self._llava_next.uninstall()
            self._llava_next = None
            self._installed_vlm = None
            return
        if self._qwen is not None:
            self._qwen.uninstall()
            self._qwen = None
            self._installed_vlm = None
            return
        if self._internvl is not None:
            self._internvl.uninstall()
            self._internvl = None
            self._installed_vlm = None
            return
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        if self._configured_stage2_layer is None:
            self._stage2_layer = DEFAULT_STAGE2_LAYER

        if self._inner_model is not None and self._original_get_image_features is not None:
            self._inner_model.get_image_features = self._original_get_image_features
        if self._installed_vlm is not None and self._original_prepare_inputs is not None:
            self._installed_vlm.prepare_inputs = self._original_prepare_inputs
        if self._language_model is not None and self._previous_llm_attn is not None:
            set_attn_implementation(self._language_model, self._previous_llm_attn)
        if self._vision_config is not None and self._previous_vision_attn is not None:
            self._vision_config._attn_implementation = self._previous_vision_attn
        unregister_capture_attention(self._attention_name)

        self._original_get_image_features = None
        self._original_prepare_inputs = None
        self._previous_llm_attn = None
        self._previous_vision_attn = None
        self._vision_config = None
        self._attention_name = None
        self._image_token_id = None
        self._installed_vlm = None
        self._inner_model = None
        self._language_model = None
        self._reset_forward_state()

    def _reset_forward_state(self) -> None:
        self._vision_positions = None
        self._benchmark_indices = None
        self._stage2_attention = None
        self._prune_state = {}
        self._suspend_rpme = False
        self._stage1_source = None

    # -------------------------------------------------------------- vision tower

    def _force_eager_vision(self, vision_tower: Any) -> None:
        """Make the ViT run eager attention -- SDPA returns no weights to capture."""
        config = getattr(vision_tower, "config", None)
        # CLIPAttention reads `_attn_implementation` off the vision config it was
        # given, which for a bare CLIPVisionModel is the top-level config.
        config = getattr(config, "vision_config", None) or config
        if config is None:
            raise TypeError("Nuwa needs a config on the vision tower to force eager attention")
        self._vision_config = config
        self._previous_vision_attn = getattr(config, "_attn_implementation", None)
        config._attn_implementation = "eager"

    def _install_vision_capture(self, vision_tower: Any) -> None:
        """Ask the vision tower for attentions; LLaVA only requests hidden states."""

        def pre_hook(_module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
            kwargs["output_attentions"] = True
            kwargs["output_hidden_states"] = True
            return args, kwargs

        self._hook_handles.append(vision_tower.register_forward_pre_hook(pre_hook, with_kwargs=True))

    # ------------------------------------------------------------------- Stage 1

    def _install_stage1(self, inner: Any) -> None:
        """Replace ``get_image_features`` with the Stage-1 merge plus re-projection."""
        original = inner.get_image_features
        self._original_get_image_features = original

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            image_outputs = original(*args, **kwargs)
            hidden_states = getattr(image_outputs, "hidden_states", None)
            attentions = getattr(image_outputs, "attentions", None)
            if not hidden_states or not attentions:
                self._warn_once(
                    "no_vision_outputs",
                    "the vision tower returned no attentions/hidden states; Stage 1 is SKIPPED and "
                    "the full token set reaches the decoder.",
                )
                self._reset_forward_state()
                return image_outputs
            return self._run_stage1(inner, image_outputs, hidden_states, attentions)

        inner.get_image_features = wrapped_get_image_features

    def _run_stage1(self, inner: Any, image_outputs: Any, hidden_states: Sequence[torch.Tensor],
                    attentions: Sequence[torch.Tensor]) -> Any:
        # Drop the CLS token: `hidden_states[vision_feature_layer]` is exactly what
        # LLaVA itself selects, and `attentions[vision_feature_layer]` is that same
        # layer's attention, so the scores describe the features being merged.
        feature_hs = hidden_states[self._vision_feature_layer][:, 1:, :]
        sim_hs = average_hidden_states(hidden_states, *self._sim_layers)[:, 1:, :]
        # Deviation 1: salience is CLS attention summed over heads, no norm factor.
        cls_attention = attentions[self._vision_feature_layer][:, :, 0, 1:].sum(dim=1)

        if feature_hs.shape[1] != self._num_patches:
            raise RuntimeError(
                f"Nuwa is configured for a {self._grid_h}x{self._grid_w} grid ({self._num_patches} patches) "
                f"but the vision tower produced {feature_hs.shape[1]}. Pass grid_h/grid_w."
            )

        merged, benchmark_indices, weights = nuwa_stage1(
            patch_features=feature_hs,
            sim_features=sim_hs,
            cls_attention=cls_attention,
            grid_h=self._grid_h,
            grid_w=self._grid_w,
            num_tokens=self._stage1_count,
            top_n_per_region=self._top_n_per_region,
            distance_threshold=self._distance_threshold,
            pillar_threshold_percentile=self._pillar_threshold_percentile,
        )
        if merged.shape[1] != self._stage1_count:
            raise RuntimeError(
                f"Stage 1 produced {merged.shape[1]} tokens but the <image> placeholders were trimmed "
                f"to {self._stage1_count}; the two must agree."
            )

        self._benchmark_indices = benchmark_indices
        # A new image invalidates the decoder-side decisions taken for the last one.
        self._prune_state = {}
        self._stage2_attention = None
        if self._track_source:
            self._stage1_source = weights[0].detach().float().cpu()
            self._source_history = [
                torch.eye(feature_hs.shape[1], dtype=torch.float32),
                self._stage1_source,
            ]

        projected = inner.multi_modal_projector(merged)
        image_outputs.pooler_output = list(projected.unbind(0))
        return image_outputs

    # ---------------------------------------------------------- image placeholders

    def _install_prepare_inputs(self, vlm: VLM) -> None:
        """Trim ``<image>`` placeholders to the Stage-1 budget and record their columns.

        LLaVA raises unless the placeholder count equals the number of image
        features, so this has to track Stage 1 exactly. The surviving columns are
        also what RPME and Stage 2 address the visual span by.
        """
        original = vlm.prepare_inputs
        self._original_prepare_inputs = original

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            inputs = original(*args, **kwargs)
            input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
            if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
                self._vision_positions = None
                return inputs
            inputs = self._trim_image_placeholders(vlm, inputs, input_ids)
            self._capture_vision_positions(inputs["input_ids"])
            return inputs

        vlm.prepare_inputs = wrapped_prepare_inputs

    def _trim_image_placeholders(self, vlm: VLM, inputs: Any, input_ids: torch.Tensor) -> Any:
        """Keep the first ``stage1_count`` ``<image>`` placeholders in every row."""
        attention_mask = inputs.get("attention_mask")
        id_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        changed = False

        for row in range(input_ids.shape[0]):
            ids = input_ids[row]
            positions = (ids == self._image_token_id).nonzero(as_tuple=True)[0]
            if positions.numel() <= self._stage1_count:
                id_rows.append(ids)
                if attention_mask is not None:
                    mask_rows.append(attention_mask[row])
                continue
            changed = True
            keep = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
            keep[positions[self._stage1_count:]] = False
            id_rows.append(ids[keep])
            if attention_mask is not None:
                mask_rows.append(attention_mask[row][keep])

        if not changed:
            return inputs
        inputs["input_ids"] = self._left_pad_stack(id_rows, self._pad_token_id(vlm))
        if attention_mask is not None:
            inputs["attention_mask"] = self._left_pad_stack(mask_rows, 0)
        return inputs

    def _capture_vision_positions(self, input_ids: torch.Tensor) -> None:
        """Record each row's visual columns, or disable the decoder stages."""
        rows = [(row == self._image_token_id).nonzero(as_tuple=True)[0] for row in input_ids]
        counts = {int(row.numel()) for row in rows}
        self._reset_forward_state()

        if counts == {0}:
            return  # text-only prompt: nothing for Nuwa to prune
        if len(counts) != 1:
            self._warn_once(
                "ragged_batch",
                f"rows carry different visual-token counts {sorted(counts)}; Stage 2 and RPME are "
                f"SKIPPED for such batches. Batch images of equal token count.",
            )
            return
        count = counts.pop()
        if count != self._stage1_count:
            self._warn_once(
                "unexpected_count",
                f"expected {self._stage1_count} visual tokens per row but found {count}; Stage 2 and "
                f"RPME are SKIPPED.",
            )
            return
        self._vision_positions = torch.stack(rows, dim=0)

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
                pad = torch.full((max_len - row.shape[0],), pad_value, dtype=row.dtype, device=row.device)
                row = torch.cat([pad, row], dim=0)
            padded.append(row)
        return torch.stack(padded, dim=0)

    # -------------------------------------------------------- Stage-2 attention

    def _install_attention_capture(self, language_model: Any) -> None:
        """Run eager attention at the capture layer only; see ``policy.py``."""

        def sink(weights: torch.Tensor) -> None:
            self._stage2_attention = weights

        self._attention_name, self._previous_llm_attn = register_capture_attention(
            language_model, self._stage2_layer - 1, sink, owner=self, label="nuwa_capture")

    # ---------------------------------------------------------------------- RPME

    def _install_rpme(self, language_model: Any) -> None:
        """Rewrite ``position_ids`` before RoPE so spatial distances survive.

        Mirrors the official ``modeling_llama.py``::

            prefill: position_ids[visual span] = benchmark_indices + first_visual
                     position_ids[after the span] += num_patches - N_stage1
            decode:  position_ids += num_patches - N_stage1

        Kept patches are re-addressed by their *original* grid index, so two
        survivors that were four patches apart still are, and the text following
        the image keeps the absolute offset it would have had without pruning.
        ``LlamaModel`` builds ``position_embeddings`` once from this call and hands
        them to every layer, so this single hook covers the whole decoder.
        """

        def rpme_hook(_module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
            positions = self._vision_positions
            benchmark = self._benchmark_indices
            if positions is None or benchmark is None or self._suspend_rpme:
                return None

            hidden_states = args[0] if args else kwargs.get("x")
            in_kwargs = "position_ids" in kwargs
            position_ids = kwargs.get("position_ids")
            if position_ids is None and len(args) >= 2:
                position_ids = args[1]
            if not isinstance(hidden_states, torch.Tensor) or not isinstance(position_ids, torch.Tensor):
                return None

            num_visual = positions.shape[1]
            offset = self._num_patches - num_visual
            batch = max(hidden_states.shape[0], position_ids.shape[0])
            if positions.shape[0] != batch:
                return None  # batch changed since prefill (e.g. beam expansion)
            position_ids = position_ids.expand(batch, -1).clone()

            if hidden_states.shape[1] > 1:  # prefill
                columns = positions.to(position_ids.device)
                if int(columns.max()) >= position_ids.shape[1]:
                    return None
                first = columns[:, :1]
                spatial = benchmark.to(device=position_ids.device, dtype=position_ids.dtype)
                if spatial.shape[0] == 1 and batch > 1:
                    spatial = spatial.expand(batch, -1)
                if spatial.shape != columns.shape:
                    return None
                position_ids.scatter_(1, columns, spatial + first)
                # Everything after the image closes the gap the pruning opened.
                grid = torch.arange(position_ids.shape[1], device=position_ids.device)
                position_ids = position_ids + (grid.unsqueeze(0) > columns[:, -1:]) * offset
            else:  # decode: the whole cache sits at the shifted offset
                position_ids = position_ids + offset

            if in_kwargs:
                kwargs["position_ids"] = position_ids
            else:
                args = (args[0], position_ids, *args[2:])
            return args, kwargs

        self._hook_handles.append(
            language_model.rotary_emb.register_forward_pre_hook(rpme_hook, with_kwargs=True)
        )

    # ------------------------------------------------------------------- Stage 2

    def _install_stage2(self, layers: Any) -> None:
        """Drop the visual tokens the prompt does not want, at the prune layer.

        At the prune layer's prefill the captured attention picks the survivors and
        the sequence is physically rewritten to
        ``[text before | kept visual | text after]``. Layers from here down run on
        the shorter sequence and cache only those keys, which is where Nuwa's
        speedup comes from -- the official code does the same thing by vendoring
        its own ``modeling_llama.py``.

        Everything the decoder builds once per forward and broadcasts to every
        layer -- ``position_ids``, ``position_embeddings``, ``attention_mask``,
        ``cache_position`` -- was sized for the full sequence, so the deeper layers
        get pruned copies re-supplied. From here on the KV cache is ragged: layers
        below the prune layer hold the full sequence and layers above it hold the
        short one, so the decode path rebuilds the mask and cache positions at the
        shorter width every step.
        """
        for index in range(self._stage2_layer, len(layers)):
            handle = layers[index].register_forward_pre_hook(
                self._make_stage2_hook(index), with_kwargs=True
            )
            self._hook_handles.append(handle)

    def _make_stage2_hook(self, layer_idx: int) -> Any:
        def hook(_module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
            return self._stage2_hook(args, kwargs, layer_idx)

        return hook

    def _stage2_hook(self, args: tuple, kwargs: dict, layer_idx: int) -> tuple[tuple, dict] | None:
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            return None

        prefill = hidden_states.shape[1] > 1
        if layer_idx == self._stage2_layer:
            if prefill:
                hidden_states = self._prune(hidden_states, kwargs)
            elif self._prune_state:
                advance_decode_step(self._prune_state, hidden_states, label="Nuwa")

        if not self._prune_state:
            return None
        apply_prune_state(kwargs, self._prune_state, prefill=prefill)
        return (hidden_states,), kwargs

    def _prune(self, hidden_states: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """Rewrite the sequence to ``[text | kept visual | text]`` and size the rest."""
        self._prune_state = {}
        attention = self._stage2_attention
        self._stage2_attention = None  # the weights are large; do not hold them

        positions = self._vision_positions
        if positions is None:
            return hidden_states
        batch, seq_len, hidden_dim = hidden_states.shape
        if attention is None:
            self._warn_once(
                "no_capture",
                f"no attention was captured at layer {self._stage2_layer - 1}; Stage 2 is SKIPPED and "
                f"only Stage 1's {self._stage1_count} tokens reach the deeper layers.",
            )
            return hidden_states

        columns = positions.to(hidden_states.device)
        if columns.shape[0] != batch or attention.shape[0] != batch or int(columns.max()) >= seq_len:
            self._warn_once(
                "geometry_changed",
                f"the visual span {tuple(columns.shape)} does not fit the decoder input "
                f"{tuple(hidden_states.shape)}; Stage 2 is SKIPPED for this batch.",
            )
            return hidden_states
        if self._stage2_keep >= columns.shape[1]:
            return hidden_states  # the budget already fits; nothing left to cut

        keep_local = select_by_text_attention(attention, columns, self._stage2_keep)
        kept_columns = columns.gather(1, keep_local)
        keep_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden_states.device)
        keep_mask.scatter_(1, columns, False)  # drop the whole visual span ...
        keep_mask.scatter_(1, kept_columns, True)  # ... then put the winners back
        keep_index = keep_columns(keep_mask)  # ascending per row
        new_len = keep_index.shape[1]

        new_hidden = hidden_states.gather(1, keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim))
        position_ids = gather_position_ids(kwargs.get("position_ids"), keep_index, seq_len)
        position_embeddings = self._pruned_position_embeddings(
            kwargs.get("position_embeddings"), keep_index, seq_len, new_hidden, position_ids,
        )
        mask, valid = pruned_attention_mask(kwargs.get("attention_mask"), keep_index, seq_len)

        self._prune_state = {
            "batch": batch,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "attention_mask": mask,
            "valid": valid,
            # Layers at or below the cut cache only the survivors, so their cache
            # positions are the compact ones; RPME's ids are RoPE's business alone.
            "cache_position": torch.arange(new_len, device=hidden_states.device),
            "kv_len": new_len,
        }
        if "first_cut" not in self._warned:
            self._warned.add("first_cut")
            print(f"[Nuwa] stage 2 @layer{self._stage2_layer}: seq {seq_len} -> {new_len} "
                  f"(visual {columns.shape[1]} -> {self._stage2_keep})", flush=True)

        if self._track_source and self._stage1_source is not None:
            self._source_history = [*self._source_history[:2],
                                    self._stage1_source[keep_local[0].detach().cpu()]]
        return new_hidden

    def _pruned_position_embeddings(self, position_embeddings: Any, keep_index: torch.Tensor,
                                    seq_len: int, new_hidden: torch.Tensor,
                                    position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the survivors' rows out of the RoPE tables built for the full sequence.

        Selecting rather than recomputing matters: the tables already carry RPME's
        rewritten positions, and re-running ``rotary_emb`` would send the pruned
        ids back through the RPME hook. The recompute fallback therefore suspends
        that hook -- which is why this wraps ``select_rope_tables`` instead
        of calling it directly, as the backbones without RPME do.
        """
        selected = select_rope_tables(position_embeddings, keep_index, seq_len)
        if selected is not None:
            return selected

        self._suspend_rpme = True
        try:
            return self._language_model.rotary_emb(new_hidden, position_ids=position_ids)
        finally:
            self._suspend_rpme = False
