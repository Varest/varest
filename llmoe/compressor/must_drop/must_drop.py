"""Faithful reimplementation of MustDrop (Multi-Stage Vision Token Dropping).

Paper: "Multi-Stage Vision Token Dropping: Towards Efficient Multimodal Large
Language Model" (arXiv:2411.10803). Equation/section numbers below refer to it.
Reference code: https://github.com/liuting20/MustDrop (the ``llava`` package:
``multimodal_encoder/local_merge.py`` + ``clip_encoder.py`` for the vision stage,
``language_model/modelling_sparse_llama.py`` for the prefilling/decoding stages).

MustDrop measures the importance of every vision token across its whole
lifecycle, dropping redundant tokens in three stages:

* **Vision encoding (§3.3).** Two modules, both training-free:
    - **Local Spatial Merging (LSM).** The ``N`` CLIP patches are laid out on
      their 2-D grid and partitioned into ``k x k`` windows. For each window the
      mean pairwise cosine similarity of its tokens is computed (Eq. 5); the most
      redundant windows are merged into a single weighted-average token. This is
      the spatial-redundancy removal that retains ~76% of tokens.
    - **Key Retaining (KR).** The penultimate CLIP layer's ``[CLS]`` -> patch
      attention (Eq. 6) ranks the merged tokens; the top ``key_rate`` form the
      key set ``O`` (Eq. 7). Key tokens are *never* dropped in any later stage.

* **Prefilling (§3.4) — Dual-Attention Filtering.** Inside the LLM, at a set of
  decoder ``pruning_layers``, the text -> vision attention is read off and two
  scores are formed per vision token: the **global** attention (sum over all text
  tokens, Eq. 8) and the **individual** attention (max over individual text
  tokens, Eq. 10). Tokens low on both are redundant and dropped; the key set is
  always re-unioned back in. Pruning is progressive across the layers.

* **Decoding (§3.5) — Output-aware cache.** After the shallow pruning layers, the
  tokens already dropped stay dropped for the rest of the forward and through
  decoding, so the KV cache only ever holds the surviving (``S_few``) tokens.

Faithfulness notes / harness adaptations
----------------------------------------
This runs inside the LLMOE harness, which drives a stock transformers
``LlavaForConditionalGeneration`` (it does NOT load the paper's patched ``llava``
package). So, exactly like the sibling ``vision_trim.py``, every stage is
expressed through forward hooks rather than by editing model source:

* The reference patches the CLIP encoder so the LSM merge happens *inside* the
  ViT at layer 0 and the KR ``[CLS]`` attention is read at layer 23, meaning
  layers 1-23 actually propagate the merged tokens. Here the vision tower runs
  unmodified; LSM + KR are applied in ``compress`` to the encoder *output*, and
  the ``[CLS]`` attention is recomputed from the penultimate layer's q/k via a
  pre-hook (the SDPA path returns no attention weights). KR's per-token ``[CLS]``
  score is aggregated over each merged group. This is the same documented
  compromise the VisionTrim port makes.

* transformers 5.x scatters image features into ``<image>`` placeholders with a
  strict count check, so the per-image token count out of the vision stage must
  be **deterministic** from the placeholder run length alone. The paper's LSM is
  similarity-thresholded (a data-dependent count); here LSM instead merges the
  ``m`` most-redundant windows where ``m`` is fixed by ``vit_retention`` (default
  440/576, the paper's reported ~76%, which the τ=0.8 similarity threshold yields
  on LLaVA). Window *selection* stays similarity-ranked (Eq. 5) — only the cut
  count is made deterministic so the placeholder count stays predictable.

* The prefilling stage masks pruned vision columns (attention ``-inf``) from the
  prune layer onward instead of physically shrinking the sequence. This keeps
  RoPE / KV / sequence length consistent (and is output-identical to dropping),
  and the persistence of the mask through later layers + decoding is exactly the
  paper's output-aware cache policy (§3.5). It requires ``batch_size == 1``.

* The prefilling cut is **threshold-driven, exactly as the reference**: a token is
  dropped iff its summed text attention < ``global_thr`` of the total AND its max
  text attention < ``individual_thr`` (``γ`` and ``α`` in the paper). The kept count
  is therefore data-dependent and NOT a fixed ``max_tokens``; ``max_tokens`` selects
  the threshold profile instead. Set ``prefill_stage=False`` to run the
  vision-encoding stage only (LSM + KR; an ablation).

Budget semantics
----------------
``max_tokens`` is **not** the final visual-token count. As in the paper -- and as in
``sparse_vlm.py``, ``duet.py`` and ``hidrop.py`` here -- the reported budget is the
token count *averaged over the decoder's layers* (the paper's
``num_token_pool / num_layers`` metric), so pruning to a fixed final count of
``max_tokens`` would keep ~2.5x too many tokens on average and inflate accuracy.
``MUSTDROP_PROFILES`` maps each supported budget onto the thresholds that land on
it::

    budget  global_thr  individual_thr  prune layers    key_rate  floor   source
    224     0.0009      0.001           (2, 6, 10, 14)  0.08       61.25  extrapolated
    192     0.001       0.001           (2, 6, 10, 14)  0.08       61.25  paper
    128     0.0012      0.001           (2, 6, 10, 14)  0.08       61.25  paper
    64      0.011       0.01            (2, 6, 10, 14)  0.08       61.25  paper
    32      0.033       0.032           (1, 4, 8, 12)   0.03       27.31  extrapolated

``floor`` is the smallest layer-average the schedule can reach on LLaVA-1.5 (576
patches -> 440 after LSM, 32 decoder layers): the layers before the first prune hold
every token and the key set ``O`` (Eq. 7) survives all of them, so
``(first * n_vis + (32 - first) * |O|) / 32`` is a hard lower bound
(``minimum_layer_average``). The paper's own schedule floors at 61.25, which is why
32 alone prunes one layer earlier and protects a smaller key set -- under
``(2, 6, 10, 14)`` with ``key_rate=0.08`` a 32-token budget is unreachable no matter
how aggressive the thresholds are. Only the listed budgets are supported; pass both
``global_thr`` and ``individual_thr`` explicitly to explore another. The per-forward
and running ``layer-avg`` are printed so the effective budget can be checked against
the label, and a budget below the reachable floor warns once rather than silently
reporting an inflated one.

Qwen2.5-VL adaptation
---------------------
Qwen has no CLS token and its visual count varies by image. Its path therefore
requires ``retain_ratio``, captures received attention and rotary keys from
the last vision block using the VisionZip mechanism, and runs LSM over the
merged 2-D grid after visual/text concatenation. Received attention replaces
CLS attention for Key Retaining. Dynamic edge windows are supported and every
stage resolves an exact ratio count. Decoder cuts are physical M-RoPE-safe
gathers with persistent shortened caches. This is an in-repo backbone
adaptation; the released MustDrop implementation and thresholds target LLaVA.
"""

from typing import Sequence, override

import torch
from torch.nn import functional as F
from torch.utils.hooks import RemovableHandle

from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.utils.depth import REFERENCE_DEPTH, LayerSchedule, scale_layers

try:  # RoPE helper for the prefilling-stage real-attention recompute.
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as _APPLY_ROPE
except Exception:  # pragma: no cover - degrade to no-RoPE if the path moved
    _APPLY_ROPE = None


# Reference defaults (scripts/v1_5/eval/*.sh): the decoder layers dual-attention
# filtering runs at, the fraction of merged tokens Key Retaining protects, and the
# LSM retention the paper's tau=0.8 similarity threshold yields on LLaVA-1.5.
DEFAULT_PRUNING_LAYERS: tuple[int, ...] = (2, 6, 10, 14)
DEFAULT_KEY_RATE: float = 0.08
DEFAULT_VIT_RETENTION: float = 0.7639
DEFAULT_NUM_LAYERS: int = REFERENCE_DEPTH

# Budgets the paper itself publishes thresholds for; the rest are extrapolated.
MUSTDROP_PAPER_BUDGETS: tuple[int, ...] = (192, 128, 64)

# Per-budget dual-attention profile. ``global_thr`` is a fraction of the total
# text->vision attention mass and ``individual_thr`` an absolute max-attention floor
# (gamma and alpha, Eq. 8-10); a token is dropped only when it falls under both.
# The key is the LAYER-AVERAGED token count the paper reports, not the final
# retained count, so ``pruning_layers``/``key_rate`` belong to the profile too: they
# set the floor the thresholds have to average down to (see the module docstring).
MUSTDROP_PROFILES: dict[int, dict[str, object]] = {
    224: {
        "global_thr": 0.0009,
        "individual_thr": 0.001,
        "pruning_layers": DEFAULT_PRUNING_LAYERS,
        "key_rate": DEFAULT_KEY_RATE,
        "source": "extrapolated: log-linear in the paper's 192/128 thresholds, pruning less than 192",
    },
    192: {
        "global_thr": 0.001,
        "individual_thr": 0.001,
        "pruning_layers": DEFAULT_PRUNING_LAYERS,
        "key_rate": DEFAULT_KEY_RATE,
        "source": "paper",
    },
    128: {
        "global_thr": 0.0012,
        "individual_thr": 0.001,
        "pruning_layers": DEFAULT_PRUNING_LAYERS,
        "key_rate": DEFAULT_KEY_RATE,
        "source": "paper",
    },
    64: {
        "global_thr": 0.011,
        "individual_thr": 0.01,
        "pruning_layers": DEFAULT_PRUNING_LAYERS,
        "key_rate": DEFAULT_KEY_RATE,
        "source": "paper",
    },
    32: {
        # The paper's (2, 6, 10, 14) schedule holds all 440 merged tokens through
        # layers 0-1 and can never drop the 36-token key set, which floors the
        # layer-average at 61.25 -- so a 32 budget is unreachable on it however
        # aggressive the thresholds get. Pruning from layer 1 and protecting 3% of
        # the merged tokens (14) drops the floor to 27.31.
        "global_thr": 0.033,
        "individual_thr": 0.032,
        "pruning_layers": (1, 4, 8, 12),
        "key_rate": 0.03,
        "source": "extrapolated: log-linear in the paper's 128/64 thresholds, on an earlier schedule "
                  "because the paper's own schedule floors at 61.25 layer-average tokens",
    },
}

SUPPORTED_BUDGETS: tuple[int, ...] = tuple(sorted(MUSTDROP_PROFILES, reverse=True))


def key_set_size(n_tokens: int, key_rate: float) -> int:
    """Size of the key set ``O`` (Eq. 7): ``ceil(key_rate * M)`` of the ``M`` merged
    tokens, at least one and never more than ``M``."""
    n_tokens = int(n_tokens)
    if n_tokens <= 0 or float(key_rate) <= 0:
        return 0
    return min(n_tokens, max(1, int(float(key_rate) * n_tokens + 0.999)))


def layer_average_tokens(n_vis: int, pruning_layers: Sequence[int], kept_counts: Sequence[int],
                         num_layers: int = DEFAULT_NUM_LAYERS) -> float:
    """The reported budget: vision tokens averaged over every decoder layer (the
    paper's ``num_token_pool / num_layers``).

    :param n_vis: vision tokens entering the decoder, i.e. after the vision stage.
    :param pruning_layers: layers that actually pruned, ascending. Every layer before
        the first of them holds all ``n_vis`` tokens.
    :param kept_counts: tokens surviving each of those layers; each count holds until
        the next pruning layer, and the last one to the end of the stack.
    :param num_layers: decoder depth (32 for LLaVA-1.5-7B, 40 for the 13B).
    """
    layers = tuple(int(layer) for layer in pruning_layers)
    counts = tuple(int(count) for count in kept_counts)
    if len(layers) != len(counts):
        raise ValueError("pruning_layers and kept_counts must have the same length")
    return LayerSchedule(layers, counts, int(num_layers), int(n_vis)).layer_average


def minimum_layer_average(n_vis: int, first_prune_layer: int, key_count: int,
                          num_layers: int = DEFAULT_NUM_LAYERS) -> float:
    """Smallest layer-average a schedule can reach, i.e. the lowest budget label it
    can serve. Every layer before ``first_prune_layer`` holds all ``n_vis`` tokens and
    the key set is never dropped after it, so no threshold, however aggressive, can
    average below this."""
    first = max(0, min(int(first_prune_layer), int(num_layers)))
    return LayerSchedule((first,), (int(key_count),), int(num_layers), int(n_vis)).layer_average


def profile_for_budget(max_tokens: int) -> dict[str, object]:
    """The dual-attention profile for one budget label, or ``{}`` when the budget has
    none (the caller must then supply both thresholds itself)."""
    return dict(MUSTDROP_PROFILES.get(int(max_tokens), {}))


class MustDropCompressor(TokenCompressor):
    """Three-stage vision-token dropper: LSM + Key Retaining in the ViT, then
    dual-attention filtering in the decoder.

    :param max_tokens: LLaVA's reported layer-average budget. One of
        ``SUPPORTED_BUDGETS`` (224/192/128/64/32), or accompanied by custom
        ``global_thr`` and ``individual_thr`` values.
    :param retain_ratio: Qwen2.5-VL-only final fraction of each image's dynamic
        visual-token count. Qwen does not use the LLaVA budget profiles.
    :param window_size: LSM window side ``k``; ``k x k`` patches merge into one.
    :param vit_retention: fraction of the CLIP patches the vision stage keeps.
    :param key_rate: fraction of the merged tokens the key set ``O`` protects from
        every later stage. Defaults to the budget's profile.
    :param prefill_stage: run the decoder stages. ``False`` is the vision-only
        ablation (LSM + KR), where ``max_tokens`` is instead approached by merging.
    :param pruning_layers: decoder layers dual-attention filtering runs at.
        Defaults to the budget's profile.
    :param global_thr: drop threshold on a token's share of the total text->vision
        attention (gamma, Eq. 8-9). Defaults to the budget's profile.
    :param individual_thr: drop threshold on a token's max attention from any single
        text token (alpha, Eq. 10). Defaults to the budget's profile.
    :param grid_size: patch-grid height override; inferred from ``N`` otherwise.
    :param eps: cosine-similarity denominator guard.
    """

    def __init__(self, max_tokens: int = 64, *, retain_ratio: float | None = None,
                 window_size: int = 3,
                 vit_retention: float = DEFAULT_VIT_RETENTION,
                 key_rate: float | None = None, prefill_stage: bool = True,
                 pruning_layers: Sequence[int] | None = None,
                 global_thr: float | None = None, individual_thr: float | None = None,
                 grid_size: int | None = None, eps: float = 1e-6) -> None:
        self._configured_max_tokens = int(max_tokens)
        self._configured_window_size = int(window_size)
        self._configured_vit_retention = float(vit_retention)
        self._configured_key_rate = key_rate is not None
        self._configured_prefill_stage = bool(prefill_stage)
        self._configured_pruning_layers = pruning_layers is not None
        self._configured_global_thr = global_thr is not None
        self._configured_individual_thr = individual_thr is not None
        self._configured_grid_size = grid_size is not None
        self._configured_eps = float(eps)
        super().__init__(int(max_tokens))
        if self._max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        # The budget label picks a whole profile (thresholds + the schedule that can
        # actually average down to it), not just a threshold pair. An unlisted budget
        # is refused rather than silently reusing another budget's thresholds, which
        # would benchmark a different token count than the one being reported. The
        # vision-only ablation never reads the thresholds, so it takes any budget:
        # there ``max_tokens`` is a merge target, not a layer-average label.
        profile = profile_for_budget(self._max_tokens)
        if (
            retain_ratio is None
            and not profile
            and prefill_stage
            and (global_thr is None or individual_thr is None)
        ):
            raise ValueError(
                f"No MustDrop profile for max_tokens={self._max_tokens}; supported budgets are "
                f"{list(SUPPORTED_BUDGETS)}. Pass both global_thr and individual_thr to explore "
                f"another budget, or prefill_stage=False for the vision-only ablation."
            )
        if key_rate is None:
            key_rate = profile.get("key_rate", DEFAULT_KEY_RATE)
        if pruning_layers is None:
            pruning_layers = profile.get("pruning_layers", DEFAULT_PRUNING_LAYERS)
        if int(window_size) < 2:
            raise ValueError("window_size must be >= 2")
        if not 0.0 < float(vit_retention) <= 1.0:
            raise ValueError("vit_retention must be in (0, 1]")
        if not 0.0 < float(key_rate) <= 1.0:
            raise ValueError("key_rate must be in (0, 1]")
        if retain_ratio is not None and not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError(
                f"retain_ratio must be in (0, 1], got {retain_ratio}"
            )
        pruning = tuple(int(layer) for layer in pruning_layers)
        if any(layer < 0 for layer in pruning):
            raise ValueError("pruning_layers must all be >= 0")
        self._requested_window_size: int = int(window_size)
        self._window_size: int = self._requested_window_size
        self._retain_ratio: float | None = (
            None if retain_ratio is None else float(retain_ratio)
        )
        self._vit_retention: float = float(vit_retention)
        self._key_rate: float = float(key_rate)
        self._prefill_stage: bool = bool(prefill_stage)
        self._pruning_layers: tuple[int, ...] = tuple(sorted(set(pruning)))
        # As configured, i.e. against the 32-layer reference decoder; `install`
        # re-solves these for the backbone and `uninstall` puts them back.
        self._reference_pruning_layers: tuple[int, ...] = self._pruning_layers
        self._grid_size: int | None = None if grid_size is None else int(grid_size)
        self._eps: float = float(eps)
        # Dual-attention thresholds (§3.4), from this budget's profile unless given
        # explicitly. They control how aggressively the prefilling stage prunes; the
        # resulting LAYER-AVERAGE token count is what the paper labels "max_tokens"
        # (so we do NOT prune to a fixed final count — that would keep far too many
        # tokens on average and inflate accuracy).
        fallback = MUSTDROP_PROFILES[64]                          # unreachable unless vision-only
        self._global_thr: float = float(
            global_thr if global_thr is not None else profile.get("global_thr", fallback["global_thr"]))
        self._individual_thr: float = float(
            individual_thr if individual_thr is not None
            else profile.get("individual_thr", fallback["individual_thr"]))
        self._profile_source: str = str(profile.get("source", "custom thresholds"))
        # Penultimate CLIP layer (layer L-1, 23rd of 24) whose [CLS] attention
        # feeds Key Retaining (Eq. 6); matches LLaVA vision_feature_layer=-2.
        self._vit_attention_layer: int = -2
        self._vit_layer = None
        self._captured_vit_hidden: torch.Tensor | None = None
        self._logged: bool = False
        # Per-image key set indices (in compress output order) from the vision
        # stage, consumed by the prefilling stage so key tokens are never pruned.
        self._key_indices: list[torch.Tensor] = []
        # Prefilling/decoding-stage state (populated at install / per forward).
        self._image_token_id: int | None = None
        self._image_mask: torch.Tensor | None = None
        self._llm_pruned_columns: torch.Tensor | None = None
        self._n_layers: int = 0
        self._prune_history: list[tuple[int, int]] = []   # (layer_idx, kept_count) this forward
        self._logged_llm_attn: bool = False
        # Diagnostics: per-forward retained-token logging for the first few forwards,
        # plus a running dataset-wide layer-average (the paper's "equal token number"
        # metric) so the effective budget can be checked against the label.
        self._forward_count: int = 0
        self._diag_forwards: int = 3
        self._token_pool: float = 0.0          # sum of per-forward layer-averages
        self._token_pool_forwards: int = 0
        self._warned_no_mask: bool = False
        self._warned_budget_floor: bool = False
        self._confirmed_mask: bool = False
        # Installation bookkeeping, all undone by uninstall(). Every stage here
        # wraps a bound method or registers a hook, so both have to be tracked for
        # the backbone to be usable un-compressed again in the same process.
        self._installed_vlm = None
        self._inner_model = None
        self._internvl = None
        self._hook_handles: list[RemovableHandle] = []
        self._original_get_image_features = None
        self._original_prepare_inputs = None
        self._qwen = None
        self._llava_next = None

    @property
    @override
    def budget(self) -> int | float:
        return self._max_tokens if self._retain_ratio is None else self._retain_ratio

    def validate_llava_next(self) -> None:
        """Enforce MustDrop's single canonical NeXT control: retain_ratio."""

        from llmoe.compressor.llava_next import validate_retain_ratio
        from llmoe.compressor.must_drop.must_drop_llava_next import (
            layer_average_schedule,
        )

        ratio = validate_retain_ratio(self._retain_ratio, label="MustDrop")
        noncanonical = (
            self._configured_max_tokens != 64
            or self._configured_window_size != 3
            or self._configured_vit_retention != DEFAULT_VIT_RETENTION
            or self._configured_key_rate
            or not self._configured_prefill_stage
            or self._configured_pruning_layers
            or self._configured_global_thr
            or self._configured_individual_thr
            or self._configured_grid_size
            or self._configured_eps != 1e-6
        )
        if noncanonical:
            raise ValueError(
                "MustDrop on LLaVA-NeXT fixes the released 3x3 LSM, 440/576 "
                "vision retention, 8% key set, thresholds, and four decoder "
                "boundaries internally; only retain_ratio is configurable."
            )
        stage1 = max(1, int(576 * DEFAULT_VIT_RETENTION))
        protected = key_set_size(stage1, DEFAULT_KEY_RATE)
        layer_average_schedule(576, ratio, DEFAULT_NUM_LAYERS, key_count=protected)

    # ------------------------------------------------------------------ helpers

    def _grid(self, n: int) -> tuple[int, int]:
        if self._grid_size is not None and n % self._grid_size == 0:
            return self._grid_size, n // self._grid_size
        side = int(round(n ** 0.5))
        if side * side == n:
            return side, side
        for height in range(side, 0, -1):
            if n % height == 0:
                return height, n // height
        return 1, n

    @staticmethod
    def _is_square(n: int) -> bool:
        side = int(round(n ** 0.5))
        return side * side == n

    def _split_cls(self, tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Drop a leading ``[CLS]`` token if present. Returns ``(patches, offset)``
        where ``offset`` is 1 when a ``[CLS]`` was removed."""
        n = tokens.shape[0]
        if self._is_square(n):
            return tokens, 0
        if n > 1 and self._is_square(n - 1):
            return tokens[1:], 1
        return tokens, 0

    def _num_merges(self, n_patch: int) -> int:
        """Number of ``k x k`` windows to merge so the merged token count equals
        ``round(vit_retention * N)`` (deterministic from ``N``). Each merged window
        of ``w = k*k`` tokens collapses to 1, i.e. removes ``w - 1`` tokens."""
        k = self._window_size
        per_merge = k * k - 1
        target_removed = n_patch - int(round(self._vit_retention * n_patch))
        n_windows = (n_patch // (k * k)) if self._grid_divisible(n_patch) else 0
        return max(0, min(n_windows, int(round(target_removed / per_merge))))

    def _grid_divisible(self, n_patch: int) -> bool:
        k = self._window_size
        if self._internvl is not None:
            return self._internvl.grid_divisible(n_patch, k)
        height, width = self._grid(n_patch)
        return height * width == n_patch and height % k == 0 and width % k == 0

    def _vit_budget(self, n_patch: int) -> int:
        """Per-image tokens kept by the vision-encoding (LSM) stage. Must be
        reproducible from ``n_patch`` alone so the placeholder trim agrees."""
        if not self._grid_divisible(n_patch):
            return n_patch
        return n_patch - self._merge_count(n_patch) * (self._window_size ** 2 - 1)

    def _merge_count(self, n_patch: int) -> int:
        """Number of windows merged for this image's grid. With the prefilling
        stage on, the vision stage keeps ``vit_retention`` (the LLM stage prunes
        the rest). Vision-only, it merges as far toward ``max_tokens`` as whole
        windows allow (LSM alone cannot reach arbitrarily low budgets)."""
        if self._prefill_stage:
            return self._num_merges(n_patch)
        per_merge = self._window_size ** 2 - 1
        n_windows = n_patch // (self._window_size ** 2)
        want_removed = max(0, n_patch - self._max_tokens)
        return max(0, min(n_windows, want_removed // per_merge))

    # ------------------------------------------------ vision encoding (§3.3)

    def _window_partition(self, n_patch: int) -> torch.Tensor | None:
        """Indices of each ``k x k`` window's tokens: ``[n_windows, k*k]`` (raster
        order), or ``None`` if the grid is not cleanly tileable by the window."""
        if not self._grid_divisible(n_patch):
            return None
        k = self._window_size
        if self._internvl is not None:
            return self._internvl.window_partition(n_patch, k)
        height, width = self._grid(n_patch)
        grid = torch.arange(n_patch).reshape(height, width)
        windows = grid.unfold(0, k, k).unfold(1, k, k)            # [gh, gw, k, k]
        return windows.reshape(-1, k * k)

    def _window_similarity(self, score_patches: torch.Tensor, windows: torch.Tensor) -> torch.Tensor:
        """Mean pairwise cosine similarity inside each window (paper Eq. 5)."""
        feats = score_patches[windows]                            # [n_windows, w, d]
        normed = feats / (feats.norm(dim=-1, keepdim=True) + self._eps)
        sims = normed @ normed.transpose(1, 2)                    # [n_windows, w, w]
        w = windows.shape[1]
        eye = torch.eye(w, device=sims.device, dtype=sims.dtype)
        sims = sims * (1.0 - eye)                                 # exclude self-similarity
        return sims.sum(dim=(1, 2)) / (w * (w - 1))               # [n_windows]

    def _local_spatial_merging(self, patches: torch.Tensor, score_patches: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """LSM (§3.3). Merge the most-redundant windows into single weighted-average
        tokens. Returns ``(merged_features, key_cls_scores, groups)`` where each
        entry of ``groups`` is the original-patch indices subsumed by output token
        ``i`` (length 1 for a kept token, ``k*k`` for a merged window)."""
        n_patch = patches.shape[0]
        windows = self._window_partition(n_patch)
        if windows is None:
            groups = [torch.tensor([i], device=patches.device) for i in range(n_patch)]
            return patches, score_patches.new_zeros(0), groups
        windows = windows.to(patches.device)
        n_merge = max(0, min(self._merge_count(n_patch), windows.shape[0]))
        sims = self._window_similarity(score_patches, windows)
        merge_win = torch.topk(sims, n_merge).indices if n_merge > 0 else sims.new_zeros(0, dtype=torch.long)
        merge_set = set(int(i) for i in merge_win.tolist())

        merged: list[torch.Tensor] = []
        groups: list[torch.Tensor] = []
        for win_id in range(windows.shape[0]):
            members = windows[win_id]
            if win_id in merge_set:                               # collapse to one token
                merged.append(patches[members].mean(dim=0))      # weighted avg (equal sizes)
                groups.append(members)
            else:                                                 # keep each token
                for idx in members.tolist():
                    merged.append(patches[idx])
                    groups.append(members.new_tensor([idx]))
        merged_features = torch.stack(merged, dim=0)
        return merged_features, sims, groups

    def _key_retaining(self, groups: list[torch.Tensor], cls_attention: torch.Tensor | None,
                       n_out: int, device: torch.device) -> torch.Tensor:
        """Key set ``O`` (§3.3, Eq. 6-7): top ``key_rate`` merged tokens by [CLS]
        attention. Each merged token's score is the mean [CLS] attention of its
        members. Returns the key indices in output order."""
        if cls_attention is None or self._key_rate <= 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        attn = cls_attention.to(device).reshape(-1)
        scores = torch.tensor([attn[g].mean() for g in groups], device=device)
        left = key_set_size(n_out, self._key_rate)                # ceil(key_rate * M)
        if left <= 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        return torch.topk(scores, left).indices

    def _process_image(self, tokens: torch.Tensor, score_tokens: torch.Tensor,
                       cls_attention: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the vision-encoding stage on one image. Returns
        ``(merged_features, source_rows, key_indices)``."""
        n_tokens = tokens.shape[0]
        patches, cls_offset = self._split_cls(tokens)
        score_patches, _ = self._split_cls(score_tokens)
        if score_patches.shape[0] != patches.shape[0]:
            score_patches = patches
        merged, _, groups = self._local_spatial_merging(patches, score_patches)
        n_out = merged.shape[0]
        key_idx = self._key_retaining(groups, cls_attention, n_out, patches.device)

        source = patches.new_zeros((n_out, n_tokens))
        for row, members in enumerate(groups):
            cols = members + cls_offset
            source[row, cols] = 1.0 / float(members.numel())
        if not self._logged:
            self._logged = True
            print(f"[MustDrop] vision encoding: N={patches.shape[0]} -> merged={n_out} "
                  f"(LSM, window={self._window_size}, vit_retention={self._vit_retention}), "
                  f"key_set={int(key_idx.numel())} (key_rate={self._key_rate}, "
                  f"cls_attention={'yes' if cls_attention is not None else 'no'})", flush=True)
        return merged, source, key_idx

    # ----------------------------------------------------------- TokenCompressor

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, **kwargs) -> CompressedFeatures:
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.dim() != 3:
            raise ValueError(f"Expected [B, N, d] or [N, d], got {tuple(features.shape)}")
        batch, n_tokens, _ = features.shape
        self._source_history = []
        self._key_indices = []
        compressed: list[torch.Tensor] = []
        clip_hidden = self._captured_vit_hidden
        for index in range(batch):
            per_image_attention = attention[index] if attention is not None else None
            output_tokens = features[index].float()
            score_tokens = output_tokens
            if isinstance(clip_hidden, torch.Tensor) and clip_hidden.dim() == 3 and index < clip_hidden.shape[0]:
                score_tokens = clip_hidden[index].float()
            merged, source, key_idx = self._process_image(output_tokens, score_tokens, per_image_attention)
            compressed.append(merged.to(features.dtype))
            self._key_indices.append(key_idx)
            if index == 0:
                identity = torch.eye(n_tokens, device=source.device, dtype=source.dtype)
                self._source_history = [identity, source]
        result = torch.stack(compressed, dim=0)
        return CompressedFeatures(tokens=result.squeeze(0) if squeeze else result)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        if self._llava_next is not None:
            return list(self._source_history)
        if self._qwen is not None:
            return self._qwen.get_source_history()
        return self._source_history

    # ------------------------------------------------------------------- install

    @override
    def install(self, vlm) -> None:
        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return  # idempotent: repeated benchmarks must not stack wrappers
            raise RuntimeError("MustDropCompressor is already installed on another backbone")
        inner = getattr(getattr(vlm, "model", None), "model", None)
        from llmoe.compressor.llava_next import is_llava_next

        if is_llava_next(vlm):
            self.validate_llava_next()
            from llmoe.compressor.must_drop.must_drop_llava_next import (
                LlavaNextMustDropAdapter,
            )

            self._installed_vlm = vlm
            self._llava_next = LlavaNextMustDropAdapter(
                self,
                vlm,
                retain_ratio=float(self._retain_ratio),
                track_source=True,
            )
            try:
                self._llava_next.install()
            except Exception:
                self._llava_next = None
                self._installed_vlm = None
                raise
            return
        if getattr(inner, "visual", None) is not None:
            if self._retain_ratio is None:
                raise ValueError(
                    "MustDrop on Qwen2.5-VL requires retain_ratio: NaViT "
                    "emits a different visual-token count for each image."
                )
            from llmoe.compressor.must_drop.must_drop_qwen import (
                QwenMustDropAdapter,
            )
            from llmoe.utils.qwen import resolve_image_token_id

            adapter = QwenMustDropAdapter(
                retain_ratio=self._retain_ratio,
                window_size=self._window_size,
                vit_retention=self._vit_retention,
                key_rate=self._key_rate,
                prefill_stage=self._prefill_stage,
                pruning_layers=self._pruning_layers,
                global_thr=self._global_thr,
                individual_thr=self._individual_thr,
            )
            adapter.install(
                vlm,
                image_token_id=resolve_image_token_id(
                    vlm.model.config,
                    vlm.processor,
                    label="MustDrop",
                ),
            )
            self._installed_vlm = vlm
            self._qwen = adapter
            return
        if self._retain_ratio is not None:
            raise ValueError(
                "retain_ratio is MustDrop's Qwen2.5-VL "
                "parameterisation; LLaVA-1.5 and InternVL use the layer-average "
                "max_tokens presets."
            )

        self._installed_vlm = vlm
        self._wrap_image_features(vlm)
        self._install_placeholder_trim(vlm)
        if self._prefill_stage:
            self._install_prefill_stage(vlm)

    def uninstall(self) -> None:
        """Detach every hook and restore the methods ``install()`` replaced.

        Without this the ViT capture hook, the ``get_image_features`` wrapper and
        the decoder's per-layer prune hooks stay live, so a later full-token or
        differently-compressed run in the same process would still be pruned.
        """
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

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

        if self._inner_model is not None and self._original_get_image_features is not None:
            self._inner_model.get_image_features = self._original_get_image_features
        if self._installed_vlm is not None and self._original_prepare_inputs is not None:
            self._installed_vlm.prepare_inputs = self._original_prepare_inputs

        self._original_get_image_features = None
        self._original_prepare_inputs = None
        self._installed_vlm = None
        self._inner_model = None
        self._internvl = None
        self._llava_next = None
        self._window_size = self._requested_window_size
        self._vit_layer = None
        self._image_token_id = None
        self._n_layers = 0
        self._pruning_layers = self._reference_pruning_layers
        self._reset_forward_state()

    def _reset_forward_state(self) -> None:
        """Drop everything a forward populates; diagnostics counters are lifetime."""
        self._captured_vit_hidden = None
        self._key_indices = []
        self._image_mask = None
        self._llm_pruned_columns = None
        self._prune_history = []

    # The base wrapper only forwards CLS-less patch features to ``compress`` — so
    # Key Retaining never sees the [CLS] attention it relies on (Eq. 6). We
    # recompute the penultimate CLIP layer's [CLS] -> patch attention from that
    # layer's q_proj/k_proj, captured via a pre-hook during the vision-tower
    # forward (backend-independent: SDPA ignores output_attentions). If the layer
    # can't be located, Key Retaining is skipped (KR is an additive protection).

    def _wrap_image_features(self, vlm) -> None:
        inner = vlm.model.model
        self._inner_model = inner

        from llmoe.compressor.must_drop.must_drop_internvl import (
            InternVLMustDropAdapter,
        )

        if InternVLMustDropAdapter.supports(vlm):
            self._internvl = InternVLMustDropAdapter(
                self, vlm, window_size=self._window_size
            )
            self._window_size = self._internvl.window_size
        internvl = self._internvl is not None

        self._vit_layer = self._locate_vit_attention_layer(getattr(inner, "vision_tower", None))
        if self._vit_layer is not None:
            self._hook_handles.append(
                self._vit_layer.register_forward_pre_hook(self._capture_vit_hidden, with_kwargs=True))
        print(f"[MustDrop] install: max_tokens={self._max_tokens}, window={self._window_size}, "
              f"vit_retention={self._vit_retention}, key_rate={self._key_rate}, "
              f"prefill_stage={self._prefill_stage}, pruning_layers={list(self._pruning_layers)}, "
              f"ViT layer located={self._vit_layer is not None}, internvl={internvl}", flush=True)
        original = inner.get_image_features
        self._original_get_image_features = original

        def wrapped(*args, **kwargs):
            self._captured_vit_hidden = None
            image_outputs = original(*args, **kwargs)
            cls_attention = self._compute_cls_attention()
            if self._internvl is not None:
                return self._internvl.compress_outputs(image_outputs, cls_attention)
            stacked = torch.stack(list(image_outputs.pooler_output), dim=0)
            result = self.compress(stacked, attention=cls_attention)
            image_outputs.pooler_output = list(result.tokens.unbind(0))
            return image_outputs

        inner.get_image_features = wrapped

    def _locate_vit_attention_layer(self, vision_tower):
        if vision_tower is None:
            return None
        vision_model = getattr(vision_tower, "vision_model", vision_tower)
        encoder = getattr(vision_model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None:
            layers = getattr(encoder, "layer", None)
        if not layers:
            return None
        try:
            return layers[self._vit_attention_layer]
        except (IndexError, TypeError):
            return None

    def _capture_vit_hidden(self, _module, args, kwargs):
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if isinstance(hidden, torch.Tensor):
            self._captured_vit_hidden = hidden.detach()
        return None

    def _compute_cls_attention(self) -> torch.Tensor | None:
        """Penultimate-layer [CLS] -> patch attention, head-averaged (Eq. 6).
        Returns ``[B, N_patch]`` or ``None`` if unavailable."""
        layer, hidden = self._vit_layer, self._captured_vit_hidden
        if layer is None or not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return None
        try:
            with torch.no_grad():
                attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
                norm = getattr(layer, "layer_norm1", None) or getattr(
                    layer,
                    "layernorm_before",
                    None,
                )
                if attn is None or norm is None:
                    return None
                normed = norm(hidden)
                query, key = attn.q_proj(normed), attn.k_proj(normed)
                q_norm, k_norm = getattr(attn, "q_norm", None), getattr(attn, "k_norm", None)
                if callable(q_norm):
                    query = q_norm(query)
                if callable(k_norm):
                    key = k_norm(key)
                b, seq, dim = query.shape
                if seq < 2:
                    return None
                heads = int(getattr(attn, "num_heads", None) or attn.config.num_attention_heads)
                head_dim = dim // heads
                scale = float(getattr(attn, "scale", getattr(attn, "scaling", head_dim ** -0.5)))
                query = query.view(b, seq, heads, head_dim).transpose(1, 2) * scale
                key = key.view(b, seq, heads, head_dim).transpose(1, 2)
                attention = (query @ key.transpose(-1, -2)).softmax(dim=-1)
                return attention[:, :, 0, 1:].mean(dim=1).float()   # [CLS] row, drop [CLS] col
        except Exception:                                            # noqa: BLE001
            return None

    # The vision stage shrinks each image to ``_vit_budget`` features, but the
    # LLaVA processor still expands <image> into N placeholders. transformers 5.x
    # scatters with a strict count check, so surplus placeholders must be removed.

    def _install_placeholder_trim(self, vlm) -> None:
        config = getattr(getattr(vlm, "model", None), "config", None)
        image_token_id = None
        for attr in ("image_token_id", "image_token_index"):
            if getattr(config, attr, None) is not None:
                image_token_id = int(getattr(config, attr))
                break
        if image_token_id is None:
            return
        tokenizer = getattr(getattr(vlm, "processor", None), "tokenizer", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        pad_id = int(pad_id) if pad_id is not None else image_token_id
        original = vlm.prepare_inputs
        self._original_prepare_inputs = original

        def wrapped(*args, **kwargs):
            return self._trim_placeholders(original(*args, **kwargs), image_token_id, pad_id)

        vlm.prepare_inputs = wrapped

    def _trim_placeholders(self, inputs, image_token_id: int, pad_id: int):
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2:
            return inputs
        attention_mask = inputs.get("attention_mask")
        has_attention = isinstance(attention_mask, torch.Tensor) and attention_mask.shape == input_ids.shape
        device = input_ids.device
        kept_ids: list[torch.Tensor] = []
        kept_attention: list[torch.Tensor] = []
        changed = False
        for row_idx in range(input_ids.shape[0]):
            row = input_ids[row_idx]
            keep = torch.ones(row.shape[0], dtype=torch.bool, device=device)
            positions = (row == image_token_id).nonzero(as_tuple=False).flatten().tolist()
            if positions:
                run_start = previous = positions[0]
                for position in positions[1:] + [None]:
                    if position == previous + 1:
                        previous = position
                        continue
                    run_len = previous - run_start + 1
                    budget = self._vit_budget(run_len)
                    if run_len > budget:
                        keep[run_start + budget: previous + 1] = False
                        changed = True
                    if position is not None:
                        run_start = previous = position
            kept_ids.append(row[keep])
            if has_attention:
                kept_attention.append(attention_mask[row_idx][keep])
        if not changed:
            return inputs
        max_len = max(row.shape[0] for row in kept_ids)
        new_ids = input_ids.new_full((len(kept_ids), max_len), pad_id)
        new_attention = attention_mask.new_zeros((len(kept_ids), max_len)) if has_attention else None
        for row_idx, row in enumerate(kept_ids):
            length = row.shape[0]
            new_ids[row_idx, max_len - length:] = row                          # tokenizer pads left
            if new_attention is not None:
                new_attention[row_idx, max_len - length:] = kept_attention[row_idx]
        inputs["input_ids"] = new_ids
        if new_attention is not None:
            inputs["attention_mask"] = new_attention
        return inputs

    # ------------------------------------------ prefilling + decoding (§3.4-3.5)
    #
    # Dual-attention filtering runs at each of ``pruning_layers`` on the LLM's real
    # text -> vision attention (recomputed from q/k + RoPE; the SDPA path returns
    # no weights). At each layer a token is dropped iff it is globally unimportant
    # (summed text attention < global_thr of the total) AND individually unimportant
    # (max text attention < individual_thr); the vision-stage key set is never
    # dropped. Pruned columns are masked (-inf) from that layer on rather than
    # physically dropped — keeping sequence length / RoPE / KV cache consistent —
    # and the mask persists through the rest of the forward and all decoding steps,
    # which is exactly the output-aware cache policy (§3.5). Requires batch_size==1.

    def _install_prefill_stage(self, vlm) -> None:
        config = getattr(vlm.model, "config", None)
        token_id = getattr(config, "image_token_id", None)
        if token_id is None:
            token_id = getattr(config, "image_token_index", None)
        if token_id is None:
            raise ValueError("MustDrop prefilling stage needs config.image_token_id / image_token_index")
        self._image_token_id = int(token_id)
        layers = vlm.model.model.language_model.layers
        n_layers = len(layers)
        self._n_layers = n_layers
        # The filtering layers are positions in the decoder, not absolute indices:
        # the paper's (2, 6, 10, 14) of 32 is "prune through the first half", which
        # on a 40-layer Vicuna-13B is (3, 8, 13, 18). The dual-attention thresholds
        # need no rescaling -- they are fractions of the text-to-vision attention
        # mass, which does not depend on how many layers follow.
        scaled = scale_layers(self._reference_pruning_layers, n_layers,
                              label="MustDrop's filtering layers")
        prune = [layer for layer in scaled if 0 <= layer < n_layers]
        if not prune:
            raise ValueError(f"no valid pruning_layers within [0, {n_layers}): {self._pruning_layers}")
        self._pruning_layers = tuple(prune)
        first = self._pruning_layers[0]

        def capture_image_mask(_module, args, kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 2 and input_ids.shape[1] > 1:
                self._image_mask = input_ids == self._image_token_id   # reset per prefill
                self._llm_pruned_columns = None
                self._prune_history = []
                self._forward_count += 1
                if self._forward_count <= self._diag_forwards:
                    n_img = int(self._image_mask[0].sum())
                    print(f"[MustDrop] prefill #{self._forward_count}: seq_len={input_ids.shape[1]}, "
                          f"image_tokens={n_img}", flush=True)
            return None

        self._hook_handles.append(
            vlm.model.model.register_forward_pre_hook(capture_image_mask, with_kwargs=True))
        for index in range(first, n_layers):
            self._hook_handles.append(
                layers[index].register_forward_pre_hook(
                    self._make_layer_hook(index), with_kwargs=True))
        print(f"[MustDrop] prefilling stage @layers{list(self._pruning_layers)}, "
              f"global_thr={self._global_thr}, individual_thr={self._individual_thr}, "
              f"key_rate={self._key_rate} (budget label: {self._max_tokens} layer-avg tokens, "
              f"profile: {self._profile_source})", flush=True)

    def _make_layer_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            return self._layer_hook(module, args, kwargs, layer_idx)
        return hook

    def _layer_hook(self, module, args, kwargs, layer_idx: int):
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return None
        batch, seq_len, _ = hidden.shape

        if layer_idx in self._pruning_layers and seq_len > 1:        # prefill prune step
            mask = self._image_mask
            if mask is not None and tuple(mask.shape) == (batch, seq_len):
                if batch != 1:
                    raise RuntimeError("MustDrop prefilling stage requires batch_size == 1.")
                self._prune_at_layer(module, hidden, kwargs, layer_idx)
            elif not self._warned_no_mask:
                self._warned_no_mask = True
                shape = None if mask is None else tuple(mask.shape)
                print(f"[MustDrop] WARNING: prune SKIPPED @layer{layer_idx} — image_mask {shape} "
                      f"!= (batch={batch}, seq_len={seq_len}); no LLM pruning will happen.", flush=True)

        pruned = self._llm_pruned_columns
        if pruned is None or pruned.numel() == 0:
            return None
        if not self._confirmed_mask:
            self._confirmed_mask = True
            print(f"[MustDrop] applying attention mask over {int(pruned.numel())} pruned image columns "
                  f"@layer{layer_idx} (and all deeper layers + decoding)", flush=True)
        kwargs = self._mask_pruned_columns(dict(kwargs), hidden, pruned)
        return tuple(args), kwargs

    def _prune_at_layer(self, module, hidden: torch.Tensor, kwargs: dict, layer_idx: int) -> None:
        """One dual-attention filtering step (§3.4), threshold-driven exactly as the
        reference (``dual_attention_filter``). A vision token is dropped iff it is
        BOTH globally unimportant (its summed text attention < ``global_thr`` of the
        total, Eq. 8-9) AND individually unimportant (its max text attention <
        ``individual_thr``, Eq. 10); the key set ``O`` is never dropped (Eq. 7).
        Pruning is cumulative across layers (output-aware cache, §3.5). The count
        kept is data-dependent — the paper tunes the thresholds so the LAYER-AVERAGE
        token count hits the reported budget."""
        image_positions = self._image_mask[0].nonzero(as_tuple=False).squeeze(-1)
        n_vis = int(image_positions.numel())
        already = self._llm_pruned_columns if self._llm_pruned_columns is not None \
            else image_positions.new_zeros(0)
        alive_mask = torch.ones(n_vis, dtype=torch.bool, device=image_positions.device)
        if already.numel():
            alive_mask[self._columns_to_local(already, image_positions)] = False
        alive_local = alive_mask.nonzero(as_tuple=False).squeeze(-1)
        alive_positions = image_positions[alive_local]
        if alive_positions.numel() == 0:
            return

        weights = self._compute_llm_attention(module, hidden, kwargs)
        global_score, individual_score = self._dual_attention(weights, hidden, alive_positions)
        # Drop = globally unimportant AND individually unimportant (reference).
        total = global_score.sum()
        drop = (global_score < total * self._global_thr) & (individual_score < self._individual_thr)
        # Key set O is never pruned (Eq. 7).
        key_idx = self._key_indices[0] if self._key_indices else image_positions.new_zeros(0, dtype=torch.long)
        key_idx = key_idx[key_idx < n_vis].to(image_positions.device)
        if key_idx.numel():
            is_key = torch.zeros(n_vis, dtype=torch.bool, device=image_positions.device)
            is_key[key_idx] = True
            drop = drop & ~is_key[alive_local]

        self._check_budget_reachable(n_vis, int(key_idx.numel()))

        newly_pruned = alive_positions[drop]
        self._llm_pruned_columns = torch.cat([already, newly_pruned])
        kept = n_vis - int(self._llm_pruned_columns.numel())
        self._prune_history.append((layer_idx, kept))
        if layer_idx == self._pruning_layers[-1]:                  # forward complete: record budget
            avg = self._layer_average(n_vis)
            self._token_pool += avg
            self._token_pool_forwards += 1
            running = self._token_pool / self._token_pool_forwards
            if self._forward_count <= self._diag_forwards or self._token_pool_forwards % 200 == 0:
                print(f"[MustDrop] prefill #{self._forward_count}: keep(final)={kept}, "
                      f"layer-avg={avg:.1f}, running layer-avg={running:.1f} "
                      f"(budget label={self._max_tokens}, key_set={int(key_idx.numel())})", flush=True)
        elif self._forward_count <= self._diag_forwards:
            print(f"[MustDrop] prefill #{self._forward_count} prune @layer{layer_idx}: "
                  f"alive={int(alive_positions.numel())} -> keep={kept} "
                  f"(dropped={int(newly_pruned.numel())})", flush=True)

    def _layer_average(self, n_vis: int) -> float:
        """Average vision-token count across all decoder layers (the paper's budget
        metric: ``num_token_pool / num_layers``), from this forward's prune history."""
        if not self._n_layers:
            return float(n_vis)
        return layer_average_tokens(n_vis, [layer for layer, _ in self._prune_history],
                                    [kept for _, kept in self._prune_history], self._n_layers)

    def _check_budget_reachable(self, n_vis: int, key_count: int) -> None:
        """Warn once when the schedule cannot average down to the budget label.

        The layers before the first prune hold every vision token and the key set
        survives all of them, so the layer-average has a hard floor that no threshold
        can beat. The built-in profiles are all above their floor; a custom
        ``key_rate`` / ``pruning_layers`` / ``vit_retention`` can fall below it, and
        the run would then quietly report a budget it never spent.
        """
        if self._warned_budget_floor or not self._n_layers or not self._pruning_layers:
            return
        self._warned_budget_floor = True
        floor = minimum_layer_average(n_vis, self._pruning_layers[0], key_count, self._n_layers)
        if floor <= self._max_tokens:
            return
        print(f"[MustDrop] WARNING: budget label {self._max_tokens} is below this schedule's floor of "
              f"{floor:.2f} layer-avg tokens (n_vis={n_vis}, first prune @layer{self._pruning_layers[0]}, "
              f"key_set={key_count}, {self._n_layers} layers). The run will average at least the floor. "
              f"Prune earlier, lower key_rate, or lower vit_retention.", flush=True)

    @staticmethod
    def _columns_to_local(columns: torch.Tensor, image_positions: torch.Tensor) -> torch.Tensor:
        """Map absolute sequence columns back to local image-token indices [0, n_vis)."""
        match = (image_positions.unsqueeze(0) == columns.unsqueeze(1))    # [n_pruned, n_vis]
        return match.float().argmax(dim=1)

    def _dual_attention(self, weights: torch.Tensor | None, hidden: torch.Tensor,
                        vis_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Global (sum over text, Eq. 8) and individual (max over text, Eq. 10)
        text -> vision attention over ``vis_positions``. ``text`` = the prompt tokens
        after the image block (causal masking already zeroes any pre-image token's
        attention to the image). Falls back to hidden-state dot products if the
        layer attention can't be recomputed."""
        img_full = self._image_mask[0]
        last_img = int(img_full.nonzero(as_tuple=False).max())
        seq_len = hidden.shape[1]
        idx = torch.arange(seq_len, device=hidden.device)
        text_positions = idx[(~img_full) & (idx > last_img)]               # question tokens after image
        if weights is not None and text_positions.numel():
            text_to_vis = weights[text_positions][:, vis_positions]        # [text, vis]
            return text_to_vis.sum(dim=0).float(), text_to_vis.max(dim=0).values.float()
        # Fallback: dot-product affinity between text and visual hidden states.
        visual = hidden[0, vis_positions].float()
        if text_positions.numel() == 0:
            uniform = torch.ones(vis_positions.numel(), device=hidden.device)
            return uniform, uniform
        text_feats = hidden[0, text_positions].float()
        scale = visual.shape[-1] ** 0.5
        affinity = F.softmax((text_feats @ visual.t()) / scale, dim=-1)    # [text, vis]
        return affinity.sum(dim=0), affinity.max(dim=0).values

    def _compute_llm_attention(self, module, hidden: torch.Tensor, kwargs: dict) -> torch.Tensor | None:
        """Head-averaged causal self-attention at this layer, recomputed from
        ``q_proj``/``k_proj`` + RoPE (and GQA). Returns ``[seq, seq]`` or None."""
        try:
            attn = module.self_attn
            normed = module.input_layernorm(hidden)                # LLaMA: RMSNorm before attention
            query, key = attn.q_proj(normed), attn.k_proj(normed)
            b, seq, _ = query.shape
            config = attn.config
            n_heads = int(config.num_attention_heads)
            n_kv = int(getattr(config, "num_key_value_heads", n_heads))
            head_dim = int(getattr(attn, "head_dim", query.shape[-1] // n_heads))
            query = query.view(b, seq, n_heads, head_dim)
            key = key.view(b, seq, n_kv, head_dim)
            q_norm, k_norm = getattr(attn, "q_norm", None), getattr(attn, "k_norm", None)
            if callable(q_norm):
                query = q_norm(query)
            if callable(k_norm):
                key = k_norm(key)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            pos_emb = kwargs.get("position_embeddings")
            if pos_emb is not None and _APPLY_ROPE is not None:
                cos, sin = pos_emb
                query, key = _APPLY_ROPE(query, key, cos, sin)
            if n_kv != n_heads:                                    # GQA: expand kv heads
                key = key.repeat_interleave(n_heads // n_kv, dim=1)
            logits = (query.float() @ key.float().transpose(-1, -2)) * (head_dim ** -0.5)
            causal = torch.ones(seq, seq, dtype=torch.bool, device=logits.device).tril()
            logits = logits.masked_fill(~causal, torch.finfo(logits.dtype).min)
            # Exclude already-pruned tokens from the softmax denominator so the
            # recomputed attention matches the reference's physically-shrunk
            # sequence (attention concentrates as tokens are removed).
            already = self._llm_pruned_columns
            if already is not None and already.numel():
                cols = already.to(logits.device)
                cols = cols[cols < seq]
                if cols.numel():
                    logits[..., cols] = torch.finfo(logits.dtype).min
            weights = logits.softmax(dim=-1)[0].mean(dim=0)        # head-averaged [seq, seq]
            if not self._logged_llm_attn:
                self._logged_llm_attn = True
                print(f"[MustDrop] LLM real-attention active (heads={n_heads}, kv={n_kv}, "
                      f"rope={pos_emb is not None and _APPLY_ROPE is not None})", flush=True)
            return weights
        except Exception as error:                                 # noqa: BLE001 - degrade to dot product
            if not self._logged_llm_attn:
                self._logged_llm_attn = True
                print(f"[MustDrop] LLM real-attention failed ({type(error).__name__}: {error}); "
                      f"using hidden-dot fallback", flush=True)
            return None

    def _mask_pruned_columns(self, kwargs: dict, hidden: torch.Tensor,
                             pruned_columns: torch.Tensor) -> dict:
        """Set ``-inf`` at the pruned key columns of the layer's attention mask,
        reusing an incoming 4-D mask or building a causal one (SDPA None path)."""
        device, dtype = hidden.device, hidden.dtype
        min_value = torch.finfo(dtype).min
        q_len = hidden.shape[1]
        incoming = kwargs.get("attention_mask")
        if isinstance(incoming, torch.Tensor) and incoming.dim() == 4:
            mask = incoming.clone()
        else:
            kv_len = q_len
            cache_position = kwargs.get("cache_position")
            position_ids = kwargs.get("position_ids")
            if isinstance(cache_position, torch.Tensor) and cache_position.numel():
                kv_len = int(cache_position[-1]) + 1
            elif isinstance(position_ids, torch.Tensor) and position_ids.numel():
                kv_len = int(position_ids.max()) + 1
            query_abs = torch.arange(q_len, device=device).unsqueeze(1) + (kv_len - q_len)
            key_abs = torch.arange(kv_len, device=device).unsqueeze(0)
            allowed = key_abs <= query_abs
            mask = torch.where(allowed, torch.zeros((), dtype=dtype, device=device),
                               torch.full((), min_value, dtype=dtype, device=device)).view(1, 1, q_len, kv_len)
        columns = pruned_columns.to(device=mask.device, dtype=torch.long)
        columns = columns[columns < mask.shape[-1]]
        if columns.numel():
            mask[..., columns] = min_value
        kwargs["attention_mask"] = mask
        return kwargs
