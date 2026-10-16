"""VFlowOpt: visual-information-flow-guided progressive token pruning.

Paper: "VFlowOpt: A Token Pruning Framework for LMMs with Visual Information
Flow-Guided Optimization" (arXiv:2508.05211, ICCV 2025). Reference code:
https://github.com/sihany077/VFlowOpt -- the pruning itself lives in
``src/LLaVA-OneVision/llava/model/multimodal_encoder/siglip_encoder.py`` (Eq. 1-4),
``src/LLaVA-OneVision/llava/model/language_model/llava_qwen_training_free.py``
(Eq. 6-7, the vision stage) and
``src/transformers-4.46.0/src/transformers/models/qwen2/modeling_qwen2.py``
(``forward_illava``, the two LLM stages). This is a faithful reimplementation for
Hugging Face ``llava-hf/llava-1.5-7b-hf``, adapted to layer-averaged token
budgets of 224 / 128 / 64 / 32.

Three pruning points, driven by a single importance map computed once in the ViT:

* **Vision stage (before the LLM).** Keep ``round(R1 * 576)`` tokens by
  importance; the pruned tokens are grouped into ``a x a`` grid cells and fused
  into one recycled token per cell (Sec. 3.2, Eq. 6-7).
* **LLM stages 2 and 3** (decoder layers 10 and 20 for a 32-layer Vicuna-7B).
  Keep the top ``R2`` and then ``R3`` fraction of the *surviving* visual tokens,
  ranked by the same ViT importance scores carried forward -- that is the
  "visual information flow" the method is named for; the LLM's own attention is
  never consulted (``combined_attn = VIT_ATTN_MAP`` upstream).

Budget semantics
----------------
The "pruning strategy" section below is pure arithmetic on the six
hyperparameters the paper searches over -- ``(R1, R2, R3, t, alpha, a)``
(Sec. 3.3, Algorithm 1) -- so it is fully testable without a GPU or a model.

The quantities that matter:

* **Stages (Sec. 3.2).** The LMM is split evenly into three stages of ``L1``,
  ``L2`` and ``L3`` decoder layers. ``R1`` is applied before the LLM (in the
  vision stage), ``R2`` at the start of stage 2 and ``R3`` at the start of
  stage 3.
* **Budget (Eq. 5).** The reported budget -- ``max_tokens`` -- is the
  *layer-averaged* visual token count::

      R_bar = (R1*L1 + R1*R2*L2 + R1*R2*R3*L3) / L

  which is what "224 / 128 / 64 / 32 tokens" means here -- NOT the number of
  tokens that survive to the last layer, and not the ``_max_tokens`` the base
  class holds (that one is the vision stage's own count). ``R3`` is never
  searched: it is solved from this constraint so every candidate spends exactly
  the budget.

Every budget in ``CALIBRATED_BUDGETS`` has its own fitted strategy;
``calibrated_strategy`` raises for an unlisted one instead of falling back to a
neighbour's parameters, so a run's label never describes a strategy fitted for a
different budget. Pass an explicit ``strategy`` to explore one.

Importance estimation (Sec. 3.1, Eq. 1-4)
-----------------------------------------
Three pieces, all training-free:

1. **Relatively important tokens (Eq. 1-2).** Inside one ViT layer, a token ``j``
   is "relatively important" when the attention it *receives* summed over every
   query exceeds ``tau = t * (1/N) * sum_ij A_ij``. For a row-stochastic ``A``
   the double sum is ``N``, so the test is simply ``sum_i A_ij > t`` -- i.e.
   ``t`` is a multiple of the average received attention. ``K`` is the set of
   such tokens.
2. **Patch entropy (Eq. 3).** The Shannon entropy of the 256-bin grey-level
   histogram of the image patch behind each token.
3. **Importance score (Eq. 4).** ``I_i = mean_{k in K} A_ki + alpha * softmax(H)_i``.

Faithfulness notes
------------------
* Eq. 4 writes the attention term as a **sum** over ``K``; the released code
  (``siglip_encoder.py``: ``attn_weights_mean[i][global_token_indice, :].mean(0)``)
  uses the **mean**. We follow the code: the mean keeps the attention term on
  the same ``O(1/N)`` scale as ``softmax(H)``, which is what makes the searched
  range ``alpha in [-1, 1]`` meaningful at all. With a sum the entropy term
  would be ``|K|`` times too small to ever matter.
* The released code computes the entropy from the **model-normalised**
  ``pixel_values`` via ``(patches * 255).to(torch.uint8)``, which wraps
  out-of-range values rather than recovering 8-bit pixels. Entropy is invariant
  to any relabelling of grey levels, so this is still a valid intra-patch
  diversity measure, and it is what produced the published numbers -- so it is
  the default (``entropy_mode="reference"``). ``"raw"`` un-normalises first and
  gives the paper's literal 8-bit grey levels.

Token recycling (Sec. 3.2, Eq. 6-7)
-----------------------------------
The first pruning stage runs *before* the LLM. It keeps the ``round(R1 * N)``
most important tokens, but rather than discarding the rest outright it groups the
pruned tokens into ``a x a`` grid cells (Eq. 6) and fuses each cell into one
recycled token by an importance-weighted average (Eq. 7). The fused token takes
the slot of the highest-importance member of its cell and re-joins the kept set.

Upstream takes ``reduce_tokens + int(N / a^2)`` low-importance tokens as the
recycling pool and then emits one token per *non-empty* cell, so the surviving
count depends on how the low-importance tokens happen to scatter over the grid.
transformers scatters image features into ``<image>`` placeholders under a strict
count check, so the count has to be predictable from ``(R1, a)`` alone. We
therefore size the pool with the exact cell count of the grid and, in the
(vanishingly rare) event that a cell draws no pruned token, top the kept set back
up with the highest-importance tokens still left in the pool. The result is
always exactly ``round(R1 * N)`` tokens, and it coincides with upstream's output
whenever every cell is populated -- which, at every budget and grid size used
here, is the overwhelmingly common case (the pool covers 60-95% of all tokens).

Harness adaptations
-------------------
LLMOE drives a stock ``LlavaForConditionalGeneration``; the reference patches a
vendored LLaVA-OneVision + transformers-4.46 tree. Everything here is therefore
expressed as forward hooks and method wrappers, and the LLMOE submodule is not
modified. The consequences:

* **Attention source.** The reference reads the full self-attention of the last
  ViT layer that feeds the projector. For LLaVA-1.5 that is the CLIP layer whose
  output is ``hidden_states[vision_feature_layer]`` (layer 22 of 24 at the
  default ``vision_feature_layer=-2``). SDPA returns no attention weights, so the
  map is recomputed from that layer's ``q_proj``/``k_proj``.
* **[CLS].** SigLIP has no class token; CLIP does. ``[CLS]`` participates in the
  attention map and may itself be selected as a "relatively important" token
  (Eq. 2) -- it is exactly the kind of global token the criterion looks for --
  but it carries no image patch, so it is dropped from the importance map before
  pruning, matching LLaVA's ``vision_feature_select_strategy="default"``.
* **Deterministic vision-stage count.** See "Token recycling" above.
* **Physical sequence shrinkage.** Stages 2 and 3 really remove tokens from the
  hidden state and the KV cache rather than masking them, so the KV-cache and
  prefill savings the paper reports are actually realised. Position IDs are
  gathered, never renumbered ("the position IDs of visual tokens remain
  unchanged following pruning", Sec. 3.2). This requires ``batch_size == 1``.

Qwen2.5-VL adaptation
---------------------
Qwen is configured with ``retain_ratio``, the visual-token fraction averaged
over its decoder layers. The three cumulative keeps follow a geometric decay
solved for the loaded decoder depth, so counts are derived independently for
every dynamic NaViT grid. The vision stage uses the shared VisionZip-style
final-block received-attention capture and rectangular cell recycling, then the
same importance is carried through physical M-RoPE-aware decoder cuts. This is
an in-repo first-pass adaptation: the released calibration and entropy term are
specific to fixed-grid LLaVA-OneVision/SigLIP and are never presented as Qwen
parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from json import loads
from os import PathLike
from pathlib import Path
from typing import Any, Final, Iterable, override

import torch
from torch.nn import functional as F
from torch.utils.hooks import RemovableHandle

from llmoe.compressor.internvl import trim_image_placeholders
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.utils.depth import REFERENCE_DEPTH
from llmoe.weights import resolve_weights

# ---------------------------------------------------------------------------
# Pruning strategy: stages, budgets, and the search space (Sec. 3.3, Alg. 1)
# ---------------------------------------------------------------------------


# LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf): CLIP-ViT-L/14-336 -> 24x24 patches,
# Vicuna-7B -> 32 decoder layers.
LLAVA15_VISUAL_TOKENS: int = 576
LLAVA15_GRID_SIDE: int = 24
LLAVA15_DECODER_LAYERS: int = REFERENCE_DEPTH

# Reference search bounds, read off `optimize_ratios` in
# VFlowOpt/src/lmms_eval-0.2.4/lmms_eval/models/llava_ov_opt_all.py:
#     Real(0.25, 0.375, "ratio_v"), Real(0.2/0.375, 1.0, "ratio_1"),
#     Real(0.0, 50.0, "vit_t"),     Real(3.0, 24, "vit_m"), Real(-1, 1, "alpha_v")
# There, TARGET_AVERAGE = 0.25, so the R1 window is [T, 1.5*T] and the R2 window
# is the constant [0.2/0.375, 1.0]. We keep the same shape and re-derive the R1
# window from whichever budget is being fitted.
R1_LOW_FACTOR: float = 1.0
R1_HIGH_FACTOR: float = 1.5
R2_LOW: float = 0.2 / 0.375  # == 8/15
R2_HIGH: float = 1.0
T_LOW: float = 0.0
T_HIGH: float = 50.0
ALPHA_LOW: float = -1.0
ALPHA_HIGH: float = 1.0
A_LOW: float = 3.0
A_HIGH: float = 24.0


def grid_cells(a: float, side: int) -> int:
    """Number of ``a x a`` grid cells covering a ``side x side`` token grid.

    Cell membership is ``floor(coord / a)`` (paper Eq. 6, and
    ``(points.float() / a).floor().long()`` in the reference), so the number of
    distinct row (or column) indices over coordinates ``0..side-1`` is
    ``floor((side - 1) / a) + 1``. ``a`` is a real number, not an integer.
    """
    if a <= 0:
        raise ValueError(f"grid side length a must be > 0, got {a}")
    if side < 1:
        raise ValueError(f"grid side must be >= 1, got {side}")
    return (int((side - 1) // a) + 1) ** 2


def min_grid_side_length(max_cells: int, side: int, *, low: float = A_LOW, high: float = A_HIGH,
                         step: float = 0.01) -> float:
    """Smallest ``a`` in ``[low, high]`` whose grid has at most ``max_cells`` cells.

    The vision stage emits exactly one recycled token per grid cell, so a run
    whose kept-token count can drop to ``max_cells`` needs ``a`` large enough
    that the cells still fit inside the budget. At the tightest budgets the
    reference's ``a >= 3`` floor is infeasible and has to be raised.

    The grid has ``(floor((side-1)/a) + 1)^2`` cells, so the constraint is
    ``a > (side - 1) / floor(sqrt(max_cells))``. That bound is exact but often
    lands on a value where binary floating point decides ``floor`` either way, so
    the result is snapped up to a ``step`` grid and then verified.
    """
    if max_cells < 1:
        raise ValueError(f"max_cells must be >= 1, got {max_cells}")
    if grid_cells(low, side) <= max_cells:
        return low
    rows = int(max_cells ** 0.5)
    # Round to the step grid *before* verifying: the exact bound (side-1)/rows is
    # often a value whose nearest double falls on the other side of the floor.
    a = round(max(low, (int((side - 1) / rows / step) + 1) * step), 6)
    while a <= high and grid_cells(a, side) > max_cells:
        a = round(a + step, 6)
    if a > high:
        raise ValueError(f"no a in [{low}, {high}] yields <= {max_cells} cells on a {side}x{side} grid")
    return a


@dataclass(frozen=True)
class StageLayers:
    """Decoder-layer counts of the three progressive-pruning stages (Sec. 3.2)."""

    l1: int
    l2: int
    l3: int

    def __post_init__(self) -> None:
        if min(self.l1, self.l2, self.l3) < 1:
            raise ValueError(f"every stage needs >= 1 layer, got {(self.l1, self.l2, self.l3)}")

    @property
    def total(self) -> int:
        return self.l1 + self.l2 + self.l3

    @property
    def prune_layers(self) -> tuple[int, int]:
        """0-indexed decoder layers at which stages 2 and 3 start pruning."""
        return self.l1, self.l1 + self.l2

    @classmethod
    def even(cls, num_layers: int) -> "StageLayers":
        """The paper's "evenly divide the LMM into three stages".

        Reproduces the released settings exactly: 28 layers -> 9/9/10, i.e.
        pruning after layers 9 and 18 (LLaVA-OneVision-7B / Qwen2-VL-7B), and
        32 layers -> 10/10/12, i.e. pruning after layers 10 and 20 -- which is
        the split the paper states for LLaVA-NeXT-7B, the other Vicuna-7B
        (32-layer) model it evaluates.
        """
        if num_layers < 3:
            raise ValueError(f"need >= 3 decoder layers, got {num_layers}")
        base = num_layers // 3
        return cls(base, base, num_layers - 2 * base)


LLAVA15_STAGES: StageLayers = StageLayers.even(LLAVA15_DECODER_LAYERS)  # 10 / 10 / 12


def layer_average_tokens(keeps: tuple[int, int, int], stages: StageLayers) -> float:
    """Eq. 5 in token units: the visual-token count averaged over all layers."""
    n1, n2, n3 = keeps
    return (n1 * stages.l1 + n2 * stages.l2 + n3 * stages.l3) / stages.total


def layer_average_ratio(r1: float, r2: float, r3: float, stages: StageLayers) -> float:
    """Eq. 5 verbatim, on continuous retention ratios."""
    return (r1 * stages.l1 + r1 * r2 * stages.l2 + r1 * r2 * r3 * stages.l3) / stages.total


def solve_r3_continuous(r1: float, r2: float, target_ratio: float, stages: StageLayers) -> float:
    """Solve Eq. 5 for ``R3``, exactly as ``bayesian_objective`` does upstream::

        ratio_list_3 = min(1.0, ((L*T - R1*L1 - R1*R2*L2) / L3) / (R1*R2))

    Upstream clamps only from above (an under-spent budget); the search bounds
    are what keep the numerator positive. We clamp from below too so a caller
    that widens the bounds cannot produce a negative retention ratio.
    """
    denominator = r1 * r2 * stages.l3
    if denominator <= 0:
        raise ValueError(f"R1*R2 must be > 0 to solve for R3, got R1={r1}, R2={r2}")
    numerator = target_ratio * stages.total - r1 * stages.l1 - r1 * r2 * stages.l2
    return float(min(1.0, max(0.0, numerator / denominator)))


def keeps_from_ratios(r1: float, r2: float, r3: float, n_visual: int,
                      stages: StageLayers) -> tuple[int, int, int]:
    """Integer tokens kept in each stage.

    ``n1`` is the vision stage's deterministic output count; ``n2``/``n3``
    follow the reference's ``r = round(n * (1 - ratio)); keep = n - r`` rule
    (``modeling_qwen2.py``: ``r = round(image_token_length * (1 - r_val))``).
    """
    n1 = int(round(n_visual * r1))
    n1 = max(0, min(n_visual, n1))
    n2 = max(0, n1 - int(round(n1 * (1.0 - r2))))
    n3 = max(0, n2 - int(round(n2 * (1.0 - r3))))
    return n1, n2, n3


def solve_keeps_integer(r1: float, r2: float, budget_tokens: float, n_visual: int,
                        stages: StageLayers) -> tuple[int, int, int]:
    """Solve the budget constraint on integer token counts instead of ratios.

    Same equation as :func:`solve_r3_continuous`, but the free variable is the
    stage-3 token count, so the *realised* layer average is guaranteed to be the
    largest achievable value that does not exceed ``budget_tokens``. Solving in
    ratio space and then rounding can overshoot the budget by a fraction of a
    token, which matters when budgets are quoted as exact counts (224/128/64/32).
    """
    n1 = int(round(n_visual * r1))
    n1 = max(0, min(n_visual, n1))
    n2 = max(0, n1 - int(round(n1 * (1.0 - r2))))
    slack = budget_tokens * stages.total - n1 * stages.l1 - n2 * stages.l2
    n3 = int(slack // stages.l3) if slack > 0 else 0
    return n1, n2, max(0, min(n2, n3))


@dataclass(frozen=True)
class PruningStrategy:
    """A fully resolved VFlowOpt pruning strategy ``s = (R1, R2, R3, t, alpha, a)``."""

    r1: float
    r2: float
    r3: float
    t: float
    alpha: float
    a: float
    n_visual: int = LLAVA15_VISUAL_TOKENS
    grid_side: int = LLAVA15_GRID_SIDE
    stages: StageLayers = LLAVA15_STAGES

    @property
    def keeps(self) -> tuple[int, int, int]:
        return keeps_from_ratios(self.r1, self.r2, self.r3, self.n_visual, self.stages)

    @property
    def prune_layers(self) -> tuple[int, int]:
        return self.stages.prune_layers

    @property
    def n_cells(self) -> int:
        return grid_cells(self.a, self.grid_side)

    @property
    def layer_average(self) -> float:
        return layer_average_tokens(self.keeps, self.stages)

    def to_dict(self) -> dict[str, Any]:
        n1, n2, n3 = self.keeps
        payload = asdict(self)
        payload["stages"] = {"l1": self.stages.l1, "l2": self.stages.l2, "l3": self.stages.l3}
        payload.update(
            prune_layers=list(self.prune_layers),
            keeps=[n1, n2, n3],
            n_cells=self.n_cells,
            layer_average_tokens=self.layer_average,
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PruningStrategy":
        stages = payload.get("stages")
        if isinstance(stages, dict):
            stages = StageLayers(int(stages["l1"]), int(stages["l2"]), int(stages["l3"]))
        elif stages is None:
            stages = LLAVA15_STAGES
        return cls(
            r1=float(payload["r1"]), r2=float(payload["r2"]), r3=float(payload["r3"]),
            t=float(payload["t"]), alpha=float(payload["alpha"]), a=float(payload["a"]),
            n_visual=int(payload.get("n_visual", LLAVA15_VISUAL_TOKENS)),
            grid_side=int(payload.get("grid_side", LLAVA15_GRID_SIDE)),
            stages=stages,
        )


@dataclass(frozen=True)
class SearchSpace:
    """Bounds for the five searched dimensions, specialised to one token budget.

    ``R3`` is absent by construction -- it is solved from the budget constraint
    (Algorithm 1, lines 7 and 14).
    """

    budget_tokens: int
    n_visual: int = LLAVA15_VISUAL_TOKENS
    grid_side: int = LLAVA15_GRID_SIDE
    stages: StageLayers = LLAVA15_STAGES

    @property
    def target_ratio(self) -> float:
        return self.budget_tokens / self.n_visual

    @property
    def r1_bounds(self) -> tuple[float, float]:
        target = self.target_ratio
        return R1_LOW_FACTOR * target, min(1.0, R1_HIGH_FACTOR * target)

    @property
    def r2_bounds(self) -> tuple[float, float]:
        return R2_LOW, R2_HIGH

    @property
    def t_bounds(self) -> tuple[float, float]:
        return T_LOW, T_HIGH

    @property
    def a_bounds(self) -> tuple[float, float]:
        # The vision stage emits one recycled token per grid cell on top of the
        # kept tokens, so the cell count must fit inside the smallest kept-token
        # count the R1 window allows, round(n_visual * R1_low) == budget_tokens.
        low = min_grid_side_length(self.budget_tokens, self.grid_side, low=A_LOW, high=A_HIGH)
        return low, A_HIGH

    @property
    def alpha_bounds(self) -> tuple[float, float]:
        return ALPHA_LOW, ALPHA_HIGH

    @property
    def names(self) -> tuple[str, ...]:
        return "r1", "r2", "t", "a", "alpha"

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return (self.r1_bounds, self.r2_bounds, self.t_bounds, self.a_bounds, self.alpha_bounds)

    def clip(self, point: Iterable[float]) -> list[float]:
        """Upstream's ``clip_point_to_bounds``: the optimizer's proposal is clipped
        into the box before it is evaluated *and* before it is told back."""
        return [float(min(max(value, low), high))
                for value, (low, high) in zip(point, self.bounds)]

    def resolve(self, point: Iterable[float], *, budget_mode: str = "integer") -> PruningStrategy:
        """Turn a searched point ``(R1, R2, t, a, alpha)`` into a full strategy.

        ``budget_mode="integer"`` (default) solves the budget constraint on
        integer token counts, so the realised layer average never exceeds the
        budget. ``budget_mode="continuous"`` reproduces the reference's ratio
        space solve verbatim.
        """
        r1, r2, t, a, alpha = self.clip(point)
        if budget_mode == "continuous":
            r3 = solve_r3_continuous(r1, r2, self.target_ratio, self.stages)
        elif budget_mode == "integer":
            n1, n2, n3 = solve_keeps_integer(r1, r2, self.budget_tokens, self.n_visual, self.stages)
            r3 = (n3 / n2) if n2 > 0 else 0.0
        else:
            raise ValueError(f"budget_mode must be 'integer' or 'continuous', got {budget_mode!r}")
        return PruningStrategy(r1=r1, r2=r2, r3=r3, t=t, alpha=alpha, a=a,
                               n_visual=self.n_visual, grid_side=self.grid_side, stages=self.stages)


# ---------------------------------------------------------------------------
# Calibrated LLaVA-1.5 strategies shipped with the baseline port
# ---------------------------------------------------------------------------


# Copied from configs/strategy_b*.json at baseline commit 6d0f44e. Budgets are
# Eq. 5 decoder-layer averages, not final token counts.
_STRATEGY_PARAMETERS: Final[dict[int, dict[str, float]]] = {
    32: {
        "r1": 0.07959425853945315,
        "r2": 0.7724896511858073,
        "r3": 0.4722222222222222,
        "t": 50.0,
        "alpha": 0.020467921635002106,
        "a": 4.61,
    },
    58: {
        "r1": 0.14047758797813364,
        "r2": 0.8161146548978053,
        "r3": 0.48484848484848486,
        "t": 46.31504392566746,
        "alpha": 0.829919351087562,
        "a": 16.773805198145272,
    },
    64: {
        "r1": 0.15500975225173363,
        "r2": 0.8161146548978053,
        "r3": 0.4794520547945205,
        "t": 46.31504392566746,
        "alpha": 0.829919351087562,
        "a": 16.672617535540837,
    },
    128: {
        "r1": 0.2893797088086464,
        "r2": 0.7852591759407808,
        "r3": 0.7099236641221374,
        "t": 10.153061236738473,
        "alpha": 0.19773093297707223,
        "a": 22.799924981717606,
    },
    144: {
        "r1": 0.32555217240972717,
        "r2": 0.7852591759407808,
        "r3": 0.7027027027027027,
        "t": 10.153061236738473,
        "alpha": 0.19773093297707223,
        "a": 22.799924981717606,
    },
    224: {
        "r1": 0.549594908834493,
        "r2": 0.9818484676783585,
        "r3": 0.2379421221864952,
        "t": 35.51411264587945,
        "alpha": -0.2771242989755692,
        "a": 14.396773675102564,
    },
    288: {
        "r1": 0.75,
        "r2": 0.7509769961744376,
        "r3": 0.42592592592592593,
        "t": 50.0,
        "alpha": -0.060852995258160236,
        "a": 24.0,
    },
}

CALIBRATED_BUDGETS: Final[tuple[int, ...]] = tuple(sorted(_STRATEGY_PARAMETERS))


def load_strategy_file(path: str | PathLike[str]) -> PruningStrategy:
    """Read a calibration JSON, as written by the search and shipped in the bundle.

    The released files wrap the six parameters under a ``"strategy"`` key
    alongside the objective and search provenance; a bare parameter mapping is
    accepted too, so hand-written files keep working.
    """
    payload = loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} does not hold a JSON object")
    inner = payload.get("strategy", payload)
    if not isinstance(inner, Mapping):
        raise TypeError(f'{path} has a non-object "strategy" entry')
    return PruningStrategy.from_dict(dict(inner))


def calibrated_strategy(budget_tokens: int) -> PruningStrategy:
    """Return the baseline's fitted LLaVA-1.5 strategy for one budget."""
    budget_tokens = int(budget_tokens)
    try:
        parameters = _STRATEGY_PARAMETERS[budget_tokens]
    except KeyError as error:
        supported = ", ".join(str(value) for value in CALIBRATED_BUDGETS)
        raise ValueError(
            f"No calibrated VFlowOpt strategy for budget {budget_tokens}. "
            f"Supported LLaVA-1.5 budgets: {supported}; pass an explicit strategy "
            "for any other budget."
        ) from error
    return PruningStrategy.from_dict(parameters)


# ---------------------------------------------------------------------------
# Visual token importance estimation (Sec. 3.1, Eq. 1-4)
# ---------------------------------------------------------------------------


# Reference constant: the three RGB channels are averaged with these literal
# weights (``torch.tensor([0.3333, 0.3333, 0.3333])``), not with 1/3.
_GREY_WEIGHTS: tuple[float, float, float] = (0.3333, 0.3333, 0.3333)
_ENTROPY_EPS: float = 1e-10
_GREY_LEVELS: int = 256


def extract_patches(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``[B, C, H, W] -> [B * n_patches, C, patch, patch]`` in raster order."""
    if pixel_values.dim() != 4:
        raise ValueError(f"expected [B, C, H, W] pixel values, got {tuple(pixel_values.shape)}")
    channels = pixel_values.shape[1]
    patches = pixel_values.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    return patches.contiguous().view(-1, channels, patch_size, patch_size)


def patch_entropy(pixel_values: torch.Tensor, *, patch_size: int = 14, mode: str = "reference",
                  image_mean: tuple[float, ...] | None = None,
                  image_std: tuple[float, ...] | None = None) -> torch.Tensor:
    """Eq. 3: grey-level entropy of every image patch. Returns ``[B, n_patches]``.

    ``mode="reference"`` reproduces ``SigLipVisionEmbeddings.forward`` byte for
    byte. ``mode="raw"`` first inverts the vision tower's normalisation with
    ``image_mean``/``image_std`` so the grey levels really are 0-255 intensities.
    """
    if pixel_values.dim() != 4:
        raise ValueError(f"expected [B, C, H, W] pixel values, got {tuple(pixel_values.shape)}")
    batch, _, height, width = pixel_values.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"image {height}x{width} is not divisible by patch size {patch_size}")
    n_patches = (height // patch_size) * (width // patch_size)

    patches = extract_patches(pixel_values.float(), patch_size)
    if mode == "reference":
        scaled = (patches * 255).to(torch.uint8).to(torch.float32)
    elif mode == "raw":
        if image_mean is None or image_std is None:
            raise ValueError("mode='raw' needs the processor's image_mean/image_std")
        mean = torch.tensor(image_mean, device=patches.device, dtype=patches.dtype).view(1, -1, 1, 1)
        std = torch.tensor(image_std, device=patches.device, dtype=patches.dtype).view(1, -1, 1, 1)
        scaled = ((patches * std + mean) * 255).clamp_(0, 255)
    else:
        raise ValueError(f"mode must be 'reference' or 'raw', got {mode!r}")

    weights = torch.tensor(_GREY_WEIGHTS, device=scaled.device, dtype=scaled.dtype).view(1, -1, 1, 1)
    grey = (scaled * weights).sum(dim=1).view(scaled.shape[0], -1)
    levels = grey.round().clamp_(0, _GREY_LEVELS - 1).to(torch.long)

    histogram = torch.zeros(levels.shape[0], _GREY_LEVELS, device=levels.device, dtype=torch.float32)
    histogram.scatter_add_(1, levels, torch.ones_like(levels, dtype=torch.float32))
    probabilities = histogram / histogram.sum(dim=1, keepdim=True)
    entropy = -(probabilities * torch.log2(probabilities + _ENTROPY_EPS)).sum(dim=1)
    return entropy.view(batch, n_patches)


def relatively_important_tokens(attention: torch.Tensor, t: float) -> torch.Tensor:
    """Eq. 1-2: boolean mask ``[B, N]`` of tokens whose received attention beats ``tau``.

    ``attention`` is the head-averaged ``[B, N, N]`` map. ``received[j]`` is the
    column mean and ``received.mean() == 1/N`` for a row-stochastic map, so
    ``received > t * received.mean()`` is exactly ``sum_i A_ij > t``.
    """
    if attention.dim() != 3 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError(f"expected a square [B, N, N] attention map, got {tuple(attention.shape)}")
    received = attention.mean(dim=1)
    return received > t * received.mean(dim=-1, keepdim=True)


def context_relevance(attention: torch.Tensor, t: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The attention half of Eq. 4: ``mean_{k in K} A_k.`` per image.

    Returns ``(scores [B, N], k_sizes [B])``. Where ``K`` is empty the score row
    is ``NaN`` -- exactly what the reference produces (``x[empty].mean(0)``), and
    the calibration objective treats such a candidate as maximally bad. Callers
    that must not emit NaN features should use :func:`context_relevance_safe`.
    """
    important = relatively_important_tokens(attention, t)
    weights = important.to(attention.dtype)
    k_sizes = weights.sum(dim=-1)
    scores = torch.einsum("bk,bkj->bj", weights, attention) / k_sizes.unsqueeze(-1)
    return scores, k_sizes


def context_relevance_safe(attention: torch.Tensor, t: float) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`context_relevance` with the empty-``K`` row replaced by the plain
    received-attention map (equivalent to ``t = 0``, i.e. every token is a
    context token). Returns ``(scores, k_sizes)``; ``k_sizes == 0`` flags the
    rows where the fallback fired."""
    scores, k_sizes = context_relevance(attention, t)
    empty = k_sizes <= 0
    if bool(empty.any()):
        scores = torch.where(empty.unsqueeze(-1), attention.mean(dim=1), scores)
    return scores, k_sizes


def importance_score(context: torch.Tensor, entropy: torch.Tensor, alpha: float) -> torch.Tensor:
    """Eq. 4: ``I = context_relevance + alpha * softmax(H)``.

    ``context`` is ``[B, n_patches]`` (already restricted to patch tokens) and
    ``entropy`` is ``[B, n_patches]``.
    """
    if context.shape != entropy.shape:
        raise ValueError(f"context {tuple(context.shape)} and entropy {tuple(entropy.shape)} differ")
    return context + alpha * F.softmax(entropy.float(), dim=-1).to(context.dtype)


def head_averaged_self_attention(layer: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Recompute one CLIP encoder layer's full ``[B, N, N]`` head-averaged attention.

    LLaVA runs the vision tower under SDPA, which returns no attention weights,
    so we replay the layer's pre-attention LayerNorm and its ``q_proj``/``k_proj``
    on the captured input hidden states. This is the same recompute the sibling
    MustDrop/VisionTrim ports use, except that VFlowOpt needs the whole softmax
    rather than just the ``[CLS]`` row.
    """
    attention = getattr(layer, "self_attn", None)
    if attention is None:
        attention = getattr(layer, "attention", None)
    norm = getattr(layer, "layer_norm1", None)
    if norm is None:
        norm = getattr(layer, "layernorm_before", None)
    if attention is None or norm is None:
        raise TypeError("VFlowOpt needs a vision layer with attention and pre-attention norm")
    normed = norm(hidden_states)
    query, key = attention.q_proj(normed), attention.k_proj(normed)
    q_norm, k_norm = getattr(attention, "q_norm", None), getattr(attention, "k_norm", None)
    if callable(q_norm):
        query = q_norm(query)
    if callable(k_norm):
        key = k_norm(key)
    batch, seq_len, dim = query.shape
    heads = int(getattr(attention, "num_heads", None) or attention.config.num_attention_heads)
    head_dim = dim // heads
    scale = float(getattr(attention, "scale", getattr(attention, "scaling", head_dim ** -0.5)))
    query = query.view(batch, seq_len, heads, head_dim).transpose(1, 2).float() * scale
    key = key.view(batch, seq_len, heads, head_dim).transpose(1, 2).float()
    logits = query @ key.transpose(-1, -2)
    return logits.softmax(dim=-1).mean(dim=1)


# ---------------------------------------------------------------------------
# Vision-stage pruning with token recycling (Sec. 3.2, Eq. 6-7)
# ---------------------------------------------------------------------------


_WEIGHT_SUM_EPS: float = 1e-8


def grid_cell_ids(n_tokens: int, side: int, a: float, device: torch.device, *,
                  grid_h: int | None = None, grid_w: int | None = None,
                  reference_side: int | None = None) -> torch.Tensor:
    """Eq. 6 cell IDs for a square grid or a normalized rectangular grid.

    reference_side maps a rectangular concatenated InternVL layout onto the
    strategy calibration grid before applying a. This preserves the calibrated
    number of recycling cells while making one global decision over every tile
    rather than repeating the cells and budget per tile.
    """
    grid_h = int(side if grid_h is None else grid_h)
    grid_w = int(side if grid_w is None else grid_w)
    if grid_h * grid_w != n_tokens:
        raise ValueError(f"{n_tokens} tokens do not form a {grid_h}x{grid_w} grid")
    coords = torch.arange(n_tokens, device=device)
    rows = torch.div(coords, grid_w, rounding_mode="floor")
    cols = coords - rows * grid_w
    if reference_side is not None:
        reference_side = int(reference_side)
        row_scale = (reference_side - 1) / max(1, grid_h - 1)
        col_scale = (reference_side - 1) / max(1, grid_w - 1)
        rows = rows.float() * row_scale
        cols = cols.float() * col_scale
        cell_extent = reference_side
    else:
        rows = rows.float()
        cols = cols.float()
        cell_extent = grid_w
    n_cols = int((cell_extent - 1) // a) + 1
    return torch.div(rows, a).floor().long() * n_cols + torch.div(cols, a).floor().long()


@dataclass
class VisionStageResult:
    """Output of the vision stage for one image."""

    features: torch.Tensor       # [keep_target, d]
    importance: torch.Tensor     # [keep_target] scores carried into the LLM stages
    keep_indices: torch.Tensor   # [keep_target] original token indices, ascending
    source: torch.Tensor         # [keep_target, N] exact selection/recycling weights
    merged_cells: int            # number of cells that actually fused >= 2 tokens
    topped_up: int               # tokens kept by the empty-cell repair path


def vision_stage(features: torch.Tensor, importance: torch.Tensor, *, keep_target: int,
                 a: float, side: int, n_cells: int, grid_h: int | None = None,
                 grid_w: int | None = None, reference_side: int | None = None) -> VisionStageResult:
    """Prune to ``keep_target`` tokens, recycling the pruned ones (Eq. 6-7).

    ``features`` is ``[N, d]`` in raster order and ``importance`` is ``[N]``.
    """
    n_tokens, _ = features.shape
    if importance.shape != (n_tokens,):
        raise ValueError(f"importance {tuple(importance.shape)} does not match {n_tokens} tokens")
    if not 0 < keep_target <= n_tokens:
        raise ValueError(f"keep_target must be in (0, {n_tokens}], got {keep_target}")
    if keep_target == n_tokens:
        keep_indices = torch.arange(n_tokens, device=features.device)
        return VisionStageResult(
            features=features,
            importance=importance,
            keep_indices=keep_indices,
            source=torch.eye(n_tokens, device=features.device, dtype=torch.float32),
            merged_cells=0,
            topped_up=0,
        )
    cell_ids = grid_cell_ids(
        n_tokens, side, a, features.device,
        grid_h=grid_h, grid_w=grid_w, reference_side=reference_side,
    )
    actual_cells = int(torch.unique(cell_ids).numel())
    if actual_cells != n_cells:
        raise ValueError(f"grid exposes {actual_cells} occupied cells, caller supplied {n_cells}")
    if actual_cells > keep_target:
        raise ValueError(f"grid has {actual_cells} cells but only {keep_target} tokens may be kept")

    device = features.device
    scores = importance.float()
    # Recycling pool: the tokens that would be pruned, plus one slot per cell so
    # the fused tokens are paid for out of the pool rather than the kept budget.
    pool_size = min(n_tokens, n_tokens - keep_target + n_cells)
    order = torch.argsort(scores, descending=False)
    pool = order[:pool_size]
    pool_scores = scores[pool]

    keep_mask = torch.ones(n_tokens, dtype=torch.bool, device=device)
    keep_mask[pool] = False

    # Group the pool by grid cell (Eq. 6). `inverse` numbers the occupied cells
    # 0..C-1; empty cells simply do not appear.
    cells = cell_ids[pool]
    _, inverse = torch.unique(cells, return_inverse=True)
    n_groups = int(inverse.max()) + 1 if inverse.numel() else 0

    # Eq. 7, importance-weighted average, computed for every cell at once.
    totals = torch.zeros(n_groups, device=device, dtype=torch.float32)
    totals.index_add_(0, inverse, pool_scores)
    counts = torch.zeros(n_groups, device=device, dtype=torch.float32)
    counts.index_add_(0, inverse, torch.ones_like(pool_scores))
    # alpha < 0 can push individual scores negative, so fall back to a plain mean
    # wherever the weights of a cell cancel out.
    degenerate = totals.abs() < _WEIGHT_SUM_EPS
    weights = torch.where(degenerate[inverse], 1.0 / counts[inverse],
                          pool_scores / torch.where(degenerate, torch.ones_like(totals), totals)[inverse])
    merged = torch.zeros(n_groups, features.shape[1], device=device, dtype=torch.float32)
    merged.index_add_(0, inverse, weights.unsqueeze(-1) * features[pool].float())

    # The fused token replaces the highest-importance member of its cell. `pool`
    # is score-ascending, so ties resolve to the first member of the group in
    # pool order -- the same token torch.argmax would pick.
    maxima = torch.full((n_groups,), float("-inf"), device=device, dtype=torch.float32)
    maxima.scatter_reduce_(0, inverse, pool_scores, reduce="amax", include_self=True)
    is_max = pool_scores >= maxima[inverse]
    slots = torch.arange(pool.numel(), device=device)
    representative_slot = torch.full((n_groups,), pool.numel(), device=device, dtype=torch.long)
    representative_slot.scatter_reduce_(0, inverse[is_max], slots[is_max], reduce="amin",
                                        include_self=True)
    representatives = pool[representative_slot]

    fused = features.clone()
    fused[representatives] = merged.to(features.dtype)
    keep_mask[representatives] = True
    merged_cells = int((counts > 1).sum())

    # Empty cells would leave us short of keep_target; refill with the best of
    # what is still pooled so the placeholder count stays exact.
    topped_up = keep_target - int(keep_mask.sum())
    if topped_up > 0:
        leftovers = pool[~keep_mask[pool]]
        best = leftovers[torch.argsort(scores[leftovers], descending=True)[:topped_up]]
        keep_mask[best] = True
    elif topped_up < 0:  # pragma: no cover - impossible: n_cells <= keep_target
        raise RuntimeError(f"vision stage kept {int(keep_mask.sum())} tokens, wanted {keep_target}")

    keep_indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)
    source = torch.zeros(
        keep_target, n_tokens, device=device, dtype=torch.float32
    )
    row_of = torch.full((n_tokens,), -1, device=device, dtype=torch.long)
    row_of[keep_indices] = torch.arange(keep_target, device=device)
    source[torch.arange(keep_target, device=device), keep_indices] = 1.0
    for group in range(n_groups):
        row = row_of[representatives[group]]
        members = pool[inverse == group]
        source[row] = 0.0
        source[row, members] = weights[inverse == group]
    return VisionStageResult(
        features=fused[keep_indices],
        importance=importance[keep_indices],
        keep_indices=keep_indices,
        source=source,
        merged_cells=merged_cells,
        topped_up=max(0, topped_up),
    )


def llm_stage_keep(importance: torch.Tensor, keep_target: int) -> torch.Tensor:
    """Stages 2-3: indices of the ``keep_target`` most important surviving tokens.

    Upstream sorts ascending and drops the first ``r = round(n * (1 - R))``
    (``modeling_qwen2.forward_illava``); the survivors stay in sequence order,
    which is what the returned ascending index tensor encodes.
    """
    n_tokens = int(importance.numel())
    keep_target = max(0, min(n_tokens, int(keep_target)))
    order = torch.argsort(importance.float(), descending=False)
    return torch.sort(order[n_tokens - keep_target:]).values


# ---------------------------------------------------------------------------
# The compressor
# ---------------------------------------------------------------------------


class EmptyContextSet(RuntimeError):
    """Raised in strict mode when Eq. 2's threshold selects no context tokens.

    The reference produces a NaN importance map in this case, which the
    calibration objective turns into zero cosine similarity, i.e. the candidate
    is rejected. Calibration reproduces that; inference falls back to plain
    received attention so a forward can never emit NaN features.
    """


class VFlowOptCompressor(TokenCompressor):
    """Training-free three-stage VFlowOpt for LLaVA, Qwen, and InternVL.

    The strategy may be a PruningStrategy, its dictionary representation, or the
    path of a calibration JSON, allowing it to pass through compressor_args. When
    it is omitted, max_tokens (or the baseline name budget_tokens) selects one of
    the supplied calibrated strategies. Both names denote the Eq. 5 layer-average
    budget, not the final token count.

    VFlowOpt is training-free: the calibrated parameters are compiled into this
    module, so nothing is downloaded by default. ``pretrained: true`` reads the
    released calibration JSON for the budget out of the ``llmoe_weights`` bundle
    instead -- the same numbers, but carrying the objective and search
    provenance the bundle records.
    """

    def __init__(self, strategy: PruningStrategy | Mapping[str, Any] | str | PathLike[str] | None = None, *,
                 max_tokens: int | None = None, budget_tokens: int | None = None,
                 retain_ratio: float | None = None,
                 pretrained: bool | str | PathLike[str] | None = None,
                 entropy_mode: str = "reference", enabled: bool = True, verbose: bool = True,
                 track_source: bool = False, strict_importance: bool = False,
                 recycle_fraction: float = 0.25, attention_chunk: int = 512,
                 backbone: str = "llava") -> None:
        self._configured_strategy = strategy is not None
        self._configured_max_tokens = max_tokens is not None
        self._configured_budget_tokens = budget_tokens is not None
        self._configured_pretrained = pretrained not in (None, False)
        self._configured_entropy_mode = str(entropy_mode)
        self._configured_enabled = bool(enabled)
        self._configured_strict_importance = bool(strict_importance)
        self._configured_recycle_fraction = float(recycle_fraction)
        self._configured_attention_chunk = int(attention_chunk)
        if retain_ratio is not None and not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
        if not 0.0 < float(recycle_fraction) <= 1.0:
            raise ValueError(
                f"recycle_fraction must be in (0, 1], got {recycle_fraction}"
            )
        if int(attention_chunk) < 1:
            raise ValueError("attention_chunk must be >= 1")
        # `retain_ratio` marks a ratio-budgeted backbone (Qwen or LLaVA-NeXT), so
        # a token count is meaningless with it. `strategy` is allowed: LLaVA-NeXT
        # accepts a profile calibrated for that backbone, and `validate_llava_next`
        # / the Qwen path decide whether one is usable.
        if retain_ratio is not None and any(
            value is not None for value in (max_tokens, budget_tokens)
        ):
            raise ValueError(
                "retain_ratio budgets a fraction of a variable visual span; "
                "max_tokens and budget_tokens configure the fixed-grid LLaVA path."
            )
        if retain_ratio is not None and pretrained not in (None, False):
            raise ValueError(
                "VFlowOpt's released calibration is fixed-grid LLaVA-specific "
                "and cannot be loaded on Qwen."
            )
        if retain_ratio is not None and strict_importance:
            raise ValueError(
                "strict_importance belongs to VFlowOpt's LLaVA calibration "
                "objective and is unavailable on the Qwen adaptation."
            )
        self._retain_ratio: float | None = (
            None if retain_ratio is None else float(retain_ratio)
        )
        if (
            max_tokens is not None
            and budget_tokens is not None
            and int(max_tokens) != int(budget_tokens)
        ):
            raise ValueError(
                f"max_tokens ({max_tokens}) and budget_tokens ({budget_tokens}) disagree")
        requested_budget = budget_tokens if budget_tokens is not None else max_tokens
        if self._retain_ratio is not None:
            super().__init__(1)
            # A ratio-budgeted backbone does not take a LLaVA-1.5 PruningStrategy,
            # so `strategy` is kept raw here rather than parsed as one. The NeXT
            # path resolves it in `_next_strategy`; Qwen ignores it.
            self.strategy: PruningStrategy | None = None
            self._raw_strategy = strategy
            self._budget_tokens = 0
        else:
            if strategy is None and (pretrained is not None and pretrained is not False):
                if requested_budget is None:
                    raise ValueError(
                        "max_tokens/budget_tokens is required to select a released strategy"
                    )
                # The compressor is built before the backbone is known, and the
                # strategies are depth-specific (Eq. 5 averages over the decoder),
                # so which release to take has to be said rather than inferred.
                strategy = resolve_weights(
                    "VFlowOpt", int(requested_budget), pretrained=pretrained,
                    backbone=backbone,
                )
            if strategy is None:
                if requested_budget is None:
                    raise ValueError("strategy or max_tokens/budget_tokens is required")
                strategy = calibrated_strategy(int(requested_budget))
            elif isinstance(strategy, (str, PathLike)):
                strategy = load_strategy_file(strategy)
            elif isinstance(strategy, Mapping):
                strategy = PruningStrategy.from_dict(dict(strategy))
            elif not isinstance(strategy, PruningStrategy):
                raise TypeError(
                    "strategy must be PruningStrategy or a mapping, got "
                    f"{type(strategy).__name__}"
                )
            super().__init__(int(round(strategy.n_visual * strategy.r1)))
            self.strategy = strategy
            self._budget_tokens = int(
                requested_budget
                if requested_budget is not None
                else round(strategy.layer_average)
            )
            if self._budget_tokens < 1:
                raise ValueError("max_tokens/budget_tokens must be >= 1")
        self._entropy_mode: str = entropy_mode
        self._enabled: bool = bool(enabled)
        self._verbose: bool = bool(verbose)
        self._track_source: bool = bool(track_source)
        # When strict, an empty "relatively important" set (Eq. 2) raises instead
        # of falling back to plain received attention. Calibration uses it to
        # reproduce the reference's NaN -> zero-similarity rejection.
        self._strict_importance: bool = bool(strict_importance)
        self._recycle_fraction: float = float(recycle_fraction)
        self._attention_chunk: int = int(attention_chunk)

        # Installation bookkeeping; all of it is undone by uninstall().
        self._installed_vlm: Any = None
        self._inner_model: Any = None
        self._language_model: Any = None
        self._vit_layer: torch.nn.Module | None = None
        self._hook_handles: list[RemovableHandle] = []
        self._original_get_image_features: Any = None
        self._original_prepare_inputs: Any = None
        self._image_token_id: int | None = None
        self._image_mean: tuple[float, ...] | None = None
        self._image_std: tuple[float, ...] | None = None
        self._vision_patch_size: int = 14
        self._internvl: Any | None = None
        # Qwen delegates immediately at install time and has no fixed-grid
        # strategy; these placeholders are replaced by the active backbone path.
        # `self.strategy` is the *parsed* fixed-grid strategy and is None on the
        # ratio-budgeted path, where the backbone adapter derives its own stages.
        # Reading the raw `strategy` argument here would break on a NeXT profile,
        # which is a mapping rather than a PruningStrategy.
        parsed = self.strategy
        self._active_stages: StageLayers = (
            parsed.stages if parsed is not None else StageLayers(1, 1, 1)
        )
        self._active_keeps: tuple[int, int, int] = (
            parsed.keeps if parsed is not None else (1, 1, 1)
        )
        self._active_source_tokens: int | None = None

        # Per-forward state.
        self._captured_hidden: torch.Tensor | None = None
        self._captured_pixels: torch.Tensor | None = None
        self._vit_importance: list[torch.Tensor] = []
        self._image_positions: torch.Tensor | None = None
        self._alive_scores: torch.Tensor | None = None
        self._position_ids: torch.Tensor | None = None
        self._position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cache_positions: torch.Tensor | None = None
        self._pruned_at: set[int] = set()
        self._forward_keeps: list[int] = []
        self._source_history: list[torch.Tensor] = []

        # Diagnostics accumulated over the whole run.
        self.stats: dict[str, float] = {
            "forwards": 0.0, "layer_average_sum": 0.0, "keep1_sum": 0.0,
            "keep2_sum": 0.0, "keep3_sum": 0.0, "merged_cells_sum": 0.0,
            "topped_up_sum": 0.0, "empty_context_set": 0.0, "k_size_sum": 0.0,
        }
        self._logged_install: bool = False
        self._diag_forwards: int = 3
        self._qwen: Any | None = None
        self._llava_next: Any | None = None

    # ------------------------------------------------------------------ basics

    @property
    @override
    def budget(self) -> int | float:
        # `max_tokens` follows the strategy's stage-1 count (`set_strategy`
        # re-derives it), so the reported budget comes from `_budget_tokens`, which
        # `set_budget` maintains across calibration.
        return self._budget_tokens if self._retain_ratio is None else self._retain_ratio

    def validate_llava_next(self) -> None:
        """Enforce VFlowOpt's single canonical NeXT control: retain_ratio."""

        from llmoe.compressor.llava_next import validate_retain_ratio
        from llmoe.compressor.vflowopt.vflowopt_llava_next import (
            layer_average_schedule,
        )

        ratio = validate_retain_ratio(self._retain_ratio, label="VFlowOpt")
        noncanonical = (
            self._configured_max_tokens
            or self._configured_budget_tokens
            or self._configured_pretrained
            or self._configured_entropy_mode != "reference"
            or not self._configured_enabled
            or self._configured_strict_importance
            or self._configured_recycle_fraction != 0.25
            or self._configured_attention_chunk != 512
        )
        if noncanonical:
            raise ValueError(
                "VFlowOpt on LLaVA-NeXT fixes the entropy weighting and the two "
                "decoder boundaries internally; retain_ratio is the compression "
                "control, and `strategy` may carry a profile calibrated for this "
                "backbone."
            )
        layer_average_schedule(576, ratio, 32, self._next_strategy())

    def _next_strategy(self):
        """The searched NeXT profile, or ``None`` for upstream's canonical one.

        A LLaVA-1.5 `PruningStrategy` is not accepted here: its ratios are solved
        against a fixed 576-token grid, while NeXT budgets a fraction of a
        variable AnyRes span, so the two are not interchangeable.
        """
        from llmoe.compressor.vflowopt.vflowopt_llava_next import NextStrategy

        raw = getattr(self, "_raw_strategy", None)
        if not raw:
            return None
        if isinstance(raw, NextStrategy):
            return raw
        if isinstance(raw, (str, PathLike)):
            payload = loads(Path(raw).read_text(encoding="utf-8"))
        elif isinstance(raw, Mapping):
            payload = raw
        else:
            payload = getattr(raw, "to_dict", dict)()
        inner = payload.get("strategy", payload)
        missing = {"r1", "r2", "r3"} - set(inner)
        if missing:
            raise ValueError(
                f"A LLaVA-NeXT VFlowOpt strategy needs {sorted(missing)}; got "
                f"{sorted(inner)}. Calibrate one with "
                "`python -m llmoe.compressor.vflowopt.calibrate --backbone llava-next-8b`."
            )
        optional = {key: float(inner[key]) for key in ("t", "a") if key in inner}
        return NextStrategy(r1=float(inner["r1"]), r2=float(inner["r2"]),
                            r3=float(inner["r3"]), **optional)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Toggle every stage at once. Disabled is bit-exact full-token LLaVA,
        which is what the calibration objective needs for its reference forward."""
        self._enabled = bool(enabled)
        if self._qwen is not None:
            self._qwen.set_enabled(self._enabled)
        self._reset_forward_state()

    def set_strategy(self, strategy: PruningStrategy) -> None:
        """Swap in a new candidate strategy between calibration iterations."""
        if self._retain_ratio is not None:
            raise RuntimeError("A Qwen ratio run has no LLaVA calibration strategy")
        self.strategy = strategy
        self._max_tokens = int(round(strategy.n_visual * strategy.r1))
        if self._internvl is None:
            self._active_stages = strategy.stages
            self._active_keeps = strategy.keeps
        self._reset_forward_state()

    def set_budget(self, budget_tokens: int) -> None:
        """Relabel the compressor (output filenames, logs) for another budget."""
        self._budget_tokens = int(budget_tokens)

    def reset_stats(self) -> None:
        for key in self.stats:
            self.stats[key] = 0.0

    @property
    def vision_stage_tokens(self) -> int:
        """Deterministic per-image token count out of the vision stage."""
        if self.strategy is None:
            raise RuntimeError("Qwen resolves its vision-stage count per image")
        return int(round(self.strategy.n_visual * self.strategy.r1))

    def mean_layer_average(self) -> float:
        forwards = self.stats["forwards"]
        return self.stats["layer_average_sum"] / forwards if forwards else float("nan")

    def summary(self) -> dict[str, Any]:
        if self._retain_ratio is not None:
            return {
                "retain_ratio": self._retain_ratio,
                "adaptation": "qwen-final-vit-received-attention",
                "recycle_fraction": self._recycle_fraction,
            }
        forwards = max(1.0, self.stats["forwards"])
        return {
            "budget_tokens": self._budget_tokens,
            "forwards": int(self.stats["forwards"]),
            "mean_layer_average_tokens": self.mean_layer_average(),
            "mean_keep_stage1": self.stats["keep1_sum"] / forwards,
            "mean_keep_stage2": self.stats["keep2_sum"] / forwards,
            "mean_keep_stage3": self.stats["keep3_sum"] / forwards,
            "mean_merged_cells": self.stats["merged_cells_sum"] / forwards,
            "mean_context_set_size": self.stats["k_size_sum"] / forwards,
            "empty_context_set_images": int(self.stats["empty_context_set"]),
            "empty_cell_top_ups": int(self.stats["topped_up_sum"]),
            "strategy": self.strategy.to_dict(),
        }

    # -------------------------------------------------------------- installing

    @override
    def install(self, vlm: Any) -> None:
        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return
            raise RuntimeError("VFlowOptCompressor is already installed on another backbone")

        model = vlm.model
        config = model.config
        inner = model.model
        from llmoe.compressor.llava_next import is_llava_next

        if is_llava_next(vlm):
            self.validate_llava_next()
            from llmoe.compressor.vflowopt.vflowopt_llava_next import (
                LlavaNextVFlowOptAdapter,
            )

            self._installed_vlm = vlm
            self._llava_next = LlavaNextVFlowOptAdapter(
                self,
                vlm,
                retain_ratio=float(self._retain_ratio),
                track_source=self._track_source,
                verbose=self._verbose,
                strategy=self._next_strategy(),
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
                    "VFlowOpt on Qwen2.5-VL requires retain_ratio: NaViT "
                    "emits a different visual-token count for each image."
                )
            from llmoe.compressor.vflowopt.vflowopt_qwen import QwenVFlowOptAdapter
            from llmoe.utils.qwen import resolve_image_token_id

            adapter = QwenVFlowOptAdapter(
                retain_ratio=self._retain_ratio,
                recycle_fraction=self._recycle_fraction,
                attention_chunk=self._attention_chunk,
                track_source=self._track_source,
                verbose=self._verbose,
                enabled=self._enabled,
            )
            adapter.install(
                vlm,
                image_token_id=resolve_image_token_id(
                    config, vlm.processor, label="VFlowOpt"
                ),
            )
            self._installed_vlm = vlm
            self._qwen = adapter
            return
        if self._retain_ratio is not None:
            raise ValueError(
                "retain_ratio is VFlowOpt's Qwen2.5-VL parameterisation; "
                "LLaVA-1.5 and InternVL use max_tokens/budget_tokens or a strategy."
            )
        from llmoe.compressor.vflowopt.vflowopt_internvl import (
            InternVLVFlowOptAdapter,
        )

        internvl = InternVLVFlowOptAdapter.supports(vlm)
        if not internvl and getattr(config, "model_type", None) != "llava":
            raise TypeError(
                "VFlowOpt supports the Hugging Face LLaVA-1.5, Qwen2.5-VL, "
                "and InternVL backbones."
            )
        language_model = inner.language_model
        layers = getattr(language_model, "layers", None)
        if not layers:
            raise TypeError("VFlowOpt needs a decoder stack at model.model.language_model.layers")

        token_id = getattr(config, "image_token_id", None)
        if token_id is None:
            token_id = getattr(config, "image_token_index", None)
        if token_id is None:
            raise ValueError("VFlowOpt needs config.image_token_id / image_token_index")

        self._installed_vlm = vlm
        self._inner_model = inner
        self._language_model = language_model
        self._image_token_id = int(token_id)
        if internvl:
            self._internvl = InternVLVFlowOptAdapter(
                self, vlm, decoder_layers=len(layers)
            )
        self._active_stages = (
            self._internvl.stages if self._internvl is not None else self.strategy.stages
        )
        self._active_keeps = self.strategy.keeps

        image_processor = getattr(vlm.processor, "image_processor", None)
        mean = getattr(image_processor, "image_mean", None)
        std = getattr(image_processor, "image_std", None)
        self._image_mean = tuple(float(v) for v in mean) if mean else None
        self._image_std = tuple(float(v) for v in std) if std else None
        vision_config = getattr(getattr(inner, "vision_tower", None), "config", None)
        vision_config = getattr(vision_config, "vision_config", None) or vision_config
        patch_size = getattr(vision_config, "patch_size", 14)
        if isinstance(patch_size, (list, tuple)):
            if not patch_size or any(int(value) != int(patch_size[0]) for value in patch_size):
                raise ValueError(
                    f"VFlowOpt needs square vision patches, got patch_size={patch_size}"
                )
            patch_size = patch_size[0]
        self._vision_patch_size = int(patch_size)
        if self._vision_patch_size < 1:
            raise ValueError(
                f"VFlowOpt needs a positive vision patch size, got {self._vision_patch_size}"
            )

        self._install_vision_stage(vlm, inner, config)
        self._install_placeholder_trim(vlm)
        self._install_llm_stages(inner)

        if self._verbose and not self._logged_install:
            self._logged_install = True
            if self._internvl is not None:
                self._internvl.log_install()
            else:
                n1, n2, n3 = self.strategy.keeps
                k1, k2 = self.strategy.prune_layers
                print(
                    f"[VFlowOpt] install budget={self._budget_tokens} "
                    f"R=({self.strategy.r1:.4f}, {self.strategy.r2:.4f}, {self.strategy.r3:.4f}) "
                    f"t={self.strategy.t:.3f} alpha={self.strategy.alpha:+.4f} "
                    f"a={self.strategy.a:.3f} cells={self.strategy.n_cells} "
                    f"keeps={n1}/{n2}/{n3} @layers(vit,{k1},{k2}) "
                    f"layer-avg={self.strategy.layer_average:.3f} "
                    f"entropy={self._entropy_mode} "
                    f"vit_layer={'ok' if self._vit_layer is not None else 'MISSING'}",
                    flush=True,
                )

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
        self._language_model = None
        self._vit_layer = None
        self._image_token_id = None
        self._internvl = None
        self._llava_next = None
        self._active_stages = self.strategy.stages
        self._active_keeps = self.strategy.keeps
        self._active_source_tokens = None
        self._vision_patch_size = 14
        self._reset_forward_state()

    def _reset_forward_state(self) -> None:
        self._captured_hidden = None
        self._captured_pixels = None
        self._vit_importance = []
        self._image_positions = None
        self._alive_scores = None
        self._position_ids = None
        self._position_embeddings = None
        self._cache_positions = None
        self._pruned_at = set()
        self._forward_keeps = []
        self._source_history = []

    # ------------------------------------------------------- vision stage wiring

    def _install_vision_stage(self, vlm: Any, inner: Any, config: Any) -> None:
        vision_tower = getattr(inner, "vision_tower", None)
        self._vit_layer = self._locate_attention_layer(vision_tower, config)
        if self._vit_layer is None:
            raise RuntimeError("Could not locate the vision encoder layer that feeds the projector")
        self._hook_handles.append(
            self._vit_layer.register_forward_pre_hook(self._capture_vit_hidden, with_kwargs=True)
        )

        original = inner.get_image_features
        self._original_get_image_features = original

        def wrapped(*args, **kwargs):
            self._captured_hidden = None
            pixel_values = kwargs.get("pixel_values")
            if pixel_values is None and args:
                pixel_values = args[0]
            self._captured_pixels = pixel_values
            image_outputs = original(*args, **kwargs)
            if not self._enabled:
                return image_outputs
            if self._internvl is not None:
                return self._internvl.compress_outputs(image_outputs)
            stacked = torch.stack(list(image_outputs.pooler_output), dim=0)
            result = self.compress(stacked)
            image_outputs.pooler_output = list(result.tokens.unbind(0))
            return image_outputs

        inner.get_image_features = wrapped

    @staticmethod
    def _locate_attention_layer(vision_tower: Any, config: Any) -> torch.nn.Module | None:
        """The CLIP layer whose *output* LLaVA feeds to the projector.

        ``hidden_states`` has ``n_layers + 1`` entries (embeddings first), so
        ``hidden_states[feature_layer]`` is the output of encoder layer
        ``feature_layer_index - 1``. At LLaVA-1.5's ``vision_feature_layer=-2``
        that is layer 22 of 24.
        """
        if vision_tower is None:
            return None
        vision_model = getattr(vision_tower, "vision_model", vision_tower)
        encoder = getattr(vision_model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None:
            layers = getattr(encoder, "layer", None)
        if not layers:
            return None
        feature_layer = getattr(config, "vision_feature_layer", None)
        if feature_layer is None:
            feature_layer = getattr(config, "vision_feature_select_layer", -2)
        if isinstance(feature_layer, (list, tuple)):
            feature_layer = feature_layer[-1]
        index = int(feature_layer) % (len(layers) + 1) - 1
        if not 0 <= index < len(layers):
            return None
        return layers[index]

    def _capture_vit_hidden(self, _module: Any, args: tuple, kwargs: dict) -> None:
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if isinstance(hidden, torch.Tensor):
            self._captured_hidden = hidden.detach()
        return None

    def _context_for_forward(self, n_images: int) -> torch.Tensor:
        hidden = self._captured_hidden
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            raise RuntimeError("VFlowOpt did not capture the ViT hidden states for this forward")
        with torch.no_grad():
            attention = head_averaged_self_attention(self._vit_layer, hidden)
        if self._strict_importance:
            context, k_sizes = context_relevance(attention, self.strategy.t)
        else:
            context, k_sizes = context_relevance_safe(attention, self.strategy.t)
        empty = int((k_sizes <= 0).sum())
        self.stats["empty_context_set"] += float(empty)
        self.stats["k_size_sum"] += float(k_sizes.sum()) / max(1, n_images)
        if empty and self._strict_importance:
            raise EmptyContextSet(
                f"threshold t={self.strategy.t} selected no context tokens "
                f"for {empty}/{attention.shape[0]} vision inputs"
            )
        return context

    def compute_importance(self, n_images: int) -> torch.Tensor:
        """Eq. 1-4 for a regular LLaVA forward."""
        context = self._context_for_forward(n_images)
        patches = context[:, 1:] if context.shape[-1] == self.strategy.n_visual + 1 else context
        if patches.shape[-1] != self.strategy.n_visual:
            raise RuntimeError(
                f"ViT produced {patches.shape[-1]} patch scores, expected {self.strategy.n_visual}"
            )

        pixels = self._captured_pixels
        if not isinstance(pixels, torch.Tensor):
            raise RuntimeError("VFlowOpt did not capture pixel_values for this forward")
        entropy = patch_entropy(
            pixels,
            patch_size=self._vision_patch_size,
            mode=self._entropy_mode,
            image_mean=self._image_mean,
            image_std=self._image_std,
        )
        if entropy.shape != patches.shape:
            raise RuntimeError(
                f"entropy shape {tuple(entropy.shape)} != attention shape {tuple(patches.shape)}"
            )
        return importance_score(patches, entropy, self.strategy.alpha)

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, **kwargs) -> CompressedFeatures:
        """Vision stage: prune to ``round(R1 * N)`` tokens with recycling."""
        if self._retain_ratio is not None:
            raise RuntimeError(
                "VFlowOptCompressor.compress() is the fixed-grid LLaVA path. "
                "Qwen ratio pruning runs after concatenation through "
                "QwenVFlowOptAdapter."
            )
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.dim() != 3:
            raise ValueError(f"expected [B, N, d] or [N, d] features, got {tuple(features.shape)}")
        batch, n_tokens, _ = features.shape
        if n_tokens != self.strategy.n_visual:
            raise ValueError(f"expected {self.strategy.n_visual} visual tokens, got {n_tokens}")

        importance = kwargs.get("importance")
        if importance is None:
            importance = self.compute_importance(batch)
        keep_target = self.vision_stage_tokens

        self._vit_importance = []
        self._source_history = []
        compressed: list[torch.Tensor] = []
        for index in range(batch):
            result = vision_stage(
                features[index], importance[index].to(features.device),
                keep_target=keep_target, a=self.strategy.a,
                side=self.strategy.grid_side, n_cells=self.strategy.n_cells,
            )
            compressed.append(result.features)
            self._vit_importance.append(result.importance)
            self.stats["merged_cells_sum"] += float(result.merged_cells) / batch
            self.stats["topped_up_sum"] += float(result.topped_up)
            if index == 0 and self._track_source:
                self._source_history = self._build_source_history(result, n_tokens)
        tokens = torch.stack(compressed, dim=0)
        return CompressedFeatures(tokens=tokens.squeeze(0) if squeeze else tokens)

    def _build_source_history(self, result: Any, n_tokens: int) -> list[torch.Tensor]:
        identity = torch.eye(n_tokens, device=result.features.device, dtype=torch.float32)
        source = result.source
        expected = (int(result.keep_indices.numel()), int(n_tokens))
        if tuple(source.shape) != expected:
            raise RuntimeError(
                f"VFlowOpt source map has shape {tuple(source.shape)}, expected {expected}"
            )
        return [identity, source]

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        if self._llava_next is not None:
            return list(self._source_history)
        if self._qwen is not None:
            return self._qwen.get_source_history()
        return self._source_history

    # --------------------------------------------------------- placeholder trim

    def _install_placeholder_trim(self, vlm: Any) -> None:
        """Trim each source-image placeholder run to the global stage-1 count."""
        tokenizer = getattr(vlm.processor, "tokenizer", None)
        tokenizer_pad = getattr(tokenizer, "pad_token_id", None)
        pad_id = (
            int(tokenizer_pad)
            if tokenizer_pad is not None
            else int(self._image_token_id)
        )
        original = vlm.prepare_inputs
        self._original_prepare_inputs = original

        def wrapped(*args, **kwargs):
            inputs = original(*args, **kwargs)
            return self._trim_placeholders(vlm, inputs, pad_id) if self._enabled else inputs

        vlm.prepare_inputs = wrapped

    def _trim_placeholders(self, vlm: Any, inputs: Any, pad_id: int) -> Any:
        if self._internvl is not None:
            return self._internvl.trim_inputs(inputs, pad_token_id=pad_id)
        keep = self.vision_stage_tokens

        return trim_image_placeholders(
            inputs,
            token_id=self._image_token_id,
            keep_for_run=lambda _run_length: keep,
            pad_token_id=pad_id,
        )

    # ------------------------------------------------------------- LLM stages

    def _install_llm_stages(self, inner: Any) -> None:
        layers = self._language_model.layers
        n_layers = len(layers)
        # "Evenly divide the LMM into three stages" (Sec. 3.2) is a rule about the
        # decoder, not a fixed 32-layer split, so the stages follow whatever depth
        # is loaded: 10/10/12 on Vicuna-7B, 13/13/14 on the 13B. The searched
        # ratios carry over unchanged -- they are per-stage retention fractions --
        # but stage 3 is re-solved against the new widths, because the budget
        # constraint (Eq. 5) weights each stage by how many layers it spans.
        if self._internvl is not None:
            self._active_stages = self._internvl.stages
        else:
            self._active_stages = StageLayers.even(n_layers)
            if self._active_stages != self.strategy.stages:
                self._active_keeps = solve_keeps_integer(
                    self.strategy.r1, self.strategy.r2, self._budget_tokens,
                    self.strategy.n_visual, self._active_stages,
                )
                print(
                    f"[VFlowOpt] resolved the strategy for a {n_layers}-layer decoder: "
                    f"stages {self._active_stages.l1}/{self._active_stages.l2}/"
                    f"{self._active_stages.l3}, prune at layers "
                    f"{list(self._active_stages.prune_layers)}, keeps "
                    f"{list(self._active_keeps)}, layer-average "
                    f"{layer_average_tokens(self._active_keeps, self._active_stages):.2f}",
                    flush=True,
                )
        self._hook_handles.append(
            inner.register_forward_pre_hook(self._capture_sequence, with_kwargs=True)
        )
        first = self._active_stages.prune_layers[0]
        for index in range(first, n_layers):
            self._hook_handles.append(
                layers[index].register_forward_pre_hook(
                    self._make_layer_hook(index), with_kwargs=True
                )
            )

    def _capture_sequence(self, _module: Any, args: tuple, kwargs: dict) -> None:
        """Pre-hook on ``LlavaModel``: it is the last place ``input_ids`` is visible
        (``LlamaModel`` only ever receives ``inputs_embeds``)."""
        if not self._enabled:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2 or input_ids.shape[1] <= 1:
            return None  # a decoding step: keep the state built during prefill
        if input_ids.shape[0] != 1:
            raise RuntimeError(
                f"VFlowOpt prunes the sequence physically and needs batch_size == 1, "
                f"got {input_ids.shape[0]}")
        self._pruned_at = set()
        self._position_ids = None
        self._position_embeddings = None
        self._cache_positions = None
        self._image_positions = (input_ids[0] == self._image_token_id).nonzero(as_tuple=False).squeeze(-1)
        # The vision stage has not run yet (get_image_features is called from
        # inside LlavaModel.forward, i.e. after this pre-hook), so the importance
        # scores are bound lazily by the first decoder-layer hook.
        self._alive_scores = None
        self._forward_keeps = [int(self._image_positions.numel())]
        self.stats["forwards"] += 1.0
        return None

    def _make_layer_hook(self, layer_idx: int):
        def hook(module: Any, args: tuple, kwargs: dict):
            return self._layer_hook(module, args, kwargs, layer_idx)
        return hook

    def _layer_hook(self, _module: Any, args: tuple, kwargs: dict, layer_idx: int):
        if not self._enabled or self._image_positions is None:
            return None
        hidden = kwargs.get("hidden_states")
        from_args = hidden is None
        if from_args:
            if not args:
                return None
            hidden = args[0]
        if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
            return None

        if hidden.shape[1] == 1:
            if not self._pruned_at:
                return None  # nothing was pruned in the prefill: leave the layer alone
            # Decoding. Layers past the first prune point hold a shorter cache
            # than the mask built by the model, so let SDPA attend over whatever
            # the layer actually cached (every cached key is a real token).
            kwargs["attention_mask"] = None
            cache = kwargs.get("past_key_values")
            cached = self._cached_length(cache, layer_idx)
            if cached is not None and "cache_position" in kwargs:
                kwargs["cache_position"] = torch.tensor([cached], device=hidden.device,
                                                        dtype=torch.long)
            return args, kwargs

        if self._image_positions.numel() == 0:
            return None  # text-only prompt: nothing to prune
        if self._alive_scores is None:
            scores = self._vit_importance[0] if self._vit_importance else None
            if scores is None:
                return None
            if scores.numel() != self._image_positions.numel():
                raise RuntimeError(
                    f"{self._image_positions.numel()} image placeholders but {scores.numel()} "
                    f"vision-stage importance scores")
            self._alive_scores = scores

        if self._position_ids is None:
            position_ids = kwargs.get("position_ids")
            if not isinstance(position_ids, torch.Tensor):
                position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
            self._position_ids = position_ids
            self._position_embeddings = kwargs.get("position_embeddings")
            self._cache_positions = kwargs.get("cache_position")

        prune_layers = self._active_stages.prune_layers
        if layer_idx in prune_layers and layer_idx not in self._pruned_at:
            self._pruned_at.add(layer_idx)
            ratio = self.strategy.r2 if layer_idx == prune_layers[0] else self.strategy.r3
            keep_target = None
            if self._internvl is not None:
                keep_target = self._active_keeps[1 if layer_idx == prune_layers[0] else 2]
            hidden = self._prune_sequence(
                hidden, ratio, layer_idx, keep_target=keep_target
            )

        kwargs["attention_mask"] = None  # batch 1, no padding: SDPA applies is_causal
        kwargs["position_ids"] = self._position_ids
        if self._position_embeddings is not None:
            kwargs["position_embeddings"] = self._position_embeddings
        if self._cache_positions is not None and "cache_position" in kwargs:
            kwargs["cache_position"] = self._cache_positions
        if from_args:
            args = (hidden,) + tuple(args[1:])
        else:
            kwargs["hidden_states"] = hidden
        return args, kwargs

    def _prune_sequence(self, hidden: torch.Tensor, ratio: float, layer_idx: int, *,
                        keep_target: int | None = None) -> torch.Tensor:
        """One progressive-pruning step: drop the least important visual tokens
        from the sequence, the position ids and (implicitly) the KV cache."""
        positions = self._image_positions
        scores = self._alive_scores
        if positions is None or scores is None or positions.numel() == 0:
            return hidden
        n_alive = int(positions.numel())
        if keep_target is None:
            keep_target = n_alive - int(round(n_alive * (1.0 - ratio)))
        keep_target = max(0, min(n_alive, int(keep_target)))
        keep_local = llm_stage_keep(scores, keep_target)

        keep_sequence = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        keep_sequence[positions] = False
        keep_sequence[positions[keep_local]] = True

        hidden = hidden[:, keep_sequence, :]
        self._position_ids = self._position_ids[:, keep_sequence]
        self._position_embeddings = self._language_model.rotary_emb(hidden, position_ids=self._position_ids)
        self._cache_positions = torch.arange(hidden.shape[1], device=hidden.device, dtype=torch.long)

        remap = torch.cumsum(keep_sequence.long(), dim=0) - 1
        self._image_positions = remap[positions[keep_local]]
        self._alive_scores = scores[keep_local]

        self._record_stage(keep_target)
        return hidden

    def _record_stage(self, kept: int) -> None:
        """Accumulate this forward's realised per-stage keeps and, once stage 3
        has fired, the Eq. 5 layer average that the budget label refers to."""
        self._forward_keeps.append(int(kept))
        if len(self._forward_keeps) < 3:
            return
        n1, n2, n3 = self._forward_keeps[:3]
        stages = self._active_stages
        layer_average = (n1 * stages.l1 + n2 * stages.l2 + n3 * stages.l3) / stages.total
        self.stats["keep1_sum"] += float(n1)
        self.stats["keep2_sum"] += float(n2)
        self.stats["keep3_sum"] += float(n3)
        self.stats["layer_average_sum"] += layer_average
        if self._verbose and self.stats["forwards"] <= self._diag_forwards:
            print(f"[VFlowOpt] forward #{int(self.stats['forwards'])}: keeps={n1}/{n2}/{n3} "
                  f"layer-avg={layer_average:.2f} (budget {self._budget_tokens})", flush=True)

    @staticmethod
    def _cached_length(cache: Any, layer_idx: int) -> int | None:
        if cache is None:
            return None
        try:
            return int(cache.layers[layer_idx].get_seq_length())
        except Exception:  # noqa: BLE001 - unknown cache layout, leave kwargs alone
            return None
