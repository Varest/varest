"""PyramidDrop: staged visual-token pruning across the language decoder.

Paper: Xing et al., "PyramidDrop: Accelerating Your Large Vision-Language Models
via Pyramid Visual Redundancy Reduction" (CVPR 2025, arXiv:2410.17247).
Reference code: https://github.com/Cooperx521/PyramidDrop, ported here from the
``llmoe_baseline_pyramiddrop`` sidecar that ran it against this harness.

PyramidDrop is training-free and does nothing to the vision tower: all 576
projected patches enter the decoder. What it changes is how long they stay there.
The decoder is cut into stages by ``layer_list``; at the end of each stage the
surviving visual tokens are ranked and only a scheduled number continue:

1. the stage ends after 1-indexed layer ``L``, so 0-indexed layers ``0..L-1`` run
   on the incoming count and layers ``L..`` run on the reduced one;
2. the *next* layer's ``input_layernorm`` + ``q_proj``/``k_proj`` (plus RoPE) give
   the attention of the **last prompt token** to every position -- the paper's
   lightweight rank, which costs one query row instead of a full attention pass;
3. the top-scoring visual tokens are kept in reading order, every non-visual token
   is kept, and the sequence is **physically shrunk**.

Pruning happens during prefill only, so each stage's KV cache holds exactly the
tokens that stage ran on and both FLOPs and cache shrink with depth -- that is the
entire speedup.

Budget semantics
----------------
``max_tokens`` is the **layer-averaged** visual-token count
(``sum(tokens held by each of the 32 decoder layers) / 32``), which is what the
paper reports and the same convention ``sparse_vlm.py``, ``must_drop.py``,
``duet.py`` and ``hidrop.py`` use here -- not a final retained count. The paper's
own LLaVA-1.5 setting (four equal 8-layer stages, ``lambda = 0.5``, i.e.
``576 -> 288 -> 144 -> 72`` after layers 8/16/24) averages 270, so none of this
harness's four budgets is the released configuration:

====== ================= =============================== ========
budget stage boundaries  visual tokens entering a stage   average
====== ================= =============================== ========
224    8, 16, 24         576, 213, 78, 29                 224.0
128    4, 8, 12          576, 217, 81, 30                 128.0
64     2, 8, 16, 24      576, 112, 22, 5, 1                64.0
32     1, 8, 16, 24      576, 48, 10, 3, 1                 32.0
====== ================= =============================== ========

``PYRAMIDDROP_PROFILES`` records where each row comes from. Adjusting ``lambda``
alone cannot reach 128 or below on the paper's equal 8-layer stages: the first
stage is never pruned, so it alone floors the average at ``8 * 576 / 32 = 144``.
224 and 128 therefore compress the equal stages (the widest division that still
reaches the label), and 64 and 32 use the uneven division the authors give for
their own low-budget comparison points -- ``(2, 8, 16, 24)``, one shallow cut
followed by the original 8/16/24 trajectory. 32 cannot use it either, because two
unpruned 576-token layers already contribute ``2 * 576 / 32 = 36``; its shallow
boundary moves to layer 1, which is the only change made to that trajectory.
``minimum_average_retained_tokens`` computes this floor and ``install()`` refuses
a schedule that cannot reach its own label.

Only these four budgets have a profile. Pass ``layer_list`` and
``stage_token_counts`` together to run another schedule -- including the paper's
own, ``layer_list=(8, 16, 24), stage_token_counts=(576, 288, 144, 72)``.

Qwen2.5-VL and LLaVA-NeXT are parameterised by ``retain_ratio`` because their
visual spans have no fixed token count. The ratio is the same decoder-layer
average, expressed as a fraction of each image's own visual span. A four-stage
geometric schedule is solved for the loaded decoder depth, and every boundary
keeps ``floor(stage_ratio * N)`` tokens (with a one-token minimum). Qwen
preserves M-RoPE coordinates; NeXT applies the schedule to one native AnyRes
packed span with Llama 3 RoPE/cache mechanics. These are in-repo adaptations.

Faithfulness notes / harness adaptations
----------------------------------------
The reference replaces ``LlamaModel.forward`` with a re-implemented decoder loop.
This harness drives a stock ``LlavaForConditionalGeneration``, so -- exactly like
``fastv.py``, ``duet.py`` and ``must_drop.py`` -- every stage is expressed through
forward pre-hooks instead:

* **Ranking.** The reference reads the rank layer's own ``input_layernorm`` +
  ``q_proj``/``k_proj``, applies RoPE, and softmaxes one query row. A pre-hook on
  the boundary layer sees precisely that layer's input, so the same computation is
  reproduced there, restricted to the single query row; the quadratic ``[L, L]``
  map is never materialised and the backend stays on SDPA.

* **The query is the last prompt token.** The reference takes the last *unpadded*
  column of each row. That column is held as an offset past the end of the visual
  span, which is the only part of the sequence that shrinks, so it stays valid at
  every later boundary.

* **Positions are compacted, not preserved.** After each cut the reference
  rebuilds ``position_ids`` from the compacted padding mask, so every stage sees a
  gapless numbering of its own shorter sequence, and -- because layers in
  different stages then hold caches of different lengths -- a decoding query sits
  at ``number of valid keys in that stage's cache``. ``sparse_vlm.py`` compacts
  for the same reason; ``fastv.py`` and ``duet.py`` keep absolute ids instead,
  because *their* references do. Each stage is written and read at one consistent
  numbering either way; the two schemes simply place survivors at different RoPE
  phases.

* **Prefill only.** Cuts run on the prefill pass. Decode steps inherit each
  stage's narrower key space through its own mask width, ``cache_position`` and
  query position; nothing is dropped once generation starts.

* **Batching.** The reference requires one contiguous visual span per row and the
  same visual count in every row. Both are enforced here, and a ragged batch warns
  once and skips pruning rather than corrupting the gather -- the treatment
  ``fastv.py`` and ``duet.py`` give it, in place of the reference's hard error.
  Beam search expands the batch after prefill, which per-stage pruned caches
  cannot follow, so it raises instead of silently producing garbage.

* **Placeholders.** Nothing to do: PyramidDrop never changes the number of visual
  features the projector emits, so LLaVA's ``<image>`` expansion is left alone and
  ``install()`` does not wrap ``get_image_features`` at all. ``compress()`` is the
  tensor-level entry point for one boundary's selection rule, and passes tokens
  through untouched without an ``attention`` argument -- which is also what the
  real vision path does.

* **Source history.** ``get_source_history()`` returns a leading identity plus one
  ``[T_cur, T_orig]`` matrix per cut, which is what ``llmoe.viz`` renders. Rows are
  one-hot: PyramidDrop prunes, it never merges.
"""

from typing import Any, Sequence, override

import torch
from torch.utils.hooks import RemovableHandle

from llmoe.backbone.prototype import VLM
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.utils.depth import REFERENCE_DEPTH, LayerSchedule

try:  # RoPE helper for the boundary-attention recompute.
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as _APPLY_ROPE
except Exception:  # pragma: no cover - degrade to no-RoPE if the path moved
    _APPLY_ROPE = None


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
#
# A PyramidDrop schedule is two lists: `layer_list`, the 1-indexed layers a stage
# ends after (equivalently, the 0-indexed layer that first runs reduced), and
# `stage_token_counts`, the visual tokens entering each stage. The first count is
# the full visual sequence, so `stage_token_counts` is one longer than
# `layer_list`.
#
# The paper's released LLaVA-1.5 setting -- four equal 8-layer stages with
# lambda = 0.5 -- is DEFAULT_LAYER_LIST / DEFAULT_STAGE_TOKEN_COUNTS below, and
# averages 270 tokens per layer. The harness's four budgets need their own
# schedules; see the module docstring for why lambda alone cannot produce them.

DEFAULT_VISUAL_TOKENS: int = 576
DEFAULT_NUM_DECODER_LAYERS: int = REFERENCE_DEPTH
DEFAULT_MAX_TOKENS: int = 128
DEFAULT_LAYER_LIST: tuple[int, ...] = (8, 16, 24)
DEFAULT_STAGE_TOKEN_COUNTS: tuple[int, ...] = (576, 288, 144, 72)


PYRAMIDDROP_PROFILES: dict[int, dict[str, object]] = {
    224: {
        "layer_list": (8, 16, 24),
        "stage_token_counts": (576, 213, 78, 29),
        "average_retained_tokens": 224.0,
        "source": "reproduction: the paper's own four equal 8-layer stages, with the three drops "
                  "retuned from lambda=0.5 to the near-uniform 0.37 that averages 224",
    },
    128: {
        "layer_list": (4, 8, 12),
        "stage_token_counts": (576, 217, 81, 30),
        "average_retained_tokens": 128.0,
        "source": "reproduction: equal stages compressed to width 4 -- the widest equal division "
                  "that reaches 128, since width 8 floors at 144 -- with three drops at 0.37",
    },
    64: {
        "layer_list": (2, 8, 16, 24),
        # Balanced rather than front-loaded. (576, 112, 22, 5, 1) also averages
        # exactly 64 but leaves one visual token for the last eight layers;
        # moving budget into the 8-15 stage improves all sixteen benchmarks
        # (MME 967 -> 1267, POPE f1 45.8 -> 70.2), monotonically in how much
        # moves, so it follows from the shape not one lucky vector.
        "stage_token_counts": (576, 64, 58, 5, 1),
        "average_retained_tokens": 64.0,
        "source": "the authors' uneven stage division for their own low-budget comparison points, "
                  "with the four drops solved for an exact 64 average",
    },
    32: {
        # The authors' (2, 8, 16, 24) holds 576 tokens through two full layers,
        # which alone average 36, so 32 is unreachable on it however hard the
        # later stages are cut. Moving the shallow boundary to layer 1 drops the
        # floor to 18.97 and is the only departure from their trajectory.
        "layer_list": (1, 8, 16, 24),
        # Balanced like budget 64, and for the same reason: the front-loaded
        # solve collapses to one token by layer 24. At an identical 32.0
        # average this takes MME 704 -> 928 and POPE f1 10.8 -> 32.2.
        "stage_token_counts": (576, 32, 22, 5, 1),
        "average_retained_tokens": 32.0,
        "source": "the authors' uneven stage division with the shallow boundary moved to layer 1, "
                  "the only division that reaches 32 (theirs floors at 36.94)",
    },
}

SUPPORTED_BUDGETS: tuple[int, ...] = tuple(sorted(PYRAMIDDROP_PROFILES, reverse=True))


def layer_schedule(layer_list: Sequence[int], stage_token_counts: Sequence[int],
                   num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS) -> LayerSchedule:
    """This schedule as a :class:`~llmoe.utils.depth.LayerSchedule`.

    PyramidDrop counts the full visual sequence as stage 0, so ``stage_token_counts``
    is one longer than ``layer_list``: its head is the span entering the decoder and
    its tail is one keep per boundary, which is exactly the shared form. Every
    budget question below -- the average, the floor, and the re-solve for another
    decoder depth -- is answered from here rather than recomputed.
    """
    counts = tuple(int(count) for count in stage_token_counts)
    boundaries = tuple(int(layer) for layer in layer_list)
    if len(counts) != len(boundaries) + 1:
        raise ValueError("stage_token_counts must hold one count per stage, i.e. one more than "
                         f"layer_list; got {len(counts)} counts for {len(boundaries) + 1} stages")
    return LayerSchedule(boundaries, counts[1:], int(num_decoder_layers), counts[0])


def stage_widths(layer_list: Sequence[int],
                 num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS) -> tuple[int, ...]:
    """Decoder layers each stage spans, one entry per stage."""
    boundaries = tuple(int(layer) for layer in layer_list)
    leading, per_stage = LayerSchedule(
        boundaries, (0,) * len(boundaries), int(num_decoder_layers), 0
    ).stage_weights()
    return (leading, *per_stage)


def visual_token_counts_by_layer(layer_list: Sequence[int], stage_token_counts: Sequence[int],
                                 num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS
                                 ) -> list[int]:
    """Visual tokens held by each 0-indexed decoder layer under this schedule."""
    return [int(count) for count in
            layer_schedule(layer_list, stage_token_counts, num_decoder_layers).counts_by_layer()]


def average_retained_tokens(layer_list: Sequence[int], stage_token_counts: Sequence[int],
                            num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS) -> float:
    """The reported budget: visual tokens averaged over every decoder layer."""
    return layer_schedule(layer_list, stage_token_counts, num_decoder_layers).layer_average


def minimum_average_retained_tokens(layer_list: Sequence[int],
                                    visual_tokens: int = DEFAULT_VISUAL_TOKENS,
                                    num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS) -> float:
    """Smallest layer-average this stage division can reach, i.e. the lowest budget
    label it can serve.

    The first stage is never pruned and every later stage keeps at least one token,
    so no schedule of counts can average below this.
    """
    boundaries = tuple(int(layer) for layer in layer_list)
    return LayerSchedule(
        boundaries, (1,) * len(boundaries), int(num_decoder_layers), int(visual_tokens)
    ).floor(1)


def profile_for_budget(max_tokens: int) -> dict[str, object]:
    """The schedule for one budget label, or ``{}`` when the budget has none."""
    return dict(PYRAMIDDROP_PROFILES.get(int(max_tokens), {}))


def qwen_pyramid_schedule(
    retain_ratio: float,
    num_decoder_layers: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Derive four Qwen stages whose continuous layer average is ``retain_ratio``.

    The first stage is necessarily full-token. Its width is chosen as late as the
    requested ratio permits, capped at the paper's quarter-depth boundary. The
    remaining decoder is split into three stages and a single geometric decay
    ``lambda`` is solved by bisection. Returned ratios are cumulative fractions
    of the original per-image visual span.
    """
    ratio = float(retain_ratio)
    depth = int(num_decoder_layers)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
    if depth < 4:
        raise ValueError(f"PyramidDrop needs at least four decoder layers, got {depth}")
    if ratio == 1.0:
        return (), ()
    if ratio <= 1.0 / depth:
        raise ValueError(
            f"retain_ratio={ratio:g} is unreachable on a {depth}-layer decoder: "
            "the first full-token layer alone spends 1/depth of the budget."
        )

    first_width = max(1, int(ratio * depth - 1e-12))
    first_width = min(depth // 4, first_width)
    remaining = depth - first_width
    boundaries = (
        first_width,
        first_width + remaining // 3,
        first_width + (2 * remaining) // 3,
    )
    widths = stage_widths(boundaries, depth)

    low, high = 0.0, 1.0
    for _ in range(80):
        decay = (low + high) / 2.0
        achieved = sum(
            width * (decay ** stage)
            for stage, width in enumerate(widths)
        ) / depth
        if achieved < ratio:
            low = decay
        else:
            high = decay
    decay = (low + high) / 2.0
    return boundaries, tuple(decay ** stage for stage in range(1, len(widths)))


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
        raise RuntimeError("Could not resolve the image placeholder token id for PyramidDrop.")
    return int(token_id)


class PyramidDropCompressor(TokenCompressor):
    """Prune the visual span in stages, ranked by the last prompt token's attention.

    Parameters
    ----------
    max_tokens:
        Published budget: the decoder-layer-averaged visual-token count, which
        selects the schedule (see the module docstring). Must be one of
        ``PYRAMIDDROP_PROFILES`` unless a schedule is given explicitly.
    retain_ratio:
        Layer-averaged fraction of each image's dynamic visual span on Qwen2.5-VL
        and LLaVA-NeXT. On NeXT this is the sole budget control; the loaded
        decoder depth determines the hidden canonical boundaries.
    layer_list:
        1-indexed layers a stage ends after; layer ``L`` (0-indexed) is the first
        to run on the reduced set. Strictly increasing. Must be given together with
        ``stage_token_counts``.
    stage_token_counts:
        Visual tokens entering each stage, non-increasing, one more entry than
        ``layer_list``. The first is the full visual sequence. The paper's own
        setting is ``(576, 288, 144, 72)`` on ``(8, 16, 24)``.
    visual_tokens:
        Visual tokens the projector emits, 576 on LLaVA-1.5. Only used to size the
        schedule; the count actually observed at run time wins.
    num_decoder_layers:
        Decoder depth the schedule was built for, 32 on LLaVA-1.5-7B. ``install()``
        re-solves the schedule when the backbone is deeper or shallower (LLaVA-1.5-13B
        has 40 layers), moving each boundary to the same fraction of the decoder and
        re-solving the counts so the layer-average label still holds.
    track_source:
        Record ``[T_current, T_original]`` matrices for ``llmoe.viz``. Required by
        ``bm --visualize``; off by default so benchmark runs don't pay for it.
    """

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS, *, retain_ratio: float | None = None,
                 layer_list: Sequence[int] | None = None,
                 stage_token_counts: Sequence[int] | None = None,
                 visual_tokens: int = DEFAULT_VISUAL_TOKENS,
                 num_decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS,
                 track_source: bool = False) -> None:
        super().__init__(int(max_tokens))
        if retain_ratio is not None and not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
        if retain_ratio is not None and (layer_list is not None or stage_token_counts is not None):
            raise ValueError(
                "retain_ratio selects PyramidDrop's canonical dynamic schedule; "
                "layer_list/stage_token_counts configure the fixed-grid LLaVA schedule."
            )
        self._retain_ratio: float | None = (
            None if retain_ratio is None else float(retain_ratio)
        )
        if (layer_list is None) != (stage_token_counts is None):
            raise ValueError(
                "layer_list and stage_token_counts must be given together: a schedule is a "
                "division into stages plus the count entering each of them."
            )
        if self._retain_ratio is not None:
            layer_list = ()
            stage_token_counts = ()
            self._schedule_source = "canonical ratio schedule (resolved at install)"
        elif layer_list is None:
            profile = profile_for_budget(max_tokens)
            if not profile:
                supported = ", ".join(str(budget) for budget in SUPPORTED_BUDGETS)
                raise ValueError(
                    f"Unsupported PyramidDrop budget: {max_tokens}. Only {{{supported}}} have a "
                    f"schedule; pass `layer_list` and `stage_token_counts` to use another."
                )
            layer_list = profile["layer_list"]
            stage_token_counts = profile["stage_token_counts"]
            self._schedule_source: str = str(profile["source"])
        else:
            self._schedule_source = "caller-supplied"

        self._visual_tokens: int = int(visual_tokens)
        self._num_decoder_layers: int = int(num_decoder_layers)
        self._layer_list: tuple[int, ...] = tuple(int(layer) for layer in layer_list)
        self._stage_token_counts: tuple[int, ...] = tuple(
            int(count) for count in stage_token_counts)
        self._warned: set[str] = set()
        if self._retain_ratio is None:
            self._validate_schedule()

        # Preserve the configured schedule while InternVL resolves a dynamic
        # schedule from each source image's globally concatenated token count.
        self._configured_num_decoder_layers: int = self._num_decoder_layers
        self._configured_layer_list: tuple[int, ...] = self._layer_list
        self._configured_stage_token_counts: tuple[int, ...] = self._stage_token_counts

        self._track_source: bool = bool(track_source)
        # Layers `[layer_list[s], layer_list[s + 1])` run under stage `s + 1`; the
        # boundary layer itself is where stage `s`'s cut is applied.
        self._boundary_stage: dict[int, int] = {
            layer: stage for stage, layer in enumerate(self._layer_list)}

        # Per-forward state, rebuilt by the inner-model pre-hook on every prefill.
        self._image_start: torch.Tensor | None = None  # [B] first visual column
        self._query_offset: torch.Tensor | None = None  # [B] rank query, past the span's end
        self._padding_mask: torch.Tensor | None = None  # [B, L] prefill padding, or None
        self._visual_count: int = 0
        self._stage_state: dict[int, dict[str, Any]] = {}
        self._composed_source: torch.Tensor | None = None
        self._forward_count: int = 0

        # Installation bookkeeping, all restored by uninstall().
        self._installed_vlm: VLM | None = None
        self._language_model: Any | None = None
        self._hook_handles: list[RemovableHandle] = []
        self._image_token_id: int | None = None
        self._qwen: Any | None = None
        self._internvl: Any | None = None
        self._llava_next: bool = False
        self._llava_next_stage_ratios: tuple[float, ...] = ()

    @property
    @override
    def budget(self) -> int | float:
        return self._max_tokens if self._retain_ratio is None else self._retain_ratio

    def validate_llava_next(self) -> None:
        """Expose only a layer-average ratio; keep method schedules internal."""

        from llmoe.compressor.llava_next import validate_retain_ratio

        validate_retain_ratio(self._retain_ratio, label="PyramidDrop")
        if self._max_tokens != DEFAULT_MAX_TOKENS:
            raise ValueError(
                "PyramidDrop on LLaVA-NeXT only accepts retain_ratio; remove max_tokens "
                f"(its constructor default is {DEFAULT_MAX_TOKENS} and is ignored on NeXT)."
            )
        if self._visual_tokens != DEFAULT_VISUAL_TOKENS:
            raise ValueError(
                "PyramidDrop on LLaVA-NeXT derives visual counts from AnyRes; remove "
                f"visual_tokens (the ignored default is {DEFAULT_VISUAL_TOKENS})."
            )
        if self._configured_num_decoder_layers != DEFAULT_NUM_DECODER_LAYERS:
            raise ValueError(
                "PyramidDrop on LLaVA-NeXT derives decoder depth from the model; remove "
                f"num_decoder_layers (the default is {DEFAULT_NUM_DECODER_LAYERS})."
            )

    def _validate_schedule(self) -> None:
        layers, counts = self._layer_list, self._stage_token_counts
        if not layers:
            raise ValueError("PyramidDrop needs at least one stage boundary")
        if len(counts) != len(layers) + 1:
            raise ValueError(
                f"stage_token_counts must hold one count per stage, i.e. {len(layers) + 1} for "
                f"{len(layers)} boundaries; got {len(counts)}"
            )
        if list(layers) != sorted(set(layers)) or layers[0] < 1:
            raise ValueError("layer_list must be strictly increasing and >= 1")
        if layers[-1] >= self._num_decoder_layers:
            raise ValueError(
                f"boundary {layers[-1]} leaves no layer to run reduced in a "
                f"{self._num_decoder_layers}-layer decoder"
            )
        if counts[0] != self._visual_tokens:
            raise ValueError(
                f"the first stage holds the whole visual sequence, so stage_token_counts[0] must "
                f"be {self._visual_tokens}, got {counts[0]}"
            )
        if any(count < 1 for count in counts):
            raise ValueError("every PyramidDrop stage must keep at least one visual token")
        if list(counts) != sorted(counts, reverse=True):
            raise ValueError("stage_token_counts must be non-increasing")

        floor = minimum_average_retained_tokens(
            layers, self._visual_tokens, self._num_decoder_layers)
        if self._max_tokens < floor:
            raise ValueError(
                f"PyramidDrop budget {self._max_tokens} is unreachable on boundaries {layers}: "
                f"{stage_widths(layers, self._num_decoder_layers)[0]} unpruned layer(s) plus one "
                f"token per later stage already average {floor:g}. Cut a stage earlier."
            )
        # Integer per-stage counts make the average discrete, so a caller-supplied
        # schedule may land near its label rather than on it. Under a token is the
        # rounding; anything more is mislabelled, and the run directory would carry
        # a budget the schedule does not spend.
        achieved = average_retained_tokens(layers, counts, self._num_decoder_layers)
        if abs(achieved - self._max_tokens) >= 1.0:
            self._warn_once(
                "budget_mismatch",
                f"this schedule averages {achieved:g} visual tokens per layer, but the run is "
                f"labelled b{self._max_tokens}. Adjust stage_token_counts or the label.",
            )

    def _adopt_schedule(self, schedule: LayerSchedule) -> None:
        """Run under `schedule` until uninstall, which restores the configured one."""
        self._num_decoder_layers = schedule.depth
        self._layer_list = schedule.boundaries
        self._stage_token_counts = (int(schedule.n_visual),
                                    *(int(keep) for keep in schedule.keeps))
        self._boundary_stage = {
            layer: stage for stage, layer in enumerate(self._layer_list)
        }

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[PyramidDrop] WARNING: {message}", flush=True)

    @property
    def achieved_average(self) -> float:
        """Layer-average the configured schedule actually realises on a full span."""
        if self._retain_ratio is not None:
            return self._retain_ratio
        return average_retained_tokens(
            self._layer_list, self._stage_token_counts, self._num_decoder_layers)

    # ----------------------------------------------------------------- selection

    def _stage_for_layer(self, layer_idx: int | None) -> int:
        """Stage whose cut fires at ``layer_idx``; the first boundary when unnamed."""
        if layer_idx is None:
            return 0
        stage = self._boundary_stage.get(int(layer_idx))
        if stage is None:
            raise ValueError(
                f"layer {layer_idx} is not a PyramidDrop boundary; boundaries are "
                f"{self._layer_list}"
            )
        return stage

    def _target_count(self, stage: int, n_tokens: int) -> int:
        """Visual tokens leaving stage ``stage``, clipped to what is actually there."""
        return max(1, min(int(n_tokens), int(self._stage_token_counts[stage + 1])))

    @staticmethod
    def _scores_from_attention(attention: torch.Tensor, batch: int, n_tokens: int) -> torch.Tensor:
        """Normalise an ``attention`` argument to ``[B, N]`` per-token scores.

        Accepts per-token scores (``[N]`` / ``[B, N]``) or raw attention weights
        (``[B, Q, N]`` / ``[B, H, Q, N]``), from which the last query row is taken
        after averaging over heads -- the paper's lightweight rank.
        """
        scores = attention.float()
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        elif scores.ndim == 3:
            scores = scores[:, -1, :]
        elif scores.ndim == 4:
            scores = scores.mean(dim=1)[:, -1, :]
        if scores.ndim != 2:
            raise ValueError(f"Unsupported attention shape {tuple(attention.shape)}")
        if scores.shape[0] == 1 and batch > 1:
            scores = scores.expand(batch, -1)
        if tuple(scores.shape) != (batch, n_tokens):
            raise ValueError(
                f"attention scores {tuple(scores.shape)} do not match the {batch} x {n_tokens} "
                f"tokens being compressed"
            )
        return scores

    @staticmethod
    def _one_hot_source(indices: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """``[keep, N]`` pruning source matrix for ``llmoe.viz`` (one-hot rows)."""
        source = torch.zeros(indices.shape[0], n_tokens, dtype=torch.float32)
        source.scatter_(1, indices.detach().cpu().unsqueeze(1), 1.0)
        return source

    # ----------------------------------------------------------- TokenCompressor

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, **kwargs: Any) -> CompressedFeatures:
        """Apply one boundary's cut to ``[B, N, D]`` (or ``[N, D]``) visual tokens.

        This is PyramidDrop's selection rule on its own, and the entry point for
        tensor-level tests. ``attention`` carries the rank scores over the visual
        span (see ``_scores_from_attention`` for accepted shapes) and ``layer_idx`` names
        the boundary whose budget to apply, defaulting to the first.

        With ``attention=None`` there is no criterion, so the tokens pass through
        unchanged -- which is also what happens on the real vision path, where all
        576 projected patches reach the decoder and only the decoder prunes.
        """
        del kwargs
        if self._retain_ratio is not None:
            raise RuntimeError(
                "PyramidDropCompressor.compress() is the LLaVA tensor path. "
                "Qwen ratio pruning runs at decoder boundaries through "
                "QwenPyramidDropAdapter."
            )

        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Expected [B, N, D] or [N, D], got {tuple(features.shape)}")
        batch, n_tokens, hidden_dim = features.shape
        if n_tokens == 0:
            raise ValueError("Cannot compress an empty token sequence")

        if attention is None:
            if self._track_source:
                identity = torch.eye(n_tokens, dtype=torch.float32)
                self._source_history = [identity, identity.clone()]
            return CompressedFeatures(tokens=features.squeeze(0) if squeeze else features)

        keep = self._target_count(self._stage_for_layer(layer_idx), n_tokens)
        scores = self._scores_from_attention(attention.to(features.device), batch, n_tokens)
        indices = scores.topk(keep, dim=1).indices.sort(dim=1).values
        tokens = features.gather(1, indices.unsqueeze(-1).expand(batch, keep, hidden_dim))
        if self._track_source:
            self._source_history = [
                torch.eye(n_tokens, dtype=torch.float32),
                self._one_hot_source(indices[0], n_tokens),
            ]
        return CompressedFeatures(tokens=tokens.squeeze(0) if squeeze else tokens)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        """``[identity, cut_0, cut_1, ...]`` for the last forward: the pre-prune
        identity followed by one one-hot ``[survivors, n_visual]`` matrix per
        boundary that fired. Empty when ``track_source`` is off or no forward has
        run."""
        if self._qwen is not None:
            return self._qwen.get_source_history()
        return list(self._source_history)

    # ------------------------------------------------------------------- install

    @override
    def install(self, vlm: VLM) -> None:
        # Deliberately not `super().install()`: PyramidDrop performs no vision-stage
        # compression, so `get_image_features` must stay untouched and LLaVA's
        # `<image>` placeholder count stays correct on its own.
        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return
            raise RuntimeError("PyramidDropCompressor is already installed on another backbone")

        inner = vlm.model.model
        from llmoe.compressor.llava_next import is_llava_next

        llava_next = is_llava_next(vlm)
        if getattr(inner, "visual", None) is not None:
            if self._retain_ratio is None:
                raise ValueError(
                    "PyramidDrop on Qwen2.5-VL requires retain_ratio: NaViT "
                    "emits a different visual-token count for each image."
                )
            from llmoe.compressor.pyramiddrop.pyramiddrop_qwen import QwenPyramidDropAdapter
            from llmoe.utils.qwen import resolve_image_token_id as resolve_qwen_image_token_id

            adapter = QwenPyramidDropAdapter(
                retain_ratio=self._retain_ratio,
                track_source=self._track_source,
            )
            adapter.install(
                vlm,
                image_token_id=resolve_qwen_image_token_id(
                    vlm.model.config, vlm.processor, label="PyramidDrop"
                ),
            )
            self._installed_vlm = vlm
            self._qwen = adapter
            return
        if self._retain_ratio is not None and not llava_next:
            raise ValueError(
                "retain_ratio is PyramidDrop's Qwen2.5-VL parameterisation; "
                "LLaVA-1.5 and InternVL use max_tokens and a count schedule."
            )
        language_model = getattr(inner, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if not layers:
            raise TypeError(
                "PyramidDrop needs a decoder stack at model.model.language_model.layers")
        if llava_next:
            self.validate_llava_next()
            self._num_decoder_layers = len(layers)
            (
                self._layer_list,
                self._llava_next_stage_ratios,
            ) = qwen_pyramid_schedule(float(self._retain_ratio), len(layers))
            self._stage_token_counts = ()
            self._boundary_stage = {
                layer: stage for stage, layer in enumerate(self._layer_list)
            }
            self._llava_next = True
        from llmoe.compressor.pyramiddrop.pyramiddrop_internvl import (
            InternVLPyramidDropAdapter,
        )

        if InternVLPyramidDropAdapter.supports(vlm):
            dynamic = self._schedule_source != "caller-supplied"
            self._internvl = InternVLPyramidDropAdapter(
                max_tokens=self._max_tokens,
                decoder_layers=len(layers),
                dynamic=dynamic,
                schedule_solver=qwen_pyramid_schedule,
            )
            if dynamic:
                self._num_decoder_layers = len(layers)
                self._layer_list = ()
                self._stage_token_counts = ()
                self._boundary_stage = {}
        elif len(layers) != self._num_decoder_layers:
            # The boundaries are fractions of the decoder, not absolute layers, so
            # re-solve them (and the counts that keep the label true) for this
            # backbone's depth rather than running a 32-layer schedule on it.
            self._adopt_schedule(
                layer_schedule(
                    self._layer_list, self._stage_token_counts, self._num_decoder_layers
                ).rescaled_to(
                    len(layers), label=f"PyramidDrop's budget-{self._max_tokens} schedule"
                )
            )
            print(
                f"[PyramidDrop] resolved the budget-{self._max_tokens} schedule for a "
                f"{len(layers)}-layer decoder: boundaries {list(self._layer_list)}, "
                f"counts {list(self._stage_token_counts)}, "
                f"layer-average {self.achieved_average:.2f}",
                flush=True,
            )

        self._installed_vlm = vlm
        self._language_model = language_model
        self._image_token_id = resolve_image_token_id(vlm.model.config, vlm.processor)

        # The visual span is read from input_ids before the vision features are
        # scattered in, so every boundary knows which columns are visual.
        self._hook_handles.append(
            inner.register_forward_pre_hook(self._capture_visual_span, with_kwargs=True)
        )
        # Dynamic InternVL boundaries can start at any nonzero layer, so install
        # the same lightweight pre-hook coverage used by the Qwen adapter.
        if self._internvl is not None and self._internvl.dynamic:
            hook_start = 1
        elif not self._layer_list:
            hook_start = len(layers)
        else:
            hook_start = self._layer_list[0]
        for layer_idx in range(hook_start, len(layers)):
            self._hook_handles.append(
                layers[layer_idx].register_forward_pre_hook(
                    self._make_layer_hook(layer_idx), with_kwargs=True)
            )

        label = self._internvl.label if self._internvl is not None else "PyramidDrop"
        if self._llava_next:
            label = "PyramidDrop/LLaVA-NeXT"
            print(
                f"[{label}] install: retain_ratio={self._retain_ratio:g} "
                "(decoder-layer-averaged), "
                f"boundaries={list(self._layer_list)}, "
                "cumulative_stage_ratios="
                f"{[round(ratio, 6) for ratio in self._llava_next_stage_ratios]}, "
                f"image_token_id={self._image_token_id}",
                flush=True,
            )
        elif self._internvl is not None and self._internvl.dynamic:
            print(
                f"[{label}] install: max_tokens={self._max_tokens} (layer-averaged); "
                "uses the Qwen-derived stage solver once over each globally "
                "concatenated tile span; boundaries and targets resolve per image, "
                f"image_token_id={self._image_token_id}",
                flush=True,
            )
        else:
            print(
                f"[{label}] install: max_tokens={self._max_tokens} "
                f"(layer-averaged; schedule averages {self.achieved_average:g}), "
                f"boundaries={list(self._layer_list)}, "
                f"stage_tokens={list(self._stage_token_counts)}, "
                f"source={self._schedule_source}, image_token_id={self._image_token_id}",
                flush=True,
            )

    def uninstall(self) -> None:
        if self._qwen is not None:
            self._qwen.uninstall()
            self._qwen = None
            self._installed_vlm = None
            return
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._installed_vlm = None
        self._language_model = None
        self._image_token_id = None
        self._reset_forward_state()
        self._internvl = None
        self._num_decoder_layers = self._configured_num_decoder_layers
        self._layer_list = self._configured_layer_list
        self._llava_next = False
        self._llava_next_stage_ratios = ()
        self._stage_token_counts = self._configured_stage_token_counts
        self._boundary_stage = {
            layer: stage for stage, layer in enumerate(self._layer_list)
        }

    def _reset_forward_state(self) -> None:
        self._image_start = None
        self._query_offset = None
        self._padding_mask = None
        self._visual_count = 0
        self._stage_state = {}
        self._composed_source = None
        if self._internvl is not None and self._internvl.dynamic:
            self._layer_list = ()
            self._stage_token_counts = ()
            self._boundary_stage = {}
        if self._llava_next:
            self._stage_token_counts = ()

    # -------------------------------------------------------------- visual span

    def _capture_visual_span(self, _module: Any, args: tuple, kwargs: dict) -> None:
        """Record the visual span and the rank query from each prefill's ``input_ids``.

        Decode steps (width-1 ``input_ids``) are left alone so the state built by
        the prefill -- and the per-stage caches it produced -- stays valid.
        """
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 \
                or input_ids.shape[1] <= 1:
            return None

        self._reset_forward_state()
        self._forward_count += 1

        if self._llava_next:
            from llmoe.compressor.llava_next import (
                decoder_visual_positions,
                retained_tokens,
            )

            positions = decoder_visual_positions(
                input_ids,
                token_id=int(self._image_token_id),
                label="PyramidDrop",
            )
            if positions is None:
                return None
            rows = [positions]
        else:
            rows = [
                (row == self._image_token_id).nonzero(as_tuple=True)[0] for row in input_ids
            ]
        counts = {int(row.numel()) for row in rows}
        if counts == {0}:
            return None  # text-only prompt: PyramidDrop has nothing to prune
        if len(counts) != 1:
            self._warn_once(
                "ragged_batch",
                f"rows carry different visual-token counts {sorted(counts)}; pruning is SKIPPED "
                f"for such batches. Batch images of equal token count.",
            )
            return None
        count = counts.pop()
        if any(int(row[-1]) - int(row[0]) + 1 != count for row in rows):
            self._warn_once(
                "ragged_span",
                "the visual tokens are not one contiguous run per row; pruning is SKIPPED.",
            )
            return None
        if self._llava_next:
            targets = tuple(
                retained_tokens(count, ratio)
                for ratio in self._llava_next_stage_ratios
            )
            self._stage_token_counts = (count, *targets)
        elif self._internvl is not None and self._internvl.dynamic:
            (
                self._layer_list,
                self._stage_token_counts,
                self._boundary_stage,
            ) = self._internvl.configure(count)
        elif count != self._visual_tokens:
            self._warn_once(
                "unexpected_count",
                f"expected {self._visual_tokens} visual tokens per row but found {count}; the "
                f"schedule's counts are clipped to {count}, so the realised layer-average will "
                f"not be {self._max_tokens}.",
            )

        starts = torch.stack([row[0] for row in rows], dim=0)
        # The reference ranks with the last unpadded column of each row. It is held
        # relative to the end of the visual span -- the only part of the sequence
        # that shrinks -- so it stays valid at every later boundary.
        padding_mask = self._padding_from(kwargs.get("attention_mask"), input_ids)
        last_valid = self._last_valid_column(padding_mask)
        if bool((last_valid < starts + count).any()):
            self._warn_once(
                "query_inside_span",
                "no prompt token follows the visual span, so there is nothing to rank the image "
                "with; pruning is SKIPPED.",
            )
            return None

        self._image_start = starts
        self._query_offset = last_valid - (starts + count)
        self._padding_mask = padding_mask
        self._visual_count = count
        if self._track_source:
            identity = torch.eye(count, dtype=torch.float32)
            self._source_history = [identity]
            self._composed_source = identity
        if self._forward_count == 1:
            label = (
                self._internvl.label
                if self._internvl is not None
                else "PyramidDrop/LLaVA-NeXT" if self._llava_next else "PyramidDrop"
            )
            achieved = average_retained_tokens(
                self._layer_list, self._stage_token_counts, self._num_decoder_layers
            )
            print(f"[{label}] first prefill: seq_len={input_ids.shape[1]}, visual={count} "
                  f"at [{int(starts[0])}, {int(starts[0]) + count}), "
                  f"boundaries={list(self._layer_list)}, "
                  f"schedule={list(self._stage_token_counts)}, average={achieved:g}", flush=True)
        return None

    @staticmethod
    def _padding_from(attention_mask: Any, input_ids: torch.Tensor) -> torch.Tensor:
        """The prefill's ``[B, L]`` padding mask; all-valid when the caller sent none."""
        batch, seq_len = input_ids.shape
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2 \
                and attention_mask.shape[-1] == seq_len:
            return attention_mask.expand(batch, seq_len).long()
        return torch.ones(batch, seq_len, dtype=torch.long, device=input_ids.device)

    @staticmethod
    def _last_valid_column(padding_mask: torch.Tensor) -> torch.Tensor:
        """Last unpadded column of each row, the reference's ranking query."""
        batch, seq_len = padding_mask.shape
        positions = torch.arange(seq_len, device=padding_mask.device).expand(batch, seq_len)
        return positions.masked_fill(~padding_mask.bool(), -1).amax(dim=1)

    # ---------------------------------------------------------------- prefill cut

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
            return self._layer_hook(module, args, kwargs, layer_idx)

        return hook

    def _layer_hook(self, module: Any, args: tuple, kwargs: dict,
                    layer_idx: int) -> tuple[tuple, dict] | None:
        hidden = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            return None

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
        kwargs["position_ids"] = state["position_ids"]
        kwargs["position_embeddings"] = state["position_embeddings"]
        if hidden.shape[1] > 1:  # prefill: everything downstream sees the short sequence
            kwargs["attention_mask"] = state["mask"]
        else:  # decode: only this stage's key space is narrower than the layers above
            kwargs["attention_mask"] = self._decode_mask(kwargs.get("attention_mask"), state)
        if "cache_position" in kwargs:
            kwargs["cache_position"] = state["cache_position"]
        return (hidden,), kwargs

    def _active_state(self, layer_idx: int) -> dict[str, Any] | None:
        """The most recent boundary state governing ``layer_idx``, if any cut ran."""
        for stage in range(len(self._layer_list) - 1, -1, -1):
            if self._layer_list[stage] <= layer_idx and stage in self._stage_state:
                return self._stage_state[stage]
        return None

    def _base_state(self, stage: int) -> dict[str, Any] | None:
        """The geometry stage ``stage`` starts from: the last earlier cut, if any."""
        for earlier in range(stage - 1, -1, -1):
            if earlier in self._stage_state:
                return self._stage_state[earlier]
        return None

    def _prune(self, layer: Any, hidden: torch.Tensor, kwargs: dict,
               stage: int) -> torch.Tensor:
        """Cut the visual span to this boundary's budget and size the rest of the stack.

        The decoder hands every layer the geometry it built once for the *full*
        sequence, so a later boundary must start from the previous boundary's state
        rather than from ``kwargs``.

        Each boundary runs once per prefill, so a second entry can only be gradient
        checkpointing recomputing the layer on the same input. That replays the cut
        this stage already decided instead of re-scoring: the counters it advanced
        describe the sequence *after* the cut.
        """
        replay = self._stage_state.get(stage)
        if replay is not None:
            keep_index = replay["keep_index"]
            return hidden.gather(
                1, keep_index.unsqueeze(-1).expand(*keep_index.shape, hidden.shape[-1]))

        base = self._base_state(stage)
        batch, seq_len, hidden_dim = hidden.shape
        target = self._target_count(stage, self._visual_count)
        if target >= self._visual_count:
            return hidden  # this boundary's budget already fits; nothing to cut

        start = self._image_start.to(hidden.device)
        columns = start.unsqueeze(1) + torch.arange(self._visual_count, device=hidden.device)
        if start.shape[0] != batch or int(columns.max()) >= seq_len:
            self._warn_once(
                "geometry_changed",
                f"the visual span {tuple(columns.shape)} does not fit the layer-"
                f"{self._layer_list[stage]} input {tuple(hidden.shape)}; pruning is SKIPPED for "
                f"this batch.",
            )
            return hidden

        base_embeddings = base["position_embeddings"] if base is not None \
            else kwargs.get("position_embeddings")
        base_mask = base["mask"] if base is not None else kwargs.get("attention_mask")
        base_padding = base["padding"] if base is not None else self._prefill_padding(hidden)

        scores = self._rank_scores(layer, hidden, base_embeddings, base_padding)
        local = scores.gather(1, columns).topk(target, dim=1).indices.sort(dim=1).values
        kept_columns = columns.gather(1, local)
        keep_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden.device)
        keep_mask.scatter_(1, columns, False)  # drop the whole visual span ...
        keep_mask.scatter_(1, kept_columns, True)  # ... then put the winners back
        keep_index = keep_mask.nonzero(as_tuple=True)[1].view(batch, -1)  # ascending per row
        new_len = keep_index.shape[1]

        new_hidden = hidden.gather(1, keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim))
        padding = base_padding.gather(1, keep_index)
        # The reference rebuilds position ids from the compacted padding mask, so
        # this stage sees a gapless numbering of its own shorter sequence.
        position_ids = (padding.long().cumsum(dim=-1) - 1).clamp_min(0)
        position_embeddings = self._language_model.rotary_emb(
            new_hidden, position_ids=position_ids)

        self._stage_state[stage] = {
            "batch": batch,
            "padding": padding,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "mask": self._pruned_mask(base_mask, keep_index, seq_len),
            "valid": padding.bool(),
            # Layers at or past this boundary cache only the survivors, so their
            # cache positions are compact.
            "cache_position": torch.arange(new_len, device=hidden.device),
            "kv_len": new_len,
            # Kept so a recomputed forward replays this exact cut (see above).
            "keep_index": keep_index,
        }
        if self._forward_count == 1:
            print(f"[PyramidDrop] stage {stage} @layer{self._layer_list[stage]}: seq {seq_len} -> "
                  f"{new_len} (visual {self._visual_count} -> {target})", flush=True)
        self._visual_count = target
        self._record_cut(local)
        return new_hidden

    def _prefill_padding(self, hidden: torch.Tensor) -> torch.Tensor:
        """The ``[B, L]`` padding mask the first cut starts from, captured at prefill."""
        padding = self._padding_mask
        if padding is None or padding.shape[1] != hidden.shape[1]:
            return torch.ones(hidden.shape[0], hidden.shape[1], dtype=torch.long,
                              device=hidden.device)
        return padding.to(hidden.device)

    def _record_cut(self, local: torch.Tensor) -> None:
        """Narrow the composed source attribution to the survivors of one cut."""
        if not self._track_source or self._composed_source is None:
            return
        self._composed_source = self._composed_source[local[0].detach().cpu()]
        self._source_history.append(self._composed_source)

    def _rank_scores(self, layer: Any, hidden: torch.Tensor,
                     position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
                     padding: torch.Tensor) -> torch.Tensor:
        """The paper's lightweight rank: the *next* layer's LN + Q/K + RoPE softmax.

        The pre-hook sits on the layer that will run first after the cut, so its
        ``input_layernorm``/``q_proj``/``k_proj`` applied to ``hidden`` reproduce the
        reference's ranking attention exactly. Only the last prompt token's query
        row is built, so the quadratic map is never materialised.
        """
        attn = getattr(layer, "self_attn", None)
        layernorm = getattr(layer, "input_layernorm", None)
        if attn is None or layernorm is None or not hasattr(attn, "q_proj") \
                or not hasattr(attn, "k_proj"):
            raise RuntimeError(
                "PyramidDrop needs LLaMA-style decoder layers exposing `input_layernorm` and "
                "`self_attn.q_proj`/`k_proj` to recompute the ranking attention."
            )
        batch, seq_len, _ = hidden.shape
        query_column = self._query_columns(hidden.device)
        if int(query_column.max()) >= seq_len:
            raise RuntimeError(
                f"PyramidDrop rank query at column {int(query_column.max())} exceeds the boundary "
                f"sequence length {seq_len}; the prompt changed between hooks."
            )

        config = attn.config
        n_heads = int(config.num_attention_heads)
        n_kv = int(getattr(config, "num_key_value_heads", n_heads))
        head_dim = int(getattr(attn, "head_dim", hidden.shape[-1] // n_heads))
        normed = layernorm(hidden)
        query = attn.q_proj(normed).view(batch, seq_len, n_heads, head_dim)
        key = attn.k_proj(normed).view(batch, seq_len, n_kv, head_dim)
        q_norm, k_norm = getattr(attn, "q_norm", None), getattr(attn, "k_norm", None)
        if callable(q_norm):
            query = q_norm(query)
        if callable(k_norm):
            key = k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        if position_embeddings is not None and _APPLY_ROPE is not None:
            cos, sin = position_embeddings
            query, key = _APPLY_ROPE(query, key, cos, sin)
        if n_kv != n_heads:  # GQA: replicate kv heads to match the query heads
            key = key.repeat_interleave(n_heads // n_kv, dim=1)

        query = query.gather(
            2, query_column[:, None, None, None].expand(batch, n_heads, 1, head_dim))
        scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
        logits = (query.float() @ key.float().transpose(-1, -2)) * scaling  # [B, H, 1, L]
        # Causal plus padding: the query may only see unpadded keys at or before it.
        keys = torch.arange(seq_len, device=hidden.device).expand(batch, seq_len)
        allowed = padding.bool() & (keys <= query_column.unsqueeze(1))
        logits = logits.masked_fill(~allowed[:, None, None, :], torch.finfo(logits.dtype).min)
        return logits.softmax(dim=-1).mean(dim=1)[:, 0, :]  # [B, L], head-averaged

    def _query_columns(self, device: torch.device) -> torch.Tensor:
        """Current-sequence column of each row's rank query, following the span."""
        start = self._image_start.to(device)
        return self._query_offset.to(device) + start + self._visual_count

    @staticmethod
    def _pruned_mask(attention_mask: Any, keep_index: torch.Tensor,
                     seq_len: int) -> torch.Tensor | None:
        """Select the kept rows and columns of the mask this boundary started from.

        Restricting a causal mask to an ascending subset of positions is still
        exactly causal over the compacted order. The incoming format is mirrored
        rather than normalised, since the backend decides it (4-D additive float
        under eager, 4-D bool or ``None`` under SDPA, 2-D padding under flash).
        """
        batch, new_len = keep_index.shape

        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4:
            if attention_mask.shape[-1] != seq_len:
                raise RuntimeError(
                    f"PyramidDrop expected a [B, H, {seq_len}, {seq_len}] mask at this boundary, "
                    f"got {tuple(attention_mask.shape)}"
                )
            mask = attention_mask.expand(batch, -1, -1, -1)
            heads = mask.shape[1]
            mask = mask.gather(
                2, keep_index[:, None, :, None].expand(batch, heads, new_len, seq_len))
            return mask.gather(
                3, keep_index[:, None, None, :].expand(batch, heads, new_len, new_len))

        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2 \
                and attention_mask.shape[-1] == seq_len:
            return attention_mask.expand(batch, seq_len).gather(1, keep_index)

        return None  # no mask supplied: the backend's causal default is correct

    # ------------------------------------------------------------------ decoding

    def _advance_decode_step(self, hidden: torch.Tensor, stage: int) -> None:
        """Extend one stage's pruned key space by the token generated this step.

        The reference gives the new query the position ``number of valid keys this
        stage has cached``, which is what keeps its renumbered prefill positions and
        its decode positions on one contiguous scale.
        """
        state = self._stage_state[stage]
        if hidden.shape[0] != state["batch"]:
            raise RuntimeError(
                f"PyramidDrop cut the cache for batch {state['batch']} but decoding now runs "
                f"batch {hidden.shape[0]}. Beam search expands the batch after prefill and is "
                f"not supported; use greedy or sampling decoding."
            )
        padding = state["padding"].to(hidden.device)
        position_ids = padding.long().sum(dim=-1, keepdim=True)
        state["position_ids"] = position_ids
        state["position_embeddings"] = self._language_model.rotary_emb(
            hidden, position_ids=position_ids)

        column = torch.ones(state["batch"], 1, dtype=padding.dtype, device=padding.device)
        state["padding"] = torch.cat([padding, column], dim=1)
        state["valid"] = state["padding"].bool()
        state["kv_len"] += 1
        state["cache_position"] = torch.tensor(
            [state["kv_len"] - 1], device=hidden.device, dtype=torch.long)

    @staticmethod
    def _decode_mask(attention_mask: Any, state: dict[str, Any]) -> torch.Tensor | None:
        """Rebuild a ``kv_len``-wide mask for a pruned stage while decoding.

        Layers above a boundary still hold the longer cache, so the mask the
        decoder built is too wide down here. When it built none there is nothing to
        fix -- the single query attends to everything cached.
        """
        if not isinstance(attention_mask, torch.Tensor):
            return attention_mask
        valid = state["valid"]
        if attention_mask.shape[-1] == valid.shape[-1]:
            return attention_mask
        if attention_mask.ndim == 2:
            return valid.to(dtype=attention_mask.dtype)
        if attention_mask.dtype == torch.bool:
            return valid[:, None, None, :]
        minimum = torch.finfo(attention_mask.dtype).min
        return torch.where(
            valid[:, None, None, :],
            torch.zeros((), dtype=attention_mask.dtype, device=valid.device),
            torch.full((), minimum, dtype=attention_mask.dtype, device=valid.device),
        )
