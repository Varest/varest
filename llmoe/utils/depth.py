"""Decoder-depth arithmetic shared by every layer-scheduled compressor.

Almost every decoder-stage baseline here reports its budget as a **layer
average** -- ``sum(visual tokens held by each decoder layer) / depth`` -- rather
than as a final retained count, because that is the quantity the papers tabulate
and the one that tracks attention FLOPs and KV-cache pressure. A published
schedule is therefore only meaningful together with the depth it was calibrated
at, which for every LLaVA-1.5 profile in this repo is Vicuna-7B's 32 layers
(:data:`REFERENCE_DEPTH`).

Run that schedule unchanged on a deeper decoder -- LLaVA-1.5-13B has 40 layers --
and the label silently stops describing the run: the layers after the last
boundary outnumber the ones before it differently, so a "64-token" FastV
schedule actually spends 53.4. This module carries a schedule across depths
instead, in the two steps the Qwen and InternVL ports here already use:

1. **Scale the boundaries.** A boundary is a *fraction of the way through the
   decoder*, not an absolute layer, so layer 16 of 32 becomes layer 20 of 40
   (:func:`scale_layer`). This preserves the shape of the published schedule --
   which stage sees how much of the decoder -- and is what makes "the middle
   layer" well defined on any backbone.
2. **Re-solve the keeps to restore the label.** Scaling alone does not preserve
   the average, so the keep curve is interpolated between its published shape
   and "prune nothing" by a single scalar ``t``, chosen so the layer average
   comes back to exactly what it was. ``t`` enters the average linearly, so this
   is one division, and at ``depth == reference`` it solves to ``t == 1`` --
   re-resolving a schedule at its own depth returns it unchanged, so no existing
   LLaVA-1.5-7B number moves.

Everything is exact rational arithmetic (:class:`fractions.Fraction`) so that
identity holds bit-for-bit rather than to within a float epsilon.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

# Vicuna-7B's decoder and LLaVA-1.5's CLIP-336 patch grid: the geometry every
# published LLaVA profile in this repo is stated against.
REFERENCE_DEPTH: int = 32
REFERENCE_VISUAL_TOKENS: int = 576


def decoder_depth(vlm: Any) -> int:
    """Decoder layers in a loaded backbone.

    Accepts a :class:`~llmoe.backbone.prototype.VLM`, the ``PreTrainedModel`` it
    wraps, or the inner ``model``, so callers can pass whichever they hold at the
    point they need the depth. Counts the modules rather than trusting
    ``num_hidden_layers``: a compressor that swaps the decoder (HiDrop, DUET)
    installs against the stack that is actually there.
    """
    for candidate in (vlm, getattr(vlm, "model", None)):
        if candidate is None:
            continue
        inner = getattr(candidate, "model", candidate)
        layers = getattr(getattr(inner, "language_model", None), "layers", None)
        if layers is not None:
            return len(layers)
    raise TypeError(
        "Could not find a decoder stack at `model.model.language_model.layers` on "
        f"{type(vlm).__name__}; a layer-averaged schedule cannot be resolved without the depth."
    )


def config_decoder_depth(vlm: Any) -> int:
    """Decoder layers of a backbone that has *not* been loaded yet.

    :func:`decoder_depth` counts modules and so needs the weights in memory. The
    trainers need the depth before that -- the schedule has to be resolved once,
    up front, and the model is then built from it -- so this reads
    ``num_hidden_layers`` off the checkpoint's config instead, which is a small
    JSON rather than a multi-gigabyte download.
    """
    from transformers import AutoConfig

    model_id = getattr(vlm, "_model_id", vlm)
    config = AutoConfig.from_pretrained(
        str(model_id), local_files_only=bool(getattr(vlm, "_local_files_only", True))
    )
    text_config = getattr(config, "text_config", config)
    depth = getattr(text_config, "num_hidden_layers", None)
    if depth is None:
        raise ValueError(
            f"Could not read `num_hidden_layers` from {model_id!r}. A layer-averaged "
            f"schedule is depth-relative and cannot be resolved without it."
        )
    return int(depth)


def scale_layer(layer: int, depth: int, *, reference: int = REFERENCE_DEPTH) -> int:
    """Move a layer index from a `reference`-deep decoder onto a `depth`-deep one.

    Proportional and round-half-up, so the boundary keeps its position *as a
    fraction of the decoder*. The index convention is the caller's -- this is
    arithmetic on the number alone -- but every schedule here is 0-indexed on
    "the first layer that runs reduced".
    """
    return int(Fraction(int(layer) * int(depth), int(reference)) + Fraction(1, 2))


def scale_layers(layers: Sequence[int], depth: int, *, reference: int = REFERENCE_DEPTH,
                 minimum: int = 1, limit: int | None = None,
                 label: str = "the schedule") -> tuple[int, ...]:
    """:func:`scale_layer` over a whole boundary list, kept strictly increasing.

    Rounding can push two boundaries onto the same layer, or the first one below
    ``minimum``, on a decoder much shallower than the reference. Silently
    accepting either would drop a pruning stage or leave no layer holding the
    full visual span -- a different method, not a re-targeted one. Each boundary
    is therefore raised to the least layer that keeps the sequence strictly
    increasing and at or above ``minimum``, which moves it by at most a layer,
    and it is an error only when the decoder has too few layers to hold them all.

    :param minimum: lowest layer the first boundary may take. ``1`` leaves one
        unpruned layer; HiDrop passes one past its late-entry layer.
    :param limit: one past the highest usable layer -- the decoder depth, or
        HiDrop's early exit. Defaults to ``depth``.
    """
    limit = depth if limit is None else int(limit)
    scaled: list[int] = []
    floor = int(minimum)
    for layer in layers:
        boundary = max(floor, scale_layer(layer, depth, reference=reference))
        scaled.append(boundary)
        floor = boundary + 1
    if scaled and scaled[-1] >= limit:
        raise ValueError(
            f"{label} boundaries {tuple(int(x) for x in layers)} do not fit a {depth}-layer "
            f"decoder: they scale to {tuple(scaled)}, which runs past layer {limit - 1}. This "
            f"decoder is too shallow for the schedule -- pass boundaries built for its depth."
        )
    return tuple(scaled)


@dataclass(frozen=True)
class LayerSchedule:
    """A visual-token schedule over one decoder, and its layer-averaged budget.

    :param boundaries: 0-indexed layers that first run reduced, strictly
        increasing. Layers before the first hold the full visual span.
    :param keeps: tokens surviving each boundary, one per boundary and
        non-increasing. Each is held until the next boundary, the last to the end
        of the stack.
    :param depth: decoder layers this schedule is stated against.
    :param n_visual: visual tokens entering the decoder. Pass ``1`` for a
        schedule written in *fractions* of a per-image span, as the Qwen ports
        must: the arithmetic is identical and the average comes back as a ratio.
    :param entry: first layer holding visual tokens at all -- HiDrop's late
        entry, ``0`` everywhere else.
    :param exit_layer: one past the last layer holding visual tokens -- HiDrop's
        early exit, the depth everywhere else.
    """

    boundaries: tuple[int, ...]
    keeps: tuple[float, ...]
    depth: int
    n_visual: float = REFERENCE_VISUAL_TOKENS
    entry: int = 0
    exit_layer: int | None = None

    def __post_init__(self) -> None:
        pin = object.__setattr__
        pin(self, "boundaries", tuple(int(layer) for layer in self.boundaries))
        pin(self, "keeps", tuple(self.keeps))
        pin(self, "depth", int(self.depth))
        pin(self, "entry", int(self.entry))
        pin(self, "exit_layer", self.depth if self.exit_layer is None else int(self.exit_layer))
        if len(self.boundaries) != len(self.keeps):
            raise ValueError(
                f"one keep per boundary is required, got {len(self.boundaries)} boundaries and "
                f"{len(self.keeps)} keeps"
            )
        if list(self.boundaries) != sorted(set(self.boundaries)):
            raise ValueError(f"boundaries must be strictly increasing, got {self.boundaries}")
        if not 0 <= self.entry < self.exit_layer <= self.depth:
            raise ValueError(
                f"window [{self.entry}, {self.exit_layer}) does not fit a {self.depth}-layer decoder"
            )
        if self.boundaries and not (self.entry <= self.boundaries[0]
                                    and self.boundaries[-1] < self.exit_layer):
            raise ValueError(
                f"boundaries {self.boundaries} fall outside the visual window "
                f"[{self.entry}, {self.exit_layer})"
            )

    # ------------------------------------------------------------------ shape

    def stage_weights(self) -> tuple[int, tuple[int, ...]]:
        """``(layers holding the full span, layers holding each stage's keep)``.

        Only layers inside the window count: outside it the schedule holds
        nothing, so those layers contribute zero to the average whatever the
        keeps are.
        """
        stage_of_layer = {layer: index for index, layer in enumerate(self.boundaries)}
        leading = 0
        per_stage = [0] * len(self.boundaries)
        stage = -1
        for layer in range(self.entry, self.exit_layer):
            if layer in stage_of_layer:
                stage = stage_of_layer[layer]
            if stage < 0:
                leading += 1
            else:
                per_stage[stage] += 1
        return leading, tuple(per_stage)

    def counts_by_layer(self) -> list[float]:
        """Visual tokens held by each 0-indexed decoder layer, zero outside the window."""
        keep_at = dict(zip(self.boundaries, self.keeps))
        counts: list[float] = []
        current: float = self.n_visual
        for layer in range(self.depth):
            current = keep_at.get(layer, current)
            counts.append(0 if layer < self.entry or layer >= self.exit_layer else current)
        return counts

    # ----------------------------------------------------------------- budget

    @property
    def exact_layer_average(self) -> Fraction:
        """:attr:`layer_average` without float rounding, for chained arithmetic."""
        leading, per_stage = self.stage_weights()
        total = Fraction(leading) * Fraction(self.n_visual)
        for width, keep in zip(per_stage, self.keeps):
            total += Fraction(width) * Fraction(keep)
        return total / self.depth

    @property
    def layer_average(self) -> float:
        """The reported budget: visual tokens averaged over every decoder layer."""
        return float(self.exact_layer_average)

    @property
    def exact_ceiling(self) -> Fraction:
        """Highest average this window can reach, i.e. pruning nothing inside it."""
        leading, per_stage = self.stage_weights()
        return Fraction(leading + sum(per_stage)) * Fraction(self.n_visual) / self.depth

    @property
    def ceiling(self) -> float:
        return float(self.exact_ceiling)

    def floor(self, min_keep: float = 1) -> float:
        """Lowest average this stage division can reach.

        The layers before the first boundary always hold the full span and every
        stage keeps at least ``min_keep``, so no choice of keeps averages below
        this -- it is the lowest budget label the division can serve.
        """
        leading, per_stage = self.stage_weights()
        total = (Fraction(leading) * Fraction(self.n_visual)
                 + Fraction(sum(per_stage)) * Fraction(min_keep))
        return float(total / self.depth)

    # ------------------------------------------------------------- retargeting

    def rescaled_to(self, depth: int, *, min_keep: float = 1, integral: bool = True,
                    label: str = "the schedule") -> "LayerSchedule":
        """The same schedule on a `depth`-layer decoder, at the same layer average.

        Boundaries and window move proportionally; the keep curve is then
        interpolated toward "prune nothing" by the single scalar that restores
        the average (see the module docstring). Returns ``self`` when the depth
        already matches, so a backbone at the reference depth is untouched.

        :param min_keep: floor on a stage's keep -- ``1`` for schedules that must
            leave a token alive at every stage, ``0`` where a stage may drop the
            visual span entirely (DUET's terminal stage).
        :param integral: round the solved keeps to whole tokens. ``False`` leaves
            them rational, which is what a ratio-valued schedule wants.
        """
        depth = int(depth)
        if depth == self.depth:
            return self
        entry = scale_layer(self.entry, depth, reference=self.depth)
        exit_layer = min(depth, scale_layer(self.exit_layer, depth, reference=self.depth))
        if entry >= exit_layer - 1:
            raise ValueError(
                f"{label} does not fit a {depth}-layer decoder: its visual window scales to "
                f"[{entry}, {exit_layer}), which is too narrow to hold a schedule."
            )
        scaled = LayerSchedule(
            # At least one layer inside the window must hold the full span before
            # anything is cut, so the first boundary starts one past the entry.
            boundaries=scale_layers(self.boundaries, depth, reference=self.depth,
                                    minimum=entry + 1, limit=exit_layer, label=label),
            keeps=self.keeps,
            depth=depth,
            n_visual=self.n_visual,
            entry=entry,
            exit_layer=exit_layer,
        )
        target = self.exact_layer_average
        leading, per_stage = scaled.stage_weights()
        span = Fraction(self.n_visual)
        gaps = [span - Fraction(keep) for keep in self.keeps]
        if all(gap <= 0 for gap in gaps):  # nothing is pruned inside the window
            return scaled
        if target < Fraction(scaled.floor(min_keep)).limit_denominator():
            raise ValueError(
                f"{label} budget of {float(target):g} is unreachable on a {depth}-layer decoder: "
                f"its {leading} unpruned layer(s) alone average {scaled.floor(min_keep):g} even "
                f"with every stage cut to {min_keep}. Pass boundaries built for this depth."
            )
        solved = _solve_keeps(
            target_mass=target * depth,
            leading_mass=Fraction(leading) * span,
            widths=per_stage,
            gaps=gaps,
            span=span,
            low=Fraction(min_keep),
        )
        keeps = tuple(_round_half_up(keep) if integral else keep for keep in solved)
        return LayerSchedule(scaled.boundaries, keeps, depth, self.n_visual,
                             scaled.entry, scaled.exit_layer)


def _solve_keeps(*, target_mass: Fraction, leading_mass: Fraction, widths: Sequence[int],
                 gaps: Sequence[Fraction], span: Fraction, low: Fraction) -> list[Fraction]:
    """Keeps whose weighted total hits `target_mass`, interpolated along `gaps`.

    Every stage moves along its own gap by a shared ``t`` -- ``keep_i = span - t *
    gap_i`` -- which is one linear solve, except that a stage cannot be cut below
    ``low`` or raised above ``span``. Because the gaps are non-decreasing, the
    stages that hit a bound are always a contiguous end of the list, so pinning
    them and re-solving over the rest converges in at most one pass per stage and
    leaves the keeps non-increasing. This is what lets a deep decoder still spend
    its label after the shallow stages have bottomed out at a single token.
    """
    pinned: dict[int, Fraction] = {}
    free = [index for index, gap in enumerate(gaps) if gap > 0]
    solved: dict[int, Fraction] = {}
    while free:
        denominator = sum(Fraction(widths[index]) * gaps[index] for index in free)
        if denominator <= 0:
            break
        pinned_mass = sum(Fraction(widths[index]) * keep for index, keep in pinned.items())
        free_mass = sum(Fraction(widths[index]) for index in free) * span
        t = (leading_mass + pinned_mass + free_mass - target_mass) / denominator
        solved = {index: span - t * gaps[index] for index in free}
        out_of_range = {
            index: (span if keep > span else low)
            for index, keep in solved.items()
            if keep > span or keep < low
        }
        if not out_of_range:
            break
        pinned.update(out_of_range)
        free = [index for index in free if index not in pinned]
    return [pinned.get(index, solved.get(index, span - gap))
            for index, gap in enumerate(gaps)]


def _round_half_up(value: Fraction) -> int:
    return int(value + Fraction(1, 2))
