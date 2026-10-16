"""Search a VFlowOpt pruning strategy for one backbone (the paper's Algorithm 1).

VFlowOpt's strategies are calibrated *per model*. Eq. 5 defines the budget as a
layer average over the three progressive-pruning stages, so a strategy solved
against one decoder depth does not describe the same budget on another:
Vicuna-7B's 32 layers split 10/10/12, LLaVA-1.5-13B's 40 split 13/13/14. The
paper is explicit that this is the method rather than an implementation detail,
criticising prior work for applying "the same strategy across different LMMs,
without considering the unique characteristics of each model".

:data:`~llmoe.compressor.vflowopt.vflowopt._STRATEGY_PARAMETERS` ships the fitted
LLaVA-1.5-7B parameters but not the search that produced them, so calibrating a
new backbone previously meant reusing another model's numbers. This module
supplies that outer loop. Everything inside the objective is the existing port:
the importance map (Eq. 1-4), token recycling (Eq. 6-7), :class:`SearchSpace`'s
bounds and clipping, and the Eq. 5 solver.

Algorithm 1, reproduced:

* Search ``(R1, R2, t, a, alpha)``; ``R3`` is *derived* from the budget
  constraint rather than searched (lines 5 and 12).
* Objective, Eq. 8: ``max f(s) = CosineSim(h_f, g_s(h_f))``, where ``h_f`` is the
  final token's hidden state in the last decoder layer -- summed over samples.
* Gaussian-Process surrogate with Expected Improvement.

Usage::

    python -m llmoe.compressor.vflowopt.calibrate \\
        --backbone llava-13b --budgets 32,64,128,224 \\
        --samples /path/to/llava_v1_5_mix665k.json \\
        --image-root /path/to/images --out ./strategies

Requires ``scikit-optimize``, which is not a runtime dependency of the package:
inference reads a calibrated strategy, only calibration searches for one.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from llmoe.backbone import BACKBONES
from llmoe.compressor.vflowopt.vflowopt import (
    SearchSpace, StageLayers, VFlowOptCompressor,
)
from llmoe.utils.depth import decoder_depth

#: Recorded in the "search" block of every released strategy JSON.
DEFAULT_N_CALLS = 50
DEFAULT_N_INITIAL_POINTS = 27
DEFAULT_RANDOM_STATE = 42
ACQ_FUNC = "EI"
BUDGET_MODE = "integer"
#: "we sample 30 unlabeled instances from each model's training datasets" (Sec 4.1).
DEFAULT_N_SAMPLES = 30
DEFAULT_SAMPLE_SEED = 42


def load_samples(mix_path: Path, image_root: Path, n_samples: int,
                 seed: int) -> list[tuple[Path, str]]:
    """`n_samples` (image, prompt) pairs drawn deterministically from a mix file.

    Entries whose image is not present under `image_root` are skipped rather than
    failing the run: a corpus is often distributed with its images in archives,
    and the calibration only needs a small deterministic sample of what is on
    disk. The seed is applied to the *filtered* list, so a given (corpus, root,
    seed) always yields the same sample.
    """
    entries = json.loads(Path(mix_path).read_text(encoding="utf-8"))
    root = Path(image_root)
    with_image = [e for e in entries if e.get("image") and (root / e["image"]).is_file()]
    if len(with_image) < n_samples:
        raise ValueError(f"{mix_path} has {len(with_image)} entries whose image "
                         f"resolves under {root}, fewer than the {n_samples} requested")
    picked = random.Random(seed).sample(with_image, n_samples)
    out = []
    for entry in picked:
        turn = next((c["value"] for c in entry["conversations"]
                     if c.get("from") == "human"), "")
        out.append((root / entry["image"], turn.replace("<image>", "").strip()))
    return out


@torch.no_grad()
def final_token_state(vlm: Any, image: Any, prompt: str) -> torch.Tensor:
    """``h_f``: the last token's hidden state in the last decoder layer."""
    inputs = vlm.prepare_inputs([image], [prompt])
    out = vlm.model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1][0, -1].float()


#: LLaVA-NeXT search space. `alpha` is absent because the entropy term of Eq. 4
#: is not implemented on the NeXT path -- importance there is pure attention
#: relevance, and upstream's CANONICAL_ALPHA is 0 for the same reason -- so a
#: fitted value would describe nothing. `a` is a cell side in patches on the
#: 27x27 reference grid the adapter normalises against.
NEXT_R2_BOUNDS = (0.2 / 0.375, 1.0)
NEXT_T_BOUNDS = (0.0, 50.0)
NEXT_A_BOUNDS = (3.0, 27.0)
NEXT_PARAM_NAMES = ("r1", "r2", "t", "a")


def next_bounds(target_ratio: float) -> list[tuple[float, float]]:
    """Box for ``(R1, R2, t, a)``; ``R3`` is derived from the Eq. 5 constraint."""
    return [(target_ratio, min(1.0, 1.5 * target_ratio)),
            NEXT_R2_BOUNDS, NEXT_T_BOUNDS, NEXT_A_BOUNDS]


def next_strategy_from_point(point, target_ratio: float, stages):
    """Resolve a searched point into a :class:`NextStrategy`.

    ``R3`` follows from Eq. 5 in *ratio* space, which is the right space here:
    NeXT budgets a fraction of a variable AnyRes span, so there is no fixed token
    count to solve against.
    """
    from llmoe.compressor.vflowopt.vflowopt import solve_r3_continuous
    from llmoe.compressor.vflowopt.vflowopt_llava_next import NextStrategy

    (r1_lo, r1_hi), (r2_lo, r2_hi), (t_lo, t_hi), (a_lo, a_hi) = next_bounds(target_ratio)
    r1 = min(max(float(point[0]), r1_lo), r1_hi)
    r2 = min(max(float(point[1]), r2_lo), r2_hi)
    t = min(max(float(point[2]), t_lo), t_hi)
    a = min(max(float(point[3]), a_lo), a_hi)
    r3 = solve_r3_continuous(r1, r2, target_ratio, stages)
    return NextStrategy(r1=r1, r2=r2, r3=max(r3, 1e-6), t=t, a=a)


def calibrate_next(vlm: Any, ratio: float, images: list, prompts: list[str],
                   reference: list[torch.Tensor], stages, *,
                   n_calls: int = DEFAULT_N_CALLS,
                   n_initial_points: int = DEFAULT_N_INITIAL_POINTS,
                   random_state: int = DEFAULT_RANDOM_STATE) -> tuple[Any, float, float, int]:
    """Search one retain_ratio on LLaVA-NeXT. Returns (strategy, best, seconds, skipped).

    Not every point is realisable: the crop-local allocator needs at least one
    cell per AnyRes region, so a large ``a`` with a small budget has no valid
    assignment. Those raise rather than scoring badly, so they are caught and
    given the worst possible objective -- the search must see them as bad points,
    not die on them.
    """
    from skopt import gp_minimize

    from llmoe.compressor.vflowopt.vflowopt import VFlowOptCompressor

    bounds = next_bounds(ratio)
    infeasible = 0
    first_error: str | None = None

    def objective(point: list[float]) -> float:
        nonlocal infeasible, first_error
        strategy = next_strategy_from_point(point, ratio, stages)
        compressor = VFlowOptCompressor(retain_ratio=ratio,
                                        strategy={"r1": strategy.r1, "r2": strategy.r2,
                                                  "r3": strategy.r3, "t": strategy.t,
                                                  "a": strategy.a})
        try:
            compressor.install(vlm)
        except Exception as error:  # noqa: BLE001 - recorded, then scored as bad
            infeasible += 1
            if first_error is None:
                first_error = f"install: {type(error).__name__}: {error}"
            return 0.0  # cosine similarity is bounded above by 1; 0 is a bad score
        try:
            total = 0.0
            for image, prompt, ref in zip(images, prompts, reference):
                got = final_token_state(vlm, image, prompt)
                total += float(torch.nn.functional.cosine_similarity(
                    ref.unsqueeze(0), got.unsqueeze(0)).item())
        except Exception as error:  # noqa: BLE001 - recorded, then scored as bad
            infeasible += 1
            if first_error is None:
                first_error = f"forward: {type(error).__name__}: {error}"
            return 0.0
        finally:
            uninstall = getattr(compressor, "uninstall", None)
            if uninstall is not None:
                uninstall()
        return -total

    started = time.perf_counter()
    result = gp_minimize(objective, bounds, n_calls=n_calls,
                         n_initial_points=n_initial_points,
                         acq_func=ACQ_FUNC, random_state=random_state)
    seconds = time.perf_counter() - started
    if infeasible >= n_calls:
        # Every point failed, so `result.x` is just wherever the optimiser stopped
        # and the objective never measured anything. Writing that out would look
        # like a calibration while describing nothing.
        raise RuntimeError(
            f"VFlowOpt/LLaVA-NeXT calibration at retain_ratio={ratio:g} evaluated "
            f"no feasible point in {n_calls} calls. First failure -- {first_error}"
        )
    best = next_strategy_from_point(result.x, ratio, stages)
    return best, -float(result.fun), seconds, infeasible


def calibrate(vlm: Any, budget: int, images: list, prompts: list[str],
              reference: list[torch.Tensor], stages: StageLayers, *,
              n_calls: int = DEFAULT_N_CALLS,
              n_initial_points: int = DEFAULT_N_INITIAL_POINTS,
              random_state: int = DEFAULT_RANDOM_STATE) -> tuple[Any, float, float]:
    """Search one budget. Returns (strategy, best similarity sum, seconds)."""
    from skopt import gp_minimize

    space = SearchSpace(budget_tokens=budget, stages=stages)
    bounds = [tuple(b) for b in space.bounds]

    def objective(point: list[float]) -> float:
        compressor = VFlowOptCompressor(strategy=space.resolve(point, budget_mode=BUDGET_MODE))
        compressor.install(vlm)
        try:
            total = 0.0
            for image, prompt, ref in zip(images, prompts, reference):
                got = final_token_state(vlm, image, prompt)
                total += float(torch.nn.functional.cosine_similarity(
                    ref.unsqueeze(0), got.unsqueeze(0)).item())
        finally:
            compressor.uninstall()
        return -total  # gp_minimize minimises; Eq. 8 maximises.

    started = time.perf_counter()
    result = gp_minimize(objective, bounds, n_calls=n_calls,
                         n_initial_points=n_initial_points,
                         acq_func=ACQ_FUNC, random_state=random_state)
    seconds = time.perf_counter() - started
    return space.resolve(result.x, budget_mode=BUDGET_MODE), -float(result.fun), seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", default="llava-13b", choices=sorted(BACKBONES))
    parser.add_argument("--budgets", default="32,64,128,224",
                        help="comma-separated visual-token budgets; on a "
                             "ratio-budgeted backbone each is read as N/576")
    parser.add_argument("--samples", required=True, type=Path,
                        help="LLaVA-style instruction mix JSON to draw calibration images from")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--n-calls", type=int, default=DEFAULT_N_CALLS)
    parser.add_argument("--n-initial-points", type=int, default=DEFAULT_N_INITIAL_POINTS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    args = parser.parse_args(argv)

    from PIL import Image

    budgets = tuple(int(b) for b in args.budgets.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.samples, args.image_root, args.n_samples, args.sample_seed)
    images = [Image.open(p).convert("RGB") for p, _ in samples]
    prompts = [q for _, q in samples]

    vlm = BACKBONES[args.backbone]()
    depth = decoder_depth(vlm)
    stages = StageLayers.even(depth)
    print(f"backbone={args.backbone} depth={depth} "
          f"stages={stages.l1}/{stages.l2}/{stages.l3} prune_layers={stages.prune_layers}",
          flush=True)

    # h_f does not depend on the strategy, so the reference pass runs once.
    reference = [final_token_state(vlm, im, q) for im, q in zip(images, prompts)]
    ref_norm = float(torch.stack(reference).norm(dim=-1).mean())

    # LLaVA-NeXT budgets a fraction of a variable AnyRes span, so its control is
    # a ratio, not a token count; the b<N> label is kept so rows line up.
    ratio_budgeted = args.backbone == "llava-next-8b"

    for budget in budgets:
        if ratio_budgeted:
            ratio = budget / 576
            strategy, best, seconds, infeasible = calibrate_next(
                vlm, ratio, images, prompts, reference, stages,
                n_calls=args.n_calls, n_initial_points=args.n_initial_points,
                random_state=args.random_state)
            realised = strategy.layer_average(stages)
            extra = {"retain_ratio": ratio,
                     "realised_layer_average_ratio": realised,
                     "searched": list(NEXT_PARAM_NAMES),
                     "derived": ["r3"],
                     "fixed": {"alpha": 0.0},
                     "infeasible_points": infeasible}
            strategy_dict = {"r1": strategy.r1, "r2": strategy.r2, "r3": strategy.r3,
                             "t": strategy.t, "a": strategy.a,
                             "stages": {"l1": stages.l1, "l2": stages.l2, "l3": stages.l3},
                             "prune_layers": list(stages.prune_layers)}
        else:
            strategy, best, seconds = calibrate(
                vlm, budget, images, prompts, reference, stages,
                n_calls=args.n_calls, n_initial_points=args.n_initial_points,
                random_state=args.random_state)
            extra = {}
            strategy_dict = strategy.to_dict()
        payload = {
            "budget_tokens": budget,
            **extra,
            "strategy": strategy_dict,
            "objective": {"best_similarity_sum": best,
                          "best_mean_similarity": best / len(samples),
                          "n_samples": len(samples),
                          "reference_hidden_state_norm": ref_norm},
            "search": {"n_calls": args.n_calls, "n_initial_points": args.n_initial_points,
                       "random_state": args.random_state, "acq_func": ACQ_FUNC,
                       "budget_mode": BUDGET_MODE, "seconds": seconds},
            "model": getattr(vlm, "_model_id", args.backbone),
            "entropy_mode": "reference",
            "calibration_samples": {"mix_path": str(args.samples),
                                    "image_root": str(args.image_root),
                                    "seed": args.sample_seed, "n_samples": len(samples)},
            "decoder": {"num_layers": depth, "stages": [stages.l1, stages.l2, stages.l3],
                        "prune_layers": list(stages.prune_layers)},
            "provenance": ("Searched with llmoe.compressor.vflowopt.calibrate, a "
                           "reimplementation of the paper's Algorithm 1 over this "
                           "package's Eq. 5 solver, search bounds and clipping. The "
                           "released LLaVA-1.5-7B strategies were produced by upstream "
                           "code that is not part of this repository."),
        }
        path = args.out / f"strategy_b{budget}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        shown = (f"layer_avg_ratio={strategy.layer_average(stages):.4f}"
                 if ratio_budgeted else f"layer_avg={strategy.layer_average:.2f}")
        print(f"b{budget}: mean_cos={best / len(samples):.4f} {shown} "
              f"{seconds:.0f}s -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
