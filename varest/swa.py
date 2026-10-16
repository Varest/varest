"""SWA5: average the last snapshots of a training run into the released weights.

The released W5 / W5RELJOINT checkpoints are the uniform average of five checkpoints of one run -- the last four
32k-step snapshots before the final step, and the final weights: 512k, 544k, 576k, 608k + 624,609 on LLaVA-1.5's 665K
mix, 640k, 672k, 704k, 736k + 738,590 on LLaVA-NeXT-Data (``swa_steps``). Every trained component is averaged: the LoRA, the
position tag, the multimodal projector and, for W5RELJOINT, the [REL] head. Tensors are accumulated in float32 and cast
back to their stored dtype; the training config must be identical in every source.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from os import PathLike

import torch

AVERAGED = ("lora", "postag", "proj", "relhead")


def swa_steps(total_steps: int, every: int = 32000, n: int = 4) -> tuple[int, ...]:
    """The last ``n`` snapshot steps (multiples of ``every``) strictly before ``total_steps``, in increasing order."""
    top = ((total_steps - 1) // every) * every
    if top - every * (n - 1) <= 0:
        raise ValueError(f"a {total_steps}-step run has fewer than {n} snapshots every {every} steps")
    return tuple(top - every * i for i in range(n - 1, -1, -1))


SWA_STEPS = swa_steps(624609)          # the LLaVA-1.5 releases (one pass over the 665K mix)


def average(sources: Sequence[dict]) -> dict:
    """Average trainer checkpoints ``{"lora", "postag", "proj", ["relhead"], "cfg", "step"}`` -> one checkpoint
    (``step = -1``)."""
    if len(sources) < 2:
        raise ValueError("SWA needs at least two checkpoints")
    s0 = sources[0]
    if any(s.get("cfg") != s0.get("cfg") for s in sources[1:]):
        raise ValueError("sources disagree on cfg -- they are not checkpoints of one run")
    out = {"cfg": s0.get("cfg"), "step": -1}
    for key in AVERAGED:
        present = [key in s and s[key] is not None for s in sources]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"{key!r} is missing from some sources")
        keys = set(s0[key])
        if any(set(s[key]) != keys for s in sources[1:]):
            raise ValueError(f"sources disagree on the tensor set of {key!r}")
        out[key] = {k: (sum(s[key][k].float() for s in sources) / len(sources)).to(s0[key][k].dtype)
                    for k in s0[key]}
    extra = sorted(set(s0) - set(out) - {"step"})
    if extra:
        raise ValueError(f"sources carry components SWA does not cover: {extra}")
    return out


def average_files(paths: Sequence[str | PathLike[str]]) -> dict:
    return average([torch.load(p, map_location="cpu", weights_only=False) for p in paths])


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Average training snapshots (SWA).")
    ap.add_argument("out")
    ap.add_argument("sources", nargs="+")
    a = ap.parse_args(argv)
    ckpt = average_files(a.sources)
    torch.save(ckpt, a.out)
    print(f"averaged {', '.join(k for k in AVERAGED if k in ckpt)} over {len(a.sources)} checkpoints -> {a.out}")


if __name__ == "__main__":
    main()
