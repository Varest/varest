"""Train the TEXT head (v5 architecture) on OCR positives WITH text-free negatives -- the released "v5n" head.

WHY NEGATIVES. A head trained only on OCR-positive images never sees a text-free image, so it fires confidently on
fabric, faces and bottle ridges, and the W tokenizers pin every cell it fires on. Adding text-free images at the
CORPUS rate buys that texture rejection at no measurable cost to detection. Only the data changes: architecture and
labels are v5's.

DATA (:func:`build_rows`).
  text-bearing pool  images with at least one cell in the ``positives`` file (:mod:`varest.train.ocr_labels`,
                     row-keyed, first row per image wins); labels are the padded-24-grid cells remapped onto the
                     image's own in-image cell grid (:func:`varest.heads.text.cell_shape`).
  text-free pool     images with NO detection at 672 AND 336 in the ``two-res`` dump, and not in the text pool.
  ratio              corpus text-free : text-bearing over the whole two-res dump.
  holdout            drawn FIRST with ``Random(seed + 999)``: ``round(holdout / (1 + ratio))`` text images, the rest
                     text-free; identical for both arms and excluded from both.
  train              ``ntext`` text images with ``Random(seed)``; ``round(ntext * ratio)`` negatives with
                     ``Random(seed + 1)`` (used by the ``neg`` arm only).
Each image is converted to grayscale and bicubic-resized so one cell is ``cell_px`` pixels at its own aspect.

TWO ARMS, ONE VARIABLE. ``neg`` (released) and ``ctl`` train on the identical text images; ``ctl`` omits the
negatives. ``pos_weight`` is computed from the TEXT images only, so the all-negative images do not inflate it; a
fixed number of optimizer steps is run and the FINAL weights are kept -- no selection, because selecting on a noisy
held-out metric makes stopping time a hidden variable.

HELD-OUT METRICS (per image, then averaged, since cells within an image are correlated): ``auc_text`` (cell AUC
inside text images), ``rec@t`` (share of OCR-positive cells with p > t), ``fp@t`` (share of cells with p > t on
text-free images) and ``anyfp@t`` (share of text-free images with any cell > t), for t in 0.5, 0.8, 0.9, 0.95, 0.972.

MODULE NOTE. The released weights were trained with the research ``PixelTextHead``, whose forward takes the per-cell
mean and variance by RESHAPING each cell's pixels; :class:`varest.heads.text.TextHead` takes them by average pooling
(Var = E[x^2] - E[x]^2). The two are mathematically identical, have the same parameters in the same construction
order (so the same seed gives the same initial weights), and differ only by float rounding (~1e-5 at the logit), so a
retrain here follows the released run up to that rounding rather than bit for bit.

usage: python -m varest.train.text_head --data-root <LLaVA-Tuning> --positives mix665k_easyocr_text.pkl \
           --negatives 'mix665k_ocr2res_*of30.pkl' --out texthead_v5n_neg_head.pt
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import pickle
import random
from collections.abc import Callable, Sequence
from multiprocessing import Pool
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from varest.heads.text import CELL_PX, TextHead, cell_shape, text_head_from_state

MIX_JSON = "llava_v1_5_mix665k.json"
GRID = 24
THRESHOLDS = (0.5, 0.8, 0.9, 0.95, 0.972)
SPLITS = ("tr_text", "tr_free", "h_text", "h_free")

Row = dict[str, Any]


# ---------------------------------------------------------------------------------------------------- data

def load_row(data_root: str, rel: str, cells: Sequence[tuple[int, int]] | None, cell_px: int = CELL_PX,
             grid: int = GRID) -> Row | None:
    """One image -> ``{"rel", "g": uint8[nr*cell_px, nc*cell_px], "y": f32[nr, nc], "text"}``, or None if unreadable.

    ``cells`` are padded-grid ``(r, c)`` positives (None marks a text-free image); cells on the letterbox are dropped.
    """
    try:
        with Image.open(os.path.join(data_root, rel)) as im:
            width, height = im.size
            r0, nr, c0, nc = cell_shape(width, height, grid=grid)
            g = im.convert("L").resize((nc * cell_px, nr * cell_px), Image.BICUBIC)
        g = np.asarray(g, dtype=np.uint8)
    except Exception:
        return None
    y = np.zeros((nr, nc), np.float32)
    for (r, c) in (cells or []):
        rr, cc = r - r0, c - c0
        if 0 <= rr < nr and 0 <= cc < nc:
            y[rr, cc] = 1
    return {"rel": rel, "g": g, "y": y, "text": cells is not None}


def _load_job(job: tuple[str, Sequence[tuple[int, int]] | None], data_root: str, cell_px: int) -> Row | None:
    return load_row(data_root, job[0], job[1], cell_px)


def _expand(patterns: Sequence[str]) -> list[str]:
    """Paths or globs -> the sorted file list (sorted, so the merge order never depends on the shell)."""
    out: set[str] = set()
    for p in patterns:
        hits = glob.glob(p)
        if not hits:
            raise FileNotFoundError(f"nothing matches {p!r}")
        out.update(hits)
    return sorted(out)


def text_pools(positives: dict[str, list], mix_rows: Sequence[dict], two_res: dict[str, dict]
               ) -> tuple[dict[str, list], list[str], float, int, int]:
    """-> ``(text {image: padded cells}, free [sorted images], ratio, n_free_all, n_text_all)``.

    ``positives`` is row-keyed; rows are visited in ascending row order so the FIRST row of an image supplies its
    cells, as in the released build (whose file was written in that order)."""
    text: dict[str, list] = {}
    for k in sorted(positives, key=int):
        v = positives[k]
        e = mix_rows[int(k)]
        if v and "image" in e and e["image"] not in text:
            text[e["image"]] = v
    nfree_all = sum(1 for v in two_res.values() if not v["d672"] and not v["d336"])
    ntext_all = len(two_res) - nfree_all
    free = sorted(k for k, v in two_res.items() if not v["d672"] and not v["d336"] and k not in text)
    return text, free, nfree_all / ntext_all, nfree_all, ntext_all


def draw_splits(text_keys: Sequence[str], free: Sequence[str], ratio: float, ntext: int = 20000,
                holdout: int = 3000, seed: int = 1234) -> dict[str, list[str]]:
    """The four image lists. Holdout first, from both pools at the corpus rate, with its own RNG."""
    tkeys = sorted(text_keys)
    rh = random.Random(seed + 999)
    h_text = rh.sample(tkeys, int(round(holdout / (1 + ratio))))
    h_free = rh.sample(list(free), holdout - len(h_text))
    hs = set(h_text) | set(h_free)
    tr_text = random.Random(seed).sample([k for k in tkeys if k not in hs], ntext)
    tr_free = random.Random(seed + 1).sample([k for k in free if k not in hs], int(round(ntext * ratio)))
    return {"tr_text": tr_text, "tr_free": tr_free, "h_text": h_text, "h_free": h_free}


def build_rows(data_root: str, positives: Sequence[str], negatives: Sequence[str], ntext: int = 20000,
               holdout: int = 3000, seed: int = 1234, cell_px: int = CELL_PX, workers: int = 8,
               mix_json: str = MIX_JSON) -> list[Row]:
    """Decode every image of every split into memory. Rows keep split order (tr_text, tr_free, h_text, h_free);
    unreadable images are dropped. Each row gains ``"split"`` and tensor ``"g"``/``"y"``."""
    with open(os.path.join(data_root, mix_json)) as f:
        mix_rows = json.load(f)
    pos: dict[str, list] = {}
    for p in _expand(positives):
        with open(p, "rb") as f:
            pos.update(pickle.load(f))
    two: dict[str, dict] = {}
    for p in _expand(negatives):
        with open(p, "rb") as f:
            two.update(pickle.load(f))
    text, free, ratio, nfree_all, ntext_all = text_pools(pos, mix_rows, two)
    print(f"  pools: {len(text):,} text-bearing images with labels; {len(free):,} text-free images "
          f"(none at 672 or 336); corpus text-free:text = {nfree_all:,}:{ntext_all:,} = {ratio:.3f}", flush=True)
    sp = draw_splits(list(text), free, ratio, ntext, holdout, seed)
    print(f"  holdout {len(sp['h_text'])} text + {len(sp['h_free'])} text-free; train {len(sp['tr_text'])} text + "
          f"{len(sp['tr_free'])} negatives (neg arm only)", flush=True)
    jobs = ([(k, text[k]) for k in sp["tr_text"]] + [(k, None) for k in sp["tr_free"]]
            + [(k, text[k]) for k in sp["h_text"]] + [(k, None) for k in sp["h_free"]])
    split = [s for s in SPLITS for _ in sp[s]]
    rows: list[Row] = []
    bad = 0
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap(functools.partial(_load_job, data_root=data_root, cell_px=cell_px),
                                        jobs, chunksize=64)):
            if r is None:
                bad += 1
                continue
            r["split"] = split[i]
            r["g"] = torch.from_numpy(r["g"])
            r["y"] = torch.from_numpy(r["y"])
            rows.append(r)
            if (i + 1) % 5000 == 0:
                print(f"    cached {i + 1}/{len(jobs)}", flush=True)
    print(f"  cached {len(rows)} rows, {bad} unreadable", flush=True)
    return rows


def split_indices(rows: Sequence[Row]) -> dict[str, list[int]]:
    return {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in SPLITS}


def positive_fraction(rows: Sequence[Row], idx: Sequence[int]) -> float:
    """Mean per-image share of positive cells over ``idx`` (the TEXT training images)."""
    return float(np.mean([float(rows[i]["y"].mean()) for i in idx]))


def buckets_of(rows: Sequence[Row], idx: Sequence[int]) -> dict[tuple[int, int], list[int]]:
    """Group by label shape: grids vary with aspect, and a batch must stack without padding."""
    b: dict[tuple[int, int], list[int]] = {}
    for i in idx:
        b.setdefault(tuple(rows[i]["y"].shape), []).append(i)
    return b


def _gray(rows: Sequence[Row], bi: Sequence[int], device: str | torch.device) -> torch.Tensor:
    return torch.stack([rows[i]["g"] for i in bi]).to(device).float().unsqueeze(1) / 255.0


# ---------------------------------------------------------------------------------------------------- metrics

@torch.no_grad()
def predict(head: nn.Module, rows: Sequence[Row], idx: Sequence[int], device: str | torch.device = "cpu",
            batch: int = 64) -> dict[int, np.ndarray]:
    head.eval()
    out: dict[int, np.ndarray] = {}
    for ii in buckets_of(rows, idx).values():
        for b in range(0, len(ii), batch):
            bi = ii[b:b + batch]
            p = torch.sigmoid(head(_gray(rows, bi, device))).float().cpu().numpy()
            for j, i in enumerate(bi):
                out[i] = p[j]
    return out


def auc(s: Any, y: Any) -> float:
    """Rank AUC with average ranks for ties; NaN when one class is absent."""
    s = np.asarray(s, np.float64)
    y = np.asarray(y, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    su = s[order]
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def holdout_metrics(head: nn.Module, rows: Sequence[Row], h_text: Sequence[int], h_free: Sequence[int],
                    device: str | torch.device = "cpu", thresholds: Sequence[float] = THRESHOLDS) -> dict[str, float]:
    p = predict(head, rows, list(h_text) + list(h_free), device)
    aucs = [a for a in (auc(p[i].ravel(), rows[i]["y"].numpy().ravel()) for i in h_text) if a == a]
    m: dict[str, float] = {"auc_text": float(np.mean(aucs)), "n_auc": len(aucs)}
    for t in thresholds:
        rec = [float((p[i].ravel()[rows[i]["y"].numpy().ravel() > 0] > t).mean()) for i in h_text
               if rows[i]["y"].sum() > 0]
        m[f"rec@{t}"] = float(np.mean(rec))
        m[f"fp@{t}"] = float(np.mean([(p[i] > t).mean() for i in h_free]))
        m[f"anyfp@{t}"] = float(np.mean([(p[i] > t).any() for i in h_free]))
    return m


def format_metrics(m: dict[str, float], thresholds: Sequence[float] = THRESHOLDS) -> str:
    return f"auc_text {m['auc_text']:.4f} | " + " ".join(
        f"t{t}: rec {m[f'rec@{t}']:.3f} fp {100 * m[f'fp@{t}']:.2f}% anyfp {100 * m[f'anyfp@{t}']:.1f}%"
        for t in thresholds)


# ---------------------------------------------------------------------------------------------------- training

def train_step(head: nn.Module, opt: torch.optim.Optimizer, lossf: nn.Module, g: torch.Tensor, y: torch.Tensor,
               clip: float = 1.0) -> float:
    """One AdamW step on a stacked batch (``g`` [B,1,H,W] in [0,1], ``y`` [B,nr,nc]); returns the loss."""
    loss = lossf(head(g), y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), clip)
    opt.step()
    return loss.item()


def train_arm(rows: Sequence[Row], by: dict[str, list[int]], arm: str = "neg", steps: int = 30000, batch: int = 16,
              seed: int = 1234, lr: float = 1e-3, weight_decay: float = 0.01, clip: float = 1.0,
              device: str | torch.device | None = None, eval_every_epoch: bool = True,
              log: Callable[[str], None] = print) -> tuple[TextHead, int, dict[str, float]]:
    """Train one arm for exactly ``steps`` optimizer steps; returns ``(head, epochs_run, final holdout metrics)``.

    Each epoch shuffles every shape bucket, cuts it into batches, and shuffles the batch list, all from one
    ``Random(seed)`` stream; the head is initialised right after ``torch.manual_seed(seed)``."""
    if arm not in ("neg", "ctl"):
        raise ValueError(f"arm must be neg or ctl, got {arm!r}")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pos = positive_fraction(rows, by["tr_text"])
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tr = by["tr_text"] + (by["tr_free"] if arm == "neg" else [])
    head = TextHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=dev))
    bk = buckets_of(rows, tr)
    log(f"### ARM {arm}: {len(tr)} training images ({len(by['tr_text'])} text"
        f"{' + ' + str(len(by['tr_free'])) + ' text-free' if arm == 'neg' else ''}), steps={steps}, "
        f"pos_weight {(1 - pos) / pos:.2f} (text images only), final weights kept")
    step, ep = 0, 0
    while step < steps:
        head.train()
        batches = []
        for idx in bk.values():
            idx = idx[:]
            rng.shuffle(idx)
            batches += [idx[b:b + batch] for b in range(0, len(idx), batch)]
        rng.shuffle(batches)
        tot, nb = 0.0, 0
        for bi in batches:
            if step >= steps:
                break
            y = torch.stack([rows[i]["y"] for i in bi]).to(dev)
            tot += train_step(head, opt, lossf, _gray(rows, bi, dev), y, clip)
            nb += 1
            step += 1
        ep += 1
        msg = f"  [{arm}] epoch {ep} step {step}/{steps} loss={tot / max(1, nb):.4f}"
        if eval_every_epoch and by["h_text"] and by["h_free"]:
            msg += f" | holdout {format_metrics(holdout_metrics(head, rows, by['h_text'], by['h_free'], dev))}"
        log(msg)
    m = holdout_metrics(head, rows, by["h_text"], by["h_free"], dev) if by["h_text"] and by["h_free"] else {}
    return head, ep, m


def save_text_head(path: str, head: TextHead, *, arm: str, steps: int, epochs_run: int, ntext: int, nneg: int,
                   metrics: dict[str, float], seed: int) -> None:
    """Write the blob :func:`varest.heads.text.text_head_from_state` loads (same keys as the released file)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"sd": head.state_dict(), "arm": arm, "steps": steps, "epochs_run": epochs_run, "ntext": ntext,
                "nneg": nneg, "holdout_metrics": metrics, "seed": seed, **head.config(), "ver": "v5",
                "selection": "none (final weights)"}, path)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m varest.train.text_head", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", required=True, help=f"directory holding {MIX_JSON} and the images")
    ap.add_argument("--positives", nargs="+", required=True, help="ocr_labels positives .pkl (paths or globs)")
    ap.add_argument("--negatives", nargs="+", required=True, help="ocr_labels two-res .pkl shards (paths or globs)")
    ap.add_argument("--out", required=True, help="output head .pt")
    ap.add_argument("--arm", choices=("neg", "ctl"), default="neg", help="neg = released; ctl = no negatives")
    ap.add_argument("--ntext", type=int, default=20000)
    ap.add_argument("--holdout", type=int, default=3000)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8, help="image decoding processes")
    ap.add_argument("--feat-cache", default=None, help="optional .pt of decoded rows (read if present, else written)")
    ap.add_argument("--ref-head", default=None, help="optional head to score on the same holdout, for reference")
    ap.add_argument("--mix-json", default=MIX_JSON)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if a.feat_cache and os.path.exists(a.feat_cache):
        rows = torch.load(a.feat_cache, map_location="cpu", weights_only=False)["rows"]
        print(f"  loaded cache {a.feat_cache}: {len(rows)} rows", flush=True)
    else:
        rows = build_rows(a.data_root, a.positives, a.negatives, a.ntext, a.holdout, a.seed, CELL_PX, a.workers,
                          a.mix_json)
        if a.feat_cache:
            torch.save({"rows": rows, "ntext": a.ntext, "holdout": a.holdout, "seed": a.seed, "cell_px": CELL_PX},
                       a.feat_cache)
            print(f"  wrote cache -> {a.feat_cache}", flush=True)
    by = split_indices(rows)
    print("  " + ", ".join(f"{s}={len(v)}" for s, v in by.items()), flush=True)
    if a.ref_head:
        ref = text_head_from_state(torch.load(a.ref_head, map_location="cpu", weights_only=False)).to(dev)
        print(f"  REFERENCE {os.path.basename(a.ref_head)}: "
              f"{format_metrics(holdout_metrics(ref, rows, by['h_text'], by['h_free'], dev))}", flush=True)

    head, ep, m = train_arm(rows, by, a.arm, a.steps, a.batch, a.seed, a.lr, a.weight_decay, a.clip, dev,
                            log=lambda s: print(s, flush=True))
    save_text_head(a.out, head, arm=a.arm, steps=a.steps, epochs_run=ep, ntext=len(by["tr_text"]),
                   nneg=len(by["tr_free"]) if a.arm == "neg" else 0, metrics=m, seed=a.seed)
    print(f"### FINAL [{a.arm}] {format_metrics(m)}\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

