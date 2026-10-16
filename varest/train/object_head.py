"""Train the OBJECT head: regress DINOv3 edge distances at the teacher's NATIVE grid (released: w96h288).

NO RESAMPLING. The student sees the raw RGB image bicubic-resized to exactly the teacher's input (``n_c*16 x n_r*16``,
long side 768) and predicts on the teacher's own patch grid, so the dumped edge map (:mod:`varest.train.object_targets`)
IS the target (:func:`targets_native`); only the two structural zeros -- no right edge in the last column, no down edge
in the last row -- are masked. Loss is masked L1: robust to the long right tail of strong edges.

SELECTION. The validation metric is the Spearman rank correlation of predicted and teacher edges over every unmasked
held-out edge (:func:`val_rho`), because the tokenizer only ever COMPARES boundary strengths: a student with the right
ordering and the wrong scale partitions identically. Every epoch that improves it by more than 1e-5 is written to
``--out`` at once (atomically), and at the end the best epoch's weights are restored and written again; so ``--out``
always holds the BEST weights, with ``best_epoch``/``best_val_rho`` recorded. The released run used a 100-epoch cosine
anneal whose rho improved every epoch, so its best epoch is also its last.

DATA. Only the teacher edges live in RAM; images are decoded by DataLoader workers, because at 768 px RGB the corpus
does not fit in memory and training is filesystem-bound, not GPU-bound. Batches come from one ``(n_r, n_c)`` bucket
each (grids vary with aspect) via :class:`BucketBatches`.

SEGMENTS. A 100-epoch run does not fit one 12-hour job. ``--wall-seconds`` stops cleanly after the epoch that crosses
it, and ``--state-path`` (default ``<out>.state.pt``) carries model, optimiser, SCHEDULER and best-so-far, so the
cosine anneals over the full planned ``--epochs`` however the wall clock is split; resuming under a different
``--epochs`` is refused. The Python RNG that orders batches is re-seeded at the start of every segment (as in the
released run), so the batch order after a resume depends on where segments were cut.

usage: python -m varest.train.object_head --data-root <LLaVA-Tuning> \
           --targets 'mix665k_dinoedge_v3vitl16_*of20.pkl' --out object_head.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import random
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from varest.heads.object import CELL_PX, DILS, HID, WIDTH, ObjectHead

TARGET = "dinov3_edge_distance_native48"

Store = dict[str, tuple[np.ndarray, tuple[int, int]]]


def targets_native(edges: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher ``[2, nr, nc]`` -> ``(targets, weights)`` with NO resampling; weights mask the structural zeros."""
    t = torch.as_tensor(np.asarray(edges), dtype=torch.float32).clone()
    w = torch.ones_like(t)
    w[0, :, -1] = 0.0                                         # no right edge in the last column
    w[1, -1, :] = 0.0                                         # no down edge in the last row
    t[0, :, -1] = 0.0
    t[1, -1, :] = 0.0
    return t, w


def _rank(v: np.ndarray) -> np.ndarray:
    """1-based ranks, ties averaged."""
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), np.float64)
    r[order] = np.arange(1, len(v) + 1)
    vs = v[order]
    i = 0
    while i < len(vs):
        j = i
        while j + 1 < len(vs) and vs[j + 1] == vs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r


def val_rho(pred: Any, targ: Any, cap: int = 2_000_000) -> float:
    """Spearman rank correlation; above ``cap`` pairs a fixed subsample (``default_rng(0)``) keeps it tractable."""
    p = np.asarray(pred, np.float64).ravel()
    t = np.asarray(targ, np.float64).ravel()
    if len(p) > cap:
        idx = np.random.default_rng(0).choice(len(p), cap, replace=False)
        p, t = p[idx], t[idx]
    if len(p) < 2:
        return float("nan")
    rp, rt = _rank(p), _rank(t)
    rp -= rp.mean()
    rt -= rt.mean()
    den = float(np.sqrt((rp ** 2).sum() * (rt ** 2).sum()))
    return float((rp * rt).sum() / den) if den > 0 else float("nan")


class EdgeSet(Dataset):
    """Index -> ``(rgb [3, nr*cell_px, nc*cell_px] in [0,1], targets [2,nr,nc], weights [2,nr,nc])``."""

    def __init__(self, keys: Sequence[str], store: Store, data_root: str, cell_px: int = CELL_PX) -> None:
        self.keys, self.store, self.data_root, self.cell_px = keys, store, data_root, cell_px

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.keys[i]
        e, _wh = self.store[k]
        nr, nc = e.shape[1], e.shape[2]
        with Image.open(os.path.join(self.data_root, k)) as im:
            rgb = im.convert("RGB").resize((nc * self.cell_px, nr * self.cell_px), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(rgb, np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        t, w = targets_native(e)
        return x, t, w


class BucketBatches(Sampler):
    """Batches of dataset indices drawn from ONE ``(n_r, n_c)`` bucket each, so they stack without padding.

    Buckets keep the order of ``indices``; with ``shuffle`` each pass shuffles every bucket, cuts it, and shuffles the
    batch list, all from ``rng``."""

    def __init__(self, indices: Sequence[int], shapes: Sequence[tuple[int, int]], batch_size: int,
                 shuffle: bool = True, rng: random.Random | None = None) -> None:
        self.buckets: dict[tuple[int, int], list[int]] = {}
        for i in indices:
            self.buckets.setdefault(tuple(shapes[i]), []).append(i)
        self.bs, self.shuffle = batch_size, shuffle
        self.rng = rng if rng is not None else random.Random()

    def __iter__(self) -> Iterator[list[int]]:
        out = []
        for idx in self.buckets.values():
            idx = idx[:]
            if self.shuffle:
                self.rng.shuffle(idx)
            out += [idx[b:b + self.bs] for b in range(0, len(idx), self.bs)]
        if self.shuffle:
            self.rng.shuffle(out)
        return iter(out)

    def __len__(self) -> int:
        return sum((len(v) + self.bs - 1) // self.bs for v in self.buckets.values())


def load_targets(paths: Sequence[str], data_root: str, ntrain: int = 341000,
                 seed: int = 1234) -> tuple[list[str], Store]:
    """Teacher shards (paths or globs; merged in SORTED path order) -> ``(keys, store)`` of the training images.

    Keys are shuffled with ``Random(seed)``, cut to ``ntrain``, then images missing from ``data_root`` are dropped."""
    files: set[str] = set()
    for p in paths:
        files.update(glob.glob(p))
    shards = sorted(files)
    if not shards:
        raise FileNotFoundError(f"no teacher shards match {list(paths)}")
    store: Store = {}
    keys: list[str] = []
    for sh in shards:
        with open(sh, "rb") as f:
            d = pickle.load(f)
        bad = 0
        for k, v in d.items():
            if not (isinstance(v, tuple) and len(v) == 2 and getattr(v[0], "ndim", 0) == 3 and v[0].shape[0] == 2):
                bad += 1
                continue
            store[k] = v
            keys.append(k)
        if bad:
            raise ValueError(f"{os.path.basename(sh)}: {bad} entries are not ([2,nr,nc] edges, (W,H))")
        print(f"    {os.path.basename(sh)}: {len(d)} images (total {len(store)})", flush=True)
        del d
    random.Random(seed).shuffle(keys)
    keys = [k for k in keys[:ntrain] if os.path.exists(os.path.join(data_root, k))]
    store = {k: store[k] for k in keys}
    print(f"  {len(shards)} shards | {len(keys)} training images", flush=True)
    return keys, store


def split_train_val(n: int, val_frac: float = 0.05, seed: int = 1234) -> tuple[list[int], list[int]]:
    """-> ``(train, val)`` image indices: ``Random(seed + 777)`` shuffle, the first ``round(val_frac*n)`` held out."""
    idx = list(range(n))
    random.Random(seed + 777).shuffle(idx)
    nval = int(round(val_frac * n))
    return idx[nval:], idx[:nval]


def make_loss(name: str = "l1", huber_delta: float = 0.1) -> nn.Module:
    """Per-element loss (``reduction="none"``; the caller masks). l1 is the released recipe."""
    if name == "l1":
        return nn.L1Loss(reduction="none")
    if name == "l2":
        return nn.MSELoss(reduction="none")
    if name == "huber":
        return nn.HuberLoss(reduction="none", delta=huber_delta)
    raise ValueError(f"loss {name!r} not recognised; use l1, l2 or huber")


def masked_loss(lossf: nn.Module, o: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (lossf(o, t) * w).sum() / w.sum().clamp_min(1.0)


def train_step(head: nn.Module, opt: torch.optim.Optimizer, lossf: nn.Module, x: torch.Tensor, t: torch.Tensor,
               w: torch.Tensor, clip: float = 1.0) -> float:
    """One AdamW step on a batch already on the device; returns the masked loss."""
    loss = masked_loss(lossf, head(x), t, w)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), clip)
    opt.step()
    return loss.item()


@torch.no_grad()
def validate(head: nn.Module, batches: Any, lossf: nn.Module, device: str | torch.device) -> tuple[float, float]:
    """-> ``(mean batch loss, val_rho)`` over the unmasked edges of ``batches`` (an iterable of ``(x, t, w)``)."""
    head.eval()
    ps, ys, vlo, vn = [], [], 0.0, 0
    for x, t, w in batches:
        x, t, w = x.to(device), t.to(device), w.to(device)
        o = head(x)
        vlo += float(masked_loss(lossf, o, t, w))
        vn += 1
        m = w.bool().cpu().numpy().ravel()
        ps.append(o.float().cpu().numpy().ravel()[m])
        ys.append(t.cpu().numpy().ravel()[m])
    return vlo / max(1, vn), val_rho(np.concatenate(ps), np.concatenate(ys))


def save_object_head(path: str, sd: dict[str, torch.Tensor], head: ObjectHead, *, ntrain: int, seed: int,
                     best_epoch: int, best_val_rho: float) -> None:
    """Atomic write (temp file beside ``path``, then rename) of the blob
    :func:`varest.heads.object.object_head_from_state` loads."""
    blob = {"sd": sd, "ntrain": ntrain, "seed": seed, "cell_px": head.cell_px, "width": head.width, "hid": head.hid,
            "dils": head.dils, "rgb": head.rgb, "use_norm": bool(head.use_norm), "best_epoch": best_epoch,
            "best_val_rho": best_val_rho, "target": TARGET}
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".pt")
    os.close(fd)
    torch.save(blob, tmp)
    os.replace(tmp, path)


def default_state_path(out: str) -> str:
    return out[:-3] + ".state.pt" if out.endswith(".pt") else out + ".state.pt"


def train(keys: Sequence[str], store: Store, data_root: str, out: str, *, state_path: str | None = None,
          epochs: int = 100, min_epochs: int = 100, patience: int = 100, batch: int = 16, lr: float = 1e-3,
          weight_decay: float = 0.01, clip: float = 1.0, lr_sched: str = "cosine", loss: str = "l1",
          huber_delta: float = 0.1, val_frac: float = 0.05, seed: int = 1234, width: int = WIDTH, hid: int = HID,
          dils: Sequence[int] = DILS, cell_px: int = CELL_PX, rgb: bool = True, use_norm: bool = False,
          workers: int = 44, prefetch_factor: int = 6, wall_seconds: float = 0,
          device: str | torch.device | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Train (or resume) one run; writes ``out`` (best weights) and the resumable state. Returns ``best``
    (``{"rho", "epoch", "sd"}``)."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = torch.device(dev).type == "cuda"
    torch.manual_seed(seed)
    rng = random.Random(seed)                                 # orders the training batches
    shapes = [tuple(store[k][0].shape[1:]) for k in keys]
    ti, vi = split_train_val(len(keys), val_frac, seed)
    log(f"  split: {len(ti)} train / {len(vi)} val images")

    ds = EdgeSet(keys, store, data_root, cell_px)
    kw: dict[str, Any] = {"pin_memory": cuda}
    if workers > 0:                                           # persistent: each worker holds a copy of `store`
        kw.update(persistent_workers=True, prefetch_factor=prefetch_factor)
    tl = DataLoader(ds, batch_sampler=BucketBatches(ti, shapes, batch, True, rng), num_workers=workers, **kw)
    vl = DataLoader(ds, batch_sampler=BucketBatches(vi, shapes, batch, False),
                    num_workers=max(2, workers // 2) if workers > 0 else 0, **kw)

    head = ObjectHead(width=width, hid=hid, dils=tuple(dils), cell_px=cell_px, rgb=rgb, use_norm=use_norm).to(dev)
    gflop = float("nan")
    try:
        from torch.utils.flop_counter import FlopCounterMode
        ref = torch.zeros(1, 3 if head.rgb else 1, 48 * cell_px, 48 * cell_px, device=dev)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            head(ref)
        gflop = fc.get_total_flops() / 1e9
    except Exception as e:                                    # accounting must never kill a run
        log(f"  (FLOP count unavailable: {e})")
    log(f"  object head: params={sum(p.numel() for p in head.parameters()) / 1e6:.3f}M {gflop:.2f} GFLOP @48x48grid "
        f"cell_px={cell_px} rgb={head.rgb} dils={head.dils} width={head.width} hid={head.hid} "
        f"use_norm={head.use_norm}")
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        log(f"  lr cosine: {lr:g} -> 0 over {epochs} epochs")
    elif lr_sched == "constant":
        sched = None
        log(f"  lr {lr:g} constant")
    else:
        raise ValueError(f"lr_sched must be cosine or constant, got {lr_sched!r}")
    lossf = make_loss(loss, huber_delta)
    log(f"  loss={loss}" + (f" delta={huber_delta}" if loss == "huber" else "")
        + "  (selection metric is val_edge_rho, which is scale-free)")

    best: dict[str, Any] = {"rho": -2.0, "epoch": -1, "sd": None}
    state_p = state_path or default_state_path(out)
    start_ep = 0
    if os.path.exists(state_p):
        st = torch.load(state_p, map_location="cpu", weights_only=False)
        if st.get("planned_epochs") != epochs:
            raise ValueError(f"{state_p} was written with epochs={st.get('planned_epochs')} but this run has "
                             f"epochs={epochs}; refusing to mix two anneal schedules")
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        if sched is not None and st.get("sched") is not None:
            sched.load_state_dict(st["sched"])
        best, start_ep = st["best"], st["epoch"]
        log(f"  RESUMED from {state_p}: epoch {start_ep}/{epochs}, best epoch {best['epoch']} rho={best['rho']:.4f}")
    else:
        log(f"  FRESH START (no {os.path.basename(state_p)})")
    t_start = time.time()

    def save_best() -> None:
        save_object_head(out, best["sd"] if best["sd"] is not None else head.state_dict(), head, ntrain=len(keys),
                         seed=seed, best_epoch=best["epoch"], best_val_rho=best["rho"])

    def save_state(ep_done: int) -> None:
        tmp = state_p + ".tmp"
        torch.save({"head": head.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict() if sched is not None else None,
                    "best": best, "epoch": ep_done, "planned_epochs": epochs, "seed": seed}, tmp)
        os.replace(tmp, state_p)

    for ep in range(start_ep, epochs):
        head.train()
        tot, nb = 0.0, 0
        for x, t, w in tl:
            x, t, w = (v.to(dev, non_blocking=True) for v in (x, t, w))
            tot += train_step(head, opt, lossf, x, t, w, clip)
            nb += 1
        vloss, rho = validate(head, vl, lossf, dev)
        msg = f"    epoch {ep + 1}/{epochs} loss={tot / max(1, nb):.4f} ({nb} batches) val_l1={vloss:.4f} " \
              f"val_edge_rho={rho:.4f}"
        if sched is not None:
            sched.step()
        if rho > best["rho"] + 1e-5:
            best = {"rho": rho, "epoch": ep + 1,
                    "sd": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
            msg += " *best*"
            save_best()                                       # every new best reaches disk at once
        elif ep + 1 >= min_epochs and (ep + 1 - best["epoch"]) >= patience:
            log(msg + f"  -> EARLY STOP (best epoch {best['epoch']}, rho={best['rho']:.4f})")
            save_state(ep + 1)
            break
        log(msg)
        save_state(ep + 1)                                    # after the scheduler step: a resume continues here
        if wall_seconds and (time.time() - t_start) > wall_seconds:
            log(f"    WALL STOP after epoch {ep + 1}/{epochs} ({time.time() - t_start:.0f}s > {wall_seconds}s); "
                f"resume from {state_p}")
            break
    if best["sd"] is not None:
        head.load_state_dict(best["sd"])
        log(f"  restored BEST epoch {best['epoch']} (val_edge_rho={best['rho']:.4f})")
    save_best()
    log(f"saved object head -> {out}")
    return best


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m varest.train.object_head", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", required=True, help="directory holding the images the target keys name")
    ap.add_argument("--targets", nargs="+", required=True, help="object_targets .pkl shards (paths or globs)")
    ap.add_argument("--out", required=True, help="output head .pt (always the best weights so far)")
    ap.add_argument("--state-path", default=None, help="resumable state (default: <out>.state.pt)")
    ap.add_argument("--wall-seconds", type=float, default=0, help="stop after the epoch that crosses this (0 = never)")
    ap.add_argument("--ntrain", type=int, default=341000)
    ap.add_argument("--epochs", type=int, default=100, help="PLANNED total; also the cosine T_max")
    ap.add_argument("--min-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lr-sched", choices=("cosine", "constant"), default="cosine")
    ap.add_argument("--loss", choices=("l1", "l2", "huber"), default="l1")
    ap.add_argument("--huber-delta", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--hid", type=int, default=HID)
    ap.add_argument("--dils", type=int, nargs="+", default=list(DILS))
    ap.add_argument("--cell-px", type=int, default=CELL_PX)
    ap.add_argument("--rgb", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-norm", action=argparse.BooleanOptionalAction, default=False,
                    help="GroupNorm before the projection (released: off)")
    ap.add_argument("--workers", type=int, default=44, help="DataLoader decoding workers (validation uses half)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    keys, store = load_targets(a.targets, a.data_root, a.ntrain, a.seed)
    train(keys, store, a.data_root, a.out, state_path=a.state_path, epochs=a.epochs, min_epochs=a.min_epochs,
          patience=a.patience, batch=a.batch, lr=a.lr, weight_decay=a.weight_decay, clip=a.clip,
          lr_sched=a.lr_sched, loss=a.loss, huber_delta=a.huber_delta, val_frac=a.val_frac, seed=a.seed,
          width=a.width, hid=a.hid, dils=a.dils, cell_px=a.cell_px, rgb=a.rgb, use_norm=a.use_norm,
          workers=a.workers, wall_seconds=a.wall_seconds, device=a.device, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
