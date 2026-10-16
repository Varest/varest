"""Precompute the W5 fields of a training corpus: text field, boundary field and k_min per image.

Fine-tuning draws a new budget every step, but the fields do not depend on it, and computing them live (two heads
plus a full agglomeration for k_min, ~50 ms per image) would dominate a 0.3 s training step. This dump runs the heads
once per UNIQUE image and stores, per image key (the image path relative to ``--data-root``):

  LLaVA-1.5   {"text": f16[24,24], "boundary": f16[2,24,24], "kmin": int, "size": (W, H)}
  LLaVA-NeXT  {"win": (gh, gw, r0, nr, c0, nc), "text": f16[nr,nc], "bound": f16[2,nr,nc], "kmin": int,
               "btext": f16[24,24], "bbound": f16[2,24,24], "bkmin": int}

``kmin`` is the RAW merge-cost knee of ``text_cut(boundary, text)`` (uncapped: the builder caps it at K/2 for the K
it is given). Keys are relative so a cache is valid from any mount of the same data. Shards are strided over the
sorted unique image list (``--shard i/n``), so shards are disjoint and their union is the corpus.

usage: python -m varest.train.dump_fields --family llava|llava_next --data-root <LLaVA-Tuning> \
          --text-head text_head.pt --object-head object_head.pt --out <dir> [--shard i/n]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from PIL import Image

from varest import next as vnext
from varest.fields import llava_fields, next_fields
from varest.heads.object import load_object_head
from varest.heads.text import load_text_head
from varest.tokenizer import KMIN_Q, TEXT_EDGE_W, kmin_from_curve, text_cut

MIX_JSON = "llava_v1_5_mix665k.json"


def corpus_images(data_root: str, mix_json: str = MIX_JSON) -> list[str]:
    """Sorted unique image paths (relative to ``data_root``) of the fine-tuning corpus."""
    with open(os.path.join(data_root, mix_json)) as f:
        rows = json.load(f)
    return sorted({r["image"] for r in rows if r.get("image")})


def _kmin(args: tuple[np.ndarray, np.ndarray]) -> int:
    b, t = args
    return int(kmin_from_curve(text_cut(np.asarray(b, np.float32), np.asarray(t, np.float32), TEXT_EDGE_W)))


def dump(family: str, data_root: str, keys: Sequence[str], text_head_path: str, object_head_path: str,
         workers: int = 8, device: str | None = None, log_every: int = 2000) -> dict:
    """-> ``{"fields": {key: record}, <metadata>}`` for ``keys`` (paths relative to ``data_root``)."""
    if family not in ("llava", "llava_next"):
        raise ValueError(f"family must be llava or llava_next, got {family!r}")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    th = load_text_head(text_head_path, dev)
    oh = load_object_head(object_head_path, dev)
    out: dict[str, dict] = {}
    futs = []
    bad = 0
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for j, key in enumerate(keys):
            if os.path.isabs(key):
                raise ValueError(f"absolute key {key!r}: keys must be relative to the data root")
            fp = os.path.join(data_root, key)
            try:
                if family == "llava":
                    f = llava_fields(fp, th, oh)
                    out[key] = {"text": f.text, "boundary": f.boundary, "size": f.size}
                    futs.append((key, "kmin", pool.submit(_kmin, (f.boundary, f.text))))
                else:
                    with Image.open(fp) as im:
                        lay = vnext.layout(*im.size)
                    lo, hi = next_fields(fp, th, oh, (lay.nr, lay.nc))
                    out[key] = {"win": (lay.gh, lay.gw, lay.r0, lay.nr, lay.c0, lay.nc), "text": hi.text,
                                "bound": hi.boundary, "btext": lo.text, "bbound": lo.boundary}
                    futs.append((key, "kmin", pool.submit(_kmin, (hi.boundary, hi.text))))
                    futs.append((key, "bkmin", pool.submit(_kmin, (lo.boundary, lo.text))))
            except Exception as e:                  # a handful of corrupt files must not kill a shard
                bad += 1
                if bad <= 3:
                    print(f"  skip {key}: {type(e).__name__} {e}", flush=True)
            if log_every and (j + 1) % log_every == 0:
                el = time.time() - t0
                print(f"  {j + 1}/{len(keys)}  {1000 * el / (j + 1):.1f} ms/img", flush=True)
        for key, name, fut in futs:
            out[key][name] = fut.result()
    print(f"### {len(out)} images, {bad} unreadable, {(time.time() - t0) / 60:.1f} min", flush=True)
    return {"fields": out, "family": family, "text_head": os.path.basename(text_head_path),
            "object_head": os.path.basename(object_head_path), "text_edge_w": TEXT_EDGE_W, "q": KMIN_Q}


def load_fields(pattern: str) -> tuple[dict[str, dict], dict]:
    """Merge every shard matching the glob ``pattern`` -> ``(fields, metadata)``; refuses shards built with
    different heads or settings."""
    import glob
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no field shard matches {pattern!r}")
    fields: dict[str, dict] = {}
    meta = None
    for f in files:
        with open(f, "rb") as fh:
            blob = pickle.load(fh)
        m = {k: blob.get(k) for k in ("family", "text_head", "object_head", "text_edge_w", "q")}
        if meta is None:
            meta = m
        elif m != meta:
            raise ValueError(f"{f} was built with {m}, the first shard with {meta}")
        fields.update(blob["fields"])
    return fields, meta or {}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Precompute W5 field caches for fine-tuning.")
    ap.add_argument("--family", required=True, choices=("llava", "llava_next"))
    ap.add_argument("--data-root", required=True, help="directory holding the corpus json and its images")
    ap.add_argument("--mix-json", default=MIX_JSON)
    ap.add_argument("--text-head", required=True)
    ap.add_argument("--object-head", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--shard", default="0/1", help="i/n: strided shard of the sorted unique image list")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="processes computing k_min")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    i, n = (int(v) for v in a.shard.split("/"))
    name = f"{a.family}_fields" + (f"_{i}of{n}" if n > 1 else "") + (f"_limit{a.limit}" if a.limit else "")
    dst = os.path.join(a.out, name + ".pkl")
    if os.path.exists(dst) and not a.force:
        raise SystemExit(f"{dst} exists; pass --force to overwrite")
    keys = corpus_images(a.data_root, a.mix_json)[i::n]
    if a.limit:
        keys = keys[:a.limit]
    print(f"### {a.family} fields: {len(keys)} images (shard {i}/{n}) -> {dst}", flush=True)
    blob = dump(a.family, a.data_root, keys, a.text_head, a.object_head, workers=a.workers)
    os.makedirs(a.out, exist_ok=True)
    with open(dst, "wb") as f:
        pickle.dump(blob, f, protocol=4)
    print(f"### wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
