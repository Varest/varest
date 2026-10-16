"""Teacher targets for the OBJECT head: DINOv3 ViT-L/16 edge distances over every unique corpus image, sharded.

WHAT IS STORED. Per image ``(edges, (W, H))`` where ``edges`` is float16 ``[2, n_r, n_c]`` on the RAW image's own
patch grid (:func:`varest.heads.object.raw_grid`: long side 768, 16-px patches): channel 0 is the cosine DISTANCE
between a patch's DINOv3 feature and its RIGHT neighbour's, channel 1 to the neighbour BELOW. The last column's
right edge and the last row's down edge do not exist and are stored as 0 (the trainer masks them).

WHY EDGES, NOT A PARTITION. A dumped k-region label map would freeze k at teacher time. Boundary evidence is
k-independent -- the tokenizer floods and merges it to whatever k it is asked for -- so the teacher stores evidence and
the student predicts evidence. Nothing is thresholded here: the value is the raw cosine distance.

NO LETTERBOX. The image is bicubic-resized, aspect preserved, to exactly ``n_c*16 x n_r*16``, ImageNet-normalised, and
fed whole: DINOv3 uses RoPE, so a non-square input is native and there is no padding for the student to get free
credit on. Patch tokens follow the CLS token and the model's register tokens.

Keys are image paths relative to ``--data-root`` in FIRST-APPEARANCE order of the json; shard ``i`` of ``n`` takes
``keys[i::n]``. The released targets are the 20 files ``mix665k_dinoedge_v3vitl16_<i>of20.pkl``
(:func:`shard_name`), consumed by :mod:`varest.train.object_head`.

usage: python -m varest.train.object_targets --data-root <LLaVA-Tuning> --shard 0 --nshard 20 --out <dir or .pkl>
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

from varest.heads.object import LONG_PX, raw_grid

MIX_JSON = "llava_v1_5_mix665k.json"
DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def image_keys(data_root: str, mix_json: str = MIX_JSON) -> list[str]:
    """Unique relative image paths in first-appearance order (stable, so shards are reproducible)."""
    with open(os.path.join(data_root, mix_json)) as f:
        rows = json.load(f)
    seen: set[str] = set()
    out: list[str] = []
    for r in rows:
        im = r.get("image")
        if im and im not in seen:
            seen.add(im)
            out.append(im)
    return out


def edge_dist(f: torch.Tensor) -> torch.Tensor:
    """``[n_r, n_c, D]`` features -> float32 ``[2, n_r, n_c]`` cosine distance to the RIGHT and BELOW neighbour."""
    x = torch.nn.functional.normalize(f, dim=-1)
    e = torch.zeros(2, f.shape[0], f.shape[1], dtype=torch.float32)
    e[0, :, :-1] = 1 - (x[:, :-1] * x[:, 1:]).sum(-1)
    e[1, :-1, :] = 1 - (x[:-1, :] * x[1:, :]).sum(-1)
    return e


def load_teacher(model_id: str = DEFAULT_MODEL, device: str | torch.device = "cpu") -> tuple[Any, int]:
    """-> ``(model, patch_size)``, via transformers ``AutoModel`` in float32."""
    from transformers import AutoConfig, AutoModel
    patch = int(AutoConfig.from_pretrained(model_id).patch_size)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
    return model, patch


@torch.no_grad()
def edges_for(img: Image.Image, model: Any, patch: int, long_px: int = LONG_PX, device: str | torch.device = "cpu",
              amp: bool = True) -> np.ndarray:
    """RGB image -> float16 ``[2, n_r, n_c]`` edge distances on its raw grid. ``amp`` = bfloat16 autocast on CUDA."""
    width, height = img.size
    nr, nc = raw_grid(width, height, long_px, patch)
    im = img.resize((nc * patch, nr * patch), Image.BICUBIC)
    x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1)[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and torch.device(device).type == "cuda"):
        out = model(pixel_values=x)
    h = out.last_hidden_state[0, 1 + getattr(model.config, "num_register_tokens", 0):]
    return edge_dist(h.reshape(nr, nc, -1).float().cpu()).numpy().astype(np.float16)


def model_tag(model_id: str) -> str:
    """``facebook/dinov3-vitl16-pretrain-lvd1689m`` -> ``v3vitl16``."""
    return model_id.rsplit("/", 1)[-1].split("-pretrain")[0].replace("dinov3-", "v3")


def shard_name(model_id: str, shard: int, nshard: int) -> str:
    return f"mix665k_dinoedge_{model_tag(model_id)}_{shard}of{nshard}.pkl"


def dump(model: Any, patch: int, data_root: str, keys: Sequence[str], long_px: int = LONG_PX,
         device: str | torch.device = "cpu", amp: bool = True,
         log_every: int = 500) -> dict[str, tuple[np.ndarray, tuple[int, int]]]:
    """``{key: (edges f16[2,n_r,n_c], (W, H))}`` for ``keys``; unreadable images are left out."""
    out: dict[str, tuple[np.ndarray, tuple[int, int]]] = {}
    bad, t0 = 0, time.time()
    for j, key in enumerate(keys):
        try:
            with Image.open(os.path.join(data_root, key)) as im:
                img = im.convert("RGB")
                out[key] = (edges_for(img, model, patch, long_px, device, amp), img.size)
        except Exception as e:
            bad += 1
            if bad <= 3:
                print(f"  skip {key}: {type(e).__name__} {e}", flush=True)
        if log_every and (j + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  {j + 1}/{len(keys)}  {1000 * el / (j + 1):.0f} ms/img  "
                  f"eta {(len(keys) - j - 1) * el / (j + 1) / 60:.0f} min", flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m varest.train.object_targets", description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", required=True, help=f"directory holding {MIX_JSON} and the images")
    ap.add_argument("--out", required=True,
                    help="output .pkl, or an existing directory to write mix665k_dinoedge_<tag>_<i>of<n>.pkl into")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=20)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="DINOv3 checkpoint (transformers id or local path)")
    ap.add_argument("--long", type=int, default=LONG_PX, help="long side in px before patching")
    ap.add_argument("--limit", type=int, default=0, help="first N images of this shard (0 = all)")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="bf16 autocast on CUDA")
    ap.add_argument("--mix-json", default=MIX_JSON)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if not 0 <= a.shard < a.nshard:
        ap.error(f"need 0 <= shard < nshard, got {a.shard}/{a.nshard}")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, patch = load_teacher(a.model, dev)
    keys = image_keys(a.data_root, a.mix_json)
    mine = keys[a.shard::a.nshard]
    if a.limit:
        mine = mine[:a.limit]
    print(f"### {a.model} patch={patch} long={a.long} | shard {a.shard}/{a.nshard}: {len(mine)} of {len(keys)} "
          f"images", flush=True)
    t0 = time.time()
    out = dump(model, patch, a.data_root, mine, a.long, dev, a.amp)
    dst = os.path.join(a.out, shard_name(a.model, a.shard, a.nshard)) if os.path.isdir(a.out) else a.out
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"### wrote {dst}: {len(out)} images, {len(mine) - len(out)} unreadable, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
