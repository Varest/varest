"""EasyOCR labels for the TEXT head: the positive cells and the two-resolution dump that yields its negatives.

Two passes over the fine-tuning corpus (``llava_v1_5_mix665k.json`` under ``--data-root``), both with EasyOCR
(CRAFT detector + CRNN recogniser). A real OCR engine rather than an open-vocabulary detector because text
RECOGNITION is what separates text from text-like texture; a detector prompted with "text. word. letter." recovered
only about a quarter of the OCR gain.

``positives``  OCR on the text-heavy sources (``ocr_vqa`` and ``textvqa`` by default), one pass per ROW, keyed by
               ``str(row_index)`` over ALL rows of the json (the image-bearing rows are contiguous, so this equals the
               image-row index). Value: sorted ``(r, c)`` cells of LLaVA's padded 24x24 square that any detection box
               touches. This is the label file the text head trains on.

``two-res``    OCR over every UNIQUE image (all sources by default) at long side 672 and at 336, NEVER upsampling: an
               image whose long side is below a target runs at native size, since interpolation would label small
               source text "legible at 672" when nothing made it so. Keyed by the image path RELATIVE to the data root
               (the same data is mounted under two prefixes, so absolute keys do not survive). Value::

                   {"c672": f16[576], "c336": f16[576],            # per-cell MAX detection confidence
                    "d672": [{"t": str, "c": float, "cells": [(r, c), ...]}, ...], "d336": [...]}

               An image with no detection at either resolution is a TEXT-FREE negative for the text head. Boxes come
               back in the resized frame and are scaled to original pixels BEFORE the letterbox transform; reversing
               the two would shift every cell on every downscaled pass.

Both passes share one geometry (:func:`xyxy_frame`, :func:`box2cells`), and shard by striding: ``positives`` over
the source rows in json order, ``two-res`` over the sorted unique image list, ``items[shard::nshard]`` after
``--limit``. Shards are disjoint and their union is the whole set.

easyocr is an optional dependency (``pip install easyocr``); it is imported only when a reader is built.

usage: python -m varest.train.ocr_labels positives --data-root <LLaVA-Tuning> --out mix665k_easyocr_text.pkl
       python -m varest.train.ocr_labels two-res --data-root <LLaVA-Tuning> --shard 0 --nshard 30 \
           --out mix665k_ocr2res_0of30.pkl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from PIL import Image

MIX_JSON = "llava_v1_5_mix665k.json"
GRID = 24
POSITIVE_SOURCES = ("ocr_vqa", "textvqa")
RESOLUTIONS = (672, 336)

Cell = tuple[int, int]


def xyxy_frame(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> list[float]:
    """Pixel box on the raw image -> ``[x0, y0, x1, y1]`` in [0, 1] on LLaVA's padded square (centred letterbox,
    integer offsets, clamped). The same convention as every cells file the tokenizer consumes."""
    s = max(width, height)
    ox = (s - width) // 2 if width < s else 0
    oy = (s - height) // 2 if height < s else 0
    return [min(max(v, 0), 1) for v in [(x0 + ox) / s, (y0 + oy) / s, (x1 + ox) / s, (y1 + oy) / s]]


def box2cells(b: Sequence[float], grid: int = GRID) -> list[Cell]:
    """Normalised ``[x0, y0, x1, y1]`` -> every ``(r, c)`` of the ``grid x grid`` square the box touches."""
    c0, c1 = max(0, int(b[0] * grid)), min(grid - 1, int(math.ceil(b[2] * grid)) - 1)
    r0, r1 = max(0, int(b[1] * grid)), min(grid - 1, int(math.ceil(b[3] * grid)) - 1)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def quad_cells(pts: Sequence[Sequence[float]], width: int, height: int, scale: float = 1.0) -> list[Cell]:
    """EasyOCR's 4-point box (in a frame ``scale`` times smaller than the raw image) -> padded-grid cells."""
    xs = [p[0] * scale for p in pts]
    ys = [p[1] * scale for p in pts]
    return box2cells(xyxy_frame(min(xs), min(ys), max(xs), max(ys), width, height))


def make_reader(model_dir: str | None = None, gpu: bool = True) -> Any:
    """English EasyOCR reader. With ``model_dir`` the models must already be there (no download), which is how the
    released labels were made; without it EasyOCR uses (and may download into) its own default directory."""
    try:
        import easyocr
    except ImportError as e:                                  # optional dependency
        raise ImportError("varest.train.ocr_labels needs EasyOCR: pip install easyocr") from e
    if model_dir is None:
        return easyocr.Reader(["en"], gpu=gpu)
    return easyocr.Reader(["en"], gpu=gpu, model_storage_directory=model_dir, download_enabled=False)


def _load_mix(data_root: str, mix_json: str = MIX_JSON) -> list[dict]:
    with open(os.path.join(data_root, mix_json)) as f:
        return json.load(f)


def positive_rows(data_root: str, sources: Iterable[str] = POSITIVE_SOURCES,
                  mix_json: str = MIX_JSON) -> list[tuple[str, str]]:
    """``[(str(row_index), relative_image_path), ...]`` for the rows whose image comes from ``sources``, in json
    order. The index runs over ALL rows, so keys index the json directly (``big[int(key)]``)."""
    want = set(sources)
    out = []
    for i, e in enumerate(_load_mix(data_root, mix_json)):
        im = e.get("image")
        if im and im.split("/")[0] in want:
            out.append((str(i), im))
    return out


def ocr_positives(reader: Any, data_root: str, items: Sequence[tuple[str, str]], conf: float = 0.0,
                  log_every: int = 200) -> dict[str, list[Cell]]:
    """OCR each ``(key, rel)`` and return ``{key: sorted cells}``. An unreadable image keeps its key with NO cells,
    so the output covers every requested row. ``conf`` drops boxes below it (0 keeps all: density is the knob)."""
    out: dict[str, list[Cell]] = {}
    nb = 0
    for n, (key, rel) in enumerate(items):
        fp = os.path.join(data_root, rel)
        try:
            with Image.open(fp) as im:
                width, height = im.convert("RGB").size       # full decode: a truncated file fails HERE
            res = reader.readtext(fp)                         # EasyOCR decodes the file itself
        except Exception as e:
            print(f"  SKIP {key}: {type(e).__name__}: {e}", flush=True)
            out[key] = []
            continue
        acc: set[Cell] = set()
        for pts, _txt, cf in res:
            if cf < conf:
                continue
            acc.update(quad_cells(pts, width, height))
        out[key] = sorted(acc)
        nb += len(acc)
        if log_every and (n + 1) % log_every == 0:
            print(f"  {n + 1}/{len(items)}  mean_cells={nb / (n + 1):.1f}", flush=True)
    return out


def unique_images(data_root: str, sources: Iterable[str] | None = None, mix_json: str = MIX_JSON) -> list[str]:
    """Sorted unique relative image paths (optionally restricted to ``sources``). Sorted, so every shard agrees on
    the split regardless of dict order."""
    want = None if sources is None else set(sources)
    return sorted({e["image"] for e in _load_mix(data_root, mix_json)
                   if e.get("image") and (want is None or e["image"].split("/")[0] in want)})


def ocr_two_res(reader: Any, img: Image.Image, resolutions: Sequence[int] = RESOLUTIONS, conf: float = 0.0,
                keep_dets: bool = True, grid: int = GRID) -> dict[str, Any]:
    """One RGB image -> ``{"c<res>": f16[grid*grid], "d<res>": [...]}`` for each long side in ``resolutions``."""
    width, height = img.size
    s = max(width, height)
    rec: dict[str, Any] = {}
    for long_px in resolutions:
        sc = long_px / float(s)
        if sc >= 1.0:                                         # never upsample
            arr, back = np.asarray(img), 1.0
        else:
            arr = np.asarray(img.resize((max(1, int(round(width * sc))), max(1, int(round(height * sc)))),
                                        Image.BICUBIC))
            back = 1.0 / sc                                   # resized px -> original px
        cmap = np.zeros((grid, grid), np.float32)
        dets = []
        for pts, txt, cf in reader.readtext(arr):
            cf = float(cf)
            if cf < conf:
                continue
            cells = quad_cells(pts, width, height, back)
            for (r, c) in cells:
                if cmap[r, c] < cf:                           # per-cell MAX confidence
                    cmap[r, c] = cf
            if keep_dets:
                dets.append({"t": txt, "c": cf, "cells": cells})
        rec[f"c{long_px}"] = cmap.reshape(-1).astype(np.float16)
        if keep_dets:
            rec[f"d{long_px}"] = dets
    return rec


def ocr_two_res_images(reader: Any, data_root: str, rels: Sequence[str], resolutions: Sequence[int] = RESOLUTIONS,
                       conf: float = 0.0, keep_dets: bool = True, log_every: int = 500) -> dict[str, dict[str, Any]]:
    """:func:`ocr_two_res` over ``rels``; unreadable images are left out (and counted in the log)."""
    out: dict[str, dict[str, Any]] = {}
    nskip = 0
    for n, rel in enumerate(rels):
        try:
            with Image.open(os.path.join(data_root, rel)) as im:
                img = im.convert("RGB")
            out[rel] = ocr_two_res(reader, img, resolutions, conf, keep_dets)
        except Exception as e:
            nskip += 1
            if nskip <= 3:
                print(f"  skip {rel}: {type(e).__name__} {e}", flush=True)
        if log_every and (n + 1) % log_every == 0:
            print(f"  {n + 1:,}/{len(rels):,} ({nskip} skipped)", flush=True)
    return out


def label_mix(out: dict[str, dict[str, Any]], hi: int = max(RESOLUTIONS), lo: int = min(RESOLUTIONS),
              thr: float = 0.5) -> dict[str, float]:
    """Share of cells in each (hi, lo) legibility class at ``thr`` -- printed so a shard that silently OCRs nothing
    shows in its own log."""
    tot = np.zeros(4, np.int64)
    for v in out.values():
        a = np.asarray(v[f"c{hi}"], np.float32) >= thr
        b = np.asarray(v[f"c{lo}"], np.float32) >= thr
        tot += np.array([int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())])
    n = max(int(tot.sum()), 1)
    return {"normal": tot[0] / n, "needs_res": tot[1] / n, "penalise": tot[2] / n, "none": tot[3] / n}


def _dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m varest.train.ocr_labels", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("positives", "row-keyed positive cells on the text-heavy sources"),
                      ("two-res", "image-keyed OCR at two long sides, for text-free negatives")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--data-root", required=True, help=f"directory holding {MIX_JSON} and the images")
        p.add_argument("--out", required=True, help="output .pkl")
        p.add_argument("--shard", type=int, default=0)
        p.add_argument("--nshard", type=int, default=1)
        p.add_argument("--limit", type=int, default=0, help="first N items before sharding (0 = all)")
        p.add_argument("--easyocr-models", default=None,
                       help="pre-downloaded EasyOCR model dir (downloads disabled); default: EasyOCR's own")
        p.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--conf", type=float, default=0.0, help="drop boxes below this confidence")
        p.add_argument("--mix-json", default=MIX_JSON)
        if name == "positives":
            p.add_argument("--sources", nargs="+", default=list(POSITIVE_SOURCES),
                           help="first path component of the image (default: ocr_vqa textvqa)")
        else:
            p.add_argument("--sources", nargs="+", default=None, help="default: every source")
            p.add_argument("--resolutions", type=int, nargs="+", default=list(RESOLUTIONS))
            p.add_argument("--keep-dets", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args(argv)
    if not 0 <= a.shard < a.nshard:
        ap.error(f"need 0 <= shard < nshard, got {a.shard}/{a.nshard}")

    if a.cmd == "positives":
        items = positive_rows(a.data_root, a.sources, a.mix_json)
        n_all = len(items)
        if a.limit:
            items = items[:a.limit]
        items = items[a.shard::a.nshard]
        print(f"### positives: sources={sorted(a.sources)} -> {n_all:,} rows; shard {a.shard}/{a.nshard} = "
              f"{len(items):,}  gpu={a.gpu} conf>={a.conf}", flush=True)
        out = ocr_positives(make_reader(a.easyocr_models, a.gpu), a.data_root, items, a.conf)
        _dump(out, a.out)
        tot = sum(len(v) for v in out.values())
        print(f"### wrote {a.out}: {len(out):,} rows, mean {tot / max(1, len(out)):.1f} cells/row", flush=True)
        return

    rels = unique_images(a.data_root, a.sources, a.mix_json)
    if a.limit:
        rels = rels[:a.limit]
    mine = rels[a.shard::a.nshard]
    print(f"### two-res: sources={'ALL' if a.sources is None else sorted(a.sources)}  unique images={len(rels):,} "
          f"-> shard {a.shard}/{a.nshard} = {len(mine):,}  res={a.resolutions} gpu={a.gpu} "
          f"keep_dets={a.keep_dets}", flush=True)
    out2 = ocr_two_res_images(make_reader(a.easyocr_models, a.gpu), a.data_root, mine, a.resolutions, a.conf,
                              a.keep_dets)
    _dump(out2, a.out)
    m = label_mix(out2, max(a.resolutions), min(a.resolutions))
    print(f"### wrote {a.out}: {len(out2):,} images, {len(mine) - len(out2)} skipped", flush=True)
    print(f"###   cells  normal={100 * m['normal']:.2f}%  NEEDS-RES={100 * m['needs_res']:.2f}%  "
          f"PENALISE={100 * m['penalise']:.2f}%  none={100 * m['none']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
