"""``tr`` / ``ft`` entry points for Varest and VarestW5 -- thin wrappers over :mod:`varest.train.finetune`.

The whole recipe lives in ``varest``; this module maps the harness's config onto it. ``training_args`` (under
``custom_args``) carries:

  fields       REQUIRED. Glob of the field caches written by ``python -m varest.train.dump_fields`` for this backbone
               family (``llava`` for LLaVA-1.5 7B/13B, ``llava_next`` for LLaVA-NeXT-8B).
  steps, snapshot_every, wall_seconds, tag, lr, ...   optional overrides of :class:`varest.train.finetune.FinetuneConfig`
  package      optional ``[text_head, object_head, out]``: after the final step, SWA the run and write the release
               checkpoint (see :func:`varest.train.finetune.package`).

The corpus is the backbone's own instruction-tuning set: ``<input_dir>/LLaVA-Tuning/llava_v1_5_mix665k.json`` for
LLaVA-1.5 (624,609 image rows) and ``<input_dir>/LLaVA-NeXT-Data/llava_next_raw_format/`` (the processed json and its
``images/`` tree; 738,590 rows) for LLaVA-NeXT-8B; ``steps`` defaults to one pass over it. Checkpoints go to
``<output_dir>/varest/<tag>*.pt``. ``ft`` runs the same loop and resumes from the newest snapshot of the same tag.
"""
from __future__ import annotations

import os
from dataclasses import fields as dc_fields

from mle.vars import ExpConfig

from llmoe.backbone.prototype import VLM
from llmoe.config import Config
from llmoe.weights import backbone_key

from varest.train.dump_fields import MIX_JSON
from varest.train.finetune import FinetuneConfig, Trainer, package

SLUGS = {"llava": "llava-1.5-7b", "llava-13b": "llava-1.5-13b", "llava-next-8b": "llama3-llava-next-8b"}
NEXT_DATA = os.path.join("LLaVA-NeXT-Data", "llava_next_raw_format")
NEXT_JSON = "llava_next_raw_format_processed.json"
NEXT_STEPS = 738590


def corpus(input_dir: str, slug: str) -> tuple[str, str, int]:
    """(data_root, corpus json, one-pass step count) of the backbone's instruction-tuning set."""
    if slug == "llama3-llava-next-8b":
        root = os.path.join(input_dir, NEXT_DATA)
        return os.path.join(root, "images"), os.path.join(root, NEXT_JSON), NEXT_STEPS
    return os.path.join(input_dir, "LLaVA-Tuning"), MIX_JSON, FinetuneConfig.steps


def varest_tr_fn_cd(config: ExpConfig) -> str:
    found = [os.path.join(r, j) for r, j, _ in (corpus(config.input_dir, s) for s in SLUGS.values())
             if os.path.exists(os.path.join(r, j))]
    if not found:
        return (f"Missing the corpus: {MIX_JSON} under LLaVA-Tuning (LLaVA-1.5) or {NEXT_JSON} under {NEXT_DATA} "
                f"(LLaVA-NeXT) in {config.input_dir}")
    args = (getattr(config, "custom_args", None) or {}).get("training_args", {})
    if not args.get("fields"):
        return ("training_args.fields is required: a glob of the caches written by "
                "`python -m varest.train.dump_fields` for this backbone")
    return "OK"


def _train(variant: str, vlm: VLM, config: Config) -> None:
    key = backbone_key(vlm)
    if key not in SLUGS:
        raise ValueError(f"Varest trains on {sorted(SLUGS)}, not backbone {key!r}")
    args = dict(config.custom_args.get("training_args", {}))
    pkg = args.pop("package", None)
    known = {f.name for f in dc_fields(FinetuneConfig)}
    unknown = sorted(set(args) - known)
    if unknown:
        raise ValueError(f"unknown Varest training_args {unknown}")
    data_root, mix_json, steps = corpus(config.input_dir, SLUGS[key])
    if not os.path.exists(os.path.join(data_root, mix_json)):
        raise FileNotFoundError(f"{SLUGS[key]} trains on {os.path.join(data_root, mix_json)}, which does not exist")
    args.setdefault("steps", steps)
    args.setdefault("mix_json", mix_json)
    cfg = FinetuneConfig(variant=variant, backbone=SLUGS[key], data_root=data_root,
                         output_dir=os.path.join(config.output_dir, "varest"),
                         model_id=getattr(vlm, "_model_id", ""), **args)
    last = Trainer(cfg, device=str(getattr(vlm, "_device", "")) or None).train()
    if pkg and os.path.exists(last):
        package(cfg, *pkg)
        print(f"varest: wrote release checkpoint {pkg[2]}", flush=True)


def train_varest(vlm: VLM, config: Config) -> None:
    """W5RELJOINT."""
    _train("w5reljoint", vlm, config)


def train_varest_w5(vlm: VLM, config: Config) -> None:
    """W5."""
    _train("w5", vlm, config)
