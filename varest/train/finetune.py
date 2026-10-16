"""Fine-tune LLaVA-1.5 (7B/13B) or LLaVA-NeXT-8B to read W5 visual tokens: W5 and W5RELJOINT.

RECIPE (the released checkpoints). One pass over the image rows of LLaVA's 665K instruction mix (first turn of each
conversation; rows shuffled once with seed 1234), batch 1, 624,609 steps. Every step draws a fresh budget
K ~ U[32, 576] (LLaVA-NeXT: the retain ratio K/576), builds the W5 tokens of the image from its cached fields
(:mod:`varest.train.dump_fields`) and trains, at LR 2e-5 constant with AdamW (weight decay 0.01), gradient clipping
at 1.0:

* a LoRA (rank 32, alpha 64, dropout 0.05) on q/k/v/o of BOTH the vision tower and the language model and on
  ``lm_head``;
* the multimodal projector (full);
* the position tag (:mod:`varest.postag`), trained from scratch.

W5RELJOINT additionally trains the [REL] head (:mod:`varest.rel`) jointly FROM SCRATCH: its 4 registers sit after the
prompt, the pruned count K' ~ U{0..K} is drawn per step, the tokens [REL] would prune are masked out of every query at
layers >= the middle layer (16 of 32, 20 of 40), and the head has its own AdamW group (LR 1e-3 with a 2000-step linear
warm-up, no weight decay, not clipped).

Snapshots every 32,000 steps; the released weights are the SWA of the 512k/544k/576k/608k snapshots and the final
weights (:mod:`varest.swa`), packaged with the two heads (:func:`package`).

The sequence is the one the engine decodes (:mod:`varest.engine`) plus the target: `` " " + answer + </s>`` on
LLaVA-1.5, ``answer + <|eot_id|>`` on LLaVA-NeXT. Sequences longer than ``max_len`` are skipped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from varest import next as vnext
from varest.checkpoint import BACKBONES, VarestCheckpoint
from varest.engine import LLAVA_SYSTEM, NEXT_ASSISTANT, NEXT_USER
from varest.postag import PosTag
from varest.runtime import _canonical
from varest.rel import RelHead, forward_loss, register_keep_attention
from varest.swa import average_files, swa_steps
from varest.tokenizer import GRID, build_partition, pool_regions
from varest.train.dump_fields import MIX_JSON, load_fields

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]


@dataclass
class FinetuneConfig:
    variant: str                                    # "w5" | "w5reljoint"
    backbone: str                                   # "llava-1.5-7b" | "llava-1.5-13b" | "llama3-llava-next-8b"
    data_root: str                                  # directory holding the corpus json and its images
    fields: str                                     # glob of varest.train.dump_fields shards
    output_dir: str
    tag: str = ""
    mix_json: str = MIX_JSON
    model_id: str = ""
    steps: int = 624609
    lr: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    kmin: int = 32
    kmax: int = 576
    max_len: int = 0                                # 0: 1400 on LLaVA-1.5, 4096 on LLaVA-NeXT
    shuffle_seed: int = 1234
    seed: int = 1234
    snapshot_every: int = 32000
    wall_seconds: int = 0                           # stop cleanly (and save) after this long; 0 = never
    rel_layer: int = 0                              # 0: the middle decoder layer
    rel_registers: int = 4
    rel_rank: int = 256
    rel_lr: float = 1e-3
    rel_warmup: int = 2000
    rel_temp: float = 0.1
    log_every: int = 100
    only_cached: bool = False                       # train only on images present in the field cache
    init_from: str = ""                             # start from these weights (a trainer checkpoint), not step 0 state
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.variant not in ("w5", "w5reljoint"):
            raise ValueError(f"variant must be w5 or w5reljoint, got {self.variant!r}")
        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {sorted(BACKBONES)}, got {self.backbone!r}")
        self.model_id = self.model_id or BACKBONES[self.backbone]
        self.tag = self.tag or f"varest_{self.variant}_{self.backbone}"
        if not self.max_len:
            self.max_len = 4096 if self.is_next else 1400

    @property
    def is_next(self) -> bool:
        return self.backbone == "llama3-llava-next-8b"

    @property
    def uses_rel(self) -> bool:
        return self.variant == "w5reljoint"


# ------------------------------------------------------------------------------------------------ data
def load_corpus(data_root: str, mix_json: str, shuffle_seed: int) -> list[dict]:
    """Image rows of the instruction mix as ``{"image", "prompt", "target"}`` (first turn), shuffled once."""
    with open(os.path.join(data_root, mix_json)) as f:
        rows = json.load(f)

    def clean(t: str) -> str:
        t = t.replace("<image>", "")
        t = re.sub(r"Reference OCR token:.*", "", t, flags=re.S)
        return t.strip()

    data = []
    for r in rows:
        if not r.get("image"):
            continue
        conv = r["conversations"]
        data.append({"image": r["image"], "prompt": clean(conv[0]["value"]), "target": conv[1]["value"].strip()})
    random.Random(shuffle_seed).shuffle(data)
    return data


# ------------------------------------------------------------------------------------------------ model
def build_model(cfg: FinetuneConfig, device: torch.device):
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoProcessor, LlavaForConditionalGeneration, LlavaImageProcessor,
                              LlavaNextForConditionalGeneration)
    proc = AutoProcessor.from_pretrained(cfg.model_id)
    if cfg.is_next:
        model = LlavaNextForConditionalGeneration.from_pretrained(cfg.model_id, dtype=torch.bfloat16)
    else:
        ip = proc.image_processor.to_dict()
        ip.pop("do_pad", None)
        ip["image_processor_type"] = "LlavaImageProcessor"
        proc.image_processor = LlavaImageProcessor(**ip, do_pad=True)       # pad to square, as LLMOE evaluates
        model = LlavaForConditionalGeneration.from_pretrained(cfg.model_id, dtype=torch.bfloat16)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    lcfg = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for n, p in model.named_parameters():
        if "multi_modal_projector" in n:
            p.requires_grad_(True)
    return proc, model


def _inner(model):
    return model.base_model.model.model                 # peft -> Llava*ForConditionalGeneration -> Llava*Model


# ------------------------------------------------------------------------------------------------ steps
class Trainer:
    def __init__(self, cfg: FinetuneConfig, device: str | None = None) -> None:
        self.cfg = cfg
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(cfg.seed)
        self.proc, self.model = build_model(cfg, self.dev)
        self.tok = self.proc.tokenizer
        inner = _inner(self.model)
        self.text_model = inner.language_model
        self.lm_head = self.model.get_output_embeddings()
        self.emb = self.model.get_input_embeddings()
        self.vfl = self.model.config.vision_feature_layer
        width = self.model.config.text_config.hidden_size
        self.postag = PosTag(width).to(self.dev, torch.bfloat16)
        self.rel = None
        n_layers = len(self.text_model.layers)
        self.rel_layer = cfg.rel_layer or n_layers // 2
        if cfg.uses_rel:
            self.rel = RelHead(d=width, nreg=cfg.rel_registers, rank=cfg.rel_rank).to(self.dev).float()
            register_keep_attention(self.text_model)
        self.fields, self.field_meta = load_fields(cfg.fields)
        want = "llava_next" if cfg.is_next else "llava"
        if self.field_meta.get("family") != want:
            raise ValueError(f"{cfg.fields} holds {self.field_meta.get('family')} fields, {cfg.backbone} needs {want}")
        self.data = load_corpus(cfg.data_root, cfg.mix_json, cfg.shuffle_seed)
        if cfg.only_cached:
            self.data = [d for d in self.data if d["image"] in self.fields]
            if not self.data:
                raise ValueError("only_cached: no corpus image has a cached field")
        self.params = [p for p in self.model.parameters() if p.requires_grad] + list(self.postag.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if self.rel is not None:
            self.opt.add_param_group({"params": list(self.rel.parameters()), "lr_mult": cfg.rel_lr / cfg.lr,
                                      "rel": True, "weight_decay": 0.0})
        self.k_rng = random.Random(cfg.seed)
        self.rel_rng = random.Random(4321)
        self.eos = (self.tok.convert_tokens_to_ids("<|eot_id|>") if cfg.is_next else self.tok.eos_token_id)
        self.start = 0
        self.stats = {"skipped_long": 0, "missing_fields": 0, "oom": 0, "nonfinite": 0}

    # -- paths / state
    def path(self, step: int | None = None) -> str:
        base = os.path.join(self.cfg.output_dir, self.cfg.tag)
        return f"{base}_step{step}.pt" if step is not None else f"{base}.pt"

    def state(self, step: int) -> dict:
        sd = {"lora": {k: v.detach().cpu() for k, v in self.model.state_dict().items() if "lora" in k.lower()},
              "postag": {k: v.detach().cpu() for k, v in self.postag.state_dict().items()},
              "proj": {k: v.detach().cpu() for k, v in self.model.state_dict().items()
                       if "multi_modal_projector" in k},
              "cfg": {**asdict(self.cfg), "rel_layer": self.rel_layer, "method": "Varest"},
              "step": step}
        if self.rel is not None:
            sd["relhead"] = {k: v.detach().cpu() for k, v in self.rel.state_dict().items()}
        return sd

    def save(self, step: int, snapshot: bool) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        st = self.state(step)
        dst = self.path(step) if snapshot else self.path()
        torch.save(st, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        torch.save({"opt": self.opt.state_dict(), "step": step, "k_rng": self.k_rng.getstate(),
                    "rel_rng": self.rel_rng.getstate()}, self.path() + ".opt.tmp")
        os.replace(self.path() + ".opt.tmp", self.path() + ".opt")

    def load_weights(self, st: dict) -> None:
        """LoRA, projector, position tag (and [REL] head) from a trainer checkpoint."""
        own = self.model.state_dict()
        # checkpoints from other transformers versions may name CLIP's encoder with or without `vision_model.`
        by_canon = {_canonical(k): k for k in own}
        src = {**st["lora"], **st["proj"]}
        missing = [k for k in src if _canonical(k) not in by_canon]
        if missing:
            raise ValueError(f"{len(missing)} checkpoint tensors do not match the model, e.g. {missing[:2]}")
        own.update({by_canon[_canonical(k)]: v.to(own[by_canon[_canonical(k)]].dtype) for k, v in src.items()})
        self.model.load_state_dict(own)
        self.postag.load_state_dict(st["postag"])
        if self.rel is not None:
            self.rel.load_state_dict(st["relhead"])

    def resume(self) -> None:
        """Continue from the newest of ``<tag>.pt`` and ``<tag>_step<N>.pt`` (a killed segment leaves the snapshot
        newer than the main file)."""
        cands = []
        for f in [self.path()] + glob.glob(self.path().replace(".pt", "_step*.pt")):
            if os.path.exists(f):
                st = torch.load(f, map_location="cpu", weights_only=False)
                if int(st.get("step", -1)) >= 0:
                    cands.append((int(st["step"]), f, st))
        if not cands:
            return
        step, f, st = max(cands, key=lambda c: c[0])
        self.load_weights(st)
        opt_path = self.path() + ".opt"
        if os.path.exists(opt_path):
            o = torch.load(opt_path, map_location="cpu", weights_only=False)
            if int(o.get("step", -1)) == step:
                self.opt.load_state_dict(o["opt"])
                self.k_rng.setstate(o["k_rng"])
                self.rel_rng.setstate(o["rel_rng"])
        self.start = step
        print(f"resumed from {f} at step {step}", flush=True)

    # -- one example
    def _visual_llava(self, img: Image.Image, rec: dict, k: int) -> torch.Tensor:
        pv = self.proc.image_processor(img, return_tensors="pt")["pixel_values"].to(self.dev, torch.bfloat16)
        inner = _inner(self.model)
        out = inner.vision_tower(pv, output_hidden_states=True)
        xf = inner.multi_modal_projector(out.hidden_states[self.vfl][:, 1:])[0]
        w, h = rec["size"]
        from varest.heads.text import cell_shape
        r0, nr, c0, nc = cell_shape(w, h)
        inimg = np.zeros((GRID, GRID), bool)
        inimg[r0:r0 + nr, c0:c0 + nc] = True
        lab, _ = build_partition(np.asarray(rec["boundary"], np.float32), np.asarray(rec["text"], np.float32), k,
                                 inimg=inimg, kmin=rec.get("kmin"))
        tokens, pos, _ = pool_regions(xf, lab, GRID)
        p = torch.as_tensor(pos, device=self.dev).to(torch.bfloat16)
        return self.postag.apply_to(tokens, p)          # float32 tokens + bf16 tag -> float32 (cast at assembly)

    def _visual_next(self, img: Image.Image, rec: dict, k7: int) -> torch.Tensor:
        w, h = img.size
        lay = vnext.layout(w, h)
        if tuple(rec["win"]) != (lay.gh, lay.gw, lay.r0, lay.nr, lay.c0, lay.nc):
            raise ValueError(f"cached window {rec['win']} does not match this image's layout {lay}")
        o = self.proc.image_processor(img, return_tensors="pt")
        pv = o["pixel_values"][0, :1 + lay.nh * lay.nw].to(self.dev, torch.bfloat16)
        inner = _inner(self.model)
        out = inner.vision_tower(pv, output_hidden_states=True)
        f = inner.multi_modal_projector(out.hidden_states[self.vfl][:, 1:])
        k = vnext.kept(lay.nr, lay.nc, k7 / 576.0)
        plan = vnext.plan((rec["bbound"], rec["btext"], rec.get("bkmin")), (rec["bound"], rec["text"], rec.get("kmin")),
                          lay, k)
        tokens, pos = vnext.tokenize(f[0], vnext.stitch(f[1:], lay), plan, lay)
        p = torch.as_tensor(pos, device=self.dev).to(torch.bfloat16)
        return self.postag.apply_to(tokens, p)

    def _ids(self, text: str, special: bool) -> torch.Tensor:
        return self.tok(text, add_special_tokens=special, return_tensors="pt").input_ids.to(self.dev)

    def example(self, item: dict, k: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int] | None] | None:
        rec = self.fields.get(item["image"])
        if rec is None:
            self.stats["missing_fields"] += 1
            return None
        with Image.open(os.path.join(self.cfg.data_root, item["image"])) as im:
            img = im.convert("RGB")
        if self.cfg.is_next:
            vis = self._visual_next(img, rec, k)
            pre = self._ids(NEXT_USER, True)
            tail = self._ids(NEXT_ASSISTANT, False)
            tgt = self._ids(item["target"], False)
        else:
            vis = self._visual_llava(img, rec, k)
            pre = self._ids(LLAVA_SYSTEM + "USER: ", True)
            tail = self._ids(" ASSISTANT:", False)
            tgt = self._ids(" " + item["target"], False)
        tids = torch.cat([tgt, torch.tensor([[self.eos]], device=self.dev)], 1)
        pe = self.emb(pre)
        if self.rel is None and not self.cfg.is_next:
            post = self.emb(self._ids("\n" + item["prompt"] + " ASSISTANT:", False))
            parts = [pe, vis.unsqueeze(0), post]
            spans = None
        else:
            pq = self.emb(self._ids("\n" + item["prompt"], False))
            parts = [pe, vis.unsqueeze(0).to(pq.dtype), pq]
            spans = None
            if self.rel is not None:
                rt = self.rel.tokens.to(pq.dtype).unsqueeze(0)
                spans = (pre.shape[1], pre.shape[1] + vis.shape[0], pre.shape[1] + vis.shape[0] + pq.shape[1],
                         rt.shape[1])
                parts.append(rt)
            parts.append(self.emb(tail))
        parts.append(self.emb(tids))
        e = torch.cat([x.to(torch.float32) if x.dtype != torch.float32 else x for x in parts], 1).to(torch.bfloat16)
        if e.shape[1] > self.cfg.max_len:
            self.stats["skipped_long"] += 1
            return None
        lab = torch.full((e.shape[1],), -100, device=self.dev, dtype=torch.long)
        lab[-tids.shape[1]:] = tids[0]
        return e, lab, spans

    def loss(self, e: torch.Tensor, lab: torch.Tensor, spans) -> torch.Tensor:
        if self.rel is None:
            out = self.model(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], device=self.dev,
                                                                         dtype=torch.long),
                             labels=lab.unsqueeze(0), use_cache=False)
            return out.loss
        a0, b0, _, _ = spans
        kp = self.rel_rng.randint(0, b0 - a0)
        loss, _ = forward_loss(self.text_model, self.lm_head, self.rel, e, lab.unsqueeze(0), spans, kp,
                               self.rel_layer, self.cfg.rel_temp)
        return loss

    # -- loop
    def train(self) -> str:
        cfg = self.cfg
        self.resume()
        if self.start == 0 and cfg.init_from:
            self.load_weights(torch.load(cfg.init_from, map_location="cpu", weights_only=False))
            print(f"initialised from {cfg.init_from}", flush=True)
        self.model.train()
        t0 = time.time()
        run, nb = 0.0, 0
        for step in range(self.start, cfg.steps):
            item = self.data[step % len(self.data)]
            k = self.k_rng.randint(cfg.kmin, cfg.kmax)
            try:
                ex = self.example(item, k)
                if ex is not None:
                    loss = self.loss(*ex)
                    loss.backward()
                    for g in self.opt.param_groups:
                        g["lr"] = cfg.lr * (g["lr_mult"] * min(1.0, (step + 1) / max(1, cfg.rel_warmup))
                                            if g.get("rel") else 1.0)
                    gn = torch.nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
                    if torch.isfinite(gn):
                        self.opt.step()
                    else:
                        self.stats["nonfinite"] += 1
                    self.opt.zero_grad(set_to_none=True)
                    run += float(loss.detach())
                    nb += 1
            except torch.cuda.OutOfMemoryError:
                self.stats["oom"] += 1
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            if cfg.log_every and (step + 1) % cfg.log_every == 0:
                print(f"step {step + 1}/{cfg.steps} loss={run / max(nb, 1):.3f} "
                      f"{(time.time() - t0) / max(1, step + 1 - self.start):.2f}s/it {self.stats}", flush=True)
                run, nb = 0.0, 0
            if cfg.snapshot_every and (step + 1) % cfg.snapshot_every == 0:
                self.save(step + 1, snapshot=True)
            if cfg.wall_seconds and time.time() - t0 > cfg.wall_seconds:
                self.save(step + 1, snapshot=True)
                self.save(step + 1, snapshot=False)
                print(f"wall stop at step {step + 1}", flush=True)
                return self.path()
        self.save(cfg.steps, snapshot=False)
        print(f"finished {cfg.steps} steps -> {self.path()}", flush=True)
        return self.path()


def package(cfg: FinetuneConfig, text_head: str, object_head: str, out: str) -> VarestCheckpoint:
    """SWA of the last four snapshots before ``cfg.steps`` and the final weights, packaged with both heads."""
    base = os.path.join(cfg.output_dir, cfg.tag)
    srcs = [f"{base}_step{s}.pt" for s in swa_steps(cfg.steps, cfg.snapshot_every)] + [f"{base}.pt"]
    missing = [s for s in srcs if not os.path.exists(s)]
    if missing:
        raise FileNotFoundError(f"SWA sources missing: {missing}")
    swa = average_files(srcs)
    th = torch.load(text_head, map_location="cpu", weights_only=False)
    oh = torch.load(object_head, map_location="cpu", weights_only=False)
    ck = VarestCheckpoint.from_training(swa, variant=cfg.variant, backbone=cfg.backbone, text_head=th,
                                        object_head=oh, rel_layer=(swa["cfg"] or {}).get("rel_layer"))
    ck.save(out)
    return ck


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fine-tune LLaVA for W5 / W5RELJOINT.")
    ap.add_argument("--variant", required=True, choices=("w5", "w5reljoint"))
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--data-root", required=True, help="directory the corpus image paths are relative to")
    ap.add_argument("--mix-json", default=MIX_JSON,
                    help="corpus json (relative to --data-root, or absolute); LLaVA-NeXT-Data for LLaVA-NeXT")
    ap.add_argument("--fields", required=True, help="glob of varest.train.dump_fields shards for this backbone family")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps", type=int, default=624609)
    ap.add_argument("--wall-seconds", type=int, default=0)
    ap.add_argument("--snapshot-every", type=int, default=32000)
    ap.add_argument("--only-cached", action="store_true", help="train only on images the field cache covers")
    ap.add_argument("--init-from", default="", help="start from the weights of this trainer checkpoint")
    ap.add_argument("--package", nargs=3, metavar=("TEXT_HEAD", "OBJECT_HEAD", "OUT"),
                    help="skip training: SWA the finished run and write the release checkpoint")
    a = ap.parse_args(argv)
    cfg = FinetuneConfig(variant=a.variant, backbone=a.backbone, data_root=a.data_root, fields=a.fields, mix_json=a.mix_json,
                         output_dir=a.output_dir, tag=a.tag, steps=a.steps, wall_seconds=a.wall_seconds,
                         snapshot_every=a.snapshot_every, only_cached=a.only_cached, init_from=a.init_from)
    if a.package:
        package(cfg, *a.package)
        print(f"wrote {a.package[2]}")
        return
    Trainer(cfg).train()


if __name__ == "__main__":
    main()
