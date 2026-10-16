"""The Varest release format: ONE self-contained file per (backbone, variant).

    {"method": "Varest", "format_version": 1,
     "variant":  "w5" | "w5reljoint",
     "backbone": "llava-1.5-7b" | "llava-1.5-13b" | "llama3-llava-next-8b",
     "model_id": the Hugging Face id the adapter was trained on,
     "lora": {name: tensor}, "lora_rank": r, "lora_alpha": a, "lora_targets": [leaf module names],
     "projector": {name: tensor},            # the fine-tuned multimodal projector
     "postag": {name: tensor},               # varest.postag.PosTag
     "relhead": {name: tensor} | None,       # varest.rel.RelHead (w5reljoint only)
     "rel_layer": int | None,                # decoder layer at which [REL] prunes
     "text_head": {"sd": ..., <config>},     # varest.heads.text.TextHead (v5n)
     "object_head": {"sd": ..., <config>},   # varest.heads.object.ObjectHead (w96h288)
     "train_cfg": {...}}                     # the recipe the weights were trained with

Parameter names are relative to the Hugging Face ``Llava*ForConditionalGeneration`` module (e.g.
``model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight``, ``model.multi_modal_projector.linear_1.weight``).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from typing import Any

import torch

METHOD = "Varest"
FORMAT_VERSION = 1
VARIANTS = ("w5", "w5reljoint")
BACKBONES = {"llava-1.5-7b": "llava-hf/llava-1.5-7b-hf", "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
             "llama3-llava-next-8b": "llava-hf/llama3-llava-next-8b-hf"}
_PEFT_PREFIX = "base_model.model."


def _strip(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {(k[len(_PEFT_PREFIX):] if k.startswith(_PEFT_PREFIX) else k): v for k, v in sd.items()}


@dataclass
class VarestCheckpoint:
    variant: str
    backbone: str
    lora: dict[str, torch.Tensor]
    projector: dict[str, torch.Tensor]
    postag: dict[str, torch.Tensor]
    text_head: dict[str, Any]
    object_head: dict[str, Any]
    relhead: dict[str, torch.Tensor] | None = None
    rel_layer: int | None = None
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_targets: list[str] = field(default_factory=list)
    model_id: str = ""
    train_cfg: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown Varest variant {self.variant!r}; expected one of {VARIANTS}")
        if self.backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {self.backbone!r}; expected one of {sorted(BACKBONES)}")
        if (self.variant == "w5reljoint") != (self.relhead is not None):
            raise ValueError(f"variant {self.variant!r} {'needs' if self.variant == 'w5reljoint' else 'must not carry'}"
                             f" a [REL] head")
        if self.relhead is not None and self.rel_layer is None:
            raise ValueError("a [REL] checkpoint must record rel_layer")
        self.lora = _strip(self.lora)
        self.projector = _strip(self.projector)
        a_keys = [k for k in self.lora if ".lora_A." in k]
        if not a_keys:
            raise ValueError("checkpoint carries no LoRA weights")
        self.lora_rank = int(self.lora[a_keys[0]].shape[0])
        self.lora_targets = sorted({k.split(".lora_A.")[0].rsplit(".", 1)[-1] for k in a_keys})
        self.model_id = self.model_id or BACKBONES[self.backbone]

    @property
    def uses_rel(self) -> bool:
        return self.relhead is not None

    def to_dict(self) -> dict[str, Any]:
        return {"method": METHOD, "format_version": FORMAT_VERSION, "variant": self.variant,
                "backbone": self.backbone, "model_id": self.model_id, "lora": self.lora, "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha, "lora_targets": self.lora_targets, "projector": self.projector,
                "postag": self.postag, "relhead": self.relhead, "rel_layer": self.rel_layer,
                "text_head": self.text_head, "object_head": self.object_head, "train_cfg": self.train_cfg}

    def save(self, path: str | PathLike[str]) -> None:
        torch.save(self.to_dict(), path)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> VarestCheckpoint:
        if d.get("method") != METHOD:
            raise ValueError(f"not a Varest checkpoint (method={d.get('method')!r})")
        if int(d.get("format_version", 0)) != FORMAT_VERSION:
            raise ValueError(f"unsupported Varest checkpoint format {d.get('format_version')!r}")
        return cls(variant=d["variant"], backbone=d["backbone"], lora=dict(d["lora"]), projector=dict(d["projector"]),
                   postag=dict(d["postag"]), text_head=dict(d["text_head"]), object_head=dict(d["object_head"]),
                   relhead=d.get("relhead"), rel_layer=d.get("rel_layer"), lora_rank=int(d.get("lora_rank", 32)),
                   lora_alpha=int(d.get("lora_alpha", 64)), model_id=d.get("model_id", ""),
                   train_cfg=dict(d.get("train_cfg") or {}))

    @classmethod
    def load(cls, path: str | PathLike[str]) -> VarestCheckpoint:
        return cls.from_dict(torch.load(path, map_location="cpu", weights_only=False))

    @classmethod
    def from_training(cls, training: Mapping[str, Any], *, variant: str, backbone: str,
                      text_head: Mapping[str, Any], object_head: Mapping[str, Any],
                      rel_layer: int | None = None) -> VarestCheckpoint:
        """Package a trainer checkpoint ``{"lora", "proj", "postag", ["relhead"], "cfg"}`` with the two heads."""
        cfg = dict(training.get("cfg") or {})
        rel = training.get("relhead")
        if rel is not None and rel_layer is None:
            rel_layer = int(cfg.get("REL_LAYER", 16))
        return cls(variant=variant, backbone=backbone, lora=dict(training["lora"]), projector=dict(training["proj"]),
                   postag=dict(training["postag"]), relhead=dict(rel) if rel is not None else None,
                   rel_layer=rel_layer if rel is not None else None, text_head=dict(text_head),
                   object_head=dict(object_head), train_cfg=cfg)
