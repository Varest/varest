"""End-to-end W5 / W5RELJOINT inference on a Hugging Face LLaVA model: one image, one prompt, one budget.

The engine owns the model-side changes (LoRA, projector) and assembles the language-model input EXACTLY as the
checkpoints were trained:

  LLaVA-1.5   [BOS <system> USER: ] [visual] ["\\n" prompt] ([REL] x R) [" ASSISTANT:"]
  LLaVA-NeXT  [BOS <|start_header_id|>user<|end_header_id|>\\n\\n] [visual] ["\\n" prompt] ([REL] x R)
              [<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n]

Each bracket is tokenized separately (that is the trained sequence; with [REL] the assistant suffix is its own
tokenization). Without [REL] decoding is Hugging Face greedy ``generate`` on the embeddings; with [REL] it is
:func:`varest.rel.generate_pruned`, which deletes the visual tokens [REL] ranks outside its top K' at the checkpoint's
[REL] layer.

Adaptive halting (``halt=HaltConfig(...)``, :mod:`varest.halt`) deletes every visual token at the first layer whose
logit lens has settled; it composes with [REL] and also runs on W5 (then through the same per-layer decoder, which
without halting is not used for W5). ``halt=None`` (default) leaves decoding exactly as released.

Budgets: LLaVA-1.5 takes the layer-averaged token budget B (32/64/128/224); LLaVA-NeXT takes a retain ratio r
(0.0556/0.1111/0.2222/0.3889), applied per image as ``floor(N_native * r)``.
"""
from __future__ import annotations

import os

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from varest import next as vnext
from varest.checkpoint import VarestCheckpoint
from varest.halt import HaltConfig
from varest.rel import generate_pruned
from varest.runtime import (LoraAttachment, ProjectorSwap, VarestRuntime, VisualTokens, attach_lora, detach_lora,
                            restore_projector, swap_projector, to_pil)

LLAVA_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. ")
NEXT_USER = "<|start_header_id|>user<|end_header_id|>\n\n"
NEXT_ASSISTANT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def family_of(model: nn.Module) -> str:
    name = type(model).__name__
    if "LlavaNext" in name:
        return "llava_next"
    if "Llava" in name:
        return "llava"
    raise TypeError(f"Varest supports LLaVA-1.5 and LLaVA-NeXT, got {name}")


class VarestEngine:
    """Attach a :class:`VarestCheckpoint` to ``model`` and generate with it. ``attach()`` / ``detach()`` are exact
    inverses; the model is otherwise untouched."""

    def __init__(self, model: nn.Module, processor: Any, ckpt: VarestCheckpoint, halt: HaltConfig | None = None) -> None:
        self.model = model
        self.halt = halt
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.ckpt = ckpt
        self.family = family_of(model)
        want = "llava_next" if ckpt.backbone == "llama3-llava-next-8b" else "llava"
        if self.family != want:
            raise ValueError(f"checkpoint {ckpt.backbone!r} cannot run on a {self.family} model")
        cfg = model.config
        width = cfg.text_config.hidden_size
        if ckpt.postag["net.2.weight"].shape[0] != width:
            raise ValueError(f"checkpoint {ckpt.backbone!r} is {ckpt.postag['net.2.weight'].shape[0]}-wide, "
                             f"the model is {width}-wide")
        self.runtime = VarestRuntime(ckpt)
        self._lora: LoraAttachment | None = None
        self._proj: ProjectorSwap | None = None

    # ------------------------------------------------------------------------------------------ lifecycle
    @property
    def attached(self) -> bool:
        return self._lora is not None

    def attach(self) -> VarestEngine:
        if self.attached:
            return self
        p = next(self.model.parameters())
        self._lora = attach_lora(self.model, self.ckpt)
        self._proj = swap_projector(self.model, self.ckpt)
        self.runtime.place(p.device, p.dtype)
        return self

    def detach(self) -> None:
        if self._proj is not None:
            restore_projector(self._proj)
            self._proj = None
        if self._lora is not None:
            detach_lora(self._lora)
            self._lora = None

    # ------------------------------------------------------------------------------------------ pieces
    @property
    def _inner(self) -> nn.Module:
        return self.model.model

    @property
    def _dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def _embed(self, text: str, special: bool) -> torch.Tensor:
        ids = self.tokenizer(text, add_special_tokens=special, return_tensors="pt").input_ids
        return self.model.get_input_embeddings()(ids.to(self.model.device))

    @torch.no_grad()
    def _project(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[P, 3, 336, 336]`` -> projected patch features ``[P, 576, D]`` (CLS dropped)."""
        vfl = self.model.config.vision_feature_layer
        out = self._inner.vision_tower(pixel_values.to(self.model.device, self._dtype), output_hidden_states=True)
        return self._inner.multi_modal_projector(out.hidden_states[vfl][:, 1:])

    @torch.no_grad()
    def visual_tokens(self, image: Any, budget: float) -> VisualTokens:
        pil = to_pil(image)
        # The fields get the ORIGINAL input when it is a file: varest.fields then decodes the object head's input with
        # PIL's JPEG draft, as the training/evaluation caches were. Only the CLIP input needs the full decode.
        src = image if isinstance(image, (str, os.PathLike)) else pil
        if self.family == "llava":
            pv = self.processor.image_processor(pil, return_tensors="pt")["pixel_values"]
            return self.runtime.llava_tokens(self._project(pv)[0], src, int(budget))
        o = self.processor.image_processor(pil, return_tensors="pt")
        w, h = pil.size
        if [int(v) for v in o["image_sizes"][0]] != [h, w]:
            raise ValueError(f"processor image_sizes {o['image_sizes'][0].tolist()} != (H, W) = ({h}, {w})")
        lay = vnext.layout(w, h)
        f = self._project(o["pixel_values"][0, :1 + lay.nh * lay.nw])
        return self.runtime.next_tokens(f[0], vnext.stitch(f[1:], lay), lay, src, float(budget))

    # ------------------------------------------------------------------------------------------ decoding
    @torch.no_grad()
    def generate(self, image: Any, prompt: str, budget: float, max_new_tokens: int,
                 generation_profiler: Any | None = None) -> tuple[list[int], dict]:
        """-> (generated token ids without the prompt, info)."""
        if not self.attached:
            raise RuntimeError("VarestEngine.generate before attach()")
        vt = self.visual_tokens(image, budget)
        if self.family == "llava":
            pre = self._embed(LLAVA_SYSTEM + "USER: ", True)
            tail, eos = " ASSISTANT:", self.tokenizer.eos_token_id
        else:
            pre = self._embed(NEXT_USER, True)
            tail, eos = NEXT_ASSISTANT, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        vis = vt.embeds.unsqueeze(0)
        measure = getattr(generation_profiler, "measure_generate", None)
        ctx = measure(self.model) if callable(measure) else nullcontext()
        info = {"visual_tokens": vt.n, **vt.info}
        with ctx:
            if self.runtime.rel_head is None and self.halt is not None:
                if self.family == "llava":
                    post = self._embed("\n" + prompt + tail, False)
                else:
                    post = torch.cat([self._embed("\n" + prompt, False), self._embed(tail, False)], 1)
                e = torch.cat([pre, vis.to(post.dtype), post], 1)
                a0 = pre.shape[1]
                text_model = getattr(self._inner, "language_model", self._inner)
                ids, ri = generate_pruned(text_model, self.model.lm_head, self.model.get_input_embeddings(), None, e,
                                          (a0, a0 + vt.n, a0 + vt.n, 0), 0, None, max_new_tokens, int(eos),
                                          halt=self.halt)
                info.update({"halt_layer": ri.get("halt_layer"), "halt_dropped": ri.get("halt_dropped", 0)})
                return ids, info
            if self.runtime.rel_head is None:
                if self.family == "llava":
                    post = self._embed("\n" + prompt + tail, False)
                    kw = {}
                else:
                    post = torch.cat([self._embed("\n" + prompt, False), self._embed(tail, False)], 1)
                    kw = {"eos_token_id": eos, "pad_token_id": eos}
                e = torch.cat([pre, vis.to(post.dtype), post], 1)
                out = self.model.generate(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], device=e.device,
                                                                                      dtype=torch.long),
                                          max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, **kw)
                ids = [int(t) for t in out[0]]
                return ids, info
            pq = self._embed("\n" + prompt, False)
            asst = self._embed(tail, False)
            rt = self.runtime.rel_head.tokens.to(pq.dtype).unsqueeze(0)
            e = torch.cat([pre, vis.to(pq.dtype), pq, rt, asst], 1)
            a0 = pre.shape[1]
            b0 = a0 + vt.n
            text_model = getattr(self._inner, "language_model", self._inner)
            ids, ri = generate_pruned(text_model, self.model.lm_head, self.model.get_input_embeddings(),
                                      self.runtime.rel_head, e, (a0, b0, b0 + pq.shape[1], rt.shape[1]),
                                      int(vt.keep), int(self.runtime.rel_layer), max_new_tokens, int(eos),
                                      halt=self.halt)
            info.update({"rel_kept": ri.get("kept"), "rel_layer": self.runtime.rel_layer})
            if self.halt is not None:
                info.update({"halt_layer": ri.get("halt_layer"), "halt_dropped": ri.get("halt_dropped", 0)})
            return ids, info
