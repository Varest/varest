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

Preprocessing (``preprocess=``, default from ``VAREST_PREPROCESS``, else ``"pil"``): ``"pil"`` decodes and resizes on the
CPU with PIL exactly as the checkpoints were trained; ``"gpu"`` keeps PIL's decode, uploads the image once and does every
conversion and resize on the device (:mod:`varest.gpuimage`, within one grey level of PIL), with the CLIP input built by
the same Hugging Face processor there; ``"gpu-nvjpeg"`` also decodes JPEG files with nvJPEG (faster, less faithful: see
:mod:`varest.gpuimage`); ``"gpu-fields"`` moves only the field heads' inputs to the device and builds the CLIP input
on the CPU exactly as ``"pil"`` does.

Budgets: LLaVA-1.5 takes the layer-averaged token budget B (32/64/128/224); LLaVA-NeXT takes a retain ratio r
(0.0556/0.1111/0.2222/0.3889), applied per image as ``floor(N_native * r)``.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from contextlib import nullcontext
from typing import Any

import torch
from PIL import Image
from torch import nn

from varest import next as vnext
from varest.checkpoint import VarestCheckpoint
from varest.halt import HaltConfig
from varest.fields import llava_fields, next_fields
from varest.gpuimage import DeviceImage, as_uint8_chw
from varest.graphs import GraphedCall
from varest.rel import generate_pruned
from varest.runtime import (LoraAttachment, ProjectorSwap, VarestRuntime, VisualTokens, attach_lora, detach_lora,
                            restore_projector, swap_projector, to_pil)

LLAVA_SYSTEM = ("A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. ")
NEXT_USER = "<|start_header_id|>user<|end_header_id|>\n\n"
NEXT_ASSISTANT = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


PREPROCESS_MODES = ("pil", "gpu", "gpu-nvjpeg", "gpu-fields")


def speed_options(preprocess: str | None = None, vision_graph: bool | None = None, decode_graph: bool | None = None
                  ) -> tuple[str, bool, bool]:
    """The engine's opt-in speed options; an argument left ``None`` is read from the environment. ALL DEFAULT OFF, which
    is the released behaviour and outputs:

    ``preprocess`` / ``VAREST_PREPROCESS``   ``"pil"`` (default): PIL on the CPU, as trained. ``"gpu-fields"``: the field
        heads' inputs on the GPU, the CLIP input on the CPU exactly as ``"pil"``. ``"gpu"``: the CLIP input on the GPU
        too. ``"gpu-nvjpeg"``: also nvJPEG for JPEG files. The GPU modes match PIL to within one grey level, not bit for
        bit (see :mod:`varest.gpuimage`).
    ``vision_graph`` / ``VAREST_VISION_GRAPH=1``   CLIP + projector replayed from a CUDA graph (:mod:`varest.graphs`):
        the same kernels, the same outputs, and the CPU is free for the field heads and the partition meanwhile.
    ``decode_graph`` / ``VAREST_DECODE_GRAPH=1``   greedy decoding replayed from a CUDA graph (:mod:`varest.graphdecode`):
        faithful, not bit-identical (FlashAttention's kv-cache kernel replaces SDPA in the decode step); W5 then
        decodes through the same per-layer loop as [REL]."""
    preprocess = preprocess or os.environ.get("VAREST_PREPROCESS", "pil")
    if preprocess not in PREPROCESS_MODES:
        raise ValueError(f"preprocess must be one of {PREPROCESS_MODES}, got {preprocess!r}")
    if vision_graph is None:
        vision_graph = os.environ.get("VAREST_VISION_GRAPH", "0") == "1"
    if decode_graph is None:
        decode_graph = os.environ.get("VAREST_DECODE_GRAPH", "0") == "1"
    return preprocess, bool(vision_graph), bool(decode_graph)


_WORKER: ThreadPoolExecutor | None = None


def _worker() -> ThreadPoolExecutor:
    global _WORKER
    if _WORKER is None:
        _WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="varest-clip-input")
    return _WORKER


def _clip_cpu_input(image: Any) -> Any:
    """What the PIL path hands the CLIP processor, or the uint8 tensor the processor turns into the same pixel values."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        return as_uint8_chw(image).cpu()
    return to_pil(image)


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

    def __init__(self, model: nn.Module, processor: Any, ckpt: VarestCheckpoint, halt: HaltConfig | None = None,
                 preprocess: str | None = None, vision_graph: bool | None = None,
                 decode_graph: bool | None = None) -> None:
        self.preprocess, self.vision_graph, self.decode_graph = speed_options(preprocess, vision_graph, decode_graph)
        self._vision: GraphedCall | None = None
        self._decoder = None
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
        # the field heads' weights are now enqueued on the current stream; the side stream waits for THIS point only
        self._ready = torch.cuda.Event() if p.device.type == "cuda" else None
        if self._ready is not None:
            self._ready.record()
        return self

    def detach(self) -> None:
        self._vision = None                               # the graphs replay the attached LoRA's kernels and weights
        self._decoder = None
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
        pv = pixel_values.to(self.model.device, self._dtype)
        if self.vision_graph and pv.is_cuda:
            if self._vision is None:
                self._vision = GraphedCall(self._project_eager)
            return self._vision(pv)
        return self._project_eager(pv)

    def _project_eager(self, pv: torch.Tensor) -> torch.Tensor:
        vfl = self.model.config.vision_feature_layer
        out = self._inner.vision_tower(pv, output_hidden_states=True)
        return self._inner.multi_modal_projector(out.hidden_states[vfl][:, 1:])

    def _graph_decoder(self):
        if not self.decode_graph or self.model.device.type != "cuda":
            return None
        if self._decoder is None:
            from varest.graphdecode import GraphDecoder
            tm = getattr(self._inner, "language_model", self._inner)
            cap = min(int(getattr(tm.config, "max_position_embeddings", 4096)), 8192)
            self._decoder = GraphDecoder(tm, self.model.lm_head, self.model.get_input_embeddings(), cap)
        return self._decoder

    def _side(self, device: torch.device, after: torch.cuda.Event | None = None):
        """A side CUDA stream for the two field heads, so they run while CLIP still occupies the default stream (the
        fields' copy to the host then waits for the heads only, and the partition is built on the CPU during CLIP).
        Same kernels, same inputs: the outputs are unchanged. No-op off CUDA."""
        if device.type != "cuda":
            return nullcontext()
        if getattr(self, "_stream", None) is None or self._stream.device != device:
            self._stream = torch.cuda.Stream(device=device)
        if getattr(self, "_ready", None) is not None:
            self._stream.wait_event(self._ready)          # the heads' weights (placed at attach), NOT the CLIP just queued
        if after is not None:
            self._stream.wait_event(after)                # the device-side decode the heads read
        return torch.cuda.stream(self._stream)

    @torch.no_grad()
    def visual_tokens(self, image: Any, budget: float) -> VisualTokens:
        device = self.model.device
        gpu = self.preprocess != "pil" and device.type == "cuda" and (
            isinstance(image, (str, Image.Image, torch.Tensor)) or hasattr(image, "__fspath__"))
        vt = self._visual_tokens_gpu(image, budget, device) if gpu else self._visual_tokens_pil(image, budget)
        vt.info["preprocess"] = "gpu" if gpu else "pil"         # which path ran, recorded with every sample
        return vt

    @torch.no_grad()
    def _visual_tokens_pil(self, image: Any, budget: float) -> VisualTokens:
        gray = None
        if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
            # ONE decode of the file for CLIP (RGB) and the text head (L): converting the same decoded image twice is
            # pixel-identical to opening the file twice, as before.
            with Image.open(image) as im:
                im.load()
                pil = im.convert("RGB")
                gray = im.convert("L") if self.family == "llava" else None
        else:
            pil = to_pil(image)
        # The fields get the ORIGINAL input when it is a file: varest.fields then decodes the object head's input with
        # PIL's JPEG draft, as the training/evaluation caches were. Only the CLIP input needs the full decode.
        src = image if isinstance(image, (str, os.PathLike)) else pil
        rt = self.runtime
        if self.family == "llava":
            pv = self.processor.image_processor(pil, return_tensors="pt")["pixel_values"]
            proj = self._project(pv)[0]                                   # enqueued; runs while the fields are built
            with self._side(proj.device):
                f = llava_fields(src, rt.text_head, rt.object_head, gray=gray)
            return rt.llava_tokens(proj, src, int(budget), fields=f)
        o = self.processor.image_processor(pil, return_tensors="pt")
        w, h = pil.size
        if [int(v) for v in o["image_sizes"][0]] != [h, w]:
            raise ValueError(f"processor image_sizes {o['image_sizes'][0].tolist()} != (H, W) = ({h}, {w})")
        lay = vnext.layout(w, h)
        f = self._project(o["pixel_values"][0, :1 + lay.nh * lay.nw])
        with self._side(f.device):
            fl = next_fields(src, rt.text_head, rt.object_head, (lay.nr, lay.nc))
        return rt.next_tokens(f[0], vnext.stitch(f[1:], lay), lay, src, float(budget), fields=fl)

    def _visual_tokens_gpu(self, image: Any, budget: float, device: torch.device) -> VisualTokens:
        """:meth:`visual_tokens` with GPU preprocessing: one device-side image (a file decoded there, or a decoded image
        uploaded once) feeds both field heads, and -- except with ``"gpu-fields"`` -- CLIP.

        ``"gpu-fields"`` builds the CLIP input on the CPU exactly as the PIL path does (the processor on the same pixels:
        a uint8 tensor is passed as is, which the processor turns into the same pixel values as its PIL image), on a
        worker thread while the field heads run; CLIP is launched as soon as it is ready, before the partition."""
        if isinstance(image, Image.Image):
            img = DeviceImage.from_pil(image, device)
        elif isinstance(image, torch.Tensor):
            img = DeviceImage.from_tensor(image, device)
        else:
            img = DeviceImage.open(image, device, nvjpeg=self.preprocess == "gpu-nvjpeg")
        decoded = torch.cuda.Event()
        decoded.record()
        rt = self.runtime
        exact_clip = self.preprocess == "gpu-fields"
        if exact_clip:
            pending = _worker().submit(self.processor.image_processor, _clip_cpu_input(image), return_tensors="pt")
        else:
            o = self.processor.image_processor(img.rgb.to(torch.uint8), return_tensors="pt", device=device)
        if self.family == "llava":
            if exact_clip:
                with self._side(device, decoded):
                    f = llava_fields(img, rt.text_head, rt.object_head)
                proj = self._project(pending.result()["pixel_values"])[0]
            else:
                proj = self._project(o["pixel_values"])[0]
                with self._side(device, decoded):
                    f = llava_fields(img, rt.text_head, rt.object_head)
            return rt.llava_tokens(proj, img, int(budget), fields=f)
        w, h = img.size
        lay = vnext.layout(w, h)
        if exact_clip:
            with self._side(device, decoded):
                fl = next_fields(img, rt.text_head, rt.object_head, (lay.nr, lay.nc))
            o = pending.result()
        if [int(v) for v in o["image_sizes"][0]] != [h, w]:
            raise ValueError(f"processor image_sizes {o['image_sizes'][0].tolist()} != (H, W) = ({h}, {w})")
        f = self._project(o["pixel_values"][0, :1 + lay.nh * lay.nw])
        if not exact_clip:
            with self._side(device, decoded):
                fl = next_fields(img, rt.text_head, rt.object_head, (lay.nr, lay.nc))
        return rt.next_tokens(f[0], vnext.stitch(f[1:], lay), lay, img, float(budget), fields=fl)

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
            if self.runtime.rel_head is None and (self.halt is not None or self.decode_graph):
                if self.family == "llava":
                    post = self._embed("\n" + prompt + tail, False)
                else:
                    post = torch.cat([self._embed("\n" + prompt, False), self._embed(tail, False)], 1)
                e = torch.cat([pre, vis.to(post.dtype), post], 1)
                a0 = pre.shape[1]
                text_model = getattr(self._inner, "language_model", self._inner)
                ids, ri = generate_pruned(text_model, self.model.lm_head, self.model.get_input_embeddings(), None, e,
                                          (a0, a0 + vt.n, a0 + vt.n, 0), 0, None, max_new_tokens, int(eos),
                                          halt=self.halt, decoder=self._graph_decoder())
                if self.halt is not None:
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
                                      halt=self.halt, decoder=self._graph_decoder())
            info.update({"rel_kept": ri.get("kept"), "rel_layer": self.runtime.rel_layer})
            if self.halt is not None:
                info.update({"halt_layer": ri.get("halt_layer"), "halt_dropped": ri.get("halt_dropped", 0)})
            return ids, info
