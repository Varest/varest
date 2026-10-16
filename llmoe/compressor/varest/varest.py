"""Varest (W5RELJOINT) and VarestW5 for LLaVA-1.5 7B/13B and LLaVA-NeXT-8B -- thin wrappers over :mod:`varest`.

All of the method lives in the ``varest`` package; this module only plugs :class:`varest.engine.VarestEngine` into the
harness. On ``install`` it resolves the released checkpoint for the loaded backbone (``methods/varest/<backbone>/...``
in the LLMOE bucket, or ``checkpoint_path``), attaches the checkpoint's LoRA and projector to the backbone, and routes
the backbone's ``generate`` through the engine, which builds the W5 visual tokens from the image and decodes the
prompt exactly as the checkpoint was trained. ``uninstall`` restores the backbone.

Budgets. ``max_tokens`` is the layer-averaged visual budget on LLaVA-1.5 (standard 32/64/128/224); ``retain_ratio``
is the per-image fraction of NeXT's native AnyRes length (standard 0.0556/0.1111/0.2222/0.3889) and is required on
LLaVA-NeXT. For W5RELJOINT the budget B splits into K = round(1.6 B) visual tokens before the LLM and K' = 2B - K
after [REL] prunes at the middle decoder layer; W5 feeds B tokens.

Adaptive halting (off by default): ``adaptive_halt: true`` deletes every visual token at the first decoder layer
``>= halt_min_layer`` whose logit lens (last prompt position, base ``lm_head``) has kept the same top-1 for
``halt_window`` layers with entropy ``<= halt_entropy`` nats (or top-1 probability ``>= halt_maxp``). Each record's
metadata carries ``halt_layer`` (None when it never fired) for the realised saving. See :mod:`varest.halt`.

Decoding runs one row at a time (the in-LLM prune is per example), greedily, as all LLMOE benchmarks do.
"""
from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, override

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.weights import backbone_key

from varest.checkpoint import VarestCheckpoint
from varest.engine import VarestEngine
from varest.halt import HaltConfig

SLUGS = {"llava": "llava-1.5-7b", "llava-13b": "llava-1.5-13b", "llava-next-8b": "llama3-llava-next-8b"}
DEFAULT_MAX_TOKENS = 64


class VarestCompressor(TokenCompressor):
    """W5RELJOINT: the W5 variable-resolution tokenizer plus [REL] pruning inside the LLM, trained jointly."""

    variant = "w5reljoint"
    method_key = "Varest"

    def __init__(self, *, max_tokens: int | None = None, retain_ratio: float | None = None,
                 checkpoint_path: str | PathLike[str] | None = None,
                 pretrained: bool | str | PathLike[str] | None = None, for_training: bool = False,
                 adaptive_halt: bool = False, halt_min_layer: int = 8, halt_window: int = 3,
                 halt_entropy: float | None = 1.0, halt_maxp: float | None = None) -> None:
        if retain_ratio is not None:
            retain_ratio = float(retain_ratio)
            if not 0.0 < retain_ratio <= 1.0:
                raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
        if max_tokens is not None and int(max_tokens) < 3:
            raise ValueError(f"{self.method_key} needs max_tokens >= 3, got {max_tokens}")
        super().__init__(int(max_tokens) if max_tokens is not None else DEFAULT_MAX_TOKENS)
        self._explicit_max_tokens = max_tokens is not None
        self._retain_ratio = retain_ratio
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._pretrained_arg = pretrained
        self.for_training = bool(for_training)
        self._vlm: VLM | None = None
        self._engine: VarestEngine | None = None
        self._orig_prepare_inputs = None
        self._orig_generate = None
        self._last_info: list[dict[str, Any]] = []
        if adaptive_halt:
            if halt_maxp is not None:
                halt_entropy = None
            self.halt: HaltConfig | None = HaltConfig(min_layer=int(halt_min_layer), window=int(halt_window),
                                                      entropy=None if halt_entropy is None else float(halt_entropy),
                                                      maxp=None if halt_maxp is None else float(halt_maxp))
        else:
            self.halt = None

    # -------------------------------------------------------------------------------------------- budget
    @property
    @override
    def budget(self) -> int | float:
        return self._retain_ratio if self._retain_ratio is not None else self._max_tokens

    def validate_llava_next(self) -> None:
        if self._retain_ratio is None:
            raise ValueError(f"{self.method_key} on LLaVA-NeXT needs retain_ratio (b/576 for the standard b); the "
                             "number of visual tokens differs per image")
        if self._explicit_max_tokens:
            raise ValueError(f"{self.method_key} on LLaVA-NeXT takes retain_ratio, not max_tokens")

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, **kwargs) -> CompressedFeatures:
        raise NotImplementedError(f"{self.method_key} builds its tokens from the image inside generate(); install() "
                                  "it on a backbone instead of calling compress()")

    # -------------------------------------------------------------------------------------------- weights
    def resolve_checkpoint(self, vlm: VLM) -> VarestCheckpoint:
        from llmoe.weights import resolve_weights
        key = backbone_key(vlm)
        if key not in SLUGS:
            raise ValueError(f"{self.method_key} supports {sorted(SLUGS)}, not backbone {key!r}")
        path = self.checkpoint_path
        if path is None:
            resolved = resolve_weights(self.method_key, int(self._max_tokens), pretrained=self._pretrained_arg,
                                       backbone=key)
            if resolved is None:
                raise ValueError(f"{self.method_key} has no untrained form: pass checkpoint_path, or leave "
                                 "`pretrained` unset / true to fetch the released weights")
            path = Path(resolved)
        ckpt = VarestCheckpoint.load(path)
        if ckpt.variant != self.variant:
            raise ValueError(f"{path} is a {ckpt.variant!r} checkpoint; {type(self).__name__} runs {self.variant!r}")
        if ckpt.backbone != SLUGS[key]:
            raise ValueError(f"{path} was trained for {ckpt.backbone}, not {SLUGS[key]}")
        self.checkpoint_path = path
        return ckpt

    # -------------------------------------------------------------------------------------------- lifecycle
    @override
    def install(self, vlm: VLM) -> None:
        if self._vlm is vlm:
            return
        if self._vlm is not None:
            raise RuntimeError(f"{self.method_key} is already installed on another backbone")
        is_next = getattr(vlm, "compressor_family", None) == "llava_next"
        if is_next:
            self.validate_llava_next()
        elif self._retain_ratio is not None:
            raise ValueError(f"{self.method_key} takes max_tokens on LLaVA-1.5; retain_ratio is for LLaVA-NeXT")
        ckpt = self.resolve_checkpoint(vlm)
        self._engine = VarestEngine(vlm.model, vlm.processor, ckpt, halt=self.halt).attach()
        self._vlm = vlm
        self._orig_prepare_inputs = vlm.prepare_inputs
        self._orig_generate = vlm.generate
        prepare = self._orig_prepare_inputs

        def prepare_inputs(images, prompts, **kwargs):
            inputs = prepare(images, prompts, **kwargs)
            inputs._varest_rows = (list(images), list(prompts))
            return inputs

        vlm.prepare_inputs = prepare_inputs
        vlm.generate = self._generate
        vlm.vision_tokens_per_layer = self.vision_tokens_per_layer

    def uninstall(self) -> None:
        if self._vlm is None:
            return
        self._vlm.prepare_inputs = self._orig_prepare_inputs
        self._vlm.generate = self._orig_generate
        self._vlm.__dict__.pop("vision_tokens_per_layer", None)
        if self._engine is not None:
            self._engine.detach()
        self._engine = None
        self._vlm = None
        self._orig_prepare_inputs = self._orig_generate = None

    # -------------------------------------------------------------------------------------------- decoding
    def _generate(self, inputs: Any, *, max_new_tokens: int = 256, generation_profiler: Any | None = None,
                  **kwargs) -> torch.Tensor:
        rows = getattr(inputs, "_varest_rows", None)
        if rows is None:
            raise RuntimeError(f"{self.method_key}: generate() received inputs that did not come through "
                               "prepare_inputs(); the image and prompt of each row are needed")
        if kwargs.get("do_sample") or int(kwargs.get("num_beams", 1) or 1) != 1:
            raise ValueError(f"{self.method_key} decodes greedily (as it was trained and evaluated); got "
                             f"do_sample={kwargs.get('do_sample')} num_beams={kwargs.get('num_beams')}")
        images, prompts = rows
        engine = self._engine
        tok = self._vlm.processor.tokenizer
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        outs, self._last_info = [], []
        with torch.inference_mode():
            for image, prompt in zip(images, prompts):
                ids, info = engine.generate(image, prompt, self.budget, max_new_tokens,
                                            generation_profiler=generation_profiler)
                outs.append(ids)
                self._last_info.append(info)
        prompt_ids = inputs["input_ids"]
        width = max((len(o) for o in outs), default=0)
        gen = torch.full((len(outs), width), int(pad), dtype=prompt_ids.dtype, device=prompt_ids.device)
        for i, o in enumerate(outs):
            if o:
                gen[i, :len(o)] = torch.tensor(o, dtype=prompt_ids.dtype, device=prompt_ids.device)
        return torch.cat([prompt_ids, gen], 1)

    def vision_tokens_per_layer(self) -> list[int] | None:
        """Vision tokens entering each decoder layer in the last generate call (one row), from the engine's info."""
        if self._engine is None or len(getattr(self, "_last_info", [])) != 1:
            return None
        text_model = getattr(self._engine._inner, "language_model", self._engine._inner)
        return vision_tokens_from_info(self._last_info[0], len(text_model.layers))

    @override
    def get_benchmark_record_metadata(self, batch_size: int) -> list[dict[str, Any]]:
        rows = [dict(r) for r in self._last_info[:batch_size]]
        return rows + [{} for _ in range(batch_size - len(rows))]


def vision_tokens_from_info(info: dict[str, Any], num_layers: int) -> list[int]:
    """Per-layer vision-token counts of one Varest generate call, from ``VarestEngine.generate``'s ``info``.

    All ``visual_tokens`` enter the decoder; with [REL], layers from ``rel_layer`` on hold ``rel_kept``
    (``varest.rel.generate_pruned`` prunes before running that layer); adaptive halting removes every visual token from
    ``halt_layer`` on. The [REL] registers are not vision tokens and are not counted.
    """
    counts = [int(info["visual_tokens"])] * int(num_layers)
    rel_layer, kept = info.get("rel_layer"), info.get("rel_kept")
    if rel_layer is not None and kept is not None:
        counts[int(rel_layer):] = [int(kept)] * (int(num_layers) - int(rel_layer))
    halt_layer = info.get("halt_layer")
    if halt_layer is not None:
        counts[int(halt_layer):] = [0] * (int(num_layers) - int(halt_layer))
    return counts


class VarestW5Compressor(VarestCompressor):
    """W5: the W5 variable-resolution tokenizer alone (no in-LLM pruning)."""

    variant = "w5"
    method_key = "VarestW5"
