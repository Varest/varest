"""FastV: attention-guided visual-token pruning in the language decoder.

FastV ranks image tokens by the head-averaged last-query attention at decoder
layer ``k - 1`` and physically removes the lowest-ranked tokens before layer
``k``. The budget, ranking, keep-set construction, and source tracking in this
module are backbone-independent. LLaVA and LLaVA-NeXT integrations also live
here; Qwen's thin M-RoPE integration lives in ``fastv_qwen.py`` and uses the
same policy functions.

Paper: Chen et al., "An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play
Inference Acceleration for Large Vision-Language Models" (arXiv:2403.06764).
"""

from collections.abc import Callable
from typing import Any, override

import torch
from torch.utils.hooks import RemovableHandle

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    advance_decode_step,
    apply_prune_state,
    gather_position_ids,
    keep_columns,
    pruned_attention_mask,
)
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.utils.depth import REFERENCE_DEPTH, LayerSchedule
from llmoe.utils.qwen import resolve_image_token_id as shared_resolve_image_token_id

try:  # RoPE helper for the layer k-1 attention recompute.
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as _APPLY_ROPE
except Exception:  # pragma: no cover - degrade to no-RoPE if the path moves
    _APPLY_ROPE = None


# Paper's AGG_LAYER for LLaVA-1.5: run layers 0..1 normally, score with layer 2's
# attention, prune immediately before layer 3.
DEFAULT_AGG_LAYER: int = 3
DEFAULT_MAX_TOKENS: int = 64

# LLaVA-1.5 geometry the budget calibration is solved against.
DEFAULT_VISUAL_TOKENS: int = 576
DEFAULT_NUM_DECODER_LAYERS: int = REFERENCE_DEPTH

# Per-budget ``(k, image tokens kept)``. As in every decoder-stage baseline here,
# the label is the visual-token count averaged over the decoder layers, so with
# layers ``0..k-1`` still carrying all 576 tokens the kept count is back-solved:
# ``keep = (32 * budget - k * 576) // (32 - k)``. Floor division lands each row
# just under its label (223.47 / 191.75 / 127.41 / 63.97 / 31.56); taking the
# label as the kept count instead realizes 257 / 228 / 170 / 112 / 83.
FASTV_PROFILES: dict[int, tuple[int, int]] = {
    224: (3, 187),
    192: (3, 152),
    128: (3, 81),
    64: (3, 11),
    # k=3 holds three full layers, flooring the average at 3 * 576 / 32 = 54, so
    # 32 is unreachable there and the cut moves to layer 1 -- the same shift
    # SparseVLM makes for its own 32 budget.
    32: (1, 14),
}

RopeFunction = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def resolve_image_token_id(config: Any, processor: Any) -> int:
    """Resolve the image placeholder id without hard-coding a tokenizer value."""
    return shared_resolve_image_token_id(config, processor, label="FastV")


def resolve_keep(n_tokens: int, *, max_tokens: int,
                 retain_ratio: float | None) -> int:
    """Resolve FastV's retained image-token count for one visual span."""
    keep = int(n_tokens * retain_ratio) if retain_ratio is not None else int(max_tokens)
    return max(1, min(int(n_tokens), keep))


def select_indices(scores: torch.Tensor, keep: int) -> torch.Tensor:
    """Return top-scoring indices in their original reading order."""
    return scores.topk(keep, dim=1).indices.sort(dim=1).values


def one_hot_source(indices: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Build the one-hot source matrix consumed by ``llmoe.viz``."""
    source = torch.zeros(indices.shape[0], n_tokens, dtype=torch.float32)
    source.scatter_(1, indices.detach().cpu().unsqueeze(1), 1.0)
    return source


def source_history(indices: torch.Tensor, n_tokens: int) -> list[torch.Tensor]:
    """Return FastV's pre-prune identity and post-prune source matrices."""
    return [
        torch.eye(n_tokens, dtype=torch.float32),
        one_hot_source(indices, n_tokens),
    ]


def normalize_attention_scores(attention: torch.Tensor, batch: int,
                               n_tokens: int) -> torch.Tensor:
    """Normalize scores or raw attention weights to FastV's ``[B, N]`` scores."""
    scores = attention.float()
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    elif scores.ndim == 3:
        scores = scores[:, -1, :]
    elif scores.ndim == 4:
        scores = scores.mean(dim=1)[:, -1, :]
    if scores.ndim != 2:
        raise ValueError(f"Unsupported attention shape {tuple(attention.shape)}")
    if scores.shape[0] == 1 and batch > 1:
        scores = scores.expand(batch, -1)
    if tuple(scores.shape) != (batch, n_tokens):
        raise ValueError(
            f"attention scores {tuple(scores.shape)} do not match the "
            f"{batch} x {n_tokens} tokens being compressed"
        )
    return scores


def last_query_attention(
    attn: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    attention_mask: torch.Tensor | None,
    *,
    apply_rope: RopeFunction | None,
    rope_args: tuple[Any, ...] = (),
) -> torch.Tensor:
    """Recompute FastV's last attention row without materializing the full map.

    LLaVA and Qwen use the same Q/K projection, GQA, masking, softmax, and head
    reduction. Only their rotary function differs, so it is injected here.
    """
    config = attn.config
    n_heads = int(config.num_attention_heads)
    n_kv = int(getattr(config, "num_key_value_heads", n_heads))
    head_dim = int(getattr(attn, "head_dim", hidden_states.shape[-1] // n_heads))
    batch, seq_len, _ = hidden_states.shape

    query = attn.q_proj(hidden_states).view(
        batch, seq_len, n_heads, head_dim
    )
    key = attn.k_proj(hidden_states).view(
        batch, seq_len, n_kv, head_dim
    )
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)
    if callable(q_norm):
        query = q_norm(query)
    if callable(k_norm):
        key = k_norm(key)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    if position_embeddings is not None and apply_rope is not None:
        cos, sin = position_embeddings
        query, key = apply_rope(query, key, cos, sin, *rope_args)
    if n_kv != n_heads:
        key = key.repeat_interleave(n_heads // n_kv, dim=1)

    scaling = float(getattr(attn, "scaling", head_dim ** -0.5))
    logits = (
        query[:, :, -1:, :].float() @ key.float().transpose(-1, -2)
    ) * scaling
    if (
        isinstance(attention_mask, torch.Tensor)
        and attention_mask.ndim == 4
        and attention_mask.shape[-1] == seq_len
    ):
        row = attention_mask[:, :, -1:, :]
        if row.dtype == torch.bool:
            logits = logits.masked_fill(~row, torch.finfo(logits.dtype).min)
        else:
            logits = logits + row.float()
    return logits.softmax(dim=-1).mean(dim=1)[:, 0, :]


def select_visual_keep(scores: torch.Tensor, image_positions: torch.Tensor,
                       keep: int, seq_len: int, *,
                       label: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local visual winners and the full prompt's ascending keep indices."""
    local = select_indices(scores.gather(1, image_positions), keep)
    kept_positions = image_positions.gather(1, local)
    keep_mask = torch.ones(
        image_positions.shape[0], seq_len, dtype=torch.bool,
        device=image_positions.device,
    )
    keep_mask.scatter_(1, image_positions, False)
    keep_mask.scatter_(1, kept_positions, True)
    return local, keep_columns(keep_mask, label=label)


class FastVCompressor(TokenCompressor):
    """Drop the image tokens layer ``k - 1`` attends to least at layer ``k``.

    ``max_tokens`` is the fixed LLaVA/InternVL budget. Dynamic-image backbones
    use ``retain_ratio``. On LLaVA-NeXT it is the decoder-layer-average visual
    fraction, and the paper's canonical ``k=3`` is fixed internally; no alternate
    schedule is exposed.
    """

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS, *, k: int | None = None,
                 retain_ratio: float | None = None,
                 track_source: bool = False) -> None:
        super().__init__(int(max_tokens))
        if self._max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if retain_ratio is not None and not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")

        # Only a profile-derived schedule is re-solved for another decoder depth;
        # an explicit `k` is the caller's own choice of layer, taken literally.
        self._profile_schedule: bool = k is None and retain_ratio is None
        # True when the caller asked for a boundary other than the paper's.
        self._k_override: bool = k is not None and int(k) != DEFAULT_AGG_LAYER
        if retain_ratio is not None:
            # The paper's own parameterization: a ratio at the paper's layer.
            k = DEFAULT_AGG_LAYER if k is None else k
            keep_tokens = self._max_tokens
        elif k is None:
            if self._max_tokens not in FASTV_PROFILES:
                supported = ", ".join(str(b) for b in sorted(FASTV_PROFILES))
                raise ValueError(
                    f"Unsupported FastV budget: {max_tokens}. Only {{{supported}}} have a "
                    f"calibrated profile; pass `k` to back-solve another, or `retain_ratio` for "
                    f"the paper's R. Reusing another budget's profile would run one budget "
                    f"under another's label."
                )
            k, keep_tokens = FASTV_PROFILES[self._max_tokens]
        else:
            keep_tokens = (DEFAULT_NUM_DECODER_LAYERS * self._max_tokens
                           - int(k) * DEFAULT_VISUAL_TOKENS) // (DEFAULT_NUM_DECODER_LAYERS - int(k))
            if keep_tokens < 1:
                raise ValueError(
                    f"A layer-averaged budget of {max_tokens} is unreachable with k={k}: the "
                    f"{k} unpruned layer(s) alone average "
                    f"{int(k) * DEFAULT_VISUAL_TOKENS / DEFAULT_NUM_DECODER_LAYERS:.1f} tokens."
                )
        if int(k) < 1:
            raise ValueError(
                "k must be >= 1; the ranking attention is read from layer k - 1"
            )
        # The schedule as published, i.e. against the 32-layer reference decoder.
        # `install` re-solves it for whatever depth the backbone actually has and
        # `uninstall` puts this back, so one compressor can serve both LLaVA sizes.
        self._reference_schedule = LayerSchedule(
            (int(k),), (int(keep_tokens),), REFERENCE_DEPTH, DEFAULT_VISUAL_TOKENS
        )
        self._schedule = self._reference_schedule
        self._k = int(k)
        self._keep_tokens = int(keep_tokens)
        self._retain_ratio = None if retain_ratio is None else float(retain_ratio)
        self._track_source = bool(track_source)

        self._image_positions: torch.Tensor | None = None
        self._attention_scores: torch.Tensor | None = None
        self._layer_state: dict[str, Any] = {}
        self._forward_count = 0
        self._warned_no_image = False

        self._installed_vlm: VLM | None = None
        self._language_model: Any | None = None
        self._hook_handles: list[RemovableHandle] = []
        self._image_token_id: int | None = None
        self._qwen: QwenPruningAdapter | None = None
        self._internvl: Any | None = None
        self._llava_next: bool = False

    @property
    def achieved_average(self) -> float:
        """Visual tokens averaged over the decoder, i.e. the budget actually run.

        Reads the schedule resolved for the installed backbone, so on a decoder
        that is not 32 layers deep this is the depth-corrected figure rather than
        the reference one.
        """
        return self._schedule.layer_average

    @property
    def budget(self) -> int | float:
        return self._max_tokens if self._retain_ratio is None else self._retain_ratio

    def validate_llava_next(self) -> None:
        """Accept only the paper layer and one layer-average ratio knob."""

        from llmoe.compressor.llava_next import validate_retain_ratio

        validate_retain_ratio(self._retain_ratio, label="FastV")
        if self._max_tokens != DEFAULT_MAX_TOKENS:
            raise ValueError(
                "FastV on LLaVA-NeXT only accepts retain_ratio; remove max_tokens "
                f"(its constructor default is {DEFAULT_MAX_TOKENS} and is ignored on NeXT)."
            )
        if self._k != DEFAULT_AGG_LAYER and not self._k_override:
            raise ValueError(
                f"FastV on LLaVA-NeXT fixes k={DEFAULT_AGG_LAYER}, matching the "
                "paper's attention layer; only retain_ratio is configurable."
            )
        if self._k_override:
            # Deliberate, per-arm deviation from the paper. `k` layers run
            # unpruned, so the reachable layer-average floor is about k/depth:
            # with the paper's k=3 on a 32-layer decoder nothing below ~0.094 is
            # expressible, putting the sweep's lowest point (0.0556) out of
            # reach. Moving the boundary earlier is the only way to express it,
            # and it also moves the ranking attention to an earlier layer -- so
            # these runs are NOT the paper's FastV and are reported as adapted.
            print(f"[FastV] ADAPTED VARIANT: k={self._k} (paper uses "
                  f"{DEFAULT_AGG_LAYER}) to make retain_ratio="
                  f"{float(self._retain_ratio):g} reachable", flush=True)

    def _resolve_keep(self, n_tokens: int) -> int:
        """Image tokens to keep out of ``n_tokens`` (the reference's ATTENTION_RANK)."""
        if self._llava_next:
            from llmoe.compressor.llava_next import layer_average_keep

            return layer_average_keep(
                n_tokens,
                retain_ratio=float(self._retain_ratio),
                full_layers=self._k,
                decoder_layers=len(self._language_model.layers),
                label="FastV/LLaVA-NeXT",
            )

        if self._retain_ratio is not None:
            return resolve_keep(
                n_tokens,
                max_tokens=self._keep_tokens,
                retain_ratio=self._retain_ratio,
            )
        if self._internvl is not None:
            return self._internvl.resolve_keep(n_tokens)
        return resolve_keep(
            n_tokens,
            max_tokens=self._keep_tokens,
            retain_ratio=None,
        )

    @staticmethod
    def _scores_from_attention(attention: torch.Tensor, batch: int,
                               n_tokens: int) -> torch.Tensor:
        """Compatibility wrapper around the shared FastV score normalizer."""
        return normalize_attention_scores(attention, batch, n_tokens)

    def _record_source(self, indices: torch.Tensor, n_tokens: int) -> None:
        if self._track_source:
            self._source_history = source_history(indices, n_tokens)

    @override
    def compress(self, features: torch.Tensor, *,
                 attention: torch.Tensor | None = None,
                 layer_idx: int | None = None,
                 **kwargs: Any) -> CompressedFeatures:
        """Apply FastV's ranking rule directly to an image-token tensor."""
        del layer_idx, kwargs

        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(
                f"Expected [B, N, D] or [N, D], got {tuple(features.shape)}"
            )
        batch, n_tokens, hidden_dim = features.shape
        if n_tokens == 0:
            raise ValueError("Cannot compress an empty token sequence")

        if attention is None:
            if self._track_source:
                identity = torch.eye(n_tokens, dtype=torch.float32)
                self._source_history = [identity, identity.clone()]
            return CompressedFeatures(
                tokens=features.squeeze(0) if squeeze else features
            )

        keep = self._resolve_keep(n_tokens)
        scores = normalize_attention_scores(
            attention.to(features.device), batch, n_tokens
        )
        indices = select_indices(scores, keep)
        tokens = features.gather(
            1, indices.unsqueeze(-1).expand(batch, keep, hidden_dim)
        )
        self._record_source(indices[0], n_tokens)
        return CompressedFeatures(
            tokens=tokens.squeeze(0) if squeeze else tokens
        )

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        if self._qwen is not None:
            return self._qwen.get_source_history()
        return list(self._source_history)

    @staticmethod
    def _detect_backbone(vlm: VLM) -> str:
        inner = getattr(getattr(vlm, "model", None), "model", None)
        if getattr(inner, "visual", None) is not None:
            return "qwen"
        if getattr(inner, "vision_tower", None) is not None:
            return "llava"
        model_name = type(getattr(vlm, "model", None)).__name__
        raise TypeError(
            "FastV supports the LLaVA-1.5, Qwen2.5-VL, and InternVL backbones; "
            f"{model_name} exposes neither `model.model.vision_tower` nor "
            "`model.model.visual`."
        )

    @override
    def install(self, vlm: VLM) -> None:
        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return
            raise RuntimeError(
                "FastVCompressor is already installed on another backbone"
            )

        from llmoe.compressor.llava_next import is_llava_next, layer_average_keep
        llava_next = is_llava_next(vlm)

        if self._detect_backbone(vlm) == "qwen":
            if self._retain_ratio is None:
                raise ValueError(
                    "FastV on Qwen2.5-VL requires `retain_ratio`: NaViT emits a "
                    "different visual-token count for each image, so a fixed "
                    "budget is not comparable across samples."
                )
            from llmoe.compressor.fastv.fastv_qwen import QwenFastVAdapter

            self._installed_vlm = vlm
            self._qwen = QwenFastVAdapter(
                retain_ratio=self._retain_ratio,
                k=self._k,
                track_source=self._track_source,
            )
            self._qwen.install(
                vlm,
                image_token_id=resolve_image_token_id(
                    vlm.model.config, vlm.processor
                ),
            )
            return

        inner = vlm.model.model
        language_model = getattr(inner, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if not layers:
            raise TypeError(
                "FastV needs a decoder stack at model.model.language_model.layers"
            )
        from llmoe.compressor.fastv.fastv_internvl import InternVLFastVAdapter

        internvl = InternVLFastVAdapter.supports(vlm)
        if self._profile_schedule and not internvl:
            # An absolute budget is a layer average, so the cut layer and the keep
            # count only mean what they say at the depth they were solved for.
            # InternVL is left alone: its adapter back-solves the keep itself from
            # the depth it is handed, over a tile span that is not 576 tokens.
            self._schedule = self._reference_schedule.rescaled_to(len(layers), label="FastV")
            self._k = self._schedule.boundaries[0]
            self._keep_tokens = int(self._schedule.keeps[0])
        if self._k >= len(layers):
            raise ValueError(
                f"k={self._k} is too large for a {len(layers)}-layer decoder"
            )
        if llava_next:
            self.validate_llava_next()
            # Validate reachability against the loaded decoder before hooks are
            # installed; the actual keep is resolved from every image's packed N.
            layer_average_keep(
                DEFAULT_VISUAL_TOKENS,
                retain_ratio=float(self._retain_ratio),
                full_layers=self._k,
                decoder_layers=len(layers),
                label="FastV/LLaVA-NeXT",
            )
            self._llava_next = True

        self._installed_vlm = vlm
        self._language_model = language_model
        if internvl:
            self._internvl = InternVLFastVAdapter(
                max_tokens=self._max_tokens,
                k=self._k,
                decoder_layers=len(layers),
            )
        self._image_token_id = resolve_image_token_id(
            vlm.model.config, vlm.processor
        )
        self._hook_handles.append(
            inner.register_forward_pre_hook(
                self._capture_image_positions, with_kwargs=True
            )
        )
        self._hook_handles.append(
            layers[self._k - 1].self_attn.register_forward_pre_hook(
                self._capture_attention, with_kwargs=True
            )
        )
        for layer_idx in range(self._k, len(layers)):
            self._hook_handles.append(
                layers[layer_idx].register_forward_pre_hook(
                    self._make_layer_hook(layer_idx), with_kwargs=True
                )
            )

        budget = (
            f"retain_ratio={self._retain_ratio}"
            if self._retain_ratio is not None
            else f"max_tokens={self._max_tokens}"
        )
        if self._internvl is not None and self._retain_ratio is None:
            achieved = self._internvl.install_note
            label = self._internvl.label
        elif self._llava_next:
            achieved = ", dynamic keep back-solved from the packed AnyRes span"
            label = "FastV/LLaVA-NeXT"
        elif self._retain_ratio is not None:
            achieved = ""
            label = "FastV"
        else:
            achieved = (
                f", keeps {self._keep_tokens} tokens for a layer average of "
                f"{self.achieved_average:.2f}"
            )
            label = "FastV"
        print(
            f"[{label}] install: {budget}, k={self._k} "
            f"(score @layer{self._k - 1}, prune @layer{self._k} of "
            f"{len(layers)}){achieved}, image_token_id={self._image_token_id}",
            flush=True,
        )

    def uninstall(self) -> None:
        if self._qwen is not None:
            self._qwen.uninstall()
            self._qwen = None
            self._installed_vlm = None
            return
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._installed_vlm = None
        self._language_model = None
        self._image_token_id = None
        self._internvl = None
        self._llava_next = False
        self._schedule = self._reference_schedule
        self._k = self._reference_schedule.boundaries[0]
        self._keep_tokens = int(self._reference_schedule.keeps[0])
        self._reset_forward_state()

    def _reset_forward_state(self) -> None:
        self._image_positions = None
        self._attention_scores = None
        self._layer_state = {}
        self._source_history = []

    def _capture_image_positions(self, _module: Any, args: tuple,
                                 kwargs: dict) -> None:
        """Record visual columns for each prefill and reject ragged batches."""
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.shape[1] <= 1
        ):
            return None

        self._reset_forward_state()
        self._forward_count += 1
        if self._llava_next:
            from llmoe.compressor.llava_next import decoder_visual_positions

            positions = decoder_visual_positions(
                input_ids,
                token_id=int(self._image_token_id),
                label="FastV",
            )
            if positions is None:
                return None
            rows = [positions]
        else:
            rows = [
                (row == self._image_token_id).nonzero(as_tuple=True)[0]
                for row in input_ids
            ]
        counts = {int(row.numel()) for row in rows}
        if counts == {0}:
            return None
        if len(counts) != 1:
            if not self._warned_no_image:
                self._warned_no_image = True
                print(
                    "[FastV] WARNING: rows carry different image-token counts "
                    f"{sorted(counts)}; pruning is skipped for this batch.",
                    flush=True,
                )
            return None

        self._image_positions = torch.stack(rows, dim=0)
        if self._forward_count == 1:
            n_visual = int(self._image_positions.shape[1])
            print(
                f"[FastV] first prefill: seq_len={input_ids.shape[1]}, "
                f"image_tokens={n_visual}, keep={self._resolve_keep(n_visual)}",
                flush=True,
            )
        return None

    def _capture_attention(self, module: Any, args: tuple,
                           kwargs: dict) -> None:
        if self._image_positions is None:
            return None
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.ndim != 3
            or hidden_states.shape[1] <= 1
        ):
            return None
        self._attention_scores = last_query_attention(
            module,
            hidden_states,
            kwargs.get("position_embeddings"),
            kwargs.get("attention_mask"),
            apply_rope=_APPLY_ROPE,
        )
        return None

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(_module: Any, args: tuple,
                 kwargs: dict) -> tuple[tuple, dict] | None:
            return self._layer_hook(args, kwargs, layer_idx)

        return hook

    def _layer_hook(self, args: tuple, kwargs: dict,
                    layer_idx: int) -> tuple[tuple, dict] | None:
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            return None

        prefill = hidden_states.shape[1] > 1
        if layer_idx == self._k:
            if prefill:
                if self._image_positions is None:
                    self._layer_state = {}
                    return None
                hidden_states = self._prune(hidden_states, kwargs)
            elif self._layer_state:
                advance_decode_step(
                    self._layer_state, hidden_states, label="FastV"
                )

        if not self._layer_state:
            return None
        apply_prune_state(kwargs, self._layer_state, prefill=prefill)
        kwargs.pop("hidden_states", None)
        return (hidden_states,), kwargs

    def _prune(self, hidden_states: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """Apply the shared FastV keep policy to LLaVA decoder geometry."""
        positions = self._image_positions
        if positions is None:
            return hidden_states
        positions = positions.to(hidden_states.device)
        batch, seq_len, hidden_dim = hidden_states.shape
        n_visual = int(positions.shape[1])
        if positions.shape[0] != batch or int(positions.max()) >= seq_len:
            raise RuntimeError(
                f"FastV image span {tuple(positions.shape)} does not fit decoder "
                f"input {tuple(hidden_states.shape)}; the prompt changed between hooks."
            )

        keep = self._resolve_keep(n_visual)
        if keep >= n_visual:
            self._layer_state = {}
            self._record_source(torch.arange(n_visual), n_visual)
            return hidden_states

        scores, self._attention_scores = self._attention_scores, None
        if scores is None:
            raise RuntimeError(
                f"FastV captured no attention at layer {self._k - 1}; the "
                "decoder must expose LLaMA-style q_proj and k_proj modules."
            )
        scores = normalize_attention_scores(
            scores.to(hidden_states.device), batch, seq_len
        )
        local, keep_index = select_visual_keep(
            scores, positions, keep, seq_len, label="FastV"
        )
        new_len = int(keep_index.shape[1])
        new_hidden = hidden_states.gather(
            1, keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim)
        )

        position_ids = gather_position_ids(
            kwargs.get("position_ids"), keep_index, seq_len
        )
        position_embeddings = self._language_model.rotary_emb(
            new_hidden, position_ids=position_ids
        )
        attention_mask, valid = pruned_attention_mask(
            kwargs.get("attention_mask"),
            keep_index,
            seq_len,
            label="FastV",
        )
        self._layer_state = {
            "batch": batch,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "attention_mask": attention_mask,
            "valid": valid,
            "cache_position": torch.arange(
                new_len, device=hidden_states.device
            ),
            "kv_len": new_len,
        }
        if self._forward_count == 1:
            print(
                f"[FastV] prune @layer{self._k}: seq {seq_len} -> {new_len} "
                f"(image {n_visual} -> {keep})",
                flush=True,
            )
        self._record_source(local[0], n_visual)
        return new_hidden
