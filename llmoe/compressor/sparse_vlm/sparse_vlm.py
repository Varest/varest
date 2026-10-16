"""SparseVLM: progressive text-guided visual-token sparsification.

SparseVLM selects visually relevant text raters, scores visual tokens from
rater-to-visual decoder attention, and progressively prunes at several layers.
Deleted tokens may be recycled through density-peak clustering. This module
owns that entire baseline policy and the LLaVA integration; ``sparse_vlm_qwen.py`` contains
only Qwen's M-RoPE and cache geometry.

Paper: Zhang et al., "SparseVLM: Visual Token Sparsification for Efficient
Vision Language Model Inference" (arXiv:2410.04417).
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
)
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.utils.depth import REFERENCE_DEPTH, LayerSchedule
from llmoe.utils.qwen import resolve_image_token_id as shared_resolve_image_token_id

try:
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb as _APPLY_ROPE,
    )
except Exception:  # pragma: no cover - degrade to no-RoPE if the path moves
    _APPLY_ROPE = None


SPARSEVLM_PROFILES: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {
    224: ((299, 245, 147), (2, 6, 15)),
    192: ((300, 200, 110), (2, 6, 15)),
    128: ((303, 110, 36), (2, 6, 15)),
    64: ((66, 30, 17), (2, 6, 15)),
    32: ((15, 14, 8), (1, 6, 15)),
}

LLAVA_VISUAL_TOKENS: int = 576

# Qwen2.5-VL takes `retain_ratio` like every other compressor here: NaViT emits a
# different token count per image, so the published budgets only transfer as the
# fraction of the visual span they were on LLaVA.
SPARSEVLM_QWEN_RATIOS: dict[float, int] = {
    round(budget / LLAVA_VISUAL_TOKENS, 4): budget for budget in SPARSEVLM_PROFILES
}
RopeFunction = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def resolve_image_token_id(config: Any, processor: Any) -> int:
    """Resolve the image placeholder id without hard-coding a tokenizer value."""
    return shared_resolve_image_token_id(config, processor, label="SparseVLM")


def cluster_and_merge(
    tokens: torch.Tensor,
    cluster_num: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Density-peak cluster ``[1, N, C]`` tokens and average each cluster."""
    _, length, channels = tokens.shape
    cluster_num = max(1, min(int(cluster_num), length))

    distance = (
        tokens[:, :, None, :] - tokens[:, None, :, :]
    ).norm(dim=-1, p=2) / (channels ** 0.5)
    nearest, _ = distance.topk(cluster_num, dim=-1, largest=False)
    density = (-(nearest ** 2).mean(dim=-1)).exp()
    density = density + torch.rand(
        density.shape,
        generator=generator,
        device=density.device,
        dtype=density.dtype,
    ) * 1e-6

    denser = (density[:, None, :] > density[:, :, None]).to(tokens.dtype)
    farthest = distance.flatten(1).max(dim=-1)[0][:, None, None]
    separation = (
        distance * denser + farthest * (1 - denser)
    ).min(dim=-1)[0]
    centers = (separation * density).topk(cluster_num, dim=-1).indices[0]

    assignment = distance[0, centers, :].argmin(dim=0)
    assignment[centers] = torch.arange(cluster_num, device=tokens.device)
    weights = torch.zeros(
        cluster_num,
        length,
        device=tokens.device,
        dtype=torch.float32,
    )
    weights[assignment, torch.arange(length, device=tokens.device)] = 1.0
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    merged = (weights.to(tokens.dtype) @ tokens[0]).unsqueeze(0)
    return merged, weights, centers


def sparsify(
    visual: torch.Tensor,
    scores: torch.Tensor,
    keep_k: int,
    *,
    recycling: bool,
    recycle_fraction: float,
    cluster_divisor: int,
    generator: torch.Generator,
    track_source: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run one keep-and-recycle stage over a ``[1, V, D]`` visual block.

    Returns the new visual block, the incoming token donating each output's
    position, and an optional source matrix.
    """
    visual_count = int(visual.shape[1])
    keep_k = max(0, min(int(keep_k), visual_count))
    keep_index = (
        scores.topk(keep_k, dim=-1).indices.sort().values
        if keep_k
        else scores.new_zeros(0, dtype=torch.long)
    )
    keep_mask = torch.zeros(
        visual_count, dtype=torch.bool, device=visual.device
    )
    keep_mask[keep_index] = True
    delete_index = (~keep_mask).nonzero(as_tuple=True)[0]

    kept = visual[:, keep_index, :]
    blocks = [kept]
    donors = [keep_index]
    merge_weights: torch.Tensor | None = None
    recycled_index: torch.Tensor | None = None
    if recycling and delete_index.numel() > 0:
        deleted_count = int(delete_index.numel())
        pool_size = min(
            deleted_count,
            int(deleted_count * recycle_fraction) + 1,
        )
        pool_local = scores[delete_index].topk(pool_size).indices
        recycled_index = delete_index[pool_local]
        merged, merge_weights, centers = cluster_and_merge(
            visual[:, recycled_index, :],
            pool_size // cluster_divisor + 1,
            generator,
        )
        blocks.append(merged.to(visual.dtype))
        donors.append(recycled_index[centers])

    new_visual = torch.cat(blocks, dim=1) if len(blocks) > 1 else kept
    donor = torch.cat(donors) if len(donors) > 1 else keep_index
    if not track_source:
        return new_visual, donor, None

    source = torch.zeros(
        new_visual.shape[1], visual_count, dtype=torch.float32
    )
    if keep_k:
        source[torch.arange(keep_k), keep_index.cpu()] = 1.0
    if merge_weights is not None and recycled_index is not None:
        source[keep_k:, recycled_index.cpu()] = merge_weights.detach().cpu()
    return new_visual, donor, source


class SparseVLMPolicy:
    """Backbone-neutral SparseVLM sparsification settings and deterministic RNG."""

    def __init__(self, *, recycling: bool, recycle_fraction: float,
                 cluster_divisor: int, seed: int,
                 track_source: bool) -> None:
        self.recycling = bool(recycling)
        self.recycle_fraction = float(recycle_fraction)
        self.cluster_divisor = int(cluster_divisor)
        self.seed = int(seed)
        self.track_source = bool(track_source)
        self._generators: dict[torch.device, torch.Generator] = {}

    def reseed(self) -> None:
        for generator in self._generators.values():
            generator.manual_seed(self.seed)

    def generator(self, device: torch.device) -> torch.Generator:
        generator = self._generators.get(device)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generators[device] = generator
        return generator

    def sparsify(
        self,
        visual: torch.Tensor,
        scores: torch.Tensor,
        keep_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return sparsify(
            visual,
            scores,
            keep_k,
            recycling=self.recycling,
            recycle_fraction=self.recycle_fraction,
            cluster_divisor=self.cluster_divisor,
            generator=self.generator(visual.device),
            track_source=self.track_source,
        )


def select_raters(inputs_embeds: torch.Tensor, image_start: int,
                  visual_count: int) -> torch.Tensor:
    """Select visually relevant post-image text tokens (paper Eq. 5-6)."""
    text_start = image_start + visual_count
    if text_start >= inputs_embeds.shape[1]:
        raise RuntimeError(
            "SparseVLM needs text tokens after the image span to rate it"
        )
    visual = inputs_embeds[:, image_start:text_start, :].float()
    text = inputs_embeds[:, text_start:, :].float()
    relation = (
        torch.matmul(visual, text.transpose(1, 2))
        .softmax(dim=2)
        .mean(dim=1)[0]
    )
    selected = (relation > relation.mean()).nonzero(as_tuple=True)[0]
    if selected.numel() == 0:
        selected = torch.arange(text.shape[1], device=inputs_embeds.device)
    return selected


def rater_attention(
    attn: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    raters: torch.Tensor,
    *,
    apply_rope: RopeFunction | None,
    rope_args: tuple[Any, ...] = (),
) -> torch.Tensor:
    """Recompute only SparseVLM's rater attention rows."""
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
        query[:, :, raters, :].float() @ key.float().transpose(-1, -2)
    ) * scaling
    keys = torch.arange(seq_len, device=hidden_states.device)
    causal = (keys[None, :] > raters[:, None])[None, None, :, :]
    logits = logits.masked_fill(causal, torch.finfo(logits.dtype).min)
    return logits.softmax(dim=-1).mean(dim=1)[0]


def causal_mask(attention_mask: Any, new_len: int,
                device: torch.device) -> torch.Tensor | None:
    """Rebuild a causal prefill mask after recycled tokens change the sequence."""
    if not isinstance(attention_mask, torch.Tensor):
        return None
    allowed = torch.ones(
        new_len, new_len, dtype=torch.bool, device=device
    ).tril().view(1, 1, new_len, new_len)
    if attention_mask.dtype == torch.bool:
        return allowed
    minimum = torch.finfo(attention_mask.dtype).min
    return torch.where(
        allowed,
        torch.zeros((), dtype=attention_mask.dtype, device=device),
        torch.full((), minimum, dtype=attention_mask.dtype, device=device),
    )


def append_source_history(
    history: list[torch.Tensor],
    composed: torch.Tensor | None,
    source: torch.Tensor | None,
) -> torch.Tensor | None:
    """Compose one stage's source matrix and append it to ``history``."""
    if source is None:
        return composed
    composed = source if composed is None else source @ composed
    if not history:
        history.append(torch.eye(source.shape[1], dtype=torch.float32))
    history.append(composed)
    return composed


def normalize_rater_scores(attention: torch.Tensor,
                           n_tokens: int) -> torch.Tensor:
    """Normalize scores or raw rater attention to a ``[N]`` score vector."""
    scores = attention.float()
    if scores.ndim == 4:
        scores = scores.mean(dim=(1, 2))
    elif scores.ndim == 3:
        scores = scores.mean(dim=1)
    if scores.ndim == 2:
        if scores.shape[0] != 1:
            raise ValueError(
                f"Expected single-image scores, got {tuple(attention.shape)}"
            )
        scores = scores[0]
    if scores.ndim != 1 or scores.shape[0] != n_tokens:
        raise ValueError(
            f"attention scores {tuple(attention.shape)} do not match the "
            f"{n_tokens} tokens being compressed"
        )
    return scores


class SparseVLMCompressor(TokenCompressor):
    """Prune visual tokens progressively using visually relevant text raters."""

    def __init__(
        self,
        max_tokens: int = 192,
        *,
        retain_ratio: float | None = None,
        pruning_layers: tuple[int, ...] | None = None,
        schedule: tuple[int, ...] | None = None,
        schedule_ratios: tuple[float, ...] | None = None,
        recycling: bool = True,
        recycle_fraction: float = 0.3,
        cluster_divisor: int = 10,
        seed: int = 42,
        track_source: bool = False,
    ) -> None:
        configured_max_tokens = int(max_tokens)
        configured_pruning_layers = pruning_layers is not None
        configured_schedule = schedule is not None
        configured_schedule_ratios = schedule_ratios is not None
        published_ratio_budget: int | None = None
        if retain_ratio is not None:
            retain_ratio = float(retain_ratio)
            if not 0.0 < retain_ratio <= 1.0:
                raise ValueError(
                    f"retain_ratio must be in (0, 1], got {retain_ratio}"
                )
            key = round(retain_ratio, 4)
            published_ratio_budget = SPARSEVLM_QWEN_RATIOS.get(key)
            # Arbitrary ratios are valid only on native NeXT, whose packed span
            # is known at runtime. Keep the canonical 192 profile as inert
            # constructor metadata until install() resolves that backbone.
            max_tokens = (
                published_ratio_budget
                if published_ratio_budget is not None
                else 192
            )
        self._retain_ratio: float | None = (
            None if retain_ratio is None else float(retain_ratio)
        )
        super().__init__(int(max_tokens))
        # Only a fully published schedule is re-solved for another decoder depth;
        # anything the caller pinned is theirs and is taken literally.
        self._profile_schedule: bool = schedule is None and pruning_layers is None
        if schedule is None or pruning_layers is None:
            if int(max_tokens) not in SPARSEVLM_PROFILES:
                supported = ", ".join(
                    str(budget) for budget in sorted(SPARSEVLM_PROFILES)
                )
                raise ValueError(
                    f"Unsupported SparseVLM budget: {max_tokens}. Only "
                    f"{{{supported}}} have a published schedule."
                )
            profile_schedule, profile_layers = SPARSEVLM_PROFILES[int(max_tokens)]
            schedule = profile_schedule if schedule is None else schedule
            pruning_layers = (
                profile_layers if pruning_layers is None else pruning_layers
            )

        self._schedule = tuple(int(keep) for keep in schedule)
        self._pruning_layers = tuple(int(layer) for layer in pruning_layers)
        if len(self._schedule) != len(self._pruning_layers):
            raise ValueError(
                "schedule and pruning_layers must have the same length, got "
                f"{len(self._schedule)} != {len(self._pruning_layers)}"
            )
        if not self._schedule:
            raise ValueError("SparseVLM needs at least one pruning stage")
        if any(keep < 1 for keep in self._schedule):
            raise ValueError("every schedule entry must be >= 1")
        if any(layer < 0 for layer in self._pruning_layers):
            raise ValueError("pruning_layers must all be >= 0")
        if list(self._pruning_layers) != sorted(set(self._pruning_layers)):
            raise ValueError("pruning_layers must be strictly increasing")
        if not 0.0 < float(recycle_fraction) <= 1.0:
            raise ValueError(
                f"recycle_fraction must be in (0, 1], got {recycle_fraction}"
            )
        if int(cluster_divisor) < 1:
            raise ValueError("cluster_divisor must be >= 1")
        if (
            schedule_ratios is not None
            and len(schedule_ratios) != len(self._schedule)
        ):
            raise ValueError(
                "schedule_ratios must have one entry per pruning layer, got "
                f"{len(schedule_ratios)} != {len(self._schedule)}"
            )

        self._schedule_ratios = (
            tuple(float(ratio) for ratio in schedule_ratios)
            if schedule_ratios is not None
            else tuple(keep / LLAVA_VISUAL_TOKENS for keep in self._schedule)
        )
        self._policy = SparseVLMPolicy(
            recycling=recycling,
            recycle_fraction=recycle_fraction,
            cluster_divisor=cluster_divisor,
            seed=seed,
            track_source=track_source,
        )
        self._recycling = self._policy.recycling
        self._track_source = self._policy.track_source
        self._configured_max_tokens = configured_max_tokens
        self._configured_pruning_layers = configured_pruning_layers
        self._configured_schedule = configured_schedule
        self._configured_schedule_ratios = configured_schedule_ratios
        self._published_ratio_budget = published_ratio_budget
        self._configured_recycling = bool(recycling)
        self._configured_recycle_fraction = float(recycle_fraction)
        self._configured_cluster_divisor = int(cluster_divisor)
        self._configured_seed = int(seed)
        # The published schedule, stated against the 32-layer reference decoder.
        # `install` re-solves it for the backbone's real depth and `uninstall`
        # restores it, so one compressor serves decoders of any depth.
        self._reference_schedule = LayerSchedule(
            self._pruning_layers, self._schedule, REFERENCE_DEPTH, LLAVA_VISUAL_TOKENS
        )
        self._trigger_layers: dict[int, int] = {}
        self._adopt_schedule(self._reference_schedule)

        self._span: tuple[int, int] | None = None
        self._rater_offsets: torch.Tensor | None = None
        self._visual_count = 0
        self._scores: dict[int, torch.Tensor] = {}
        self._layer_state: dict[str, Any] = {}
        self._composed_source: torch.Tensor | None = None
        self._stage_states: dict[int, dict[str, Any]] = {}
        self._next_counts: tuple[int, ...] = ()
        self._forward_count = 0

        self._installed_vlm: VLM | None = None
        self._language_model: Any | None = None
        self._hook_handles: list[RemovableHandle] = []
        self._image_token_id: int | None = None
        self._qwen: QwenPruningAdapter | None = None
        self._internvl: Any | None = None
        self._llava_next = False

    def _adopt_schedule(self, schedule: LayerSchedule) -> None:
        """Run under `schedule`; each stage's cut fires on the layer after its boundary."""
        self._schedule = tuple(int(keep) for keep in schedule.keeps)
        self._pruning_layers = schedule.boundaries
        self._resolved_schedule = schedule
        self._trigger_layers = {
            layer + 1: stage for stage, layer in enumerate(self._pruning_layers)
        }

    @property
    def achieved_average(self) -> float:
        """Visual tokens averaged over the decoder under the resolved schedule."""
        return self._resolved_schedule.layer_average

    @property
    @override
    def budget(self) -> int | float:
        return self._max_tokens if self._retain_ratio is None else self._retain_ratio

    def validate_llava_next(self) -> None:
        """Expose only the canonical v1.5 profile scale as retain_ratio."""

        from llmoe.compressor.llava_next import validate_retain_ratio
        from llmoe.compressor.sparse_vlm.sparse_vlm_llava_next import (
            layer_average_schedule,
        )

        ratio = validate_retain_ratio(self._retain_ratio, label="SparseVLM")
        if self._configured_max_tokens != 192:
            raise ValueError(
                "SparseVLM on LLaVA-NeXT only accepts retain_ratio; remove "
                "max_tokens (its constructor default is 192 and is ignored on NeXT)."
            )
        if (
            self._configured_schedule
            or self._configured_schedule_ratios
            or not self._configured_recycling
            or self._configured_recycle_fraction != 0.3
            or self._configured_cluster_divisor != 10
            or self._configured_seed != 42
        ):
            raise ValueError(
                "SparseVLM on LLaVA-NeXT fixes the released v1.5 "
                "300:200:110 profile shape, layers 2/6/15, top-30% recycling, "
                "cluster divisor, and seed; only retain_ratio and "
                "pruning_layers are configurable."
            )
        if self._configured_pruning_layers:
            print(f"[SparseVLM] ADAPTED VARIANT: pruning_layers="
                  f"{self._pruning_layers} (canonical is (2, 6, 15)) so "
                  f"retain_ratio={float(self._retain_ratio):g} is reachable",
                  flush=True)
        # Fail early against the canonical 32-layer/576-token geometry. The
        # image-specific scheduler repeats this check for every AnyRes span.
        layer_average_schedule(
            LLAVA_VISUAL_TOKENS,
            ratio,
            32,
            self._pruning_layers if self._configured_pruning_layers else None,
        )

    def _keep_for_stage(self, stage: int, visual_count: int) -> int:
        if visual_count <= 0:
            return 0
        if self._llava_next and self._retain_ratio == 1.0:
            return visual_count
        return min(int(self._schedule[stage]), visual_count - 1)

    def _stage_for_layer(self, layer_idx: int | None) -> int:
        if layer_idx is None:
            return 0
        if int(layer_idx) not in self._pruning_layers:
            raise ValueError(
                f"layer_idx={layer_idx} is not a pruning layer; expected one of "
                f"{list(self._pruning_layers)}"
            )
        return self._pruning_layers.index(int(layer_idx))

    @staticmethod
    def _rater_scores(attention: torch.Tensor,
                      n_tokens: int) -> torch.Tensor:
        return normalize_rater_scores(attention, n_tokens)

    @override
    def compress(
        self,
        features: torch.Tensor,
        *,
        attention: torch.Tensor | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> CompressedFeatures:
        del kwargs
        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(
                f"Expected [B, N, D] or [N, D], got {tuple(features.shape)}"
            )
        batch, n_tokens, _ = features.shape
        if n_tokens == 0:
            raise ValueError("Cannot compress an empty token sequence")
        if attention is None:
            if self._track_source:
                identity = torch.eye(n_tokens, dtype=torch.float32)
                self._source_history = [identity, identity.clone()]
            return CompressedFeatures(
                tokens=features.squeeze(0) if squeeze else features
            )
        if batch != 1:
            raise ValueError("SparseVLM supports batch_size=1 only")

        scores = normalize_rater_scores(
            attention.to(features.device), n_tokens
        )
        keep_k = self._keep_for_stage(
            self._stage_for_layer(layer_idx), n_tokens
        )
        tokens, _, source = self._policy.sparsify(features, scores, keep_k)
        if source is not None:
            self._source_history = [
                torch.eye(n_tokens, dtype=torch.float32),
                source,
            ]
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
            "SparseVLM supports LLaVA-1.5, Qwen2.5-VL, and InternVL; "
            f"{model_name} exposes neither supported backbone structure."
        )

    @override
    def install(self, vlm: VLM) -> None:
        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return
            raise RuntimeError(
                "SparseVLMCompressor is already installed on another backbone"
            )

        from llmoe.compressor.llava_next import is_llava_next

        llava_next = is_llava_next(vlm)
        if llava_next:
            self.validate_llava_next()
        backbone = self._detect_backbone(vlm)
        if backbone == "qwen":
            if self._retain_ratio is None:
                supported = ", ".join(
                    f"{ratio:g}" for ratio in sorted(SPARSEVLM_QWEN_RATIOS)
                )
                raise ValueError(
                    f"SparseVLM on Qwen2.5-VL requires `retain_ratio` ({{{supported}}}), "
                    f"not `max_tokens`: NaViT emits a different visual-token count for "
                    f"each image, so a fixed budget is not comparable across samples. "
                    f"The published schedule is applied as a fraction of the prompt's "
                    f"own visual span."
                )
            if self._published_ratio_budget is None:
                supported = ", ".join(
                    f"{ratio:g}" for ratio in sorted(SPARSEVLM_QWEN_RATIOS)
                )
                raise ValueError(
                    f"SparseVLM on Qwen2.5-VL only has published schedules for "
                    f"retain_ratio in {{{supported}}}; arbitrary ratios are "
                    f"supported only on native LLaVA-NeXT."
                )
            from llmoe.compressor.sparse_vlm.sparse_vlm_qwen import QwenSparseVLMAdapter

            self._installed_vlm = vlm
            self._qwen = QwenSparseVLMAdapter(
                schedule_ratios=self._schedule_ratios,
                pruning_layers=self._pruning_layers,
                policy=self._policy,
            )
            self._qwen.install(
                vlm,
                image_token_id=resolve_image_token_id(
                    vlm.model.config, vlm.processor
                ),
            )
            return

        if (
            self._retain_ratio is not None
            and self._published_ratio_budget is None
            and not llava_next
        ):
            raise ValueError(
                "Arbitrary SparseVLM retain_ratio values are supported only on "
                "native LLaVA-NeXT; LLaVA-1.5 and InternVL use published profiles."
            )
        inner = vlm.model.model
        language_model = getattr(inner, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if not layers:
            raise TypeError(
                "SparseVLM needs model.model.language_model.layers"
            )
        if self._profile_schedule:
            # Published boundaries are fractions of the decoder, so they move with
            # its depth and the keeps are re-solved to hold the budget label. A
            # caller-supplied schedule is taken literally, as everywhere else.
            self._adopt_schedule(
                self._reference_schedule.rescaled_to(
                    len(layers), label=f"SparseVLM's budget-{self._max_tokens} schedule"
                )
            )
        if self._pruning_layers[-1] + 1 >= len(layers):
            raise ValueError(
                f"pruning layer {self._pruning_layers[-1]} leaves no layer "
                f"to prune into in a {len(layers)}-layer decoder"
            )

        from llmoe.compressor.sparse_vlm.sparse_vlm_internvl import (
            InternVLSparseVLMAdapter,
        )

        self._installed_vlm = vlm
        self._language_model = language_model
        self._llava_next = llava_next
        if InternVLSparseVLMAdapter.supports(vlm):
            self._internvl = InternVLSparseVLMAdapter()
        self._image_token_id = resolve_image_token_id(
            vlm.model.config, vlm.processor
        )
        self._hook_handles.append(
            inner.register_forward_pre_hook(
                self._capture_span, with_kwargs=True
            )
        )
        self._hook_handles.append(
            language_model.register_forward_pre_hook(
                self._select_raters, with_kwargs=True
            )
        )
        for layer in self._pruning_layers:
            self._hook_handles.append(
                layers[layer].self_attn.register_forward_pre_hook(
                    self._make_score_hook(layer), with_kwargs=True
                )
            )
        for layer_idx in range(self._pruning_layers[0] + 1, len(layers)):
            self._hook_handles.append(
                layers[layer_idx].register_forward_pre_hook(
                    self._make_layer_hook(layer_idx), with_kwargs=True
                )
            )

        label = (
            "SparseVLM/LLaVA-NeXT"
            if self._llava_next
            else self._internvl.label if self._internvl is not None else "SparseVLM"
        )
        span = self._internvl.install_note if self._internvl is not None else ""
        if self._llava_next:
            schedule_note = (
                f"retain_ratio={self._retain_ratio:g}, "
                f"schedule=image-specific @layers{list(self._pruning_layers)}"
            )
        else:
            schedule_note = (
                f"budget/profile={self._max_tokens} "
                f"(schedule averages {self.achieved_average:.2f} over "
                f"{len(layers)} layers), schedule={list(self._schedule)} "
                f"@layers{list(self._pruning_layers)}"
            )
        print(
            f"[{label}] install: {schedule_note}, "
            f"recycling={self._recycling}{span}, image_token_id={self._image_token_id}",
            flush=True,
        )

    def uninstall(self) -> None:
        if self._qwen is not None:
            self._qwen.uninstall()
            self._qwen = None
            self._installed_vlm = None
            return
        source_history = list(self._source_history)
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._installed_vlm = None
        self._language_model = None
        self._image_token_id = None
        self._adopt_schedule(self._reference_schedule)
        self._internvl = None
        self._llava_next = False
        self._reset_forward_state()
        self._source_history = source_history

    def _reset_forward_state(self) -> None:
        self._span = None
        self._rater_offsets = None
        self._visual_count = 0
        self._scores = {}
        self._layer_state = {}
        self._stage_states = {}
        self._next_counts = ()
        self._composed_source = None
        self._source_history = []

    def _capture_span(self, _module: Any, args: tuple,
                      kwargs: dict) -> None:
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
        self._policy.reseed()
        self._forward_count += 1
        if input_ids.shape[0] != 1:
            raise RuntimeError(
                f"SparseVLM supports batch_size=1, got {input_ids.shape[0]}"
            )
        positions = (
            input_ids[0] == self._image_token_id
        ).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            return None
        start, end = int(positions[0]), int(positions[-1]) + 1
        if positions.numel() != end - start:
            raise RuntimeError(
                "SparseVLM expects one contiguous visual span"
            )
        self._span = (start, end - start)
        self._visual_count = end - start
        if self._llava_next:
            from llmoe.compressor.sparse_vlm.sparse_vlm_llava_next import (
                layer_average_schedule,
            )

            schedule, counts = layer_average_schedule(
                self._visual_count,
                float(self._retain_ratio),
                len(self._language_model.layers),
                self._pruning_layers if self._configured_pruning_layers else None,
            )
            self._schedule = schedule
            self._next_counts = counts
        if self._forward_count == 1:
            print(
                f"[SparseVLM] first prefill: seq_len={input_ids.shape[1]}, "
                f"image_tokens={self._visual_count} at [{start}, {end})",
                flush=True,
            )
        return None

    def _select_raters(self, _module: Any, args: tuple,
                       kwargs: dict) -> None:
        if self._span is None:
            return None
        hidden_states = kwargs.get("inputs_embeds")
        if hidden_states is None and args:
            hidden_states = args[0]
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.ndim != 3
            or hidden_states.shape[1] <= 1
        ):
            return None
        start, count = self._span
        self._rater_offsets = select_raters(hidden_states, start, count)
        if self._forward_count == 1:
            text_count = hidden_states.shape[1] - start - count
            print(
                f"[SparseVLM] raters: {self._rater_offsets.numel()} "
                f"of {text_count} text tokens",
                flush=True,
            )
        return None

    def _make_score_hook(self, layer: int) -> Any:
        def hook(module: Any, args: tuple, kwargs: dict) -> None:
            return self._score_hook(module, args, kwargs, layer)

        return hook

    def _score_hook(self, module: Any, args: tuple,
                    kwargs: dict, layer: int) -> None:
        if self._span is None or self._rater_offsets is None:
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
        start = self._span[0]
        raters = (
            self._rater_offsets.to(hidden_states.device)
            + start
            + self._visual_count
        )
        if int(raters.max()) >= hidden_states.shape[1]:
            raise RuntimeError(
                f"SparseVLM rater index {int(raters.max())} exceeds "
                f"layer-{layer} length {hidden_states.shape[1]}"
            )
        attention = rater_attention(
            module,
            hidden_states,
            kwargs.get("position_embeddings"),
            raters,
            apply_rope=_APPLY_ROPE,
        )
        self._scores[layer] = attention[
            :, start:start + self._visual_count
        ].mean(dim=0)
        return None

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(_module: Any, args: tuple,
                 kwargs: dict) -> tuple[tuple, dict] | None:
            return self._layer_hook(args, kwargs, layer_idx)

        return hook

    def _layer_hook(self, args: tuple, kwargs: dict,
                    layer_idx: int) -> tuple[tuple, dict] | None:
        if self._llava_next:
            return self._llava_next_layer_hook(args, kwargs, layer_idx)

        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            return None

        stage = self._trigger_layers.get(layer_idx)
        if (
            stage is not None
            and hidden_states.shape[1] > 1
            and self._span is not None
        ):
            hidden_states = self._prune_stage(hidden_states, kwargs, stage)

        if not self._layer_state:
            return None
        if hidden_states.shape[1] > 1:
            kwargs["position_ids"] = self._layer_state["position_ids"]
            kwargs["position_embeddings"] = self._layer_state[
                "position_embeddings"
            ]
            kwargs["attention_mask"] = self._layer_state["attention_mask"]
            if "cache_position" in kwargs:
                kwargs["cache_position"] = self._layer_state[
                    "cache_position"
                ]
        else:
            self._apply_decode_state(hidden_states, kwargs, layer_idx)
        kwargs.pop("hidden_states", None)
        return (hidden_states,), kwargs

    def _llava_next_layer_hook(
        self,
        args: tuple,
        kwargs: dict,
        layer_idx: int,
    ) -> tuple[tuple, dict] | None:
        """Apply the matching cache geometry for each progressive NeXT cut."""

        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            return None
        prefill = hidden_states.shape[1] > 1
        stage = self._trigger_layers.get(layer_idx)
        if stage is not None:
            if prefill and self._span is not None:
                hidden_states = self._prune_stage(hidden_states, kwargs, stage)
            elif not prefill and stage in self._stage_states:
                advance_decode_step(
                    self._stage_states[stage],
                    hidden_states,
                    label="SparseVLM/LLaVA-NeXT",
                )

        state = self._active_llava_next_state(layer_idx)
        if state is None:
            return None
        apply_prune_state(kwargs, state, prefill=prefill)
        if not prefill:
            position_ids = state["cache_position"].view(1, 1)
            kwargs["position_ids"] = position_ids
            kwargs["position_embeddings"] = self._language_model.rotary_emb(
                hidden_states, position_ids=position_ids
            )
        kwargs.pop("hidden_states", None)
        return (hidden_states,), kwargs

    def _active_llava_next_state(
        self, layer_idx: int
    ) -> dict[str, Any] | None:
        for stage in range(len(self._pruning_layers) - 1, -1, -1):
            if (
                self._pruning_layers[stage] < layer_idx
                and stage in self._stage_states
            ):
                return self._stage_states[stage]
        return None

    def _prune_stage(self, hidden_states: torch.Tensor,
                     kwargs: dict, stage: int) -> torch.Tensor:
        layer = self._pruning_layers[stage]
        scores = self._scores.pop(layer, None)
        if scores is None:
            raise RuntimeError(
                f"SparseVLM captured no attention at pruning layer {layer}"
            )
        start, _ = self._span
        end = start + self._visual_count
        if scores.shape[0] != self._visual_count or end > hidden_states.shape[1]:
            raise RuntimeError(
                f"SparseVLM visual span [{start}, {end}) is inconsistent "
                f"with layer-{layer} input {tuple(hidden_states.shape)}"
            )

        keep_k = self._keep_for_stage(stage, self._visual_count)
        visual, _, source = self._policy.sparsify(
            hidden_states[:, start:end, :], scores, keep_k
        )
        hidden_states = torch.cat(
            [
                hidden_states[:, :start, :],
                visual,
                hidden_states[:, end:, :],
            ],
            dim=1,
        )
        self._visual_count = int(visual.shape[1])
        new_len = int(hidden_states.shape[1])
        position_ids = torch.arange(
            new_len, device=hidden_states.device
        ).unsqueeze(0)
        state = {
            "position_ids": position_ids,
            "position_embeddings": self._language_model.rotary_emb(
                hidden_states, position_ids=position_ids
            ),
            "attention_mask": causal_mask(
                kwargs.get("attention_mask"),
                new_len,
                hidden_states.device,
            ),
            "cache_position": torch.arange(
                new_len, device=hidden_states.device
            ),
        }
        if self._llava_next:
            state.update(
                {
                    "batch": int(hidden_states.shape[0]),
                    "valid": torch.ones(
                        hidden_states.shape[0],
                        new_len,
                        dtype=torch.bool,
                        device=hidden_states.device,
                    ),
                    "kv_len": new_len,
                }
            )
            self._stage_states[stage] = state
            if self._next_counts and self._visual_count != self._next_counts[stage]:
                raise RuntimeError(
                    f"SparseVLM stage {stage} produced {self._visual_count} visual "
                    f"rows; its NeXT schedule requires {self._next_counts[stage]}."
                )
        self._layer_state = state
        if self._track_source:
            self._composed_source = append_source_history(
                self._source_history,
                self._composed_source,
                source,
            )
        if self._forward_count == 1:
            print(
                f"[SparseVLM] stage {stage} @layer{layer}: "
                f"keep_k={keep_k}, visual -> {self._visual_count}, "
                f"seq -> {new_len}",
                flush=True,
            )
        return hidden_states

    def _apply_decode_state(self, hidden_states: torch.Tensor,
                            kwargs: dict, layer_idx: int) -> None:
        past_len = self._cache_length(
            kwargs.get("past_key_values"), layer_idx
        )
        if past_len is None:
            return
        position_ids = torch.full(
            (1, 1),
            past_len,
            dtype=torch.long,
            device=hidden_states.device,
        )
        kwargs["position_ids"] = position_ids
        kwargs["position_embeddings"] = self._language_model.rotary_emb(
            hidden_states, position_ids=position_ids
        )
        mask = kwargs.get("attention_mask")
        if isinstance(mask, torch.Tensor) and mask.shape[-1] != past_len + 1:
            shape = (1, 1, 1, past_len + 1)
            kwargs["attention_mask"] = (
                torch.ones(shape, dtype=torch.bool, device=hidden_states.device)
                if mask.dtype == torch.bool
                else torch.zeros(
                    shape, dtype=mask.dtype, device=hidden_states.device
                )
            )
        if "cache_position" in kwargs:
            kwargs["cache_position"] = torch.tensor(
                [past_len], device=hidden_states.device, dtype=torch.long
            )

    @staticmethod
    def _cache_length(past_key_values: Any,
                      layer_idx: int) -> int | None:
        if past_key_values is None:
            return None
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length):
            try:
                return int(get_seq_length(layer_idx))
            except (TypeError, IndexError):
                pass
        layers = getattr(past_key_values, "layers", None)
        if layers is not None and layer_idx < len(layers):
            keys = getattr(layers[layer_idx], "keys", None)
            if isinstance(keys, torch.Tensor):
                return int(keys.shape[2])
        return None
