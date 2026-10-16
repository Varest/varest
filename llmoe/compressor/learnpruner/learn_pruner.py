"""LearnPruner for LLaVA-1.5 7B/13B and LLaVA-NeXT (arXiv:2604.23950).

The LPM scores unprojected CLIP patch features. Inference keeps the highest
keep-probability patches plus 10% max-min cosine diversity patches, then uses
query-to-image attention at decoder layer 12 of the paper's 32-layer reference
decoder for a second physical cut. On another decoder depth the repository's
shared layer-schedule solver scales that boundary and re-solves the late keep.
LLaVA-NeXT scores its complete native AnyRes sequence only after all tiles have
been projected, spatially joined, unpadded, and furnished with newline rows.
Training keeps every position and applies the paper's diagonal-preserving STE attention
mask; only the LPM is optimized. ``max_tokens`` is the layer-averaged visual
budget, matching Table 1 of the paper, not the stage-one count. The standard
repository budgets are 32, 64, 128, and 224. ``retain_ratio`` expresses the
same layer average as a fraction of the native visual span and is required for
the variable-length NeXT backbone.
"""

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, override

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.modeling_outputs import BaseModelOutputWithPooling

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    advance_decode_step, apply_prune_state, decoder_attention_rows,
    gather_position_ids, keep_columns, pruned_attention_mask,
)
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.utils.depth import LayerSchedule, REFERENCE_DEPTH, scale_layer
from llmoe.weights import backbone_key


PRUNING_LAYER = 12  # number of full decoder layers before the physical cut
DIVERSITY_RATIO = 0.10
FEATURE_DIM = 1024
HIDDEN_DIM = 512  # 1024*512 + 512*2 + biases = 0.526M parameters
DEFAULT_MAX_TOKENS = 64

# Table 1 reports 32/64/128; Section 4.1 fixes k=12 and R1:R2=3, which uniquely
# determines the integral stage counts up to nearest-token rounding. The repo's
# standard 224 row is the same stated rule extrapolated to its fourth budget.
REFERENCE_PROFILES: dict[int, dict[str, int | str]] = {
    32: {"stage1": 54, "stage2": 18, "source": "paper Table 1 / Section 4.1"},
    64: {"stage1": 111, "stage2": 37, "source": "paper Table 1 / Section 4.1"},
    128: {"stage1": 219, "stage2": 73, "source": "paper Table 1 / Section 4.1"},
    224: {"stage1": 384, "stage2": 128, "source": "Section 4.1 extrapolation"},
}


def stage_counts(budget: int, depth: int = REFERENCE_DEPTH,
                 layer: int = PRUNING_LAYER) -> tuple[int, int, float]:
    """Resolve the paper's 3:1 reference profile onto a decoder depth."""
    if depth < 2 or layer != PRUNING_LAYER:
        raise ValueError("Invalid LearnPruner budget, decoder depth, or pruning layer")
    resolved = stage_schedule(budget, depth)
    return int(resolved.n_visual), int(resolved.keeps[0]), resolved.layer_average


def stage_schedule(budget: int, depth: int, *, stage1_tokens: int | None = None,
                   stage2_tokens: int | None = None) -> LayerSchedule:
    """Return the depth-resolved schedule used by installation."""
    if (stage1_tokens is None) != (stage2_tokens is None):
        raise ValueError("stage1_tokens and stage2_tokens must be provided together")
    if stage1_tokens is None:
        profile = REFERENCE_PROFILES.get(int(budget))
        if profile is None:
            raise ValueError(
                f"LearnPruner has profiles for {tuple(REFERENCE_PROFILES)}; "
                "pass explicit stage1_tokens and stage2_tokens to explore another budget"
            )
        r1, r2 = int(profile["stage1"]), int(profile["stage2"])
    else:
        r1, r2 = int(stage1_tokens), int(stage2_tokens)
        if r1 != 3 * r2:
            raise ValueError("LearnPruner requires the paper's stage1_tokens:stage2_tokens=3:1")
    if r2 < 1:
        raise ValueError("LearnPruner stage token counts must be positive")
    return LayerSchedule(
        boundaries=(PRUNING_LAYER,),
        keeps=(r2,),
        depth=REFERENCE_DEPTH,
        n_visual=r1,
    ).rescaled_to(depth, label="the LearnPruner schedule")


def pruning_boundary(depth: int, reference_layer: int = PRUNING_LAYER) -> int:
    """Scale the paper's layer boundary to the loaded decoder depth."""
    depth = int(depth)
    if depth < 2:
        raise ValueError(f"LearnPruner needs at least two decoder layers, got {depth}")
    boundary = scale_layer(int(reference_layer), depth, reference=REFERENCE_DEPTH)
    if not 0 < boundary < depth:
        raise ValueError(
            f"LearnPruner's layer {reference_layer}/{REFERENCE_DEPTH} boundary does not fit "
            f"a {depth}-layer decoder"
        )
    return boundary


def ratio_stage_counts(token_count: int, retain_ratio: float, depth: int,
                       layer: int) -> tuple[int, int, float]:
    """Resolve the paper's 3:1 stages from a layer-average retention ratio.

    Counts are chosen by nearest-token rounding. Except for the explicit
    identity ratio, ``R1`` is exactly three times ``R2`` as prescribed by the
    paper. The result is resolved independently for every NeXT image because
    its packed AnyRes span is variable length.
    """
    native = int(token_count)
    depth = int(depth)
    layer = int(layer)
    ratio = float(retain_ratio)
    if native < 1:
        raise ValueError("LearnPruner cannot schedule an empty visual span")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
    if not 0 < layer < depth:
        raise ValueError(
            f"LearnPruner needs 0 < pruning layer < decoder depth, got {layer}/{depth}"
        )
    if ratio == 1.0:
        return native, native, 1.0

    denominator = depth + 2 * layer  # layer*3R2 + (depth-layer)*R2
    maximum_r2 = native // 3
    if maximum_r2 < 1:
        raise ValueError(
            f"LearnPruner's 3:1 stages need at least three visual tokens, got {native}"
        )
    target_mass = ratio * depth * native
    maximum_mass = denominator * maximum_r2
    minimum_mass = denominator
    if target_mass > maximum_mass + 1e-9:
        maximum_ratio = maximum_mass / (depth * native)
        raise ValueError(
            f"LearnPruner retain_ratio={ratio:g} is above the 3:1 schedule's maximum "
            f"{maximum_ratio:g} for a {native}-token span; use retain_ratio=1 for an "
            "identity pass-through."
        )
    if target_mass < minimum_mass - 1e-9:
        minimum_ratio = minimum_mass / (depth * native)
        raise ValueError(
            f"LearnPruner retain_ratio={ratio:g} is too small for a non-empty 3:1 "
            f"schedule; this image requires at least {minimum_ratio:g}."
        )

    exact_r2 = target_mass / denominator
    lower = max(1, min(maximum_r2, int(exact_r2)))
    upper = max(1, min(maximum_r2, lower + 1))
    r2 = min((lower, upper), key=lambda value: (abs(value - exact_r2), value))
    r1 = 3 * r2
    achieved = denominator * r2 / (depth * native)
    return r1, r2, achieved


def max_min_diversity(features: torch.Tensor, seeds: torch.Tensor,
                      count: int) -> torch.Tensor:
    """Iteratively select the token least similar to its nearest selected token."""
    if features.ndim != 3 or seeds.ndim != 2 or seeds.shape[0] != features.shape[0]:
        raise ValueError("Expected features [B,N,D] and seeds [B,K]")
    batch, n_tokens, _ = features.shape
    if seeds.shape[1] < 1 or seeds.shape[1] + count > n_tokens:
        raise ValueError("Diversity selection needs nonempty seeds and an available budget")
    normalized = F.normalize(features.float(), dim=-1)
    chosen = torch.zeros(batch, n_tokens, dtype=torch.bool, device=features.device)
    chosen.scatter_(1, seeds, True)
    seed_features = normalized.gather(1, seeds.unsqueeze(-1).expand(-1, -1, normalized.shape[-1]))
    maximum = (normalized @ seed_features.transpose(1, 2)).amax(dim=-1)
    additions = []
    for _ in range(count):
        candidate = maximum.masked_fill(chosen, float("inf")).argmin(dim=1)
        additions.append(candidate)
        chosen.scatter_(1, candidate[:, None], True)
        similarity = (normalized * normalized[torch.arange(batch, device=features.device), candidate][:, None]).sum(-1)
        maximum = torch.maximum(maximum, similarity)
    return torch.stack(additions, dim=1) if additions else seeds[:, :0]


def ste_keep_mask(probabilities: torch.Tensor) -> torch.Tensor:
    """Argmax hard mask in the forward pass, keep probability in the backward."""
    keep = probabilities[..., 1]
    hard = (probabilities.argmax(dim=-1) == 1).to(keep.dtype)
    return hard.detach() - keep.detach() + keep


def graph_attention_weights(logits: torch.Tensor, causal_mask: torch.Tensor | None,
                            token_mask: torch.Tensor) -> torch.Tensor:
    """Eq. 6: masked keys cannot reach other queries; the diagonal stays live."""
    batch, _, query_len, key_len = logits.shape
    if query_len != key_len or token_mask.shape != (batch, key_len):
        raise ValueError("LearnPruner training requires full square self-attention")
    columns = torch.arange(key_len, device=logits.device)
    logits = logits.float().masked_fill(columns[None, None, None, :] > columns[None, None, :, None],
                                        torch.finfo(torch.float32).min)
    if causal_mask is not None:
        if causal_mask.ndim != 4 or causal_mask.shape[-2:] != (query_len, key_len):
            raise ValueError("Expected a full 4-D causal/padding mask")
        if causal_mask.dtype == torch.bool:
            logits = logits.masked_fill(~causal_mask, torch.finfo(torch.float32).min)
        else:
            logits = logits + causal_mask.float()
    # Stable form of exp(logits) * G / sum(exp(logits) * G).
    adjacency = token_mask.float()[:, None, None, :].expand(batch, 1, query_len, key_len)
    identity = torch.eye(key_len, device=logits.device, dtype=adjacency.dtype)[None, None]
    adjacency = adjacency * (1 - identity) + identity
    unpruned = logits.softmax(dim=-1)
    weighted = unpruned * adjacency
    return weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(1e-20)


class LearnPrunerCompressor(TokenCompressor):
    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS, *,
                 checkpoint_path: str | PathLike[str] | None = None,
                 pretrained: bool | str | PathLike[str] | None = None,
                 retain_ratio: float | None = None,
                 diversity_ratio: float = DIVERSITY_RATIO, pruning_layer: int = PRUNING_LAYER,
                 feature_dim: int | None = None, hidden_dim: int = HIDDEN_DIM,
                 stage1_tokens: int | None = None, stage2_tokens: int | None = None,
                 for_training: bool = False, track_source: bool = True) -> None:
        super().__init__(max_tokens=int(max_tokens))
        if retain_ratio is not None and not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
        resolved_feature_dim = FEATURE_DIM if feature_dim is None else int(feature_dim)
        if (not 0 <= diversity_ratio < 1 or resolved_feature_dim < 1
                or hidden_dim < 1 or self._max_tokens < 1):
            raise ValueError("Invalid diversity ratio or LPM width")
        if int(pruning_layer) != PRUNING_LAYER:
            raise ValueError(
                f"LearnPruner fixes its reference pruning boundary at layer {PRUNING_LAYER}"
            )
        self.diversity_ratio = float(diversity_ratio)
        self.pruning_layer = int(pruning_layer)
        self.feature_dim = resolved_feature_dim
        self._feature_dim_explicit = feature_dim is not None
        self.hidden_dim = int(hidden_dim)
        self.stage1_tokens = stage1_tokens
        self.stage2_tokens = stage2_tokens
        self._retain_ratio = None if retain_ratio is None else float(retain_ratio)
        self.for_training = bool(for_training)
        self.track_source = bool(track_source)
        self.lpm = self._new_lpm(self.feature_dim)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._installed_vlm: VLM | None = None
        self._original_image_forward: Any = None
        self._original_prepare_inputs: Any = None
        self._original_attention_forwards: list[tuple[Any, Any]] = []
        self._handles: list[Any] = []
        self._image_token_id: int | None = None
        reference_schedule = stage_schedule(
            int(self._max_tokens), REFERENCE_DEPTH,
            stage1_tokens=self.stage1_tokens, stage2_tokens=self.stage2_tokens,
        )
        self._stage1_count = int(reference_schedule.n_visual)
        self._stage2_count = int(reference_schedule.keeps[0])
        self._resolved_pruning_layer: int = 0
        self._input_tokens: int = 576
        self._mask: torch.Tensor | None = None
        self._image_start: torch.Tensor | None = None
        self._query_columns: torch.Tensor | None = None
        self._query_valid: torch.Tensor | None = None
        self._cut_state: dict[str, Any] | None = None
        self._pending_scores: torch.Tensor | None = None
        self._source_history: list[torch.Tensor] = []
        self._stage1_source: torch.Tensor | None = None
        self._last_keep_index: torch.Tensor | None = None
        self._original_count: int = 0
        self._feature_space = "vision_encoder"
        self._checkpoint_metadata: Mapping[str, Any] | None = None
        self._training_masks: tuple[torch.Tensor, ...] = ()
        self._llava_next_adapter: Any | None = None
        self._installed_backbone_key: str | None = None
        self._pretrained_arg = pretrained
        if self.checkpoint_path is not None:
            self.load_checkpoint(self.checkpoint_path)

    def _new_lpm(self, feature_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(int(feature_dim), self.hidden_dim), nn.GELU(),
                             nn.Linear(self.hidden_dim, 2))

    def _configure_lpm(self, feature_dim: int, *, feature_space: str) -> None:
        """Materialize the backbone-specific LPM input space before use."""
        feature_dim = int(feature_dim)
        if feature_dim < 1:
            raise ValueError(f"LearnPruner received invalid feature width {feature_dim}")
        if self.feature_dim != feature_dim:
            if self._feature_dim_explicit or self._checkpoint_metadata is not None:
                raise ValueError(
                    f"LearnPruner needs {feature_dim}-wide {feature_space} features on this "
                    f"backbone, but the configured/checkpoint LPM expects {self.feature_dim}."
                )
            self.feature_dim = feature_dim
            self.lpm = self._new_lpm(feature_dim)
        checkpoint_space = (
            self._checkpoint_metadata.get("feature_space")
            if self._checkpoint_metadata is not None else None
        )
        if checkpoint_space is not None and checkpoint_space != feature_space:
            raise ValueError(
                f"LearnPruner checkpoint scores {checkpoint_space} features, not "
                f"{feature_space} features required by this backbone."
            )
        self._feature_space = feature_space

    @property
    def budget(self) -> int | float:
        """Fixed token count on LLaVA-1.5; layer-average ratio on NeXT."""
        return self._max_tokens if self._retain_ratio is None else self._retain_ratio

    @property
    def retain_ratio(self) -> float | None:
        return self._retain_ratio

    @property
    def feature_space(self) -> str:
        return self._feature_space

    @property
    def resolved_pruning_layer(self) -> int:
        return self._resolved_pruning_layer

    def validate_llava_next(self) -> None:
        """NeXT has one public, ratio-valued compression control."""
        from llmoe.compressor.llava_next import validate_retain_ratio

        validate_retain_ratio(self._retain_ratio, label="LearnPruner")
        if self._max_tokens != DEFAULT_MAX_TOKENS:
            raise ValueError(
                "LearnPruner on LLaVA-NeXT only accepts retain_ratio; remove max_tokens "
                f"(its constructor default is {DEFAULT_MAX_TOKENS} and is ignored on NeXT)."
            )
        if self.stage1_tokens is not None or self.stage2_tokens is not None:
            raise ValueError(
                "LearnPruner/LLaVA-NeXT derives both stages from retain_ratio; remove "
                "stage1_tokens and stage2_tokens."
            )

    @property
    def trainable_parameters(self) -> Any:
        return self.lpm.parameters()

    def resolve_next_stage_counts(self, token_count: int) -> tuple[int, int, float]:
        if self._installed_vlm is None or self._retain_ratio is None:
            raise RuntimeError("LearnPruner/LLaVA-NeXT is not installed")
        layers = self._installed_vlm.model.model.language_model.layers
        return ratio_stage_counts(
            int(token_count), self._retain_ratio, len(layers), self._resolved_pruning_layer,
        )

    def use_next_stage_counts(self, token_count: int) -> tuple[int, int, float]:
        stage1, stage2, average = self.resolve_next_stage_counts(token_count)
        self._input_tokens = int(token_count)
        self._stage1_count = stage1
        self._stage2_count = stage2
        return stage1, stage2, average

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, **kwargs: Any) -> CompressedFeatures:
        del attention, layer_idx, kwargs
        squeezed = features.ndim == 2
        if squeezed:
            features = features.unsqueeze(0)
        if features.ndim != 3 or features.shape[-1] != self.feature_dim or features.shape[1] < 1:
            raise ValueError("LearnPruner needs nonempty [B,N,feature_dim] image features")
        n_tokens = features.shape[1]
        count = min(self._stage1_count or self._max_tokens, n_tokens)
        diversity = min(count - 1, round(count * self.diversity_ratio))
        important = count - diversity
        probabilities = self.lpm(features.float()).softmax(dim=-1)
        seeds = probabilities[..., 1].topk(important, dim=1).indices
        additions = max_min_diversity(features, seeds, diversity)
        indices = torch.cat((seeds, additions), dim=1).sort(dim=1).values
        selected = features.gather(1, indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
        if self.track_source:
            self._original_count = n_tokens
            self._stage1_source = features.new_zeros((count, n_tokens))
            self._stage1_source.scatter_(1, indices[0, :, None], 1.0)
            self._source_history = [
                torch.eye(n_tokens, device=features.device, dtype=features.dtype),
                self._stage1_source,
            ]
        return CompressedFeatures(tokens=selected.squeeze(0) if squeezed else selected)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        return self._source_history

    def checkpoint_dict(self) -> dict[str, Any]:
        return {"format_version": 2, "method": "LearnPruner",
                "feature_dim": self.feature_dim, "hidden_dim": self.hidden_dim,
                "activation": "gelu", "classes": 2,
                "feature_space": self._feature_space,
                "backbone": self._installed_backbone_key,
                "lpm": self.lpm.state_dict()}

    def save_checkpoint(self, path: str | PathLike[str]) -> None:
        torch.save(self.checkpoint_dict(), path)

    def load_checkpoint(self, path: str | PathLike[str]) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or payload.get("method") != "LearnPruner":
            raise ValueError("Expected a LearnPruner LPM checkpoint")
        version = int(payload.get("format_version", 0))
        if version not in (1, 2):
            raise ValueError(f"Unsupported LearnPruner checkpoint format {version}")
        checkpoint_dim = int(payload.get("feature_dim", 0))
        if payload.get("hidden_dim") != self.hidden_dim or checkpoint_dim < 1:
            raise ValueError("LearnPruner checkpoint LPM widths differ from configuration")
        if payload.get("activation") != "gelu" or payload.get("classes") != 2:
            raise ValueError("LearnPruner checkpoint uses a different LPM architecture")
        if checkpoint_dim != self.feature_dim:
            if self._feature_dim_explicit:
                raise ValueError("LearnPruner checkpoint LPM widths differ from configuration")
            self.feature_dim = checkpoint_dim
            self.lpm = self._new_lpm(checkpoint_dim)
        self.lpm.load_state_dict(payload["lpm"], strict=True)
        # v1 is the original raw-CLIP LLaVA-1.5 format. Keeping that default
        # makes existing 7B checkpoints load while still preventing them from
        # being mistaken for post-packed NeXT checkpoints.
        self._checkpoint_metadata = dict(payload)
        self._checkpoint_metadata.setdefault("feature_space", "vision_encoder")
        if version == 1:
            # Format 1 predates multiple backbones and was produced only by the
            # original LLaVA-1.5-7B trainer.
            self._checkpoint_metadata.setdefault("backbone", "llava")

    def _resolve_pretrained(self, vlm: VLM) -> None:
        """Fetch the released LPM for this backbone, when one is wanted.

        Deferred to ``install`` rather than done in ``__init__`` because the
        releases are per-backbone and the backbone is not known until then --
        the same reason QuietPrune resolves its adapter here. Training is left
        alone: a run that starts from released weights is a finetune, and says
        so by passing ``checkpoint_path``.
        """
        if self.checkpoint_path is not None or self.for_training:
            return
        if self._pretrained_arg is False:
            return
        from llmoe.weights import resolve_weights

        resolved = resolve_weights(
            "LearnPruner", int(self._max_tokens),
            pretrained=self._pretrained_arg,
            backbone=backbone_key(vlm),
        )
        if resolved is None:
            return
        self.checkpoint_path = Path(resolved)
        self.load_checkpoint(self.checkpoint_path)

    def _validate_checkpoint_backbone(self, vlm: VLM) -> None:
        expected = backbone_key(vlm)
        actual = (self._checkpoint_metadata or {}).get("backbone")
        if actual is not None and expected is not None and actual != expected:
            raise ValueError(
                f"LearnPruner checkpoint was trained for {actual}, not {expected}. "
                "Train or select the LPM checkpoint for this backbone."
            )

    @override
    def install(self, vlm: VLM) -> None:
        if self._installed_vlm is vlm:
            return
        if self._installed_vlm is not None:
            raise RuntimeError("LearnPruner is already installed on another backbone")
        inner = getattr(vlm.model, "model", None)
        tower = getattr(inner, "vision_tower", None)
        projector = getattr(inner, "multi_modal_projector", None)
        language = getattr(inner, "language_model", None)
        layers = getattr(language, "layers", None)
        if tower is None or projector is None or layers is None or getattr(inner, "visual", None) is not None:
            raise TypeError("LearnPruner requires a CLIP/Llama LLaVA or LLaVA-NeXT backbone")

        from llmoe.compressor.llava_next import is_llava_next

        # Before the LPM is sized: a checkpoint carries its own feature width, and
        # `_configure_lpm` checks the backbone's width against it.
        self._resolve_pretrained(vlm)
        llava_next = is_llava_next(vlm)
        depth = len(layers)
        config = vlm.model.config
        self._resolved_pruning_layer = pruning_boundary(depth, self.pruning_layer)
        self._input_tokens = int(getattr(config, "image_seq_length", 576))
        if llava_next:
            self.validate_llava_next()
            text_config = getattr(config, "text_config", None)
            feature_dim = getattr(text_config, "hidden_size", None)
            if feature_dim is None:
                raise TypeError("LearnPruner could not resolve LLaVA-NeXT's projected width")
            self._configure_lpm(int(feature_dim), feature_space="packed_projected_anyres")
            # Real AnyRes spans are scheduled independently; this validates the
            # ratio against a normal single-crop span before hooks are installed.
            self._stage1_count, self._stage2_count, average = ratio_stage_counts(
                self._input_tokens, float(self._retain_ratio), depth,
                self._resolved_pruning_layer,
            )
        else:
            if self._retain_ratio is not None:
                raise ValueError(
                    "LearnPruner on LLaVA-1.5 7B/13B uses the constant max_tokens "
                    "budget; retain_ratio is only for LLaVA-NeXT."
                )
            schedule = stage_schedule(
                int(self._max_tokens), depth,
                stage1_tokens=self.stage1_tokens, stage2_tokens=self.stage2_tokens,
            )
            if schedule.boundaries[0] != self._resolved_pruning_layer:
                raise RuntimeError("LearnPruner depth solvers disagree on the pruning boundary")
            self._stage1_count = int(schedule.n_visual)
            self._stage2_count = int(schedule.keeps[0])
            average = schedule.layer_average
            vision_config = getattr(config, "vision_config", None)
            feature_dim = getattr(vision_config, "hidden_size", FEATURE_DIM)
            self._configure_lpm(int(feature_dim), feature_space="vision_encoder")
            if self._stage1_count > self._input_tokens:
                raise ValueError(
                    f"This layer-average budget needs {self._stage1_count} stage-one tokens, "
                    f"but the LLaVA configuration provides {self._input_tokens}."
                )

        if self.checkpoint_path is None and not self.for_training:
            raise ValueError(
                "LearnPruner inference requires a trained LPM: pass `pretrained: true` "
                "for the released one, or `checkpoint_path` for your own."
            )
        self._validate_checkpoint_backbone(vlm)
        token_id = getattr(config, "image_token_id", None)
        if token_id is None:
            token_id = getattr(config, "image_token_index", None)
        if token_id is None:
            raise TypeError("LLaVA image token ID is unavailable")
        self._image_token_id = int(token_id)
        self.lpm.to(device=next(projector.parameters()).device, dtype=torch.float32)
        self._installed_vlm = vlm
        self._installed_backbone_key = backbone_key(vlm)
        if llava_next:
            from llmoe.compressor.learnpruner.learn_pruner_llava_next import (
                LlavaNextLearnPrunerAdapter,
            )

            self._llava_next_adapter = LlavaNextLearnPrunerAdapter(self)
            self._llava_next_adapter.install(vlm)
        else:
            self._original_image_forward = inner.get_image_features
            inner.get_image_features = self._get_image_features
        if self.for_training:
            self._handles.append(inner.register_forward_pre_hook(self._capture_training_span, with_kwargs=True))
            for layer in layers:
                attn = layer.self_attn
                self._original_attention_forwards.append((attn, attn.forward))
                attn.forward = self._make_training_attention(attn)
        else:
            if not llava_next:
                self._original_prepare_inputs = vlm.prepare_inputs
                vlm.prepare_inputs = self._prepare_inputs
            self._handles.append(inner.register_forward_pre_hook(self._capture_inference_span, with_kwargs=True))
            score_layer = layers[self._resolved_pruning_layer - 1]
            self._handles.append(score_layer.register_forward_pre_hook(
                self._score_query_attention, with_kwargs=True))
            self._handles.append(score_layer.register_forward_hook(
                self._finish_scoring_layer, with_kwargs=True))
            for idx in range(self._resolved_pruning_layer, depth):
                self._handles.append(layers[idx].register_forward_pre_hook(
                    self._make_decoder_hook(idx), with_kwargs=True))
        stages = (
            f"dynamic R1:R2=3:1, retain_ratio={self._retain_ratio:g}"
            if llava_next and self._retain_ratio != 1.0
            else ("identity, retain_ratio=1" if llava_next else
                  f"stage1={self._stage1_count}, stage2={self._stage2_count}, "
                  f"layer-average={average:g}")
        )
        print(
            f"[LearnPruner{'/LLaVA-NeXT' if llava_next else ''}] {stages}, "
            f"boundary={self._resolved_pruning_layer}/{depth} "
            f"(paper={self.pruning_layer}/{REFERENCE_DEPTH}), "
            f"feature_space={self._feature_space}, diversity={self.diversity_ratio:g}",
            flush=True,
        )

    @override
    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for attn, original in self._original_attention_forwards:
            attn.forward = original
        self._original_attention_forwards.clear()
        if self._llava_next_adapter is not None:
            self._llava_next_adapter.uninstall()
            self._llava_next_adapter = None
        elif self._installed_vlm is not None:
            self._installed_vlm.model.model.get_image_features = self._original_image_forward
        if self._installed_vlm is not None and self._original_prepare_inputs is not None:
            self._installed_vlm.prepare_inputs = self._original_prepare_inputs
        self._installed_vlm = None
        self._installed_backbone_key = None
        self._original_image_forward = None
        self._original_prepare_inputs = None
        self._resolved_pruning_layer = 0
        self._mask = None
        self._training_masks = ()
        self._image_start = None
        self._query_columns = None
        self._query_valid = None
        self._cut_state = None
        self._pending_scores = None

    def _get_image_features(self, pixel_values: torch.Tensor, vision_feature_layer: int,
                            vision_feature_select_strategy: str, **kwargs: Any) -> BaseModelOutputWithPooling:
        inner = self._installed_vlm.model.model
        if vision_feature_layer is None:
            vision_feature_layer = self._installed_vlm.model.config.vision_feature_layer
        if vision_feature_select_strategy is None:
            vision_feature_select_strategy = self._installed_vlm.model.config.vision_feature_select_strategy
        if not isinstance(vision_feature_layer, int) or vision_feature_select_strategy != "default":
            raise ValueError("LearnPruner requires LLaVA-1.5's single CLIP patch feature layer")
        if kwargs.get("image_sizes") is not None:
            raise ValueError("LearnPruner's LLaVA-1.5 path does not support image_sizes tiling")
        outputs = inner.vision_tower(pixel_values, output_hidden_states=True, return_dict=True)
        patches = outputs.hidden_states[vision_feature_layer][:, 1:]
        if patches.shape[-1] != self.feature_dim:
            raise ValueError(f"CLIP feature width {patches.shape[-1]} != LPM width {self.feature_dim}")
        if self.for_training:
            self.set_training_features(tuple(patches.unbind(0)))
            features = inner.multi_modal_projector(patches)
        else:
            selected = self.compress(patches).tokens
            features = inner.multi_modal_projector(selected)
        outputs.pooler_output = list(features)
        return outputs

    def set_training_features(self, features: tuple[torch.Tensor, ...]) -> None:
        """Build STE masks for fixed or ragged post-encoder image sequences."""
        masks: list[torch.Tensor] = []
        for sequence in features:
            if (sequence.ndim != 2 or sequence.shape[0] < 1
                    or sequence.shape[-1] != self.feature_dim):
                raise ValueError(
                    "LearnPruner training needs nonempty image features shaped "
                    f"[tokens,{self.feature_dim}], got {tuple(sequence.shape)}"
                )
            probabilities = self.lpm(sequence.float()).softmax(dim=-1)
            masks.append(ste_keep_mask(probabilities))
        if not masks:
            raise ValueError("LearnPruner training received no image feature sequences")
        self._training_masks = tuple(masks)
        self._mask = (
            torch.stack(masks, dim=0)
            if len({int(mask.shape[0]) for mask in masks}) == 1 else None
        )

    def _capture_training_span(self, _module: Any, args: tuple, kwargs: dict) -> None:
        ids = kwargs.get("input_ids", args[0] if args else None)
        if isinstance(ids, torch.Tensor) and ids.ndim == 2:
            self._training_ids = ids
            self._training_attention = kwargs.get("attention_mask")

    def full_training_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        ids = getattr(self, "_training_ids", None)
        if (not isinstance(ids, torch.Tensor) or ids.shape[1] != seq_len
                or len(self._training_masks) != ids.shape[0]):
            raise RuntimeError("LearnPruner training span/mask was not captured")
        visual = ids.to(device) == self._image_token_id
        expected = [int(mask.shape[0]) for mask in self._training_masks]
        if visual.sum(dim=1).tolist() != expected:
            raise RuntimeError(
                "Training image placeholders differ from LearnPruner image features: "
                f"placeholders={visual.sum(dim=1).tolist()}, features={expected}"
            )
        mask = torch.ones(ids.shape, dtype=self._training_masks[0].dtype, device=device)
        for row, row_mask in enumerate(self._training_masks):
            mask[row, visual[row]] = row_mask.to(device)
        return mask

    def pruning_loss(self) -> torch.Tensor:
        if not self._training_masks:
            raise RuntimeError("LearnPruner did not produce an LPM mask")
        return torch.stack([
            mask.mean().square() for mask in self._training_masks
        ]).mean()  # Eq. 7, tau=0

    def _make_training_attention(self, attn: Any) -> Any:
        def forward(hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor],
                    attention_mask: torch.Tensor | None = None, past_key_values: Any = None,
                    **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            if past_key_values is not None:
                raise ValueError("LearnPruner STE training must use_cache=False")
            batch, seq_len, _ = hidden_states.shape
            head_dim = attn.head_dim
            heads = attn.config.num_attention_heads
            kv_heads = attn.config.num_key_value_heads
            query = attn.q_proj(hidden_states).view(batch, seq_len, heads, head_dim).transpose(1, 2)
            key = attn.k_proj(hidden_states).view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
            value = attn.v_proj(hidden_states).view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
            key = repeat_kv(key, attn.num_key_value_groups)
            value = repeat_kv(value, attn.num_key_value_groups)
            logits = (query.float() @ key.float().transpose(-1, -2)) * attn.scaling
            weights = graph_attention_weights(logits, attention_mask,
                                              self.full_training_mask(seq_len, hidden_states.device))
            result = (weights.to(value.dtype) @ value).transpose(1, 2).reshape(batch, seq_len, -1)
            return attn.o_proj(result), weights
        return forward

    def _prepare_inputs(self, *args: Any, **kwargs: Any) -> Any:
        inputs = self._original_prepare_inputs(*args, **kwargs)
        ids = inputs.get("input_ids")
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
            return inputs
        mask = inputs.get("attention_mask")
        rows, masks = [], []
        for row_idx, row in enumerate(ids):
            positions = (row == self._image_token_id).nonzero(as_tuple=True)[0]
            if positions.numel() not in (0, self._input_tokens):
                raise ValueError(
                    f"LearnPruner expects one {self._input_tokens}-patch LLaVA image per prompt"
                )
            keep = torch.ones_like(row, dtype=torch.bool)
            if positions.numel():
                keep[positions[self._stage1_count:]] = False
            rows.append(row[keep])
            if mask is not None:
                masks.append(mask[row_idx][keep])
        width = max(row.numel() for row in rows)
        pad = getattr(self._installed_vlm.processor.tokenizer, "pad_token_id", None) or 0
        new_ids = ids.new_full((len(rows), width), int(pad))
        new_mask = mask.new_zeros((len(rows), width)) if mask is not None else None
        for i, row in enumerate(rows):
            new_ids[i, -row.numel():] = row
            if new_mask is not None:
                new_mask[i, -row.numel():] = masks[i]
        inputs["input_ids"] = new_ids
        if new_mask is not None:
            inputs["attention_mask"] = new_mask
        return inputs

    def _capture_inference_span(self, _module: Any, args: tuple, kwargs: dict) -> None:
        ids = kwargs.get("input_ids", args[0] if args else None)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[1] <= 1:
            return
        self._cut_state = None
        self._pending_scores = None
        self._last_keep_index = None
        visual = [(row == self._image_token_id).nonzero(as_tuple=True)[0] for row in ids]
        if all(row.numel() == 0 for row in visual):
            self._image_start = None
            return
        if any(row.numel() != self._stage1_count or
               int(row[-1] - row[0] + 1) != self._stage1_count for row in visual):
            raise ValueError("LearnPruner stage-one visual spans must be contiguous and equal")
        starts = torch.stack([row[0] for row in visual])
        mask = kwargs.get("attention_mask")
        query_lists = []
        for i, row in enumerate(ids):
            positions = torch.arange(row.numel(), device=row.device)
            valid = (positions >= starts[i] + self._stage1_count)
            if isinstance(mask, torch.Tensor):
                valid &= mask[i].bool()
            query_lists.append(positions[valid])
        if any(q.numel() == 0 for q in query_lists):
            raise ValueError("LearnPruner requires query text after the image span")
        width = max(q.numel() for q in query_lists)
        query = ids.new_zeros((ids.shape[0], width))
        valid = torch.zeros_like(query, dtype=torch.bool)
        for i, q in enumerate(query_lists):
            query[i, :q.numel()] = q
            query[i, q.numel():] = q[-1]
            valid[i, :q.numel()] = True
        self._image_start = starts
        self._query_columns = query
        self._query_valid = valid

    def _make_decoder_hook(self, index: int) -> Any:
        def hook(module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
            hidden = args[0] if args else kwargs.get("hidden_states")
            if not isinstance(hidden, torch.Tensor) or self._image_start is None:
                return None
            if index == self._resolved_pruning_layer and hidden.shape[1] == 1 \
                    and self._cut_state is not None:
                advance_decode_step(self._cut_state, hidden, label="LearnPruner")
            if self._cut_state is None:
                return None
            apply_prune_state(kwargs, self._cut_state, prefill=hidden.shape[1] > 1)
            return (hidden,), kwargs
        return hook

    def _score_query_attention(self, layer: Any, args: tuple, kwargs: dict) -> None:
        """Read the twelfth layer's text-to-vision map while it is still full-width."""
        hidden = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[1] <= 1 \
                or self._image_start is None:
            return None
        start = self._image_start.to(hidden.device)
        columns = start[:, None] + torch.arange(self._stage1_count, device=hidden.device)
        rows = decoder_attention_rows(layer, hidden, kwargs.get("position_embeddings"),
                                      kwargs.get("attention_mask"),
                                      self._query_columns.to(hidden.device),
                                      apply_rope=apply_rotary_pos_emb, label="LearnPruner")
        vision_rows = rows.gather(2, columns[:, None].expand(-1, rows.shape[1], -1))
        valid = self._query_valid.to(hidden.device)
        self._pending_scores = ((vision_rows * valid[:, :, None]).sum(dim=1)
                                / valid.sum(dim=1)[:, None])
        return None

    def _finish_scoring_layer(self, _layer: Any, args: tuple, kwargs: dict,
                              output: torch.Tensor) -> torch.Tensor:
        """Prune after layer 12, so exactly the first 12 layers see R1 tokens."""
        if self._pending_scores is None or self._cut_state is not None \
                or not isinstance(output, torch.Tensor) or output.shape[1] <= 1:
            return output
        scores = self._pending_scores
        self._pending_scores = None
        self._cut_state, return_value = self._cut(output, kwargs, scores)
        return return_value

    def _cut(self, hidden: torch.Tensor, kwargs: dict,
             scores: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
        batch, seq_len, width = hidden.shape
        start = self._image_start.to(hidden.device)
        columns = start[:, None] + torch.arange(self._stage1_count, device=hidden.device)
        local = scores.topk(self._stage2_count, dim=1).indices.sort(dim=1).values
        kept_columns = columns.gather(1, local)
        keep = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden.device)
        keep.scatter_(1, columns, False)
        keep.scatter_(1, kept_columns, True)
        keep_index = keep_columns(keep, label="LearnPruner")
        shortened = hidden.gather(1, keep_index[:, :, None].expand(-1, -1, width))
        position_ids = gather_position_ids(kwargs.get("position_ids"), keep_index, seq_len)
        language = self._installed_vlm.model.model.language_model
        rope = language.rotary_emb(shortened, position_ids=position_ids)
        attention, valid_keys = pruned_attention_mask(kwargs.get("attention_mask"), keep_index,
                                                       seq_len, label="LearnPruner")
        state = {"batch": batch, "position_ids": position_ids, "position_embeddings": rope,
                 "attention_mask": attention, "valid": valid_keys,
                 "cache_position": torch.arange(shortened.shape[1], device=hidden.device),
                 "kv_len": shortened.shape[1]}
        self._last_keep_index = keep_index
        if self.track_source and self._stage1_source is not None:
            self._source_history.append(self._stage1_source[local[0]])
        return state, shortened
