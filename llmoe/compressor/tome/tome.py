"""Token Merging (ToMe) for LLaVA, LLaVA-NeXT, and Qwen visual tokens.

Paper: Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023.
Reference code: https://github.com/facebookresearch/ToMe.

``ToMeCompressor`` is the faithful *in-ViT* variant: instead of compressing the
projector output it patches every CLIP encoder layer so merging happens between
attention and the MLP, where the paper puts it. Each patched layer computes Q/K/V
manually (to expose head-averaged ``K`` as the matching metric and to add
``log(size)`` to the attention logits), runs bipartite soft matching on the
post-attention residual, and carries the updated ``size`` into the next layer.

``[CLS]`` is protected from merging and stays at index 0, so LLaVA's
``vision_feature_select_strategy="default"`` keeps working. The per-layer ``r_i``
schedule is chosen so the layer LLaVA reads (``config.vision_feature_layer``, -2
on LLaVA-1.5) outputs ``max_tokens + 1`` tokens — ``max_tokens`` after the
``[CLS]`` drop.

Harness notes
-------------
* Like ``must_drop.py`` and ``apet.py``, this drives a stock transformers
  ``LlavaForConditionalGeneration``: the ViT is patched through ``forward`` swaps
  rather than by editing model source. ``install()`` is idempotent and
  ``uninstall()`` restores every original.
* Each image now yields ``max_tokens`` features, so surplus ``<image>``
  placeholders are trimmed in a ``prepare_inputs`` wrapper (transformers 5.x
  scatters image features with a strict count check).
* Qwen2.5-VL follows the VisionZip hook pattern: the last windowed-ViT block's
  rotary keys are captured and folded through the patch merger, then the same
  bipartite policy merges the projected visual span after concatenation. Its
  target is ``retain_ratio`` of the image's dynamic token count. This is an
  adaptation; the ToMe paper validates the in-ViT formulation, not Qwen.
* LLaVA-NeXT captures head-averaged keys at the checkpoint-selected CLIP layer,
  applies the same bipartite similarity matching and size-weighted averaging
  independently to the base crop and each valid local tile region, then restores
  raster order from merged centroids and inserts protected newline rows. Stock
  AnyRes packing cannot consume progressively shortened square CLIP grids, so
  this preserves ToMe's defining matching rule but adapts its placement from the
  paper's exact in-ViT path.
* Source matrices for ``llmoe.viz`` are only recorded when ``track_source`` is
  set; pass ``track_source: true`` under ``compressor_args`` before running
  ``bm --visualize``.
"""

from typing import Any, Literal, override

import torch
import torch.nn.functional as F

from llmoe.backbone.prototype import VLM
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor

MetricSource = Literal["k_mean", "hidden_state"]
Schedule = Literal["uniform", "front_loaded"]

# Matching rounds used by the post-projector fallback, which has no layer stack
# to spread the reduction over. Mirrors the CLIP layers feeding LLaVA-1.5.
DEFAULT_FALLBACK_ROUNDS = 23


def _max_r(num_tokens: int, protect_cls: bool) -> int:
    """Largest feasible ``r`` for one bipartite step.

    Sources come from the even-indexed half ``A`` (size ``ceil(T / 2)``); a
    protected ``[CLS]`` sits in ``A`` and cannot be one, leaving one fewer.
    """
    half = (num_tokens + 1) // 2
    return max(half - 1, 0) if protect_cls else max(half, 0)


def _nominal_uniform_schedule(total_reductions: int, rounds: int) -> list[int]:
    """Spread ``total_reductions`` evenly, extras first (where the cap is loosest)."""
    base, extra = divmod(total_reductions, rounds)
    return [base + 1] * extra + [base] * (rounds - extra)


def _nominal_front_loaded_schedule(
    initial_tokens: int, total_reductions: int, rounds: int
) -> list[int]:
    """Geometric schedule: shrink by a fixed *fraction* each round, not a fixed count.

    Uses a constant decay ``d = (final / initial) ** (1 / rounds)``, so ``r`` is
    largest in the first round. Useful when few rounds are available and a
    uniform ``r`` would exceed the feasibility cap. The last round absorbs the
    rounding remainder.
    """
    final_tokens = initial_tokens - total_reductions
    if final_tokens <= 0:
        raise ValueError(f"final token count must stay positive, got {final_tokens}")

    decay = (final_tokens / initial_tokens) ** (1.0 / rounds)
    schedule: list[int] = []
    tokens = initial_tokens
    tokens_float = float(initial_tokens)
    remaining = total_reductions
    for index in range(rounds):
        tokens_float *= decay
        r = remaining if index == rounds - 1 else tokens - round(tokens_float)
        r = max(0, min(r, remaining))
        schedule.append(r)
        tokens -= r
        remaining -= r
    return schedule


def _build_schedule(
    initial_tokens: int, total_reductions: int, rounds: int, *, kind: Schedule, protect_cls: bool
) -> list[int]:
    """Build a per-round ``r`` schedule summing to ``total_reductions``.

    ``initial_tokens`` includes ``[CLS]`` when ``protect_cls`` is set. A schedule
    that cannot reach the budget is rejected rather than silently under-merging,
    since the output count must match the trimmed placeholder count exactly.
    """
    if rounds <= 0:
        raise ValueError("the number of merge rounds must be >= 1")
    if total_reductions < 0:
        raise ValueError("total_reductions must be >= 0")
    if total_reductions == 0:
        return [0] * rounds

    if kind == "uniform":
        schedule = _nominal_uniform_schedule(total_reductions, rounds)
    else:
        schedule = _nominal_front_loaded_schedule(initial_tokens, total_reductions, rounds)

    tokens = initial_tokens
    for index, r in enumerate(schedule):
        cap = _max_r(tokens, protect_cls)
        if r > cap:
            raise ValueError(
                f"ToMe schedule is infeasible: round {index} wants to merge {r} tokens but only "
                f"{cap} of the {tokens} remaining tokens are mergeable. Use more merge layers "
                f"(num_merge_layers), a larger max_tokens, or schedule='front_loaded'."
            )
        tokens -= r
    return schedule


def _bipartite_soft_matching(
    x: torch.Tensor,
    metric: torch.Tensor,
    r: int,
    size: torch.Tensor,
    source: torch.Tensor | None,
    *,
    protect_cls: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """One round of bipartite soft matching (Bolya et al., §3.2).

    Tokens split into the even-indexed half ``A`` (sources) and odd-indexed half
    ``B`` (destinations); each source is matched to its most similar destination
    and the ``r`` strongest edges are merged by size-weighted averaging. With
    ``protect_cls``, token 0 is masked out of the scores and stays at the front.

    Args:
        x: ``[B, T, D]`` features to merge.
        metric: ``[B, T, C]`` similarity signal (head-averaged ``K`` or the
            post-attention hidden state).
        r: number of sources to merge into destinations.
        size: ``[B, T, 1]`` original tokens each token represents.
        source: optional ``[B, T, T_original]`` provenance matrix.

    Returns:
        ``(x, size, source)``, each shortened to ``T - r`` tokens.
    """
    _, _, dim = x.shape

    with torch.no_grad():
        m = metric / metric.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        a = m[..., ::2, :]
        b = m[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        if protect_cls:
            scores[..., 0, :] = -float("inf")  # never merge [CLS] (A row 0)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)

        unm_idx = edge_idx[..., r:, None]
        src_idx = edge_idx[..., :r, None]
        dst_idx = node_idx.gather(dim=-1, index=src_idx.squeeze(-1)).unsqueeze(-1)
        unm_idx, _ = unm_idx.sort(dim=1)  # keeps [CLS] (A row 0) at the front

    original_dtype = x.dtype
    x_f32 = x.to(torch.float32)
    size_f32 = size.to(torch.float32)

    src_x = x_f32[..., ::2, :]
    dst_x = x_f32[..., 1::2, :]
    src_size = size_f32[..., ::2, :]
    dst_size = size_f32[..., 1::2, :]

    src_weighted = (src_x * src_size).gather(dim=1, index=src_idx.expand(-1, -1, dim))
    merged_weighted = (dst_x * dst_size).scatter_add(
        dim=1, index=dst_idx.expand(-1, -1, dim), src=src_weighted
    )
    merged_size = dst_size.scatter_add(
        dim=1, index=dst_idx, src=src_size.gather(dim=1, index=src_idx)
    )

    unm_x = src_x.gather(dim=1, index=unm_idx.expand(-1, -1, dim))
    unm_size = src_size.gather(dim=1, index=unm_idx)

    new_x = torch.cat([unm_x, merged_weighted / merged_size], dim=1).to(original_dtype)
    new_size = torch.cat([unm_size, merged_size], dim=1).to(size.dtype)

    new_source: torch.Tensor | None = None
    if source is not None:
        # Same A/B split and indices as `x`, so each merged destination row
        # accumulates the original-token contributions folded into it.
        t_original = source.shape[-1]
        src_source = source[..., ::2, :]
        dst_source = source[..., 1::2, :]
        merged_source = dst_source.scatter_add(
            dim=1,
            index=dst_idx.expand(-1, -1, t_original),
            src=src_source.gather(dim=1, index=src_idx.expand(-1, -1, t_original)),
        )
        new_source = torch.cat(
            [src_source.gather(dim=1, index=unm_idx.expand(-1, -1, t_original)), merged_source],
            dim=1,
        )

    return new_x, new_size, new_source


class ToMeCompressor(TokenCompressor):
    """Merge redundant visual tokens with a backbone-specific ToMe adapter.

    Parameters
    ----------
    max_tokens:
        LLaVA-1.5/InternVL visual-token budget.
    retain_ratio:
        Fraction retained from each dynamic Qwen2.5-VL or LLaVA-NeXT visual span.
    prop_attention:
        Add ``log(size)`` to the attention logits (proportional attention, §3.3).
        Principled, but occasionally harmful as a training-free drop-in since the
        pretrained weights never saw the bias. This applies to the exact LLaVA-1.5
        in-ViT path; NeXT merges at its crop-packing boundary.
    num_merge_layers:
        Merge only in the first N CLIP layers, capped at the layers whose output
        LLaVA reads. Stopping earlier keeps the later layers' feature
        distribution closer to what the LLM was trained on. Defaults to all of
        them.
    metric_source:
        ``"k_mean"`` (head-averaged ``K``, the paper's choice) or
        ``"hidden_state"`` (post-attention residual, full hidden width and often
        a more discriminative signal).
    schedule:
        ``"uniform"`` or ``"front_loaded"``. Front-loading helps when
        ``num_merge_layers`` is small enough that a uniform ``r`` would exceed the
        feasibility cap.
    track_source:
        Record per-layer ``[T_current, T_original]`` matrices for ``llmoe.viz``.
        Required by ``bm --visualize``; off by default because it materializes a
        ``[B, T, T_original]`` matrix per merge layer.
    vision_feature_layer:
        Override the CLIP layer LLaVA reads. Taken from the model config when
        left as ``None``.
    """

    def __init__(
        self,
        max_tokens: int = 64,
        *,
        retain_ratio: float | None = None,
        prop_attention: bool = True,
        num_merge_layers: int | None = None,
        metric_source: MetricSource = "k_mean",
        schedule: Schedule = "uniform",
        track_source: bool = False,
        vision_feature_layer: int | None = None,
    ) -> None:
        self._configured_max_tokens = int(max_tokens)
        self._configured_prop_attention = bool(prop_attention)
        self._configured_num_merge_layers = num_merge_layers is not None
        self._configured_metric_source = metric_source
        self._configured_schedule = schedule
        self._configured_vision_feature_layer = vision_feature_layer is not None
        super().__init__(int(max_tokens))
        if self._max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if num_merge_layers is not None and int(num_merge_layers) < 1:
            raise ValueError("num_merge_layers must be >= 1")
        if metric_source not in ("k_mean", "hidden_state"):
            raise ValueError(
                f"metric_source must be 'k_mean' or 'hidden_state', got {metric_source!r}"
            )
        if schedule not in ("uniform", "front_loaded"):
            raise ValueError(f"schedule must be 'uniform' or 'front_loaded', got {schedule!r}")
        if retain_ratio is not None and not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")

        self._retain_ratio: float | None = None if retain_ratio is None else float(retain_ratio)
        self._prop_attention: bool = bool(prop_attention)
        self._num_merge_layers: int | None = (
            None if num_merge_layers is None else int(num_merge_layers)
        )
        self._metric_source: MetricSource = metric_source
        self._schedule_type: Schedule = schedule
        self._track_source: bool = bool(track_source)
        self._vision_feature_layer: int | None = (
            None if vision_feature_layer is None else int(vision_feature_layer)
        )

        # Resolved in install(): how many leading CLIP layers may merge.
        self._merge_layer_count: int = 0
        # Cached per input shape, so one compressor handles several resolutions.
        self._schedules: dict[tuple[int, int, int, bool], list[int]] = {}

        # Per-forward state shared between the encoder and layer patches.
        self._size: torch.Tensor | None = None
        self._layer_counter: int = 0
        self._source: torch.Tensor | None = None
        self._pending_history: list[torch.Tensor] = []
        self._active_schedule: list[int] = []

        # Installation bookkeeping, all restored by uninstall().
        self._installed_vlm: VLM | None = None
        self._encoder: Any | None = None
        self._inner_model: Any | None = None
        self._original_encoder_forward: Any | None = None
        self._original_layer_forwards: list[tuple[Any, Any]] = []
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._qwen: Any | None = None
        self._internvl: Any | None = None
        self._llava_next: Any | None = None

    @property
    @override
    def budget(self) -> int | float:
        return self._max_tokens if self._retain_ratio is None else self._retain_ratio

    def validate_llava_next(self) -> None:
        """Validate the single canonical NeXT parameterisation."""

        from llmoe.compressor.llava_next import validate_retain_ratio

        validate_retain_ratio(self._retain_ratio, label="ToMe")
        # `schedule` and `num_merge_layers` are the two knobs the infeasibility
        # error itself recommends, so an explicit choice of either is treated as
        # a deliberate adaptation rather than a rejected parameterisation.
        adapted_schedule = (
            self._configured_schedule != "uniform" or self._configured_num_merge_layers
        )
        noncanonical = (
            self._configured_max_tokens != 64
            or not self._configured_prop_attention
            or self._configured_metric_source != "k_mean"
            or self._configured_vision_feature_layer
        )
        if adapted_schedule:
            # ToMe merges crop-locally and splits the budget across AnyRes crops,
            # so a crop can be allocated a far deeper reduction than the global
            # retain_ratio suggests. Bipartite matching merges at most half the
            # remaining tokens per round, and a uniform schedule asks for a
            # constant r every round, so the tail can demand more merges than
            # remain mergeable. A geometric schedule decays r instead.
            print(f"[ToMe] ADAPTED VARIANT: schedule={self._configured_schedule!r}"
                  f"{', num_merge_layers set' if self._configured_num_merge_layers else ''}"
                  f" (canonical is uniform) so retain_ratio="
                  f"{float(self._retain_ratio):g} is reachable per crop", flush=True)
        if noncanonical:
            raise ValueError(
                "ToMe/LLaVA-NeXT fixes head-averaged K matching, uniform rounds "
                "over the checkpoint-selected CLIP depth, size-weighted averaging, "
                "crop-local grouping, and newline protection; only retain_ratio "
                "is configurable."
            )

    # ----------------------------------------------------------------- compress

    @override
    def compress(
        self,
        features: torch.Tensor,
        *,
        attention: torch.Tensor | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> CompressedFeatures:
        """Apply ToMe to an already-projected ``[B, N, D]`` (or ``[N, D]``) block.

        The post-projector fallback, and the entry point for tensor-level tests.
        With no layer stack to spread the reduction over, the schedule runs over
        ``num_merge_layers`` (default ``DEFAULT_FALLBACK_ROUNDS``) rounds using
        the features themselves as the metric. No ``[CLS]`` is present here.
        """
        del attention, layer_idx, kwargs
        if self._retain_ratio is not None:
            raise RuntimeError(
                "ToMeCompressor.compress() is the fixed-count tensor path. "
                "Ratio-based Qwen2.5-VL and LLaVA-NeXT runs use their installed "
                "architecture adapters."
            )

        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Expected [B, N, D] or [N, D], got {tuple(features.shape)}")
        if features.shape[1] == 0:
            raise ValueError("Cannot compress an empty token sequence")

        batch, num_tokens, _ = features.shape
        identity = torch.eye(num_tokens, device=features.device, dtype=torch.float32)
        if num_tokens <= self._max_tokens:
            if self._track_source:
                self._source_history = [identity.cpu(), identity.cpu()]
            return CompressedFeatures(tokens=features.squeeze(0) if squeeze else features)

        rounds = self._num_merge_layers or DEFAULT_FALLBACK_ROUNDS
        schedule = self._resolve_schedule(
            num_tokens, num_tokens - self._max_tokens, rounds, protect_cls=False
        )

        tokens = features
        size = features.new_ones((batch, num_tokens, 1))
        source = (
            identity.unsqueeze(0).expand(batch, num_tokens, num_tokens).contiguous()
            if self._track_source
            else None
        )
        history: list[torch.Tensor] = [identity.cpu()] if self._track_source else []

        for r in schedule:
            if r < 1:
                continue
            tokens, size, source = _bipartite_soft_matching(
                tokens,
                tokens,
                r,
                size,
                source,
                protect_cls=False,
            )
            if source is not None:
                history.append(source[0].detach().cpu())

        if self._track_source:
            self._source_history = history
        return CompressedFeatures(tokens=tokens.squeeze(0) if squeeze else tokens)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        """Per-layer ``[T_current, T_original]`` snapshots from the last forward.

        Index 0 is the pre-merge identity; later entries follow each merge layer.
        Empty when ``track_source`` is off or no forward has run.
        """
        if self._llava_next is not None:
            return self._llava_next.get_source_history()
        if self._qwen is not None:
            return self._qwen.get_source_history()
        return list(self._source_history)

    def _resolve_schedule(
        self, initial_tokens: int, total_reductions: int, rounds: int, *, protect_cls: bool
    ) -> list[int]:
        key = (initial_tokens, total_reductions, rounds, protect_cls)
        cached = self._schedules.get(key)
        if cached is None:
            cached = _build_schedule(
                initial_tokens,
                total_reductions,
                rounds,
                kind=self._schedule_type,
                protect_cls=protect_cls,
            )
            self._schedules[key] = cached
        return cached

    # ------------------------------------------------------------------ install

    @override
    def install(self, vlm: VLM) -> None:
        from llmoe.compressor.llava_next import is_llava_next

        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return
            raise RuntimeError("ToMeCompressor is already installed on another backbone")

        if is_llava_next(vlm):
            self.validate_llava_next()
            assert self._retain_ratio is not None
            from llmoe.compressor.tome.tome_llava_next import LlavaNextToMeAdapter

            adapter = LlavaNextToMeAdapter(
                self,
                vlm,
                retain_ratio=self._retain_ratio,
                track_source=self._track_source,
            )
            adapter.install()
            self._installed_vlm = vlm
            self._llava_next = adapter
            return

        inner = getattr(getattr(vlm, "model", None), "model", None)
        if getattr(inner, "visual", None) is not None:
            if self._retain_ratio is None:
                raise ValueError(
                    "ToMe on Qwen2.5-VL requires retain_ratio: NaViT emits "
                    "a different visual-token count for each image."
                )
            from llmoe.compressor.tome.tome_qwen import QwenToMeAdapter
            from llmoe.utils.qwen import resolve_image_token_id

            adapter = QwenToMeAdapter(
                compressor=self,
                retain_ratio=self._retain_ratio,
                track_source=self._track_source,
            )
            adapter.install(
                vlm,
                image_token_id=resolve_image_token_id(
                    vlm.model.config,
                    vlm.processor,
                    label="ToMe",
                ),
            )
            self._installed_vlm = vlm
            self._qwen = adapter
            return
        if self._retain_ratio is not None:
            raise ValueError(
                "retain_ratio is ToMe's Qwen2.5-VL parameterisation; "
                "LLaVA-1.5 and InternVL use max_tokens."
            )

        from llmoe.compressor.tome.tome_internvl import InternVLToMeAdapter

        if InternVLToMeAdapter.supports(vlm):
            adapter = InternVLToMeAdapter(self)
            adapter.install(vlm)
            self._installed_vlm = vlm
            self._internvl = adapter
            print(
                f"[ToMe/InternVL] install: max_tokens={self._max_tokens}, "
                "concatenate tiles per source",
                flush=True,
            )
            return

        self._installed_vlm = vlm
        encoder = self._locate_clip_encoder(vlm)
        if encoder is None:
            # No CLIP-style vision tower: compress the projector output instead.
            self._wrap_get_image_features(vlm)
        else:
            self._merge_layer_count = self._resolve_merge_layer_count(vlm, len(encoder.layers))
            self._patch_encoder(encoder)
        self._install_placeholder_trim(vlm)

    def uninstall(self) -> None:
        if self._installed_vlm is None:
            return
        if self._llava_next is not None:
            self._llava_next.uninstall()
            self._llava_next = None
            self._installed_vlm = None
            return
        if self._qwen is not None:
            self._qwen.uninstall()
            self._qwen = None
            self._installed_vlm = None
            return
        if self._internvl is not None:
            self._internvl.uninstall()
            self._internvl = None
            self._installed_vlm = None
            return
        if self._encoder is not None and self._original_encoder_forward is not None:
            self._encoder.forward = self._original_encoder_forward
        for layer, original_forward in self._original_layer_forwards:
            layer.forward = original_forward
        if self._inner_model is not None and self._original_get_image_features is not None:
            self._inner_model.get_image_features = self._original_get_image_features
        if self._original_prepare_inputs is not None:
            self._installed_vlm.prepare_inputs = self._original_prepare_inputs
        self._original_layer_forwards = []
        self._original_encoder_forward = None
        self._original_get_image_features = None
        self._original_prepare_inputs = None
        self._encoder = None
        self._inner_model = None
        self._installed_vlm = None
        self._size = None
        self._source = None
        self._pending_history = []
        self._active_schedule = []
        self._layer_counter = 0

    def _wrap_get_image_features(self, vlm: VLM) -> None:
        """Post-projector fallback wrapper.

        The inherited ``wrap_get_image_features`` keeps no handle on the original,
        so it cannot be undone; this one can.
        """
        inner = getattr(getattr(vlm, "model", None), "model", None)
        original_get_image_features = getattr(inner, "get_image_features", None)
        if not callable(original_get_image_features):
            raise TypeError(
                "Backbone exposes neither a CLIP vision encoder nor model.model.get_image_features"
            )

        self._inner_model = inner
        self._original_get_image_features = original_get_image_features

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            image_outputs = original_get_image_features(*args, **kwargs)
            features = torch.stack(list(image_outputs.pooler_output), dim=0)
            image_outputs.pooler_output = list(self.compress(features).tokens.unbind(0))
            return image_outputs

        inner.get_image_features = wrapped_get_image_features

    @staticmethod
    def _locate_clip_encoder(vlm: VLM) -> Any | None:
        """Return the CLIP vision encoder, or ``None`` for other backbones.

        transformers 5.x hangs the encoder directly off ``CLIPVisionModel``;
        earlier layouts nest it under ``vision_model``.
        """
        vision_tower = getattr(
            getattr(getattr(vlm, "model", None), "model", None), "vision_tower", None
        )
        if vision_tower is None:
            return None
        vision_model = getattr(vision_tower, "vision_model", vision_tower)
        encoder = getattr(vision_model, "encoder", None)
        if encoder is None or not getattr(encoder, "layers", None):
            return None
        return encoder

    def _resolve_merge_layer_count(self, vlm: VLM, num_clip_layers: int) -> int:
        """How many leading CLIP layers may merge.

        ``capture_outputs`` builds ``hidden_states`` as ``[input_to_layer0,
        output_of_layer0, ...]``, so ``vision_feature_layer = -k`` selects the
        output of layer ``N - k`` and reflects the merges of layers ``0..N-k``.
        With several selected layers, merging must finish before the shallowest,
        or the concatenated hidden states would have mismatched token counts.
        """
        selected = self._vision_feature_layer
        if selected is None:
            config = getattr(getattr(vlm, "model", None), "config", None)
            selected = getattr(config, "vision_feature_layer", None)
            if selected is None:
                selected = getattr(config, "vision_feature_select_layer", -2)

        layers = [selected] if isinstance(selected, int) else list(selected)
        resolved = [layer - 1 if layer >= 0 else num_clip_layers + layer for layer in layers]
        for layer, index in zip(layers, resolved):
            if not 0 <= index < num_clip_layers:
                raise ValueError(
                    f"vision_feature_layer={layer} resolves to out-of-range layer index {index} "
                    f"for a {num_clip_layers}-layer vision tower"
                )
        max_merge_layers = min(resolved) + 1

        if self._num_merge_layers is None:
            return max_merge_layers
        if self._num_merge_layers > max_merge_layers:
            raise ValueError(
                f"num_merge_layers must be <= {max_merge_layers} (the layers whose output feeds "
                f"the projector), got {self._num_merge_layers}"
            )
        return self._num_merge_layers

    def _patch_encoder(self, encoder: Any) -> None:
        """Bracket the encoder forward with per-batch state, then patch each layer.

        The wrapper merges nothing itself; it seeds the running ``size`` (and
        ``source``), the schedule, and the layer counter the layer patches use.
        """
        self._encoder = encoder
        self._original_encoder_forward = encoder.forward
        original_encoder_forward = encoder.forward

        def patched_encoder_forward(*args: Any, **kwargs: Any) -> Any:
            # HF calls this as `encoder(inputs_embeds=...)`, but recover the
            # tensor either way rather than assuming the caller's convention.
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None and args:
                inputs_embeds = args[0]
            if inputs_embeds is None:
                raise RuntimeError("ToMeCompressor expected inputs_embeds to the CLIP encoder")

            batch, num_tokens, _ = inputs_embeds.shape
            # Every token starts as one original patch; merges accumulate here and
            # the attention patch reads it back as the log-size bias.
            self._size = torch.ones(
                batch, num_tokens, 1, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            # Only place the full token stream is visible, so the schedule is
            # resolved here. `[CLS]` is outside the budget, hence the -1.
            self._active_schedule = self._resolve_schedule(
                num_tokens,
                max(num_tokens - 1 - self._max_tokens, 0),
                self._merge_layer_count,
                protect_cls=True,
            )
            # Reset per batch so layer i reads schedule[i].
            self._layer_counter = 0

            if self._track_source:
                identity = torch.eye(num_tokens, device=inputs_embeds.device, dtype=torch.float32)
                self._source = (
                    identity.unsqueeze(0).expand(batch, num_tokens, num_tokens).contiguous()
                )
                self._pending_history = [identity.cpu()]
            else:
                self._source = None
                self._pending_history = []

            try:
                return original_encoder_forward(*args, **kwargs)
            finally:
                # Publish the history before clearing, so `llmoe.viz` can read it
                # after the forward returns.
                if self._track_source:
                    self._source_history = self._pending_history
                self._size = None
                self._source = None
                self._pending_history = []
                self._active_schedule = []
                self._layer_counter = 0

        encoder.forward = patched_encoder_forward
        for layer in encoder.layers:
            self._patch_encoder_layer(layer)

    def _patch_encoder_layer(self, layer: Any) -> None:
        """Swap one ``CLIPEncoderLayer.forward`` for the ToMe-aware version.

        Mirrors the stock forward (pre-norm -> attention -> residual -> pre-norm
        -> MLP -> residual) with a merge step between the attention residual and
        the MLP.
        """
        self._original_layer_forwards.append((layer, layer.forward))

        def patched_layer_forward(
            hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: Any
        ) -> torch.Tensor:
            layer_idx = self._layer_counter
            self._layer_counter += 1

            # Empty unless the encoder wrapper seeded it, so a forward bypassing
            # that wrapper runs the stock layer untouched.
            schedule = self._active_schedule
            will_merge = (
                self._size is not None and layer_idx < len(schedule) and schedule[layer_idx] > 0
            )
            need_k_mean = will_merge and self._metric_source == "k_mean"

            residual = hidden_states
            normed = layer.layer_norm1(hidden_states)
            attention_output, k_mean = self._attention_forward(
                layer.self_attn,
                normed,
                attention_mask,
                need_k_mean=need_k_mean,
                **kwargs,
            )
            hidden_states = residual + attention_output

            if will_merge:
                metric = k_mean if self._metric_source == "k_mean" else hidden_states
                hidden_states, self._size, self._source = _bipartite_soft_matching(
                    hidden_states,
                    metric,
                    schedule[layer_idx],
                    self._size,
                    self._source,
                    protect_cls=True,
                )
                if self._source is not None:
                    # Detach to CPU to keep VRAM flat across the encoder forward.
                    self._pending_history.append(self._source[0].detach().cpu())

            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.layer_norm2(hidden_states))
            return hidden_states

        layer.forward = patched_layer_forward

    def _attention_forward(
        self,
        self_attn: Any,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        need_k_mean: bool,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run CLIP self-attention, optionally returning head-averaged ``K``.

        Needing neither ``k_mean`` nor the log-size bias, the module's own forward
        runs so the configured backend (SDPA) is preserved. Otherwise this mirrors
        ``modeling_clip.eager_attention_forward`` (fp32 softmax included), which
        matches the stock path when ``size`` is uniform.
        """
        needs_bias = self._prop_attention and self._size is not None
        if not need_k_mean and not needs_bias:
            attention_output, _ = self_attn(
                hidden_states=hidden_states, attention_mask=attention_mask, **kwargs
            )
            return attention_output, None

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self_attn.head_dim)

        queries = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        keys = self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        values = self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self_attn.scale
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        if needs_bias:
            # [B, T, 1] -> additive log-size bias per key column, broadcast over
            # heads and query rows: [B, 1, 1, T].
            log_size = self._size.to(attention_weights.dtype).clamp(min=1.0).log()
            attention_weights = attention_weights + log_size.squeeze(-1)[:, None, None, :]

        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(
            queries.dtype
        )
        attention_output = torch.matmul(attention_weights, values).transpose(1, 2).contiguous()
        attention_output = self_attn.out_proj(attention_output.reshape(*input_shape, -1))

        return attention_output, keys.mean(dim=1) if need_k_mean else None

    # The vision stage now emits `max_tokens` features per image, but the LLaVA
    # processor still expands <image> into one placeholder per patch, and
    # transformers 5.x scatters image features with a strict count check.

    def _install_placeholder_trim(self, vlm: VLM) -> None:
        image_token_id = self._image_token_id(vlm)
        if image_token_id is None:
            return
        tokenizer = getattr(getattr(vlm, "processor", None), "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        pad_token_id = image_token_id if pad_token_id is None else int(pad_token_id)

        self._original_prepare_inputs = vlm.prepare_inputs
        original_prepare_inputs = vlm.prepare_inputs

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            inputs = original_prepare_inputs(*args, **kwargs)
            return self._trim_placeholders(inputs, image_token_id, pad_token_id)

        vlm.prepare_inputs = wrapped_prepare_inputs

    @staticmethod
    def _image_token_id(vlm: VLM) -> int | None:
        config = getattr(getattr(vlm, "model", None), "config", None)
        for attribute in ("image_token_id", "image_token_index"):
            value = getattr(config, attribute, None)
            if value is not None:
                return int(value)
        return None

    def _trim_placeholders(self, inputs: Any, image_token_id: int, pad_token_id: int) -> Any:
        """Shorten every contiguous image-placeholder run to ``max_tokens``."""
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return inputs

        attention_mask = inputs.get("attention_mask")
        has_attention = (
            isinstance(attention_mask, torch.Tensor) and attention_mask.shape == input_ids.shape
        )
        kept_ids: list[torch.Tensor] = []
        kept_attention: list[torch.Tensor] = []
        changed = False

        for row_index, row in enumerate(input_ids):
            keep = torch.ones(row.shape[0], dtype=torch.bool, device=row.device)
            positions = (row == image_token_id).nonzero(as_tuple=False).flatten()
            if positions.numel():
                # Split non-consecutive positions into one run per image.
                split_points = (positions[1:] != positions[:-1] + 1).nonzero(
                    as_tuple=False
                ).flatten() + 1
                for run in torch.tensor_split(positions, split_points.tolist()):
                    if run.numel() > self._max_tokens:
                        keep[run[self._max_tokens :]] = False
                        changed = True
            kept_ids.append(row[keep])
            if has_attention:
                kept_attention.append(attention_mask[row_index][keep])

        if not changed:
            return inputs

        max_length = max(row.numel() for row in kept_ids)
        new_ids = input_ids.new_full((len(kept_ids), max_length), pad_token_id)
        new_attention = (
            attention_mask.new_zeros((len(kept_ids), max_length)) if has_attention else None
        )
        for row_index, row in enumerate(kept_ids):
            length = row.numel()
            new_ids[row_index, max_length - length :] = row
            if new_attention is not None:
                new_attention[row_index, max_length - length :] = kept_attention[row_index]
        inputs["input_ids"] = new_ids
        if new_attention is not None:
            inputs["attention_mask"] = new_attention
        return inputs
