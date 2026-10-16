"""Nuwa on InternVL -- the LLaVA-NeXT release, retargeted at InternVL's tiling.

Reference code: ``CLIPVisionTower_Nuwa_Next.forward`` and ``prune_vision_tokens``
in ``nuwa/clip_encoder.py``, plus ``prepare_inputs_labels_for_multimodal_nuwa``
(``nuwa/llava_arch.py``) and the shared ``modeling_llama.py``, from
github.com/Man-PaperRejected/Nuwa. Published configuration:
``exp_bash/llava-next.sh`` -- ``NT`` in ``{160, 320, 640}``, ``DIST=280``.

Owned by :class:`~llmoe.compressor.nuwa.nuwa.NuwaCompressor`, which detects the
backbone at ``install()`` and delegates here.

Both backbones tile the same way -- cut the image into a grid of tiles, encode each
on its own, encode one downscaled view alongside -- and Nuwa's LLaVA-NeXT release
handles that by stitching the per-tile grids back into one mosaic before
partitioning it into 2x2 regions, so a region can straddle a tile boundary and the
distance penalty measures distance in the picture rather than within a tile. The
global view is pruned as a grid of its own, and each group's budget is
``Stage1 // num_tiles`` per tile.

Differences from LLaVA-NeXT, all of them simplifications:

* the thumbnail is **last**, not first, and the tile grid comes from
  ``get_optimal_tiled_canvas`` rather than ``anyres`` pinpoints;
* InternVL resizes the image *to* the canvas instead of letterboxing it, so
  upstream's ``unpad_image`` / ``unpad_size`` bookkeeping has nothing to do;
* the prompt is one flat ``<IMG_CONTEXT>`` run, so
  ``insert_newlines_after_pruning`` has no separators to maintain;
* survivors stay in InternVL's own tile-major slots (LLaVA-NeXT emits mosaic
  raster order because that *is* its order); the mosaic is used for selection
  only, and ``tile_stitch_index`` maps back;
* RPME needs no ``rotary_emb`` hook: cutting the sequence in the language model's
  pre-hook and handing it the survivors' original absolute columns is the same
  rewrite upstream does by hand.

The one new step is the encoder's token merge -- InternVL pixel-shuffles each
tile's 32x32 patch grid into 16x16 LLM tokens -- so the salience is mean-pooled
over each 2x2 patch block and the similarity features are pixel-shuffled alongside
the aggregation features, as the Qwen2.5-VL release does for its patch merger.

``max_tokens`` is the decoder-layer-averaged count, as on LLaVA-1.5: ``NT=160``
puts 290 tokens through layers 0-15 and ``int(290 * 0.1) = 29`` after that. Since
the tile count varies, the realised Stage-1 count is ``(Stage1 // tiles) * tiles``
-- upstream divides the same way.

Harness adaptations: upstream rebuilds the prompt embeddings by hand, which the
strict placeholder/feature count check makes impossible, so Stage 1 writes the
merged features back into the survivors' own slots and the sequence is cut in a
``language_model`` pre-hook. ``batch_size=1`` only; video, multi-image prompts,
beam search and ``use_cache=False`` raise.
"""

import math
from typing import Any, Sequence

import torch
from torch.utils.hooks import RemovableHandle

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    advance_decode_step,
    apply_prune_state,
    gather_position_ids,
    keep_columns,
    pruned_attention_mask,
    select_rope_tables,
)
from llmoe.compressor.nuwa.policy import (
    average_hidden_states,
    nuwa_stage1,
    pool_grid_blocks,
    preserve_count,
    register_capture_attention,
    resolve_region_budget,
    select_by_text_attention,
    set_attn_implementation,
    tile_stitch_index,
    unregister_capture_attention,
)

# Published LLaVA-NeXT budgets (`exp_bash/llava-next.sh`, `Stage1_CONFIGS` and
# `TOKEN_CONFIGS` of `CLIPVisionTower_Nuwa_Next`). The Stage-1 counts satisfy
# `(S + int(S * 0.1)) / 2 == budget`, i.e. the same layer-weighted average the
# LLaVA-1.5 budgets use, and the fallback below solves that for any other budget.
NEXT_STAGE1_CONFIGS: dict[int, int] = {64: 112, 128: 224, 160: 290, 192: 336, 320: 582, 640: 1162}
NEXT_TOPN_CONFIGS: dict[int, int] = {64: 1, 128: 2, 160: 2, 192: 3, 320: 3, 640: 3}
# `llava-next.sh` exports THRE=0.5, not the 0.55 hardcoded in the LLaVA-1.5 tower.
NEXT_PILLAR_PERCENTILE: float = 0.5


def next_stage1_count(max_tokens: int) -> int:
    """Stage-1 budget for a tiled backbone: the published table, else the formula."""
    if max_tokens in NEXT_STAGE1_CONFIGS:
        return NEXT_STAGE1_CONFIGS[max_tokens]
    return max(1, round(2 * max_tokens / 1.1))


def next_top_n(max_tokens: int) -> int:
    """Candidates per 2x2 region: the published table, else the largest published."""
    if max_tokens in NEXT_TOPN_CONFIGS:
        return NEXT_TOPN_CONFIGS[max_tokens]
    return 3


class InternVLNuwaAdapter:
    """Nuwa's tiled-backbone half, driving InternVL's dynamic tiles.

    :param stage1_count: total Stage-1 budget, split evenly over the tiles.
    :param top_n_per_region: candidates per ``2 x 2`` region, before
        :func:`resolve_region_budget` raises it for a large mosaic.
    :param distance: squared spatial cutoff for the aggregation penalty, 280
        upstream.
    :param pillar_threshold_percentile: Pillar/Collector split, 0.5 in
        ``llava-next.sh``.
    :param stage2_layer: first decoder layer to run on the reduced set; its
        predecessor supplies the ranking attention.
    :param vision_feature_layer: encoder hidden state used as the merged features;
        taken from the model config when ``None`` (-1 on InternVL).
    :param sim_layers: ``[start, end)`` encoder hidden states averaged into the
        similarity features, clamped to the tower's depth.
    :param track_source: record ``[T_current, T_original]`` matrices for
        ``llmoe.viz``.
    """

    def __init__(self, *, stage1_count: int, top_n_per_region: int, distance: float,
                 pillar_threshold_percentile: float, stage2_layer: int,
                 vision_feature_layer: int | None, sim_layers: Sequence[int],
                 track_source: bool) -> None:
        self._stage1_count: int = int(stage1_count)
        self._top_n_per_region: int = int(top_n_per_region)
        self._distance_threshold: float = math.sqrt(float(distance))
        self._pillar_threshold_percentile: float = float(pillar_threshold_percentile)
        self._stage2_layer: int = int(stage2_layer)
        self._vision_feature_layer: int | None = (
            None if vision_feature_layer is None else int(vision_feature_layer))
        self._sim_layers: tuple[int, int] = (int(sim_layers[0]), int(sim_layers[1]))
        self._track_source: bool = bool(track_source)
        if self._stage1_count < 1:
            raise ValueError(f"stage1_count must be >= 1, got {stage1_count}")

        # Installation bookkeeping, all undone by uninstall().
        self._handles: list[RemovableHandle] = []
        self._installed_vlm: VLM | None = None
        self._inner: Any | None = None
        self._language_model: Any | None = None
        self._image_token_id: int | None = None
        self._select_strategy: str = "default"
        self._downsample_ratio: float = 0.5
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._attention_name: str | None = None
        self._previous_llm_attn: str | None = None
        self._previous_vision_attn: str | None = None
        self._vision_config: Any | None = None
        self._tile_size: tuple[int, int] = (448, 448)
        self._min_patches: int = 1
        self._max_patches: int = 12

        # Per-forward state.
        self._image_sizes: list[tuple[int, int]] = []
        self._input_ids: torch.Tensor | None = None
        self._keep_visual: torch.Tensor | None = None  # [num_visual] bool
        self._image_start: int = 0
        self._visual_count: int = 0
        self._vision_positions: torch.Tensor | None = None  # [1, V] current columns
        self._stage2_attention: torch.Tensor | None = None
        self._stage1_state: dict[str, Any] | None = None
        self._prune_state: dict[str, Any] = {}
        self._composed_source: torch.Tensor | None = None
        self._source_history: list[torch.Tensor] = []
        self._forward_count: int = 0
        self._warned: set[str] = set()

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(f"[Nuwa/InternVL] WARNING: {message}", flush=True)

    # ------------------------------------------------------------------ install

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        inner = vlm.model.model
        vision_tower = getattr(inner, "vision_tower", None)
        language_model = getattr(inner, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if vision_tower is None or getattr(inner, "multi_modal_projector", None) is None \
                or not layers:
            raise TypeError(
                "Nuwa's InternVL path needs `model.model.vision_tower`, "
                "`model.model.multi_modal_projector` and `model.model.language_model.layers`."
            )
        if not 1 <= self._stage2_layer < len(layers):
            raise ValueError(
                f"stage2_layer={self._stage2_layer} does not fit a {len(layers)}-layer decoder")

        config = vlm.model.config
        self._installed_vlm = vlm
        self._inner = inner
        self._language_model = language_model
        self._image_token_id = int(image_token_id)
        self._downsample_ratio = float(getattr(config, "downsample_ratio", 0.5))
        self._select_strategy = str(getattr(config, "vision_feature_select_strategy", "default"))
        if self._vision_feature_layer is None:
            layer = getattr(config, "vision_feature_layer", -1)
            if not isinstance(layer, int):
                raise TypeError(
                    f"Nuwa needs a single vision_feature_layer, but the config holds {layer!r}. "
                    f"Pass `vision_feature_layer` explicitly."
                )
            self._vision_feature_layer = int(layer)
        self._read_tiling_config(vlm)

        self._force_eager_vision(vision_tower)
        # `input_ids` are only visible on the outer model's own call -- it embeds
        # them and scatters the image features in before the language model runs --
        # and every prefill has to invalidate the previous sample's decisions.
        self._handles.append(
            inner.register_forward_pre_hook(self._capture_model_inputs, with_kwargs=True))
        self._handles.append(
            vision_tower.register_forward_pre_hook(self._vision_capture_hook, with_kwargs=True))
        self._wrap_get_image_features(inner)
        self._wrap_prepare_inputs(vlm)
        self._handles.append(
            language_model.register_forward_pre_hook(self._language_model_pre_hook,
                                                     with_kwargs=True))
        self._attention_name, self._previous_llm_attn = register_capture_attention(
            language_model, self._stage2_layer - 1, self._on_stage2_attention,
            owner=self, label="nuwa_internvl")
        for index in range(self._stage2_layer, len(layers)):
            self._handles.append(
                layers[index].register_forward_pre_hook(self._make_layer_hook(index),
                                                        with_kwargs=True))

        print(f"[Nuwa/InternVL] install: stage1={self._stage1_count} (split over the tiles), "
              f"top_n_per_region={self._top_n_per_region}, pillars>=p"
              f"{self._pillar_threshold_percentile:g}, feature layer "
              f"{self._vision_feature_layer}, sim layers {list(self._sim_layers)}, "
              f"prune @layer{self._stage2_layer} of {len(layers)} "
              f"(score @layer{self._stage2_layer - 1}), tiles {self._min_patches}-"
              f"{self._max_patches} of {self._tile_size[0]}x{self._tile_size[1]}", flush=True)

    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self._inner is not None and self._original_get_image_features is not None:
            self._inner.get_image_features = self._original_get_image_features
        if self._installed_vlm is not None and self._original_prepare_inputs is not None:
            self._installed_vlm.prepare_inputs = self._original_prepare_inputs
        if self._language_model is not None and self._previous_llm_attn is not None:
            set_attn_implementation(self._language_model, self._previous_llm_attn)
        if self._vision_config is not None and self._previous_vision_attn is not None:
            self._vision_config._attn_implementation = self._previous_vision_attn
        unregister_capture_attention(self._attention_name)
        self._attention_name = None
        self._previous_llm_attn = None
        self._previous_vision_attn = None
        self._vision_config = None
        self._original_get_image_features = None
        self._original_prepare_inputs = None
        self._installed_vlm = None
        self._inner = None
        self._language_model = None
        self._image_token_id = None
        self._reset_forward_state()
        self._source_history = []

    def _capture_model_inputs(self, _module: Any, args: tuple, kwargs: dict) -> None:
        """Record the prompt ids, and drop the previous sample's state at a prefill."""
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 \
                or input_ids.shape[1] <= 1:
            return None  # a decode step: the prefill's capture and state stay valid
        # Clear first, so a text-only prompt (or a tower that somehow does not run)
        # cannot be pruned with the last sample's keep-set.
        self._reset_forward_state()
        self._forward_count += 1
        self._input_ids = input_ids
        return None

    def _reset_forward_state(self) -> None:
        self._input_ids = None
        self._keep_visual = None
        self._image_start = 0
        self._visual_count = 0
        self._vision_positions = None
        self._stage2_attention = None
        self._stage1_state = None
        self._prune_state = {}
        self._composed_source = None

    def get_source_history(self) -> list[torch.Tensor]:
        return list(self._source_history)

    def _read_tiling_config(self, vlm: VLM) -> None:
        """Copy the image processor's own tiling parameters, to rebuild its grid."""
        image_processor = getattr(vlm.processor, "image_processor", None)
        if image_processor is None:
            raise TypeError("Nuwa's InternVL path needs the processor's `image_processor`.")
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            height, width = int(size["height"]), int(size["width"])
        else:
            height, width = int(size.height), int(size.width)
        self._tile_size = (height, width)
        self._min_patches = int(getattr(image_processor, "min_patches", 1))
        self._max_patches = int(getattr(image_processor, "max_patches", 12))

    def _force_eager_vision(self, vision_tower: Any) -> None:
        """Make the encoder run eager attention -- SDPA returns no weights to capture."""
        config = getattr(vision_tower, "config", None)
        config = getattr(config, "vision_config", None) or config
        if config is None:
            raise TypeError("Nuwa needs a config on the vision tower to force eager attention")
        self._vision_config = config
        self._previous_vision_attn = getattr(config, "_attn_implementation", None)
        config._attn_implementation = "eager"

    @staticmethod
    def _vision_capture_hook(_module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        """Ask the encoder for attentions; InternVL requests neither by default."""
        kwargs["output_attentions"] = True
        kwargs["output_hidden_states"] = True
        return args, kwargs

    def _wrap_prepare_inputs(self, vlm: VLM) -> None:
        """Record each prompt's image sizes: the tile grid cannot be recovered later.

        The number of tiles is visible in ``pixel_values``, but not how they were
        laid out -- and ``2 x 4`` and ``4 x 2`` are different pictures. The image
        processor picks the layout from the *original* size with
        ``get_optimal_tiled_canvas``, so that size is captured here and the same
        function is re-run in :meth:`_resolve_tile_grid`.
        """
        original = vlm.prepare_inputs
        self._original_prepare_inputs = original

        def wrapped_prepare_inputs(images: Any, *args: Any, **kwargs: Any) -> Any:
            self._image_sizes = self._measure_images(images)
            return original(images, *args, **kwargs)

        vlm.prepare_inputs = wrapped_prepare_inputs

    @staticmethod
    def _measure_images(images: Any) -> list[tuple[int, int]]:
        """``(height, width)`` per image, for tensors, arrays or PIL images."""
        if isinstance(images, torch.Tensor):
            if images.ndim == 4:
                return [(int(images.shape[-2]), int(images.shape[-1]))] * images.shape[0]
            images = [images]
        sizes: list[tuple[int, int]] = []
        for image in images:
            shape = getattr(image, "shape", None)
            if shape is not None and len(shape) >= 2:
                sizes.append((int(shape[-2]), int(shape[-1])))
                continue
            size = getattr(image, "size", None)  # PIL: (width, height)
            if isinstance(size, tuple) and len(size) == 2:
                sizes.append((int(size[1]), int(size[0])))
                continue
            raise TypeError(f"Nuwa cannot measure an image of type {type(image).__name__}")
        return sizes

    # ------------------------------------------------------------------- Stage 1

    def _wrap_get_image_features(self, inner: Any) -> None:
        """Replace ``get_image_features`` with the tiled Stage-1 merge."""
        original = inner.get_image_features
        self._original_get_image_features = original

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            image_outputs = original(*args, **kwargs)
            hidden_states = getattr(image_outputs, "hidden_states", None)
            attentions = getattr(image_outputs, "attentions", None)
            if not hidden_states or not attentions:
                self._warn_once(
                    "no_vision_outputs",
                    "the vision tower returned no attentions/hidden states; Stage 1 is SKIPPED "
                    "and the full token set reaches the decoder.",
                )
                return image_outputs
            return self._run_stage1(inner, image_outputs, hidden_states, attentions)

        inner.get_image_features = wrapped_get_image_features

    def _run_stage1(self, inner: Any, image_outputs: Any, hidden_states: Sequence[torch.Tensor],
                    attentions: Sequence[torch.Tensor]) -> Any:
        features = image_outputs.pooler_output  # [tiles, tokens, D], already projected
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise RuntimeError(
                f"Nuwa expected [tiles, tokens, D] image features, got "
                f"{type(features).__name__}")
        tiles, tokens_per_tile, _ = features.shape

        # `get_image_features` reads `last_hidden_state` rather than
        # `hidden_states[-1]` for the default layer, and the two differ whenever the
        # tower's final layernorm is not an identity, so follow the model.
        feature_hs = (image_outputs.last_hidden_state if self._vision_feature_layer == -1
                      else hidden_states[self._vision_feature_layer])
        sim_hs = average_hidden_states(hidden_states, *self._sim_layers)
        cls_attention = attentions[self._vision_feature_layer][:, :, 0, 1:].sum(dim=1)
        if self._select_strategy == "default":
            feature_hs = feature_hs[:, 1:, :]
            sim_hs = sim_hs[:, 1:, :]
        num_patches = feature_hs.shape[1]
        side = int(round(math.sqrt(num_patches)))
        if side * side != num_patches:
            raise RuntimeError(
                f"Nuwa needs a square encoder grid; the tower produced {num_patches} patches.")

        # Fold the encoder patches onto the tokens the decoder sees: pixel shuffle
        # for the features (the model's own method, so the projector still fits),
        # a 2x2 mean for the salience.
        factor = int(round(1.0 / self._downsample_ratio))
        patch_features = self._pixel_shuffle(inner, feature_hs, side)
        sim_features = self._pixel_shuffle(inner, sim_hs, side)
        metric = pool_grid_blocks(cls_attention.float(), side, side, factor)
        tile_side = side // factor
        if patch_features.shape[1] != tokens_per_tile or tile_side * tile_side != tokens_per_tile:
            raise RuntimeError(
                f"Nuwa folded the encoder grid to {patch_features.shape[1]} tokens per tile but "
                f"the model emits {tokens_per_tile}."
            )

        rows, cols, has_thumbnail = self._resolve_tile_grid(tiles)
        per_tile = max(1, self._stage1_count // tiles)
        groups = self._tile_groups(tiles, rows, cols, tile_side, has_thumbnail, per_tile,
                                   features.device)

        keep_visual = torch.zeros(tiles * tokens_per_tile, dtype=torch.bool, device=features.device)
        merged_rows: list[torch.Tensor] = []
        merged_positions: list[torch.Tensor] = []
        sources: list[tuple[torch.Tensor, torch.Tensor]] = []
        flat_patches = patch_features.reshape(1, tiles * tokens_per_tile, -1)
        flat_sim = sim_features.reshape(1, tiles * tokens_per_tile, -1)
        flat_metric = metric.reshape(1, tiles * tokens_per_tile)

        for order, grid_h, grid_w, budget in groups:
            merged, positions, weights = self._prune_group(
                flat_patches, flat_sim, flat_metric, order, grid_h, grid_w, budget)
            keep_visual[positions] = True
            merged_rows.append(merged)
            merged_positions.append(positions)
            if self._track_source:
                sources.append((order, weights))

        sequence_order = torch.argsort(torch.cat(merged_positions))
        merged_all = torch.cat(merged_rows, dim=0)[sequence_order]
        kept_positions = torch.cat(merged_positions)[sequence_order]
        projected = inner.multi_modal_projector(merged_all.unsqueeze(0)).squeeze(0)

        # The merged features go back into the survivors' own slots, so the strict
        # placeholder/feature count check still passes; the language-model pre-hook
        # then drops everything `keep_visual` rejects.
        updated = features.reshape(tiles * tokens_per_tile, -1).clone()
        updated[kept_positions] = projected.to(updated.dtype)
        image_outputs.pooler_output = updated.view(tiles, tokens_per_tile, -1)
        self._keep_visual = keep_visual
        if self._track_source:
            self._record_stage1_source(tiles * tokens_per_tile, sources, sequence_order)
        if self._forward_count == 1:
            print(f"[Nuwa/InternVL] stage 1: {tiles} tiles ({rows}x{cols}"
                  f"{' + thumbnail' if has_thumbnail else ''}) of {tile_side}x{tile_side} tokens, "
                  f"visual {tiles * tokens_per_tile} -> {int(keep_visual.sum())} "
                  f"({per_tile} per tile)", flush=True)
        return image_outputs

    @staticmethod
    def _pixel_shuffle(inner: Any, hidden_states: torch.Tensor, side: int) -> torch.Tensor:
        """The model's own pixel shuffle, so the merged tokens still fit the projector."""
        tiles, _, channels = hidden_states.shape
        shuffled = inner.pixel_shuffle(
            hidden_states.reshape(tiles, side, side, channels),
            scale_factor=inner.config.downsample_ratio,
        )
        return shuffled.reshape(tiles, -1, shuffled.shape[-1])

    def _tile_groups(self, tiles: int, rows: int, cols: int, tile_side: int,
                     has_thumbnail: bool, per_tile: int,
                     device: torch.device) -> list[tuple[torch.Tensor, int, int, int]]:
        """The token groups Stage 1 prunes independently, as upstream splits them.

        :return: ``(order, grid_h, grid_w, budget)`` per group, where ``order`` maps
            mosaic raster position to the token's position in the visual span.
        """
        tokens_per_tile = tile_side * tile_side
        local_tiles = tiles - 1 if has_thumbnail else tiles
        groups: list[tuple[torch.Tensor, int, int, int]] = []
        if local_tiles > 0:
            order = tile_stitch_index(rows, cols, tile_side, tile_side, device)
            groups.append((order, rows * tile_side, cols * tile_side, per_tile * local_tiles))
        if has_thumbnail:
            # The thumbnail is the last tile and a plain grid of its own -- upstream's
            # `global_tokens`, which gets one tile's share of the budget.
            offset = (tiles - 1) * tokens_per_tile
            order = offset + torch.arange(tokens_per_tile, device=device)
            groups.append((order, tile_side, tile_side, per_tile))
        return groups

    def _prune_group(self, patches: torch.Tensor, sim: torch.Tensor, metric: torch.Tensor,
                     order: torch.Tensor, grid_h: int, grid_w: int,
                     budget: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Stage 1 over one mosaic and map the survivors back to the span.

        :return: ``(merged [K, D], positions [K] into the visual span, weights
            [K, N] over the group's own tokens)``.
        """
        top_n, budget = resolve_region_budget(budget, self._top_n_per_region, grid_h, grid_w)
        merged, benchmark, weights = nuwa_stage1(
            patch_features=patches[:, order],
            sim_features=sim[:, order],
            cls_attention=metric[:, order],
            grid_h=grid_h,
            grid_w=grid_w,
            num_tokens=budget,
            top_n_per_region=top_n,
            distance_threshold=self._distance_threshold,
            pillar_threshold_percentile=self._pillar_threshold_percentile,
        )
        return merged[0], order[benchmark[0]], weights[0]

    def _resolve_tile_grid(self, tiles: int) -> tuple[int, int, bool]:
        """Rebuild the image processor's ``rows x cols`` tile layout.

        The processor appends a thumbnail whenever it cut more than one tile, so
        ``tiles - 1`` tiles form the mosaic; a single tile means the image was small
        enough to encode whole and there is no thumbnail.
        """
        if tiles <= 1:
            return 1, 1, False
        blocks = tiles - 1
        height, width = self._image_sizes[0] if self._image_sizes else (0, 0)
        rows, cols = 0, 0
        if height and width:
            try:
                from transformers.models.got_ocr2.image_processing_got_ocr2 import (
                    get_optimal_tiled_canvas,
                )

                cols, rows = get_optimal_tiled_canvas(
                    (height, width), self._tile_size, self._min_patches, self._max_patches)
            except Exception:  # pragma: no cover - keep working if the path moves
                rows, cols = 0, 0
        if rows * cols != blocks:
            rows, cols = self._closest_factor_pair(blocks, height, width)
            self._warn_once(
                "grid_fallback",
                f"could not reproduce the processor's tile layout for a {height}x{width} image "
                f"cut into {blocks} tiles; falling back to the closest {rows}x{cols} factor pair.",
            )
        return rows, cols, True

    @staticmethod
    def _closest_factor_pair(blocks: int, height: int, width: int) -> tuple[int, int]:
        """``rows x cols == blocks`` with the aspect ratio closest to the image's."""
        target = (width / height) if height and width else 1.0
        best = (1, blocks)
        best_diff = float("inf")
        for rows in range(1, blocks + 1):
            if blocks % rows:
                continue
            cols = blocks // rows
            diff = abs(target - cols / rows)
            if diff < best_diff:
                best_diff, best = diff, (rows, cols)
        return best

    # ------------------------------------------------------- sequence-side cut

    def _language_model_pre_hook(self, _module: Any, args: tuple,
                                 kwargs: dict) -> tuple[tuple, dict] | None:
        inputs_embeds = kwargs.get("inputs_embeds")
        if not isinstance(inputs_embeds, torch.Tensor) or inputs_embeds.ndim != 3:
            return None
        if inputs_embeds.shape[1] == 1:
            return self._decode(args, kwargs, inputs_embeds)
        return self._prefill(args, kwargs, inputs_embeds)

    def _prefill(self, args: tuple, kwargs: dict,
                 inputs_embeds: torch.Tensor) -> tuple[tuple, dict] | None:
        keep_visual = self._keep_visual
        if keep_visual is None:
            return None  # text-only prompt, or Stage 1 was skipped
        batch = inputs_embeds.shape[0]
        if batch != 1:
            raise NotImplementedError(
                f"Nuwa on InternVL supports batch_size=1 only, got {batch}. InternVL gives each "
                f"image its own tile count, so a padded batch cannot share one keep-set."
            )
        input_ids = self._locate_input_ids(kwargs, inputs_embeds.shape[1])
        if input_ids is None:
            raise NotImplementedError(
                "Nuwa on InternVL locates the image span in `input_ids`, so it cannot prune a "
                "forward driven by `inputs_embeds` alone."
            )
        if kwargs.get("use_cache") is False:
            raise NotImplementedError(
                "Nuwa on InternVL prunes the KV cache, so `use_cache=False` would run the "
                "unpruned sequence on every step. Run with use_cache=True."
            )

        device = inputs_embeds.device
        positions = (input_ids[0] == self._image_token_id).nonzero(as_tuple=True)[0]
        num_visual = int(positions.numel())
        if num_visual == 0:
            return None
        first, last = int(positions[0]), int(positions[-1])
        if num_visual != last - first + 1:
            raise NotImplementedError(
                "Nuwa on InternVL expects one contiguous image span per prompt; this prompt "
                "carries several."
            )
        if num_visual != keep_visual.numel():
            raise RuntimeError(
                f"Nuwa merged {keep_visual.numel()} visual tokens but the prompt carries "
                f"{num_visual} placeholders."
            )

        self._image_start = first
        keep = torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=device)
        keep[first:first + num_visual] = keep_visual.to(device)
        visual_count = int(keep_visual.sum())
        self._visual_count = visual_count
        if visual_count == num_visual:
            # Nothing was dropped; the span stays where it is and Stage 2 ranks it there.
            self._vision_positions = (
                first + torch.arange(num_visual, device=device)).unsqueeze(0)
            return None

        keep_index = keep.nonzero(as_tuple=True)[0].unsqueeze(0)
        new_length = int(keep_index.shape[1])
        new_embeds = inputs_embeds[:, keep]
        # RPME: every survivor keeps the absolute position it held in the unpruned
        # sequence, so RoPE still sees the true distance across the deleted span and
        # the text after the image keeps its own offset.
        position_ids = gather_position_ids(
            kwargs.get("position_ids"), keep_index, inputs_embeds.shape[1])
        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            kwargs["attention_mask"] = attention_mask[:, keep]
        elif attention_mask is not None:
            raise NotImplementedError(
                f"Nuwa on InternVL needs the 2-D padding mask to prune; got "
                f"{tuple(attention_mask.shape)}."
            )
        kwargs["inputs_embeds"] = new_embeds
        kwargs["position_ids"] = position_ids
        if "cache_position" in kwargs:
            kwargs["cache_position"] = torch.arange(new_length, device=device)

        self._vision_positions = (first + torch.arange(visual_count, device=device)).unsqueeze(0)
        self._stage1_state = {
            "batch": 1,
            "prefill_len": new_length,
            "kv_len": new_length,
            "base_position": position_ids[..., -1:].clone(),
        }
        if self._forward_count == 1:
            print(f"[Nuwa/InternVL] prefill: sequence {inputs_embeds.shape[1]} -> {new_length} "
                  f"(visual {num_visual} -> {visual_count} at [{first}, {first + visual_count}))",
                  flush=True)
        return args, kwargs

    def _locate_input_ids(self, kwargs: dict, seq_len: int) -> torch.Tensor | None:
        """The prompt ids for this forward, which the language model no longer gets.

        ``InternVLModel.forward`` embeds ``input_ids`` and scatters the image
        features in before calling the language model, so the ids are not in this
        hook's ``kwargs`` -- they come from the outer model's own call, captured by
        :meth:`_capture_model_inputs`. A length mismatch means the ids describe a
        different sequence, so they are refused rather than trusted.
        """
        input_ids = kwargs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            input_ids = self._input_ids
        if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2 \
                and input_ids.shape[1] == seq_len:
            return input_ids
        return None

    def _decode(self, args: tuple, kwargs: dict,
                inputs_embeds: torch.Tensor) -> tuple[tuple, dict] | None:
        """Re-address one generated token against the Stage-1 pruned cache."""
        state = self._stage1_state
        if state is None:
            return None
        if inputs_embeds.shape[0] != state["batch"]:
            raise RuntimeError(
                f"Nuwa pruned the cache for batch {state['batch']} but decoding now runs batch "
                f"{inputs_embeds.shape[0]}. Beam search expands the batch after prefill and is "
                f"not supported; use greedy or sampling decoding."
            )
        state["kv_len"] += 1
        kv_len = int(state["kv_len"])
        step = kv_len - int(state["prefill_len"])
        kwargs["position_ids"] = state["base_position"] + step
        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            # batch_size == 1 and the prompt is unpadded, so every cached key is real.
            kwargs["attention_mask"] = attention_mask.new_ones((attention_mask.shape[0], kv_len))
        if "cache_position" in kwargs:
            kwargs["cache_position"] = torch.arange(
                kv_len - 1, kv_len, device=inputs_embeds.device)
        return args, kwargs

    # ------------------------------------------------------------------- Stage 2

    def _on_stage2_attention(self, weights: torch.Tensor) -> None:
        self._stage2_attention = weights

    def _make_layer_hook(self, layer_idx: int) -> Any:
        def hook(_module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict] | None:
            return self._layer_hook(args, kwargs, layer_idx)

        return hook

    def _layer_hook(self, args: tuple, kwargs: dict,
                    layer_idx: int) -> tuple[tuple, dict] | None:
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            return None

        prefill = hidden_states.shape[1] > 1
        if layer_idx == self._stage2_layer:
            if prefill:
                hidden_states = self._prune(hidden_states, kwargs)
            elif self._prune_state:
                advance_decode_step(self._prune_state, hidden_states, label="Nuwa/InternVL")
        if not self._prune_state:
            return None
        apply_prune_state(kwargs, self._prune_state, prefill=prefill)
        kwargs.pop("hidden_states", None)  # it moves to args, so it cannot be passed twice
        return (hidden_states,), kwargs

    def _prune(self, hidden_states: torch.Tensor, kwargs: dict) -> torch.Tensor:
        """Cut the visual span to the Stage-2 budget and size the rest of the stack."""
        self._prune_state = {}
        attention = self._stage2_attention
        self._stage2_attention = None  # the weights are large; do not hold them

        columns = self._vision_positions
        if columns is None:
            return hidden_states
        batch, seq_len, hidden_dim = hidden_states.shape
        if attention is None:
            self._warn_once(
                "no_capture",
                f"no attention was captured at layer {self._stage2_layer - 1}; Stage 2 is SKIPPED "
                f"and only Stage 1's tokens reach the deeper layers.",
            )
            return hidden_states
        columns = columns.to(hidden_states.device)
        if columns.shape[0] != batch or attention.shape[0] != batch \
                or int(columns.max()) >= seq_len:
            self._warn_once(
                "geometry_changed",
                f"the visual span {tuple(columns.shape)} does not fit the decoder input "
                f"{tuple(hidden_states.shape)}; Stage 2 is SKIPPED for this batch.",
            )
            return hidden_states

        target = preserve_count(self._visual_count)
        if target >= self._visual_count:
            return hidden_states  # the budget already fits; nothing left to cut

        local = select_by_text_attention(attention, columns, target)
        kept_columns = columns.gather(1, local)
        keep_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden_states.device)
        keep_mask.scatter_(1, columns, False)  # drop the whole visual span ...
        keep_mask.scatter_(1, kept_columns, True)  # ... then put the winners back
        keep_index = keep_columns(keep_mask)
        new_len = keep_index.shape[1]

        new_hidden = hidden_states.gather(
            1, keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim))
        position_ids = gather_position_ids(kwargs.get("position_ids"), keep_index, seq_len)
        position_embeddings = select_rope_tables(
            kwargs.get("position_embeddings"), keep_index, seq_len)
        if position_embeddings is None:
            position_embeddings = self._language_model.rotary_emb(
                new_hidden, position_ids=position_ids)
        mask, valid = pruned_attention_mask(kwargs.get("attention_mask"), keep_index, seq_len)

        self._prune_state = {
            "batch": batch,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "attention_mask": mask,
            "valid": valid,
            # Layers at or below the cut cache only the survivors, so their cache
            # positions are the compact ones; the absolute ids above are RoPE's alone.
            "cache_position": torch.arange(new_len, device=hidden_states.device),
            "kv_len": new_len,
        }
        if self._forward_count == 1:
            print(f"[Nuwa/InternVL] stage 2 @layer{self._stage2_layer}: seq {seq_len} -> "
                  f"{new_len} (visual {self._visual_count} -> {target})", flush=True)
        self._visual_count = target
        self._vision_positions = (
            self._image_start + torch.arange(target, device=hidden_states.device)).unsqueeze(0)
        self._record_cut(local)
        return new_hidden

    # -------------------------------------------------------------------- source

    def _record_stage1_source(self, num_visual: int,
                              sources: Sequence[tuple[torch.Tensor, torch.Tensor]],
                              sequence_order: torch.Tensor) -> None:
        """``[identity, stage1]`` attribution over the whole visual span.

        Each group's merge weights are indexed by *its own mosaic*, so column ``j``
        is scattered onto span column ``order[j]`` before the rows of the two groups
        are put into sequence order. There is no ``[CLS]`` column here, unlike the
        LLaVA path.
        """
        rows: list[torch.Tensor] = []
        for order, weights in sources:
            weights = weights.detach().float().cpu()
            group = torch.zeros(weights.shape[0], num_visual, dtype=torch.float32)
            group[:, order.detach().cpu()] = weights
            rows.append(group)
        source = torch.cat(rows, dim=0)[sequence_order.detach().cpu()]
        self._composed_source = source
        self._source_history = [torch.eye(num_visual, dtype=torch.float32), source]

    def _record_cut(self, local: torch.Tensor) -> None:
        """Narrow the composed attribution to the survivors of the Stage-2 cut."""
        if not self._track_source or self._composed_source is None or local.shape[1] == 0:
            return
        self._composed_source = self._composed_source[local[0].detach().cpu()]
        self._source_history.append(self._composed_source)
