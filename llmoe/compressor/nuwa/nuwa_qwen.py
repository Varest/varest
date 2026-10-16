"""Nuwa on Qwen2.5-VL -- the half of the method that cannot be shared with CLIP.

Reference code: ``Vscan/qwen/model/qwen2_5vl_nuwa.py`` in
github.com/Man-PaperRejected/Nuwa (identical to
``lmms-eval/lmms_eval/models/qwen2_5vl_nuwa.py`` bar where ``retain_ratio`` is read
from), a fork of the transformers ~4.49 modeling file with both stages spliced
into ``Qwen2_5_VLForConditionalGeneration.forward``.

Owned by :class:`~llmoe.compressor.nuwa.nuwa.NuwaCompressor`, which detects the
backbone at ``install()`` and delegates here. The Stage-1/Stage-2 policy is
shared in ``policy.py`` and decoder geometry is shared in ``_pruning.py``; the
backbone plumbing every Qwen port needs (hook lifecycle,
prompt capture, the last-block attention replay, span location, M-RoPE narrowing
and decode re-addressing) is
:class:`~llmoe.compressor._pruning.QwenPruningAdapter`. What lives here is what
Nuwa does differently.

Qwen2.5-VL has no ``[CLS]``, no fixed square grid, and a patch merger that folds
every ``2 x 2`` block of encoder patches into one LLM token, so upstream replaces
every CLIP-side signal:

* salience = the last ViT block's attention *received* (head mean, query sum),
  averaged within each merger group and un-permuted out of window order;
* similarity features = ``merger(mean of ViT blocks 7..23)``;
* aggregation target = the LLM-side embeddings, after the merger and after the
  scatter into the prompt;
* grid = the post-merger ``H/2 x W/2`` grid, zero-padded to even sides, with
  survivors landing on the padding dropped afterwards;
* distance cutoff = ``0.8 * diagonal`` rather than the fixed ``sqrt(280)``;
* Pillar split = 0.1, not 0.55, so only the bottom decile merges anything;
* no self-weight override after normalising (``clip_encoder.py`` does it, Qwen
  does not);
* Stage 2 ranks by the mean of the last **three** query rows reduced over heads by
  **max**, read off layer 14, cutting immediately after it;
* budgets are ratios: ``retain_ratio`` in ``{0.25, 0.5, 0.75}`` selects a published
  ``(Stage-1 ratio, Stage-2 ratio, top_n_per_region)`` triple, and ``max_tokens``
  means nothing because the token count varies per image.

Harness adaptations
-------------------
* Upstream prunes ``inputs_embeds`` after the features are scattered in, so the
  placeholder count check passes and ``get_rope_index`` still sees the untouched
  sequence; this port hooks ``model.model.language_model``, the same call.
  Survivors keep their original M-RoPE coordinates, which is RPME by construction.
* The last block's ``[H, L, L]`` softmax is recomputed in chunks from its own
  ``qkv`` instead of being returned by a forked ViT: the query-row reduction is a
  sum, so chunking is exact. Mid-stack hidden states accumulate as one float32
  running sum instead of a 17-deep stack.
* **One upstream bug fixed.** Upstream never applies ``argsort(window_index)`` to
  the similarity features, so they stay in window order while the aggregation
  features and the distance penalty are in raster order and the cosine similarity
  compares unrelated tokens. ``_merged_token_signals`` un-permutes them.
* Only layer 14 runs eager (a registered attention function delegates the rest to
  SDPA), and the decode path rebuilds the mask at the pruned cache's real width,
  which ``generate`` does not know about.
* ``batch_size=1`` only, as upstream requires. Video, multi-image prompts, beam
  search and ``use_cache=False`` raise.
"""

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from llmoe.backbone.prototype import VLM
from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    advance_decode_step,
    apply_prune_state,
    gather_mrope_position_ids,
    keep_columns,
    pruned_attention_mask,
    select_rope_tables,
)
from llmoe.compressor.nuwa.policy import (
    aggregate_tokens,
    build_distance_penalty_matrix,
    pool_merge_groups,
    region_candidate_selection,
    register_capture_attention,
    select_benchmark_tokens,
    select_by_text_attention,
    set_attn_implementation,
    unregister_capture_attention,
)
from llmoe.utils.qwen import DEFAULT_ATTENTION_CHUNK

# Published `config` table: retain_ratio -> (Stage-1 ratio, Stage-2 ratio of Stage 1,
# top_n_per_region). Verbatim from the reference forward.
QWEN_PROFILES: dict[float, tuple[float, float, int]] = {
    0.25: (0.35, 0.42, 3),
    0.50: (0.60, 0.66, 3),
    0.75: (0.85, 0.70, 4),
}

# Upstream constants for the Qwen path, all different from the CLIP releases'.
QWEN_PILLAR_PERCENTILE: float = 0.1
QWEN_DISTANCE_RATIO: float = 0.8
QWEN_SIM_BLOCKS: tuple[int, int] = (7, 24)  # `7 <= layer_num <= 23`, inclusive
QWEN_STAGE2_LAYER: int = 15  # upstream prunes *after* layer 14
QWEN_STAGE2_QUERY_ROWS: int = 3


class QwenNuwaAdapter(QwenPruningAdapter):
    """Nuwa's Qwen2.5-VL half: ratio-parameterised Stage 1 on the merged tokens.

    :param retain_ratio: published retention, one of ``QWEN_PROFILES``.
    :param stage1_ratio: fraction of the image's tokens surviving Stage 1.
        Defaults to the profile's.
    :param stage2_ratio: fraction of the *Stage-1* tokens surviving Stage 2.
    :param top_n_per_region: candidates per ``2 x 2`` region.
    :param pillar_threshold_percentile: Pillar/Collector split, 0.1 upstream.
    :param distance_ratio: the spatial cutoff as a fraction of the padded grid's
        diagonal, 0.8 upstream.
    :param stage2_layer: first decoder layer to run on the reduced set; its
        predecessor supplies the ranking attention.
    :param sim_layers: ``[start, end)`` ViT blocks averaged into the similarity
        features.
    :param attention_chunk: query rows per matmul in the vision-tower recompute.
        Lower it if the tower runs out of memory on very large images.
    :param track_source: record ``[T_current, T_original]`` matrices for
        ``llmoe.viz``.
    """

    method = "Nuwa"
    capture_vision_attention = True

    def __init__(self, *, retain_ratio: float, stage1_ratio: float, stage2_ratio: float,
                 top_n_per_region: int, pillar_threshold_percentile: float,
                 distance_ratio: float, stage2_layer: int, sim_layers: Sequence[int],
                 attention_chunk: int, track_source: bool) -> None:
        super().__init__(attention_chunk=attention_chunk, track_source=track_source)
        self._retain_ratio: float = float(retain_ratio)
        self._stage1_ratio: float = float(stage1_ratio)
        self._stage2_ratio: float = float(stage2_ratio)
        self._top_n_per_region: int = int(top_n_per_region)
        self._pillar_threshold_percentile: float = float(pillar_threshold_percentile)
        self._distance_ratio: float = float(distance_ratio)
        self._stage2_layer: int = int(stage2_layer)
        self._sim_layers: tuple[int, int] = (int(sim_layers[0]), int(sim_layers[1]))
        if not 0.0 < self._stage1_ratio <= 1.0:
            raise ValueError(f"stage1_ratio must be in (0, 1], got {stage1_ratio}")
        if not 0.0 < self._stage2_ratio <= 1.0:
            raise ValueError(f"stage2_ratio must be in (0, 1], got {stage2_ratio}")

        # Installation bookkeeping beyond the base class's, undone by uninstall().
        self._attention_name: str | None = None
        self._previous_llm_attn: str | None = None

        # Per-forward capture and per-prefill state beyond the base class's.
        self._sim_sum: torch.Tensor | None = None  # [L_patches, C] window order, float32
        self._sim_count: int = 0
        self._image_start: int = 0
        self._visual_count: int = 0  # visual tokens the decoder currently carries
        self._vision_positions: torch.Tensor | None = None  # [1, V] current columns
        self._stage2_attention: torch.Tensor | None = None
        self._prune_state: dict[str, Any] = {}
        self._composed_source: torch.Tensor | None = None

    # ------------------------------------------------------------------ install

    def install(self, vlm: VLM, *, image_token_id: int) -> None:
        self._bind(vlm, image_token_id=image_token_id)
        blocks = self._visual.blocks
        layers = self._language_model.layers
        if not 1 <= self._stage2_layer < len(layers):
            raise ValueError(
                f"stage2_layer={self._stage2_layer} does not fit a {len(layers)}-layer decoder; "
                f"Qwen2.5-VL-7B has 28 layers and upstream prunes at {QWEN_STAGE2_LAYER}."
            )
        start, end = self._sim_layers
        if not 0 <= start < end or end > len(blocks):
            raise ValueError(
                f"sim_layers {self._sim_layers} do not fit a {len(blocks)}-block vision tower")

        self._register_common_hooks()
        for index in range(start, end):
            self._handles.append(blocks[index].register_forward_hook(self._capture_sim_block))
        self._attention_name, self._previous_llm_attn = register_capture_attention(
            self._language_model, self._stage2_layer - 1, self._on_stage2_attention,
            owner=self, label="nuwa_qwen")
        for index in range(self._stage2_layer, len(layers)):
            self._handles.append(
                layers[index].register_forward_pre_hook(self._make_layer_hook(index),
                                                        with_kwargs=True))

        print(f"[Nuwa/Qwen] install: retain_ratio={self._retain_ratio:g} (stage1={self._stage1_ratio:g} "
              f"of the image, stage2={self._stage2_ratio:g} of stage1), "
              f"top_n_per_region={self._top_n_per_region}, sim blocks [{start}, {end}) of "
              f"{len(blocks)}, prune @layer{self._stage2_layer} of {len(layers)} "
              f"(score @layer{self._stage2_layer - 1})", flush=True)

    def uninstall(self) -> None:
        if self._language_model is not None and self._previous_llm_attn is not None:
            set_attn_implementation(self._language_model, self._previous_llm_attn)
        unregister_capture_attention(self._attention_name)
        self._attention_name = None
        self._previous_llm_attn = None
        super().uninstall()

    def _reset_forward_state(self) -> None:
        super()._reset_forward_state()
        self._sim_sum = None
        self._sim_count = 0
        self._image_start = 0
        self._visual_count = 0
        self._vision_positions = None
        self._stage2_attention = None
        self._prune_state = {}
        self._composed_source = None

    # -------------------------------------------------------------- ViT capture

    def _capture_sim_block(self, _module: Any, _args: tuple, output: Any) -> None:
        """Accumulate the mid-stack hidden states the similarity features average.

        Upstream appends blocks 7..23 to a list and takes ``vstack(...).mean(0)``;
        a float32 running sum is the same mean without holding 17 copies of a
        tensor that is ~40 MB per block at 1024x1024.
        """
        hidden_states = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden_states, torch.Tensor):
            return None
        with torch.no_grad():
            if self._sim_sum is None or self._sim_sum.shape != hidden_states.shape:
                self._sim_sum = hidden_states.float().clone()
                self._sim_count = 1
            else:
                self._sim_sum += hidden_states.float()
                self._sim_count += 1
        return None

    def _merged_token_signals(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold the per-patch capture onto the merged tokens the decoder sees.

        The salience is group-averaged and un-permuted by ``argsort(window_index)``
        exactly as the base class does; what Nuwa swaps in is the second signal.
        Upstream builds its similarity features as ``merger(mean(blocks 7..23))``
        and never applies ``reverse_indices`` to them, which leaves them in window
        order while the aggregation features and the distance penalty are in
        raster order -- so the cosine similarity is taken between unrelated
        tokens. That is a bug, and it is fixed here.
        """
        merge_unit, reverse_indices = self._window_reverse_indices()
        scores = pool_merge_groups(self._received, merge_unit)[reverse_indices]
        sim_mean = (self._sim_sum / max(self._sim_count, 1)).to(self._visual.dtype)
        sim_features = self._visual.merger(sim_mean)[reverse_indices]
        return scores, sim_features

    # ----------------------------------------------------------------- Stage 1

    def _prefill(self, args: tuple, kwargs: dict,
                 inputs_embeds: torch.Tensor) -> tuple[tuple, dict] | None:
        span = self._locate_image_span(inputs_embeds)
        if span is None or self._received is None or self._sim_sum is None \
                or self._image_grid_thw is None:
            return None  # text-only prompt, or a tower that did not run
        first, num_visual = span
        self._require_kv_cache(kwargs)

        device = inputs_embeds.device
        merged, keep_visual = self._stage1(inputs_embeds, first, num_visual)
        if merged is None:
            # Stage 1 found nothing to work with: the decoder keeps the full span,
            # and Stage 2 still ranks it where it stands.
            self._image_start = first
            self._visual_count = num_visual
            self._vision_positions = (
                first + torch.arange(num_visual, device=device)).unsqueeze(0)
            return None

        keep = torch.ones(inputs_embeds.shape[1], dtype=torch.bool, device=device)
        keep[first:first + num_visual] = keep_visual
        modified = inputs_embeds.clone()
        modified[0, first:first + num_visual][keep_visual] = merged.to(modified.dtype)

        state = self._apply_keep(kwargs, inputs_embeds, modified, keep)
        new_length = int(state["prefill_len"])
        visual_count = int(keep_visual.sum())
        self._image_start = first
        self._visual_count = visual_count
        self._vision_positions = (first + torch.arange(visual_count, device=device)).unsqueeze(0)
        if self._forward_count == 1:
            print(f"[Nuwa/Qwen] stage 1: visual {num_visual} -> {visual_count}, sequence "
                  f"{inputs_embeds.shape[1]} -> {new_length}", flush=True)
        return args, kwargs

    def _stage1(self, inputs_embeds: torch.Tensor, first: int,
                num_visual: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Separation, Alignment and Aggregation on the padded post-merger grid.

        :return: ``(merged [K, D], keep_visual [num_visual] bool)``, or
            ``(None, None)`` when every candidate landed on the padding. A budget
            that happens to keep every token still merges: upstream aggregates
            first and prunes second, so the features change even when the count
            does not.
        """
        device = inputs_embeds.device
        grid = self._image_grid_thw
        if grid is None or grid.shape[0] != 1:
            raise NotImplementedError(
                "Nuwa on Qwen2.5-VL supports exactly one image per prompt; this sample carries "
                f"{0 if grid is None else int(grid.shape[0])}."
            )
        # `image_grid_thw` is the *pre-merge* patch grid; the merger halves both sides.
        merge_size = int(getattr(self._visual, "spatial_merge_size", 2))
        height = int(grid[0][1]) // merge_size
        width = int(grid[0][2]) // merge_size
        if height * width != num_visual:
            raise RuntimeError(
                f"Nuwa expected {height * width} post-merger tokens from grid {grid[0].tolist()} "
                f"but the prompt carries {num_visual} image placeholders."
            )

        scores, sim_features = self._merged_token_signals()
        if scores.numel() != num_visual:
            raise RuntimeError(
                f"Nuwa scored {scores.numel()} visual tokens but the prompt carries "
                f"{num_visual} image placeholders."
            )
        keep_count = math.ceil(num_visual * self._stage1_ratio)

        # Pad the grid up to even sides so the 2x2 regions tile it. Padded entries
        # score 0 and carry zero features, so they never win a region that holds a
        # real token and contribute nothing to a merge; the ones that do win an
        # all-padding region are dropped again below.
        pad_h, pad_w = (2 - height % 2) % 2, (2 - width % 2) % 2
        padded_h, padded_w = height + pad_h, width + pad_w
        image_embeds = inputs_embeds[0, first:first + num_visual]
        metric = self._pad_grid(scores.unsqueeze(-1), height, width, pad_h, pad_w).squeeze(-1)
        sim_padded = self._pad_grid(sim_features, height, width, pad_h, pad_w)
        agg_padded = self._pad_grid(image_embeds, height, width, pad_h, pad_w)

        candidates = region_candidate_selection(
            metric.unsqueeze(0), padded_h, padded_w, self._top_n_per_region)
        keep_count = min(keep_count, int(candidates.shape[1]))
        benchmark = select_benchmark_tokens(metric.unsqueeze(0), candidates, keep_count)

        # Drop the survivors that landed on the padding, so only real tokens remain.
        rows, columns = benchmark[0] // padded_w, benchmark[0] % padded_w
        benchmark = benchmark[0][(rows < height) & (columns < width)].unsqueeze(0)
        if benchmark.shape[1] == 0:
            return None, None

        # The cutoff scales with the grid instead of the fixed sqrt(280), because
        # the grid size changes with the image.
        diagonal = math.sqrt((padded_h - 1) ** 2 + (padded_w - 1) ** 2)
        dist_penalty = build_distance_penalty_matrix(
            padded_h, padded_w, diagonal * self._distance_ratio if diagonal > 0 else 1.0, device)
        merged, weights = aggregate_tokens(
            agg_padded.unsqueeze(0), sim_padded.unsqueeze(0), metric.unsqueeze(0), benchmark,
            dist_penalty, self._pillar_threshold_percentile, self_weight_override=False,
        )

        # Back from the padded grid to the visual span's own indexing.
        padded_keep = torch.zeros(padded_h * padded_w, dtype=torch.bool, device=device)
        padded_keep[benchmark[0]] = True
        keep_visual = padded_keep.view(padded_h, padded_w)[:height, :width].reshape(-1)
        if int(keep_visual.sum()) != merged.shape[1]:
            raise RuntimeError(
                f"Nuwa selected {int(keep_visual.sum())} tokens but aggregated "
                f"{merged.shape[1]}; the padding filter and the merge disagree."
            )
        if self._track_source:
            self._record_stage1_source(height, width, padded_h, padded_w, weights[0])
        return merged[0], keep_visual

    @staticmethod
    def _pad_grid(values: torch.Tensor, height: int, width: int, pad_h: int,
                  pad_w: int) -> torch.Tensor:
        """Zero-pad a ``[H * W, C]`` grid signal to ``[(H + pad_h) * (W + pad_w), C]``."""
        if not pad_h and not pad_w:
            return values
        channels = values.shape[-1]
        grid = values.view(height, width, channels)
        # F.pad's last-dim-first ordering: (channel, channel, width, width, height, height).
        grid = F.pad(grid, (0, 0, 0, pad_w, 0, pad_h), mode="constant", value=0)
        return grid.reshape((height + pad_h) * (width + pad_w), channels)

    # ----------------------------------------------------------------- Stage 2

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
                advance_decode_step(self._prune_state, hidden_states, label="Nuwa/Qwen")
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

        target = max(1, int(self._visual_count * self._stage2_ratio))
        if target >= self._visual_count:
            return hidden_states  # the budget already fits; nothing left to cut

        local = select_by_text_attention(
            attention, columns, target, query_rows=QWEN_STAGE2_QUERY_ROWS,
            head_reduction="max")
        kept_columns = columns.gather(1, local)
        keep_mask = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden_states.device)
        keep_mask.scatter_(1, columns, False)  # drop the whole visual span ...
        keep_mask.scatter_(1, kept_columns, True)  # ... then put the winners back
        keep_index = keep_columns(keep_mask)
        new_len = keep_index.shape[1]

        new_hidden = hidden_states.gather(
            1, keep_index.unsqueeze(-1).expand(batch, new_len, hidden_dim))
        position_ids, text_ids = gather_mrope_position_ids(
            kwargs.get("position_ids"), keep_index, seq_len, new_len)
        position_embeddings = select_rope_tables(
            kwargs.get("position_embeddings"), keep_index, seq_len)
        if position_embeddings is None:
            # The layer may carry only the 2-D packing row (transformers 5.x), which
            # cannot drive rotary_emb; the Stage-1 cut stored the survivors' full
            # M-RoPE ids for exactly this case.
            mrope_ids = position_ids if position_ids.ndim == 3 else None
            if mrope_ids is None and self._prefill_state is not None:
                stored = self._prefill_state.get("position_ids")
                if isinstance(stored, torch.Tensor) and stored.shape[-1] == seq_len:
                    mrope_ids = stored[..., keep_index[0]]
            if mrope_ids is None:
                raise RuntimeError(
                    "Nuwa could not derive M-RoPE tables for the Stage-2 cut: the decoder "
                    "supplied neither selectable rope tables nor 3-row position ids."
                )
            position_embeddings = self._language_model.rotary_emb(new_hidden, mrope_ids[-3:])
        mask, valid = pruned_attention_mask(kwargs.get("attention_mask"), keep_index, seq_len)

        self._prune_state = {
            "batch": batch,
            "position_ids": text_ids,
            "position_embeddings": position_embeddings,
            "attention_mask": mask,
            "valid": valid,
            # Layers at or below the cut cache only the survivors, so their cache
            # positions are the compact ones; the absolute M-RoPE coordinates above
            # are RoPE's business alone.
            "cache_position": torch.arange(new_len, device=hidden_states.device),
            "kv_len": new_len,
        }
        if self._forward_count == 1:
            print(f"[Nuwa/Qwen] stage 2 @layer{self._stage2_layer}: seq {seq_len} -> {new_len} "
                  f"(visual {self._visual_count} -> {target})", flush=True)
        self._visual_count = target
        # The span is contiguous and everything before it survived, so after the
        # compaction the survivors sit at `image_start + 0..target`.
        self._vision_positions = (
            self._image_start + torch.arange(target, device=hidden_states.device)).unsqueeze(0)
        self._record_cut(local)
        return new_hidden

    # -------------------------------------------------------------------- source

    def _record_stage1_source(self, height: int, width: int, padded_h: int, padded_w: int,
                              weights: torch.Tensor) -> None:
        """``[identity, stage1]`` attribution over the image span, for ``llmoe.viz``.

        The columns are re-expressed on the unpadded grid, so the matrix describes
        the visual tokens the prompt actually carries; the padding columns it drops
        contribute nothing to any merge. There is no ``[CLS]`` column here, unlike
        the LLaVA path.
        """
        unpadded = torch.arange(padded_h * padded_w).view(padded_h, padded_w)[:height, :width]
        source = weights.detach().float().cpu()[:, unpadded.reshape(-1)]
        self._composed_source = source
        self._source_history = [torch.eye(height * width, dtype=torch.float32), source]

    def _record_cut(self, local: torch.Tensor) -> None:
        """Narrow the composed attribution to the survivors of the Stage-2 cut."""
        if not self._track_source or self._composed_source is None or local.shape[1] == 0:
            return
        self._composed_source = self._composed_source[local[0].detach().cpu()]
        self._source_history.append(self._composed_source)
