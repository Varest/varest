"""Native Transformers LLaVA-NeXT adapter for the released Nuwa path."""

import math
from typing import Any

import torch

from llmoe.compressor.internvl import image_token_id
from llmoe.compressor.llava_next import (
    LlavaNextFeatureTransformAdapter,
    crop_counts,
    decoder_visual_positions,
    patch_grid,
    split_crop_tensor,
    stitch_and_unpad_local_crops,
    two_stage_layer_average_schedule,
)
from llmoe.compressor.nuwa.nuwa_internvl import NEXT_PILLAR_PERCENTILE
from llmoe.compressor.nuwa.policy import (
    STAGE2_RATIO,
    average_hidden_states,
    nuwa_stage1,
    set_attn_implementation,
    unregister_capture_attention,
)

NEXT_TOP_N_PER_REGION = 3


def layer_average_schedule(
    native_tokens: int,
    retain_ratio: float,
    stage2_layer: int,
    decoder_layers: int,
) -> tuple[int, int]:
    """Largest released two-stage schedule not exceeding the requested ratio."""

    return two_stage_layer_average_schedule(
        native_tokens,
        retain_ratio=retain_ratio,
        stage2_layer=stage2_layer,
        decoder_layers=decoder_layers,
        stage2_ratio=STAGE2_RATIO,
        label="Nuwa/LLaVA-NeXT",
    )


def _candidate_top_n(tokens: int, grid_h: int, grid_w: int) -> int:
    regions = max(1, (grid_h // 2) * (grid_w // 2))
    return min(4, max(NEXT_TOP_N_PER_REGION, math.ceil(tokens / regions)))


def _split_patch_budget(
    total: int,
    crops: int,
    global_capacity: int,
    local_capacity: int,
) -> tuple[int, int]:
    """Keep the official equal-per-crop split while respecting unpadding."""

    if not 1 <= total <= global_capacity + local_capacity:
        raise ValueError(
            f"Nuwa cannot allocate {total} patch rows into capacities "
            f"{global_capacity}+{local_capacity}."
        )
    preferred_global = math.ceil(total / crops)
    global_tokens = min(global_capacity, preferred_global)
    local_tokens = total - global_tokens
    if local_tokens > local_capacity:
        shift = local_tokens - local_capacity
        global_tokens += shift
        local_tokens -= shift
    if global_tokens > global_capacity:
        shift = global_tokens - global_capacity
        global_tokens -= shift
        local_tokens += shift
    return global_tokens, local_tokens


class LlavaNextNuwaAdapter:
    """Nuwa Stage 1 over base/unpadded mosaics plus its existing decoder cut."""

    def __init__(self, owner: Any, vlm: Any, *, retain_ratio: float,
                 track_source: bool) -> None:
        self._owner = owner
        self._vlm = vlm
        self._inner = vlm.model.model
        self._language_model = self._inner.language_model
        self._retain_ratio = float(retain_ratio)
        self._track_source = bool(track_source)
        self._depth = len(self._language_model.layers)
        self._schedules: dict[int, tuple[int, int]] = {}
        self._lifecycle = LlavaNextFeatureTransformAdapter(
            label="Nuwa",
            retain_ratio=self._retain_ratio,
            transform_outputs=self._transform_outputs,
            target_for_length=self._stage1_target,
            on_prepared=self._on_prepared,
        )

    def _stage1_target(self, native_tokens: int) -> int:
        schedule = layer_average_schedule(
            native_tokens,
            self._retain_ratio,
            self._owner._stage2_layer,
            self._depth,
        )
        self._schedules[int(native_tokens)] = schedule
        return schedule[0]

    def install(self) -> None:
        layers = self._language_model.layers
        if not 1 <= self._owner._stage2_layer < len(layers):
            raise ValueError(
                f"Nuwa's canonical stage2_layer={self._owner._stage2_layer} "
                f"does not fit a {len(layers)}-layer LLaVA-NeXT decoder."
            )
        strategy = getattr(
            self._vlm.model.config, "vision_feature_select_strategy", "default"
        )
        if strategy != "default":
            raise ValueError(
                "Nuwa/LLaVA-NeXT supports the checkpoint's canonical "
                "vision_feature_select_strategy='default' only."
            )

        self._owner._inner_model = self._inner
        self._owner._language_model = self._language_model
        self._owner._image_token_id = image_token_id(
            self._vlm, label="Nuwa/LLaVA-NeXT"
        )
        try:
            self._owner._force_eager_vision(self._inner.vision_tower)
            self._owner._install_vision_capture(self._inner.vision_tower)
            self._lifecycle.install(self._vlm)
            self._owner._install_attention_capture(self._language_model)
            self._owner._install_rpme(self._language_model)
            self._owner._install_stage2(layers)
        except Exception:
            self.uninstall()
            raise

    def uninstall(self) -> None:
        self._lifecycle.uninstall()
        for handle in self._owner._hook_handles:
            handle.remove()
        self._owner._hook_handles = []
        if self._owner._previous_llm_attn is not None:
            set_attn_implementation(
                self._language_model, self._owner._previous_llm_attn
            )
        if (
            self._owner._vision_config is not None
            and self._owner._previous_vision_attn is not None
        ):
            self._owner._vision_config._attn_implementation = (
                self._owner._previous_vision_attn
            )
        unregister_capture_attention(self._owner._attention_name)
        self._owner._attention_name = None
        self._owner._previous_llm_attn = None
        self._owner._previous_vision_attn = None
        self._owner._vision_config = None
        self._owner._inner_model = None
        self._owner._language_model = None
        self._owner._image_token_id = None
        self._owner._reset_forward_state()
        self._schedules = {}

    def _on_prepared(
        self,
        inputs: Any,
        native_lengths: tuple[int, ...],
        stage1_lengths: tuple[int, ...],
    ) -> None:
        if len(native_lengths) != 1 or len(stage1_lengths) != 1:
            raise NotImplementedError(
                "Nuwa/LLaVA-NeXT follows the released batch_size=1, one-image "
                "path; evaluate this baseline with batch_size=1."
            )
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("Nuwa/LLaVA-NeXT needs input_ids to locate the image span")
        positions = decoder_visual_positions(
            input_ids,
            token_id=int(self._owner._image_token_id),
            label="Nuwa",
        )
        if positions is None:
            return

        native = native_lengths[0]
        stage1, stage2 = self._schedules.get(native) or layer_average_schedule(
            native,
            self._retain_ratio,
            self._owner._stage2_layer,
            self._depth,
        )
        if stage1 != stage1_lengths[0] or int(positions.numel()) != stage1:
            raise RuntimeError(
                "Nuwa's scheduled Stage-1 count does not match the trimmed image span"
            )
        self._owner._reset_forward_state()
        self._owner._num_patches = native
        self._owner._stage1_count = stage1
        self._owner._stage2_keep = stage2
        self._owner._vision_positions = positions.unsqueeze(0)
        if self._track_source:
            self._owner._source_history = []

    @staticmethod
    def _argument(args: tuple[Any, ...], kwargs: dict[str, Any],
                  name: str, position: int) -> Any:
        value = kwargs.get(name)
        if value is None and len(args) > position:
            value = args[position]
        return value

    @torch.no_grad()
    def _transform_outputs(
        self,
        image_outputs: Any,
        targets: tuple[int, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        hidden_states = getattr(image_outputs, "hidden_states", None)
        attentions = getattr(image_outputs, "attentions", None)
        if not hidden_states or not attentions:
            raise RuntimeError(
                "Nuwa/LLaVA-NeXT needs CLIP hidden states and eager attention "
                "weights for its released Stage 1."
            )
        image_sizes = self._argument(args, kwargs, "image_sizes", 1)
        if not isinstance(image_sizes, torch.Tensor):
            raise ValueError("Nuwa/LLaVA-NeXT needs image_sizes from the processor")
        counts = crop_counts(self._inner, image_sizes)
        feature_groups = split_crop_tensor(
            hidden_states[self._owner._vision_feature_layer][:, 1:],
            counts,
            label="Nuwa features",
        )
        sim_groups = split_crop_tensor(
            average_hidden_states(hidden_states, *self._owner._sim_layers)[:, 1:],
            counts,
            label="Nuwa similarity features",
        )
        attention_groups = split_crop_tensor(
            attentions[self._owner._vision_feature_layer][:, :, 0, 1:].sum(dim=1),
            counts,
            label="Nuwa CLS attention",
        )
        native_outputs = list(image_outputs.pooler_output)
        outputs: list[torch.Tensor] = []
        for image_index, (features, similarity, salience, target) in enumerate(
            zip(
                feature_groups,
                sim_groups,
                attention_groups,
                targets,
                strict=True,
            )
        ):
            native_count = int(native_outputs[image_index].shape[0])
            if target == native_count:
                outputs.append(native_outputs[image_index])
                benchmark = torch.arange(
                    native_count, device=features.device, dtype=torch.long
                )
                self._owner._benchmark_indices = benchmark.unsqueeze(0)
                if self._track_source and image_index == 0:
                    identity = torch.eye(native_count, dtype=torch.float32)
                    self._owner._stage1_source = identity
                    self._owner._source_history = [identity, identity.clone()]
                continue
            packed, benchmark, source = self._run_stage1(
                features,
                similarity,
                salience,
                image_sizes[image_index],
                target,
                native_count,
            )
            outputs.append(packed)
            if image_index == 0:
                self._owner._benchmark_indices = benchmark.unsqueeze(0)
                if self._track_source and source is not None:
                    self._owner._stage1_source = source
                    self._owner._source_history = [
                        torch.eye(native_count, dtype=torch.float32), source
                    ]
        image_outputs.pooler_output = outputs
        return image_outputs

    def _run_stage1(
        self,
        features: torch.Tensor,
        similarity: torch.Tensor,
        salience: torch.Tensor,
        image_size: torch.Tensor,
        target: int,
        native_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        crop_h, crop_w = patch_grid(self._inner)
        patches = crop_h * crop_w
        if int(features.shape[1]) != patches:
            raise ValueError(
                f"Nuwa expected {patches} patches per crop, got {features.shape[1]}"
            )
        crop_count = int(features.shape[0])
        if crop_count > 1:
            local_features, (local_h, local_w) = stitch_and_unpad_local_crops(
                self._inner, features[1:], image_size
            )
            local_similarity, _ = stitch_and_unpad_local_crops(
                self._inner, similarity[1:], image_size
            )
            local_salience, _ = stitch_and_unpad_local_crops(
                self._inner, salience[1:], image_size
            )
            local_salience = local_salience.squeeze(-1)
            newline_count = local_h
            local_capacity = local_h * local_w
        else:
            local_features = features[:0].reshape(1, 0, features.shape[-1])
            local_similarity = similarity[:0].reshape(1, 0, similarity.shape[-1])
            local_salience = salience[:0].reshape(1, 0)
            local_h = local_w = newline_count = local_capacity = 0

        patch_budget = target - newline_count
        if patch_budget < 1:
            raise ValueError(
                f"Nuwa's {newline_count} protected AnyRes row separators exhaust "
                f"the Stage-1 budget {target}; use a larger retain_ratio."
            )
        # The released 2x2 regional selector cannot nominate an unmatched odd
        # border as a benchmark token. It can still merge that border into the
        # selected rows, so use the candidate capacity for budgeting while the
        # provenance matrix continues to cover the full unpadded mosaic.
        local_candidate_capacity = (
            4 * (local_h // 2) * (local_w // 2)
            if crop_count > 1
            else 0
        )
        global_budget, local_budget = _split_patch_budget(
            patch_budget, crop_count, patches, local_candidate_capacity
        )

        global_tokens, global_indices, global_weights = nuwa_stage1(
            patch_features=features[:1],
            sim_features=similarity[:1],
            cls_attention=salience[:1],
            grid_h=crop_h,
            grid_w=crop_w,
            num_tokens=global_budget,
            top_n_per_region=_candidate_top_n(global_budget, crop_h, crop_w),
            distance_threshold=self._owner._distance_threshold,
            pillar_threshold_percentile=NEXT_PILLAR_PERCENTILE,
        )
        projected_global = self._inner.multi_modal_projector(global_tokens)[0]
        output_parts = [projected_global]
        benchmark_parts = [global_indices[0]]
        source_parts: list[torch.Tensor] = []
        if self._track_source:
            global_source = torch.zeros(
                global_budget, native_count, dtype=torch.float32
            )
            global_source[:, :patches] = (
                global_weights[0].detach().float().cpu()
            )
            source_parts.append(global_source)

        projected_local = projected_global.new_empty(
            (0, projected_global.shape[-1])
        )
        local_indices = torch.empty(
            0, dtype=torch.long, device=features.device
        )
        local_weights: torch.Tensor | None = None
        if local_budget:
            local_tokens, local_indices_batch, local_weights_batch = nuwa_stage1(
                patch_features=local_features,
                sim_features=local_similarity,
                cls_attention=local_salience,
                grid_h=local_h,
                grid_w=local_w,
                num_tokens=local_budget,
                top_n_per_region=_candidate_top_n(local_budget, local_h, local_w),
                distance_threshold=self._owner._distance_threshold,
                pillar_threshold_percentile=NEXT_PILLAR_PERCENTILE,
            )
            projected_local = self._inner.multi_modal_projector(local_tokens)[0]
            local_indices = local_indices_batch[0]
            if self._track_source:
                local_weights = (
                    local_weights_batch[0].detach().float().cpu()
                )

        if crop_count > 1:
            native_local_columns = (
                patches
                + torch.arange(local_capacity)
                + torch.arange(local_capacity) // local_w
            ).long()
            cursor = 0
            for row in range(local_h):
                count = int(((local_indices // local_w) == row).sum())
                if count:
                    output_parts.append(projected_local[cursor : cursor + count])
                    selected = local_indices[cursor : cursor + count]
                    benchmark_parts.append(
                        patches + selected + selected // local_w
                    )
                    if self._track_source and local_weights is not None:
                        selected_source = torch.zeros(
                            count, native_count, dtype=torch.float32
                        )
                        selected_source[:, native_local_columns] = local_weights[
                            cursor : cursor + count
                        ]
                        source_parts.append(selected_source)
                    cursor += count
                newline_position = patches + (row + 1) * local_w + row
                output_parts.append(self._inner.image_newline[None])
                benchmark_parts.append(
                    torch.tensor([newline_position], device=features.device)
                )
                if self._track_source:
                    newline_source = torch.zeros(
                        1, native_count, dtype=torch.float32
                    )
                    newline_source[0, newline_position] = 1.0
                    source_parts.append(newline_source)
            if cursor != local_budget:
                raise RuntimeError(
                    f"Nuwa placed {cursor} of {local_budget} selected local tokens"
                )

        output = torch.cat(output_parts, dim=0)
        benchmark = torch.cat(benchmark_parts).long()
        source = torch.cat(source_parts, dim=0) if self._track_source else None
        if int(output.shape[0]) != target or int(benchmark.numel()) != target:
            raise RuntimeError(
                f"Nuwa Stage 1 produced {output.shape[0]} rows/"
                f"{benchmark.numel()} positions, expected {target}."
            )
        return output, benchmark, source
