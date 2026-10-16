"""InternVL adapter for VisionZip's attention-guided token reduction."""

from typing import Any

import torch

from llmoe.compressor.internvl import (
    concatenate_tile_signal,
    image_sequences,
    is_internvl,
    pool_tile_signal,
    recorded_tile_counts,
    tokens_per_tile,
)


class InternVLVisionZipAdapter:
    """Capture tile salience and compress one concatenated span per image."""

    def __init__(self, compressor: Any, vlm: Any) -> None:
        self._compressor = compressor
        self._vlm = vlm
        self._attn_handle: Any | None = None
        self._inner: Any | None = None
        self._original_get_image_features: Any | None = None

    @staticmethod
    def supports(vlm: Any) -> bool:
        return is_internvl(vlm)

    def install(self) -> None:
        layers = self._compressor._vision_encoder_layers(self._vlm)
        layer = self._compressor._resolve_feature_layer(self._vlm)
        target = layers[layer] if layer < 0 else layers[layer - 1]
        attention = getattr(target, "attention", None)
        if attention is None:
            raise TypeError("VisionZip needs InternVL vision layers exposing `attention`.")

        self._attn_handle = attention.register_forward_pre_hook(
            self._compressor._attn_pre_hook, with_kwargs=True
        )
        inner = self._vlm.model.model
        original = inner.get_image_features
        self._inner = inner
        self._original_get_image_features = original

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            self._compressor._cls_attention_sum = None
            self._compressor._metric_patches = None
            image_outputs = original(*args, **kwargs)
            return self.compress_outputs(inner, image_outputs, layer)

        inner.get_image_features = wrapped_get_image_features

    def uninstall(self) -> None:
        if self._attn_handle is not None:
            self._attn_handle.remove()
        if self._inner is not None and self._original_get_image_features is not None:
            self._inner.get_image_features = self._original_get_image_features
        self._attn_handle = None
        self._inner = None
        self._original_get_image_features = None

    def compress_outputs(
        self,
        inner: Any,
        image_outputs: Any,
        feature_layer: int,
    ) -> Any:
        """Pixel-shuffle tile signals, concatenate by image, then run VisionZip."""

        attention = self._compressor._cls_attention_sum
        metric = self._compressor._metric_patches
        if not isinstance(attention, torch.Tensor) or not isinstance(metric, torch.Tensor):
            raise RuntimeError(
                "VisionZip did not capture InternVL's CLS attention and key metric."
            )

        if feature_layer == -1:
            hidden = getattr(image_outputs, "last_hidden_state", None)
        else:
            hidden_states = getattr(image_outputs, "hidden_states", None)
            hidden = None if hidden_states is None else hidden_states[feature_layer]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise RuntimeError("VisionZip could not read InternVL's selected vision hidden state.")

        strategy = str(getattr(inner.config, "vision_feature_select_strategy", "default"))
        patches = hidden[:, 1:, :] if strategy == "default" else hidden
        patch_count = int(patches.shape[1])
        side = int(round(patch_count**0.5))
        if side * side != patch_count:
            raise ValueError(
                f"VisionZip needs a square InternVL patch grid, got {patch_count} tokens per tile."
            )

        shuffled = inner.pixel_shuffle(
            patches.reshape(patches.shape[0], side, side, patches.shape[-1]),
            scale_factor=inner.config.downsample_ratio,
        )
        tile_features = shuffled.reshape(shuffled.shape[0], -1, shuffled.shape[-1])
        per_tile = tokens_per_tile(self._vlm)
        if int(tile_features.shape[1]) != per_tile:
            raise RuntimeError(
                f"VisionZip pixel-shuffled {tile_features.shape[1]} tokens per tile, "
                f"but InternVL expects {per_tile}."
            )

        boundaries = image_sequences(
            image_outputs,
            tile_counts=recorded_tile_counts(self._vlm),
            output_tokens_per_tile=per_tile,
        )
        feature_groups = concatenate_tile_signal(tile_features, boundaries, per_tile)
        attention_groups = concatenate_tile_signal(
            pool_tile_signal(attention, per_tile, has_cls=False, reduction="sum"),
            boundaries,
            per_tile,
        )
        metric_groups = concatenate_tile_signal(
            pool_tile_signal(metric, per_tile, has_cls=False, reduction="mean"),
            boundaries,
            per_tile,
        )

        projected: list[torch.Tensor] = []
        first_source: torch.Tensor | None = None
        first_input_count = 0
        for index, features in enumerate(feature_groups):
            result = self._compressor.compress(
                features,
                attention=attention_groups[index],
                keys=metric_groups[index],
            )
            projected.append(
                inner.multi_modal_projector(result.tokens).to(boundaries[index].dtype)
            )
            if (
                index == 0
                and self._compressor._track_source
                and self._compressor._patch_source is not None
            ):
                first_source = self._compressor._patch_source[0].detach().cpu().float()
                first_input_count = int(features.shape[0])

        if first_source is not None:
            self._compressor._source_history = [
                torch.eye(first_input_count, dtype=torch.float32),
                first_source,
            ]
        self._compressor._patch_source = None
        self._compressor._cls_attention_sum = None
        self._compressor._metric_patches = None
        image_outputs.pooler_output = torch.cat(projected, dim=0).unsqueeze(0)
        return image_outputs
