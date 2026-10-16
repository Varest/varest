"""Shared LLaVA-NeXT plumbing for training-free token compressors.

LLaVA-NeXT's native Transformers path returns one already-packed ``[tokens,
hidden]`` sequence per source image.  Those sequences are legitimately ragged:
AnyRes chooses a different crop grid for each image, removes padded patches, and
adds learned image-newline rows before the language model sees the result.

Method adapters should not reimplement that lifecycle. This module owns the
backbone-specific mechanics shared by packed-feature and crop-aware compressors:

* identify the LLaVA-NeXT backbone without confusing it with LLaVA-1.5;
* resolve one canonical, per-image ``retain_ratio`` with floor semantics;
* expose native crop counts, AnyRes geometry, mosaic reconstruction, and unpadding;
* run a policy over either the packed sequence or raw per-crop features;
* trim each matching ``<image>`` run to the exact method output length; and
* restore both wrapped methods on uninstall.

The selector itself remains method-local. In particular, the packed selector does not
protect thumbnails or newline rows: released DivPrune and OTPrune LLaVA-1.6
paths rank the whole packed span.  Methods whose papers prescribe different
newline semantics reuse the crop-aware lifecycle with their own transform.

Generic sequence and placeholder operations are reused from ``internvl.py``.
They were deliberately written for both ragged packed sequences and dynamic
placeholder runs, so sharing them keeps the two dynamic-image backbones on one
well-tested implementation.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from llmoe.compressor._pruning import ratio_keep_count
from llmoe.compressor.internvl import (
    compress_image_sequences,
    contiguous_runs,
    image_sequences,
    image_token_id,
    left_pad,
    tokenizer_pad_token_id,
    trim_image_placeholders,
)

CompressOne = Callable[[torch.Tensor, int, int], torch.Tensor]
SelectIndices = Callable[[torch.Tensor, int], torch.Tensor]
SourceRecorder = Callable[[list[torch.Tensor]], None]
TransformOutputs = Callable[[Any, tuple[int, ...], tuple[Any, ...], dict[str, Any]], Any]
PreparedCallback = Callable[[Any, tuple[int, ...], tuple[int, ...]], None]


def is_llava_next(vlm: Any) -> bool:
    """Return whether ``vlm`` exposes the native Transformers NeXT layout."""

    if getattr(vlm, "compressor_family", None) == "llava_next":
        return True
    model = getattr(vlm, "model", None)
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "llava_next":
        return True
    inner = getattr(model, "model", None)
    return type(inner).__name__ == "LlavaNextModel"


def validate_retain_ratio(retain_ratio: float | None, *, label: str) -> float:
    """Validate and return NeXT's single public compression-strength control."""

    if retain_ratio is None:
        raise ValueError(
            f"{label} on LLaVA-NeXT requires retain_ratio; fixed token counts are "
            "not comparable across AnyRes images."
        )
    ratio = float(retain_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
    return ratio


def retained_tokens(token_count: int, retain_ratio: float) -> int:
    """Canonical NeXT budget: ``max(1, floor(N * retain_ratio))`` per image."""

    return ratio_keep_count(token_count, retain_ratio, label="LLaVA-NeXT compression")


def balanced_group_budgets(
    total: int,
    capacities: Sequence[int],
    *,
    minimum: int = 1,
    label: str = "LLaVA-NeXT compression",
) -> tuple[int, ...]:
    """Water-fill one exact image budget over crop-local patch groups.

    Crop-aware methods operate independently on the base crop and on each
    surviving region of the unpadded local mosaic. This allocator gives those
    groups deterministic, capacity-aware budgets while keeping the image-wide
    total exact. A method can raise ``minimum`` when its local reducer needs a
    fixed number of basis or representative rows.
    """

    target = int(total)
    floor = int(minimum)
    caps = tuple(int(capacity) for capacity in capacities)
    if floor < 0:
        raise ValueError(f"{label} minimum must be >= 0, got {minimum}")
    if not caps or any(capacity < floor for capacity in caps):
        raise ValueError(f"{label} needs non-empty group capacities >= {floor}, got {caps}.")
    required = floor * len(caps)
    if target < required:
        raise ValueError(
            f"{label} needs at least {required} rows across {len(caps)} groups "
            f"(minimum {floor} each), but this retain_ratio budgets {target}."
        )
    if target > sum(caps):
        raise ValueError(f"{label} cannot allocate {target} rows into group capacities {caps}.")

    budgets = [floor] * len(caps)
    remaining = target - required
    while remaining:
        active = [
            index
            for index, (budget, capacity) in enumerate(zip(budgets, caps, strict=True))
            if budget < capacity
        ]
        if not active:  # pragma: no cover - guarded by the capacity check above
            raise RuntimeError(f"{label} group-budget water filling exhausted its capacity")
        share = max(1, remaining // len(active))
        for index in active:
            addition = min(share, caps[index] - budgets[index], remaining)
            budgets[index] += addition
            remaining -= addition
            if not remaining:
                break
    return tuple(budgets)


def placeholder_run_lengths(inputs: Any, *, token_id: int) -> tuple[int, ...]:
    """Return image-placeholder lengths in batch-row and prompt order."""

    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return ()
    lengths: list[int] = []
    for row in input_ids:
        positions = (row == int(token_id)).nonzero(as_tuple=True)[0]
        lengths.extend(int(run.numel()) for run in contiguous_runs(positions))
    return tuple(lengths)


def trim_image_placeholders_to_lengths(
    inputs: Any,
    *,
    token_id: int,
    lengths: Sequence[int],
    pad_token_id: int = 0,
) -> Any:
    """Trim image runs to explicit per-image lengths in prompt order.

    Crop-aware methods can change the packing itself, so their output length is
    not necessarily a function of the native run length alone. Keeping this
    traversal beside placeholder_run_lengths gives every such adapter the same
    batch ordering and left-padding behavior.
    """

    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return inputs
    targets = tuple(int(length) for length in lengths)
    if any(length < 1 for length in targets):
        raise ValueError(f"LLaVA-NeXT image lengths must be positive, got {targets}")

    runs = [
        run
        for row in input_ids
        for run in contiguous_runs((row == int(token_id)).nonzero(as_tuple=True)[0])
    ]
    if len(runs) != len(targets):
        raise RuntimeError(
            f"LLaVA-NeXT found {len(runs)} image-placeholder runs but the "
            f"compressor produced {len(targets)} output lengths."
        )

    keep_rows: list[torch.Tensor] = []
    target_index = 0
    for ids in input_ids:
        keep = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
        positions = (ids == int(token_id)).nonzero(as_tuple=True)[0]
        for run in contiguous_runs(positions):
            wanted = targets[target_index]
            target_index += 1
            if wanted > int(run.numel()):
                raise ValueError(
                    f"LLaVA-NeXT compressor requested {wanted} rows from a native "
                    f"placeholder run of length {run.numel()}."
                )
            if wanted < int(run.numel()):
                keep[run[wanted:]] = False
        keep_rows.append(keep)

    sequence_fields = {
        key: value
        for key, value in list(inputs.items())
        if isinstance(value, torch.Tensor)
        and value.ndim == 2
        and tuple(value.shape) == tuple(input_ids.shape)
    }
    pad_values: dict[str, int | float | bool] = {
        "input_ids": int(pad_token_id),
        "attention_mask": 0,
        "labels": -100,
        "token_type_ids": 0,
        "position_ids": 0,
    }
    for key, value in sequence_fields.items():
        rows = [value[index][keep] for index, keep in enumerate(keep_rows)]
        inputs[key] = left_pad(rows, pad_values.get(key, 0))
    if "attention_mask" not in sequence_fields:
        inputs["attention_mask"] = left_pad(
            [
                torch.ones(int(keep.sum()), dtype=torch.long, device=keep.device)
                for keep in keep_rows
            ],
            0,
        )
    if "position_ids" in sequence_fields:
        mask = inputs["attention_mask"].bool()
        positions = torch.zeros_like(inputs["position_ids"])
        for row in range(positions.shape[0]):
            positions[row, mask[row]] = torch.arange(
                int(mask[row].sum()), device=positions.device, dtype=positions.dtype
            )
        inputs["position_ids"] = positions
    return inputs


def crop_counts(inner: Any, image_sizes: torch.Tensor) -> tuple[int, ...]:
    """Return native AnyRes crop counts (base image included) per source image."""

    if not isinstance(image_sizes, torch.Tensor) or image_sizes.ndim != 2:
        raise ValueError("LLaVA-NeXT crop-aware compression needs image_sizes shaped [images, 2]")
    from transformers.models.llava_next.modeling_llava_next import (
        image_size_to_num_patches,
    )

    config = inner.config
    return tuple(
        int(
            image_size_to_num_patches(
                image_size=size,
                grid_pinpoints=config.image_grid_pinpoints,
                patch_size=config.vision_config.image_size,
            )
        )
        for size in image_sizes
    )


def patch_grid(inner: Any) -> tuple[int, int]:
    """Return the regular CLIP patch grid used by every AnyRes crop."""

    vision = inner.config.vision_config
    image_size = int(vision.image_size)
    patch_size = int(vision.patch_size)
    if image_size % patch_size:
        raise ValueError(
            f"LLaVA-NeXT vision image_size={image_size} is not divisible by "
            f"patch_size={patch_size}."
        )
    side = image_size // patch_size
    return side, side


def anyres_grid_shape(
    inner: Any,
    image_size: torch.Tensor | Sequence[int],
) -> tuple[int, int]:
    """Return the processor-selected local-crop grid as (rows, columns)."""

    from transformers.models.llava_next.modeling_llava_next import (
        get_anyres_image_grid_shape,
    )

    height, width = get_anyres_image_grid_shape(
        image_size,
        inner.config.image_grid_pinpoints,
        inner.config.vision_config.image_size,
    )
    return int(height), int(width)


def _size_pair(image_size: torch.Tensor | Sequence[int]) -> tuple[int, int]:
    values = (
        image_size.detach().cpu().tolist()
        if isinstance(image_size, torch.Tensor)
        else image_size
    )
    if len(values) != 2:
        raise ValueError(f"Expected image size [height, width], got {values}")
    return int(values[0]), int(values[1])


def unpadded_spatial_shape(
    full_height: int,
    full_width: int,
    image_size: torch.Tensor | Sequence[int],
) -> tuple[int, int]:
    """Return Transformers' AnyRes unpadded height and width without allocating pixels."""

    original_height, original_width = _size_pair(image_size)
    original_ratio = original_width / original_height
    current_ratio = int(full_width) / int(full_height)
    if original_ratio > current_ratio:
        new_height = int(round(original_height * (full_width / original_width), 7))
        padding = (full_height - new_height) // 2
        return full_height - 2 * padding, full_width
    new_width = int(round(original_width * (full_height / original_height), 7))
    padding = (full_width - new_width) // 2
    return full_height, full_width - 2 * padding


@dataclass(frozen=True)
class LlavaNextGeometry:
    """Exact packed-sequence geometry for one source image."""

    crop_count: int
    patch_height: int
    patch_width: int
    tile_rows: int
    tile_columns: int
    local_height: int
    local_width: int
    newline_count: int
    native_tokens: int

    @property
    def patches_per_view(self) -> int:
        return self.patch_height * self.patch_width

    @property
    def local_patch_count(self) -> int:
        return self.local_height * self.local_width


def geometry_for_image(
    inner: Any,
    image_size: torch.Tensor | Sequence[int],
) -> LlavaNextGeometry:
    """Resolve crop, unpadding, newline, and native-token counts for one image."""

    patch_h, patch_w = patch_grid(inner)
    size_tensor = (
        image_size.unsqueeze(0)
        if isinstance(image_size, torch.Tensor) and image_size.ndim == 1
        else torch.as_tensor([_size_pair(image_size)])
    )
    count = crop_counts(inner, size_tensor)[0]
    if count == 1:
        return LlavaNextGeometry(
            crop_count=1,
            patch_height=patch_h,
            patch_width=patch_w,
            tile_rows=0,
            tile_columns=0,
            local_height=0,
            local_width=0,
            newline_count=1,
            native_tokens=patch_h * patch_w + 1,
        )
    tile_h, tile_w = anyres_grid_shape(inner, image_size)
    if count != 1 + tile_h * tile_w:
        raise RuntimeError(
            f"AnyRes selected {count} views but its local grid is {tile_h}x{tile_w}"
        )
    local_h, local_w = unpadded_spatial_shape(
        tile_h * patch_h,
        tile_w * patch_w,
        image_size,
    )
    return LlavaNextGeometry(
        crop_count=count,
        patch_height=patch_h,
        patch_width=patch_w,
        tile_rows=tile_h,
        tile_columns=tile_w,
        local_height=local_h,
        local_width=local_w,
        newline_count=local_h,
        native_tokens=patch_h * patch_w + local_h * local_w + local_h,
    )


def split_crop_tensor(
    tensor: torch.Tensor,
    counts: Sequence[int],
    *,
    label: str,
) -> list[torch.Tensor]:
    """Restore source-image boundaries on a concatenated vision-crop batch."""

    resolved = tuple(int(count) for count in counts)
    if tensor.ndim < 1 or sum(resolved) != int(tensor.shape[0]):
        leading = tensor.shape[0] if tensor.ndim else 0
        raise ValueError(
            f"{label} captured crop counts {resolved} (sum={sum(resolved)}) but "
            f"the vision tensor has leading size {leading}."
        )
    return list(torch.split(tensor, resolved, dim=0))


def stitch_and_unpad_local_crops(
    inner: Any,
    local_crops: torch.Tensor,
    image_size: torch.Tensor | Sequence[int],
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Rebuild the native local-crop mosaic and remove letterbox padding.

    local_crops is [tiles, patch_rows, ...] in processor order. The returned
    tensor is [1, unpadded_rows, ...] in the raster order used by Transformers
    immediately before it inserts image-newline rows.
    """

    if local_crops.ndim < 2:
        raise ValueError(
            f"Expected [tiles, patches, ...] local crops, got {tuple(local_crops.shape)}"
        )
    from transformers.models.llava_next.modeling_llava_next import unpad_image

    crop_h, crop_w = patch_grid(inner)
    if int(local_crops.shape[1]) != crop_h * crop_w:
        raise ValueError(
            f"LLaVA-NeXT local crops carry {local_crops.shape[1]} patches, expected "
            f"{crop_h}x{crop_w}={crop_h * crop_w}."
        )
    grid_h, grid_w = anyres_grid_shape(inner, image_size)
    if int(local_crops.shape[0]) != int(grid_h * grid_w):
        raise ValueError(
            f"LLaVA-NeXT AnyRes grid is {grid_h}x{grid_w}, but received "
            f"{local_crops.shape[0]} local crops."
        )

    trailing = local_crops.shape[2:]
    mosaic = local_crops.reshape(grid_h, grid_w, crop_h, crop_w, *trailing)
    trailing_dims = len(trailing)
    permute = tuple(range(4, 4 + trailing_dims)) + (0, 2, 1, 3)
    mosaic = mosaic.permute(permute).contiguous()
    mosaic = mosaic.flatten(-4, -3).flatten(-2, -1)
    if trailing_dims == 0:
        mosaic = mosaic.unsqueeze(0)
    elif trailing_dims > 1:
        mosaic = mosaic.reshape(-1, mosaic.shape[-2], mosaic.shape[-1])
    mosaic = unpad_image(mosaic, image_size)
    height, width = int(mosaic.shape[-2]), int(mosaic.shape[-1])
    rows = mosaic.flatten(1, 2).transpose(0, 1).unsqueeze(0)
    return rows, (height, width)


def decoder_visual_positions(
    input_ids: torch.Tensor,
    *,
    token_id: int,
    label: str,
) -> torch.Tensor | None:
    """Return the sole contiguous visual span for canonical decoder pruning."""

    if input_ids.ndim != 2:
        raise ValueError(
            f"{label}/LLaVA-NeXT expected [batch, sequence] input_ids, got {tuple(input_ids.shape)}"
        )
    if input_ids.shape[0] != 1:
        raise ValueError(
            f"{label}/LLaVA-NeXT requires batch_size=1 for physical decoder pruning; "
            f"got {input_ids.shape[0]}."
        )
    positions = (input_ids[0] == int(token_id)).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        return None
    runs = contiguous_runs(positions)
    if len(runs) != 1:
        raise NotImplementedError(
            f"{label}/LLaVA-NeXT supports exactly one contiguous image span per prompt"
        )
    return runs[0]


def layer_average_keep(
    token_count: int,
    *,
    retain_ratio: float,
    full_layers: int,
    decoder_layers: int,
    label: str,
) -> int:
    """Back-solve one post-prune count from a decoder-layer-average ratio."""

    count = int(token_count)
    depth = int(decoder_layers)
    shallow = int(full_layers)
    ratio = validate_retain_ratio(retain_ratio, label=label)
    if count < 1:
        raise ValueError(f"{label} cannot schedule an empty visual span")
    if not 0 <= shallow < depth:
        raise ValueError(
            f"{label} needs 0 <= full_layers < decoder_layers, got {shallow} and {depth}"
        )
    if ratio == 1.0:
        return count
    keep = int(count * (depth * ratio - shallow) / (depth - shallow))
    if keep < 1:
        floor = (shallow * count + (depth - shallow)) / depth / count
        raise ValueError(
            f"{label} retain_ratio={ratio:g} is unreachable for {count} tokens: "
            f"{shallow} full layer(s) plus one survivor require at least {floor:g}."
        )
    return min(count, keep)


def two_stage_layer_average_schedule(
    token_count: int,
    *,
    retain_ratio: float,
    stage2_layer: int,
    decoder_layers: int,
    stage2_ratio: float,
    label: str,
) -> tuple[int, int]:
    """Resolve a fixed-shape two-stage method from one layer-average ratio.

    ``stage2_ratio`` is part of the method, not a user-facing tuning knob. The
    first stage keeps ``S`` rows and the decoder boundary keeps
    ``max(1, floor(S * stage2_ratio))``. We choose the largest integer ``S``
    whose total decoder-layer token cost does not exceed the requested ratio.
    This gives Nuwa and ApET identical floor semantics without duplicating the
    binary-search and reachability checks in each method adapter.
    """

    native = int(token_count)
    shallow = int(stage2_layer)
    depth = int(decoder_layers)
    ratio = validate_retain_ratio(retain_ratio, label=label)
    deep_ratio = float(stage2_ratio)
    if native < 1 or not 0 < shallow < depth:
        raise ValueError(
            f"{label} cannot schedule native={native}, stage2_layer={shallow}, "
            f"decoder_layers={depth}."
        )
    if not 0.0 < deep_ratio <= 1.0:
        raise ValueError(f"{label} stage2_ratio must be in (0, 1], got {stage2_ratio}")
    if ratio == 1.0:
        return native, native

    target = math.floor(native * depth * ratio)

    def deep(stage1: int) -> int:
        return max(1, math.floor(stage1 * deep_ratio))

    maximum = shallow * native + (depth - shallow) * deep(native)
    if target > maximum:
        achieved = maximum / (depth * native)
        raise ValueError(
            f"{label} retain_ratio={ratio:g} is above the released two-stage "
            f"schedule's maximum {achieved:g}. Use <= {achieved:g}, or "
            "retain_ratio=1 for an identity pass-through."
        )
    minimum = shallow + (depth - shallow)
    if target < minimum:
        floor_ratio = minimum / (depth * native)
        raise ValueError(
            f"{label} retain_ratio={ratio:g} is too small for a non-empty "
            f"two-stage schedule; this image requires at least {floor_ratio:g}."
        )

    low, high = 1, native
    while low < high:
        middle = (low + high + 1) // 2
        cost = shallow * middle + (depth - shallow) * deep(middle)
        if cost <= target:
            low = middle
        else:
            high = middle - 1
    return low, deep(low)


class LlavaNextFeatureTransformAdapter:
    """Reusable lifecycle for methods replacing native packed image features.

    Crop-aware methods need the vision tower's raw hidden states and therefore
    cannot use LlavaNextProjectedFeatureAdapter. Their transform receives the
    original image outputs plus the exact target length of every source image;
    this class still owns ratio resolution, placeholder trimming, ragged-output
    validation, and reversible installation.
    """

    def __init__(
        self,
        *,
        label: str,
        retain_ratio: float,
        transform_outputs: TransformOutputs,
        target_for_length: Callable[[int], int] | None = None,
        on_prepared: PreparedCallback | None = None,
    ) -> None:
        self._label = str(label)
        self._retain_ratio = validate_retain_ratio(retain_ratio, label=self._label)
        self._transform_outputs = transform_outputs
        self._target_for_length = target_for_length
        self._on_prepared = on_prepared
        self._vlm: Any | None = None
        self._inner: Any | None = None
        self._image_token_id: int | None = None
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._pending_lengths: tuple[int, ...] | None = None

    @staticmethod
    def supports(vlm: Any) -> bool:
        return is_llava_next(vlm)

    def keep_for(self, token_count: int) -> int:
        if self._target_for_length is not None:
            keep = int(self._target_for_length(int(token_count)))
            if not 1 <= keep <= int(token_count):
                raise ValueError(
                    f"{self._label} resolved {keep} rows from a native "
                    f"LLaVA-NeXT span of {token_count}."
                )
            return keep
        return retained_tokens(token_count, self._retain_ratio)

    def install(self, vlm: Any) -> None:
        if self._vlm is not None:
            if self._vlm is vlm:
                return
            raise RuntimeError(
                f"{self._label}'s LLaVA-NeXT adapter is already installed on another backbone"
            )
        if not self.supports(vlm):
            raise TypeError(f"{self._label}'s crop-aware adapter needs llava-next-8b")

        inner = getattr(getattr(vlm, "model", None), "model", None)
        original = getattr(inner, "get_image_features", None)
        if not callable(original):
            raise TypeError(
                f"{self._label}'s LLaVA-NeXT adapter needs model.model.get_image_features"
            )

        self._vlm = vlm
        self._inner = inner
        self._image_token_id = image_token_id(vlm, label=self._label)
        self._original_get_image_features = original
        self._original_prepare_inputs = vlm.prepare_inputs

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            image_outputs = original(*args, **kwargs)
            native_lengths = tuple(
                int(sequence.shape[0]) for sequence in image_sequences(image_outputs)
            )
            expected = self._pending_lengths
            self._pending_lengths = None
            targets = (
                expected
                if expected is not None
                else tuple(self.keep_for(length) for length in native_lengths)
            )
            if len(targets) != len(native_lengths):
                raise RuntimeError(
                    f"{self._label} received {len(native_lengths)} LLaVA-NeXT "
                    f"images but has {len(targets)} target lengths."
                )
            transformed = self._transform_outputs(image_outputs, targets, args, kwargs)
            actual = tuple(int(sequence.shape[0]) for sequence in image_sequences(transformed))
            if actual != targets:
                raise RuntimeError(
                    f"{self._label} produced LLaVA-NeXT image lengths {actual}; "
                    f"retain_ratio={self._retain_ratio:g} requires {targets}."
                )
            return transformed

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            inputs = self._original_prepare_inputs(*args, **kwargs)
            native_lengths = placeholder_run_lengths(inputs, token_id=int(self._image_token_id))
            targets = tuple(self.keep_for(length) for length in native_lengths)
            self._pending_lengths = targets
            inputs = trim_image_placeholders_to_lengths(
                inputs,
                token_id=int(self._image_token_id),
                lengths=targets,
                pad_token_id=tokenizer_pad_token_id(vlm),
            )
            if self._on_prepared is not None:
                self._on_prepared(inputs, native_lengths, targets)
            return inputs

        inner.get_image_features = wrapped_get_image_features
        vlm.prepare_inputs = wrapped_prepare_inputs

    def uninstall(self) -> None:
        if self._inner is not None and self._original_get_image_features is not None:
            self._inner.get_image_features = self._original_get_image_features
        if self._vlm is not None and self._original_prepare_inputs is not None:
            self._vlm.prepare_inputs = self._original_prepare_inputs
        self._vlm = None
        self._inner = None
        self._image_token_id = None
        self._original_get_image_features = None
        self._original_prepare_inputs = None
        self._pending_lengths = None


class LlavaNextProjectedFeatureAdapter:
    """Reusable lifecycle for methods reducing packed NeXT image features.

    ``compress_one`` receives ``(packed_features, keep, image_index)``.  It must
    return exactly ``keep`` rows in the order the language model should consume
    them.  The helper intentionally resolves the count so every baseline uses
    identical retain-ratio semantics on this backbone.
    """

    def __init__(
        self,
        *,
        label: str,
        retain_ratio: float,
        compress_one: CompressOne,
    ) -> None:
        self._label = str(label)
        self._retain_ratio = validate_retain_ratio(retain_ratio, label=self._label)
        self._compress_one = compress_one
        self._vlm: Any | None = None
        self._inner: Any | None = None
        self._image_token_id: int | None = None
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._pending_keep_counts: tuple[int, ...] | None = None

    @staticmethod
    def supports(vlm: Any) -> bool:
        return is_llava_next(vlm)

    def keep_for(self, token_count: int) -> int:
        return retained_tokens(token_count, self._retain_ratio)

    def install(self, vlm: Any) -> None:
        if self._vlm is not None:
            if self._vlm is vlm:
                return
            raise RuntimeError(
                f"{self._label}'s LLaVA-NeXT adapter is already installed on another backbone"
            )
        if not self.supports(vlm):
            raise TypeError(f"{self._label}'s LLaVA-NeXT adapter needs llava-next-8b")

        inner = getattr(getattr(vlm, "model", None), "model", None)
        original = getattr(inner, "get_image_features", None)
        if not callable(original):
            raise TypeError(
                f"{self._label}'s LLaVA-NeXT adapter needs model.model.get_image_features"
            )

        self._vlm = vlm
        self._inner = inner
        self._image_token_id = image_token_id(vlm, label=self._label)
        self._original_get_image_features = original
        self._original_prepare_inputs = vlm.prepare_inputs

        def compress_one(sequence: torch.Tensor, image_index: int) -> torch.Tensor:
            keep = self.keep_for(int(sequence.shape[0]))
            result = self._compress_one(sequence, keep, image_index)
            if int(result.shape[0]) != keep:
                raise RuntimeError(
                    f"{self._label} returned {result.shape[0]} rows for LLaVA-NeXT "
                    f"image {image_index}; retain_ratio={self._retain_ratio:g} requires {keep}."
                )
            return result

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            image_outputs = original(*args, **kwargs)
            result = compress_image_sequences(image_outputs, compress_one)
            actual = tuple(int(features.shape[0]) for features in result.pooler_output)
            expected = self._pending_keep_counts
            self._pending_keep_counts = None
            if expected is not None and actual != expected:
                raise RuntimeError(
                    f"{self._label} compressed LLaVA-NeXT images to {actual}, but the "
                    f"matching placeholder runs were trimmed to {expected}."
                )
            return result

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            inputs = self._original_prepare_inputs(*args, **kwargs)
            run_lengths = placeholder_run_lengths(
                inputs,
                token_id=int(self._image_token_id),
            )
            self._pending_keep_counts = tuple(self.keep_for(length) for length in run_lengths)
            return trim_image_placeholders(
                inputs,
                token_id=int(self._image_token_id),
                keep_for_run=self.keep_for,
                pad_token_id=tokenizer_pad_token_id(vlm),
            )

        inner.get_image_features = wrapped_get_image_features
        vlm.prepare_inputs = wrapped_prepare_inputs

    def uninstall(self) -> None:
        if self._inner is not None and self._original_get_image_features is not None:
            self._inner.get_image_features = self._original_get_image_features
        if self._vlm is not None and self._original_prepare_inputs is not None:
            self._vlm.prepare_inputs = self._original_prepare_inputs
        self._vlm = None
        self._inner = None
        self._image_token_id = None
        self._original_get_image_features = None
        self._original_prepare_inputs = None
        self._pending_keep_counts = None


class LlavaNextPackedSelectionAdapter(LlavaNextProjectedFeatureAdapter):
    """Shared NeXT path for selectors operating over the whole packed span."""

    def __init__(
        self,
        *,
        label: str,
        retain_ratio: float,
        select_indices: SelectIndices,
        source_recorder: SourceRecorder | None = None,
    ) -> None:
        self._select_indices = select_indices
        self._source_recorder = source_recorder
        super().__init__(
            label=label,
            retain_ratio=retain_ratio,
            compress_one=self._select,
        )

    def _select(
        self,
        sequence: torch.Tensor,
        keep: int,
        image_index: int,
    ) -> torch.Tensor:
        indices = self._select_indices(sequence, keep)
        if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
            shape = (
                tuple(indices.shape)
                if isinstance(indices, torch.Tensor)
                else type(indices).__name__
            )
            raise TypeError(f"{self._label} selector must return one index vector, got {shape}")
        if indices.numel() != keep:
            raise RuntimeError(
                f"{self._label} selected {indices.numel()} LLaVA-NeXT rows; expected {keep}"
            )
        indices = indices.to(device=sequence.device, dtype=torch.long)
        if indices.unique().numel() != keep:
            raise RuntimeError(f"{self._label} returned duplicate LLaVA-NeXT indices")
        if int(indices.min()) < 0 or int(indices.max()) >= sequence.shape[0]:
            raise IndexError(
                f"{self._label} returned an index outside a {sequence.shape[0]}-token span"
            )

        if self._source_recorder is not None and image_index == 0:
            source = torch.zeros(keep, sequence.shape[0], dtype=torch.float32)
            source[torch.arange(keep), indices.detach().cpu()] = 1.0
            self._source_recorder([torch.eye(sequence.shape[0], dtype=torch.float32), source])
        return sequence.index_select(0, indices)
