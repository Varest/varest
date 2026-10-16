"""Shared InternVL plumbing for training-free visual-token compressors.

InternVL's image processor dynamically splits one source image into several
``448 x 448`` tiles.  The vision tower sees those tiles as a batch, and
``InternVLModel.get_image_features`` returns that flat tile batch.  The
processor has already removed its ``num_patches`` metadata at that point, so
the InternVL backbone records the equivalent boundaries from the original
placeholder runs.  A compressor must regroup and flatten the tiles for each
source image before applying its policy.

This module centralises that contract.  Method modules continue to own their
selection or merging policy; the helpers below only:

* detect the Hugging Face InternVL layout structurally;
* reconstruct one concatenated visual sequence per source image;
* align per-tile ViT signals (attention, keys, hidden states) with InternVL's
  pixel-shuffled LLM tokens before concatenating them in the same order; and
* trim contiguous ``<IMG_CONTEXT>`` runs while retaining left-padding semantics.

Keeping this logic in one place prevents subtly different tile accounting from
turning nominally matched baseline budgets into per-tile budgets.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch


def is_internvl(vlm: Any) -> bool:
    """Return whether ``vlm`` exposes the pinned Transformers InternVL layout.

    LLaVA and InternVL both have ``vision_tower`` and
    ``multi_modal_projector``.  InternVL is distinguished by the pixel-shuffle
    step between them, which is also the architecture detail these adapters
    need to account for.
    """

    inner = getattr(getattr(vlm, "model", None), "model", None)
    return (
        getattr(inner, "vision_tower", None) is not None
        and getattr(inner, "multi_modal_projector", None) is not None
        and callable(getattr(inner, "pixel_shuffle", None))
    )


def image_token_id(vlm: Any, *, label: str = "InternVL compression") -> int:
    """Resolve InternVL's ``<IMG_CONTEXT>`` token id."""

    config = getattr(getattr(vlm, "model", None), "config", None)
    token_id = getattr(config, "image_token_id", None)
    if token_id is None:
        token_id = getattr(config, "image_token_index", None)
    if token_id is None:
        processor = getattr(vlm, "processor", None)
        token = getattr(processor, "image_token", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if token is not None and tokenizer is not None:
            token_id = tokenizer.convert_tokens_to_ids(str(token))
    if token_id is None:
        raise RuntimeError(f"Could not resolve the image placeholder token id for {label}.")
    return int(token_id)


def tokens_per_tile(vlm: Any) -> int:
    """Number of projected LLM tokens emitted by one InternVL tile."""

    config = getattr(getattr(vlm, "model", None), "config", None)
    count = int(getattr(config, "image_seq_length", 256))
    if count < 1:
        raise ValueError(f"InternVL image_seq_length must be >= 1, got {count}")
    return count


def recorded_tile_counts(vlm: Any) -> tuple[int, ...]:
    """Return source-image tile boundaries captured by the InternVL backbone."""

    counts = getattr(vlm, "tile_counts", ())
    if callable(counts):
        counts = counts()
    result = tuple(int(count) for count in counts)
    if not result:
        raise RuntimeError(
            "InternVL tile boundaries are unavailable. Call vlm.prepare_inputs() "
            "after installing the compressor and before invoking the model."
        )
    if any(count < 1 for count in result):
        raise ValueError(f"InternVL tile counts must all be positive, got {result}")
    return result


def image_sequences(
    image_outputs: Any,
    *,
    tile_counts: Sequence[int] | None = None,
    output_tokens_per_tile: int | None = None,
) -> list[torch.Tensor]:
    """Return one concatenated ``[tokens, hidden]`` tensor per source image.

    For InternVL, ``pooler_output`` is ``[sum(tiles), per_tile, hidden]``.
    ``tile_counts`` restores its source-image boundaries, after which every
    tile in a group is flattened in processor order. For backbones that already
    return a sequence per image (such as LLaVA), omit both grouping arguments.

    The result is deliberately a list: dynamic tiling makes source-image
    sequences legitimately ragged across a batch.
    """

    pooler = getattr(image_outputs, "pooler_output", None)
    if isinstance(pooler, torch.Tensor):
        if tile_counts is not None:
            counts = tuple(int(count) for count in tile_counts)
            if any(count < 1 for count in counts):
                raise ValueError(f"InternVL tile counts must all be positive, got {counts}")
            if output_tokens_per_tile is None:
                if pooler.ndim != 3:
                    raise ValueError(
                        "output_tokens_per_tile is required to regroup a flat InternVL tensor"
                    )
                output_tokens_per_tile = int(pooler.shape[1])
            per_tile = int(output_tokens_per_tile)
            if pooler.ndim == 3:
                if int(pooler.shape[1]) != per_tile:
                    raise ValueError(
                        f"InternVL emits {pooler.shape[1]} tokens per tile, expected {per_tile}"
                    )
                if sum(counts) != int(pooler.shape[0]):
                    raise ValueError(
                        f"InternVL captured tile counts {counts} (sum={sum(counts)}) but the "
                        f"vision tower returned {pooler.shape[0]} tiles"
                    )
                sequences = []
                cursor = 0
                for count in counts:
                    sequences.append(pooler[cursor : cursor + count].flatten(0, 1))
                    cursor += count
            elif pooler.ndim == 2:
                expected = sum(counts) * per_tile
                if int(pooler.shape[0]) != expected:
                    raise ValueError(
                        f"InternVL captured {sum(counts)} tiles x {per_tile} tokens but the "
                        f"flat feature tensor holds {pooler.shape[0]} rows"
                    )
                sequences = list(torch.split(pooler, [count * per_tile for count in counts]))
            else:
                raise ValueError(
                    f"InternVL pooler_output must be [tiles, tokens, hidden] or "
                    f"[all_tokens, hidden], got {tuple(pooler.shape)}"
                )
        elif pooler.ndim == 2:
            sequences = [pooler]
        elif pooler.ndim == 3 and pooler.shape[0] == 1:
            sequences = list(pooler.unbind(0))
        else:
            raise ValueError(
                "A multi-row [tiles, tokens, hidden] pooler_output is ambiguous without "
                "source-image tile_counts; refusing to compress tiles independently"
            )
    elif isinstance(pooler, Sequence):
        sequences = list(pooler)
    else:
        raise TypeError("get_image_features must return pooler_output as a tensor or sequence")

    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, torch.Tensor) or sequence.ndim != 2:
            shape = (
                tuple(sequence.shape)
                if isinstance(sequence, torch.Tensor)
                else type(sequence).__name__
            )
            raise ValueError(
                f"Image {index} must be a [tokens, hidden] tensor after tile "
                f"concatenation, got {shape}"
            )
    return sequences


def compress_image_sequences(
    image_outputs: Any,
    compress_one: Callable[[torch.Tensor, int], torch.Tensor],
    *,
    tile_counts: Sequence[int] | None = None,
    output_tokens_per_tile: int | None = None,
    flatten_output: bool = False,
) -> Any:
    """Apply ``compress_one`` once per concatenated source-image sequence."""

    compressed: list[torch.Tensor] = []
    sequences = image_sequences(
        image_outputs,
        tile_counts=tile_counts,
        output_tokens_per_tile=output_tokens_per_tile,
    )
    for index, sequence in enumerate(sequences):
        result = compress_one(sequence, index)
        if not isinstance(result, torch.Tensor) or result.ndim != 2:
            shape = (
                tuple(result.shape) if isinstance(result, torch.Tensor) else type(result).__name__
            )
            raise ValueError(
                f"InternVL compressor output for image {index} must be "
                f"[tokens, hidden], got {shape}"
            )
        if result.shape[-1] != sequence.shape[-1]:
            raise ValueError(
                f"InternVL compressor changed hidden width for image {index}: "
                f"{sequence.shape[-1]} -> {result.shape[-1]}"
            )
        compressed.append(result)
    if flatten_output:
        # InternVL's forward only checks/scatters the flattened feature count.
        # A singleton leading dimension keeps ragged batches representable while
        # retaining every image's sequence order.
        if not compressed:
            raise ValueError("Cannot flatten an empty InternVL image batch")
        image_outputs.pooler_output = torch.cat(compressed, dim=0).unsqueeze(0)
    else:
        image_outputs.pooler_output = compressed
    return image_outputs


def _square_side(count: int, *, label: str) -> int:
    side = int(count**0.5)
    if side * side != count:
        raise ValueError(f"{label} needs a square token grid, got {count} tokens")
    return side


def pool_tile_signal(
    signal: torch.Tensor,
    output_tokens_per_tile: int,
    *,
    has_cls: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """Align a per-patch ViT signal with InternVL's pixel-shuffled tile tokens.

    ``signal`` is ``[tiles, patches(+CLS), ...]``.  InternVL3.5 uses a 32x32
    patch grid and a 0.5 pixel shuffle, so every 2x2 patch block becomes one of
    the 16x16 LLM tokens.  The implementation derives the factor from the two
    token counts instead of baking those sizes in.
    """

    if not isinstance(signal, torch.Tensor) or signal.ndim < 2:
        raise ValueError("InternVL tile signal must have shape [tiles, tokens, ...]")
    patches = signal[:, 1:] if has_cls else signal
    patch_count = int(patches.shape[1])
    input_side = _square_side(patch_count, label="InternVL ViT signal")
    output_side = _square_side(int(output_tokens_per_tile), label="InternVL projected tile")
    if input_side % output_side:
        raise ValueError(
            f"InternVL cannot align a {input_side}x{input_side} ViT grid with a "
            f"{output_side}x{output_side} projected grid"
        )
    factor = input_side // output_side
    trailing = patches.shape[2:]
    view = patches.reshape(patches.shape[0], output_side, factor, output_side, factor, *trailing)
    reduce_dims = (2, 4)
    if reduction == "mean":
        pooled = view.mean(dim=reduce_dims)
    elif reduction == "sum":
        pooled = view.sum(dim=reduce_dims)
    else:
        raise ValueError(f"Unsupported InternVL tile-signal reduction: {reduction}")
    return pooled.reshape(patches.shape[0], output_tokens_per_tile, *trailing)


def concatenate_tile_signal(
    pooled_tiles: torch.Tensor,
    sequences: Sequence[torch.Tensor],
    output_tokens_per_tile: int,
) -> list[torch.Tensor]:
    """Group aligned tile signals using the same boundaries as image features."""

    groups: list[torch.Tensor] = []
    cursor = 0
    for image_index, sequence in enumerate(sequences):
        count = int(sequence.shape[0])
        if count % output_tokens_per_tile:
            raise ValueError(
                f"InternVL image {image_index} has {count} projected tokens, which is not "
                f"a multiple of the per-tile count {output_tokens_per_tile}"
            )
        tile_count = count // output_tokens_per_tile
        end = cursor + tile_count
        if end > pooled_tiles.shape[0]:
            raise ValueError("InternVL tile signals ended before all image features were grouped")
        group = pooled_tiles[cursor:end].reshape(count, *pooled_tiles.shape[2:])
        groups.append(group)
        cursor = end
    if cursor != pooled_tiles.shape[0]:
        raise ValueError(
            f"InternVL captured {pooled_tiles.shape[0]} tile signals but grouped only {cursor}"
        )
    return groups


def contiguous_runs(positions: torch.Tensor) -> list[torch.Tensor]:
    """Split ascending token positions into contiguous image-placeholder runs."""

    if positions.numel() == 0:
        return []
    splits = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=True)[0] + 1
    return list(torch.tensor_split(positions, splits.tolist()))


def trim_image_placeholders(
    inputs: Any,
    *,
    token_id: int,
    keep_for_run: Callable[[int], int],
    pad_token_id: int = 0,
) -> Any:
    """Trim each image run to its concatenate-first compression output length."""

    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        return inputs
    attention_mask = inputs.get("attention_mask")
    has_attention = (
        isinstance(attention_mask, torch.Tensor) and attention_mask.shape == input_ids.shape
    )
    id_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    changed = False

    for row_index, ids in enumerate(input_ids):
        keep = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
        positions = (ids == int(token_id)).nonzero(as_tuple=True)[0]
        for run in contiguous_runs(positions):
            wanted = max(1, min(int(run.numel()), int(keep_for_run(int(run.numel())))))
            if wanted < run.numel():
                keep[run[wanted:]] = False
                changed = True
        id_rows.append(ids[keep])
        if has_attention:
            mask_rows.append(attention_mask[row_index][keep])

    if not changed:
        return inputs
    inputs["input_ids"] = left_pad(id_rows, int(pad_token_id))
    if has_attention:
        inputs["attention_mask"] = left_pad(mask_rows, 0)
    return inputs


def left_pad(rows: Sequence[torch.Tensor], value: int) -> torch.Tensor:
    """Stack ragged 1-D tensors using the tokenizer's left-padding convention."""

    if not rows:
        raise ValueError("Cannot pad an empty batch")
    width = max(int(row.shape[0]) for row in rows)
    padded: list[torch.Tensor] = []
    for row in rows:
        missing = width - int(row.shape[0])
        if missing:
            prefix = torch.full((missing,), value, dtype=row.dtype, device=row.device)
            row = torch.cat((prefix, row), dim=0)
        padded.append(row)
    return torch.stack(padded, dim=0)


def tokenizer_pad_token_id(vlm: Any) -> int:
    tokenizer = getattr(getattr(vlm, "processor", None), "tokenizer", None)
    value = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    return int(value) if value is not None else 0


class InternVLProjectedFeatureAdapter:
    """Reusable lifecycle for methods that reduce projected InternVL features.

    Method-local ``*_internvl.py`` modules provide the actual compression and
    budget callbacks. This architecture helper owns only the common wiring:
    wrapping ``get_image_features``, regrouping dynamic tiles by source image,
    trimming the matching placeholder runs, and restoring both methods on
    uninstall.
    """

    def __init__(
        self,
        *,
        label: str,
        compress_one: Callable[[torch.Tensor, int], torch.Tensor],
        keep_for_run: Callable[[int], int],
    ) -> None:
        self._label = str(label)
        self._compress_one = compress_one
        self._keep_for_run = keep_for_run
        self._vlm: Any | None = None
        self._inner: Any | None = None
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None

    @staticmethod
    def supports(vlm: Any) -> bool:
        return is_internvl(vlm)

    def install(self, vlm: Any, *, wrap_prepare_inputs: bool = True) -> None:
        if self._vlm is not None:
            if self._vlm is vlm:
                return
            raise RuntimeError(
                f"{self._label}'s InternVL adapter is already installed on another backbone"
            )
        if not self.supports(vlm):
            raise TypeError(f"{self._label}'s InternVL adapter needs an InternVL backbone")

        inner = getattr(getattr(vlm, "model", None), "model", None)
        original = getattr(inner, "get_image_features", None)
        if not callable(original):
            raise TypeError(
                f"{self._label}'s InternVL adapter needs model.model.get_image_features"
            )

        self._vlm = vlm
        self._inner = inner
        self._original_get_image_features = original

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            image_outputs = original(*args, **kwargs)
            return compress_image_sequences(
                image_outputs,
                self._compress_one,
                tile_counts=recorded_tile_counts(vlm),
                output_tokens_per_tile=tokens_per_tile(vlm),
                flatten_output=True,
            )

        inner.get_image_features = wrapped_get_image_features

        if wrap_prepare_inputs:
            original_prepare = vlm.prepare_inputs
            self._original_prepare_inputs = original_prepare

            def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
                return self.trim_inputs(vlm, original_prepare(*args, **kwargs))

            vlm.prepare_inputs = wrapped_prepare_inputs

    def trim_inputs(self, vlm: Any, inputs: Any) -> Any:
        return trim_image_placeholders(
            inputs,
            token_id=image_token_id(vlm, label=self._label),
            keep_for_run=self._keep_for_run,
            pad_token_id=tokenizer_pad_token_id(vlm),
        )

    def uninstall(self) -> None:
        if self._inner is not None and self._original_get_image_features is not None:
            self._inner.get_image_features = self._original_get_image_features
        if self._vlm is not None and self._original_prepare_inputs is not None:
            self._vlm.prepare_inputs = self._original_prepare_inputs
        self._vlm = None
        self._inner = None
        self._original_get_image_features = None
        self._original_prepare_inputs = None
