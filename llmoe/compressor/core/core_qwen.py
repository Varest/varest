"""CORE's object-centric merge over Qwen2.5-VL's visual tokens.

CORE (arXiv:2511.14072) is normally a whole VLM: frozen ConvNeXt-L produces ``F_3``,
a frozen Mask2Former head produces object masks over the same grid, Eq. (1) merges
``F_3`` per object, and the merged tokens go to a Vicuna decoder. Running it "on
Qwen" could mean either keeping that vision stack and swapping the decoder, or
keeping *Qwen's* vision stack and applying CORE's merge to its tokens. This module
does the second, because the point of a baseline sweep is to compare compression
methods on one backbone: swapping the decoder would leave the visual features
non-Qwen and the numbers incomparable with the other four.

What transposes cleanly
-----------------------
Eq. (1) is a weighted average over a *flattened 2-D feature map*::

    t_n = sum_i w_{n,i} f_i / sum_i w_{n,i}      c_n = sum_i w_{n,i} i / sum_i w_{n,i}

with the masks living on the same ``H x W`` grid as the features, and the merged
tokens then sorted by centroid ``c_n``. Nothing in that is specific to ``F_3``: it
needs a raster grid of features and masks at the same resolution. Qwen's
post-merger tokens are exactly such a grid (``llm_grid_h x llm_grid_w``), so the
merge applies verbatim.

The resolution gap is the paper's own problem, solved the paper's own way: "the
probability masks are then down-sampled via an interpolation function to match the
resolution of ``F_3``". Here they are down-sampled to Qwen's token grid instead.

What does not transpose
-----------------------
The segmentation head still needs a real image, not Qwen's patch-flattened,
Qwen-normalised ``pixel_values``. The images are therefore captured on the way
through ``prepare_inputs`` and preprocessed separately for the frozen stack -- the
same thing CORE's own model does internally.

The frozen ConvNeXt-L + OMG-Seg weights are required for a real run; the merge
itself is pure tensor work and is exercised against the stub head.
"""

import torch

from llmoe.compressor.core.core import (
    MergedTokens,
    ObjectCentricMerger,
    _resize_to,
)

__all__ = ["merge_qwen_tokens", "centroid_grid_positions", "QwenCoreMerge"]


def merge_qwen_tokens(tokens: torch.Tensor, masks: torch.Tensor, *,
                      grid: tuple[int, int],
                      merger: ObjectCentricMerger) -> MergedTokens:
    """Apply CORE's Eq. (1) to one image's post-merger Qwen tokens.

    :param tokens: ``[T, C]`` Qwen visual tokens in raster order, ``T = H' * W'``.
    :param masks: ``[N, H, W]`` mask probabilities at *any* resolution; they are
        interpolated onto the token grid, which is the paper's own step.
    :param grid: ``(H', W')`` -- Qwen's post-merger token grid.
    :param merger: the shared object-centric merger.
    """
    height, width = (int(value) for value in grid)
    if tokens.ndim != 2 or tokens.shape[0] != height * width:
        raise ValueError(
            f"CORE/Qwen expected [{height * width}, C] tokens for a {height}x{width} "
            f"grid, got {tuple(tokens.shape)}"
        )
    if masks.ndim != 3:
        raise ValueError(f"masks must be [N, H, W], got {tuple(masks.shape)}")
    resized = _resize_to(masks.unsqueeze(0), (height, width)).squeeze(0)
    return merger.merge(tokens, resized.to(tokens.device))


def centroid_grid_positions(centroids: torch.Tensor, grid: tuple[int, int]
                            ) -> torch.Tensor:
    """Turn CORE's raster centroids into ``(h, w)`` cells for M-RoPE.

    A merged token is an *object*, not a surviving patch, so it has no grid cell of
    its own -- but Eq. (1) already gives it a position: the weighted-average raster
    index ``c_n``. Rounding that back to a cell keeps each object's coordinates
    where its mass actually is, which is also the order CORE sorts by.

    :param centroids: ``[K]`` raster indices from :class:`MergedTokens`.
    :param grid: ``(H', W')``.
    :return: ``[K, 2]`` integer ``(h, w)`` per token.
    """
    height, width = (int(value) for value in grid)
    index = centroids.detach().float().clamp(0, height * width - 1).round().long()
    return torch.stack([index // width, index % width], dim=-1)


class QwenCoreMerge:
    """Hold the merge configuration and the per-forward masks for one backbone."""

    def __init__(self, *, merger: ObjectCentricMerger) -> None:
        self.merger = merger
        #: Masks for the image currently being processed, ``[N, H, W]``.
        self.masks: torch.Tensor | None = None
        #: Centroids of the last merge, for the M-RoPE gather.
        self.centroids: torch.Tensor | None = None

    @property
    def token_count(self) -> int | None:
        """Tokens the merge will emit, when the rate makes that knowable."""
        if self.merger.rate == "fixed" and self.merger.fixed_tokens is not None:
            return int(self.merger.fixed_tokens)
        return None

    def apply(self, tokens: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """Merge one image's tokens with the masks captured for it."""
        if self.masks is None:
            raise RuntimeError(
                "CORE/Qwen has no object masks for this image; the segmentation "
                "stack did not run."
            )
        merged = merge_qwen_tokens(tokens, self.masks, grid=grid, merger=self.merger)
        self.centroids = merged.centroids
        return merged.tokens
