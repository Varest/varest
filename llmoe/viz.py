"""Token-compression visualization for the LLMOE pipeline.

The renderer draws one snapshot of a compressor's
`source` matrix on top of the image the ViT actually saw. `visualize` is the
orchestration entry point: it takes a backbone `model` (with a `TokenCompressor`
already installed by the caller), the `compressor` itself, and a `Config`
injected by the caller (the same `Config` the `llmoe.run` pipeline builds — it
must carry `config.benchmark` and, optionally, `config.dataset_limit`), runs the
first few benchmark images through the model, and writes one PNG per layer per
image showing how patches were regrouped or dropped. The forward is a full
multimodal pass, so it captures compression done in the vision tower as well as
compression done inside the LLM layers.

Outputs are written under `<config.output_dir>/<compressor.name>/image_<i>_<tag>/`
as `00_original.png` (the 336x336 canvas the ViT saw) plus per-layer PNGs.

This module also provides per-word attention saliency maps over the visual
tokens (`visualize_saliency`): for each whitespace word of the raw benchmark
question, the LLM's text->visual attention is rendered as a heatmap overlay on
the ViT canvas. Saliency works with or without a compressor installed; with a
compressor, saliency over compressed visual tokens is back-projected onto the
original patch grid via the compressor's source matrices. The
`SaliencyMethod` protocol keeps the door open for gradient-based methods
(Grad-CAM / attention-x-gradient relevancy) behind the same driver.
"""


import random
import re
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterator, List, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mipcandy import save_image
from scipy.ndimage import binary_erosion
from transformers import BatchFeature, PreTrainedModel

from llmoe.backbone import LLaVA, VLM
from llmoe.benchmarks import DATASET_REQUIRED_BY_BENCHMARK
from llmoe.compressor import TokenCompressor
from llmoe.compressor.prototype import VisualizationView
from llmoe.config import Config
from llmoe.resolvers import resolve_benchmark
from llmoe.visualization import pixel_values_to_canvas


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Equidistant random colormap with N entries (seeded for reproducibility)."""
    rng = random.Random(seed)
    return [(rng.random(), rng.random(), rng.random()) for _ in range(N)]

def make_visualization(
    image: torch.Tensor,
    source: torch.Tensor,
    patch_size: int = 14,
) -> torch.Tensor:
    """Render a compression snapshot onto `image`.

    Args:
        image: `[C, H, W]` tensor (float in [0, 1] or uint8 in [0, 255]). Must
            already be sized to `(ph * patch_size, pw * patch_size)` so its grid
            matches the ViT patch grid. For LLaVA-1.5 (CLIP-L/14 @ 336px) that
            is 336x336.
        source: `[T_current, T_original]` source matrix from one merge layer
            (or the pre-merge identity), for a single image (batch size 1).
            Row i records which original patches map to current token i.
        patch_size: ViT patch size in pixels. 14 for CLIP-L/14.

    Returns:
        A `[C, H, W]` float tensor in [0, 1] (ready for `mipcandy.save_image`).
    """
    img = image.detach().cpu().float()
    if img.ndim != 3 or img.shape[0] not in (1, 3):
        raise ValueError(f"image must be a [C, H, W] tensor with 1 or 3 channels, got {tuple(img.shape)}.")
    if img.shape[0] == 1:
        img = img.expand(3, -1, -1)
    if float(img.max()) > 1.0:
        img = img / 255.0
    img_arr = img.permute(1, 2, 0).contiguous().numpy()  # [H, W, 3] in [0, 1]

    src = source.detach().cpu()
    if src.ndim != 2:
        raise ValueError(
            f"source must be [T_current, T_original] (single image, no batch dim), "
            f"got shape {tuple(src.shape)}."
        )

    h, w, _ = img_arr.shape
    ph = h // patch_size
    pw = w // patch_size

    # Strip CLS only when present
    if src.shape[-1] == ph * pw + 1:
        src = src[:, 1:]
    elif src.shape[-1] != ph * pw:
        raise ValueError(
            f"source has {src.shape[-1]} original-patch columns but the image "
            f"yields {ph * pw} patches ({ph * pw + 1} with CLS) at patch_size={patch_size}."
        )
    # Infer merge vs prune from the source tensor.
    if _is_pruning(src):
        kept = (src.sum(dim=0) > 0).float()  # [T_original]
        kept_grid = kept.view(1, 1, ph, pw)
        kept_full = F.interpolate(kept_grid, size=(h, w), mode="nearest")
        kept_mask = kept_full.view(h, w, 1).numpy()
        vis_img = img_arr * kept_mask
    else: 
        # For each original patch, which current group does it most belong to?
        vis = src.argmax(dim=0)  # [T_original]
        num_groups = int(vis.max().item()) + 1

        cmap = generate_colormap(num_groups)
        vis_img = np.zeros_like(img_arr)

        for i in range(num_groups):
            mask = (vis == i).float().view(1, 1, ph, pw)
            mask = F.interpolate(mask, size=(h, w), mode="nearest")
            mask_np = mask.view(h, w, 1).numpy()
            denom = mask_np.sum()
            if denom < 1e-6:
                # No original patches landed in this group.
                continue
            color = (mask_np * img_arr).sum(axis=(0, 1)) / denom
            if not np.isfinite(color).all():
                color = np.zeros(3)
            mask_eroded = binary_erosion(mask_np[..., 0]).astype(mask_np.dtype)[..., None]
            mask_edge = mask_np - mask_eroded
            vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
            vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Back to a [C, H, W] float tensor in [0, 1] for mipcandy.save_image.
    return torch.from_numpy(np.clip(vis_img, 0.0, 1.0)).permute(2, 0, 1).contiguous().float()


def _is_pruning(source: torch.Tensor) -> bool:
    """Finds the renderer style straight from the source matrix.
    A source matrix reflects pruning iff every row has at most one nonzero entry. 
    """
    nnz_per_row = (source.detach() > 0).sum(dim=-1)
    return bool((nnz_per_row <= 1).all())


def _pixel_values_to_canvas(pixel_values: torch.Tensor, image_processor) -> torch.Tensor:
    """Backward-compatible alias for the shared normalization inverse."""

    return pixel_values_to_canvas(pixel_values, image_processor)


def _llava_next_view_strip(
    pixel_values: torch.Tensor,
    image_processor,
) -> torch.Tensor:
    """Render base then tile-major AnyRes views as one patch-aligned strip."""

    if pixel_values.ndim != 4 or pixel_values.shape[0] < 1:
        raise ValueError(
            "LLaVA-NeXT visualization expects [views, channels, height, width], "
            f"got {tuple(pixel_values.shape)}"
        )
    views = [
        _pixel_values_to_canvas(pixel_values[index], image_processor)
        for index in range(int(pixel_values.shape[0]))
    ]
    return torch.cat(views, dim=2)


def _llava_next_patch_source(
    source: torch.Tensor,
    *,
    view_count: int,
    patches_per_view: int,
) -> torch.Tensor:
    """Drop per-view CLS and learned-newline source columns for rendering."""

    if source.ndim != 2:
        raise ValueError(f"LLaVA-NeXT source must be 2-D, got {tuple(source.shape)}")
    rows_per_view = int(patches_per_view) + 1
    content_width = int(view_count) * rows_per_view
    if int(source.shape[1]) < content_width:
        raise ValueError(
            f"LLaVA-NeXT source has {source.shape[1]} columns; expected {content_width} "
            "view columns followed by any learned-newline source columns"
        )
    keep = torch.zeros(source.shape[1], dtype=torch.bool, device=source.device)
    keep[:content_width] = True
    keep[torch.arange(int(view_count), device=source.device) * rows_per_view] = False
    return source[:, keep]


def _source_for_visualization_view(
    source: torch.Tensor,
    view: VisualizationView,
    *,
    patch_size: int,
) -> torch.Tensor:
    """Map a compressor source matrix into one declared display canvas."""

    mapped = source
    if view.source_columns is not None:
        columns = view.source_columns.to(device=source.device, dtype=torch.long)
        if columns.numel() and int(columns.max()) >= int(source.shape[1]):
            raise RuntimeError(
                f"Visualization source column {int(columns.max())} exceeds width {source.shape[1]}."
            )
        mapped = source.index_select(1, columns)
    if view.canvas_columns is not None:
        canvas_columns = view.canvas_columns.to(device=source.device, dtype=torch.long)
        if int(canvas_columns.numel()) != int(mapped.shape[1]):
            raise RuntimeError("Visualization source and canvas column maps must have equal length.")
        height, width = view.canvas.shape[-2:]
        canvas_patches = (int(height) // int(patch_size)) * (int(width) // int(patch_size))
        if canvas_columns.numel() and int(canvas_columns.max()) >= canvas_patches:
            raise RuntimeError("Visualization canvas column exceeds the declared canvas patch grid.")
        expanded = source.new_zeros((source.shape[0], canvas_patches))
        expanded[:, canvas_columns] = mapped
        mapped = expanded
    if view.source_columns is not None:
        mapped = mapped[mapped.abs().sum(dim=1) > 0]
    return mapped


def visualize(model: VLM, compressor: TokenCompressor, config: Config) -> None:
    """Render per-layer token-compression visualizations for a benchmark.

    Standard source histories use the default LLaVA canvas. A compressor whose
    source columns follow another coordinate system declares generic
    visualization views through its compressor interface.
    """
    benchmark_type = resolve_benchmark(config.benchmark)
    config.dataset_name = DATASET_REQUIRED_BY_BENCHMARK[benchmark_type]
    benchmark = benchmark_type(config, type(model))
    limit = config.dataset_limit if config.dataset_limit is not None else 4

    patch_size = int(model.model.config.vision_config.patch_size)
    image_processor = model.processor.image_processor

    run_dir = Path(config.output_dir) / compressor.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing visualizations to {run_dir}")

    for b, row in enumerate(islice(benchmark.load_dataset(), limit)):
        img = row.get("image")
        if not isinstance(img, torch.Tensor):
            raise RuntimeError(f"Benchmark row {b} did not yield an image tensor.")
        tag = str(row.get("question_id") or row.get("item_id") or f"row_{b:03d}")
        prompt = str(row.get("question") or "")

        inputs = model.prepare_inputs([img], [prompt], pixel_dtype=model._dtype)
        pixel_values = inputs["pixel_values"]
        llava_next = getattr(model, "compressor_family", None) == "llava_next"

        with torch.inference_mode():
            model.model(**inputs)

        history = compressor.get_source_history()
        if not history:
            raise RuntimeError(
                "No source history captured — the compressor did not record per-layer "
                "source matrices during the encoder forward."
            )

        visual_views = compressor.get_visualization_views(model, inputs, image_processor)
        if visual_views is None:
            if llava_next:
                canvas = _llava_next_view_strip(pixel_values[0], image_processor)
                view_count = int(pixel_values.shape[1])
                patches_per_view = (
                    int(model.model.config.vision_config.image_size) // patch_size
                ) ** 2
                history = [
                    _llava_next_patch_source(
                        source,
                        view_count=view_count,
                        patches_per_view=patches_per_view,
                    )
                    for source in history
                ]
            else:
                canvas = _pixel_values_to_canvas(pixel_values[0], image_processor)
            visual_views = [VisualizationView("image", canvas)]
        if not visual_views:
            raise RuntimeError("Compressor returned no visualization views.")

        safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
        img_dir = run_dir / f"image_{b:03d}_{safe_tag}"
        img_dir.mkdir(parents=True, exist_ok=True)
        for view in visual_views:
            suffix = "" if view.name == "image" else f"_{view.name}"
            save_image(view.canvas, str(img_dir / f"00_original{suffix}.png"))

        for layer_idx, source in enumerate(history):
            label = "pre_merge" if layer_idx == 0 else f"layer_{layer_idx - 1:02d}"
            for view in visual_views:
                mapped = _source_for_visualization_view(
                    source,
                    view,
                    patch_size=patch_size,
                )
                vis = make_visualization(view.canvas, mapped, patch_size=patch_size)
                suffix = "" if view.name == "image" else f"_{view.name}"
                save_image(
                    vis,
                    str(img_dir / f"{layer_idx + 1:02d}_{label}{suffix}.png"),
                )
        print(
            f"  image {b}: {len(visual_views)} original view(s) + "
            f"{len(history) * len(visual_views)} snapshots -> {img_dir.name}"
        )

# ---------------------------------------------------------------------------
# Per-word attention saliency over visual tokens
# ---------------------------------------------------------------------------

# 10 evenly spaced anchors of matplotlib's viridis; linear interpolation to a
# 256-entry LUT is visually indistinguishable at print scale and avoids a
# matplotlib dependency (pyproject pins are frozen for cluster parity).
_VIRIDIS_ANCHORS: Tuple[Tuple[float, float, float], ...] = (
    (0.2667, 0.0039, 0.3294), (0.2824, 0.1569, 0.4706), (0.2431, 0.2863, 0.5373),
    (0.1922, 0.4078, 0.5569), (0.1490, 0.5098, 0.5569), (0.1216, 0.6196, 0.5373),
    (0.2078, 0.7176, 0.4745), (0.4275, 0.8039, 0.3490), (0.7059, 0.8706, 0.1725),
    (0.9922, 0.9059, 0.1451),
)

# Google's published 5th-degree polynomial fit of the turbo colormap.
_TURBO_COEFFS: Tuple[Tuple[float, ...], ...] = (
    (0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943),
    (0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604),
    (0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973),
)

_LUT_CACHE: dict[str, torch.Tensor] = {}


def _get_lut(cmap: str) -> torch.Tensor:
    """Return a `[256, 3]` float colormap LUT for `cmap` ("viridis" or "turbo")."""
    if cmap in _LUT_CACHE:
        return _LUT_CACHE[cmap]
    x = torch.linspace(0.0, 1.0, 256)
    if cmap == "viridis":
        anchors = torch.tensor(_VIRIDIS_ANCHORS)
        pos = x * (anchors.shape[0] - 1)
        lo = pos.floor().long().clamp(max=anchors.shape[0] - 2)
        frac = (pos - lo.float()).unsqueeze(-1)
        lut = anchors[lo] * (1 - frac) + anchors[lo + 1] * frac
    elif cmap == "turbo":
        channels = []
        for coeffs in _TURBO_COEFFS:
            powers = torch.stack([x ** k for k in range(len(coeffs))], dim=-1)
            channels.append(powers @ torch.tensor(coeffs))
        lut = torch.stack(channels, dim=-1).clamp(0, 1)
    else:
        raise ValueError(f"Unsupported colormap: {cmap!r}. Valid choices: viridis, turbo.")
    _LUT_CACHE[cmap] = lut.float()
    return _LUT_CACHE[cmap]


def make_saliency_visualization(
    canvas: torch.Tensor,
    heat_grid: torch.Tensor,
    *,
    alpha: float = 0.7,
    alpha_mode: str = "scaled",
    cmap: str = "viridis",
    vmax: float | None = None,
    gamma: float = 0.5,
) -> torch.Tensor:
    """Overlay a patch-grid saliency map onto the ViT canvas.

    Args:
        canvas: `[C, H, W]` float tensor in [0, 1] (the de-normalized ViT input).
        heat_grid: `[ph, pw]` non-negative saliency over the patch grid.
        alpha: peak overlay opacity.
        alpha_mode: "scaled" modulates opacity by the saliency itself so
            low-saliency regions stay photographic; "constant" uses `alpha`
            everywhere.
        cmap: "viridis" or "turbo".
        vmax: when given, maps are normalized by this shared maximum (with 0 as
            the floor) so maps are comparable across words; when `None`, each
            map is min-max normalized independently.
        gamma: contrast exponent applied to the normalized heat (`heat ** gamma`).
            Attention maps are peak-dominated, so values < 1 lift the mid-range
            and make the distribution visible; 1.0 disables the stretch.

    Returns:
        A `[C, H, W]` float tensor in [0, 1] (ready for `mipcandy.save_image`).
    """
    img = canvas.detach().cpu().float()
    if img.ndim != 3 or img.shape[0] not in (1, 3):
        raise ValueError(f"canvas must be a [C, H, W] tensor with 1 or 3 channels, got {tuple(img.shape)}.")
    if img.shape[0] == 1:
        img = img.expand(3, -1, -1)
    if alpha_mode not in ("scaled", "constant"):
        raise ValueError(f"Unsupported alpha_mode: {alpha_mode!r}. Valid choices: scaled, constant.")

    h, w = img.shape[1], img.shape[2]
    heat = heat_grid.detach().cpu().float()[None, None]
    heat = F.interpolate(heat, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    if vmax is None:
        lo, hi = heat.min(), heat.max()
        heat01 = (heat - lo) / (hi - lo).clamp_min(1e-12)
    else:
        heat01 = (heat / max(vmax, 1e-12)).clamp(0, 1)
    if gamma != 1.0:
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}.")
        heat01 = heat01 ** gamma

    lut = _get_lut(cmap)
    color = lut[(heat01 * 255).round().long().clamp(0, 255)].permute(2, 0, 1)  # [3, H, W]
    a = alpha * heat01 if alpha_mode == "scaled" else torch.full_like(heat01, alpha)
    return ((1 - a) * img + a * color).clamp(0, 1)


def _burn_label(image: torch.Tensor, text: str) -> torch.Tensor:
    """Burn a small text strip into the top of a `[C, H, W]` image tensor."""
    from PIL import Image, ImageDraw  # available transitively via torchvision

    arr = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 14), fill=(0, 0, 0))
    draw.text((2, 1), text, fill=(255, 255, 255))
    out = np.asarray(pil).astype(np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).contiguous()


def _attention_rollout(mats: Sequence[torch.Tensor]) -> torch.Tensor:
    """Attention rollout (Abnar & Zuidema) over per-layer `[L, L]` head-means.

    R = prod_l rownorm(A_l + I); the conventional 0.5 residual factor cancels
    under row normalization.
    """
    size = mats[0].shape[-1]
    eye = torch.eye(size, dtype=torch.float32)
    rollout = eye.clone()
    for attn in mats:
        a_hat = attn.float() + eye
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = a_hat @ rollout
    return rollout


def _project_to_patch_grid(saliency: torch.Tensor, source: torch.Tensor | None,
                           mode: str = "broadcast") -> torch.Tensor:
    """Map saliency over current (possibly compressed) visual tokens back to patches.

    Args:
        saliency: `[T_cur]` saliency over the visual tokens the LLM saw.
        source: `[T_cur, T_orig]` source matrix (CLS column already stripped),
            or `None` for the identity (no compressor).
        mode: "broadcast" assigns each original patch its argmax-owner token's
            saliency (consistent with `make_visualization`'s argmax rendering);
            "weighted" distributes via the row-normalized source transpose.

    Returns:
        `[T_orig]` saliency over the original patch grid.
    """
    if source is None:
        return saliency
    src = source.detach().cpu().float()
    if src.shape[0] != saliency.shape[0]:
        raise ValueError(
            f"source has {src.shape[0]} current-token rows but saliency has {saliency.shape[0]} entries."
        )
    if mode == "broadcast":
        owner = src.argmax(dim=0)  # [T_orig]
        out = saliency[owner]
        out = out * (src.sum(dim=0) > 0).float()  # pruned patches -> 0
        return out
    if mode == "weighted":
        weights = src / src.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return weights.t() @ saliency
    raise ValueError(f"Unsupported back_projection mode: {mode!r}. Valid choices: broadcast, weighted.")


@dataclass
class SaliencyMaps:
    """Per-word saliency over the visual tokens, per layer plus aggregates."""
    per_layer: torch.Tensor                # [n_layers, T_vis]
    aggregates: dict[str, torch.Tensor]    # e.g. {"mean": [T_vis], "rollout": [T_vis]}


class SaliencyMethod(Protocol):
    """A saliency backend: collects per-layer matrices, then scores words.

    `requires_grad` lets the driver pick `torch.no_grad()` vs
    `torch.enable_grad()`; attention saliency is grad-free, future
    Grad-CAM/relevancy methods are not.
    """
    name: str
    requires_grad: bool

    def collect(self, model: VLM, inputs: BatchFeature) -> list[torch.Tensor]: ...

    def compute(self, layer_mats: Sequence[torch.Tensor], query_positions: Sequence[int],
                visual_slice: slice, *, layer_range: tuple[int, int] | None = None) -> SaliencyMaps: ...


class AttentionSaliency:
    """Raw-attention saliency: head-mean text->visual attention per LLM layer.

    Per-layer maps are renormalized over the visual positions by default —
    causal mass also flows to the BOS sink and text tokens, so the conditional
    "where in the image does this word look, given it looks at the image" is
    the comparable quantity across words and layers.
    """

    name: str = "attention"
    requires_grad: bool = False

    def __init__(self, *, renormalize: bool = True,
                 aggregates: Sequence[str] = ("mean", "rollout")) -> None:
        unknown = set(aggregates) - {"mean", "rollout"}
        if unknown:
            raise ValueError(f"Unsupported aggregates: {sorted(unknown)}. Valid choices: mean, rollout.")
        self._renormalize: bool = renormalize
        self._aggregates: tuple[str, ...] = tuple(aggregates)

    def collect(self, model: VLM, inputs: BatchFeature) -> list[torch.Tensor]:
        """Run one forward and return one `[L, L]` float32 CPU head-mean per LLM layer."""
        n_layers = int(model.model.config.text_config.num_hidden_layers)
        with torch.no_grad():
            out = model.model(**inputs, output_attentions=True, use_cache=False)
        attentions = getattr(out, "attentions", None) or ()
        if len(attentions) != n_layers:
            raise RuntimeError(
                f"Got {len(attentions)} attention tensors, expected {n_layers}. The attention "
                f"implementation is not 'eager' (sdpa returns no weights and the capture is "
                f"silently empty) — check the eager switch and the installed transformers version."
            )
        return [attn[0].float().mean(dim=0).cpu() for attn in attentions]

    def _renorm(self, saliency: torch.Tensor) -> torch.Tensor:
        return saliency / saliency.sum().clamp_min(1e-12) if self._renormalize else saliency

    def compute(self, layer_mats: Sequence[torch.Tensor], query_positions: Sequence[int],
                visual_slice: slice, *, layer_range: tuple[int, int] | None = None) -> SaliencyMaps:
        """Score one word (its token positions `query_positions`) against the visual slice.

        `per_layer` always covers every layer; `layer_range` (half-open) only
        restricts the aggregates.
        """
        query = torch.as_tensor(list(query_positions), dtype=torch.long)
        if query.numel() == 0:
            raise ValueError("query_positions must be non-empty.")
        per_layer = torch.stack([
            self._renorm(attn[query][:, visual_slice].mean(dim=0)) for attn in layer_mats
        ])
        lo, hi = layer_range if layer_range is not None else (0, len(layer_mats))
        if not 0 <= lo < hi <= len(layer_mats):
            raise ValueError(f"layer_range {layer_range} out of bounds for {len(layer_mats)} layers.")
        aggregates: dict[str, torch.Tensor] = {}
        if "mean" in self._aggregates:
            aggregates["mean"] = self._renorm(per_layer[lo:hi].mean(dim=0))
        if "rollout" in self._aggregates:
            rollout = _attention_rollout(list(layer_mats)[lo:hi])
            aggregates["rollout"] = self._renorm(rollout[query][:, visual_slice].mean(dim=0))
        return SaliencyMaps(per_layer=per_layer, aggregates=aggregates)


@contextmanager
def _eager_attention(model: PreTrainedModel) -> Iterator[None]:
    """Temporarily switch every sub-model to eager attention so weights are returned.

    Under sdpa the per-layer `attn_weights` is `None` and the transformers
    output capture silently records nothing, so the switch is mandatory.
    Previous implementations are restored on exit.
    """
    config = model.config
    sub_configs = [getattr(config, name) for name in ("text_config", "vision_config")
                   if getattr(config, name, None) is not None]
    previous = [(cfg, getattr(cfg, "_attn_implementation", None)) for cfg in (config, *sub_configs)]
    set_impl = getattr(model, "set_attn_implementation", None)
    if callable(set_impl):
        set_impl("eager")
    else:
        for cfg, _ in previous:
            cfg._attn_implementation = "eager"
    try:
        yield
    finally:
        for cfg, impl in previous:
            if impl is not None:
                cfg._attn_implementation = impl


def _resolve_image_token_id(model: VLM) -> int:
    config = model.model.config
    token_id = getattr(config, "image_token_id", None)
    if token_id is None:
        token_id = getattr(config, "image_token_index", None)
    if token_id is None:
        token_id = getattr(model.processor, "image_token_id", None)
    if token_id is None:
        raise RuntimeError("Could not resolve the image token id from the model config or processor.")
    return int(token_id)


def _find_visual_positions(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Positions of the (contiguous) image-token block in a 1-D `input_ids`."""
    positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    if positions.numel() == 0:
        raise RuntimeError("No image tokens found in input_ids.")
    if not bool((positions.diff() == 1).all()):
        raise RuntimeError("Image-token positions are not contiguous; cannot map to the patch grid.")
    return positions


@dataclass
class _SequenceGeometry:
    """How the attention matrix relates to `input_ids` and the patch grid."""
    visual_slice: slice              # visual-token columns/rows in the attention matrix
    pos_shift: int                   # add to input_ids text positions to index attention
    source: torch.Tensor | None      # [T_cur, T_orig] back-projection, None = identity


def _resolve_sequence_geometry(l_ids: int, l_attn: int, visual_pos: torch.Tensor,
                               compressor: TokenCompressor | None, num_patches: int) -> _SequenceGeometry:
    """Reconcile attention size with `input_ids` under optional compression.

    Cases: (1) no compressor — identity over the placeholder slots; (2) the
    compressor shrank `input_ids` itself, slots == T_cur; (3) the compressor
    shrank the embeddings after tokenization, so attention is `delta` shorter
    than `input_ids` and text positions after the visual block shift left.
    """
    n_slots = int(visual_pos.numel())
    vis_start = int(visual_pos[0])
    if compressor is None:
        if l_attn != l_ids:
            raise RuntimeError(f"Attention length {l_attn} != input length {l_ids} without a compressor.")
        if n_slots != num_patches:
            raise RuntimeError(f"Found {n_slots} image-token slots but expected {num_patches} patches.")
        return _SequenceGeometry(slice(vis_start, vis_start + n_slots), 0, None)

    history = compressor.get_source_history()
    if not history:
        raise RuntimeError("Compressor installed but recorded no source history during the forward.")
    source = history[-1].detach().cpu().float()
    # Strip CLS only when present, mirroring make_visualization.
    if source.shape[-1] == num_patches + 1:
        source = source[:, 1:]
    elif source.shape[-1] != num_patches:
        raise RuntimeError(
            f"Final source matrix has {source.shape[-1]} original-token columns, expected "
            f"{num_patches} (or {num_patches + 1} with CLS)."
        )
    if l_attn > l_ids:
        raise RuntimeError(f"Attention length {l_attn} exceeds input length {l_ids}; unsupported geometry.")
    expected_t_cur = n_slots if l_attn == l_ids else n_slots - (l_ids - l_attn)
    # Vision-side compressors (e.g. CLS-attention scorers) often keep a CLS row
    # in their source matrix, but LLaVA's default select strategy drops that
    # token before the LLM. After the CLS column strip such a row carries no
    # patch mass, so it can be dropped without disturbing row alignment.
    # `make_visualization` tolerates the same convention (the row just wins no
    # patches in its argmax), which is why compression viz runs but this check
    # would otherwise trip.
    if source.shape[0] == expected_t_cur + 1 and float(source[0].abs().sum()) <= 1e-8:
        source = source[1:]
    t_cur = source.shape[0]
    if t_cur != expected_t_cur:
        raise RuntimeError(
            f"The source matrix has {t_cur} rows but the LLM received {expected_t_cur} visual "
            f"tokens ({n_slots} image slots, attention {l_attn} vs input_ids {l_ids}). The "
            f"compressor must record source rows only for the tokens the LLM receives, in patch "
            f"order (a leading zero-mass CLS row is dropped automatically)."
        )
    if l_attn == l_ids:
        return _SequenceGeometry(slice(vis_start, vis_start + t_cur), 0, source)
    return _SequenceGeometry(slice(vis_start, vis_start + t_cur), l_attn - l_ids, source)


def _split_words_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Whitespace words of `text` with their character spans."""
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _word_spans_by_offsets(processor, formatted_prompt: str, raw_question: str, input_ids: torch.Tensor,
                           n_image_tokens: int) -> list[tuple[str, list[int]]] | None:
    """Map question words to token positions via the fast tokenizer's offset mapping.

    Rebuilds the exact expanded string the tokenizer saw (the processor repeats
    the image token once per placeholder slot) and verifies the re-tokenization
    reproduces `input_ids` exactly; returns `None` when that ground truth cannot
    be established so the caller can fall back to prefix matching.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    image_token = getattr(processor, "image_token", None)
    if tokenizer is None or image_token is None:
        return None
    expanded = formatted_prompt.replace(image_token, image_token * n_image_tokens)
    try:
        enc = tokenizer(expanded, return_offsets_mapping=True)
    except (TypeError, ValueError, NotImplementedError):
        print("  offset path declined: tokenizer does not support offset mapping")
        return None
    if list(enc["input_ids"]) != input_ids.tolist():
        print("  offset path declined: re-tokenized prompt does not match input_ids")
        return None
    question_start = expanded.rfind(raw_question)
    if question_start < 0:
        print("  offset path declined: raw question not found verbatim in the prompt")
        return None
    words = _split_words_with_spans(raw_question)
    spans: list[tuple[str, list[int]]] = [(word, []) for word, _, _ in words]
    for position, (start, end) in enumerate(enc["offset_mapping"]):
        if end <= start:
            continue
        best, best_overlap = -1, 0
        for w, (_, w_start, w_end) in enumerate(words):
            overlap = min(end, question_start + w_end) - max(start, question_start + w_start)
            if overlap > best_overlap:
                best, best_overlap = w, overlap
        if best >= 0:
            spans[best][1].append(position)
    return spans


def _word_spans_by_prefix(tokenizer, raw_question: str, input_ids: torch.Tensor,
                          search_start: int) -> list[tuple[str, list[int]]]:
    """Fallback word->token mapping: growing-prefix tokenization + subsequence search.

    SentencePiece tokenization is context dependent at word boundaries, so
    per-word token counts come from diffing the lengths of growing word-prefix
    tokenizations; the question's token subsequence is then located inside
    `input_ids` after the image block (retrying without the first token for
    boundary merges).
    """
    words = [w for w, _, _ in _split_words_with_spans(raw_question)]
    if not words:
        raise RuntimeError("Raw question contains no words.")
    counts: list[int] = []
    previous = 0
    for k in range(1, len(words) + 1):
        ids = tokenizer(" ".join(words[:k]), add_special_tokens=False)["input_ids"]
        counts.append(len(ids) - previous)
        previous = len(ids)
    question_ids = tokenizer(" ".join(words), add_special_tokens=False)["input_ids"]
    haystack = input_ids.tolist()

    def find(needle: list[int]) -> int:
        for i in range(search_start, len(haystack) - len(needle) + 1):
            if haystack[i:i + len(needle)] == needle:
                return i
        return -1

    first_word_extra: list[int] = []
    start = find(question_ids)
    if start < 0 and len(question_ids) > 1:
        # The first token often tokenizes differently in context (e.g. "▁What"
        # standalone vs "What" after a newline); retry without it and treat the
        # token right before the match as the first word's in-context token.
        start = find(question_ids[1:])
        if start >= 0:
            counts[0] -= 1
            if start - 1 >= search_start:
                first_word_extra = [start - 1]
    if start < 0:
        raise RuntimeError(
            "Could not locate the question's token subsequence in input_ids; per-word "
            "saliency cannot be mapped for this prompt."
        )
    spans: list[tuple[str, list[int]]] = []
    cursor = start
    for index, (word, count) in enumerate(zip(words, counts)):
        count = max(count, 0)
        positions = list(range(cursor, cursor + count))
        if index == 0:
            positions = first_word_extra + positions
        spans.append((word, positions))
        cursor += count
    return spans


def _map_words_to_token_spans(model: VLM, formatted_prompt: str, raw_question: str,
                              input_ids: torch.Tensor, visual_pos: torch.Tensor) -> list[tuple[str, list[int]]]:
    """Per-word token positions in `input_ids` for the raw question words."""
    spans = _word_spans_by_offsets(model.processor, formatted_prompt, raw_question, input_ids,
                                   int(visual_pos.numel()))
    if spans is None:
        print("  offset-mapping path unavailable; falling back to prefix tokenization")
        spans = _word_spans_by_prefix(model.processor.tokenizer, raw_question, input_ids,
                                      int(visual_pos[-1]) + 1)
    return spans


def _resolve_raw_question(model: VLM, row: dict, prompt: str) -> str:
    """Pick the text whose words receive saliency maps.

    Benchmarks may rewrite the question they feed (e.g. MME strips its native
    "Please answer yes or no." and appends the backbone's short-answer suffix),
    so a benchmark-provided raw question is only trusted when it occurs
    verbatim in the fed prompt; otherwise the prompt minus any known backbone
    suffix is the raw question.
    """
    for key in ("question_raw", "question_mme"):
        candidate = str(row.get(key) or "").strip()
        if candidate and candidate in prompt:
            return candidate
    for suffix in (model.short_answer_suffix, model.option_letter_suffix, model.direct_answer_suffix):
        # format_*_prompt appends the suffix verbatim but also accepts prompts
        # already ending in its stripped form, so check both.
        for variant in (suffix, suffix.strip() if suffix else ""):
            if variant and prompt.endswith(variant):
                return prompt[: -len(variant)].rstrip()
    return prompt


def visualize_saliency(model: VLM, compressor: TokenCompressor | None, config: Config) -> None:
    """Render per-word attention saliency maps for a benchmark's first images.

    Mirrors `visualize`'s orchestration: `config` must carry `config.benchmark`
    and `config.dataset_limit` bounds how many images are processed (default 4).
    Unlike `visualize`, the `compressor` may be `None` (vanilla full-token
    baseline). Knobs are read from `config.custom_args["saliency_args"]`:
    `renormalize`, `aggregates`, `layers` ("all" | "none" | list), `layer_range`,
    `alpha`, `alpha_mode`, `gamma`, `cmap`, `scale` ("per_map" | "shared"),
    `back_projection`, `label`, `max_words`.

    Outputs land under `<output_dir>/<compressor.name or "full_token">/
    image_<i>_<tag>/saliency/<q_full | wNN_word>/` as per-layer and aggregate PNGs.
    """
    if not isinstance(model, LLaVA):
        raise NotImplementedError(
            f"Saliency visualization currently supports the LLaVA backbone only, got {type(model).__name__}."
        )
    args = dict(config.custom_args.get("saliency_args", {})) if config.custom_args else {}
    renormalize = bool(args.get("renormalize", True))
    aggregates = tuple(args.get("aggregates", ("mean", "rollout")))
    layer_range = tuple(args["layer_range"]) if args.get("layer_range") else None
    layers = args.get("layers", "all")
    alpha = float(args.get("alpha", 0.7))
    alpha_mode = str(args.get("alpha_mode", "scaled"))
    gamma = float(args.get("gamma", 0.5))
    cmap = str(args.get("cmap", "viridis"))
    scale = str(args.get("scale", "per_map"))
    back_projection = str(args.get("back_projection", "broadcast"))
    label = bool(args.get("label", False))
    max_words = int(args.get("max_words", 24))
    if scale not in ("per_map", "shared"):
        raise ValueError(f"Unsupported scale: {scale!r}. Valid choices: per_map, shared.")

    benchmark_type = resolve_benchmark(config.benchmark)
    config.dataset_name = DATASET_REQUIRED_BY_BENCHMARK[benchmark_type]
    benchmark = benchmark_type(config, type(model))
    limit = config.dataset_limit if config.dataset_limit is not None else 4

    patch_size = int(model.model.config.vision_config.patch_size)
    image_processor = model.processor.image_processor
    image_token_id = _resolve_image_token_id(model)
    method = AttentionSaliency(renormalize=renormalize, aggregates=aggregates)

    run_dir = Path(config.output_dir) / (compressor.name if compressor is not None else "full_token")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing saliency visualizations to {run_dir}")

    with _eager_attention(model.model):
        for b, row in enumerate(islice(benchmark.load_dataset(), limit)):
            img = row.get("image")
            if not isinstance(img, torch.Tensor):
                raise RuntimeError(f"Benchmark row {b} did not yield an image tensor.")
            tag = str(row.get("question_id") or row.get("item_id") or f"row_{b:03d}")
            prompt = str(row.get("question") or "")
            raw_question = _resolve_raw_question(model, row, prompt)

            inputs = model.prepare_inputs([img], [prompt], pixel_dtype=model._dtype)
            input_ids = inputs["input_ids"]
            if input_ids.shape[0] != 1:
                raise RuntimeError("Saliency visualization expects batch size 1 (padding shifts positions).")
            canvas = _pixel_values_to_canvas(inputs["pixel_values"][0], image_processor)
            h, w = canvas.shape[1], canvas.shape[2]
            ph, pw = h // patch_size, w // patch_size
            visual_pos = _find_visual_positions(input_ids[0], image_token_id)

            layer_mats = method.collect(model, inputs)
            if any(m.shape != layer_mats[0].shape for m in layer_mats):
                raise NotImplementedError(
                    "Per-layer attention shapes differ — saliency v1 supports pre-LLM compression "
                    "only; LLM-layer sequence-length changes need per-layer source remapping."
                )
            geometry = _resolve_sequence_geometry(int(input_ids.shape[1]), int(layer_mats[0].shape[-1]),
                                                  visual_pos, compressor, ph * pw)

            formatted_prompt = model.format_prompts([prompt])[0]
            word_spans = _map_words_to_token_spans(model, formatted_prompt, raw_question,
                                                   input_ids[0], visual_pos)
            question_positions = sorted({p for _, positions in word_spans for p in positions})
            if not question_positions:
                print(f"  image {b}: no question tokens located, skipping")
                continue
            subjects: list[tuple[str, list[int]]] = [("q_full", question_positions)]
            for w_idx, (word, positions) in enumerate(word_spans[:max_words], start=1):
                if not positions:
                    print(f"  image {b}: word {word!r} maps to no tokens, skipping")
                    continue
                safe_word = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in word) or "_"
                subjects.append((f"w{w_idx:02d}_{safe_word}", positions))
            skipped = max(0, len(word_spans) - max_words)
            if skipped:
                print(f"  image {b}: {skipped} word(s) beyond max_words={max_words} skipped")

            maps: dict[str, SaliencyMaps] = {}
            for name, positions in subjects:
                shifted = [p + geometry.pos_shift for p in positions]
                maps[name] = method.compute(layer_mats, shifted, geometry.visual_slice,
                                            layer_range=layer_range)

            def to_grid(saliency: torch.Tensor) -> torch.Tensor:
                return _project_to_patch_grid(saliency, geometry.source, back_projection).view(ph, pw)

            vmax: float | None = None
            if scale == "shared":
                peak = 0.0
                for sal_maps in maps.values():
                    for agg in sal_maps.aggregates.values():
                        peak = max(peak, float(to_grid(agg).max()))
                    for layer in sal_maps.per_layer:
                        peak = max(peak, float(to_grid(layer).max()))
                vmax = peak

            safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
            saliency_dir = run_dir / f"image_{b:03d}_{safe_tag}" / "saliency"
            saliency_dir.mkdir(parents=True, exist_ok=True)
            save_image(canvas, str(saliency_dir / "00_original.png"))

            n_files = 0
            if layers == "all":
                layer_indices = list(range(len(layer_mats)))
            elif layers == "none":
                layer_indices = []
            else:
                layer_indices = [int(i) for i in layers]
            for name, sal_maps in maps.items():
                subject_dir = saliency_dir / name
                subject_dir.mkdir(parents=True, exist_ok=True)
                word_text = raw_question if name == "q_full" else name.split("_", 1)[1]
                for agg_name, agg in sal_maps.aggregates.items():
                    vis = make_saliency_visualization(canvas, to_grid(agg), alpha=alpha,
                                                      alpha_mode=alpha_mode, cmap=cmap, vmax=vmax,
                                                      gamma=gamma)
                    if label:
                        vis = _burn_label(vis, f"{word_text} | {agg_name}")
                    save_image(vis, str(subject_dir / f"agg_{agg_name}.png"))
                    n_files += 1
                for layer_idx in layer_indices:
                    vis = make_saliency_visualization(canvas, to_grid(sal_maps.per_layer[layer_idx]),
                                                      alpha=alpha, alpha_mode=alpha_mode, cmap=cmap,
                                                      vmax=vmax, gamma=gamma)
                    if label:
                        vis = _burn_label(vis, f"{word_text} | layer {layer_idx}")
                    save_image(vis, str(subject_dir / f"layer_{layer_idx:02d}.png"))
                    n_files += 1
            print(f"  image {b}: {len(subjects)} subjects, {n_files} maps -> {saliency_dir.parent.name}")
