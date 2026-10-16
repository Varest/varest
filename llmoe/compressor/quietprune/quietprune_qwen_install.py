"""Install QuietPrune's in-ViT schedule onto a Qwen2.5-VL backbone.

Pruning inside the tower means the decoder receives fewer visual tokens than the
prompt was built for, so three things follow that a post-merger reduction never
has to deal with:

* **Placeholders.** ``<|image_pad|>`` runs must be trimmed to the surviving count
  before the forward, or the strict feature/placeholder check rejects the batch.
  The count is a pure function of the grid and ``R`` (:func:`groups_kept_after_schedule`),
  so it is known without looking at the image.
* **M-RoPE.** Survivors must keep the ``(t, h, w)`` of the groups they came from.
  ``get_rope_index`` lays the grid mesh over the placeholder run, so it is gathered
  at the survivor ids instead.
* **Ordering.** ``get_rope_index`` runs *before* the vision tower during
  ``generate``, so on that first call the survivors are not known yet. The tower is
  therefore run early and memoised, exactly as the DUET Qwen port has to.

One image per prompt, matching the runner and the other Qwen adapters.
"""

from typing import Any

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor.quietprune.quietprune_qwen_forward import (
    QuietPruneVisionRunner,
    groups_kept_after_schedule,
)
from llmoe.compressor.quietprune.quietprune_qwen_qcls import QwenQCLSAdapter
from llmoe.compressor.quietprune.quietprune_qwen_vit import quietprune_layers

__all__ = ["QwenQuietPruneInstaller"]


def _patch(owner: Any, name: str, factory: Any, undo: list[Any]) -> None:
    """Replace an attribute and record a clean undo (delete the shadow if new)."""
    original = getattr(owner, name)
    had_own = name in vars(owner)
    setattr(owner, name, factory(original))

    def restore() -> None:
        if had_own:
            setattr(owner, name, original)
            return
        try:
            delattr(owner, name)
        except AttributeError:  # pragma: no cover - defensive
            setattr(owner, name, original)

    undo.append(restore)


class QwenQuietPruneInstaller:
    """Wire the in-ViT schedule, the placeholder trim and the M-RoPE gather."""

    def __init__(self, *, rate: float, aggregate: bool, normalize_aggregation: bool,
                 eps: float = 1e-6) -> None:
        self.rate = float(rate)
        self.aggregate = bool(aggregate)
        self.normalize_aggregation = bool(normalize_aggregation)
        self.eps = float(eps)
        self.adapter: QwenQCLSAdapter | None = None
        self._runner: QuietPruneVisionRunner | None = None
        self._undo: list[Any] = []
        self._hooks: list[Any] = []
        self._image_token_id: int | None = None
        self._prompt_embeds: torch.Tensor | None = None
        self._prompt_mask: torch.Tensor | None = None
        self._primed: tuple | None = None
        self._vlm: VLM | None = None
        #: The owning compressor, consulted for ``bypass()``. Self-distillation runs
        #: the teacher as the same weights *without* pruning, so the tower and the
        #: placeholder trim have to lift together or the two disagree on length.
        self.compressor: Any | None = None

    @property
    def bypassed(self) -> bool:
        return bool(getattr(self.compressor, "_bypass", False))

    # ------------------------------------------------------------------ install

    def install(self, vlm: VLM) -> None:
        inner = vlm.model.model
        visual = inner.visual
        self._vlm = vlm
        self._image_token_id = int(vlm.model.config.image_token_id)
        self.adapter = QwenQCLSAdapter.from_visual(visual)
        # fp32, never the tower's dtype: this is the only trained module, and
        # torch.amp's GradScaler refuses to unscale fp16 gradients. The LLaVA path
        # pins the adapter to fp32 for the same reason; autocast still runs the
        # forward in half precision, and `_query` casts the result back.
        self.adapter.to(device=next(visual.parameters()).device, dtype=torch.float32)
        self._runner = QuietPruneVisionRunner(
            visual,
            pruning_layers=quietprune_layers(len(visual.blocks)),
            rate=self.rate,
            aggregate=self.aggregate,
            normalize_aggregation=self.normalize_aggregation,
            eps=self.eps,
        )
        # The prompt is needed to build [Q-CLS], and it is only available on the
        # way into the model -- before the tower runs.
        self._hooks.append(
            inner.register_forward_pre_hook(self._capture_prompt, with_kwargs=True))
        self._patch_visual(inner)
        self._patch_rope_index(inner)
        self._wrap_prepare_inputs(vlm)

    def uninstall(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        for restore in reversed(self._undo):
            restore()
        self._undo = []
        self._runner = None
        self._prompt_embeds = None
        self._prompt_mask = None
        self._primed = None
        self._vlm = None

    # ------------------------------------------------------------------ [Q-CLS]

    def _capture_prompt(self, module: Any, args: tuple, kwargs: dict) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        self._ensure_prompt(module, input_ids, kwargs.get("attention_mask"))
        return None

    def _ensure_prompt(self, inner: Any, input_ids: Any, attention_mask: Any) -> None:
        """Build ``[Q-CLS]``'s inputs from a prefill's ``input_ids``.

        Called from the model pre-hook and *also* from ``get_rope_index``: during
        ``generate`` the positions are laid out before ``inner.forward`` runs, so
        the hook has not fired yet and the tower would otherwise be primed with no
        query -- silently returning unpruned features against trimmed placeholders.
        """
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return
        if input_ids.shape[1] <= 1:      # a decode step; keep the prefill's query
            return
        # The placeholders carry no query information, and neither does padding.
        mask = (input_ids != self._image_token_id).float()
        if isinstance(attention_mask, torch.Tensor) and attention_mask.shape == input_ids.shape:
            mask = mask * attention_mask.float()
        self._prompt_embeds = inner.get_input_embeddings()(input_ids)
        self._prompt_mask = mask

    def _query(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """``[1, context_dim]``: the adapter's group-space query, folded to a patch.

        The adapter inverts ``merger.mlp``, so it lands in the merger's *input*
        width -- the concatenation of a ``2x2`` block. Averaging those four patch
        slots gives a query in single-patch space, which is what the ViT's keys
        live in.
        """
        if self.adapter is None:
            raise RuntimeError("QuietPrune/Qwen has no adapter installed.")
        if self._prompt_embeds is None:
            raise RuntimeError("QuietPrune/Qwen captured no prompt for [Q-CLS].")
        group = self.adapter(self._prompt_embeds.to(self.adapter.body[0].weight.dtype),
                             self._prompt_mask)
        patch_dim = self._runner.visual.blocks[0].attn.dim
        folded = group.reshape(group.shape[0], -1, patch_dim).mean(dim=1)
        return folded[:1].to(device=device, dtype=dtype)

    # ------------------------------------------------------------------- tower

    def _patch_visual(self, inner: Any) -> None:
        """Replace ``get_image_features``, not ``visual.forward``.

        The model splits the tower's output with sizes derived from ``grid_thw``
        (``prod // merge**2``), which is the *unpruned* count -- so returning fewer
        tokens from ``visual.forward`` alone fails in ``torch.split`` before the
        decoder is ever reached. Owning the whole call sidesteps that: the split is
        over what the tower actually produced.
        """

        def factory(original: Any) -> Any:
            def patched(pixel_values: torch.Tensor,
                        image_grid_thw: torch.Tensor | None = None,
                        **kwargs: Any) -> Any:
                if self.bypassed or self._prompt_embeds is None or image_grid_thw is None:
                    return original(pixel_values, image_grid_thw=image_grid_thw, **kwargs)
                primed = self._take_primed(pixel_values)
                if primed is not None:
                    return primed
                from transformers.modeling_outputs import BaseModelOutputWithPooling

                query = self._query(pixel_values.dtype, pixel_values.device)
                merged = self._runner.forward(
                    pixel_values.type(self._runner.visual.dtype), image_grid_thw, query)
                return BaseModelOutputWithPooling(
                    last_hidden_state=merged, pooler_output=(merged,))

            return patched

        _patch(inner, "get_image_features", factory, self._undo)

    def _take_primed(self, pixel_values: torch.Tensor) -> Any:
        """Serve (once) the features computed early by :meth:`_prime`."""
        primed = self._primed
        if primed is None:
            return None
        self._primed = None
        fingerprint, outputs = primed
        current = (pixel_values.data_ptr(), tuple(pixel_values.shape))
        return outputs if fingerprint == current else None

    # ------------------------------------------------------------- placeholders

    def surviving_tokens(self, grid_thw: torch.Tensor) -> int:
        """Visual tokens the decoder will receive, before the image is seen."""
        merge = int(self._runner.visual.spatial_merge_size)
        temporal, height, width = (int(v) for v in grid_thw[0].tolist())
        groups = temporal * (height // merge) * (width // merge)
        kept = groups_kept_after_schedule(
            groups, self.rate, len(self._runner.pruning_layers))[-1]
        return kept + (1 if self.aggregate and kept < groups else 0)

    def _wrap_prepare_inputs(self, vlm: VLM) -> None:
        def factory(original: Any) -> Any:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                return self._trim(original(*args, **kwargs))

            return wrapped

        _patch(vlm, "prepare_inputs", factory, self._undo)

    def _trim(self, inputs: Any) -> Any:
        grid = inputs.get("image_grid_thw") if hasattr(inputs, "get") else None
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if not isinstance(grid, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
            return inputs
        target = self.surviving_tokens(grid)
        companions = [name for name in ("attention_mask", "mm_token_type_ids")
                      if isinstance(inputs.get(name), torch.Tensor)
                      and inputs[name].shape == input_ids.shape]
        rows: dict[str, list[torch.Tensor]] = {"input_ids": []}
        rows.update({name: [] for name in companions})
        changed = False
        for row in range(input_ids.shape[0]):
            ids = input_ids[row]
            positions = (ids == self._image_token_id).nonzero(as_tuple=True)[0]
            if positions.numel() <= target:
                rows["input_ids"].append(ids)
                for name in companions:
                    rows[name].append(inputs[name][row])
                continue
            changed = True
            keep = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
            keep[positions[target:]] = False
            rows["input_ids"].append(ids[keep])
            for name in companions:
                rows[name].append(inputs[name][row][keep])
        if not changed:
            return inputs
        for name, values in rows.items():
            inputs[name] = torch.stack(values, dim=0)
        return inputs

    # ------------------------------------------------------------------- M-RoPE

    def _patch_rope_index(self, inner: Any) -> None:
        installer = self

        def factory(original: Any) -> Any:
            def patched(input_ids: torch.Tensor, mm_token_type_ids: torch.Tensor,
                        image_grid_thw: torch.Tensor | None = None,
                        video_grid_thw: torch.Tensor | None = None,
                        second_per_grid_ts: torch.Tensor | None = None,
                        attention_mask: torch.Tensor | None = None,
                        **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
                if installer.bypassed or image_grid_thw is None or video_grid_thw is not None:
                    return original(input_ids, mm_token_type_ids,
                                    image_grid_thw=image_grid_thw,
                                    video_grid_thw=video_grid_thw,
                                    second_per_grid_ts=second_per_grid_ts,
                                    attention_mask=attention_mask, **kwargs)
                installer._ensure_prompt(inner, input_ids, attention_mask)
                installer._prime(inner, image_grid_thw, kwargs.get("pixel_values"))
                return installer._rope_index(inner, input_ids, mm_token_type_ids,
                                             image_grid_thw, attention_mask)

            return patched

        _patch(inner, "get_rope_index", factory, self._undo)

    def _prime(self, inner: Any, grid: torch.Tensor,
               pixel_values: torch.Tensor | None) -> None:
        """Run the tower early so the survivors exist before positions are laid out."""
        if self._runner is not None and self._runner.survivor_groups is not None:
            return
        if not isinstance(pixel_values, torch.Tensor):
            return
        self._primed = None
        outputs = inner.get_image_features(pixel_values, image_grid_thw=grid)
        self._primed = ((pixel_values.data_ptr(), tuple(pixel_values.shape)), outputs)

    def _rope_index(self, inner: Any, input_ids: torch.Tensor,
                    mm_token_type_ids: torch.Tensor, grid: torch.Tensor,
                    attention_mask: torch.Tensor | None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Give each survivor the grid coordinates of the group it came from."""
        import itertools

        merge = inner.config.vision_config.spatial_merge_size
        device = input_ids.device
        positions = torch.zeros(3, input_ids.shape[0], input_ids.shape[1],
                                dtype=input_ids.dtype, device=device)
        deltas: list[torch.Tensor] = []
        survivors = self._runner.survivor_groups if self._runner is not None else None

        for row in range(input_ids.shape[0]):
            row_ids, row_types = input_ids[row], mm_token_type_ids[row]
            if attention_mask is not None:
                row_ids = row_ids[attention_mask[row].bool()]
                row_types = row_types[attention_mask[row].bool()]
            current, pieces = 0, []
            for modality, group in itertools.groupby(enumerate(row_types.tolist()),
                                                     lambda item: item[1]):
                members = list(group)
                length = members[-1][0] - members[0][0] + 1
                if modality == 0:
                    pieces.append(torch.arange(length, device=device)
                                  .view(1, -1).expand(3, -1) + current)
                    current += length
                    continue
                mesh = inner.get_vision_position_ids(current, grid[0], 1, merge, 1,
                                                     device=device)
                keep = survivors.to(device) if survivors is not None else None
                if keep is None or keep.numel() > mesh.shape[1]:
                    keep = torch.arange(min(length, mesh.shape[1]), device=device)
                if keep.numel() < length:
                    # The aggregate token rides at the end of the run; it has no
                    # grid cell of its own, so it reuses the last survivor's.
                    keep = torch.cat([keep, keep[-1:].expand(length - keep.numel())])
                pieces.append(mesh[:, keep[:length]])
                current += int(max(grid[0][1], grid[0][2])) // merge
            row_positions = torch.cat(pieces, dim=1).reshape(3, -1)
            if attention_mask is not None:
                positions[:, row, attention_mask[row].bool()] = row_positions.to(
                    positions.dtype)
            else:
                positions[:, row] = row_positions.to(positions.dtype)
            deltas.append(row_positions.max() + 1 - len(row_ids))
        return positions, torch.tensor(deltas, device=device).unsqueeze(1)
