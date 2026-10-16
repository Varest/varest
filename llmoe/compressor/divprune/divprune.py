"""DivPrune: diversity-based visual-token pruning as a Max-Min Diversity Problem.

Paper: Alvar et al., "DivPrune: Diversity-based Visual Token Pruning for Large
Multimodal Models" (CVPR 2025, arXiv:2503.02175). Reference code:
https://github.com/vbdi/divprune (``LLaVA/llava/model/llava_arch.py`` -- the
``DivPrune`` branch of ``prepare_inputs_labels_for_multimodal``).

DivPrune is training-free and, like ``otprune.py``, never looks at attention, at
the prompt, or at the decoder. It casts pruning as the Max-Min Diversity Problem
(Section 3.2 / Algorithm 1): keep the subset of visual tokens whose *minimum*
pairwise distance is as large as possible, which the released code solves with
the standard greedy heuristic:

1. L2-normalise the post-projector tokens ``V`` (``[N, H]``) and build the
   all-pairs cosine-distance matrix ``D = 1 - V_hat V_hat^T``;
2. seed with the token whose second-smallest column distance is largest (the
   smallest entry is its own zero self-distance);
3. repeatedly add the token maximising its minimum distance to the chosen set,
   until the budget is filled;
4. sort the kept indices back into raster order before the LLM sees them.

Budget semantics
----------------
As in ``otprune.py``, and unlike the layer-averaged labels of ``sparse_vlm.py``,
``must_drop.py`` and ``duet.py``, ``max_tokens`` is the **literal final count**:
exactly ``min(max_tokens, N)`` tokens reach the LLM and stay for the whole
forward. ``subset_ratio`` is the alternative the paper and the released code use
(their LLaVA-1.5 default is ``0.098``, i.e. 56 of 576 tokens); when set it
overrides ``max_tokens`` and the count is resolved per input length as
``int(round(ratio * N))``, exactly as in the reference. Any budget works, so
DivPrune produces a full matched-budget curve.

Qwen2.5-VL and LLaVA-NeXT use ``retain_ratio`` because both emit dynamic
visual-token counts. Qwen preserves M-RoPE coordinates over its concatenated
NaViT span. NeXT resolves the count for each native AnyRes packed image, ranks
that complete span as the released LLaVA-1.6 path does, and trims its matching
``<image>`` placeholder run.

Faithfulness notes / harness adaptations
----------------------------------------
* **Selection dtype.** The distance matrix is built in the projected features'
  own dtype, as the released code does -- it runs the computation directly on
  the model tensor. ``otprune.py`` promotes to float32 for Cholesky stability;
  there is no such stability need here, and promoting fp16 to fp32 can flip
  close argmax decisions away from the published behaviour.

* **Batching.** The official code is batch-size-1 and slices the visual tokens
  out of the merged sequence. This harness hands ``compress()`` the
  post-projector visual tokens directly, so the per-image selection is simply
  looped over the batch -- the kept sets are independent per image, so this is
  equivalent, and any batch size works. LLaVA's padded preprocessing gives every
  image the same ``N``, hence the same kept count, which is what lets one
  placeholder trim serve the whole batch.

* **Duplicate prevention.** The reference relies on a token's zero self-distance
  to keep it from being re-selected. That breaks down for exactly-tied features,
  where a duplicate of an already-kept token can be picked again and the budget
  silently comes up short. Already-selected indices are masked out explicitly
  here, which changes nothing on non-degenerate images and returns a true subset
  on degenerate ones.

* **Zero-norm guard.** Cosine distance is undefined for a zero-norm token, and
  the resulting NaNs would quietly turn every ``argmax`` into index 0. Such a
  token cannot come out of a working projector, so this raises rather than
  scoring garbage for a whole sweep.

* **Placeholders.** transformers 5.x scatters image features into ``<image>``
  placeholders with a strict count check, so ``install()`` also wraps
  ``prepare_inputs`` to trim each placeholder run to the kept count -- the same
  thing ``vision_zip.py`` and ``otprune.py`` do. The reference framework handled
  this outside the compressor. This is the only reason DivPrune needs an
  ``install()`` at all; the selection itself is pure ``compress()``.

Cost
----
One ``[N, N]`` distance matrix per image plus an inherently sequential
budget-length greedy loop. Unlike ``otprune.py``, the loop keeps the running
argmax on the device and never reads it back, so a budget of 128 costs 128 small
kernels rather than 128 host synchronisations.
"""

from typing import Any, override

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor

# Below this a token's norm counts as zero, i.e. its cosine distance is undefined.
# Only reachable on a broken projector output; see the module docstring.
ZERO_NORM_EPS: float = 1e-12
DEFAULT_MAX_TOKENS: int = 64


def resolve_image_token_id(config: Any, processor: Any) -> int:
    """Resolve the ``<image>`` placeholder id, mirroring LLaVA's own lookup order."""
    token_id = getattr(config, "image_token_id", None)
    if token_id is None:
        token_id = getattr(config, "image_token_index", None)
    if token_id is None:
        image_token = getattr(processor, "image_token", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if image_token is not None and tokenizer is not None:
            token_id = tokenizer.convert_tokens_to_ids(str(image_token))
    if token_id is None:
        raise RuntimeError("Could not resolve the image placeholder token id for DivPrune.")
    return int(token_id)


class DivPruneCompressor(TokenCompressor):
    """Keep the max-min-diverse visual tokens, before the language model.

    Parameters
    ----------
    max_tokens:
        Visual tokens kept, the literal final count (see the module docstring on
        budgets). Ignored when ``subset_ratio`` is given.
    subset_ratio:
        Fraction of visual tokens to keep, the paper's retained ratio. Overrides
        ``max_tokens`` when set; the kept count is ``int(round(N * ratio))``
        exactly as in the reference, whose default is 0.098.
    retain_ratio:
        Fraction of each image's dynamic visual-token count on Qwen2.5-VL and
        LLaVA-NeXT. It is the sole LLaVA-NeXT budget control.
    track_source:
        Record ``[T_current, T_original]`` matrices for ``llmoe.viz``. Required by
        ``bm --visualize``; off by default so benchmark runs don't pay for it.
    """

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS, *,
                 subset_ratio: float | None = None,
                 retain_ratio: float | None = None, track_source: bool = False) -> None:
        super().__init__(int(max_tokens))
        if self._max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if subset_ratio is not None and not 0.0 < float(subset_ratio) <= 1.0:
            raise ValueError(f"subset_ratio must be in (0, 1], got {subset_ratio}")
        if retain_ratio is not None and not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError(f"retain_ratio must be in (0, 1], got {retain_ratio}")
        if subset_ratio is not None and retain_ratio is not None:
            raise ValueError(
                "subset_ratio is DivPrune's released LLaVA parameterisation; "
                "retain_ratio is its dynamic-backbone parameterisation. Pass only one."
            )
        self._subset_ratio: float | None = None if subset_ratio is None else float(subset_ratio)
        self._retain_ratio: float | None = None if retain_ratio is None else float(retain_ratio)
        self._track_source: bool = bool(track_source)

        # Installation bookkeeping, all restored by uninstall().
        self._installed_vlm: VLM | None = None
        self._inner_model: Any | None = None
        self._image_token_id: int | None = None
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._qwen: Any | None = None
        self._internvl: Any | None = None
        self._llava_next: Any | None = None

    @property
    @override
    def budget(self) -> int | float:
        if self._retain_ratio is not None:
            return self._retain_ratio
        # `subset_ratio` overrides `max_tokens` entirely, so a ratio run reports
        # the ratio and a matched-count run reports the count.
        return self._subset_ratio if self._subset_ratio is not None else self._max_tokens

    def validate_llava_next(self) -> None:
        """Enforce the sole canonical NeXT configuration: ``retain_ratio``."""

        from llmoe.compressor.llava_next import validate_retain_ratio

        validate_retain_ratio(self._retain_ratio, label="DivPrune")
        if self._max_tokens != DEFAULT_MAX_TOKENS:
            raise ValueError(
                "DivPrune on LLaVA-NeXT only accepts retain_ratio; remove max_tokens "
                f"(its constructor default is {DEFAULT_MAX_TOKENS} and is ignored on NeXT)."
            )
        if self._subset_ratio is not None:
            raise ValueError(
                "DivPrune on LLaVA-NeXT only accepts retain_ratio; subset_ratio is the "
                "released fixed-grid LLaVA parameterisation."
            )

    # ----------------------------------------------------------- selection core

    def _resolve_keep(self, n_tokens: int) -> int:
        """Visual tokens kept out of ``n_tokens`` (the reference's subset size).

        ``round`` is ties-to-even, matching ``int(round(ratio * length))`` in the
        released code (Python's and torch's rounding agree there).
        """
        keep = int(round(self._subset_ratio * n_tokens)) if self._subset_ratio is not None \
            else self._max_tokens
        return max(1, min(int(n_tokens), keep))

    @torch.no_grad()
    def _select_indices(self, visual: torch.Tensor, keep: int) -> torch.Tensor:
        """Kept indices for one image's ``[N, H]`` tokens, ascending and distinct.

        Mirrors the official greedy MMDP solver: cosine-distance matrix, the
        second-smallest-column seed, then ``keep - 1`` max-min steps.
        """
        n_tokens = visual.shape[0]
        if keep >= n_tokens:
            return torch.arange(n_tokens, device=visual.device)

        norms = visual.norm(dim=1, keepdim=True)
        if bool((norms <= ZERO_NORM_EPS).any()):
            raise ValueError(
                "DivPrune cosine distance is undefined for zero-norm visual tokens; the "
                "projector emitted an all-zero (or non-finite) token."
            )
        normalised = visual / norms
        distances = 1.0 - normalised @ normalised.transpose(0, 1)

        selected = torch.empty(keep, dtype=torch.long, device=visual.device)
        # Reference seed: per column the second-smallest distance (the smallest is
        # normally the zero diagonal), then the column where that is largest.
        selected[0] = torch.argmax(torch.topk(distances, 2, dim=0, largest=False).values[1])

        min_distance = distances[selected[0]].clone()
        min_distance[selected[0]] = -torch.inf
        for step in range(1, keep):
            chosen = torch.argmax(min_distance)
            selected[step] = chosen
            # `torch.minimum` carries the -inf of every earlier pick through, so
            # masking the new one is enough to keep the result a true subset (see
            # the module docstring on duplicates).
            min_distance = torch.minimum(min_distance, distances[chosen])
            min_distance[chosen] = -torch.inf

        # Sorted into raster order, reproducing the reference's final
        # `keep_indexs.sort()`: the kept set is diverse, the order is positional.
        return selected.sort().values

    # ----------------------------------------------------------- TokenCompressor

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, **kwargs: Any) -> CompressedFeatures:
        """Keep the most diverse tokens under the max-min criterion.

        :param features: ``[B, N, H]`` or ``[N, H]`` post-projector visual tokens.
        :param attention: unused -- DivPrune scores from the embeddings alone, with
            no attention and no text guidance.
        :return: ``[B, keep, H]``, same rank, dtype and device.
        """
        del attention, layer_idx, kwargs
        if self._retain_ratio is not None:
            raise RuntimeError(
                "DivPruneCompressor.compress() is the LLaVA tensor path. "
                "On Qwen2.5-VL, ratio pruning runs over the concatenated "
                "prompt through QwenDivPruneAdapter."
            )

        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Expected [B, N, H] or [N, H], got {tuple(features.shape)}")
        batch, n_tokens, _ = features.shape
        if n_tokens == 0:
            raise ValueError("Cannot compress an empty sequence")

        keep = self._resolve_keep(n_tokens)
        if keep >= n_tokens:  # budget already fits: pass through
            if self._track_source:
                identity = torch.eye(n_tokens, dtype=torch.float32)
                self._source_history = [identity, identity.clone()]
            return CompressedFeatures(tokens=features.squeeze(0) if squeeze else features)

        kept: list[torch.Tensor] = []
        first_indices: torch.Tensor | None = None
        for image in range(batch):
            indices = self._select_indices(features[image], keep)
            if first_indices is None:
                first_indices = indices
            kept.append(features[image].index_select(0, indices))
        tokens = torch.stack(kept, dim=0)

        if self._track_source:
            # Pure selection: every row is one-hot at the token it keeps, which the
            # visualiser reads as pruning rather than merging.
            source = torch.zeros(first_indices.numel(), n_tokens, dtype=torch.float32)
            source[torch.arange(first_indices.numel()), first_indices.detach().cpu()] = 1.0
            self._source_history = [torch.eye(n_tokens, dtype=torch.float32), source]

        return CompressedFeatures(tokens=tokens.squeeze(0) if squeeze else tokens)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        """``[identity, selection]`` for the last forward, ``[T_cur, N]`` each.

        The second entry's rows are one-hot over the original tokens, since
        DivPrune only ever selects. Empty when ``track_source`` is off or no
        forward has run.
        """
        if self._qwen is not None:
            return self._qwen.get_source_history()
        return list(self._source_history)

    # ------------------------------------------------------------------- install

    @override
    def install(self, vlm: VLM) -> None:
        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return  # idempotent: repeated benchmarks must not stack wrappers
            raise RuntimeError("DivPruneCompressor is already installed on another backbone")

        inner = vlm.model.model
        from llmoe.compressor.llava_next import LlavaNextPackedSelectionAdapter

        if LlavaNextPackedSelectionAdapter.supports(vlm):
            self.validate_llava_next()
            adapter = LlavaNextPackedSelectionAdapter(
                label="DivPrune",
                retain_ratio=float(self._retain_ratio),
                select_indices=self._select_indices,
                source_recorder=(
                    lambda history: setattr(self, "_source_history", history)
                ) if self._track_source else None,
            )
            adapter.install(vlm)
            self._installed_vlm = vlm
            self._llava_next = adapter
            print(
                f"[DivPrune/LLaVA-NeXT] install: retain_ratio={self._retain_ratio:g}, "
                "select the complete native packed span",
                flush=True,
            )
            return
        if getattr(inner, "visual", None) is not None:
            if self._retain_ratio is None:
                raise ValueError(
                    "DivPrune on Qwen2.5-VL requires retain_ratio: NaViT emits "
                    "a different visual-token count for each image."
                )
            from llmoe.compressor.divprune.divprune_qwen import QwenDivPruneAdapter
            from llmoe.utils.qwen import resolve_image_token_id as resolve_qwen_image_token_id

            adapter = QwenDivPruneAdapter(
                compressor=self,
                retain_ratio=self._retain_ratio,
                track_source=self._track_source,
            )
            adapter.install(
                vlm,
                image_token_id=resolve_qwen_image_token_id(
                    vlm.model.config, vlm.processor, label="DivPrune"
                ),
            )
            self._installed_vlm = vlm
            self._qwen = adapter
            return
        if self._retain_ratio is not None:
            raise ValueError(
                "retain_ratio is DivPrune's Qwen2.5-VL parameterisation; "
                "LLaVA-1.5 and InternVL use max_tokens or subset_ratio."
            )

        from llmoe.compressor.divprune.divprune_internvl import (
            InternVLDivPruneAdapter,
        )

        if InternVLDivPruneAdapter.supports(vlm):
            adapter = InternVLDivPruneAdapter(self)
            adapter.install(vlm)
            self._installed_vlm = vlm
            self._internvl = adapter
            budget = (
                f"subset_ratio={self._subset_ratio}"
                if self._subset_ratio is not None
                else f"max_tokens={self._max_tokens} (final count)"
            )
            print(
                f"[DivPrune/InternVL] install: {budget}, concatenate tiles per source",
                flush=True,
            )
            return

        if getattr(inner, "get_image_features", None) is None:
            raise TypeError("DivPrune needs a get_image_features on model.model")

        self._installed_vlm = vlm
        self._inner_model = inner
        self._image_token_id = resolve_image_token_id(vlm.model.config, vlm.processor)
        self._wrap_get_image_features(inner)
        self._wrap_prepare_inputs(vlm)

        budget = f"subset_ratio={self._subset_ratio}" if self._subset_ratio is not None \
            else f"max_tokens={self._max_tokens} (final count)"
        print(f"[DivPrune] install: {budget}, image_token_id={self._image_token_id}", flush=True)

    def uninstall(self) -> None:
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
        if self._inner_model is not None and self._original_get_image_features is not None:
            self._inner_model.get_image_features = self._original_get_image_features
        if self._installed_vlm is not None and self._original_prepare_inputs is not None:
            self._installed_vlm.prepare_inputs = self._original_prepare_inputs
            self._installed_vlm._divprune_prepare_wrapped = False
        self._original_get_image_features = None
        self._original_prepare_inputs = None
        self._image_token_id = None
        self._inner_model = None
        self._installed_vlm = None

    def _wrap_get_image_features(self, inner: Any) -> None:
        """Prune the projected features, mirroring the inherited base wrapper.

        The base ``wrap_get_image_features`` does not keep the original around, so
        it is re-implemented here to make ``uninstall()`` possible.
        """
        original = inner.get_image_features
        self._original_get_image_features = original

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            image_outputs = original(*args, **kwargs)
            # Preserve the established LLaVA path exactly: its images are a
            # regular batch and the selector runs once over that batch.
            stacked = torch.stack(list(image_outputs.pooler_output), dim=0)
            result = self.compress(stacked)
            image_outputs.pooler_output = list(result.tokens.unbind(0))
            return image_outputs

        inner.get_image_features = wrapped_get_image_features

    def _wrap_prepare_inputs(self, vlm: VLM) -> None:
        if getattr(vlm, "_divprune_prepare_wrapped", False):
            return
        original = vlm.prepare_inputs
        self._original_prepare_inputs = original

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            return self._trim_image_placeholders(vlm, original(*args, **kwargs))

        vlm.prepare_inputs = wrapped_prepare_inputs
        vlm._divprune_prepare_wrapped = True

    def _trim_image_placeholders(self, vlm: VLM, inputs: Any) -> Any:
        """Trim each ``<image>`` run to the kept count LLaVA will be handed."""
        input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return inputs
        attention_mask = inputs.get("attention_mask")
        has_attention = isinstance(attention_mask, torch.Tensor) \
            and attention_mask.shape == input_ids.shape
        id_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        changed = False

        for row in range(input_ids.shape[0]):
            ids = input_ids[row]
            keep_mask = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
            positions = (ids == self._image_token_id).nonzero(as_tuple=True)[0]
            if positions.numel():
                # A prompt may hold several images; each contiguous placeholder run
                # is one image and gets its own budget, as transformers treats them.
                splits = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=True)[0] + 1
                for run in torch.tensor_split(positions, splits.tolist()):
                    keep = self._resolve_keep(int(run.numel()))
                    if run.numel() > keep:
                        keep_mask[run[keep:]] = False
                        changed = True
            id_rows.append(ids[keep_mask])
            if has_attention:
                mask_rows.append(attention_mask[row][keep_mask])

        if not changed:
            return inputs
        inputs["input_ids"] = self._left_pad_stack(id_rows, self._pad_token_id(vlm))
        if has_attention:
            inputs["attention_mask"] = self._left_pad_stack(mask_rows, 0)
        return inputs

    @staticmethod
    def _pad_token_id(vlm: VLM) -> int:
        tokenizer = getattr(vlm.processor, "tokenizer", None)
        pad = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        return int(pad) if pad is not None else 0

    @staticmethod
    def _left_pad_stack(rows: list[torch.Tensor], pad_value: int) -> torch.Tensor:
        """Stack ragged rows with left padding, matching the tokenizer's side."""
        max_len = max(row.shape[0] for row in rows)
        padded: list[torch.Tensor] = []
        for row in rows:
            if row.shape[0] < max_len:
                pad = torch.full((max_len - row.shape[0],), pad_value, dtype=row.dtype,
                                 device=row.device)
                row = torch.cat([pad, row], dim=0)
            padded.append(row)
        return torch.stack(padded, dim=0)
