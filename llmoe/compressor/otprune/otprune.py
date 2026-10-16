"""OTPrune: distribution-aligned visual-token pruning via optimal transport.

Paper: Chen et al., "OTPrune: Distribution-Aligned Visual Token Pruning via
Optimal Transport" (arXiv:2602.20205). Reference code:
https://github.com/xiwenc1/OTPrune (``llava/model/llava_arch.py`` -- the
module-level ``greedy_select`` and ``LlavaMetaForCausalLM.OTPrune``).

OTPrune is training-free and the simplest pruner in this package: it never looks
at attention, at the prompt, or at the decoder. It keeps the subset of visual
tokens whose *distribution* best matches the full set, which the paper derives as
an optimal-transport alignment and the released code optimises as **DPP greedy
MAP inference**:

1. per-channel L2-normalise the post-projector tokens ``V`` (``[N, H]``);
2. build the kernel ``K = I + gamma * (V Vᵀ)²`` (``gamma = 0.01``);
3. greedily pick the ``max_tokens`` items maximising ``log det`` of the principal
   submatrix, via incremental Cholesky (``greedy_select``);
4. gather those rows in spatial order.

The paper motivates a general OT objective; this is a port of the **released
code**, not of that objective, because the published numbers come from this exact
instantiation. Selection is deterministic -- there is no sampling anywhere.

Budget semantics
----------------
Unlike ``sparse_vlm.py``, ``must_drop.py`` and ``duet.py``, whose labels are
layer-averaged, ``max_tokens`` here is the **literal final count**: exactly
``min(max_tokens, N)`` tokens reach the LLM and stay for the whole forward.
That is the fixed-grid LLaVA parameterisation. Qwen2.5-VL and LLaVA-NeXT
instead require ``retain_ratio``. Qwen resolves the count independently for
each dynamic NaViT grid; NeXT does the same for each native AnyRes packed image
and trims its matching ``<image>`` placeholder run. OTPrune supports any count
or ratio budget, so it produces a full matched-budget curve.

Faithfulness notes / harness adaptations
----------------------------------------
* **float32 selection.** The official code normalises in float32 (so the kernel is
  float32) but allocates the Cholesky buffer ``cis`` in the input dtype (fp16).
  The whole DPP is run in float32 here for numerical stability, and the features
  are gathered back in their original dtype. Reproduced from the baseline port.

* **Batching.** The official code is batch-size-1 and slices the visual tokens out
  of the merged sequence at a hardcoded ``SYS_TOKEN_LEN = 35``. This harness hands
  ``compress()`` the post-projector visual tokens directly, so the per-image
  selection is simply looped over the batch -- the kept sets are independent per
  image, so this is equivalent, and any batch size works.

* **Spatial order.** The official active branch leaves the kept block in
  ``list(set(...))`` iteration order (only its commented-out DivPrune branch
  sorts). The kept *set* is identical here; the rows are sorted so tokens enter
  the LLM in raster order, which makes the intra-block order deterministic
  instead of dependent on CPython's set iteration.

* **Placeholders.** On LLaVA, transformers 5.x scatters image features into
  ``<image>`` placeholders with a strict count check, so ``install()`` also
  trims each run. On Qwen, the full visual block first passes that check and
  M-RoPE construction; DPP selection then runs over the concatenated span and
  the shared adapter gathers it. Qwen's input placeholders are never
  pre-trimmed.

* **Dedup bound.** The official dedup loop re-runs the greedy selection until it
  has ``max_tokens`` distinct indices. Every iteration provably adds at least one
  new index on a well-formed kernel, so it terminates in under ``max_tokens``
  rounds; a hard bound at ``max_tokens`` is enforced and raises rather than
  spinning forever if a degenerate (e.g. all-zero, NaN-normalising) image ever
  breaks that. The bound cannot trigger on any input the official loop completes.

Cost
----
``greedy_select`` is an inherently sequential ``max_tokens``-step loop, and each
step reads back an ``argmax`` -- so a budget of 128 costs 128 device
synchronisations per image, on top of the ``[N, N]`` kernel. That is the
algorithm, not the port (the official code is identical), but it does make
OTPrune markedly slower per sample than the other static pruners here.
"""

from typing import Any, override

import torch

from llmoe.backbone.prototype import VLM
from llmoe.compressor.internvl import trim_image_placeholders
from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor

# Official `gamma` in OTPrune(); scales the quadratic similarity term.
DEFAULT_GAMMA: float = 0.01
DEFAULT_MAX_TOKENS: int = 128
# Official early-stop floor on the residual gain; effectively "no volume left".
DEFAULT_EPSILON: float = 1e-100


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
        raise RuntimeError("Could not resolve the image placeholder token id for OTPrune.")
    return int(token_id)


def greedy_select(kernel_matrix: torch.Tensor, max_length: int, item_size: int,
                  device: torch.device, dtype: torch.dtype,
                  epsilon: float = DEFAULT_EPSILON) -> list[int]:
    """DPP greedy MAP selection by incremental Cholesky.

    Direct port of the official ``greedy_select``. Picks up to ``max_length``
    items greedily maximising the log-determinant of the principal submatrix of
    ``kernel_matrix``, maintaining the Cholesky factor rows in ``cis`` and the
    residual gains in ``di2s``.

    The returned list may contain repeats when the residual gains collapse (the
    caller's dedup loop handles that), and may be shorter than ``max_length`` when
    the best remaining gain falls below ``epsilon``.

    :param kernel_matrix: ``[item_size, item_size]`` PSD kernel.
    :param max_length: number of items to select.
    :param item_size: number of candidates.
    :param device: device for the Cholesky buffer.
    :param dtype: dtype for the Cholesky buffer (float32 here; see the module docstring).
    :return: selected item indices, in selection order.
    """
    if max_length < 1:
        return []
    cis = torch.zeros((max_length, item_size), device=device, dtype=dtype)
    di2s = torch.diag(kernel_matrix).clone()

    selected_items: list[int] = []
    selected_item = int(torch.argmax(di2s).item())
    selected_items.append(selected_item)

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - (ci_optimal.unsqueeze(0) @ cis[:k, :]).squeeze()) / di_optimal
        cis[k, :] = eis
        di2s = di2s - torch.square(eis)
        selected_item = int(torch.argmax(di2s).item())
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

    return selected_items


class OTPruneCompressor(TokenCompressor):
    """Keep the DPP-most-diverse visual tokens, before the language model.

    Parameters
    ----------
    max_tokens:
        Fixed-grid LLaVA literal final visual-token count.
    retain_ratio:
        Fraction of each image's dynamic visual-token count on Qwen2.5-VL and
        LLaVA-NeXT. It is the sole LLaVA-NeXT budget control.
    gamma:
        Weight of the quadratic similarity term in ``K = I + gamma * (V Vᵀ)²``.
        The official default is 0.01.
    track_source:
        Record ``[T_current, T_original]`` matrices for ``llmoe.viz``. Required by
        ``bm --visualize``; off by default so benchmark runs don't pay for it.
    """

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS, *, gamma: float = DEFAULT_GAMMA,
                 retain_ratio: float | None = None,
                 track_source: bool = False) -> None:
        super().__init__(int(max_tokens))
        if self._max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if float(gamma) < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if retain_ratio is not None and not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError(
                f"retain_ratio must be in (0, 1], got {retain_ratio}"
            )
        self._gamma: float = float(gamma)
        self._retain_ratio: float | None = (
            None if retain_ratio is None else float(retain_ratio)
        )
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
        return self._max_tokens if self._retain_ratio is None else self._retain_ratio

    def validate_llava_next(self) -> None:
        """Enforce the canonical NeXT ratio path and released DPP kernel."""

        from llmoe.compressor.llava_next import validate_retain_ratio

        validate_retain_ratio(self._retain_ratio, label="OTPrune")
        if self._max_tokens != DEFAULT_MAX_TOKENS:
            raise ValueError(
                "OTPrune on LLaVA-NeXT only accepts retain_ratio; remove max_tokens "
                f"(its constructor default is {DEFAULT_MAX_TOKENS} and is ignored on NeXT)."
            )
        if self._gamma != DEFAULT_GAMMA:
            raise ValueError(
                f"OTPrune on LLaVA-NeXT fixes gamma={DEFAULT_GAMMA:g}, matching the "
                "released method; only retain_ratio is configurable."
            )

    # ----------------------------------------------------------- selection core

    @torch.no_grad()
    def _select_indices(
        self, visual: torch.Tensor, keep: int | None = None
    ) -> torch.Tensor:
        """Kept indices for one image's ``[N, H]`` tokens, ascending and distinct.

        Mirrors the official ``OTPrune()``: per-channel L2 normalisation, the
        ``I + gamma (V Vᵀ)²`` kernel, DPP greedy selection, then a dedup loop that
        zeroes the chosen rows/columns and re-selects until enough distinct
        indices are gathered.
        """
        device = visual.device
        n_tokens = visual.shape[0]
        keep = (
            min(self._max_tokens, n_tokens)
            if keep is None
            else max(1, min(int(keep), n_tokens))
        )

        values = visual.to(torch.float32)
        # Official normalisation is over dim=-2, i.e. per channel across tokens.
        values = values / values.norm(dim=-2, keepdim=True)
        gram = values @ values.t()
        kernel = torch.eye(n_tokens, device=device, dtype=torch.float32) \
            + self._gamma * (gram @ gram.t())

        selected = greedy_select(kernel, keep, n_tokens, device, torch.float32)
        rounds = 0
        while len(set(selected)) < keep:
            rounds += 1
            if rounds > keep:
                # Unreachable while each round adds an index; see the module
                # docstring. Raising beats spinning inside a benchmark sweep.
                raise RuntimeError(
                    f"OTPrune could not find {keep} distinct tokens for an image with "
                    f"{n_tokens} of them; the DPP kernel is degenerate (all-zero or "
                    f"non-finite features?)."
                )
            for index in selected:
                kernel[index, :] = 0
                kernel[:, index] = 0
            selected.extend(
                greedy_select(kernel, keep - len(set(selected)), n_tokens, device, torch.float32))

        # Sorted into raster order: the kept set matches the official code, only
        # the intra-block order is made deterministic.
        return torch.tensor(sorted(set(selected))[:keep], dtype=torch.long, device=device)

    # ----------------------------------------------------------- TokenCompressor

    @override
    def compress(self, features: torch.Tensor, *, attention: torch.Tensor | None = None,
                 layer_idx: int | None = None, **kwargs: Any) -> CompressedFeatures:
        """Keep the ``max_tokens`` most distribution-representative tokens.

        :param features: ``[B, N, H]`` or ``[N, H]`` post-projector visual tokens.
        :param attention: unused -- OTPrune scores from the embeddings alone, with
            no attention and no text guidance.
        :return: ``[B, min(max_tokens, N), H]``, same rank, dtype and device.
        """
        del attention, layer_idx, kwargs
        if self._retain_ratio is not None:
            raise RuntimeError(
                "OTPruneCompressor.compress() is the fixed-count LLaVA path. "
                "On Qwen2.5-VL the ratio reduction runs on the concatenated "
                "prompt through QwenOTPruneAdapter."
            )

        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Expected [B, N, H] or [N, H], got {tuple(features.shape)}")
        batch, n_tokens, _ = features.shape
        if n_tokens == 0:
            raise ValueError("Cannot compress an empty sequence")

        if self._max_tokens >= n_tokens:  # budget already fits: pass through
            if self._track_source:
                identity = torch.eye(n_tokens, dtype=torch.float32)
                self._source_history = [identity, identity.clone()]
            return CompressedFeatures(tokens=features.squeeze(0) if squeeze else features)

        kept = []
        first_indices: torch.Tensor | None = None
        for image in range(batch):
            indices = self._select_indices(features[image])
            if first_indices is None:
                first_indices = indices
            kept.append(features[image].index_select(0, indices))
        tokens = torch.stack(kept, dim=0)

        if self._track_source:
            # Pure selection: every row is one-hot at the token it keeps, which
            # the visualiser reads as pruning rather than merging.
            source = torch.zeros(first_indices.numel(), n_tokens, dtype=torch.float32)
            source[torch.arange(first_indices.numel()), first_indices.detach().cpu()] = 1.0
            self._source_history = [torch.eye(n_tokens, dtype=torch.float32), source]

        return CompressedFeatures(tokens=tokens.squeeze(0) if squeeze else tokens)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        """``[identity, selection]`` for the last forward, ``[T_cur, N]`` each.

        The second entry's rows are one-hot over the original tokens, since
        OTPrune only ever selects. Empty when ``track_source`` is off or no
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
            raise RuntimeError("OTPruneCompressor is already installed on another backbone")

        inner = vlm.model.model
        from llmoe.compressor.llava_next import LlavaNextPackedSelectionAdapter

        if LlavaNextPackedSelectionAdapter.supports(vlm):
            self.validate_llava_next()
            adapter = LlavaNextPackedSelectionAdapter(
                label="OTPrune",
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
                f"[OTPrune/LLaVA-NeXT] install: retain_ratio={self._retain_ratio:g}, "
                f"gamma={self._gamma:g}, select the complete native packed span",
                flush=True,
            )
            return
        if getattr(inner, "visual", None) is not None:
            if self._retain_ratio is None:
                raise ValueError(
                    "OTPrune on Qwen2.5-VL requires retain_ratio: NaViT emits "
                    "a different visual-token count for each image."
                )
            from llmoe.compressor.otprune.otprune_qwen import (
                QwenOTPruneAdapter,
            )

            adapter = QwenOTPruneAdapter(
                compressor=self,
                retain_ratio=self._retain_ratio,
                track_source=self._track_source,
            )
            adapter.install(
                vlm,
                image_token_id=resolve_image_token_id(
                    vlm.model.config, vlm.processor
                ),
            )
            self._installed_vlm = vlm
            self._qwen = adapter
            return
        if self._retain_ratio is not None:
            raise ValueError(
                "retain_ratio is OTPrune's Qwen2.5-VL parameterisation; "
                "LLaVA-1.5 and InternVL use max_tokens."
            )

        from llmoe.compressor.otprune.otprune_internvl import (
            InternVLOTPruneAdapter,
        )

        if InternVLOTPruneAdapter.supports(vlm):
            adapter = InternVLOTPruneAdapter(self)
            adapter.install(vlm)
            self._installed_vlm = vlm
            self._internvl = adapter
            print(
                f"[OTPrune/InternVL] install: max_tokens={self._max_tokens} "
                f"(final count), gamma={self._gamma}, concatenate tiles per source",
                flush=True,
            )
            return

        if getattr(inner, "get_image_features", None) is None:
            raise TypeError("OTPrune needs a get_image_features on model.model")

        self._installed_vlm = vlm
        self._inner_model = inner
        self._image_token_id = resolve_image_token_id(vlm.model.config, vlm.processor)
        self._wrap_get_image_features(inner)
        self._wrap_prepare_inputs(vlm)

        print(f"[OTPrune] install: max_tokens={self._max_tokens} (final count), "
              f"gamma={self._gamma}, image_token_id={self._image_token_id}", flush=True)

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
            self._installed_vlm._otprune_prepare_wrapped = False
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
            stacked = torch.stack(list(image_outputs.pooler_output), dim=0)
            result = self.compress(stacked)
            image_outputs.pooler_output = list(result.tokens.unbind(0))
            return image_outputs

        inner.get_image_features = wrapped_get_image_features

    def _wrap_prepare_inputs(self, vlm: VLM) -> None:
        if getattr(vlm, "_otprune_prepare_wrapped", False):
            return
        original = vlm.prepare_inputs
        self._original_prepare_inputs = original

        def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            return self._trim_image_placeholders(vlm, original(*args, **kwargs))

        vlm.prepare_inputs = wrapped_prepare_inputs
        vlm._otprune_prepare_wrapped = True

    def _trim_image_placeholders(self, vlm: VLM, inputs: Any) -> Any:
        """Trim each source-image run independently to the global kept count."""
        return trim_image_placeholders(
            inputs,
            token_id=self._image_token_id,
            keep_for_run=lambda run_length: min(self._max_tokens, run_length),
            pad_token_id=self._pad_token_id(vlm),
        )

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
