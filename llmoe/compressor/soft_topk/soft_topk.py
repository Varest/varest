"""Learned, differentiable top-k visual-token pruning.

The compressor uses a small MLP to score every visual token.  Its forward pass
always contains the real hard top-k tokens, while training uses a straight-
through relaxation so gradients can reach the scorer.  In evaluation mode the
relaxation is skipped entirely.

LLaVA uses a fixed ``max_tokens`` budget. Qwen2.5-VL requires ``retain_ratio`` and
scores the projected visual span only after Qwen has concatenated it with the
text prompt; the shared adapter then narrows M-RoPE and the generation cache.
The trainer gathers labels through the same keep mask, so the learned scorer is
trainable on either backbone.
"""

from collections.abc import Mapping
from os import PathLike
from typing import Any, override

import torch
from torch import nn
from torch.nn import functional as F

from llmoe.compressor.prototype import CompressedFeatures, TokenCompressor
from llmoe.utils.qwen import resolve_image_token_id


class SoftTopkCompressor(TokenCompressor):
    """Select the most useful visual tokens with a trainable scorer.

    Parameters
    ----------
    max_tokens:
        LLaVA-only maximum number of visual tokens returned for each image.
    retain_ratio:
        Qwen2.5-VL-only fraction of each image's dynamic visual-token count to
        retain. A fixed count is not comparable across NaViT image grids.
    scorer_hidden_dim:
        Width of the scorer MLP.  The input projection is lazy, allowing the
        compressor to work with different vision/projector hidden sizes.
    temperature:
        Temperature of the differentiable top-k relaxation used in training.
    dropout:
        Scorer dropout.  It is disabled automatically in evaluation mode.
    checkpoint_path:
        Optional checkpoint produced by :meth:`save_checkpoint` or
        ``train_soft_topk``.
    """

    def __init__(
        self,
        max_tokens: int = 64,
        *,
        retain_ratio: float | None = None,
        scorer_hidden_dim: int = 256,
        temperature: float = 1.0,
        dropout: float = 0.0,
        checkpoint_path: str | PathLike[str] | None = None,
    ) -> None:
        super().__init__(int(max_tokens))
        if self._max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if int(scorer_hidden_dim) < 1:
            raise ValueError("scorer_hidden_dim must be >= 1")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be > 0")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if retain_ratio is not None and not 0.0 < float(retain_ratio) <= 1.0:
            raise ValueError(
                f"retain_ratio must be in (0, 1], got {retain_ratio}"
            )

        self._retain_ratio = (
            None if retain_ratio is None else float(retain_ratio)
        )
        self._scorer_hidden_dim = int(scorer_hidden_dim)
        self._temperature = float(temperature)
        self._dropout = float(dropout)
        self.scorer = nn.Sequential(
            nn.LazyLinear(self._scorer_hidden_dim),
            nn.GELU(),
            nn.Dropout(self._dropout),
            nn.Linear(self._scorer_hidden_dim, 1),
        )

        self._installed_vlm: Any | None = None
        self._original_get_image_features: Any | None = None
        self._original_prepare_inputs: Any | None = None
        self._qwen: Any | None = None
        self._for_training = False
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    @property
    def budget(self) -> int | float:
        return self._max_tokens if self._retain_ratio is None else self._retain_ratio

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if float(value) <= 0.0:
            raise ValueError("temperature must be > 0")
        self._temperature = float(value)

    def _scores(self, features: torch.Tensor) -> torch.Tensor:
        # Per-token layer normalization makes one scorer checkpoint much less
        # sensitive to the scale used by a backbone's multimodal projector.
        normalized = F.layer_norm(features, (features.shape[-1],))
        return self.scorer(normalized).squeeze(-1)

    def _soft_rank_assignments(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Return a differentiable ``[B, k, N]`` approximation to top-k.

        Rows are ordered from highest to lowest score.  At every step a softmax
        selects a token and its remaining mass is reduced, which is the common
        deterministic relaxation of sampling without replacement.
        """
        work_scores = scores.float()
        remaining = torch.ones_like(work_scores)
        assignments: list[torch.Tensor] = []
        eps = torch.finfo(work_scores.dtype).eps
        for _ in range(k):
            logits = work_scores / self._temperature + remaining.clamp_min(eps).log()
            probability = logits.softmax(dim=-1)
            assignments.append(probability)
            remaining = remaining * (1.0 - probability)
        return torch.stack(assignments, dim=1)

    @staticmethod
    def _hard_indices(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return spatially ordered indices and the permutation from score rank."""
        ranked = torch.topk(scores, k, dim=-1, largest=True, sorted=True).indices
        spatial, rank_to_spatial = ranked.sort(dim=-1)
        return spatial, rank_to_spatial

    @staticmethod
    def _gather_tokens(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return features.gather(
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]),
        )

    def _compress_to(
        self, features: torch.Tensor, keep: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the learned selector for an already-resolved runtime budget."""
        batch, n_tokens, _ = features.shape
        keep = max(1, min(int(keep), n_tokens))
        if keep == n_tokens:
            indices = torch.arange(
                n_tokens, device=features.device
            ).unsqueeze(0).expand(batch, -1)
            return features, indices

        scores = self._scores(features)
        indices, rank_to_spatial = self._hard_indices(scores, keep)
        hard_tokens = self._gather_tokens(features, indices)
        if self.training and torch.is_grad_enabled():
            soft_source = self._soft_rank_assignments(scores, keep)
            soft_source = soft_source.gather(
                1,
                rank_to_spatial.unsqueeze(-1).expand(
                    -1, -1, n_tokens
                ),
            )
            soft_tokens = torch.bmm(
                soft_source.to(features.dtype), features
            )
            hard_tokens = (
                hard_tokens + soft_tokens - soft_tokens.detach()
            )
        return hard_tokens, indices

    def _record_selection(
        self, indices: torch.Tensor, n_tokens: int, device: torch.device,
    ) -> None:
        identity = torch.eye(
            n_tokens, device=device, dtype=torch.float32
        )
        source = torch.zeros(
            indices.shape[1], n_tokens, device=device, dtype=torch.float32
        )
        source.scatter_(1, indices[0].unsqueeze(-1), 1.0)
        self._source_history = [identity, source]

    @override
    def compress(
        self,
        features: torch.Tensor,
        *,
        attention: torch.Tensor | None = None,
        layer_idx: int | None = None,
        **kwargs: Any,
    ) -> CompressedFeatures:
        del attention, layer_idx, kwargs
        squeeze = features.ndim == 2
        if squeeze:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"Expected [B, N, d] or [N, d], got {tuple(features.shape)}")
        if features.shape[1] == 0:
            raise ValueError("Cannot compress an empty token sequence")

        _, n_tokens, _ = features.shape
        k = min(self._max_tokens, n_tokens)
        output, indices = self._compress_to(features, k)
        self._record_selection(indices, n_tokens, features.device)
        return CompressedFeatures(tokens=output.squeeze(0) if squeeze else output)

    @override
    def get_source_history(self) -> list[torch.Tensor]:
        if self._qwen is not None:
            return self._qwen.get_source_history()
        return self._source_history

    def checkpoint_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "compressor": self.__class__.__name__,
            "config": {
                "max_tokens": self._max_tokens,
                "retain_ratio": self._retain_ratio,
                "scorer_hidden_dim": self._scorer_hidden_dim,
                "temperature": self._temperature,
                "dropout": self._dropout,
            },
            "state_dict": self.state_dict(),
        }

    def save_checkpoint(self, path: str | PathLike[str]) -> None:
        torch.save(self.checkpoint_dict(), path)

    def load_checkpoint(self, path: str | PathLike[str]) -> None:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0 compatibility
            payload = torch.load(path, map_location="cpu")
        state_dict: Mapping[str, torch.Tensor]
        if isinstance(payload, Mapping) and "state_dict" in payload:
            checkpoint_config = payload.get("config", {})
            checkpoint_k = (
                checkpoint_config.get("max_tokens")
                if isinstance(checkpoint_config, Mapping)
                else None
            )
            checkpoint_ratio = (
                checkpoint_config.get("retain_ratio")
                if isinstance(checkpoint_config, Mapping)
                else None
            )
            if (
                self._retain_ratio is None
                and checkpoint_k is not None
                and int(checkpoint_k) != self._max_tokens
            ):
                raise ValueError(
                    f"Checkpoint uses max_tokens={checkpoint_k}, compressor uses {self._max_tokens}"
                )
            if (
                self._retain_ratio is not None
                and checkpoint_ratio is not None
                and float(checkpoint_ratio) != self._retain_ratio
            ):
                raise ValueError(
                    f"Checkpoint uses retain_ratio={checkpoint_ratio}, "
                    f"compressor uses {self._retain_ratio}"
                )
            state_dict = payload["state_dict"]
        elif isinstance(payload, Mapping):
            state_dict = payload
        else:
            raise TypeError(f"Unsupported SoftTopk checkpoint payload: {type(payload).__name__}")
        self.load_state_dict(state_dict)

    @staticmethod
    def _image_token_id(vlm: Any) -> int | None:
        config = getattr(getattr(vlm, "model", None), "config", None)
        for name in ("image_token_id", "image_token_index"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        return None

    def trim_placeholders(
        self,
        inputs: Any,
        image_token_id: int,
        pad_token_id: int,
    ) -> Any:
        """Trim every contiguous image-placeholder run to ``max_tokens``.

        Transformers checks that the number of image placeholders equals the
        number of image features.  Token compression therefore has to update
        both sides of that equality.
        """
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            return inputs

        attention_mask = inputs.get("attention_mask")
        has_attention = (
            isinstance(attention_mask, torch.Tensor) and attention_mask.shape == input_ids.shape
        )
        kept_ids: list[torch.Tensor] = []
        kept_attention: list[torch.Tensor] = []
        changed = False

        for row_index, row in enumerate(input_ids):
            keep = torch.ones(row.shape[0], dtype=torch.bool, device=row.device)
            positions = (row == image_token_id).nonzero(as_tuple=False).flatten()
            if positions.numel():
                # Split non-consecutive positions into one run per image.
                split_points = (
                    (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=False).flatten() + 1
                )
                runs = torch.tensor_split(positions, split_points.tolist())
                for run in runs:
                    if run.numel() > self._max_tokens:
                        keep[run[self._max_tokens:]] = False
                        changed = True
            kept_ids.append(row[keep])
            if has_attention:
                kept_attention.append(attention_mask[row_index][keep])

        if not changed:
            return inputs

        max_length = max(row.numel() for row in kept_ids)
        new_ids = input_ids.new_full((len(kept_ids), max_length), pad_token_id)
        new_attention = (
            attention_mask.new_zeros((len(kept_ids), max_length)) if has_attention else None
        )
        for row_index, row in enumerate(kept_ids):
            length = row.numel()
            new_ids[row_index, max_length - length:] = row
            if new_attention is not None:
                new_attention[row_index, max_length - length:] = kept_attention[row_index]
        inputs["input_ids"] = new_ids
        if new_attention is not None:
            inputs["attention_mask"] = new_attention
        return inputs

    @override
    def install(self, vlm: Any) -> None:
        if self._installed_vlm is not None:
            if self._installed_vlm is vlm:
                return
            raise RuntimeError("SoftTopkCompressor is already installed on another backbone")

        inner = getattr(getattr(vlm, "model", None), "model", None)
        if getattr(inner, "visual", None) is not None:
            if self._retain_ratio is None:
                raise ValueError(
                    "SoftTopk on Qwen2.5-VL requires retain_ratio: NaViT emits "
                    "a different visual-token count for each image."
                )
            from llmoe.compressor.soft_topk.soft_topk_qwen import (
                QwenSoftTopkAdapter,
            )

            adapter = QwenSoftTopkAdapter(
                compressor=self,
                retain_ratio=self._retain_ratio,
                for_training=self._for_training,
            )
            adapter.install(
                vlm,
                image_token_id=resolve_image_token_id(
                    vlm.model.config,
                    vlm.processor,
                    label="SoftTopk",
                ),
            )
            self._installed_vlm = vlm
            self._qwen = adapter
            return
        if self._retain_ratio is not None:
            raise ValueError(
                "retain_ratio is SoftTopk's Qwen2.5-VL parameterisation; "
                "LLaVA-1.5 uses max_tokens."
            )
        original_get_image_features = getattr(inner, "get_image_features", None)
        if not callable(original_get_image_features):
            raise TypeError("Backbone does not expose model.model.get_image_features")

        self._installed_vlm = vlm
        self._original_get_image_features = original_get_image_features
        self._original_prepare_inputs = vlm.prepare_inputs

        def wrapped_get_image_features(*args: Any, **kwargs: Any) -> Any:
            image_outputs = original_get_image_features(*args, **kwargs)
            features = torch.stack(list(image_outputs.pooler_output), dim=0)
            compressed = self.compress(features)
            image_outputs.pooler_output = list(compressed.tokens.unbind(0))
            return image_outputs

        inner.get_image_features = wrapped_get_image_features

        image_token_id = self._image_token_id(vlm)
        if image_token_id is not None:
            tokenizer = getattr(getattr(vlm, "processor", None), "tokenizer", None)
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            pad_token_id = image_token_id if pad_token_id is None else int(pad_token_id)
            original_prepare_inputs = self._original_prepare_inputs

            def wrapped_prepare_inputs(*args: Any, **kwargs: Any) -> Any:
                inputs = original_prepare_inputs(*args, **kwargs)
                return self.trim_placeholders(inputs, image_token_id, pad_token_id)

            vlm.prepare_inputs = wrapped_prepare_inputs

    def uninstall(self) -> None:
        if self._installed_vlm is None:
            return
        if self._qwen is not None:
            self._qwen.uninstall()
            self._qwen = None
            self._installed_vlm = None
            return
        vlm = self._installed_vlm
        inner = getattr(getattr(vlm, "model", None), "model", None)
        if inner is not None and self._original_get_image_features is not None:
            inner.get_image_features = self._original_get_image_features
        if self._original_prepare_inputs is not None:
            vlm.prepare_inputs = self._original_prepare_inputs
        self._installed_vlm = None
        self._original_get_image_features = None
        self._original_prepare_inputs = None
