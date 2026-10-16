"""FLOP accounting for generation.

`FlopCounter` measures FLOPs by counting matmul, convolution and attention ops as
they execute (``torch.utils.flop_counter``): 2 FLOPs per multiply-accumulate,
elementwise ops excluded, attention over the full score matrix. Its attention
formula also covers grouped-query attention and the CPU SDPA kernel.

`theoretical_flops` computes the per-layer theoretical FLOPs used in the
token-pruning literature.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import torch
from torch.utils.flop_counter import FlopCounterMode

aten = torch.ops.aten

MODULE_DEPTH = 2  # e.g. model.vision_tower, model.language_model
_MATMUL_LIKE_OP = re.compile(r"attention|attn|mm|conv|matmul|(?<![a-z])linear")


def _attention_flops(query_shape: Any, key_shape: Any, value_shape: Any, *args: Any,
                     out_shape: Any = None, **kwargs: Any) -> int:
    """``Q @ K^T`` and ``P @ V`` per query head (keys may have fewer heads under GQA)."""
    batch, heads, query_len, head_dim = query_shape
    key_len = key_shape[-2]
    value_dim = value_shape[-1]
    return 2 * batch * heads * query_len * key_len * (head_dim + value_dim)


ATTENTION_FLOP_FORMULAS = {
    op: _attention_flops
    for op in (
        aten._scaled_dot_product_flash_attention,
        aten._scaled_dot_product_efficient_attention,
        aten._scaled_dot_product_cudnn_attention,
        aten._scaled_dot_product_flash_attention_for_cpu,
        aten._scaled_dot_product_fused_attention_overrideable,
    )
}


class _TrackingFlopCounterMode(FlopCounterMode):
    def __init__(self) -> None:
        super().__init__(display=False, custom_mapping=ATTENTION_FLOP_FORMULAS)
        self.uncounted_ops: set[str] = set()

    def _count_flops(self, func_packet: Any, out: Any, args: Any, kwargs: Any) -> Any:
        if func_packet not in self.flop_registry:
            name = str(func_packet)
            if _MATMUL_LIKE_OP.search(name):
                self.uncounted_ops.add(name)
        return super()._count_flops(func_packet, out, args, kwargs)


class FlopCounter:
    """Counts FLOPs run inside ``with counter:``; `total` can be read while active."""

    def __init__(self) -> None:
        self._mode = _TrackingFlopCounterMode()

    def __enter__(self) -> "FlopCounter":
        self._mode.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._mode.__exit__(*exc)

    @property
    def total(self) -> int:
        return int(self._mode.get_total_flops())

    @property
    def uncounted_ops(self) -> list[str]:
        return sorted(self._mode.uncounted_ops)

    def by_module(self, depth: int = MODULE_DEPTH) -> dict[str, int]:
        """FLOPs per module up to ``depth`` levels below the root, partitioning the total."""
        counts = {
            name: sum(ops.values())
            for name, ops in self._mode.get_flop_counts().items()
            if name != "Global"
        }
        # A module's count includes its tracked descendants; keep only its own share so
        # each op is attributed once, even under containers that are never called.
        own = dict(counts)
        for name, flops in counts.items():
            parent = name.rpartition(".")[0]
            while parent and parent not in counts:
                parent = parent.rpartition(".")[0]
            if parent:
                own[parent] -= flops
        result: dict[str, int] = {}
        for name, flops in own.items():
            if flops:
                root, *path = name.split(".")
                label = ".".join(path[:depth]) or root
                result[label] = result.get(label, 0) + int(flops)
        outside = self.total - sum(result.values())
        if outside:
            result["<outside modules>"] = int(outside)
        return result


def theoretical_flops(vision_tokens_per_layer: Sequence[int], hidden_size: int, intermediate_size: int) -> int:
    """Theoretical prefill FLOPs of the vision tokens.

    Uses the per-layer formula of HiDrop (arXiv:2602.23699), summed over decoder layers::

        FLOPs = sum_l (4 n_l d^2 + 2 n_l^2 d + 3 n_l d m)

    where ``n_l`` is the number of vision tokens at layer ``l``, ``d`` the hidden size
    and ``m`` the FFN intermediate size.
    """
    d, m = int(hidden_size), int(intermediate_size)
    return sum(4 * n * d * d + 2 * n * n * d + 3 * n * d * m for n in map(int, vision_tokens_per_layer))
