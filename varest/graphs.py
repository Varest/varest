"""CUDA-graph replay of a fixed-shape forward.

At batch 1 the vision tower is launch-bound: CLIP-L/14's 24 layers plus their LoRA updates are several hundred small
kernels, and issuing them occupied the one CPU thread that also has to run the field heads and the partition. A
captured graph issues the same kernels, with the same arguments, in one call, so its outputs are the eager outputs
(checked bit for bit by the tests) and the CPU is free while the GPU runs them.

:class:`GraphedCall` keeps one graph per input shape (LLaVA-1.5 always feeds one 336x336 image; LLaVA-NeXT feeds
1 + the AnyRes grid's 2-4 crops) and falls back to the eager call beyond ``max_graphs`` shapes or off CUDA."""
from __future__ import annotations

from typing import Callable

import torch


class GraphedCall:
    """``f(x)`` for a CUDA tensor ``x`` of a recurring shape, replayed from a CUDA graph captured on first use.

    ``f`` must be a pure function of ``x`` and of state that does not change between calls (module weights). The
    returned tensor is a fresh copy of the graph's static output, safe to keep across calls."""

    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor], max_graphs: int = 8, warmup: int = 2) -> None:
        self.f = f
        self.max_graphs = max_graphs
        self.warmup = warmup
        self.graphs: dict[tuple, tuple[torch.Tensor, torch.cuda.CUDAGraph, torch.Tensor]] = {}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return self.f(x)
        key = (tuple(x.shape), x.dtype, x.device)
        entry = self.graphs.get(key)
        if entry is None:
            if len(self.graphs) >= self.max_graphs:
                return self.f(x)
            entry = self.graphs[key] = self._capture(x)
        static_in, graph, static_out = entry
        static_in.copy_(x)
        graph.replay()
        return static_out.clone()

    def _capture(self, x: torch.Tensor):
        # a NORMAL tensor even when the first call runs under torch.inference_mode, so later calls in either mode can
        # copy into it (an in-place update of an inference tensor outside inference mode is an error)
        with torch.inference_mode(False):
            static_in = torch.empty_like(x)
        static_in.copy_(x)
        side = torch.cuda.Stream(device=x.device)
        side.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(side):                    # warm-up off the capture (cuBLAS handles, autotuned kernels)
            for _ in range(self.warmup):
                self.f(static_in)
        torch.cuda.current_stream(x.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.f(static_in)
        return static_in, graph, static_out
