"""Optional profiler-backed performance metrics for benchmark runs.

``torch.profiler`` records every CPU and CUDA op event into host memory, so
leaving it recording for a whole benchmark costs memory proportional to the
dataset -- roughly 8 GB by 320 MME samples, an OOM long before the full 2374 --
and it roughly doubles wall time. It is therefore given a bounded ``schedule``:
it traces a few batches near the start and then stops recording entirely.

That split is safe because only three things are scoped to the sampled window
(``sampled_batches``/``sampled_examples`` say how wide it was): the operator
table, the ``self_cpu``/``cpu``/``self_cuda``/``cuda`` second counts under
``profiler_metadata``, and the measured FLOPs under ``compute``. Everything under
``generation`` comes from this class's own perf_counter and CUDA-event timers,
which run unconditionally and still describe the entire run.

Measured FLOPs (`llmoe.flops.FlopCounter`) cover the whole backbone call of each
traced batch. For per-query latency and FLOPs, use the ``varest_bench`` benchmark.

``performance_profile.schedule`` overrides the window; setting ``active`` to
``null`` restores unbounded recording, with the memory cost above.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import torch
from torch.profiler import ProfilerActivity, profile, schedule as profiler_schedule

from llmoe.flops import FlopCounter


SUMMARY_ROW_LIMIT = 200
# Trace a handful of batches, then stop recording. `wait` skips the first batch
# (whose allocator warmup and lazy model materialisation are not representative),
# `warmup` primes the profiler without keeping events, `active` is what is kept.
DEFAULT_PROFILER_SCHEDULE: dict[str, int] = {"wait": 1, "warmup": 1, "active": 3, "repeat": 1}
PROFILER_OPTIONS: dict[str, bool] = {
    "record_shapes": False,
    "profile_memory": True,
    "with_stack": False,
    "acc_events": True,
}


def performance_profile_enabled(custom_args: dict[str, Any]) -> bool:
    """Return whether profiler-backed performance tracking is requested.

    Profiler-backed metrics (torch.profiler, measured FLOPs, per-forward timing) are
    opt-out: enabled by default and disabled only when ``performance_profile``
    is explicitly set to a falsy value (or ``{"enabled": False}``). Plain
    total-inference timing in ``run.py`` is always collected regardless.
    """
    setting = custom_args.get("performance_profile", True)
    if isinstance(setting, dict):
        return bool(setting.get("enabled", True))
    return bool(setting)


def performance_profile_schedule(custom_args: dict[str, Any]) -> dict[str, int] | None:
    """The trace window for this run, or ``None`` to record every batch.

    ``performance_profile.schedule`` accepts any subset of ``wait``/``warmup``/
    ``active``/``repeat``; unnamed keys keep their defaults. An explicit
    ``"active": null`` asks for the unbounded profiler.
    """
    setting = custom_args.get("performance_profile", True)
    if not isinstance(setting, dict):
        return dict(DEFAULT_PROFILER_SCHEDULE)
    window = setting.get("schedule", {})
    if window is None:
        return None
    if not isinstance(window, dict):
        raise TypeError(f"performance_profile.schedule must be a mapping, got {type(window).__name__}")
    if "active" in window and window["active"] is None:
        return None
    resolved = {**DEFAULT_PROFILER_SCHEDULE, **{k: int(v) for k, v in window.items() if v is not None}}
    if resolved["active"] < 1:
        raise ValueError("performance_profile.schedule.active must be >= 1, or null for no bound")
    if any(resolved[key] < 0 for key in ("wait", "warmup", "repeat")):
        raise ValueError("performance_profile.schedule values must be non-negative")
    return resolved


def _event_value(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value or 0.0)
    return 0.0


def _sum_event_seconds(events: list[Any], *names: str) -> float:
    return sum(_event_value(event, *names) for event in events) / 1_000_000.0


def _safe_rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _safe_percent(numerator: float, denominator: float) -> float | None:
    rate = _safe_rate(numerator, denominator)
    return rate * 100.0 if rate is not None else None


def _format_section(title: str, values: dict[str, Any]) -> list[str]:
    return [title, *(f"  {key}: {value}" for key, value in values.items())]


def performance_profile_flops(custom_args: dict[str, Any]) -> bool:
    """Whether traced batches also count FLOPs (``performance_profile.flops``)."""
    setting = custom_args.get("performance_profile", True)
    if isinstance(setting, dict):
        return bool(setting.get("flops", True))
    return True


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class GenerateTrace:
    """Per-phase timing and FLOPs of one ``model.generate`` call.

    The first forward is the prefill, the rest are decode steps. FLOP fields are
    ``None`` unless a `FlopCounter` was active.
    """

    wall_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    prefill_calls: int = 0
    decode_calls: int = 0
    prefill_flops: int | None = None
    decode_flops: int | None = None

    @property
    def forward_calls(self) -> int:
        return self.prefill_calls + self.decode_calls


def generation_module(model: Any) -> Any:
    """The module whose ``forward`` generation calls; PEFT wrappers delegate to their base model."""
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


def _decoder_layers(model: Any) -> list[Any]:
    """The decoder blocks of a causal LM (or a multimodal wrapper around one), or ``[]``."""
    get_decoder = getattr(model, "get_decoder", None)
    try:
        decoder = get_decoder() if callable(get_decoder) else None
    except (AttributeError, NotImplementedError):
        decoder = None
    layers = getattr(decoder, "layers", None)
    return list(layers) if layers is not None else []


def _output_head(model: Any) -> Any | None:
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    try:
        return get_output_embeddings() if callable(get_output_embeddings) else None
    except (AttributeError, NotImplementedError):
        return None


@contextmanager
def trace_generate(model: Any, flop_counter: FlopCounter | None = None) -> Iterator[GenerateTrace]:
    """Time (and, with ``flop_counter``, count FLOPs for) each forward pass of the generate call in the block.

    A pass is normally one call of the model's ``forward``. Decoders that are driven layer by layer -- calling the
    decoder blocks and the output head directly, never ``forward`` (Varest's [REL] loop, ``varest.rel.generate_pruned``)
    -- are traced through the blocks instead: a pass runs from the first decoder block to the output head. Those hooks
    are inert while ``forward`` is running, so a model that does call ``forward`` is never counted twice.
    """
    model = generation_module(model)
    trace = GenerateTrace()
    if flop_counter is not None:
        trace.prefill_flops = 0
        trace.decode_flops = 0
    model_dict = getattr(model, "__dict__", {})
    had_instance_forward = "forward" in model_dict
    instance_forward = model_dict.get("forward")
    original_forward = model.forward
    use_cuda_events = torch.cuda.is_available()
    cuda_samples: list[tuple[bool, Any, Any]] = []
    state: dict[str, Any] = {"in_forward": 0, "layer_pass": None}

    def begin_pass() -> tuple[bool, Any, int]:
        is_prefill = trace.prefill_calls == 0 and trace.decode_calls == 0
        if is_prefill:
            trace.prefill_calls += 1
        else:
            trace.decode_calls += 1
        flops_before = flop_counter.total if flop_counter is not None else 0
        if use_cuda_events:
            start: Any = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = perf_counter()
        return is_prefill, start, flops_before

    def end_pass(is_prefill: bool, start: Any, flops_before: int) -> None:
        if use_cuda_events:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            cuda_samples.append((is_prefill, start, end_event))
        else:
            elapsed = perf_counter() - start
            if is_prefill:
                trace.prefill_seconds += elapsed
            else:
                trace.decode_seconds += elapsed
        if flop_counter is not None:
            flops = flop_counter.total - flops_before
            if is_prefill:
                trace.prefill_flops += flops
            else:
                trace.decode_flops += flops

    @wraps(original_forward)
    def timed_forward(*args: Any, **kwargs: Any) -> Any:
        opened = begin_pass()
        state["in_forward"] += 1
        try:
            return original_forward(*args, **kwargs)
        finally:
            state["in_forward"] -= 1
            end_pass(*opened)

    def on_first_layer(module: Any, args: Any) -> None:
        if not state["in_forward"] and state["layer_pass"] is None:
            state["layer_pass"] = begin_pass()

    def on_output_head(module: Any, args: Any, output: Any) -> None:
        if not state["in_forward"] and state["layer_pass"] is not None:
            opened, state["layer_pass"] = state["layer_pass"], None
            end_pass(*opened)

    handles: list[Any] = []
    layers, head = _decoder_layers(model), _output_head(model)
    if layers and head is not None:
        handles.append(layers[0].register_forward_pre_hook(on_first_layer))
        handles.append(head.register_forward_hook(on_output_head))
    model.forward = timed_forward
    synchronize_cuda()
    generate_start = perf_counter()
    try:
        yield trace
    finally:
        synchronize_cuda()
        trace.wall_seconds = perf_counter() - generate_start
        for handle in handles:
            handle.remove()
        for is_prefill, start_event, end_event in cuda_samples:
            elapsed = start_event.elapsed_time(end_event) / 1000.0
            if is_prefill:
                trace.prefill_seconds += elapsed
            else:
                trace.decode_seconds += elapsed
        if had_instance_forward:
            model.forward = instance_forward
        else:
            delattr(model, "forward")


class PerformanceProfiler:
    """Collect aggregate torch.profiler metrics and measured FLOPs when enabled."""

    def __init__(self, enabled: bool, schedule: dict[str, int] | None = None, *,
                 count_flops: bool = True) -> None:
        self.enabled = enabled
        # None means "record every batch" -- correct only for short runs; see the
        # module docstring for what that costs.
        self._schedule: dict[str, int] | None = schedule
        self.count_flops = count_flops
        self.profiled_batches = 0
        self.profiled_examples = 0
        # The subset of the above that torch.profiler actually traced, and so the
        # scope of every torch-derived number in the report.
        self.sampled_batches = 0
        self.sampled_examples = 0
        self._activity_names: list[str] = []
        self._profiler: Any | None = None
        self._wall_start: float | None = None
        self._wall_seconds: float = 0.0
        self.generate_wall_time_seconds: float = 0.0
        self.prefill_time_seconds: float = 0.0
        self.decode_time_seconds: float = 0.0
        self.generation_calls: int = 0
        self.prefill_forward_calls: int = 0
        self.decode_forward_calls: int = 0
        # Measured FLOPs, summed over the traced batches only.
        self._flop_counter: FlopCounter | None = None
        self.flops_examples: int = 0
        self.total_flops: int = 0
        self.prefill_flops: int = 0
        self.decode_flops: int = 0
        self.flops_by_module: dict[str, int] = {}
        self.uncounted_ops: set[str] = set()

    @classmethod
    def from_custom_args(cls, custom_args: dict[str, Any]) -> "PerformanceProfiler":
        return cls(
            performance_profile_enabled(custom_args),
            performance_profile_schedule(custom_args),
            count_flops=performance_profile_flops(custom_args),
        )

    def __enter__(self) -> "PerformanceProfiler":
        if not self.enabled:
            return self
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        self._activity_names = [activity.name for activity in activities]
        self._profiler = profile(
            activities=activities,
            schedule=profiler_schedule(**self._schedule) if self._schedule else None,
            **PROFILER_OPTIONS,
        )
        self._wall_start = perf_counter()
        self._profiler.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
        if not self.enabled or self._profiler is None:
            return None
        if self._wall_start is not None:
            self._wall_seconds = perf_counter() - self._wall_start
        return self._profiler.__exit__(exc_type, exc, tb)

    def step(self, examples: int) -> None:
        if not self.enabled:
            return
        # The batch that just ran is the step the profiler is currently on; it
        # advances to the next one below.
        if self._records_step(self.profiled_batches):
            self.sampled_batches += 1
            self.sampled_examples += int(examples)
        self.profiled_batches += 1
        self.profiled_examples += int(examples)
        if self._profiler is not None:
            self._profiler.step()

    def _records_step(self, index: int) -> bool:
        """Whether `torch.profiler` keeps the events of 0-based step `index`.

        Mirrors ``torch.profiler.schedule``: each cycle skips ``wait`` steps,
        traces ``warmup`` steps without keeping them, and keeps ``active`` steps;
        after ``repeat`` cycles nothing is recorded again.
        """
        window = self._schedule
        if window is None:
            return True
        cycle = window["wait"] + window["warmup"] + window["active"]
        if cycle <= 0:
            return False
        if window["repeat"] and index // cycle >= window["repeat"]:
            return False
        return index % cycle >= window["wait"] + window["warmup"]

    @contextmanager
    def measure_batch(self, examples: int) -> Iterator[None]:
        """Count FLOPs of the whole backbone call when this batch is traced."""
        if not (self.enabled and self.count_flops and self._records_step(self.profiled_batches)):
            yield
            return
        counter = FlopCounter()
        self._flop_counter = counter
        try:
            with counter:
                yield
        finally:
            self._flop_counter = None
        self.flops_examples += int(examples)
        self.total_flops += counter.total
        for name, flops in counter.by_module().items():
            self.flops_by_module[name] = self.flops_by_module.get(name, 0) + flops
        self.uncounted_ops.update(counter.uncounted_ops)

    @contextmanager
    def measure_generate(self, model: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        with trace_generate(model, self._flop_counter) as trace:
            yield
        self.generation_calls += 1
        self.generate_wall_time_seconds += trace.wall_seconds
        self.prefill_time_seconds += trace.prefill_seconds
        self.decode_time_seconds += trace.decode_seconds
        self.prefill_forward_calls += trace.prefill_calls
        self.decode_forward_calls += trace.decode_calls
        if trace.prefill_flops is not None:
            self.prefill_flops += trace.prefill_flops
            self.decode_flops += trace.decode_flops

    def _flops_report(self) -> dict[str, Any] | None:
        if not self.flops_examples:
            return None
        per_example = self.flops_examples
        outside_generate = self.total_flops - self.prefill_flops - self.decode_flops
        return {
            "counter": "torch.utils.flop_counter (llmoe.flops)",
            "scope": f"{self.flops_examples} of {self.profiled_examples} examples (traced batches)",
            "examples": self.flops_examples,
            "total_flops": self.total_flops,
            "flops_per_example": self.total_flops / per_example,
            "tflops_per_example": self.total_flops / per_example / 1e12,
            "prefill_flops_per_example": self.prefill_flops / per_example,
            "decode_flops_per_example": self.decode_flops / per_example,
            "outside_generate_flops_per_example": outside_generate / per_example,
            "by_module_flops_per_example": {
                name: flops / per_example for name, flops in self.flops_by_module.items()
            },
            "uncounted_ops": sorted(self.uncounted_ops),
        }

    def _build_profile_report(
        self,
        *,
        summary_path: Path,
        self_cpu_seconds: float,
        cpu_seconds: float,
        self_cuda_seconds: float,
        cuda_seconds: float,
    ) -> dict[str, Any]:
        model_forward_time_seconds = self.prefill_time_seconds + self.decode_time_seconds
        generate_overhead_time_seconds = max(
            self.generate_wall_time_seconds - model_forward_time_seconds,
            0.0,
        )

        return {
            "generation": {
                "batches": self.profiled_batches,
                "examples": self.profiled_examples,
                "total_time_seconds": self.generate_wall_time_seconds,
                "model_forward_time_seconds": model_forward_time_seconds,
                "prefill_time_seconds": self.prefill_time_seconds,
                "decode_time_seconds": self.decode_time_seconds,
                "generate_overhead_time_seconds": generate_overhead_time_seconds,
                "prefill_percent_of_forward_time": _safe_percent(
                    self.prefill_time_seconds,
                    model_forward_time_seconds,
                ),
                "decode_percent_of_forward_time": _safe_percent(
                    self.decode_time_seconds,
                    model_forward_time_seconds,
                ),
                "avg_generation_time_per_batch_seconds": _safe_rate(
                    self.generate_wall_time_seconds,
                    self.generation_calls,
                ),
                "avg_generation_time_per_example_seconds": _safe_rate(
                    self.generate_wall_time_seconds,
                    self.profiled_examples,
                ),
                "avg_prefill_time_per_batch_seconds": _safe_rate(
                    self.prefill_time_seconds,
                    self.generation_calls,
                ),
                "avg_decode_time_per_batch_seconds": _safe_rate(
                    self.decode_time_seconds,
                    self.generation_calls,
                ),
            },
            "compute": {
                "flops": self._flops_report(),
            },
            "artifacts": {
                "profile_summary_path": str(summary_path),
            },
            "profiler_metadata": {
                "profiler": "torch.profiler",
                "activities": self._activity_names,
                "profile_wall_time_seconds": self._wall_seconds,
                # The op table, the four *_time_seconds below and `compute.flops`
                # cover only these batches, not the whole run; `generation` covers
                # the run.
                "trace_schedule": self._schedule,
                "sampled_batches": self.sampled_batches,
                "sampled_examples": self.sampled_examples,
                "scope": (
                    "whole run" if self._schedule is None
                    else f"{self.sampled_batches} of {self.profiled_batches} batches"
                ),
                "generation_calls": self.generation_calls,
                "prefill_forward_calls": self.prefill_forward_calls,
                "decode_forward_calls": self.decode_forward_calls,
                "self_cpu_time_seconds": self_cpu_seconds if self.sampled_batches else None,
                "cpu_time_seconds": cpu_seconds if self.sampled_batches else None,
                "self_cuda_time_seconds": (
                    self_cuda_seconds
                    if self.sampled_batches and "CUDA" in self._activity_names
                    else None
                ),
                "cuda_time_seconds": (
                    cuda_seconds
                    if self.sampled_batches and "CUDA" in self._activity_names
                    else None
                ),
                **PROFILER_OPTIONS,
            },
        }

    def write_artifacts(self, summary_path: Path) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if self._profiler is None:
            raise RuntimeError(
                "Performance profiling was enabled but the profiler was not started."
            )

        events = self._profiler.key_averages()
        event_list = list(events)
        self_cpu_seconds = _sum_event_seconds(event_list, "self_cpu_time_total")
        cpu_seconds = _sum_event_seconds(event_list, "cpu_time_total")
        self_cuda_seconds = _sum_event_seconds(
            event_list,
            "self_cuda_time_total",
            "self_device_time_total",
        )
        cuda_seconds = _sum_event_seconds(event_list, "cuda_time_total", "device_time_total")

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        result = self._build_profile_report(
            summary_path=summary_path,
            self_cpu_seconds=self_cpu_seconds,
            cpu_seconds=cpu_seconds,
            self_cuda_seconds=self_cuda_seconds,
            cuda_seconds=cuda_seconds,
        )
        generation = result["generation"]
        flops = result["compute"]["flops"] or {}
        compute = {
            key: flops.get(key)
            for key in (
                "scope",
                "tflops_per_example",
                "prefill_flops_per_example",
                "decode_flops_per_example",
                "outside_generate_flops_per_example",
                "uncounted_ops",
            )
        }

        sort_by = (
            "self_cuda_time_total" if "CUDA" in self._activity_names else "self_cpu_time_total"
        )
        try:
            table = events.table(sort_by=sort_by, row_limit=SUMMARY_ROW_LIMIT)
        except Exception:
            table = events.table(sort_by="self_cpu_time_total", row_limit=SUMMARY_ROW_LIMIT)
        profiler_summary = {
            "activities": ", ".join(self._activity_names),
            "trace_scope": result["profiler_metadata"]["scope"],
            "sampled_batches": self.sampled_batches,
            "sampled_examples": self.sampled_examples,
            "profile_wall_time_seconds": self._wall_seconds,
            "generation_calls": self.generation_calls,
            "prefill_forward_calls": self.prefill_forward_calls,
            "decode_forward_calls": self.decode_forward_calls,
            **PROFILER_OPTIONS,
        }
        summary_path.write_text(
            "\n".join(
                [
                    "LLMOE performance profile",
                    "",
                    *_format_section(
                        "Generation",
                        {
                            "batches": generation["batches"],
                            "examples": generation["examples"],
                            "total_time_seconds": generation["total_time_seconds"],
                            "model_forward_time_seconds": generation["model_forward_time_seconds"],
                            "prefill_time_seconds": generation["prefill_time_seconds"],
                            "decode_time_seconds": generation["decode_time_seconds"],
                            "prefill_percent_of_forward_time": (
                                generation["prefill_percent_of_forward_time"]
                            ),
                            "decode_percent_of_forward_time": (
                                generation["decode_percent_of_forward_time"]
                            ),
                            "avg_generation_time_per_example_seconds": (
                                generation["avg_generation_time_per_example_seconds"]
                            ),
                        },
                    ),
                    "",
                    *_format_section("Compute", compute),
                    "",
                    *_format_section("Profiler metadata", profiler_summary),
                    "",
                    f"Operator table ({result['profiler_metadata']['scope']})",
                    "",
                    table,
                ]
            ),
            encoding="utf-8",
        )

        return result
