"""Varest efficiency benchmark: per-query latency and FLOPs.

180 short-answer VQA cases in two sweeps, ``input_text_tokens`` (prompt length) and
``reference_answer_tokens`` (answer length). Prompts are the case text plus the
short-answer suffix, which reproduces each case's LLaVA-1.5 token count.

Each case runs at batch size 1: ``varest_bench_warmup`` warmup runs (first case
only), ``varest_bench_repeats`` timed runs (the median is reported), one run under
`FlopCounter` for measured and theoretical FLOPs, and (on CUDA) one run sampled by
`_MemorySampler` for the peak and time-weighted mean of allocated GPU memory. Only the
timed runs are timed. Responses are not scored.

Expects ``<input_dir>/Varest_Benchmark/`` with ``varest_benchmark.json`` and ``images/``.
"""

import json
import threading
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any, Callable, Iterator, override

import torch

from llmoe.backbone import VLM
from llmoe.benchmarks.prototype import Benchmark, DataType
from llmoe.config import Config
from llmoe.flops import FlopCounter, theoretical_flops
from llmoe.performance import GenerateTrace, generation_module, synchronize_cuda, trace_generate
from llmoe.utils import coerce_image_tensor


CASES_FILE = "varest_benchmark.json"
EXPECTED_CASES = 180
DEFAULT_WARMUP_RUNS = 3
DEFAULT_TIMING_REPEATS = 3
DEFAULT_MEMORY_INTERVAL_MS = 1.0
MIB = 1024 * 1024


class _GenerateMeter:
    """A `generation_profiler` that keeps the trace of the one generate call it sees."""

    def __init__(self, flop_counter: FlopCounter | None = None) -> None:
        self._flop_counter = flop_counter
        self.trace: GenerateTrace | None = None

    @contextmanager
    def measure_generate(self, model: Any) -> Iterator[None]:
        with trace_generate(model, self._flop_counter) as trace:
            self.trace = trace
            yield


class _VisionTokenTrace:
    """Vision tokens entering each decoder layer during one prefill (batch size 1).

    The text-token count comes from the input-embedding call that carries image
    tokens; a layer's vision tokens are its input length minus that count. When no
    image token reaches the embedding (a compressor that feeds the decoder its own
    embeddings, e.g. Varest), the backbone's ``vision_tokens_per_layer()`` report is
    used instead; ``None`` when neither is available.
    """

    def __init__(self, model: torch.nn.Module, report: Any | None = None) -> None:
        self._report = report if callable(report) else None
        llm = getattr(model, "llm", None)  # CORE's LoRA-wrapped decoder
        self._lm = model if llm is None else generation_module(llm)
        image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
        if image_token_id is None and hasattr(model, "_image_token"):  # CORE
            image_token_id = model._image_token()
        self._image_token_id = image_token_id
        text_config = self._lm.config.get_text_config()
        self.hidden_size = int(text_config.hidden_size)
        self.intermediate_size = int(text_config.intermediate_size)
        self._layers = list(self._lm.get_decoder().layers)
        self._text_tokens: int | None = None
        self._layer_lengths: dict[int, int] = {}
        self._handles: list[Any] = []

    @classmethod
    def for_backbone(cls, backbone: Any) -> "_VisionTokenTrace | None":
        model = getattr(backbone, "model", None)
        lm = getattr(model, "llm", model)
        if not all(hasattr(lm, name) for name in ("get_input_embeddings", "get_decoder", "config")):
            return None
        return cls(model, getattr(backbone, "vision_tokens_per_layer", None))

    def __enter__(self) -> "_VisionTokenTrace":
        self._text_tokens = None
        self._layer_lengths = {}
        self._handles.append(self._lm.get_input_embeddings().register_forward_pre_hook(self._on_embed))
        for index, layer in enumerate(self._layers):
            self._handles.append(layer.register_forward_pre_hook(partial(self._on_layer, index), with_kwargs=True))
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _on_embed(self, module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        if self._text_tokens is not None or self._image_token_id is None or not args or not torch.is_tensor(args[0]):
            return
        image_tokens = int((args[0] == self._image_token_id).sum())
        if image_tokens:
            self._text_tokens = args[0].numel() - image_tokens

    def _on_layer(self, index: int, module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if index not in self._layer_lengths:
            hidden_states = args[0] if args else kwargs["hidden_states"]
            self._layer_lengths[index] = int(hidden_states.shape[-2])

    @property
    def vision_tokens_per_layer(self) -> list[int] | None:
        if self._text_tokens is None:
            reported = self._report() if self._report is not None else None
            return None if reported is None else [int(n) for n in reported]
        return [
            max(self._layer_lengths.get(index, self._text_tokens) - self._text_tokens, 0)
            for index in range(len(self._layers))
        ]


class _MemorySampler:
    """Peak and time-weighted mean of allocated memory over one run, sampled from a background thread.

    ``read`` returns the allocated bytes now (``torch.cuda.memory_allocated`` on CUDA). It is polled every
    ``interval`` seconds on the host clock, plus once at entry and once at exit. A sample holds until the next, so the
    mean is the area under that step function divided by the run's length. The allocator updates its counter when a
    tensor is created or freed on the host, so this is the host-side allocation timeline, not a device trace. The peak
    comes from ``peak`` (``torch.cuda.max_memory_allocated`` after a reset) when given, so short spikes between
    samples are not missed.
    """

    def __init__(self, read: Callable[[], int], interval: float, *, reset_peak: Callable[[], None] | None = None,
                 peak: Callable[[], int] | None = None) -> None:
        self._read, self._interval = read, float(interval)
        self._reset_peak, self._peak = reset_peak, peak
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, int]] = []
        self.baseline_bytes = 0
        self.peak_bytes = 0

    def _poll(self) -> None:
        while not self._stop.wait(self._interval):
            self.samples.append((perf_counter(), int(self._read())))

    def __enter__(self) -> "_MemorySampler":
        synchronize_cuda()
        if self._reset_peak is not None:
            self._reset_peak()
        self.baseline_bytes = int(self._read())
        self.samples = [(perf_counter(), self.baseline_bytes)]
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, name="varest-bench-memory", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        synchronize_cuda()
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.samples.append((perf_counter(), int(self._read())))
        sampled_peak = max(value for _, value in self.samples)
        self.peak_bytes = max(sampled_peak, int(self._peak())) if self._peak is not None else sampled_peak

    @property
    def mean_bytes(self) -> float:
        """Time-weighted mean: each sample holds until the next one."""
        span = self.samples[-1][0] - self.samples[0][0]
        if span <= 0:
            return float(self.samples[-1][1])
        area = sum(value * (later - now) for (now, value), (later, _) in zip(self.samples, self.samples[1:]))
        return area / span

    def report(self) -> dict[str, Any]:
        mean, base = self.mean_bytes, self.baseline_bytes
        return {
            "memory_baseline_mib": base / MIB,
            "memory_mean_mib": mean / MIB,
            "memory_peak_mib": self.peak_bytes / MIB,
            "memory_mean_above_baseline_mib": (mean - base) / MIB,
            "memory_peak_above_baseline_mib": (self.peak_bytes - base) / MIB,
            "memory_samples": len(self.samples),
        }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _mean_of(records: list[dict[str, Any]], key: str, scale: float = 1.0) -> float | None:
    values = [record["efficiency"][key] for record in records if record["efficiency"].get(key) is not None]
    return mean(values) / scale if values else None


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [record["efficiency"]["latency_seconds"] for record in records]
    peaks = [record["efficiency"]["peak_memory_mib"] for record in records
             if record["efficiency"].get("peak_memory_mib") is not None]
    return {
        "n": len(records),
        "latency_seconds_mean": mean(latencies),
        "latency_seconds_median": median(latencies),
        "latency_seconds_p90": _percentile(latencies, 0.9),
        "prefill_seconds_mean": _mean_of(records, "prefill_seconds"),
        "decode_seconds_mean": _mean_of(records, "decode_seconds"),
        "outside_generate_seconds_mean": _mean_of(records, "outside_generate_seconds"),
        "decode_ms_per_token_mean": _mean_of(records, "decode_ms_per_token"),
        "generated_tokens_mean": _mean_of(records, "generated_tokens"),
        "theoretical_tflops_mean": _mean_of(records, "theoretical_tflops"),
        "tflops_mean": _mean_of(records, "flops", 1e12),
        "prefill_tflops_mean": _mean_of(records, "prefill_flops", 1e12),
        "decode_tflops_mean": _mean_of(records, "decode_flops", 1e12),
        "outside_generate_tflops_mean": _mean_of(records, "outside_generate_flops", 1e12),
        "peak_memory_mib_max": max(peaks) if peaks else None,
        "peak_memory_mib_mean": mean(peaks) if peaks else None,
        "memory_mean_mib_mean": _mean_of(records, "memory_mean_mib"),
        "memory_peak_mib_mean": _mean_of(records, "memory_peak_mib"),
        "memory_mean_above_baseline_mib_mean": _mean_of(records, "memory_mean_above_baseline_mib"),
        "memory_peak_above_baseline_mib_mean": _mean_of(records, "memory_peak_above_baseline_mib"),
    }


class VarestBenchBenchmark(Benchmark):
    """Per-query latency and measured FLOPs over the Varest efficiency sweeps."""

    measures_performance = True

    def __init__(self, config: Config, backbone: type[VLM]) -> None:
        super().__init__(config, backbone, paired=False)
        self._cases: list[dict[str, Any]] | None = None
        self._warmed_up = False
        self._vision_trace: _VisionTokenTrace | None = None
        self._uncounted_ops: set[str] = set()
        self._dataset_length = len(self._load_cases())

    @property
    def _warmup_runs(self) -> int:
        return int(self._config.custom_args.get("varest_bench_warmup", DEFAULT_WARMUP_RUNS))

    @property
    def _timing_repeats(self) -> int:
        return int(self._config.custom_args.get("varest_bench_repeats", DEFAULT_TIMING_REPEATS))

    @property
    def _count_flops(self) -> bool:
        return bool(self._config.custom_args.get("varest_bench_flops", True))

    @property
    def _sample_memory(self) -> bool:
        return bool(self._config.custom_args.get("varest_bench_memory", True)) and torch.cuda.is_available()

    @property
    def _memory_interval_ms(self) -> float:
        return float(self._config.custom_args.get("varest_bench_memory_interval_ms", DEFAULT_MEMORY_INTERVAL_MS))

    def _load_cases(self) -> list[dict[str, Any]]:
        if self._cases is not None:
            return self._cases
        root = Path(self._config.dataset_dir)
        cases_file = root / CASES_FILE
        if not cases_file.exists():
            raise FileNotFoundError(f"Varest benchmark cases not found: {cases_file}")
        cases = json.loads(cases_file.read_text(encoding="utf-8"))
        if self._config.dataset_limit is None and len(cases) != EXPECTED_CASES:
            raise ValueError(f"Varest benchmark expected {EXPECTED_CASES} cases, got {len(cases)} from {cases_file}.")
        for case in cases:
            if case.get("prompt_type") != "short_answer":
                raise ValueError(f"Unsupported prompt_type {case.get('prompt_type')!r} for {case['question_id']}.")
            if not (root / case["image"]).exists():
                raise FileNotFoundError(f"Varest benchmark image not found: {root / case['image']}")
        if self._config.dataset_limit is not None:
            cases = cases[: self._config.dataset_limit]
        self._cases = cases
        print(f"[info] VarestBench: using {cases_file} ({len(cases)} cases)")
        return cases

    @override
    def validate_config(self) -> None:
        if self._warmup_runs < 0:
            raise ValueError("varest_bench_warmup must be >= 0")
        if self._timing_repeats < 1:
            raise ValueError("varest_bench_repeats must be >= 1")
        if self._memory_interval_ms <= 0:
            raise ValueError("varest_bench_memory_interval_ms must be > 0")

    @override
    def load_dataset(self) -> Iterator[dict[str, DataType]]:
        root = Path(self._config.dataset_dir)
        for case in self._load_cases():
            image_path = root / case["image"]
            yield {
                "question_id": case["question_id"],
                "item_id": f"varest_bench/{case['question_id']}",
                "category": case["category"],
                "image_path": str(image_path),
                "question": self.format_prompt(case["text"]),
                "question_raw": case["text"],
                "answer": case["answer"],
                "answers": case["answers"],
                "image": coerce_image_tensor(image_path, source="VarestBench"),
                "record_extra": {
                    "level": case["level"],
                    "level_index": case["level_index"],
                    "dataset": case["source"]["dataset"],
                    "workload": case["workload"],
                },
            }

    @override
    def format_prompt(self, raw_question: str) -> str:
        return self.format_short_answer_prompt(raw_question)

    @override
    def generate(self, backbone: VLM, batch: list[dict[str, DataType]], *, max_new_tokens: int,
                 generation_profiler: Any | None = None, **generation_kwargs: Any) -> list[str]:
        if len(batch) != 1:
            raise RuntimeError(f"VarestBench measures one query at a time, got a batch of {len(batch)}.")
        item = batch[0]

        def run(meter: _GenerateMeter | None) -> list[str]:
            return backbone(
                images=[item["image"]],
                prompts=[item["question"]],
                max_new_tokens=max_new_tokens,
                generation_profiler=meter,
                **generation_kwargs,
            )

        if not self._warmed_up:
            for _ in range(self._warmup_runs):
                run(None)
            self._warmed_up = True

        timed: list[tuple[float, GenerateTrace | None, float | None]] = []
        raws: list[str] | None = None
        for _ in range(self._timing_repeats):
            meter = _GenerateMeter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            synchronize_cuda()
            start = perf_counter()
            outputs = run(meter)
            synchronize_cuda()
            seconds = perf_counter() - start
            peak = torch.cuda.max_memory_allocated() / MIB if torch.cuda.is_available() else None
            timed.append((seconds, meter.trace, peak))
            raws = raws if raws is not None else outputs

        latency, trace, peak = sorted(timed, key=lambda run_: run_[0])[(len(timed) - 1) // 2]
        efficiency: dict[str, Any] = {
            "latency_seconds": latency,
            "latency_runs_seconds": [run_[0] for run_ in timed],
            "generate_seconds": None if trace is None else trace.wall_seconds,
            "prefill_seconds": None if trace is None else trace.prefill_seconds,
            "decode_seconds": None if trace is None else trace.decode_seconds,
            "outside_generate_seconds": None if trace is None else latency - trace.wall_seconds,
            "decode_ms_per_token": (
                1000.0 * trace.decode_seconds / trace.decode_calls
                if trace is not None and trace.decode_calls else None
            ),
            "generated_tokens": None if trace is None else trace.forward_calls,
            "peak_memory_mib": peak,
        }

        if self._count_flops:
            if self._vision_trace is None:
                self._vision_trace = _VisionTokenTrace.for_backbone(backbone)
            vision_trace = self._vision_trace
            counter = FlopCounter()
            meter = _GenerateMeter(counter)
            with counter:
                if vision_trace is None:
                    flop_raws = run(meter)
                else:
                    with vision_trace:
                        flop_raws = run(meter)
            vision_tokens = None if vision_trace is None else vision_trace.vision_tokens_per_layer
            flop_trace = meter.trace
            in_generate = (
                0 if flop_trace is None or flop_trace.prefill_flops is None
                else flop_trace.prefill_flops + flop_trace.decode_flops
            )
            efficiency.update({
                "theoretical_tflops": (
                    None if vision_tokens is None
                    else theoretical_flops(vision_tokens, vision_trace.hidden_size, vision_trace.intermediate_size) / 1e12
                ),
                "vision_tokens_per_layer": vision_tokens,
                "flops": counter.total,
                "prefill_flops": None if flop_trace is None else flop_trace.prefill_flops,
                "decode_flops": None if flop_trace is None else flop_trace.decode_flops,
                "outside_generate_flops": counter.total - in_generate,
                "flops_by_module": counter.by_module(),
                "flop_pass_matches_output": flop_raws == raws,
                "flop_pass_generated_tokens": None if flop_trace is None else flop_trace.forward_calls,
            })
            self._uncounted_ops.update(counter.uncounted_ops)

        if self._sample_memory:
            device = torch.cuda.current_device()
            sampler = _MemorySampler(
                partial(torch.cuda.memory_allocated, device), self._memory_interval_ms / 1000.0,
                reset_peak=partial(torch.cuda.reset_peak_memory_stats, device),
                peak=partial(torch.cuda.max_memory_allocated, device),
            )
            with sampler:
                memory_raws = run(None)
            efficiency.update(sampler.report())
            efficiency["memory_pass_matches_output"] = memory_raws == raws

        item["record_extra"] = {**item.get("record_extra", {}), "efficiency": efficiency}
        return raws

    @override
    def parse_prediction(self, raw_text: str, row: dict[str, DataType]) -> str:
        return str(raw_text).strip()

    @override
    def is_correct(self, prediction: str, row: dict[str, DataType]) -> None:
        return None

    @override
    def compute_metrics(self, records: list[dict[str, DataType]]) -> dict[str, Any]:
        per_level: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        per_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            per_level[str(record["category"])][str(record["level"])].append(record)
            per_dataset[str(record["dataset"])].append(record)

        by_module: dict[str, list[float]] = defaultdict(list)
        for record in records:
            for name, flops in (record["efficiency"].get("flops_by_module") or {}).items():
                by_module[name].append(flops / 1e12)

        return {
            "num_examples": len(records),
            "metrics": _summarize(records) if records else {"n": 0},
            "tflops_by_module_mean": {name: sum(values) / len(records) for name, values in by_module.items()},
            "per_level": {
                category: {
                    level: _summarize(level_records)
                    for level, level_records in sorted(levels.items(), key=lambda item: float(item[0]))
                }
                for category, levels in sorted(per_level.items())
            },
            "per_dataset": {name: _summarize(group) for name, group in sorted(per_dataset.items())},
            "measurement": {
                "batch_size": 1,
                "warmup_runs": self._warmup_runs,
                "timing_repeats": self._timing_repeats,
                "latency_statistic": "run with the median end-to-end latency",
                "latency_scope": "backbone call: preprocessing, generation and decoding to text",
                "flops_counted": self._count_flops,
                "memory_sampled": self._sample_memory,
                "memory_interval_ms": self._memory_interval_ms if self._sample_memory else None,
                "memory_scope": "torch.cuda.memory_allocated in a separate untimed run; mean = time-weighted over "
                                "host-clock samples; baseline = allocated before the run (weights, caches)",
                "theoretical_flops": "sum over decoder layers of 4nd^2 + 2n^2d + 3ndm, n = vision tokens "
                                     "(HiDrop, arXiv:2602.23699)",
                "flop_counter": "torch.utils.flop_counter (llmoe.flops); 2 FLOPs per MAC; "
                                "matmul, convolution and attention only; full (non-causal) attention",
                "uncounted_ops": sorted(self._uncounted_ops),
                "flop_pass_mismatches": sum(
                    1 for record in records if record["efficiency"].get("flop_pass_matches_output") is False
                ),
                "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        }

    @override
    def summary_line(self, metrics_dict: dict[str, Any]) -> str:
        metrics = metrics_dict["metrics"]
        tflops = metrics.get("tflops_mean")
        return (
            f"latency={metrics['latency_seconds_mean']:.4f}s  "
            f"tflops={'n/a' if tflops is None else f'{tflops:.4f}'}  "
            f"n={metrics['n']}"
        )
