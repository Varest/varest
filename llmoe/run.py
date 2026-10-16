from dataclasses import asdict
from datetime import datetime
from gc import collect
from json import dumps
from os.path import exists
from pathlib import Path
from time import perf_counter
from typing import Sequence, Callable

import torch
from mle.sanity import check_environment, print_environment_check_results, check_satisfied_or_throw
from mle.vars import ExpConfig
from tqdm import tqdm

from llmoe.backbone import BackboneLiteral, VLM
from llmoe.compressor import CompressorLiteral, COMPRESSOR_TRAINERS, COMPRESSOR_TUNERS
from llmoe.benchmarks import BENCHMARKS, Benchmark, DataType, DATASET_REQUIRED_BY_BENCHMARK, MAX_NEW_TOKENS_BY_BENCHMARK
from llmoe.compressor import TokenCompressor
from llmoe.config import Config
from llmoe.performance import PerformanceProfiler
from llmoe.resolvers import resolve_backbone, resolve_benchmarks
from llmoe.utils import round_numeric
from llmoe.viz import visualize, visualize_saliency



def run_training(backbone: BackboneLiteral, compressor: CompressorLiteral, config: Config) -> None:
    fn_cd, trainer = COMPRESSOR_TRAINERS[compressor]
    env_check_results = check_environment(config, fn_cd=fn_cd, fn_cpd=lambda _: "OK (N/A)")
    print_environment_check_results(env_check_results)
    check_satisfied_or_throw(env_check_results, True, False, True)
    model = resolve_backbone(backbone, config.custom_args.get("backbone_args", {}))
    model.validate_compressor(compressor)
    trainer(model, config)


def run_finetuning(backbone: BackboneLiteral, compressor: CompressorLiteral, config: Config) -> None:
    fn_cd, tuner = COMPRESSOR_TUNERS[compressor]
    env_check_results = check_environment(config, fn_cd=fn_cd, fn_cpd=lambda _: "OK (N/A)")
    print_environment_check_results(env_check_results)
    check_satisfied_or_throw(env_check_results, True, False, True)
    model = resolve_backbone(backbone, config.custom_args.get("backbone_args", {}))
    model.validate_compressor(compressor)
    tuner(model, config)


def make_fn_cd_for_benchmarks(benchmarks: Sequence[type[Benchmark]]) -> Callable[[ExpConfig], str]:
    def check_datasets(config: ExpConfig) -> str:
        missing = []
        for benchmark in benchmarks:
            required_dataset = DATASET_REQUIRED_BY_BENCHMARK[benchmark]
            if not exists(f"{config.input_dir}/{required_dataset}"):
                missing.append(required_dataset)
        return f"Missing datasets under {config.input_dir}: {', '.join(missing)}" if missing else "OK"

    return check_datasets


def run_benchmark(benchmark: type[Benchmark], backbone: BackboneLiteral, compressor: TokenCompressor | None, 
                  config: Config, *, backbone_instance: VLM | None = None) -> None:
    benchmark_key = next(name for name, value in BENCHMARKS.items() if value is benchmark)
    benchmark_config_dict = asdict(config)
    benchmark_config_dict["custom_args"] = dict(config.custom_args)
    benchmark_config = Config(**benchmark_config_dict)
    benchmark_config.benchmark = benchmark_key
    benchmark_config.dataset_name = DATASET_REQUIRED_BY_BENCHMARK[benchmark]
    benchmark_config.custom_args.setdefault("dataset_limit", benchmark_config.dataset_limit)

    batch_size = getattr(benchmark_config, "batch_size", None)
    if batch_size is None:
        batch_size = benchmark_config.custom_args.get("batch_size", 16)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if benchmark.measures_performance and batch_size != 1:
        print(f"[info] {benchmark_key}: measures per-query latency, so it runs at batch_size 1 "
              f"(configured {batch_size}).")
        batch_size = 1

    max_new_tokens = getattr(benchmark_config, "max_new_tokens", None)
    if max_new_tokens is None:
        max_new_tokens = benchmark_config.custom_args.get(
            "max_new_tokens", MAX_NEW_TOKENS_BY_BENCHMARK.get(benchmark, 128),
        )
    max_new_tokens = int(max_new_tokens)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    do_sample = bool(
        getattr(benchmark_config, "do_sample", benchmark_config.custom_args.get("do_sample", False))
    )
    num_beams = int(
        getattr(benchmark_config, "num_beams", benchmark_config.custom_args.get("num_beams", 1))
    )
    use_cache = bool(
        getattr(benchmark_config, "use_cache", benchmark_config.custom_args.get("use_cache", True))
    )
    temperature = float(
        getattr(benchmark_config, "temperature", benchmark_config.custom_args.get("temperature", 0.0))
    )
    top_p = getattr(benchmark_config, "top_p", benchmark_config.custom_args.get("top_p"))

    owns_backbone = backbone_instance is None
    if backbone_instance is None:
        backbone_instance = resolve_backbone(backbone, benchmark_config.custom_args.get("backbone_args", {}))

    performance_profiler = (
        PerformanceProfiler(enabled=False) if benchmark.measures_performance
        else PerformanceProfiler.from_custom_args(benchmark_config.custom_args)
    )
    records: list[dict[str, DataType]] = []
    batch: list[dict[str, DataType]] = []
    total_inference_time_seconds = 0.0

    def process_batch() -> None:
        nonlocal total_inference_time_seconds

        generation_kwargs: dict[str, object] = {
            "num_beams": num_beams,
            "use_cache": use_cache,
        }
        if do_sample:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            if top_p is not None:
                generation_kwargs["top_p"] = top_p

        inference_start = perf_counter()
        try:
            with performance_profiler.measure_batch(len(batch)):
                raws = benchmark_instance.generate(
                    backbone_instance,
                    batch,
                    max_new_tokens=max_new_tokens,
                    generation_profiler=performance_profiler,
                    **generation_kwargs,
                )
        finally:
            total_inference_time_seconds += perf_counter() - inference_start

        performance_profiler.step(len(batch))
        if compressor is None:
            compressor_metadata = [{} for _ in batch]
        else:
            compressor_metadata = compressor.get_benchmark_record_metadata(len(batch))
            if len(compressor_metadata) != len(batch):
                raise RuntimeError(
                    "Compressor returned benchmark metadata for "
                    f"{len(compressor_metadata)} samples, expected {len(batch)}."
                )
        for item, raw, metadata in zip(batch, raws, compressor_metadata, strict=True):
            if not isinstance(metadata, dict):
                raise TypeError(
                    "Compressor benchmark metadata entries must be dictionaries, got "
                    f"{type(metadata).__name__}."
                )
            prediction = benchmark_instance.parse_prediction(raw, item)
            correct = benchmark_instance.is_correct(prediction, {**item, "prediction": prediction})
            record = {
                "question_id": item.get("question_id"),
                "item_id": item.get("item_id"),
                "category": item.get("category"),
                "image_path": item.get("image_path"),
                "question": item.get("question"),
                "question_raw": item.get("question_raw") or item.get("question_mme"),
                "answer": item.get("answer"),
                "answers": item.get("answers"),
                "prediction_raw": raw,
                "prediction": prediction,
                "correct": None if correct is None else bool(correct),
            }
            extra = item.get("record_extra")
            if isinstance(extra, dict):
                for key, value in extra.items():
                    record.setdefault(key, value)
            for key, value in metadata.items():
                if key in record:
                    raise RuntimeError(
                        f"Compressor benchmark metadata cannot replace result field {key!r}."
                    )
                record[key] = value
            records.append(record)

    try:
        if owns_backbone and compressor is not None:
            backbone_instance.validate_compressor(compressor)
            compressor.to(device=backbone_instance._device, dtype=backbone_instance._dtype)
            compressor.install(backbone_instance)

        benchmark_instance = benchmark(benchmark_config, type(backbone_instance))
        benchmark_instance.validate_config()
        if (
                benchmark_instance.paired
                and benchmark_instance.dataset_length is not None
                and benchmark_instance.dataset_length % 2 != 0
        ):
            raise ValueError(
                f"{benchmark_key} rows are paired by image. Use an even dataset limit; "
                f"got {benchmark_instance.dataset_length}."
            )

        if benchmark_config.visualize:
            visualize(backbone_instance, compressor, benchmark_config)
            return
        if benchmark_config.visualize_saliency:
            visualize_saliency(backbone_instance, compressor, benchmark_config)
            return

        with performance_profiler:
            with tqdm(total=benchmark_instance.dataset_length, desc=benchmark_key, unit="sample") as progress:
                for row in benchmark_instance.load_dataset():
                    batch.append(row)
                    if len(batch) < batch_size:
                        continue
                    process_batch()
                    progress.update(len(batch))
                    batch.clear()
                if batch:
                    process_batch()
                    progress.update(len(batch))

        benchmark_metrics = benchmark_instance.compute_metrics(records)
        output_method = compressor.name if compressor is not None else "full_token"
        out_dir = Path(benchmark_config.output_dir) / benchmark_key / output_method
        out_path = out_dir / (
            f"{benchmark_key}_{output_method}_"
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        )
        artifact_metrics = benchmark_instance.build_artifacts(records, compressor, out_path)
        if artifact_metrics:
            benchmark_metrics = {**benchmark_metrics, **artifact_metrics}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        performance_profile = performance_profiler.write_artifacts(
            out_path.with_name(f"{out_path.stem}_profile_summary.txt"),
        )
        overall = {
            "total_examples": len(records),
            "total_inference_time_seconds": total_inference_time_seconds,
            "avg_inference_time_per_example_seconds": (
                total_inference_time_seconds / len(records) if records else None
            ),
            "evaluation_date": datetime.now().isoformat(timespec="seconds"),
            "output_path": str(out_path),
        }
        if performance_profile is not None:
            overall["performance_profile"] = performance_profile
        result = round_numeric(
            {
                benchmark_key: benchmark_metrics,
                "overall": overall,
                "config": {
                    "experiment_name": benchmark_config.experiment_name,
                    "benchmark": benchmark_key,
                    "backbone": backbone,
                    "dataset": benchmark_config.dataset_name,
                    "dataset_dir": benchmark_config.dataset_dir,
                    "dataset_limit": benchmark_config.dataset_limit,
                    "compressor": compressor.name if compressor is not None else None,
                    "device": str(benchmark_config.device),
                    "batch_size": batch_size,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_beams": num_beams,
                    "use_cache": use_cache,
                    "custom_args": benchmark_config.custom_args,
                },
                "predictions": records,
            },
            digits=4,
        )
        out_path.write_text(dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"{benchmark_key}: {benchmark_instance.summary_line(benchmark_metrics)}  ->  {out_path}")
    finally:
        if owns_backbone:
            try:
                uninstall = getattr(compressor, "uninstall", None) if compressor is not None else None
                if uninstall is not None:
                    uninstall()
            finally:
                del backbone_instance
                collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


def run_benchmarks(benchmarks: Sequence[type[Benchmark]] | Sequence[str], backbone: BackboneLiteral,
                   compressor: TokenCompressor | None, config: Config) -> None:
    benchmarks = resolve_benchmarks(benchmarks)
    env_check_results = check_environment(config, fn_cd=make_fn_cd_for_benchmarks(benchmarks), fn_cpd=lambda _: "OK (N/A)")
    print_environment_check_results(env_check_results)
    check_satisfied_or_throw(env_check_results, True, False, True)
    model = resolve_backbone(backbone, config.custom_args.get("backbone_args", {}))
    try:
        if compressor is not None:
            model.validate_compressor(compressor)
            compressor.to(device=model._device, dtype=model._dtype)
            compressor.install(model)
        for benchmark_type in benchmarks:
            run_benchmark(benchmark_type, backbone, compressor, config, backbone_instance=model)
    finally:
        try:
            uninstall = getattr(compressor, "uninstall", None) if compressor is not None else None
            if uninstall is not None:
                uninstall()
        finally:
            del model
            collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
