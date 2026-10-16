from typing import Sequence, Any

from llmoe.backbone import VLM, BACKBONES, BackboneLiteral
from llmoe.benchmarks import BENCHMARKS, Benchmark
from llmoe.compressor import TokenCompressor, COMPRESSORS, CompressorLiteral


def resolve_backbone(backbone: BackboneLiteral, backbone_args: dict[str, Any]) -> VLM:
    return BACKBONES[backbone](**backbone_args)


def resolve_compressor(compressor: CompressorLiteral | str | None, compressor_args: dict[str, Any]) -> TokenCompressor | None:
    if compressor is None or compressor == "none":
        return None
    if compressor not in COMPRESSORS:
        choices = ", ".join(["none", *COMPRESSORS])
        raise ValueError(f"Unsupported compressor: {compressor}. Valid choices: {choices}")
    return COMPRESSORS[compressor](**compressor_args)


def resolve_benchmark(benchmark: type[Benchmark] | str) -> type[Benchmark]:
    return BENCHMARKS[benchmark] if isinstance(benchmark, str) else benchmark


def resolve_benchmarks(benchmarks: Sequence[type[Benchmark]] | Sequence[str]) -> list[type[Benchmark]]:
    return [resolve_benchmark(benchmark) for benchmark in benchmarks]
