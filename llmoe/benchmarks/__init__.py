from llmoe.benchmarks.ai2d import AI2DBenchmark
from llmoe.benchmarks.chartqa import ChartQABenchmark
from llmoe.benchmarks.countbench import CountBenchBenchmark
from llmoe.benchmarks.docvqa import DocVQABenchmark
from llmoe.benchmarks.gqa import GQABenchmark
from llmoe.benchmarks.mmbench import MMBenchBenchmark
from llmoe.benchmarks.mme import MMEBenchmark
from llmoe.benchmarks.mmstar import MMStarBenchmark
from llmoe.benchmarks.pope import POPEBenchmark
from llmoe.benchmarks.refcoco import RefCOCOBenchmark
from llmoe.benchmarks.textvqa import TextVQABenchmark
from llmoe.benchmarks.varest_bench import VarestBenchBenchmark
from llmoe.benchmarks.vstar import VStarBenchmark
from llmoe.benchmarks.prototype import DataType, Benchmark

BENCHMARKS: dict[str, type[Benchmark]] = {
    "ai2d": AI2DBenchmark,
    "chartqa": ChartQABenchmark,
    "countbench": CountBenchBenchmark,
    "docvqa": DocVQABenchmark,
    "gqa": GQABenchmark,
    "mmbench": MMBenchBenchmark,
    "mme": MMEBenchmark,
    "mmstar": MMStarBenchmark,
    "pope": POPEBenchmark,
    "refcoco": RefCOCOBenchmark,
    "textvqa": TextVQABenchmark,
    "varest_bench": VarestBenchBenchmark,
    "vstar": VStarBenchmark,
}
DATASET_REQUIRED_BY_BENCHMARK: dict[type[Benchmark], str] = {
    AI2DBenchmark: "ai2d",
    ChartQABenchmark: "ChartQA",
    CountBenchBenchmark: "CountBenchQA",
    DocVQABenchmark: "docvqa",
    GQABenchmark: "GQA",
    MMBenchBenchmark: "MMBench",
    MMEBenchmark: "MME_Benchmark_release_version",
    MMStarBenchmark: "MMStar",
    POPEBenchmark: "pope",
    RefCOCOBenchmark: "RefCOCO",
    TextVQABenchmark: "textvqa",
    VarestBenchBenchmark: "Varest_Benchmark",
    VStarBenchmark: "vstar-bench",
}
MAX_NEW_TOKENS_BY_BENCHMARK: dict[type[Benchmark], int] = {
    CountBenchBenchmark: 16,
    MMBenchBenchmark: 1024,
    VStarBenchmark: 16,
}
