from argparse import ArgumentParser
from dataclasses import asdict
from json import load
from os import environ

from erbium.api import ResourceMonitor
from mle.vars import erbium_config, slurm_config, local_config
from yaml import safe_load, safe_dump

from llmoe.backbone import BACKBONES
from llmoe.benchmarks import BENCHMARKS, Benchmark
from llmoe.compressor import COMPRESSORS, TokenCompressor
from llmoe.config import Config
from llmoe.resolvers import resolve_compressor
from llmoe.run import run_training, run_finetuning, run_benchmarks
from llmoe.weights import ARTIFACTS, check_weights


def __entry__(*, custom_benchmarks: dict[str, type[Benchmark]] | None = None,
              custom_compressors: dict[str, type[TokenCompressor]] | None = None) -> None:
    if custom_benchmarks:
        BENCHMARKS.update(custom_benchmarks)
    if custom_compressors:
        COMPRESSORS.update(custom_compressors)
    parser = ArgumentParser(prog="llmoe", description="LLMOE",
                            epilog="GitHub: https://github.com/ProjectNeura/LLMOE")
    parser.add_argument("-n", "--experiment_name", default=None, help="Experiment name")
    parser.add_argument("-c", "--config", choices=("slurm", "erbium", "local"), default="erbium",
                        help="Configuration to use")
    parser.add_argument("--suser", help="SLURM username")
    parser.add_argument("--custom_args", default=None, help="Custom arguments to pass to the engine")
    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test")
    subparsers = parser.add_subparsers(dest="system", required=True)
    # training / pretraining
    subparsers.add_parser("tr", help="Train a compressor")
    # fine-tuning
    subparsers.add_parser("ft", help="Fine-tune a compressor")
    # benchmarking
    bm_parser = subparsers.add_parser("bm", help="Run benchmarks for a compressor")
    bm_parser.add_argument("benchmarks", nargs="+", choices=BENCHMARKS.keys(),
                           help="Names of the benchmarks to run")
    bm_parser.add_argument("--visualize", action="store_true",
                           help="Render token-compression visualizations instead of running the benchmark")
    bm_parser.add_argument("--visualize-saliency", action="store_true",
                           help="Render per-word attention saliency maps instead of running the benchmark "
                                "(works with or without a compressor)")
    # weight bundle management
    weights_parser = subparsers.add_parser(
        "weights", help="Check the released method weights, and fetch the missing ones")
    weights_parser.add_argument("methods", nargs="*", choices=sorted({name for name, *_ in ARTIFACTS}),
                                help="Restrict the report to these methods; all of them by default")
    weights_parser.add_argument("--download", action="store_true",
                                help="Download whatever is missing (asks before each artifact)")
    # shared arguments
    parser.add_argument("--backbone", choices=BACKBONES.keys(), default="llava", help="Backbone to use")
    parser.add_argument("--compressor", choices=("none", *COMPRESSORS.keys()), default=None,
                           help="Token compressor to use")
    # config overrides
    parser.add_argument("--root_dir", default=None, help="Input directory")
    parser.add_argument("--input_dir", default=None, help="Input directory")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--hf_home", default=None,
                        help="Hugging Face cache root, holding both the model cache and the "
                             "llmoe_weights bundle; defaults to <root_dir>/output/hf_cache")
    parser.add_argument("--dataset-limit", default=None, type=int, help="Limit the number of samples to process")
    parser.add_argument("--device", default=None, help="Device to use, auto by default")
    args = parser.parse_args()
    experiment_name = args.experiment_name or f"llmoe-{args.system}-{args.config}"
    match args.config:
        case "erbium":
            config = erbium_config(experiment_name, "dataset_undefined", root_dir=args.root_dir)
        case "slurm":
            if not args.suser:
                raise ValueError("SLURM username is required for SLURM configuration")
            config = slurm_config(
                experiment_name, args.suser, "dataset_undefined", root_dir=args.root_dir
            )
        case "local":
            config = local_config(experiment_name, "dataset_undefined", root_dir=args.root_dir)
        case _:
            raise ValueError(f"Unsupported configuration: {args.config}")
    if args.input_dir:
        config._input_dir = args.input_dir
    if args.output_dir:
        config._output_dir = args.output_dir
    else:
        config._output_dir = f"{config.output_dir}/LLMOE"
    custom_args = {}
    if args.custom_args:
        with open(args.custom_args) as f:
            if args.custom_args.endswith(".json"):
                custom_args.update(load(f))
            elif args.custom_args.endswith(".yaml"):
                custom_args.update(safe_load(f))
            else:
                raise ValueError(f"Unsupported custom arguments file type: {args.custom_args}, expected JSON or YAML")
    config = Config(**asdict(config), custom_args=custom_args,
                    _hf_home=args.hf_home or environ.get("HF_HOME"))
    # `llmoe.bootstrap` normally did this before `huggingface_hub` was imported;
    # repeating it is a no-op then, and the safety net when `__entry__` is called
    # from another program's `main` instead of the CLI.
    config.setup_hf_env()
    # Before anything writes to `output_dir`: managing the weight bundle is not an
    # experiment and must work on a machine that has no results directory yet.
    if args.system == "weights":
        raise SystemExit(check_weights(args.methods or None, download=args.download))
    if args.dataset_limit is not None:
        config.dataset_limit = args.dataset_limit
    if args.device:
        config.device = args.device
    config.visualize = getattr(args, "visualize", False)
    config.visualize_saliency = getattr(args, "visualize_saliency", False)
    if args.smoke_test:
        monitor = ResourceMonitor(f"{config.output_dir}/{config.experiment_name}-resource_monitor")
        monitor.start()
    with open(f"{config.output_dir}/config.yaml", "w") as f:
        safe_dump(asdict(config), f)
    match args.system:
        case "tr":
            run_training(args.backbone, args.compressor, config)
        case "ft":
            run_finetuning(args.backbone, args.compressor, config)
        case "bm":
            compressor = resolve_compressor(args.compressor, compressor_args=custom_args.get("compressor_args", {}))
            if config.visualize and compressor is None:
                raise ValueError("Visualization requires a compressor, pass --compressor and try again.")
            run_benchmarks(args.benchmarks, args.backbone, compressor, config)
