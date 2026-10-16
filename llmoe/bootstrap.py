"""Settle the Hugging Face cache root before anything imports ``huggingface_hub``.

``huggingface_hub`` freezes ``HF_HOME`` and every path derived from it at import
time, and ``llmoe/__init__.py`` pulls in ``transformers`` -- so by the time the CLI
has parsed its arguments it is far too late to redirect the cache. This module is
therefore imported first, from the top of the package, and does the one thing that
has to happen early: work out where this run's cache belongs and export
``HF_HOME``.

The rules, in order:

* An ``HF_HOME`` already in the environment always wins. Nothing here overrides an
  operator who set it in the shell or a scheduler job.
* Otherwise, for a ``python -m llmoe`` / ``llmoe`` invocation, the cache is
  ``<root_dir>/output/hf_cache`` -- shared by every experiment under that root,
  and where ``llmoe.weights`` then looks for the ``llmoe_weights`` bundle.
  ``--hf_home`` overrides it.
* Otherwise (imported as a library) nothing is set and Hugging Face's own default
  applies.

Keep this module free of heavy imports; the whole point is that it runs before
them.
"""

import sys
from argparse import ArgumentParser
from os import environ
from os.path import basename
from typing import Sequence

__all__ = ["cache_root_from_args", "bootstrap_hf_home"]

# Mirrors mle.vars' backend roots. Duplicated rather than imported because
# importing anything real here would defeat the purpose of running first.
_BACKEND_ROOTS = {"erbium": "/workspace", "local": "~/Documents"}


def _cli_arguments(argv: Sequence[str]) -> list[str] | None:
    """The llmoe CLI's own arguments, or ``None`` if this is not the llmoe CLI.

    A test runner, a notebook, or any other program that imports ``llmoe`` must
    not have its command line read as ours, so each launch style is recognised
    explicitly:

    * ``python -m llmoe ...`` -- ``sys.argv[0]`` is still ``"-m"`` while the
      package is being imported (the real path is only substituted once
      ``__main__`` runs), so the module name is read from ``sys.orig_argv``.
      That also distinguishes us from ``python -m pytest``, which imports
      ``llmoe`` with argv shaped exactly the same way.
    * the ``llmoe`` console script.
    * ``python path/to/llmoe/__main__.py``.
    """
    if not argv:
        return None
    original = list(getattr(sys, "orig_argv", ()))
    if "-m" in original:
        index = original.index("-m")
        module = original[index + 1] if index + 1 < len(original) else ""
        return original[index + 2:] if module.split(".")[0] == "llmoe" else None
    name = basename(argv[0])
    if name == "llmoe" or (name == "__main__.py" and "llmoe" in argv[0].replace("\\", "/")):
        return list(argv[1:])
    return None


def cache_root_from_args(argv: Sequence[str] | None = None) -> str | None:
    """The ``HF_HOME`` a CLI invocation implies, or ``None`` if it implies none.

    Reads only the options that determine output paths, and tolerates everything
    else, so it stays correct without duplicating the real parser in
    ``llmoe.__entry__``.
    """
    arguments = _cli_arguments(list(sys.argv if argv is None else argv))
    if arguments is None:
        return None
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--hf_home", default=None)
    parser.add_argument("-c", "--config", default="erbium")
    parser.add_argument("--suser", default=None)
    parser.add_argument("--root_dir", default=None)
    try:
        args, _ = parser.parse_known_args(arguments)
    except SystemExit:  # a malformed argument; let the real parser report it
        return None
    if args.hf_home:
        return args.hf_home
    root = args.root_dir
    if root is None:
        if args.config == "slurm":
            if not args.suser:
                return None
            root = f"/scratch/{args.suser}"
        else:
            root = _BACKEND_ROOTS.get(args.config)
    if root is None:
        return None
    # Deliberately `<root>/output`, ignoring `--output_dir`: the cache belongs to
    # the machine, not to one experiment's results directory, so every run under a
    # root shares one copy of the models and the weight bundle. `--hf_home` moves
    # it for the rare case that needs moving.
    return f"{root}/output/hf_cache"


def bootstrap_hf_home(argv: Sequence[str] | None = None) -> str | None:
    """Export ``HF_HOME`` for this run if it is not already set.

    Returns the value in force afterwards, or ``None`` when Hugging Face's own
    default is left to apply.
    """
    existing = environ.get("HF_HOME")
    if existing:
        return existing
    cache_root = cache_root_from_args(argv)
    if cache_root:
        environ["HF_HOME"] = cache_root
    return cache_root
