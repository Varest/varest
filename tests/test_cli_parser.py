"""The CLI parser is built at import time, so anything it touches is load-bearing.

Widening the ``ARTIFACTS`` key to ``(method, backbone, budget)`` broke a two-element
unpack in ``__entry__``. Because the parser is constructed unconditionally before
any subcommand is dispatched, that took down *every* invocation -- ``--help``
included -- not just ``llmoe weights``. Nothing caught it: the compressor tests
never build the parser.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "llmoe", *args],
        cwd=REPO, capture_output=True, text=True, timeout=900,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO),
             "HOME": str(Path.home()), "HF_HUB_OFFLINE": "1"},
    )


def test_help_works():
    """The regression: --help alone died on the ARTIFACTS unpack."""
    result = _run("--help")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage: llmoe" in result.stdout
    assert "too many values to unpack" not in result.stderr


@pytest.mark.parametrize("subcommand", ["tr", "ft", "bm", "weights"])
def test_every_subcommand_parses(subcommand):
    """Every subcommand shares the parser, so all of them fell over together."""
    result = _run(subcommand, "--help")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "too many values to unpack" not in result.stderr


def test_weights_subcommand_lists_the_released_methods():
    """Its `methods` choices come straight from the ARTIFACTS keys."""
    result = _run("weights", "--help")
    assert result.returncode == 0, result.stderr[-2000:]
    for method in ("QuietPrune", "HiDrop", "DUET", "CORE", "VisionZip"):
        assert method in result.stdout


def test_entry_point_survives_a_wider_artifacts_key():
    """Whatever the key's arity, the parser must still build from it.

    Deliberately silent on *how*: a starred unpack and a helper both qualify.
    The key has already widened once (budget -> backbone, budget) and may again,
    so this pins the behaviour rather than the spelling.
    """
    from llmoe.weights import ARTIFACTS

    assert sorted({key[0] for key in ARTIFACTS}), "the registry should list methods"
    source = (REPO / "llmoe" / "__entry__.py").read_text()
    assert "for name, _ in ARTIFACTS" not in source, (
        "a fixed two-element unpack breaks the moment the key widens")
