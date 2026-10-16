"""LearnPruner's released LPMs: one per backbone, resolved at install."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from llmoe import weights as weights_module
from llmoe.compressor.learnpruner.learn_pruner import LearnPrunerCompressor
from llmoe.weights import artifact_for, backbones_for


BUDGETS = (32, 64, 128, 224)


def test_every_budget_resolves_to_the_one_release_per_backbone() -> None:
    """The budget is an inference-time knob, so it cannot select a file.

    LearnPruner trains its LPM against an argmax over the module's own scores,
    so the retention budget never enters the loss -- the released
    `*_training.json` records the budget each run was configured with, and runs
    at different budgets produced bitwise-identical weights. A per-budget key
    would therefore promise four checkpoints that do not exist.
    """
    for backbone, leaf in (("llava", "learnpruner_lpm_7b_b64.pt"),
                           ("llava-13b", "learnpruner_lpm_13b_b64.pt"),
                           ("llava-next-8b", "learnpruner_lpm_next.pt")):
        paths = {artifact_for("LearnPruner", budget, backbone).path for budget in BUDGETS}
        assert len(paths) == 1
        assert paths.pop().endswith(leaf)


def test_releases_are_not_interchangeable_across_backbones() -> None:
    assert backbones_for("LearnPruner") == ["llava", "llava-13b", "llava-next-8b"]
    paths = {artifact_for("LearnPruner", 64, backbone).path
             for backbone in backbones_for("LearnPruner")}
    assert len(paths) == 3
    # Ambiguous without a backbone: three releases, and the NeXT LPM scores a
    # 4096-wide packed AnyRes sequence where the LLaVA-1.5 ones score 1024-wide
    # raw CLIP features.
    with pytest.raises(KeyError, match="pass backbone="):
        artifact_for("LearnPruner", 64)


def _seed_checkpoint(tmp_path: Path, backbone: str) -> Path:
    seed = LearnPrunerCompressor(max_tokens=64, for_training=True)
    seed._installed_backbone_key = backbone
    path = tmp_path / "lpm.pt"
    seed.save_checkpoint(path)
    return path


def test_pretrained_true_fetches_the_backbone_release(tmp_path: Path, monkeypatch) -> None:
    """Resolution waits for install, where the backbone is finally known."""
    checkpoint = _seed_checkpoint(tmp_path, "llava-13b")
    seen: dict[str, object] = {}

    def fake_resolve(method, budget, *, pretrained=None, backbone=None, **kwargs):
        seen.update(method=method, budget=budget, pretrained=pretrained, backbone=backbone)
        return checkpoint

    monkeypatch.setattr(weights_module, "resolve_weights", fake_resolve)
    compressor = LearnPrunerCompressor(max_tokens=128, pretrained=True)
    assert compressor.checkpoint_path is None  # nothing fetched at construction

    compressor._resolve_pretrained(SimpleNamespace(weights_key="llava-13b"))

    assert seen == {"method": "LearnPruner", "budget": 128,
                    "pretrained": True, "backbone": "llava-13b"}
    assert compressor.checkpoint_path == checkpoint


def test_pretrained_false_and_training_never_fetch(tmp_path: Path, monkeypatch) -> None:
    """`pretrained: false` is the untrained ablation; training starts from scratch.

    A training run that means to start from released weights says so with
    `checkpoint_path`, which makes it a finetune rather than a silent one.
    """
    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("resolve_weights should not be consulted here")

    monkeypatch.setattr(weights_module, "resolve_weights", explode)
    for compressor in (LearnPrunerCompressor(max_tokens=64, pretrained=False),
                       LearnPrunerCompressor(max_tokens=64, for_training=True),
                       LearnPrunerCompressor(max_tokens=64, pretrained=True,
                                             for_training=True)):
        compressor._resolve_pretrained(SimpleNamespace(weights_key="llava"))
        assert compressor.checkpoint_path is None


def test_explicit_checkpoint_outranks_the_release(tmp_path: Path, monkeypatch) -> None:
    checkpoint = _seed_checkpoint(tmp_path, "llava")

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("an explicit checkpoint_path should win outright")

    monkeypatch.setattr(weights_module, "resolve_weights", explode)
    compressor = LearnPrunerCompressor(max_tokens=64, checkpoint_path=checkpoint,
                                       pretrained=True)
    compressor._resolve_pretrained(SimpleNamespace(weights_key="llava"))
    assert compressor.checkpoint_path == checkpoint
