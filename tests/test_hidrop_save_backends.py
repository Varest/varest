"""HiDrop's checkpoint save, on every backbone it supports.

This routine has now failed twice on Qwen in production, both times at the first
checkpoint an hour into a run, and both times because it reached past the backend
into a LLaVA-shaped attribute: first `model.multi_modal_projector`, then
`hidrop_config.hidrop_metadata()`. Nothing exercised it -- the training tests stop
before a save -- so each fix only revealed the next line.

These call `_save` directly, per backend, so the whole routine is covered without
an hour of H100 time.
"""

import json
from pathlib import Path

import pytest
import torch

from llmoe.compressor.hidrop.hidrop import BACKENDS
from llmoe.compressor.hidrop.hidrop_tr import HiDropTrainer


class _StubConfig:
    """A config that deliberately lacks `hidrop_metadata`, as Qwen's does."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_dict(self) -> dict:
        return dict(self._payload)


class _LlavaStubConfig(_StubConfig):
    def hidrop_metadata(self) -> dict:
        return {"schedule": "llava"}


class _StubModel:
    def __init__(self, backend, config, state: dict) -> None:
        self.backend = backend
        self.hidrop_config = config
        self._state = state

    def state_dict(self) -> dict:
        return self._state


class _StubArgs:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        # `is_world_process_zero()` reads this; single-process here.
        self.process_index = 0
        self.local_process_index = 0
        self.should_save = True


def _trainer(backend_key: str, tmp_path: Path) -> HiDropTrainer:
    """A HiDropTrainer with just the attributes `_save` touches."""
    backend = BACKENDS[backend_key]
    if backend_key == "qwen":
        config = _StubConfig({"retain_ratio": 0.25, "debug": False, "enabled": True,
                              "batch_size_supported": True})
    else:
        config = _LlavaStubConfig({})
    # The gathered dict is the *training wrapper's*, so every key carries the
    # `base_model.` prefix the extractor keys off.
    prefix = f"base_model.{backend.stage1_module_path}"
    state = {f"{prefix}.linear_1.weight": torch.zeros(2, 2),
             f"{prefix}.linear_1.bias": torch.zeros(2),
             "base_model.model.language_model.layers.0.weight": torch.zeros(2, 2)}

    trainer = HiDropTrainer.__new__(HiDropTrainer)
    trainer.model = _StubModel(backend, config, state)
    trainer.args = _StubArgs(str(tmp_path))
    trainer.run_manifest = {"run": {"stage": "pretrain"}}
    return trainer


@pytest.mark.parametrize("backend_key", sorted(BACKENDS))
def test_pretrain_save_writes_the_backends_own_artifact(backend_key, tmp_path):
    backend = BACKENDS[backend_key]
    trainer = _trainer(backend_key, tmp_path)

    trainer._save(output_dir=str(tmp_path))

    artifact = tmp_path / backend.stage1_artifact_name
    assert artifact.exists(), (
        f"{backend_key} should write {backend.stage1_artifact_name}; "
        f"found {sorted(p.name for p in tmp_path.iterdir())}"
    )
    saved = torch.load(artifact, weights_only=True)
    assert saved, "the stage-1 state was empty"
    # Only the stage-1 module, not the frozen decoder.
    assert not any("language_model" in key for key in saved)


@pytest.mark.parametrize("backend_key", sorted(BACKENDS))
def test_pretrain_save_writes_schedule_metadata(backend_key, tmp_path):
    """The Qwen config has no `hidrop_metadata`; the backend supplies it."""
    trainer = _trainer(backend_key, tmp_path)

    trainer._save(output_dir=str(tmp_path))

    written = json.loads((tmp_path / "hidrop_config.json").read_text())
    assert isinstance(written, dict) and written


def test_qwen_metadata_comes_from_the_backend_not_the_config(tmp_path):
    """Pin the actual regression: a config without the LLaVA-only method."""
    trainer = _trainer("qwen", tmp_path)
    assert not hasattr(trainer.model.hidrop_config, "hidrop_metadata")

    trainer._save(output_dir=str(tmp_path))

    written = json.loads((tmp_path / "hidrop_config.json").read_text())
    assert written["retain_ratio"] == 0.25
    # The backend strips run-local diagnostics from the manifest.
    assert "debug" not in written


def test_a_missing_stage1_module_names_the_path_it_looked_for(tmp_path):
    trainer = _trainer("qwen", tmp_path)
    trainer.model._state = {
        "base_model.model.language_model.layers.0.weight": torch.zeros(2, 2)}

    with pytest.raises(RuntimeError, match="model.visual.merger"):
        trainer._save(output_dir=str(tmp_path))
