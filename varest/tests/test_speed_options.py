"""The engine's speed options are opt-in: with nothing set, Varest runs exactly as released."""
import pytest

from varest.engine import speed_options

ENV = ("VAREST_PREPROCESS", "VAREST_VISION_GRAPH", "VAREST_DECODE_GRAPH")


@pytest.fixture
def clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_default_is_the_released_configuration(clean_env):
    assert speed_options() == ("pil", False, False)


def test_environment_switches_each_option_on(clean_env):
    clean_env.setenv("VAREST_PREPROCESS", "gpu-fields")
    clean_env.setenv("VAREST_VISION_GRAPH", "1")
    clean_env.setenv("VAREST_DECODE_GRAPH", "1")
    assert speed_options() == ("gpu-fields", True, True)


def test_arguments_override_the_environment(clean_env):
    clean_env.setenv("VAREST_PREPROCESS", "gpu")
    clean_env.setenv("VAREST_DECODE_GRAPH", "1")
    assert speed_options("pil", False, False) == ("pil", False, False)


def test_unknown_preprocess_mode_is_refused(clean_env):
    with pytest.raises(ValueError):
        speed_options("cuda")
