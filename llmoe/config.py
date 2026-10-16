from dataclasses import dataclass, field
from os import environ
from os.path import exists
from typing import Any, override

from mipcandy import Device, auto_device
from mle.vars import ExpConfig

from llmoe.weights import WEIGHTS_DIR_NAME


@dataclass
class Config(ExpConfig):
    dataset_limit: int | None = None
    custom_args: dict[str, Any] = field(default_factory=dict)
    device: Device = auto_device()
    visualize: bool = False
    visualize_saliency: bool = False
    # Resolved by `llmoe.bootstrap` before `huggingface_hub` is imported, and
    # carried here so the config snapshot records the cache the run really used.
    _hf_home: str | None = None

    @property
    def hf_home(self) -> str:
        """Hugging Face cache root for this run: base models and method weights.

        Tied to ``root_dir`` rather than ``output_dir`` so one cache serves every
        experiment on the machine; see `llmoe.bootstrap`, which normally settles
        this before ``huggingface_hub`` is imported.
        """
        return self._hf_home or environ.get("HF_HOME") or f"{self.root_dir}/output/hf_cache"

    @property
    def weights_dir(self) -> str:
        """Released LLMOE method weights, a sibling of the model cache."""
        return f"{self.hf_home}/{WEIGHTS_DIR_NAME}"

    def setup_hf_env(self) -> None:
        """Point Hugging Face at this run's cache before any model is loaded.

        Split out of :meth:`initialize` because the benchmark path never calls
        that (it would demand a training dataset directory), yet still needs the
        cache root -- `llmoe.weights` reads ``HF_HOME`` to find the bundle.
        Existing ``HF_HOME``/``HF_TOKEN`` values are left alone so an operator can
        override from the shell, and the token file stays optional: public
        weights and an already-populated cache need no credentials.
        """
        environ.setdefault("HF_HOME", self.hf_home)
        token_file = f"{self.root_dir}/hf_token"
        if "HF_TOKEN" not in environ and exists(token_file):
            with open(token_file) as f:
                token = f.read().strip()
            if token:
                environ["HF_TOKEN"] = token

    @override
    def initialize(self) -> None:
        super().initialize()
        self.setup_hf_env()
