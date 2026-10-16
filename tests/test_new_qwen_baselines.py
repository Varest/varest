import unittest

import torch

from llmoe.compressor.divprune import DivPruneCompressor
from llmoe.compressor.divprune.divprune_qwen import QwenDivPruneAdapter
from llmoe.compressor.fastv import FastVCompressor
from llmoe.compressor.hidrop import HiDropCompressor
from llmoe.compressor.nuwa import NuwaCompressor
from llmoe.compressor.pyramiddrop import (
    PyramidDropCompressor,
    qwen_pyramid_schedule,
    stage_widths,
)
from llmoe.compressor.quietprune import QuietPruneCompressor
from llmoe.compressor.quietprune.quietprune_qwen import quiet_qwen_reduce
from llmoe.compressor.soft_topk import SoftTopkCompressor
from llmoe.compressor.sparse_vlm import SparseVLMCompressor
from llmoe.compressor.vflowopt import StageLayers, VFlowOptCompressor
from llmoe.compressor.vflowopt.vflowopt_qwen import (
    qwen_vflow_decay,
    qwen_vision_stage,
)


def _position_ids(length: int) -> torch.Tensor:
    return torch.stack(
        [
            torch.arange(length),
            torch.arange(10, 10 + length),
            torch.arange(20, 20 + length),
            torch.arange(30, 30 + length),
        ]
    ).unsqueeze(1)


def _prompt_kwargs(hidden: torch.Tensor) -> dict:
    length = hidden.shape[1]
    return {
        "inputs_embeds": hidden,
        "position_ids": _position_ids(length),
        "attention_mask": torch.ones((1, length), dtype=torch.long),
        "cache_position": torch.arange(length),
        "use_cache": True,
    }


class DivPruneQwenTest(unittest.TestCase):
    def test_ratio_path_selects_the_concatenated_visual_span(self) -> None:
        compressor = DivPruneCompressor(retain_ratio=0.5, track_source=True)
        compressor._select_indices = (
            lambda _visual, keep: torch.tensor([1, 3])
        )
        adapter = QwenDivPruneAdapter(
            compressor=compressor,
            retain_ratio=0.5,
            track_source=True,
        )
        adapter._image_token_id = 99
        adapter._input_ids = torch.tensor([[7, 99, 99, 99, 99, 8]])
        hidden = torch.arange(12).view(1, 6, 2).float()
        kwargs = _prompt_kwargs(hidden)

        self.assertIsNotNone(adapter._prefill((), kwargs, hidden))
        torch.testing.assert_close(kwargs["inputs_embeds"], hidden[:, [0, 2, 4, 5]])
        self.assertEqual(compressor.name, "DivPrune-b0.5")


class PyramidDropQwenTest(unittest.TestCase):
    def test_derived_schedule_matches_layer_average_ratio(self) -> None:
        boundaries, ratios = qwen_pyramid_schedule(0.25, 28)
        widths = stage_widths(boundaries, 28)
        achieved = sum(
            width * ratio
            for width, ratio in zip(widths, (1.0, *ratios))
        ) / 28

        self.assertAlmostEqual(achieved, 0.25, places=10)
        self.assertEqual(list(boundaries), sorted(boundaries))
        self.assertEqual(list(ratios), sorted(ratios, reverse=True))
        self.assertEqual(
            PyramidDropCompressor(retain_ratio=0.25).name,
            "PyramidDrop-b0.25",
        )

    def test_unreachable_first_stage_floor_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "first full-token layer"):
            qwen_pyramid_schedule(1 / 28, 28)


class QuietPruneQwenTest(unittest.TestCase):
    def test_group_reducer_hits_ratio_count_on_rectangular_grid(self) -> None:
        tokens = torch.arange(45).view(15, 3).float()
        scores = torch.linspace(0.1, 1.5, 15)
        result = quiet_qwen_reduce(
            tokens,
            scores,
            grid=(3, 5),
            keep=7,
            group_size=2,
            keep_border_tokens=True,
            aggregate=True,
            normalize_aggregation=True,
            eps=1e-6,
        )

        self.assertEqual(result.tokens.shape, (7, 3))
        self.assertTrue(bool((result.positions[1:] > result.positions[:-1]).all()))
        torch.testing.assert_close(result.source.sum(dim=1), torch.ones(7))
        self.assertEqual(
            QuietPruneCompressor(retain_ratio=0.5, pretrained=False).name,
            "QuietPrune-b0.5",
        )


class VFlowOptQwenTest(unittest.TestCase):
    def test_geometric_decay_matches_requested_layer_average(self) -> None:
        stages = StageLayers.even(28)
        decay = qwen_vflow_decay(0.25, stages)
        achieved = (
            decay * stages.l1
            + decay**2 * stages.l2
            + decay**3 * stages.l3
        ) / stages.total

        self.assertAlmostEqual(achieved, 0.25, places=10)
        self.assertEqual(
            VFlowOptCompressor(retain_ratio=0.25).name,
            "VFlowOpt-b0.25",
        )

    def test_recycling_is_exact_on_a_rectangular_grid(self) -> None:
        features = torch.randn(15, 6)
        importance = torch.linspace(0.1, 1.5, 15)
        result = qwen_vision_stage(
            features,
            importance,
            keep_target=7,
            grid=(3, 5),
            recycle_fraction=0.25,
        )

        self.assertEqual(result.tokens.shape, (7, 6))
        self.assertTrue(bool((result.positions[1:] > result.positions[:-1]).all()))
        torch.testing.assert_close(result.source.sum(dim=1), torch.ones(7))
        torch.testing.assert_close(
            result.source @ features.float(),
            result.tokens.float(),
        )


class SharedRunLabelTest(unittest.TestCase):
    def test_legacy_qwen_paths_inherit_the_budget_only_format(self) -> None:
        compressors = (
            HiDropCompressor(retain_ratio=0.25),
            SoftTopkCompressor(retain_ratio=0.25),
            FastVCompressor(retain_ratio=0.25),
            NuwaCompressor(retain_ratio=0.25),
            SparseVLMCompressor(max_tokens=64, recycling=False),
        )
        self.assertEqual(
            [compressor.name for compressor in compressors],
            [
                "HiDrop-b0.25",
                "SoftTopk-b0.25",
                "FastV-b0.25",
                "Nuwa-b0.25",
                "SparseVLM-b64",
            ],
        )


if __name__ == "__main__":
    unittest.main()
