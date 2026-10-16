import unittest
from types import SimpleNamespace

import torch

from llmoe.compressor._pruning import (
    place_visual_tokens,
    ratio_keep_count,
)
from llmoe.compressor.must_drop.must_drop_qwen import (
    QwenMustDropAdapter,
    local_spatial_merge,
    progressive_keep_counts,
)
from llmoe.compressor.otprune import OTPruneCompressor
from llmoe.compressor.otprune.otprune_qwen import QwenOTPruneAdapter
from llmoe.compressor.soft_topk import SoftTopkCompressor
from llmoe.compressor.soft_topk.soft_topk_qwen import QwenSoftTopkAdapter
from llmoe.compressor.tome import ToMeCompressor
from llmoe.compressor.tome.tome_qwen import QwenToMeAdapter


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
        "attention_mask": torch.ones(
            (1, length), dtype=torch.long
        ),
        "cache_position": torch.arange(length),
        "use_cache": True,
    }


class _RotaryModel:
    def rotary_emb(
        self, _hidden: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        table = position_ids.to(torch.float32)
        return table, table.clone()


class SharedRatioGeometryTest(unittest.TestCase):
    def test_ratio_budget_floors_and_keeps_at_least_one(self) -> None:
        self.assertEqual(ratio_keep_count(10, 0.35), 3)
        self.assertEqual(ratio_keep_count(2, 0.1), 1)
        self.assertEqual(ratio_keep_count(5, 1.0), 5)
        with self.assertRaisesRegex(ValueError, r"in \(0, 1\]"):
            ratio_keep_count(4, 0.0)

    def test_visual_placement_preserves_text_and_survivor_order(self) -> None:
        prompt = torch.arange(12).view(1, 6, 2).float()
        reduced = torch.tensor([[[100.0, 101.0], [200.0, 201.0]]])

        modified, keep = place_visual_tokens(
            prompt,
            start=1,
            visual_count=4,
            tokens=reduced,
            positions=torch.tensor([1, 3]),
        )

        torch.testing.assert_close(
            keep,
            torch.tensor([True, False, True, False, True, True]),
        )
        torch.testing.assert_close(
            modified[:, [2, 4]], reduced
        )


class SoftTopkQwenTest(unittest.TestCase):
    def test_prefill_scores_concatenated_visual_span_by_ratio(self) -> None:
        compressor = SoftTopkCompressor(
            retain_ratio=0.5,
            scorer_hidden_dim=2,
        )
        compressor._scores = lambda _features: torch.tensor(
            [[0.1, 0.9, 0.2, 0.8]]
        )
        adapter = QwenSoftTopkAdapter(
            compressor=compressor,
            retain_ratio=0.5,
        )
        adapter._image_token_id = 99
        adapter._input_ids = torch.tensor(
            [[7, 99, 99, 99, 99, 8]]
        )
        hidden = torch.arange(12).view(1, 6, 2).float()
        kwargs = _prompt_kwargs(hidden)

        result = adapter._prefill((), kwargs, hidden)

        self.assertIsNotNone(result)
        torch.testing.assert_close(
            kwargs["inputs_embeds"],
            hidden[:, [0, 2, 4, 5]],
        )
        self.assertEqual(
            adapter.get_source_history()[-1].shape, (2, 4)
        )
        self.assertEqual(adapter._prefill_state["kv_len"], 4)


class OTPruneQwenTest(unittest.TestCase):
    def test_prefill_applies_dpp_survivors_to_concatenated_span(self) -> None:
        compressor = OTPruneCompressor(
            retain_ratio=0.5,
            track_source=True,
        )
        compressor._select_indices = (
            lambda _visual, keep=None: torch.tensor([0, 3])
        )
        adapter = QwenOTPruneAdapter(
            compressor=compressor,
            retain_ratio=0.5,
            track_source=True,
        )
        adapter._image_token_id = 99
        adapter._input_ids = torch.tensor(
            [[7, 99, 99, 99, 99, 8]]
        )
        hidden = torch.arange(12).view(1, 6, 2).float()
        kwargs = _prompt_kwargs(hidden)

        result = adapter._prefill((), kwargs, hidden)

        self.assertIsNotNone(result)
        torch.testing.assert_close(
            kwargs["inputs_embeds"],
            hidden[:, [0, 1, 4, 5]],
        )
        torch.testing.assert_close(
            adapter.get_source_history()[-1],
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0]]
            ),
        )


class ToMeQwenTest(unittest.TestCase):
    def test_prefill_uses_vision_keys_and_exact_ratio(self) -> None:
        compressor = ToMeCompressor(
            retain_ratio=0.5,
            num_merge_layers=2,
            track_source=True,
        )
        adapter = QwenToMeAdapter(
            compressor=compressor,
            retain_ratio=0.5,
            track_source=True,
        )
        adapter._image_token_id = 99
        adapter._input_ids = torch.tensor(
            [[7, 99, 99, 99, 99, 8]]
        )
        adapter._image_grid_thw = torch.tensor([[1, 4, 4]])
        adapter._received = torch.ones(1)
        adapter._keys = torch.ones(1)
        adapter._visual = SimpleNamespace(blocks=[None, None])
        adapter._merged_token_signals = lambda: (
            torch.ones(4),
            torch.tensor(
                [[1.0, 0.0], [1.0, 0.0],
                 [0.0, 1.0], [0.0, 1.0]]
            ),
        )
        hidden = torch.arange(12).view(1, 6, 2).float()
        kwargs = _prompt_kwargs(hidden)

        result = adapter._prefill((), kwargs, hidden)

        self.assertIsNotNone(result)
        self.assertEqual(kwargs["inputs_embeds"].shape, (1, 4, 2))
        self.assertEqual(
            int(adapter.last_keep_mask[1:5].sum()), 2
        )
        final_source = adapter.get_source_history()[-1]
        self.assertEqual(final_source.shape, (2, 4))
        torch.testing.assert_close(
            final_source.sum(dim=0), torch.ones(4)
        )


class MustDropQwenTest(unittest.TestCase):
    def test_local_merge_handles_dynamic_edge_windows_exactly(self) -> None:
        tokens = torch.arange(45).view(15, 3).float()
        metric = tokens.flip(0)
        importance = torch.arange(15).float()

        merged = local_spatial_merge(
            tokens,
            metric,
            importance,
            grid_thw=(1, 3, 5),
            window_size=3,
            keep=11,
        )

        self.assertEqual(merged.tokens.shape, (11, 3))
        self.assertEqual(merged.source.shape, (11, 15))
        self.assertTrue(
            bool((merged.positions[1:] > merged.positions[:-1]).all())
        )
        torch.testing.assert_close(
            merged.source.sum(dim=1), torch.ones(11)
        )

    def test_progressive_schedule_reaches_ratio_target(self) -> None:
        counts = progressive_keep_counts(12, 3, 4)
        self.assertEqual(counts[-1], 3)
        self.assertEqual(list(counts), sorted(counts, reverse=True))

    def test_vision_prefill_uses_received_attention_for_key_set(self) -> None:
        adapter = QwenMustDropAdapter(
            retain_ratio=0.5,
            window_size=2,
            vit_retention=0.75,
            key_rate=0.25,
            prefill_stage=True,
            pruning_layers=(1,),
            global_thr=0.01,
            individual_thr=0.01,
        )
        adapter._image_token_id = 99
        adapter._input_ids = torch.tensor(
            [[7, 99, 99, 99, 99, 8]]
        )
        adapter._image_grid_thw = torch.tensor([[1, 4, 4]])
        adapter._received = torch.ones(1)
        adapter._keys = torch.ones(1)
        adapter._visual = SimpleNamespace(
            spatial_merge_size=2,
            blocks=[None, None],
        )
        adapter._merged_token_signals = lambda: (
            torch.tensor([0.1, 0.9, 0.2, 0.3]),
            torch.eye(4),
        )
        adapter._pruning_layers = (1,)
        hidden = torch.arange(12).view(1, 6, 2).float()
        kwargs = _prompt_kwargs(hidden)

        result = adapter._prefill((), kwargs, hidden)

        self.assertIsNotNone(result)
        self.assertEqual(adapter._visual_count, 3)
        self.assertEqual(adapter._stage_counts, (2,))
        self.assertEqual(int(adapter._key_mask.sum()), 1)
        self.assertEqual(kwargs["inputs_embeds"].shape, (1, 5, 2))

    def test_decoder_cut_preserves_keys_and_shared_mrope_geometry(self) -> None:
        adapter = QwenMustDropAdapter(
            retain_ratio=0.5,
            window_size=2,
            vit_retention=0.75,
            key_rate=0.25,
            prefill_stage=True,
            pruning_layers=(1,),
            global_thr=0.01,
            individual_thr=0.01,
        )
        adapter._image_start = 1
        adapter._visual_count = 3
        adapter._stage_counts = (2,)
        adapter._pruning_layers = (1,)
        adapter._position_ids = _position_ids(5)
        adapter._key_mask = torch.tensor([False, True, False])
        adapter._composed_source = torch.eye(3)
        adapter._source_history = [torch.eye(3)]
        adapter._language_model = _RotaryModel()
        adapter._dual_importance = (
            lambda *_args, **_kwargs: torch.tensor([0.1, 0.2, 0.9])
        )
        hidden = torch.arange(10).view(1, 5, 2).float()
        kwargs = {
            "attention_mask": torch.tril(
                torch.ones((1, 1, 5, 5), dtype=torch.bool)
            ),
            "cache_position": torch.arange(5),
            "position_embeddings": (
                torch.ones((1, 5, 2)),
                torch.zeros((1, 5, 2)),
            ),
        }

        pruned = adapter._prune(
            SimpleNamespace(), hidden, kwargs, stage=0
        )

        torch.testing.assert_close(
            pruned, hidden[:, [0, 2, 3, 4]]
        )
        self.assertEqual(adapter._visual_count, 2)
        torch.testing.assert_close(
            adapter._key_mask, torch.tensor([True, False])
        )
        self.assertEqual(
            adapter._stage_state[0]["kv_len"], 4
        )


if __name__ == "__main__":
    unittest.main()

