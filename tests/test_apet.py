import unittest

import torch

from llmoe.compressor.apet.apet import _qwen_prune_block
from llmoe.compressor.apet.apet_qwen import QwenApETAdapter


class _RotaryModel:
    def rotary_emb(
        self, _hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        table = position_ids.to(torch.float32)
        return table, table.clone()


def _adapter() -> QwenApETAdapter:
    return QwenApETAdapter(
        retain_ratio=0.5,
        layer_list=(2,),
        image_token_ratio_list=(0.5,),
        basis_tokens=1,
        selection_method="fps",
        ridge=1e-5,
        layer_ridge=0.1,
        track_source=True,
    )


class ApETPolicyTest(unittest.TestCase):
    def test_qwen_prune_returns_ordered_tokens_and_normalized_sources(self) -> None:
        features = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )

        kept, order, source = _qwen_prune_block(
            features,
            keep=2,
            basis_count=1,
            selection_method="fps",
            ridge=1e-5,
            ascending_blend=True,
        )

        self.assertEqual(kept.shape, (2, 3))
        self.assertTrue(bool((order[1:] >= order[:-1]).all()))
        torch.testing.assert_close(source.sum(dim=1), torch.ones(2))


class ApETQwenAdapterTest(unittest.TestCase):
    def test_stage_a_uses_shared_prompt_geometry(self) -> None:
        adapter = _adapter()
        adapter._image_token_id = 99
        adapter._input_ids = torch.tensor([[7, 99, 99, 99, 99, 8]])
        hidden = torch.tensor(
            [
                [
                    [2.0, 2.0, 2.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [3.0, 3.0, 3.0],
                ]
            ]
        )
        position_ids = torch.stack(
            [
                torch.arange(6),
                torch.arange(10, 16),
                torch.arange(20, 26),
                torch.arange(30, 36),
            ]
        ).unsqueeze(1)
        kwargs = {
            "inputs_embeds": hidden,
            "position_ids": position_ids,
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "cache_position": torch.arange(6),
        }

        result = adapter._prefill((), kwargs, hidden)

        self.assertIsNotNone(result)
        self.assertEqual(kwargs["inputs_embeds"].shape, (1, 4, 3))
        self.assertEqual(adapter._visual_count, 2)
        self.assertEqual(adapter._prefill_state["kv_len"], 4)
        self.assertEqual(int(adapter.last_keep_mask.sum()), 4)
        self.assertEqual(adapter.get_source_history()[-1].shape, (2, 4))

    def test_boundary_prune_uses_shared_mrope_mask_and_source_composition(self) -> None:
        adapter = _adapter()
        adapter._image_start = 1
        adapter._visual_count = 4
        adapter._position_ids = torch.stack(
            [
                torch.arange(6),
                torch.arange(10, 16),
                torch.arange(20, 26),
                torch.arange(30, 36),
            ]
        ).unsqueeze(1)
        adapter._language_model = _RotaryModel()
        identity = torch.eye(4)
        adapter._composed_source = identity.clone()
        adapter._source_history = [identity.clone()]
        hidden = torch.tensor(
            [
                [
                    [2.0, 2.0, 2.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [3.0, 3.0, 3.0],
                ]
            ]
        )
        kwargs = {
            "attention_mask": torch.tril(
                torch.ones((1, 1, 6, 6), dtype=torch.bool)
            ),
            "cache_position": torch.arange(6),
        }

        pruned = adapter._boundary_prune(hidden, kwargs, stage=0)

        self.assertEqual(pruned.shape, (1, 4, 3))
        state = adapter._stage_state[0]
        self.assertEqual(state["kv_len"], 4)
        self.assertEqual(state["attention_mask"].shape, (1, 1, 4, 4))
        torch.testing.assert_close(
            state["position_ids"], torch.arange(4).unsqueeze(0)
        )
        source = adapter.get_source_history()[-1]
        self.assertEqual(source.shape, (2, 4))
        torch.testing.assert_close(source.sum(dim=1), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
