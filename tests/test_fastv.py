import math
import unittest
from types import SimpleNamespace

import torch

from llmoe.compressor.fastv import FastVCompressor, resolve_keep, select_indices
from llmoe.compressor.fastv.fastv import (
    last_query_attention,
    normalize_attention_scores,
    select_visual_keep,
)
from llmoe.compressor.fastv.fastv_qwen import QwenFastVAdapter


class _IdentityAttention:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            num_attention_heads=1,
            num_key_value_heads=1,
        )
        self.head_dim = 2
        self.scaling = 1 / math.sqrt(2)
        self.q_proj = lambda tensor: tensor
        self.k_proj = lambda tensor: tensor


class _RotaryModel:
    def rotary_emb(self, _hidden_states: torch.Tensor,
                   position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        table = position_ids.to(torch.float32)
        return table, table.clone()


class FastVPolicyTest(unittest.TestCase):
    def test_budget_and_topk_preserve_reading_order(self) -> None:
        self.assertEqual(resolve_keep(10, max_tokens=4, retain_ratio=None), 4)
        self.assertEqual(resolve_keep(10, max_tokens=9, retain_ratio=0.3), 3)
        self.assertEqual(resolve_keep(2, max_tokens=9, retain_ratio=None), 2)

        scores = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.7]])
        torch.testing.assert_close(
            select_indices(scores, 3), torch.tensor([[1, 3, 4]])
        )

    def test_attention_normalization_uses_last_query_and_head_mean(self) -> None:
        attention = torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [0.2, 0.4, 0.6]],
                    [[0.0, 0.0, 0.0], [0.4, 0.6, 0.8]],
                ]
            ]
        )
        scores = normalize_attention_scores(attention, batch=1, n_tokens=3)
        torch.testing.assert_close(scores, torch.tensor([[0.3, 0.5, 0.7]]))

    def test_visual_keep_plan_retains_all_text_columns(self) -> None:
        scores = torch.tensor([[0.0, 0.1, 0.9, 0.2, 0.8, 0.0]])
        image_positions = torch.tensor([[1, 2, 3, 4]])

        local, keep_index = select_visual_keep(
            scores,
            image_positions,
            keep=2,
            seq_len=6,
            label="FastV test",
        )

        torch.testing.assert_close(local, torch.tensor([[1, 3]]))
        torch.testing.assert_close(keep_index, torch.tensor([[0, 2, 4, 5]]))

    def test_last_query_attention_supports_backbone_rope_injection(self) -> None:
        attention = _IdentityAttention()
        hidden_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        calls: list[str] = []

        def identity_rope(query: torch.Tensor, key: torch.Tensor,
                          _cos: torch.Tensor, _sin: torch.Tensor,
                          tag: str) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append(tag)
            return query, key

        mask = torch.ones((1, 1, 3, 3), dtype=torch.bool)
        mask[:, :, -1, 1] = False
        scores = last_query_attention(
            attention,
            hidden_states,
            (torch.ones(1), torch.zeros(1)),
            mask,
            apply_rope=identity_rope,
            rope_args=("qwen",),
        )

        expected_logits = torch.tensor([[1.0, -torch.inf, 2.0]]) / math.sqrt(2)
        torch.testing.assert_close(scores, expected_logits.softmax(dim=-1))
        self.assertEqual(calls, ["qwen"])

    def test_tensor_compress_uses_the_same_policy_and_tracks_sources(self) -> None:
        # retain_ratio, not max_tokens: the budget label is the layer-averaged count,
        # while this test wants a literal 2 survivors out of 5.
        compressor = FastVCompressor(max_tokens=64, retain_ratio=0.4, track_source=True)
        features = torch.arange(10).view(5, 2).float()
        scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.7])

        compressed = compressor.compress(features, attention=scores)

        torch.testing.assert_close(compressed.tokens, features[[1, 3]])
        history = compressor.get_source_history()
        self.assertEqual(len(history), 2)
        torch.testing.assert_close(
            history[1],
            torch.tensor(
                [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]]
            ),
        )


class FastVBackboneParityTest(unittest.TestCase):
    def test_llava_and_qwen_choose_identical_visual_survivors(self) -> None:
        hidden_states = torch.arange(10).view(1, 5, 2).float()
        image_positions = torch.tensor([[1, 2, 3]])
        scores = torch.tensor([[0.0, 0.1, 0.9, 0.8, 0.0]])
        mask = torch.tril(torch.ones((1, 1, 5, 5), dtype=torch.bool))

        # 2 survivors out of 3 image tokens; see the note above on retain_ratio.
        llava = FastVCompressor(max_tokens=64, retain_ratio=0.7, track_source=True)
        llava._image_positions = image_positions.clone()
        llava._attention_scores = scores.clone()
        llava._language_model = _RotaryModel()
        llava_kwargs = {
            "position_ids": torch.arange(5).unsqueeze(0),
            "attention_mask": mask.clone(),
            "cache_position": torch.arange(5),
        }
        llava_hidden = llava._prune(hidden_states, llava_kwargs)

        qwen = QwenFastVAdapter(retain_ratio=0.67, k=3, track_source=True)
        qwen._image_positions = image_positions.clone()
        qwen._scores = scores.clone()
        qwen._position_ids = torch.stack(
            [torch.arange(5), torch.arange(10, 15),
             torch.arange(20, 25), torch.arange(30, 35)]
        ).unsqueeze(1)
        qwen._language_model = _RotaryModel()
        qwen_kwargs = {
            "attention_mask": mask.clone(),
            "cache_position": torch.arange(5),
        }
        qwen_hidden = qwen._prune(hidden_states, qwen_kwargs)

        expected = hidden_states[:, [0, 2, 3, 4]]
        torch.testing.assert_close(llava_hidden, expected)
        torch.testing.assert_close(qwen_hidden, expected)
        torch.testing.assert_close(
            llava.get_source_history()[1], qwen.get_source_history()[1]
        )
        self.assertEqual(llava._layer_state["kv_len"], 4)
        self.assertEqual(qwen._prune_state["kv_len"], 4)


if __name__ == "__main__":
    unittest.main()
