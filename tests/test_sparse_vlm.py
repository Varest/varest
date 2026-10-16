import math
import unittest
from types import SimpleNamespace

import torch

from llmoe.compressor.sparse_vlm import SparseVLMCompressor, sparsify
from llmoe.compressor.sparse_vlm.sparse_vlm_qwen import QwenSparseVLMAdapter
from llmoe.compressor.sparse_vlm.sparse_vlm import (
    SparseVLMPolicy,
    normalize_rater_scores,
    rater_attention,
    select_raters,
)


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


class SparseVLMPolicyTest(unittest.TestCase):
    def test_rater_selection_falls_back_to_all_text_tokens(self) -> None:
        inputs_embeds = torch.zeros((1, 5, 3))
        torch.testing.assert_close(
            select_raters(inputs_embeds, image_start=1, visual_count=2),
            torch.tensor([0, 1]),
        )

    def test_rater_score_normalization_averages_heads_and_raters(self) -> None:
        attention = torch.tensor(
            [
                [
                    [[0.0, 0.2, 0.4], [0.2, 0.4, 0.6]],
                    [[0.4, 0.6, 0.8], [0.6, 0.8, 1.0]],
                ]
            ]
        )
        torch.testing.assert_close(
            normalize_rater_scores(attention, n_tokens=3),
            torch.tensor([0.3, 0.5, 0.7]),
        )

    def test_rater_attention_is_causal_and_accepts_rope_injection(self) -> None:
        attention = _IdentityAttention()
        hidden_states = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]]
        )
        calls: list[str] = []

        def identity_rope(query: torch.Tensor, key: torch.Tensor,
                          _cos: torch.Tensor, _sin: torch.Tensor,
                          tag: str) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append(tag)
            return query, key

        scores = rater_attention(
            attention,
            hidden_states,
            (torch.ones(1), torch.zeros(1)),
            torch.tensor([2, 3]),
            apply_rope=identity_rope,
            rope_args=("qwen",),
        )

        self.assertEqual(scores.shape, (2, 4))
        self.assertEqual(float(scores[0, 3]), 0.0)
        torch.testing.assert_close(scores.sum(dim=1), torch.ones(2))
        self.assertEqual(calls, ["qwen"])

    def test_recycling_is_deterministic_after_reseed(self) -> None:
        policy = SparseVLMPolicy(
            recycling=True,
            recycle_fraction=0.5,
            cluster_divisor=2,
            seed=17,
            track_source=True,
        )
        visual = torch.arange(24).view(1, 6, 4).float()
        scores = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

        policy.reseed()
        first = policy.sparsify(visual, scores, keep_k=2)
        policy.reseed()
        second = policy.sparsify(visual, scores, keep_k=2)

        for left, right in zip(first, second):
            torch.testing.assert_close(left, right)
        torch.testing.assert_close(
            first[2].sum(dim=1), torch.ones(first[2].shape[0])
        )

    def test_public_sparsify_keeps_winners_in_reading_order(self) -> None:
        visual = torch.arange(12).view(1, 4, 3).float()
        scores = torch.tensor([0.1, 0.9, 0.8, 0.2])
        generator = torch.Generator().manual_seed(3)

        tokens, donors, source = sparsify(
            visual,
            scores,
            keep_k=2,
            recycling=False,
            recycle_fraction=0.3,
            cluster_divisor=10,
            generator=generator,
            track_source=True,
        )

        torch.testing.assert_close(tokens, visual[:, [1, 2], :])
        torch.testing.assert_close(donors, torch.tensor([1, 2]))
        torch.testing.assert_close(
            source,
            torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        )


class SparseVLMBackboneParityTest(unittest.TestCase):
    def test_llava_and_qwen_run_the_same_sparsification_policy(self) -> None:
        hidden_states = torch.arange(14).view(1, 7, 2).float()
        scores = torch.tensor([0.1, 0.9, 0.8, 0.2])
        mask = torch.tril(torch.ones((1, 1, 7, 7), dtype=torch.bool))

        llava = SparseVLMCompressor(
            max_tokens=2,
            schedule=(2,),
            schedule_ratios=(0.5,),
            pruning_layers=(0,),
            recycling=False,
            track_source=True,
        )
        llava._span = (1, 4)
        llava._visual_count = 4
        llava._scores = {0: scores.clone()}
        llava._language_model = _RotaryModel()
        llava_kwargs = {
            "attention_mask": mask.clone(),
            "cache_position": torch.arange(7),
        }
        llava_hidden = llava._prune_stage(hidden_states, llava_kwargs, stage=0)

        qwen = QwenSparseVLMAdapter(
            schedule_ratios=(0.5,),
            pruning_layers=(0,),
            policy=llava._policy,
        )
        qwen._image_start = 1
        qwen._original_count = 4
        qwen._visual_count = 4
        qwen._scores = {0: scores.clone()}
        qwen._position_ids = torch.stack(
            [torch.arange(7), torch.arange(10, 17),
             torch.arange(20, 27), torch.arange(30, 37)]
        ).unsqueeze(1)
        qwen._language_model = _RotaryModel()
        qwen_kwargs = {
            "attention_mask": mask.clone(),
            "cache_position": torch.arange(7),
        }
        qwen_hidden = qwen._prune_stage(hidden_states, qwen_kwargs, stage=0)

        expected = hidden_states[:, [0, 2, 3, 5, 6]]
        torch.testing.assert_close(llava_hidden, expected)
        torch.testing.assert_close(qwen_hidden, expected)
        torch.testing.assert_close(
            llava.get_source_history()[1], qwen.get_source_history()[1]
        )
        self.assertEqual(llava._layer_state["cache_position"].numel(), 5)
        self.assertEqual(qwen._stage_state[0]["kv_len"], 5)


if __name__ == "__main__":
    unittest.main()
