import unittest

import torch

from llmoe.compressor._pruning import (
    QwenPruningAdapter,
    advance_decode_step,
    apply_prune_state,
    gather_mrope_position_ids,
    keep_columns,
    pruned_attention_mask,
)


class DecoderPruningHelpersTest(unittest.TestCase):
    def test_keep_columns_requires_equal_survivor_counts(self) -> None:
        keep = torch.tensor(
            [[True, False, True, False], [False, True, False, True]]
        )
        torch.testing.assert_close(
            keep_columns(keep), torch.tensor([[0, 2], [1, 3]])
        )

        with self.assertRaisesRegex(RuntimeError, "same number of survivors"):
            keep_columns(torch.tensor([[True, False], [True, True]]))

    def test_gather_mrope_recompacts_only_the_packing_row(self) -> None:
        position_ids = torch.tensor(
            [
                [[0, 1, 2, 3, 4]],
                [[10, 11, 12, 13, 14]],
                [[20, 21, 22, 23, 24]],
                [[30, 31, 32, 33, 34]],
            ]
        )
        keep_index = torch.tensor([[0, 2, 4]])

        narrowed, text_ids = gather_mrope_position_ids(
            position_ids, keep_index, seq_len=5, new_len=3
        )

        expected = torch.tensor(
            [
                [[0, 1, 2]],
                [[10, 12, 14]],
                [[20, 22, 24]],
                [[30, 32, 34]],
            ]
        )
        torch.testing.assert_close(narrowed, expected)
        torch.testing.assert_close(text_ids, expected[0])

    def test_pruned_attention_mask_selects_rows_and_columns(self) -> None:
        attention_mask = torch.tril(
            torch.ones((1, 1, 5, 5), dtype=torch.bool)
        )
        keep_index = torch.tensor([[0, 2, 4]])

        narrowed, valid = pruned_attention_mask(
            attention_mask, keep_index, seq_len=5
        )

        self.assertEqual(narrowed.shape, (1, 1, 3, 3))
        torch.testing.assert_close(
            narrowed, torch.tril(torch.ones((1, 1, 3, 3), dtype=torch.bool))
        )
        torch.testing.assert_close(valid, torch.ones((1, 3), dtype=torch.bool))

    def test_decode_state_extends_and_rebuilds_the_mask(self) -> None:
        state = {
            "batch": 1,
            "valid": torch.tensor([[True, False, True]]),
            "kv_len": 3,
            "cache_position": torch.tensor([2]),
        }
        advance_decode_step(state, torch.zeros((1, 1, 4)))

        kwargs = {
            "attention_mask": torch.ones((1, 1, 1, 8), dtype=torch.bool),
            "cache_position": torch.tensor([7]),
        }
        apply_prune_state(kwargs, state, prefill=False)

        self.assertEqual(state["kv_len"], 4)
        torch.testing.assert_close(
            state["valid"], torch.tensor([[True, False, True, True]])
        )
        torch.testing.assert_close(
            kwargs["attention_mask"],
            torch.tensor([[[[True, False, True, True]]]]),
        )
        torch.testing.assert_close(kwargs["cache_position"], torch.tensor([3]))


class QwenPruningAdapterTest(unittest.TestCase):
    def test_locate_image_span_rejects_disjoint_placeholders(self) -> None:
        adapter = QwenPruningAdapter()
        adapter._image_token_id = 99
        inputs_embeds = torch.zeros((1, 5, 2))

        adapter._input_ids = torch.tensor([[1, 99, 99, 2, 3]])
        self.assertEqual(adapter._locate_image_span(inputs_embeds), (1, 2))

        adapter._input_ids = torch.tensor([[1, 99, 2, 99, 3]])
        with self.assertRaisesRegex(NotImplementedError, "contiguous image span"):
            adapter._locate_image_span(inputs_embeds)

    def test_apply_keep_and_decode_share_one_compacted_geometry(self) -> None:
        adapter = QwenPruningAdapter()
        adapter._image_token_id = 99
        adapter._input_ids = torch.tensor([[1, 99, 99, 99, 2]])

        inputs_embeds = torch.arange(10).view(1, 5, 2).float()
        modified_embeds = inputs_embeds + 100
        keep = torch.tensor([True, True, False, True, True])
        position_ids = torch.tensor(
            [
                [[0, 1, 2, 3, 4]],
                [[10, 11, 12, 13, 14]],
                [[20, 21, 22, 23, 24]],
                [[30, 31, 32, 33, 34]],
            ]
        )
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "cache_position": torch.arange(5),
        }

        state = adapter._apply_keep(kwargs, inputs_embeds, modified_embeds, keep)

        torch.testing.assert_close(kwargs["inputs_embeds"], modified_embeds[:, keep])
        torch.testing.assert_close(
            kwargs["position_ids"][0], torch.tensor([[0, 1, 2, 3]])
        )
        torch.testing.assert_close(
            kwargs["position_ids"][1], torch.tensor([[10, 11, 13, 14]])
        )
        torch.testing.assert_close(kwargs["cache_position"], torch.arange(4))
        self.assertEqual(state["prefill_len"], 4)

        decode_kwargs = {
            "inputs_embeds": torch.zeros((1, 1, 2)),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "cache_position": torch.tensor([5]),
        }
        result = adapter._decode((), decode_kwargs, decode_kwargs["inputs_embeds"])

        self.assertIsNotNone(result)
        self.assertEqual(state["kv_len"], 5)
        torch.testing.assert_close(
            decode_kwargs["position_ids"], state["base_position"] + 1
        )
        torch.testing.assert_close(
            decode_kwargs["attention_mask"], torch.ones((1, 5), dtype=torch.long)
        )
        torch.testing.assert_close(
            decode_kwargs["cache_position"], torch.tensor([4])
        )


if __name__ == "__main__":
    unittest.main()
