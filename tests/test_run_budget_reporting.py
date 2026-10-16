"""Evaluation-result persistence for dynamic per-image compression budgets."""

import pytest

from llmoe.compressor.quietprune import QuietPruneCompressor


def test_per_sample_budget_report_is_snapshotted_for_evaluation_json() -> None:
    report = {
        "original_content_tokens": 1152,
        "requested_retention_ratio": 0.25,
        "target_content_tokens": 288,
        "realized_content_tokens": 289,
        "realized_retention_ratio": 289 / 1152,
        "structural_tokens": 24,
    }
    compressor = QuietPruneCompressor(max_tokens=64, pretrained=False)
    compressor._last_budget_report = report

    captured = compressor.get_benchmark_record_metadata(1)
    assert captured == [{"compression_budget": report}]
    assert captured[0]["compression_budget"] is not report

    with pytest.raises(RuntimeError, match="batch_size=1"):
        compressor.get_benchmark_record_metadata(2)
    compressor._last_budget_report = None
    assert compressor.get_benchmark_record_metadata(2) == [{}, {}]
