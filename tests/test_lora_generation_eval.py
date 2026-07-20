import pytest

from training.evaluate_lora_generation import summarize_pairs


def _pair(base: float, lora: float, recovered: float | None) -> dict:
    return {
        "base": {"aggregate": {"spatial_loss": base}},
        "lora": {"aggregate": {"spatial_loss": lora}},
        "recovered_target_improvement": recovered,
    }


def test_summarize_pairs_reports_paired_improvement() -> None:
    summary = summarize_pairs(
        [_pair(0.4, 0.2, 0.6), _pair(0.2, 0.3, -0.5)]
    )

    assert summary["num_pairs"] == 2
    assert summary["mean_base_spatial_loss"] == pytest.approx(0.3)
    assert summary["mean_lora_spatial_loss"] == pytest.approx(0.25)
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["mean_recovered_target_improvement"] == pytest.approx(0.05)


def test_summarize_pairs_requires_results() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_pairs([])


def test_summarize_pairs_supports_heldout_results_without_targets() -> None:
    summary = summarize_pairs([_pair(0.4, 0.25, None)])

    assert summary["win_rate"] == 1.0
    assert summary["mean_recovered_target_improvement"] is None


def test_summarize_pairs_reports_p95_when_available() -> None:
    pair = _pair(0.4, 0.25, None)
    pair["base"]["aggregate"]["p95_normalized_depth_error"] = 0.3
    pair["lora"]["aggregate"]["p95_normalized_depth_error"] = 0.18

    summary = summarize_pairs([pair])

    assert summary["mean_base_p95_depth_error"] == pytest.approx(0.3)
    assert summary["mean_lora_p95_depth_error"] == pytest.approx(0.18)
    assert summary["p95_win_rate"] == 1.0
