import pytest

from dqat.compare_calibration import paired_comparison


def test_paired_calibration_discordance_and_interval():
    reference = {i: i < 467 for i in range(500)}
    candidate = reference.copy()
    for i in range(17):
        candidate[i] = False
    for i in range(467, 482):
        candidate[i] = True
    result = paired_comparison(reference, candidate)
    assert result["difference_percentage_points"] == -0.4
    assert result["candidate_only_successes"] == 15
    assert result["reference_only_successes"] == 17
    assert result["shared_successes"] == 450
    assert result["shared_failures"] == 18
    assert result["descriptive_paired_bootstrap_95_interval_pp"] == pytest.approx([-2.6, 1.8])


def test_calibration_comparison_rejects_mismatched_or_empty_pairs():
    with pytest.raises(ValueError, match="same nonempty"):
        paired_comparison({0: True}, {1: True})
    with pytest.raises(ValueError, match="same nonempty"):
        paired_comparison({}, {})
