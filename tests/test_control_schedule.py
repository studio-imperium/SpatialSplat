import pytest

from spatial_control import control_scale_for_step


def test_control_schedule_leaves_final_steps_uncontrolled() -> None:
    scales = [control_scale_for_step(1.25, step, 20, 0.7) for step in range(20)]

    assert scales[:14] == [1.25] * 14
    assert scales[14:] == [0.0] * 6


def test_full_control_schedule_preserves_previous_behavior() -> None:
    scales = [control_scale_for_step(0.8, step, 5, 1.0) for step in range(5)]

    assert scales == [0.8] * 5


def test_control_schedule_rejects_invalid_cutoff() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        control_scale_for_step(1.0, 0, 20, 1.1)
