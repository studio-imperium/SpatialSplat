from training.select_base_candidates import candidate_rank, candidate_viability


def _aggregate(loss, bbox=0.8, centroid=0.05, extent=0.08, signal=0.9):
    return {
        "structure_loss": loss,
        "object": {
            "min_bbox_iou": bbox,
            "max_centroid_error": centroid,
            "max_extent_error": extent,
            "min_signal_ratio": signal,
        },
    }


def test_viable_candidate_beats_lower_loss_flipped_candidate():
    viable = _aggregate(0.3)
    flipped = _aggregate(0.2, bbox=0.2, centroid=0.3)
    viable_checks = candidate_viability(viable)
    flipped_checks = candidate_viability(flipped)
    assert all(viable_checks.values())
    assert not all(flipped_checks.values())
    assert candidate_rank(viable, viable_checks) < candidate_rank(flipped, flipped_checks)
