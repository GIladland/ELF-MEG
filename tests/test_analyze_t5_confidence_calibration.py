import numpy as np

from scripts.analyze_t5_confidence_calibration import auc, calibration_bins


def test_auc_handles_ordering_and_ties():
    labels = np.asarray([False, False, True, True])
    assert auc(np.asarray([0.0, 0.1, 0.8, 0.9]), labels) == 1.0
    assert auc(np.asarray([0.8, 0.9, 0.0, 0.1]), labels) == 0.0
    assert auc(np.ones(4), labels) == 0.5


def test_calibration_bins_returns_weighted_gap():
    rows, error = calibration_bins(
        np.asarray([0.1, 0.2, 0.8, 0.9]),
        np.asarray([False, False, True, True]),
        bins=2,
    )
    assert len(rows) == 2
    assert abs(error - 0.15) < 1e-12
