"""Tests for trajectory diagnostic metrics."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_trajectories import directional_lag_projection


class TrajectoryDiagnosticTests(unittest.TestCase):
    def test_positive_projection_means_estimate_lags_motion(self):
        value = directional_lag_projection(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.8, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(value, 0.2, places=7)

    def test_negative_projection_means_estimate_leads_motion(self):
        value = directional_lag_projection(
            np.array([1.0, 0.0, 0.0]),
            np.array([1.2, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(value, -0.2, places=7)

    def test_no_motion_has_no_directional_projection(self):
        value = directional_lag_projection(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.9, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        self.assertIsNone(value)

    def test_first_sample_has_no_previous_direction(self):
        value = directional_lag_projection(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.9, 0.0, 0.0]),
            None,
        )
        self.assertIsNone(value)


if __name__ == "__main__":
    unittest.main()
