"""Tests for trajectory diagnostics and non-invasive evaluator tracing."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_trajectories import directional_lag_projection
from evaluate_robustness import evaluate


class FakePolicy:
    def predict(self, observation, deterministic=True):
        return np.array([0.1, -0.1, 0.2, 0.0], dtype=np.float32), None


class FakeFetchEnv:
    def __init__(self):
        self.steps = 0

    def _observation(self):
        state = np.zeros(25, dtype=np.float32)
        state[0:3] = [0.1 + self.steps * 0.001, 0.2, 0.3]
        state[3:6] = [0.4, 0.5, 0.6]
        state[6:9] = state[3:6] - state[0:3]
        return {
            "observation": state,
            "achieved_goal": state[3:6].copy(),
            "desired_goal": np.array([0.5, 0.5, 0.6], dtype=np.float32),
        }

    def reset(self, seed=None):
        self.steps = 0
        return self._observation(), {}

    def step(self, action):
        # The evaluator must pass the policy action through without dtype coercion.
        if action.dtype != np.float32:
            raise AssertionError(f"expected float32 action, got {action.dtype}")
        self.steps += 1
        return self._observation(), -1.0, False, self.steps >= 2, {"is_success": False}


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

    def test_evaluator_trace_callback_preserves_action_dtype(self):
        events = []
        records = evaluate(
            FakePolicy(),
            FakeFetchEnv(),
            episode_count=1,
            seed_start=1000,
            sigmas=[0.005],
            alpha=0.35,
            max_steps=2,
            trace_callback=events.append,
        )
        self.assertEqual(len(records), 3)  # clean + noisy + EMA
        self.assertEqual(len(events), 6)    # two steps per condition
        self.assertTrue(all(event["action"].dtype == np.float32 for event in events))
        self.assertTrue(all(event["true_pre"].shape == (6,) for event in events))
        self.assertTrue(all(event["post_observation"].shape == (25,) for event in events))


if __name__ == "__main__":
    unittest.main()
