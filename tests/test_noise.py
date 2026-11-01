"""Fast deterministic tests for observation integrity, statistics and evaluation loop.

Uses a synthetic mock environment: its success rates are NOT MuJoCo results.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_robustness import (
    PositionEMA, conditions, corrupt_position_observation, evaluate,
    failure_bucket, summarize, wilson_interval,
)


def make_obs(z=0.42):
    state = np.zeros(25, dtype=np.float32)
    state[0:3] = [1.0, 2.0, z + 0.1]
    state[3:6] = [1.03, 2.0, z]
    state[6:9] = state[3:6] - state[0:3]
    return {
        "observation": state,
        "achieved_goal": state[3:6].copy(),
        "desired_goal": np.array([1.2, 2.0, z + 0.1], dtype=np.float32),
    }


class MockModel:
    def predict(self, observation, deterministic=False):
        assert deterministic
        return np.zeros(4, dtype=np.float32), None


class MockEnv:
    def __init__(self):
        self.step_count = 0

    def reset(self, seed=None):
        assert seed is not None
        self.step_count = 0
        return make_obs(), {}

    def step(self, action):
        self.step_count += 1
        obs = make_obs(z=0.42 + 0.015 * self.step_count)
        # Synthetic success at steps 2 and 3, not a real robot result.
        return obs, -1.0, False, self.step_count == 3, {
            "is_success": self.step_count >= 2
        }


class RobustnessTests(unittest.TestCase):
    def test_noise_deterministic_and_non_mutating(self):
        obs = make_obs()
        a = corrupt_position_observation(obs, 0.01, np.random.default_rng(42))
        b = corrupt_position_observation(obs, 0.01, np.random.default_rng(42))
        np.testing.assert_array_equal(a["observation"], b["observation"])
        np.testing.assert_array_equal(obs["observation"], make_obs()["observation"])
        np.testing.assert_array_equal(a["achieved_goal"], a["observation"][3:6])
        np.testing.assert_allclose(
            a["observation"][6:9],
            a["observation"][3:6] - a["observation"][0:3],
            atol=1e-6,
        )
        np.testing.assert_array_equal(a["desired_goal"], obs["desired_goal"])

    def test_clean_is_exact_copy(self):
        obs = make_obs()
        clean = corrupt_position_observation(obs, 0, np.random.default_rng(0))
        np.testing.assert_array_equal(clean["observation"], obs["observation"])
        self.assertIsNot(clean["observation"], obs["observation"])

    def test_filter_reset_and_consistency(self):
        ema = PositionEMA(0.5)
        start = make_obs()
        ema.transform(start)
        changed = make_obs()
        changed["observation"][3:6] += 0.2
        result = ema.transform(changed)
        np.testing.assert_allclose(
            result["observation"][3:6],
            start["observation"][3:6] + 0.1,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            result["achieved_goal"], result["observation"][3:6]
        )
        ema.reset()
        np.testing.assert_allclose(
            ema.transform(changed)["observation"][3:6],
            changed["observation"][3:6],
        )

    def test_invalid_config_rejected(self):
        for sigma in [-.1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                corrupt_position_observation(make_obs(), sigma, np.random.default_rng(1))
        for alpha in [0, -1, 1.01, float("nan")]:
            with self.assertRaises(ValueError):
                PositionEMA(alpha)
        with self.assertRaises(ValueError):
            conditions([.01, .01])
        with self.assertRaises(ValueError):
            conditions([0])

    def test_invalid_shape_rejected(self):
        obs = make_obs()
        obs["observation"] = np.zeros(9, dtype=np.float32)
        with self.assertRaises(ValueError):
            corrupt_position_observation(obs, .01, np.random.default_rng(1))

    def test_wilson_and_summary(self):
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)
        for successes in (0, 5, 10):
            lower, upper = wilson_interval(successes, 10)
            self.assertTrue(0 <= lower <= successes / 10 <= upper <= 1)
        rows = [
            {"method": "clean", "sigma_m": 0.0, "ever_success": 1,
             "final_success": 1, "first_success_step": 7, "episode_return": -7.0,
             "failure_bucket_heuristic": "success"},
            {"method": "clean", "sigma_m": 0.0, "ever_success": 0,
             "final_success": 0, "first_success_step": "", "episode_return": -50.0,
             "failure_bucket_heuristic": "little_object_lift_heuristic"},
        ]
        summary = summarize(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["ever_success_rate"], 0.5)
        self.assertEqual(summary[0]["final_success_count"], 1)
        self.assertEqual(summary[0]["mean_first_success_step_conditional"], 7)

    def test_mock_env_evaluation_pipeline(self):
        rows = evaluate(MockModel(), MockEnv(), 3, 100, [.005, .01], .35, max_steps=3)
        self.assertEqual(len(rows), 15)  # 5 conditions x 3 paired seeds
        self.assertTrue(all(r["ever_success"] == 1 for r in rows))
        self.assertTrue(all(r["first_success_step"] == 2 for r in rows))
        self.assertTrue(all(r["final_success"] == 1 for r in rows))
        self.assertTrue(all(r["episode_steps"] == 3 for r in rows))
        self.assertEqual(len(summarize(rows)), 5)

    def test_failure_buckets_are_explicitly_heuristic(self):
        self.assertEqual(failure_bucket(.12, 0, False), "no_close_approach_heuristic")
        self.assertEqual(failure_bucket(.03, 0, False), "little_object_lift_heuristic")
        self.assertEqual(failure_bucket(.03, .08, False), "no_success_after_lift_heuristic")
        self.assertEqual(failure_bucket(.03, .08, True), "success")


if __name__ == "__main__":
    unittest.main()
