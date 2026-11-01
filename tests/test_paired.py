"""使用小规模合成案例检查精确 McNemar 配对检验。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_paired import exact_mcnemar, paired_analysis


class PairedAnalysisTests(unittest.TestCase):
    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar(0, 0), 1)
        self.assertEqual(exact_mcnemar(14, 1), 0.0009765625)
        self.assertEqual(exact_mcnemar(5, 0), 0.0625)

    def test_pairing(self):
        rows = [
            {"method": method, "sigma_m": "0.02", "seed": seed,
             "final_success": value, "ever_success": value}
            for method, seed, value in [
                ("noisy", 3, 0), ("noisy", 4, 1),
                ("ema", 3, 1), ("ema", 4, 1),
            ]
        ]
        result = paired_analysis(rows)
        d = result["results"][0]
        self.assertEqual(d["paired_seeds"], 2)
        self.assertEqual(d["ema_only_succeeded"], 1)
        self.assertEqual(d["both_succeeded"], 1)
        self.assertEqual(d["ema_minus_noisy_percentage_points"], 50)

    def test_reject_unpaired(self):
        rows = [
            {"method": "noisy", "sigma_m": "0.01", "seed": 100,
             "final_success": 1},
            {"method": "ema", "sigma_m": "0.01", "seed": 101,
             "final_success": 1},
        ]
        with self.assertRaises(ValueError):
            paired_analysis(rows)


if __name__ == "__main__":
    unittest.main()
