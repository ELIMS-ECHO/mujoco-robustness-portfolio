"""Explore paired differences for real evaluations without external statistics libs.

This tool uses exact two-sided McNemar (binomial) p-values. Treat the output as
exploratory and do NOT claim causal robustness across different checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


def exact_mcnemar(b: int, c: int) -> float:
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be nonnegative")
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n))


def paired_analysis(records: list[dict], outcome: str = "final_success") -> dict:
    if outcome not in ("final_success", "ever_success"):
        raise ValueError("invalid outcome")
    paired: dict[tuple[str, float], dict[int, int]] = {}
    for r in records:
        method = str(r["method"])
        if method not in ("noisy", "ema"):
            continue
        sigma = float(r["sigma_m"])
        seed = int(r["seed"])
        val = int(r[outcome])
        if val not in (0, 1):
            raise ValueError("outcome must be binary")
        group = paired.setdefault((method, sigma), {})
        if seed in group:
            raise ValueError(f"duplicate seed={seed} for method={method}, sigma={sigma}")
        group[seed] = val
    sigmas = sorted({sigma for (method, sigma) in paired})
    if not sigmas:
        raise ValueError("no noisy/EMA paired records found")
    results = []
    for sigma in sigmas:
        a = paired.get(("noisy", sigma))
        b = paired.get(("ema", sigma))
        if a is None or b is None or set(a) != set(b):
            raise ValueError(f"unpaired data at sigma={sigma}: same seeds required")
        counts = Counter((a[seed], b[seed]) for seed in sorted(a))
        n = len(a)
        better = counts[(0, 1)]
        worse = counts[(1, 0)]
        results.append({
            "sigma_m": sigma,
            "paired_seeds": n,
            "both_failed": counts[(0, 0)],
            "both_succeeded": counts[(1, 1)],
            "ema_only_succeeded": better,
            "noisy_only_succeeded": worse,
            "ema_minus_noisy_percentage_points": 100.0 * (better - worse) / n,
            "two_sided_exact_mcnemar_p_exploratory": exact_mcnemar(better, worse),
        })
    return {
        "outcome": outcome,
        "method": "exact two-sided McNemar; exploratory without multiplicity correction",
        "warning": "Pairing uses same initial environment seed; one fixed pretrained policy and Gaussian noise model only.",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.episodes.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    final = paired_analysis(records, "final_success")
    ever = paired_analysis(records, "ever_success")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "final_success": final, "ever_success": ever,
        "source_episodes_csv": str(args.episodes),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    for x in final["results"]:
        print(f"sigma={x['sigma_m']:.3f}, paired n={x['paired_seeds']}, "
              f"EMA-noisy={x['ema_minus_noisy_percentage_points']:+.1f} pp, "
              f"exact p={x['two_sided_exact_mcnemar_p_exploratory']:.6g}")


if __name__ == "__main__":
    main()
