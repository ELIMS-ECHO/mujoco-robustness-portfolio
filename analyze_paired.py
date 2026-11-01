"""对真实评估结果进行配对差异分析，不依赖额外统计库。

本工具计算精确双侧 McNemar（二项分布）检验的 p 值。结果仅供探索性分析，
不得据此宣称策略在不同模型版本之间具有因果意义上的鲁棒性。
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
        raise ValueError("不一致配对计数不能为负数")
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n))


def paired_analysis(records: list[dict], outcome: str = "final_success") -> dict:
    if outcome not in ("final_success", "ever_success"):
        raise ValueError("结果指标无效")
    paired: dict[tuple[str, float], dict[int, int]] = {}
    for r in records:
        method = str(r["method"])
        if method not in ("noisy", "ema"):
            continue
        sigma = float(r["sigma_m"])
        seed = int(r["seed"])
        val = int(r[outcome])
        if val not in (0, 1):
            raise ValueError("结果必须是二元值")
        group = paired.setdefault((method, sigma), {})
        if seed in group:
            raise ValueError(f"随机种子重复：{seed}，方法={method}，噪声标准差={sigma}")
        group[seed] = val
    sigmas = sorted({sigma for (method, sigma) in paired})
    if not sigmas:
        raise ValueError("没有找到未滤波策略与 EMA 策略的配对记录")
    results = []
    for sigma in sigmas:
        a = paired.get(("noisy", sigma))
        b = paired.get(("ema", sigma))
        if a is None or b is None or set(a) != set(b):
            raise ValueError(f"噪声标准差为{sigma}时数据未配对：两种方法必须使用相同随机种子")
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
        "method": "精确双侧 McNemar 检验；探索性结果，未进行多重比较校正",
        "warning": "配对依据为相同的初始环境随机种子；结果仅适用于一个固定预训练策略和当前高斯噪声模型。",
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
        print(f"噪声标准差={x['sigma_m']:.3f}，配对样本数={x['paired_seeds']}, "
              f"EMA-noisy={x['ema_minus_noisy_percentage_points']:+.1f} 个百分点，"
              f"exact p={x['two_sided_exact_mcnemar_p_exploratory']:.6g}")


if __name__ == "__main__":
    main()
