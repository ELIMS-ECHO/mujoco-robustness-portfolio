"""根据真实实验生成的 summary.json 绘制适合报告展示的成功率对比图。

合成数据或手工拼装的 JSON 不能作为真实实验测量证据。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(summary_path: Path, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "measured_on_local_environment":
        raise ValueError("该文件不是环境实测结果")
    if len(str(payload.get("model_sha256", ""))) != 64:
        raise ValueError("缺少模型来源对应的 SHA-256")
    rows = payload.get("summary", [])
    clean = next((r for r in rows if r["method"] == "clean"), None)
    if clean is None:
        raise ValueError("缺少无噪声基线")
    sigmas = sorted({float(r["sigma_m"]) for r in rows if r["method"] != "clean"})
    if not sigmas:
        raise ValueError("没有噪声实验条件")

    fig, ax = plt.subplots(figsize=(8.5, 4.7), constrained_layout=True)
    clean_rate = float(clean["ever_success_rate"])
    ax.axhline(clean_rate * 100, linestyle="--", linewidth=1.5,
               label=f"Clean baseline: {clean_rate:.1%}")
    for method, title in (("noisy", "未滤波策略"), ("ema", "噪声观测 + EMA 滤波")):
        found = {float(r["sigma_m"]): r for r in rows if r["method"] == method}
        missing = [s for s in sigmas if s not in found]
        if missing:
            raise ValueError(f"缺少 {method} 噪声条件：{missing}")
        y = [float(found[s]["ever_success_rate"]) * 100 for s in sigmas]
        limits = [found[s]["ever_success_95pct_wilson"] for s in sigmas]
        lower = [y[i] - float(limits[i][0]) * 100 for i in range(len(sigmas))]
        upper = [float(limits[i][1]) * 100 - y[i] for i in range(len(sigmas))]
        ax.errorbar([s * 1000 for s in sigmas], y, yerr=[lower, upper],
                    label=title, marker="o", capsize=3, linewidth=1.8)
    ax.set_ylim(-2, 102)
    ax.set_xticks([s * 1000 for s in sigmas])
    ax.set_xlabel("位置观测高斯噪声标准差 σ（mm）")
    ax.set_ylabel("曾达成目标比例（%）")
    ax.set_title(
        f"FetchPickAndPlace-v4 · SAC 策略鲁棒性\\n"
        f"n={payload['episodes_per_condition']} 每组条件 · 95% Wilson 置信区间"
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot(args.summary, args.output)
    print(f"实测数据图表已保存至 {args.output}")


if __name__ == "__main__":
    main()
