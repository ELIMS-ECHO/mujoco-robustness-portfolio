"""Render a publication-style success-rate comparison from genuine summary.json.

Synthetic or manually assembled JSON does not qualify as measured experiment evidence.
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
        raise ValueError("Not a measured environment run")
    if len(str(payload.get("model_sha256", ""))) != 64:
        raise ValueError("Model provenance SHA-256 is missing")
    rows = payload.get("summary", [])
    clean = next((r for r in rows if r["method"] == "clean"), None)
    if clean is None:
        raise ValueError("Missing clean baseline")
    sigmas = sorted({float(r["sigma_m"]) for r in rows if r["method"] != "clean"})
    if not sigmas:
        raise ValueError("No noise conditions")

    fig, ax = plt.subplots(figsize=(8.5, 4.7), constrained_layout=True)
    clean_rate = float(clean["ever_success_rate"])
    ax.axhline(clean_rate * 100, linestyle="--", linewidth=1.5,
               label=f"Clean baseline: {clean_rate:.1%}")
    for method, title in (("noisy", "Noisy observation"), ("ema", "Noise + EMA")):
        found = {float(r["sigma_m"]): r for r in rows if r["method"] == method}
        missing = [s for s in sigmas if s not in found]
        if missing:
            raise ValueError(f"Missing {method} noise conditions: {missing}")
        y = [float(found[s]["ever_success_rate"]) * 100 for s in sigmas]
        limits = [found[s]["ever_success_95pct_wilson"] for s in sigmas]
        lower = [y[i] - float(limits[i][0]) * 100 for i in range(len(sigmas))]
        upper = [float(limits[i][1]) * 100 - y[i] for i in range(len(sigmas))]
        ax.errorbar([s * 1000 for s in sigmas], y, yerr=[lower, upper],
                    label=title, marker="o", capsize=3, linewidth=1.8)
    ax.set_ylim(-2, 102)
    ax.set_xticks([s * 1000 for s in sigmas])
    ax.set_xlabel("Position observation noise, Gaussian σ (mm)")
    ax.set_ylabel("Ever-success rate (%)")
    ax.set_title(
        f"FetchPickAndPlace-v4 · SAC policy robustness\\n"
        f"n={payload['episodes_per_condition']} per condition · 95% Wilson intervals"
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
    print(f"Saved measured-data figure to {args.output}")


if __name__ == "__main__":
    main()
