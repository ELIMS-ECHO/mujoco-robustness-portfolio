"""Record and analyze step-level FetchPickAndPlace trajectories for EMA diagnostics.

This is a supplementary diagnostic experiment. It does not train or modify the
upstream SAC policy, and it does not replace the 210-episode headline evaluation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_robustness import (
    ENV_ID,
    conditions,
    dependency_versions,
    evaluate,
)

TRACE_FIELDS = [
    "method", "sigma_m", "seed", "step",
    "gripper_true_x_m", "gripper_true_y_m", "gripper_true_z_m",
    "object_true_x_m", "object_true_y_m", "object_true_z_m",
    "gripper_noisy_x_m", "gripper_noisy_y_m", "gripper_noisy_z_m",
    "object_noisy_x_m", "object_noisy_y_m", "object_noisy_z_m",
    "gripper_policy_x_m", "gripper_policy_y_m", "gripper_policy_z_m",
    "object_policy_x_m", "object_policy_y_m", "object_policy_z_m",
    "relative_true_x_m", "relative_true_y_m", "relative_true_z_m",
    "relative_noisy_x_m", "relative_noisy_y_m", "relative_noisy_z_m",
    "relative_policy_x_m", "relative_policy_y_m", "relative_policy_z_m",
    "gripper_raw_error_m", "object_raw_error_m", "relative_raw_error_m",
    "gripper_policy_error_m", "object_policy_error_m", "relative_policy_error_m",
    "gripper_lag_projection_m", "object_lag_projection_m", "relative_lag_projection_m",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "post_gripper_x_m", "post_gripper_y_m", "post_gripper_z_m",
    "post_object_x_m", "post_object_y_m", "post_object_z_m",
    "object_goal_distance_m", "gripper_object_distance_m",
    "object_z_rise_m", "reward", "is_success",
]

EPISODE_METRICS = [
    "gripper_raw_rmse_m",
    "gripper_policy_rmse_m",
    "object_raw_rmse_m",
    "object_policy_rmse_m",
    "relative_raw_rmse_m",
    "relative_policy_rmse_m",
    "gripper_lag_projection_mean_m",
    "object_lag_projection_mean_m",
    "relative_lag_projection_mean_m",
    "mean_object_goal_distance_m",
    "final_object_goal_distance_m",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directional_lag_projection(
    true_position: np.ndarray,
    estimated_position: np.ndarray,
    previous_true_position: np.ndarray | None,
) -> float | None:
    """Project estimation error onto recent motion direction.

    Positive values mean the estimate is behind the true position along the
    latest displacement. This is a directional diagnostic, not an exact delay.
    """
    if previous_true_position is None:
        return None
    current = np.asarray(true_position, dtype=np.float64)
    estimate = np.asarray(estimated_position, dtype=np.float64)
    previous = np.asarray(previous_true_position, dtype=np.float64)
    displacement = current - previous
    distance = float(np.linalg.norm(displacement))
    if distance < 1e-8:
        return None
    return float(np.dot(current - estimate, displacement / distance))


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _mean_or_none(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = float(statistics.mean(values))
    sd = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return mean, sd


def _position_fields(prefix: str, xyz: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_x_m": float(xyz[0]),
        f"{prefix}_y_m": float(xyz[1]),
        f"{prefix}_z_m": float(xyz[2]),
    }


def run_diagnostics(model, env, episode_count: int, seed_start: int,
                    sigmas: list[float], alpha: float, max_steps: int):
    """Use the primary evaluator's exact loop and attach a read-only trace callback."""
    trace_rows: list[dict] = []
    previous_positions: dict[tuple[str, float, int], tuple[np.ndarray, np.ndarray]] = {}
    actions: dict[tuple[float, int, int, str], np.ndarray] = {}

    def collect_step(event: dict) -> None:
        method = str(event["method"])
        sigma = float(event["sigma_m"])
        seed = int(event["seed"])
        step = int(event["step"])
        true_pre = np.asarray(event["true_pre"], dtype=np.float64)
        noisy_pre = np.asarray(event["noisy_pre"], dtype=np.float64)
        policy_pre = np.asarray(event["policy_pre"], dtype=np.float64)
        previous = previous_positions.get((method, sigma, seed))
        previous_gripper = None if previous is None else previous[0]
        previous_object = None if previous is None else previous[1]
        previous_relative = (
            None if previous is None else previous[1] - previous[0]
        )
        relative_true = true_pre[3:6] - true_pre[0:3]
        relative_noisy = noisy_pre[3:6] - noisy_pre[0:3]
        relative_policy = policy_pre[3:6] - policy_pre[0:3]

        gripper_lag = directional_lag_projection(
            true_pre[0:3], policy_pre[0:3], previous_gripper
        )
        object_lag = directional_lag_projection(
            true_pre[3:6], policy_pre[3:6], previous_object
        )
        relative_lag = directional_lag_projection(
            relative_true, relative_policy, previous_relative
        )

        action = np.asarray(event["action"])
        action_flat = action.reshape(-1)
        if action_flat.size != 4:
            raise ValueError(f"策略动作维度应为 4，实际为 {action_flat.size}")
        actions[(sigma, seed, step, method)] = action_flat.astype(np.float64, copy=True)

        post_state = np.asarray(event["post_observation"], dtype=np.float64)
        post_gripper = post_state[0:3].copy()
        post_object = post_state[3:6].copy()
        goal_distance = float(
            np.linalg.norm(event["post_achieved_goal"] - event["post_desired_goal"])
        )
        gripper_object_distance = float(np.linalg.norm(post_state[6:9]))
        object_z_rise = float(post_object[2] - float(event["initial_object_z"]))

        row = {
            "method": method,
            "sigma_m": sigma,
            "seed": seed,
            "step": step,
            **_position_fields("gripper_true", true_pre[0:3]),
            **_position_fields("object_true", true_pre[3:6]),
            **_position_fields("gripper_noisy", noisy_pre[0:3]),
            **_position_fields("object_noisy", noisy_pre[3:6]),
            **_position_fields("gripper_policy", policy_pre[0:3]),
            **_position_fields("object_policy", policy_pre[3:6]),
            **_position_fields("relative_true", relative_true),
            **_position_fields("relative_noisy", relative_noisy),
            **_position_fields("relative_policy", relative_policy),
            "gripper_raw_error_m": float(np.linalg.norm(noisy_pre[0:3] - true_pre[0:3])),
            "object_raw_error_m": float(np.linalg.norm(noisy_pre[3:6] - true_pre[3:6])),
            "relative_raw_error_m": float(np.linalg.norm(relative_noisy - relative_true)),
            "gripper_policy_error_m": float(np.linalg.norm(policy_pre[0:3] - true_pre[0:3])),
            "object_policy_error_m": float(np.linalg.norm(policy_pre[3:6] - true_pre[3:6])),
            "relative_policy_error_m": float(np.linalg.norm(relative_policy - relative_true)),
            "gripper_lag_projection_m": "" if gripper_lag is None else gripper_lag,
            "object_lag_projection_m": "" if object_lag is None else object_lag,
            "relative_lag_projection_m": "" if relative_lag is None else relative_lag,
            "action_dx": float(action_flat[0]),
            "action_dy": float(action_flat[1]),
            "action_dz": float(action_flat[2]),
            "action_gripper": float(action_flat[3]),
            **_position_fields("post_gripper", post_gripper),
            **_position_fields("post_object", post_object),
            "object_goal_distance_m": goal_distance,
            "gripper_object_distance_m": gripper_object_distance,
            "object_z_rise_m": object_z_rise,
            "reward": float(event["reward"]),
            "is_success": int(bool(event["is_success"])),
        }
        trace_rows.append(row)
        previous_positions[(method, sigma, seed)] = (
            true_pre[0:3].copy(), true_pre[3:6].copy()
        )

    # 关键：不在诊断脚本里复制一套 env.step 循环。
    # 由主评估器执行同一份策略推理、噪声注入和环境推进逻辑；
    # callback 只读取每一步的副本，不改变被测环境或 action dtype。
    records = evaluate(
        model, env, episode_count, seed_start, sigmas, alpha, max_steps,
        trace_callback=collect_step,
    )

    traces_by_episode: dict[tuple[str, float, int], list[dict]] = defaultdict(list)
    for row in trace_rows:
        traces_by_episode[(row["method"], float(row["sigma_m"]), int(row["seed"]))].append(row)

    episode_rows = []
    for record in records:
        key = (record["method"], float(record["sigma_m"]), int(record["seed"]))
        rows = traces_by_episode[key]
        gripper_raw = [float(row["gripper_raw_error_m"]) for row in rows]
        gripper_policy = [float(row["gripper_policy_error_m"]) for row in rows]
        object_raw = [float(row["object_raw_error_m"]) for row in rows]
        object_policy = [float(row["object_policy_error_m"]) for row in rows]
        gripper_lag = [
            float(row["gripper_lag_projection_m"])
            for row in rows if row["gripper_lag_projection_m"] != ""
        ]
        object_lag = [
            float(row["object_lag_projection_m"])
            for row in rows if row["object_lag_projection_m"] != ""
        ]
        relative_raw = [float(row["relative_raw_error_m"]) for row in rows]
        relative_policy = [float(row["relative_policy_error_m"]) for row in rows]
        relative_lag = [
            float(row["relative_lag_projection_m"])
            for row in rows if row["relative_lag_projection_m"] != ""
        ]
        goal_distances = [float(row["object_goal_distance_m"]) for row in rows]
        episode_rows.append({
            **record,
            "gripper_raw_rmse_m": _rms(gripper_raw),
            "gripper_policy_rmse_m": _rms(gripper_policy),
            "object_raw_rmse_m": _rms(object_raw),
            "object_policy_rmse_m": _rms(object_policy),
            "relative_raw_rmse_m": _rms(relative_raw),
            "relative_policy_rmse_m": _rms(relative_policy),
            "gripper_lag_projection_mean_m": _mean_or_none(gripper_lag),
            "object_lag_projection_mean_m": _mean_or_none(object_lag),
            "relative_lag_projection_mean_m": _mean_or_none(relative_lag),
            "mean_object_goal_distance_m": _mean_or_none(goal_distances),
            "final_object_goal_distance_m": goal_distances[-1] if goal_distances else None,
        })

    action_divergence: dict[float, list[float]] = defaultdict(list)
    seeds = range(seed_start, seed_start + episode_count)
    for sigma in sigmas:
        for seed in seeds:
            for step in range(1, max_steps + 1):
                noisy_action = actions.get((sigma, seed, step, "noisy"))
                ema_action = actions.get((sigma, seed, step, "ema"))
                if noisy_action is not None and ema_action is not None:
                    action_divergence[sigma].append(
                        float(np.linalg.norm(ema_action - noisy_action))
                    )

    summaries = []
    for method, sigma in conditions(sigmas):
        group = [
            row for row in episode_rows
            if row["method"] == method and float(row["sigma_m"]) == float(sigma)
        ]
        count = len(group)
        result = {
            "method": method,
            "sigma_m": sigma,
            "episodes": count,
            "ever_success_count": sum(int(row["ever_success"]) for row in group),
            "ever_success_rate": (
                sum(int(row["ever_success"]) for row in group) / count if count else None
            ),
            "final_success_count": sum(int(row["final_success"]) for row in group),
            "final_success_rate": (
                sum(int(row["final_success"]) for row in group) / count if count else None
            ),
            "mean_first_success_step_conditional": _mean_or_none([
                float(row["first_success_step"]) for row in group
                if row["first_success_step"] != ""
            ]),
        }
        for metric in EPISODE_METRICS:
            values = [
                float(row[metric]) for row in group if row[metric] is not None
            ]
            mean, sd = _mean_sd(values)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
        result["mean_episode_return"] = _mean_or_none([
            float(row["episode_return"]) for row in group
        ])
        result["mean_action_l2_difference_vs_noisy"] = (
            _mean_or_none(action_divergence[sigma]) if method == "ema" else None
        )
        summaries.append(result)

    curve_groups: dict[tuple[str, float, int], list[dict]] = defaultdict(list)
    for row in trace_rows:
        curve_groups[(row["method"], float(row["sigma_m"]), int(row["step"]))].append(row)

    step_curves = []
    for (method, sigma, step), group in sorted(curve_groups.items()):
        curve = {"method": method, "sigma_m": sigma, "step": step}
        for metric in (
            "gripper_policy_error_m",
            "object_policy_error_m",
            "object_goal_distance_m",
        ):
            values = [float(row[metric]) for row in group]
            mean, sd = _mean_sd(values)
            curve[f"{metric}_mean"] = mean
            curve[f"{metric}_sd"] = sd
        step_curves.append(curve)

    return episode_rows, summaries, step_curves, trace_rows

def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# EMA 观测滤波轨迹诊断报告",
        "",
        f"- 环境：{payload['environment']}",
        f"- 模型 SHA-256：{payload['model_sha256']}",
        f"- 每个条件的环境随机种子：{payload['seed_start']}–{payload['seed_start'] + payload['episodes_per_condition'] - 1}",
        f"- 每条件回合数：{payload['episodes_per_condition']}",
        f"- EMA 系数：{payload['alpha']}",
        f"- 最大步数：{payload['max_steps']}",
        "",
        "## 观测误差：原始测量与策略实际输入",
        "",
        "| 方法 | σ (mm) | 夹爪原始 RMSE (mm) | 夹爪输入 RMSE (mm) | 物块原始 RMSE (mm) | 物块输入 RMSE (mm) | 相对位置原始 RMSE (mm) | 相对位置输入 RMSE (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        values = [
            row["method"],
            f"{row['sigma_m'] * 1000:.0f}",
            row["gripper_raw_rmse_m_mean"],
            row["gripper_policy_rmse_m_mean"],
            row["object_raw_rmse_m_mean"],
            row["object_policy_rmse_m_mean"],
            row["relative_raw_rmse_m_mean"],
            row["relative_policy_rmse_m_mean"],
        ]
        formatted = [str(values[0]), str(values[1])]
        formatted.extend("—" if value is None else f"{value * 1000:.2f}" for value in values[2:])
        lines.append("| " + " | ".join(formatted) + " |")

    lines.extend([
        "",
        "## 闭环任务表现与方向滞后诊断",
        "",
        "| 方法 | σ (mm) | 终态成功率 | 夹爪方向滞后投影 (mm) | 物块方向滞后投影 (mm) | 相对位置方向滞后投影 (mm) | 平均物块-目标距离 (m) | 动作差异 L2（相对未滤波） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["summary"]:
        values = [
            row["method"],
            f"{row['sigma_m'] * 1000:.0f}",
            f"{row['final_success_rate']:.1%}",
            row["gripper_lag_projection_mean_m_mean"],
            row["object_lag_projection_mean_m_mean"],
            row["relative_lag_projection_mean_m_mean"],
            row["mean_object_goal_distance_m_mean"],
            row["mean_action_l2_difference_vs_noisy"],
        ]
        formatted = [str(values[0]), str(values[1]), str(values[2])]
        formatted.extend("—" if value is None else f"{value * 1000:.2f}" for value in values[3:6])
        formatted.append("—" if values[6] is None else f"{values[6]:.4f}")
        formatted.append("—" if values[7] is None else f"{values[7]:.4f}")
        lines.append("| " + " | ".join(formatted) + " |")

    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 这是轨迹机制诊断，不是新的独立测试集。它使用与主评估相同的环境随机种子范围，目的是解释逐步行为，不替代 210 回合总体评估。",
        "- 仅对策略输入中的夹爪与物块 XYZ 位置加入独立高斯噪声；速度等其他观测字段仍来自仿真环境，未加入噪声。",
        "- 相对位置定义为物块位置减去夹爪位置；策略输入中的相对位置由滤波后的两个位置重新计算。",
        "- 策略输入 RMSE 是策略实际接收到的位置与当前仿真真实位置之差，包含测量噪声与滤波动态效应。",
        "- 方向滞后投影是估计误差沿最近真实位移方向的投影。正值表示估计位置落后于真实运动方向；它是诊断指标，不是精确时间延迟。",
        "- 未滤波与 EMA 的闭环动作不同，之后的真实轨迹也会不同。因此轨迹差异包含策略反馈后的状态变化，不能只归因于滤波器单一因素。",
        "- 只有当观测误差、相对位置误差、方向滞后、动作变化和任务轨迹共同支持时，才提出机制解释；不把相关性写成因果证明。",
        "",
        "## 生成文件",
        "",
        "- trajectory_steps.csv.gz：每个仿真步的真实位置、噪声位置、策略输入、动作和任务状态。",
        "- episode_summary.csv：每个回合的成功结果与轨迹误差汇总。",
        "- diagnostics_summary.json：机器可读的逐条件汇总与逐步曲线。",
        "- position_error_comparison.png：夹爪、物块与相对位置误差对比。",
        "- goal_distance_20mm.png：20 mm 条件下实际物块到目标距离的逐步轨迹。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")

def create_plots(output_dir: Path, summaries: list[dict], step_curves: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    metrics = [
        ("gripper_policy_rmse_m", "夹爪位置估计误差"),
        ("object_policy_rmse_m", "物块位置估计误差"),
        ("relative_policy_rmse_m", "夹爪-物块相对位置误差"),
    ]
    for axis, (metric, title) in zip(axes, metrics):
        for method, label in (("noisy", "未滤波"), ("ema", "EMA")):
            rows = [r for r in summaries if r["method"] == method]
            rows.sort(key=lambda r: r["sigma_m"])
            x = [r["sigma_m"] * 1000 for r in rows]
            y = [r[f"{metric}_mean"] * 1000 for r in rows]
            sd = [r[f"{metric}_sd"] * 1000 for r in rows]
            axis.errorbar(x, y, yerr=sd, marker="o", capsize=3, label=label)
        axis.set_title(title)
        axis.set_xlabel("位置观测噪声标准差 σ（mm）")
        axis.set_ylabel("每回合 RMSE 均值 ± SD（mm）")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output_dir / "position_error_comparison.png", dpi=170)
    plt.close(fig)

    sigmas = sorted({float(row["sigma_m"]) for row in summaries if row["method"] != "clean"})
    sigma = max(sigmas) if sigmas else 0.02
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for method, label in (("noisy", "未滤波"), ("ema", "EMA")):
        rows = [
            row for row in step_curves
            if row["method"] == method and abs(float(row["sigma_m"]) - sigma) < 1e-12
        ]
        rows.sort(key=lambda row: row["step"])
        if not rows:
            continue
        x = np.asarray([row["step"] for row in rows], dtype=np.int32)
        y = np.asarray([row["object_goal_distance_m_mean"] for row in rows], dtype=np.float64)
        sd = np.asarray([row["object_goal_distance_m_sd"] or 0.0 for row in rows], dtype=np.float64)
        ax.plot(x, y, label=label, linewidth=2)
        ax.fill_between(x, np.maximum(0, y - sd), y + sd, alpha=0.16)
    ax.set_title(f"{sigma * 1000:.0f} mm 噪声：物块到目标的真实距离轨迹")
    ax.set_xlabel("环境控制步")
    ax.set_ylabel("物块到目标距离（m）")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "goal_distance_20mm.png", dpi=170)
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("results/trajectory-diagnostics"))
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"未找到模型文件：{args.model}")
    if args.episodes < 1 or args.seed_start < 0 or args.max_steps < 1:
        parser.error("episodes/max-steps 必须为正数，seed-start 不能为负数")
    if not 0 < args.alpha <= 1 or not np.isfinite(args.alpha):
        parser.error("alpha 必须位于 (0, 1]")

    import gymnasium as gym
    import gymnasium_robotics
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import DictReplayBuffer

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, max_episode_steps=args.max_steps)
    try:
        model = SAC.load(str(args.model), env=env, device="cpu", custom_objects={
            "replay_buffer_class": DictReplayBuffer,
            "replay_buffer_kwargs": {},
            "buffer_size": 1,
        })
        episode_rows, summaries, step_curves, trace_rows = run_diagnostics(
            model, env, args.episodes, args.seed_start, args.sigmas,
            args.alpha, args.max_steps,
        )
    finally:
        env.close()

    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output / "trajectory_steps.csv.gz", "wt",
                   newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(trace_rows)

    episode_fields = list(episode_rows[0].keys())
    with (args.output / "episode_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=episode_fields)
        writer.writeheader()
        writer.writerows(episode_rows)

    payload = {
        "status": "measured_in_mujoco_simulation",
        "environment": ENV_ID,
        "model_sha256": sha256_file(args.model),
        "model_path": "upstream-model/best_model.zip (downloaded at runtime; not stored in repository)",
        "execution_platform": platform.platform(),
        "dependency_versions": dependency_versions(),
        "episodes_per_condition": args.episodes,
        "seed_start": args.seed_start,
        "alpha": args.alpha,
        "sigmas_m": args.sigmas,
        "max_steps": args.max_steps,
        "conditions": len(summaries),
        "trajectory_rows": len(trace_rows),
        "noise_scope": "独立高斯噪声仅注入夹爪与物块 XYZ 位置；速度等其他观测字段保持仿真环境原值",
        "relative_position_definition": "物块 XYZ 位置减去夹爪 XYZ 位置；由策略实际接收的两个位置重新计算",
        "lag_metric": "真实位置减策略输入位置，沿最近真实位移方向投影；正值表示估计位置落后于运动方向",
        "evaluation_loop": "复用 evaluate_robustness.evaluate 的同一环境推进循环；轨迹回调只读状态副本，动作原始 dtype 不变",
        "summary": summaries,
        "step_curves": step_curves,
    }
    (args.output / "diagnostics_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output / "diagnostics_report.md", payload)
    create_plots(args.output, summaries, step_curves)

    print(f"轨迹步数记录：{len(trace_rows)}")
    print(f"逐条件汇总：{args.output / 'diagnostics_summary.json'}")
    print(f"诊断报告：{args.output / 'diagnostics_report.md'}")


if __name__ == "__main__":
    main()
