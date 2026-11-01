"""评估预训练 FetchPickAndPlace SAC 策略在位置观测噪声下的表现。

本程序不执行模型训练。这是独立编写的评估代码，不是上游 SAC/HER 训练实现的复制版本。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

ENV_ID = "FetchPickAndPlace-v4"
DEFAULT_MAX_STEPS = 50


def _consistent_state(obs: dict) -> None:
    state = obs["observation"]
    if state.shape != (25,) or obs["achieved_goal"].shape != (3,) or obs["desired_goal"].shape != (3,):
        raise ValueError("FetchPickAndPlace-v4 的 Dict 观测维度应为 25/3/3")
    state[6:9] = state[3:6] - state[0:3]
    obs["achieved_goal"] = np.array(state[3:6], copy=True)


def corrupt_position_observation(observation: dict, sigma: float, rng: np.random.Generator) -> dict:
    """对观测到的夹爪与物块位置（单位：米）加入噪声，同时保持字段一致。

    FetchPickAndPlace observation[0:3]：夹爪位置
                           [3:6]：物块位置
                           [6:9]：物块位置减去夹爪位置
    achieved_goal 表示物块位置，也必须同步加入噪声，避免目标条件策略
    通过未受扰动的物块坐标获得信息。
    desired_goal 保持为环境指定的目标位置。
    """
    if sigma < 0 or not np.isfinite(sigma):
        raise ValueError("sigma 必须是有限的非负数")
    result = {key: np.array(value, copy=True) for key, value in observation.items()}
    _consistent_state(result)
    if sigma == 0:
        return result
    state = result["observation"]
    state[0:3] += rng.normal(0.0, sigma, size=3).astype(state.dtype)
    state[3:6] += rng.normal(0.0, sigma, size=3).astype(state.dtype)
    _consistent_state(result)
    return result


class PositionEMA:
    """对测得的夹爪与物块位置计算指数移动平均。"""

    def __init__(self, alpha: float):
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("alpha 必须位于区间 (0, 1] 内")
        self.alpha = alpha
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def transform(self, observation: dict) -> dict:
        result = {key: np.array(value, copy=True) for key, value in observation.items()}
        _consistent_state(result)
        state = result["observation"]
        measured = np.array(state[0:6], copy=True)
        if self._previous is None:
            filtered = measured
        else:
            filtered = self.alpha * measured + (1.0 - self.alpha) * self._previous
        self._previous = np.array(filtered, copy=True)
        state[0:6] = filtered
        _consistent_state(result)
        return result


def wilson_interval(successes: int, count: int, z: float = 1.9599639845) -> list[float]:
    """近似计算二项比例的 95% 置信区间；计算时将各回合视为相互独立。"""
    if count <= 0 or successes < 0 or successes > count:
        raise ValueError("必须满足 0 <= successes <= count 且 count > 0")
    p = successes / count
    denom = 1.0 + z * z / count
    center = (p + z * z / (2 * count)) / denom
    half_width = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return [max(0.0, min(p, center - half_width)), min(1.0, max(p, center + half_width))]


def conditions(sigmas: list[float]) -> list[tuple[str, float]]:
    if not sigmas or any(s <= 0 or not np.isfinite(s) for s in sigmas):
        raise ValueError("所有噪声标准差都必须是有限正数")
    if len(set(sigmas)) != len(sigmas):
        raise ValueError("噪声标准差不能重复")
    return [("clean", 0.0)] + [
        (method, sigma) for sigma in sigmas for method in ("noisy", "ema")
    ]


def failure_bucket(min_gripper_distance: float, max_object_rise: float, ever_success: bool) -> str:
    """探索性诊断启发式规则，并非经过验证的抓取失败标签。"""
    if ever_success:
        return "success"
    if min_gripper_distance > 0.08:
        return "no_close_approach_heuristic"
    if max_object_rise < 0.04:
        return "little_object_lift_heuristic"
    return "no_success_after_lift_heuristic"


def evaluate(model, env, episode_count: int, seed_start: int, sigmas: list[float],
             alpha: float, max_steps: int = DEFAULT_MAX_STEPS) -> list[dict]:
    if episode_count < 1 or seed_start < 0 or max_steps < 1:
        raise ValueError("episode_count 和 max_steps 必须为正数，seed_start 不能为负数")
    records: list[dict] = []
    for method, sigma in conditions(sigmas):
        for index in range(episode_count):
            seed = seed_start + index
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(seed * 1009 + 17)
            filter_ = PositionEMA(alpha)
            filter_.reset()
            steps = 0
            first_success: int | None = None
            last_success = False
            episode_return = 0.0
            initial_z = float(obs["observation"][5])
            max_rise = 0.0
            min_gripper_distance = float("inf")
            min_goal_distance = float("inf")
            while steps < max_steps:
                if method == "clean":
                    policy_obs = {key: np.array(value, copy=True) for key, value in obs.items()}
                else:
                    policy_obs = corrupt_position_observation(obs, sigma, rng)
                    if method == "ema":
                        policy_obs = filter_.transform(policy_obs)
                action, _ = model.predict(policy_obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                episode_return += float(reward)
                state = obs["observation"]
                max_rise = max(max_rise, float(state[5]) - initial_z)
                min_gripper_distance = min(min_gripper_distance, float(np.linalg.norm(state[6:9])))
                min_goal_distance = min(
                    min_goal_distance,
                    float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
                )
                last_success = bool(info.get("is_success", False))
                if last_success 和 first_success is None:
                    first_success = steps
                if terminated or truncated:
                    break
            records.append({
                "method": method,
                "sigma_m": sigma,
                "episode_index": index,
                "seed": seed,
                "ever_success": int(first_success is not None),
                "final_success": int(last_success),
                "first_success_step": first_success if first_success is not None else "",
                "episode_steps": steps,
                "episode_return": episode_return,
                "min_object_goal_distance_m": min_goal_distance,
                "min_gripper_object_distance_m": min_gripper_distance,
                "max_object_z_rise_m": max_rise,
                "failure_bucket_heuristic": failure_bucket(
                    min_gripper_distance, max_rise, first_success is not None
                ),
            })
        print(f"已完成方法={method} sigma={sigma:g}，回合数={episode_count}")
    return records


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = {}
    for record in records:
        key = (record["method"], record["sigma_m"])
        groups.setdefault(key, []).append(record)
    summaries = []
    for (method, sigma), group in groups.items():
        count = len(group)
        ever = sum(int(r["ever_success"]) for r in group)
        final = sum(int(r["final_success"]) for r in group)
        first_steps = [int(r["first_success_step"]) for r in group if r["first_success_step"] != ""]
        buckets: dict[str, int] = {}
        for r in group:
            key = r["failure_bucket_heuristic"]
            buckets[key] = buckets.get(key, 0) + 1
        summaries.append({
            "method": method,
            "sigma_m": sigma,
            "episodes": count,
            "ever_success_count": ever,
            "ever_success_rate": ever / count,
            "ever_success_95pct_wilson": wilson_interval(ever, count),
            "final_success_count": final,
            "final_success_rate": final / count,
            "final_success_95pct_wilson": wilson_interval(final, count),
            "mean_first_success_step_conditional": (
                sum(first_steps) / len(first_steps) if first_steps else None
            ),
            "mean_episode_return": sum(float(r["episode_return"]) for r in group) / count,
            "failure_buckets_heuristic": buckets,
        })
    return summaries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "gymnasium", "gymnasium-robotics", "mujoco", "stable-baselines3", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="已有的 SAC .zip 模型检查点")
    parser.add_argument("--episodes", type=int, default=20, help="配对评估使用的随机种子数量")
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--sigmas", nargs="+", type=float, default=[0.005, 0.01, 0.02],
                        help="位置噪声标准差，单位为米")
    parser.add_argument("--alpha", type=float, default=0.35, help="EMA 平滑系数")
    parser.add_argument(
        "--hf-compatible", action="store_true",
        help="按照 Hugging Face 模型卡建议，使用 DictReplayBuffer 兼容加载参数"
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"未找到模型文件：{args.model}")
    if args.episodes < 1 or args.seed_start < 0 or args.max_steps < 1:
        parser.error("episodes 必须为正数，seed-start 不能为负数")
    if not np.isfinite(args.alpha) or not 0 < args.alpha <= 1:
        parser.error("alpha 必须位于区间 (0, 1] 内")
    try:
        conditions(args.sigmas)
    except ValueError as error:
        parser.error(str(error))

    import gymnasium as gym
    import gymnasium_robotics
    from stable_baselines3 import SAC

    gym.register_envs(gymnasium_robotics)
    env = gym.make(ENV_ID, max_episode_steps=args.max_steps)
    try:
        load_kwargs = {}
        if args.hf_compatible:
            from stable_baselines3.common.buffers import DictReplayBuffer
            # 按照模型卡建议，仅为推理加载设置兼容参数。
            # 预训练策略权重保持不变。
            load_kwargs["custom_objects"] = {
                "replay_buffer_class": DictReplayBuffer,
                "replay_buffer_kwargs": {},
                "buffer_size": 1,
            }
        model = SAC.load(str(args.model), env=env, device="cpu", **load_kwargs)
        records = evaluate(
            model, env, args.episodes, args.seed_start, args.sigmas, args.alpha, args.max_steps
        )
    finally:
        env.close()

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "status": "measured_on_local_environment",
        "model_sha256": sha256_file(args.model),
        "model_path": str(args.model),
        "hf_compatible_loading": bool(args.hf_compatible),
        "environment": ENV_ID,
        "max_steps": args.max_steps,
        "sigmas_m": args.sigmas,
        "failure_buckets_warning": "探索性启发式阈值；不是经过验证的物理标签",
        "dependency_versions": dependency_versions(),
        "observation_noise": "分别对夹爪与物块 XYZ 位置观测加入独立高斯噪声，并重新计算相对 XYZ 与 achieved_goal",
        "episodes_per_condition": args.episodes,
        "seed_start": args.seed_start,
        "alpha": args.alpha,
        "summary": summarize(records),
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"已写入 {args.output / 'episodes.csv'} 和 {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
