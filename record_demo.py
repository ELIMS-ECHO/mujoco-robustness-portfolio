"""使用第三方预训练 SAC 策略录制 MuJoCo 仿真对照 GIF。

该文件只录制一个事先选定的演示案例，不能代表整体实验表现。
需要可用的 MuJoCo 离屏渲染器（MUJOCO_GL=osmesa 或 egl）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_robustness import ENV_ID, PositionEMA, corrupt_position_observation


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def one_run(model, env, render_env, seed: int, method: str, sigma: float, alpha: float,
            max_steps: int) -> tuple[list[np.ndarray], dict]:
    """只推进无画面的评估环境；将状态复制到独立的展示环境中渲染。"""
    import mujoco

    render_env.reset(seed=seed)
    def render_copy() -> np.ndarray:
        # 不要在被测环境中渲染或额外推进仿真：渲染过程可能影响
        # 数值轨迹或初始化状态，进而破坏不同回合之间的可比性。
        src, dst = env.unwrapped.data, render_env.unwrapped.data
        dst.qpos[:] = src.qpos
        dst.qvel[:] = src.qvel
        if len(src.act):
            dst.act[:] = src.act
        dst.mocap_pos[:] = src.mocap_pos
        dst.mocap_quat[:] = src.mocap_quat
        dst.ctrl[:] = src.ctrl
        dst.time = src.time
        mujoco.mj_forward(render_env.unwrapped.model, dst)
        frame = render_env.render()
        if frame is None:
            raise RuntimeError("MuJoCo render() 未生成 RGB 画面")
        return np.array(frame, copy=True)

    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed * 1009 + 17)
    ema = PositionEMA(alpha)
    frames: list[np.ndarray] = []
    first_success = None
    last_success = False
    for step in range(1, max_steps + 1):
        if method == "clean":
            policy_obs = {k: np.array(v, copy=True) for k, v in obs.items()}
        else:
            policy_obs = corrupt_position_observation(obs, sigma, rng)
            if method == "ema":
                policy_obs = ema.transform(policy_obs)
        action, _ = model.predict(policy_obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        frames.append(render_copy())
        last_success = bool(info.get("is_success", False))
        if last_success and first_success is None:
            first_success = step
        if terminated or truncated:
            break
    return frames, {
        "method": method, "seed": seed, "sigma_m": sigma, "steps": len(frames),
        "ever_success": first_success is not None, "final_success": last_success,
        "first_success_step": first_success,
    }


def make_gif(runs: list[tuple[list[np.ndarray], dict]], out_path: Path, fps: int) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    # 演示画面包含中文标签，必须使用具备中文字形的字体。
    font_candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    font_path = next((path for path in font_candidates if path.is_file()), None)
    if font_path is None:
        raise RuntimeError("未找到支持中文的字体。请安装 Noto Sans CJK，或在 Windows 上安装微软雅黑。")
    title_font = ImageFont.truetype(str(font_path), 18)
    status_font = ImageFont.truetype(str(font_path), 14)

    length = max(len(frames) for frames, _ in runs)
    result_frames = []
    for index in range(length):
        columns = []
        for frames, info in runs:
            image = Image.fromarray(frames[min(index, len(frames) - 1)]).convert("RGB")
            image.thumbnail((480, 380))
            canvas = Image.new("RGB", (500, 420), (8, 18, 26))
            x = (500 - image.width) // 2
            canvas.paste(image, (x, 40))
            draw = ImageDraw.Draw(canvas)
            name = "未滤波策略 20 mm" if info["method"] == "noisy" else "EMA 策略 20 mm"
            draw.text((14, 12), f"{name}｜随机种子={info['seed']}", fill=(210, 241, 230), font=title_font)
            label = "终态成功" if info["final_success"] else "终态失败"
            draw.text((14, 397), f"{label}（回合结束结果；非实时状态）", fill=(238, 189, 126), font=status_font)
            columns.append(np.asarray(canvas))
        result_frames.append(np.concatenate(columns, axis=1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), result_frames, format="GIF",
                    duration=1.0 / fps, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--noise-sigma", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("results/demo-20mm.gif"))
    parser.add_argument("--expect-contrast", action="store_true",
                        help="要求指定随机种子下未滤波策略终态失败、EMA 策略终态成功")
    args = parser.parse_args()
    if not args.model.is_file() or args.noise_sigma <= 0 or args.max_steps < 1 or args.fps < 1:
        parser.error("缺少模型文件或正数参数无效")
    import gymnasium as gym
    import gymnasium_robotics
    from stable_baselines3 import SAC
    from stable_baselines3.common.buffers import DictReplayBuffer

    gym.register_envs(gymnasium_robotics)
    # 该环境与正式评估基准相同，不启用渲染模式。
    env = gym.make(ENV_ID, max_episode_steps=args.max_steps)
    # 画面采集在独立的展示环境中进行，该环境不负责推进实验回合。
    render_env = gym.make(ENV_ID, render_mode="rgb_array", max_episode_steps=args.max_steps)
    try:
        model = SAC.load(str(args.model), env=env, device="cpu", custom_objects={
            "replay_buffer_class": DictReplayBuffer,
            "replay_buffer_kwargs": {},
            "buffer_size": 1,
        })
        cases = []
        for method in ("noisy", "ema"):
            frames, record = one_run(model, env, render_env, args.seed, method,
                                     args.noise_sigma, args.alpha, args.max_steps)
            cases.append((frames, record))
            print(json.dumps(record))
        if args.expect_contrast:
            no, ema_record = cases[0][1], cases[1][1]
            if no["final_success"] or not ema_record["final_success"]:
                raise RuntimeError(
                    "指定随机种子不符合预期的对照结果："
                    f"未滤波策略终态成功={no['final_success']}, EMA 策略终态成功={ema_record['final_success']}. "
                    "不要将该演示作为算法改进的证据发布。"
                )
        make_gif(cases, args.output, args.fps)
        manifest = {
            "label": "事先挑选的说明性演示案例，不代表总体成功率",
            "environment": ENV_ID,
            "model_sha256": digest_file(args.model),
            "seed": args.seed,
            "noise_sigma_m": args.noise_sigma,
            "alpha": args.alpha,
            "episodes": [record for _, record in cases],
            "gif": str(args.output),
        }
        manifest_path = args.output.with_suffix(".json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"已保存 {args.output} 和 {manifest_path}")
    finally:
        render_env.close()
        env.close()


if __name__ == "__main__":
    main()
