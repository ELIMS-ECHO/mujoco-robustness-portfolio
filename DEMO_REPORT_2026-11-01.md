# MuJoCo 真实抓放对比演示 · 2026-11-01

## 演示目标和数据

![同一初始种子下 Noisy 与 EMA 的 MuJoCo 仿真对比](./assets/demo.gif)

在 20mm 观测噪声下，展示 **同一初始种子、同一 SAC 外部预训练模型** 的策略行为：

- 初始环境随机种子：`10001`。
- 未滤波 Noisy 策略：50步，**ever-success = False，final-success = False**。
- 位置 EMA 滤波策略：50步，**ever-success = True，final-success = True**（首次成功步数37）。
- 录制成果：50帧、1000×420 像素、左右同步对比 GIF。
- 模型 SHA-256：`1aad52960bc0b7baaa5249f3aa1a2de09009c300dd2b0bf61c9c51d7467b017b`。
- 真实 GIF 与逐案例 JSON 已在本公开仓库归档：[`assets/demo.gif`](./assets/demo.gif)、[`assets/demo.json`](./assets/demo.json)、[`assets/model_provenance.json`](./assets/model_provenance.json)。公开仓库的 [demo workflow](./.github/workflows/demo.yml) 可重新生成该演示。

## 一个必要的技术修复

第一次录制时直接在被测环境里调用 `env.render()` 后，这个 seed 的录制结果与批量评估结果不一致，因而没有将第一次视频当作证明。

修复方法：**单独创建无渲染的被测环境与仅用于渲染的展示环境**。在被测环境中运行动作，复制位置/速度、mocap 及时间等 MuJoCo 状态到展示环境，只在展示环境调用 `mj_forward()` 和 `render()`，以免绘图影响被测物理轨迹。

重新录制后，结果与30种子批量评估中的 seed10001 一致（Noisy最终失败，EMA最终成功）；额外增加 `--expect-contrast` 检查，不满足该案例的真实成功/失败条件时拒绝发布误导性演示。

## 解释限制

此例是**事后有意识挑选的反差案例**，并非随机抽样、平均轨迹或提升幅度。总体结论必须以 [210回合实验报告](./EVAL30_REPORT_2026-11-01.md) 和原始 CSV 为依据。

这段 GIF 只是 MuJoCo 仿真，而不是真实实体机械臂录像。模型权重由 [hhmm1122](https://huggingface.co/hhmm1122/fetch-pickandplace-sac-her) 提供；本项目新增的是实验、对照、观察和录制流程。
