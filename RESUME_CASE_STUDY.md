# 求职作品集项目说明（基于已验证证据）

## 项目名称

**基于 SAC+HER 的机械臂抓放策略观测噪声鲁棒性分析**  
*Robustness Evaluation of SAC+HER Robotic Manipulation under Noisy Observations*

**应用环境：** MuJoCo / FetchPickAndPlace-v4  
**工具：** Python、Gymnasium-Robotics、Stable-Baselines3、NumPy、GitHub Actions、Matplotlib  
**项目类型：** 基于第三方预训练模型的工程实验与评估扩展，**不是从零训练出 SAC+HER 策略**。

## 简历建议描述（真实验证过的部分）

- 基于公开 SAC+HER 预训练模型，独立实现机械臂抓放策略的**位置观测噪声注入、EMA 滤波对照和批量评估流程**，同步修正 `achieved_goal` 与相对坐标，防止未受扰动状态泄漏。
- 使用 GitHub Actions 构建可复现的 Linux/MuJoCo 实验环境，完成**35回合预实验及210回合独立验证**，固定随机种子、依赖版本和第三方 checkpoint SHA-256，自动输出逐回合 CSV、结果 JSON 和比较曲线。
- 在30个独立测试种子下，分析低、中、高位置观测误差对策略成功率的影响：20 mm 高噪声时，**最终成功率从未滤波的 33.3%（10/30）提高到 EMA 的 76.7%（23/30）**；同时明确报告5 mm和10 mm条件下 EMA 导致的性能下降，拒绝将局部改善包装成普适优势。
- 对 same-seed 成对终态结果执行探索性精确 McNemar 检验，分离“曾经到达目标”和“最终保持成功”两类评价口径，并通过回归测试防止无效参数和状态不一致。

## 面试中能讲清楚的核心技术点

1. **为什么不只报告成功率？** 曾经到达目标与最终保持成功是不同指标。20mm原始策略 ever=24/30，final=10/30；忽略这个差异容易高估稳定性。
2. **为何要处理 `achieved_goal`？** SAC + HER 的 Dict 观测包括 gripper/object 位置及 `achieved_goal`，只噪声化状态数组却保留干净 `achieved_goal` 会泄漏物体位置。
3. **为什么 EMA 在20mm有效，却在较小扰动下可能失效？** 它可能降低随机误差，但引入时间滞后；目前只是推测，尚未通过轨迹分解证实。
4. **为什么需要配对种子和模型哈希？** 减少不同初始条件和 checkpoint 混淆造成的比较误差，让测试可复现。
5. **为什么不宣称自己实现 SAC+HER？** 本项目的增量是控制输入质量的实验设计、诊断、评估和工程复现，不是原始策略学习方法。

## 证据目录

- [项目主页](./README.md)
- [作品集展示网页（静态 HTML）](./index.html)
- [30种子独立真实评估报告](./EVAL30_REPORT_2026-11-01.md)
- [35回合预实验报告](./PILOT_REPORT_2026-11-01.md)
- [GitHub Actions 210回合运行](https://github.com/ELIMS-ECHO/code-study/actions/runs/41253970185)
- 运行 artifact：`mujoco-30-seeds-per-condition-evidence`（包含原始 CSV、JSON、曲线及来源信息）
- 源码：`evaluate_robustness.py`、`analyze_paired.py`、`plot_results.py`、`record_demo.py`

## 尚未完成，不能写进简历的内容

- Windows 本机机械臂演示录制验收。
- 推广到另一预训练 checkpoint、真实机器人或视觉感知。
- 自己从零训练 SAC + HER，并建立跨训练随机种子的比较。
- 公开独立项目与可直接对外访问的 GitHub Pages 作品集网站（本仓库目前仍然私有）。
- 精确定位 EMA 的失败机制（当前失败类型是探索性启发式诊断）。

### 外部作者与许可

原始权重来自 [hhmm1122/fetch-pickandplace-sac-her](https://huggingface.co/hhmm1122/fetch-pickandplace-sac-her)，模型卡未明确给出许可。报告中保留作者信息，不上传其权重，不将第三方训练成果归为自己的个人贡献。
