# 求职作品集项目说明：机械臂抓放策略的位置观测噪声鲁棒性分析

## 项目名称

**基于 SAC + HER 的机械臂抓放策略位置观测噪声鲁棒性分析**

**应用环境：** MuJoCo / FetchPickAndPlace-v4  
**技术工具：** Python、Gymnasium-Robotics、Stable-Baselines3、NumPy、GitHub Actions、Matplotlib  
**项目类型：** 基于第三方预训练模型开展的工程实验与评估扩展，**不是从零训练 SAC+HER 策略**。

## 简历描述建议（已有证据支持）

- 基于公开 SAC+HER 预训练模型，独立实现机械臂抓放策略的**位置观测噪声注入、EMA 滤波对照和批量评估流程**；同步修正 achieved_goal 与相对坐标，避免未受扰动的状态信息泄漏。
- 使用 GitHub Actions 构建可复现的 Linux / MuJoCo 实验环境，完成**35 回合预实验与 210 回合独立验证**；固定随机种子、依赖版本和第三方模型权重 SHA-256，自动输出逐回合 CSV、结果 JSON 与对比曲线。
- 在 30 个独立测试随机种子下，分析低、中、高位置观测误差对策略成功率的影响：20 mm 高噪声时，**终态成功率从未滤波策略的 33.3%（10/30）提高到 EMA 策略的 76.7%（23/30）**；同时如实报告 5 mm 和 10 mm 条件下 EMA 带来的性能下降，不将局部改善包装成普遍优势。
- 对相同随机种子的成对终态结果执行探索性精确 McNemar 检验，区分“曾经到达目标”与“回合结束时仍然成功”两种评价口径，并通过回归测试检查无效参数和状态一致性。

## 面试中应能解释的关键技术问题

1. **为什么不只报告成功率？** 曾经到达目标与回合结束时仍然成功是不同指标。20 mm 条件下，未滤波策略曾达成目标 24/30，但终态成功只有 10/30；忽略这个差异会高估稳定性。
2. **为什么必须处理 achieved_goal？** SAC + HER 的 Dict 观测包含夹爪 / 物块位置以及 achieved_goal。只对状态数组加入噪声，却保留干净的 achieved_goal，会泄漏物块位置。
3. **为什么 EMA 在 20 mm 条件下有效，却可能在较小扰动下失效？** EMA 可能降低随机测量误差，但也会引入时间滞后。目前这只是待验证的解释，尚未通过轨迹分解证实。
4. **为什么需要配对随机种子和模型哈希？** 这可以减少初始条件和模型版本混淆造成的比较误差，并提高实验可复现性。
5. **为什么不能宣称自己实现了 SAC+HER？** 本项目的新增工作是控制输入质量相关的实验设计、诊断、评估和工程复现，不是原始策略学习算法。

## 证据目录

- [项目主页](./README.md)
- [作品集展示网页（静态 HTML）](./index.html)
- [30 种子独立评估报告](./EVAL30_REPORT_2026-11-01.md)
- [35 回合预实验报告](./PILOT_REPORT_2026-11-01.md)
- [210 回合原始数据与汇总证据](./evidence/README.md)
- [公开自动化测试记录](https://github.com/ELIMS-ECHO/mujoco-robustness-portfolio/actions/workflows/tests.yml)
- [公开仿真演示工作流记录](https://github.com/ELIMS-ECHO/mujoco-robustness-portfolio/actions/workflows/demo.yml)
- 源码：evaluate_robustness.py、analyze_paired.py、plot_results.py、record_demo.py。

## 尚未完成，不能写进简历的内容

- Windows 本机环境下的独立运行与人工演示验收。
- 推广到另一预训练模型、真实机器人或视觉感知任务。
- 从零训练 SAC + HER，并建立跨训练随机种子的比较。
- Windows 本机演示与复现尚待独立验收；GitHub Pages 公网作品集已部署：[在线作品集](https://elims-echo.github.io/mujoco-robustness-portfolio/)。
- 精确定位 EMA 的失败机制。目前失败类型仅为探索性启发式诊断。

## 上游作者与许可

原始权重来自 [hhmm1122/fetch-pickandplace-sac-her](https://huggingface.co/hhmm1122/fetch-pickandplace-sac-her)，模型卡未明确给出许可。本项目保留上游作者信息，不上传其权重，也不将第三方训练成果归为个人贡献。
