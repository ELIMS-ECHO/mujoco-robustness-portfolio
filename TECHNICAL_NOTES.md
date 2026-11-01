# 机械臂抓放策略的观测噪声鲁棒性评估

> **作品集网页（已创建）：** [index.html](./index.html)。这是独立、响应式的案例展示页，包含问题、系统架构、个人贡献、进度状态和可导入真实 `summary.json` 的实验面板。下载该 HTML 文件后可以直接在浏览器离线打开。**它目前不等于已部署的公网网站，也不包含虚构的实验成绩。**

**Robustness Evaluation for FetchPickAndPlace-v4 (SAC + HER)**

> **状态：GitHub 云端真实 MuJoCo 已完成245回合（35预实验 + 独立210评估）、真实动作对比 GIF、自动化逻辑测试；Windows 本机运行与公开网站部署仍待验收。** 固定一个外部预训练模型，没有重新训练策略，也不声称跨模型泛化。

## 一句话说明

优先复用 [hhmm1122/fetch-pickandplace-sac-her](https://huggingface.co/hhmm1122/fetch-pickandplace-sac-her) 提供的 **SAC + HER 预训练机械臂抓放策略**，独立增加**位置观测噪声评估与 EMA 滤波对照**。研究不同测量误差条件下策略的表现，而不重新训练 100 万步。

### 个人贡献边界

| 来源 | 内容 | 当前状态 |
| --- | --- | --- |
| 外部模型作者 hhmm1122 | SAC+HER 训练和 Hugging Face 预训练权重 | 第三方外部基线；**不属于我的成果** |
| 本项目 | 位置噪声注入、EMA 滤波、配对种子评估、可追溯 CSV/JSON、统计区间、图表、响应式作品集网页 | 已提交；云端真实 MuJoCo 245 回合与 CI 通过 |
| 本项目后续 | 失败机制深入分析、跨模型复现、Windows 本机复现与公开独立网站 | 尚未完成 |

历史参考仓库 NickJohr 的 README 曾报告 **250 次确定性评估中 96.8% 成功率**，但 GitHub 云端运行器于 2026-11-01 尝试克隆时失败，故**不再将它作为运行依赖**。新基线来自 Hugging Face：作者自报 30 多回合确定性评估中成功率 100%，**这不是我的测量结果，不应与旧仓库的 96.8% 混用**。Hugging Face 模型卡未明确标示许可证，当前项目仅下载用于实验，不重新分发其权重。

## 核心成果：独立30种子验证（2026-11-01）

**当前作品集采用的主要结果**：[公开原始证据包](./evidence/README.md) — 7组条件 × 每组30个环境种子，**210回合真实 MuJoCo 仿真**。随机种子10000–10029，与下方35回合预实验完全独立。完整方法、统计与限制见 [30种子实验报告](./EVAL30_REPORT_2026-11-01.md)；可直接检查本仓库归档的 `episodes.csv.gz`、`summary.json`、`summary.csv` 和 `model_provenance.json`。

| 位置观测噪声 | 原策略 ever / final | 原策略 + EMA ever / final |
| --- | --- | --- |
| 0 mm | 30/30 (100%) / 30/30 (100%) | 不适用 |
| 5 mm | 30/30 (100%) / 30/30 (100%) | 27/30 (90.0%) / 25/30 (83.3%) |
| 10 mm | 30/30 (100%) / 28/30 (93.3%) | 29/30 (96.7%) / 23/30 (76.7%) |
| 20 mm | 24/30 (80.0%) / 10/30 (33.3%) | 26/30 (86.7%) / 23/30 (76.7%) |

**更值得讨论的是终态成功率的权衡**：在高噪声20 mm下，EMA 提高 **43.3个百分点**；在低噪声5和10 mm下，分别降低 **16.7个百分点**。因此不能把 EMA 解释成全面更优的方法。

对同一噪声条件的30对 seed，20mm 终态结果中：**仅 EMA 成功14对、仅未滤波成功1对、两者都成功9对、都失败6对**；探索性 McNemar 双侧精确 `p≈0.00098`，未修正多重比较，且只针对一个外部固定预训练模型和理想化高斯测量误差。可用 [analyze_paired.py](./analyze_paired.py) 从原始 CSV 独立复算。

**作品集网页**：[index.html](./index.html) 默认展示以上30种子真实实验数据，亦可导入后续 `summary.json`。**真实机器人的性能、跨模型泛化和本机可运行性都还没有验证。**

## 真实 MuJoCo GIF 演示（已通过云端验收）

[公开仓库的 demo workflow](./.github/workflows/demo.yml) 生成并归档左右对比的真实 MuJoCo 仿真 GIF：**同一随机种子10001、20mm位置噪声、50步/方法**，未滤波策略最终失败；EMA策略最终成功，第37步首次到达目标。

GIF、每条轨迹的 outcome JSON 和模型 provenance 已归档在 `assets/`，详见 [演示与渲染一致性报告](./DEMO_REPORT_2026-11-01.md)。

**重要：** 这是事后挑选的一个有反差的案例，不是随机代表性样本。首次录制发现直接在仿真环境 render 导致结果与批量评估不一致，已修复为独立渲染环境，并由 `--expect-contrast` 校验后重新录制。作品集正式性能数据仅使用独立210回合实验。

如需在本机重录，可在 `RUN_WINDOWS.ps1` 完成模型下载后执行：

```powershell
python .\record_demo.py --model .\upstream-model\best_model.zip --seed 10001 --noise-sigma 0.02 --expect-contrast --output .\results\demo-seed10001.gif
```

> 这不是实体机器人录像，也不是本人 Windows 本机验证。

## 第一批真实实验结果（2026-11-01，预实验）

35回合预实验曾由 GitHub Actions 实际完成；其原始 artifact 未纳入本公开证据包。汇总见 [预实验报告](./PILOT_REPORT_2026-11-01.md)。已完成 **7 个条件 × 各 5 个环境种子，共 35 个真实仿真回合**（seed 2026–2030）。所有方法均使用同一固定模型、同一初始种子列表、确定性预测、每回合最多 50 步。

| 观测噪声标准差 | 原策略 ever / final | EMA ever / final |
| --- | --- | --- |
| 0 mm | 5/5 · 5/5 | 不适用 |
| 5 mm | 5/5 · 5/5 | 5/5 · 4/5 |
| 10 mm | 5/5 · 5/5 | 4/5 · 2/5 |
| 20 mm | 2/5 · 0/5 | 4/5 · 4/5 |

- **ever**：整回合内曾达到目标；**final**：最后一步仍达到目标。两者不同，不能把瞬间达标当成稳定放置。
- 本次 20 mm 下 EMA 的 ever-success 为 80%，原策略为 40%；但在 10 mm 下 EMA 更差。**n=5 太小，不足以得出 EMA 改善鲁棒性的结论**。
- 例如 5/5 的成功率，其 Wilson 95% 区间仍约为 **[56.6%, 100%]**。这正说明不能过度解读本次比例。
- 模型：[`hhmm1122/fetch-pickandplace-sac-her`](https://huggingface.co/hhmm1122/fetch-pickandplace-sac-her)，revision `c817f4a3033fb4bde985bfcfe93ed5d8b949b04b`；权重 SHA-256：`1aad52960bc0b7baaa5249f3aa1a2de09009c300dd2b0bf61c9c51d7467b017b`。
- 已验证依赖：Python 3.11.17 / NumPy 2.3.5 / MuJoCo 3.3.7 / Gymnasium-Robotics 1.4.2 / SB3 2.8.0 / torch 2.15.0+cpu。
- 原始证据是该 GitHub Actions 运行的 artifact `mujoco-real-model-5-episode-smoke`，包括 `episodes.csv`、`summary.json`、`success_rate_comparison.png` 与 `model_provenance.json`。

**已完成独立30种子验证；仍待更多预训练策略、失败机制核查和视频验收。**

## 研究问题

1. 给定同一预训练策略，测量位置噪声是否降低抓放成功率？
2. 同样的环境种子下，低/中/高噪声的效果如何变化？
3. EMA 滤波是否改善性能？滤波延迟是否反而降低抓取精度？

> **注意：** 本项目加入的是状态向量的 **XYZ 位置测量误差**，并非摄像头图像噪声，不是 VLA 或视觉端到端控制，也没有改变 MuJoCo 物理状态。

## 原理和公平对照

```text
真实 FetchPickAndPlace-v4 环境
  ├─ clean：原始观测 ────────────────> 预训练 SAC 策略 ─> 动作
  ├─ noisy：位置观测 + 高斯噪声 ────> 同一 SAC 策略 ─> 动作
  └─ ema：位置观测 + 同分布噪声
                       └─ EMA 滤波 -> 同一 SAC 策略 ─> 动作

所有方法：相同模型、相同环境初始种子、确定性 predict
评估指标：ever_success / final_success / 首次成功步数 / 回报 / 观测轨迹诊断
```

- gripper XYZ、object XYZ 分别加入独立高斯噪声；object-gripper 相对位置重新计算。
- `achieved_goal` 同步使用扰动后的 object XYZ，防止目标条件策略从未扰动的坐标获得信息泄漏；`desired_goal` 不扰动。
- EMA 仅处理策略输入，不修改仿真物理状态；每回合清空历史，避免回合间泄漏。
- 使用相同的 seed 列表比较各方法；由于不同动作导致后续状态不同，**配对的是初始条件和噪声生成器**，不是强制对齐每一步的状态。
- `ever_success` 是一个回合内至少到达目标一次；`final_success` 是最后一步是否成功。**两者不可混为一谈**。
- 成功率给出 Wilson 二项比例区间，作为探索性统计；环境种子之间的独立性和跨随机训练种子的泛化尚未验证。

## 当前可核查进度（2026-11-01）

- **已提交**：完整评估循环、目标一致性检查、EMA 对照、受限评估步数、真实状态诊断指标与模型 SHA-256、依赖版本记录。
- **已运行（隔离的 Python 环境）**：8 项核心逻辑与 Mock 环境回归测试通过。Mock 成功标记是测试替身，**不是 MuJoCo 实验结果**。
- **首次真实云端任务失败**：旧 GitHub 基线克隆失败（无法读取认证用户名）。现已切换到 Hugging Face 上可浏览的 `best_model.zip`，增加不可变 revision 与 SHA-256 记录及兼容加载参数。
- **真实云端 Smoke Test 与独立验证均通过**：35回合预实验 + 210回合独立环境种子评估；模型 provenance、CSV/JSON、图表已由 GitHub Actions 留存。未验证 Windows 本机视频和跨模型泛化。
- GitHub Actions **逻辑与模拟环境回归测试已多次运行成功**，并实际跑通真实模型 Smoke Test 和30种子评估。

**一键入口**（在已激活的 Python 3.11 环境执行）：

```powershell
cd D:\CODEX\mujoco-robustness-portfolio
.\RUN_WINDOWS.ps1 -InstallDependencies -Episodes 5
# 5回合通过后，再跑正式对照
.\RUN_WINDOWS.ps1 -Episodes 100
```

脚本从 Hugging Face 下载已标记来源的模型权重，记录 revision 和 SHA-256，运行单元测试并进行真实评估；**不会启动训练或覆盖模型**。首次使用前请确认当前终端已激活正确的 conda 环境。执行脚本如果被 PowerShell 策略限制，可在当前进程执行 `Set-ExecutionPolicy -Scope Process Bypass`，结束该终端后不会改变系统级策略。

## Windows / PowerShell 快速运行

以下操作在你自己的 Windows 机器运行；代码提交不代表已经执行过。

```powershell
cd D:\CODEX

# 取得我已提交的实验实现（公开作品集仓库，无需私有仓库权限）
git clone https://github.com/ELIMS-ECHO/mujoco-robustness-portfolio.git

conda create -n mujoco-portfolio python=3.11 -y
conda activate mujoco-portfolio

cd .\mujoco-robustness-portfolio

# 先只做少量真实评估；自动下载并固定 Hugging Face 模型 revision
.\RUN_WINDOWS.ps1 -InstallDependencies -Episodes 5

# 5 回合通过后才进行扩展评估
.\RUN_WINDOWS.ps1 -Episodes 100
```

需要的外部依赖是 Gymnasium、Gymnasium-Robotics、MuJoCo、Stable-Baselines3、PyTorch、NumPy。**可能存在** Windows OpenGL、MuJoCo 或原项目模型依赖版本兼容问题，发生时必须记录完整报错，不应假装已解决。

## 实验设计

| 条件 | 输入噪声标准差 | 是否滤波 |
| --- | --- | --- |
| clean | 0 mm | 否 |
| noisy-5mm | 5 mm | 否 |
| ema-5mm | 5 mm | 是 |
| noisy-10mm | 10 mm | 否 |
| ema-10mm | 10 mm | 是 |
| noisy-20mm | 20 mm | 否 |
| ema-20mm | 20 mm | 是 |

初版固定 EMA 系数 `alpha=0.35`；**该数值不是最优参数，也没有声称可以提升性能**。如需调参，必须将调参种子与最终报告种子分离，避免验证集泄漏。

评估输出：

- `results/<run>/episodes.csv`：每回合 seed、策略、噪声、是否成功、首次成功步数、回报、最小目标距离、最小末端物块距离、最大物块抬升高度、**探索性失败分类**。分类阈值仅用于诊断，不能冒充已验证的物理抓取阶段标签。
- `results/<run>/summary.json`：按条件汇总成功率及 Wilson 区间、平均回报、探索性失败计数、模型 SHA-256、Python 与依赖版本、噪声和随机种子参数。

结果目录不应混入版本库。不要在数据生成前填写性能数字。

## 项目结构

```text
mujoco-robustness-portfolio/
├── README.md                  # 项目主页、来源声明、复现步骤
├── index.html                 # 响应式作品集网页（导入真实 JSON）
├── RUN_WINDOWS.ps1            # Windows 一键安装（可选）、单测、评估
├── download_checkpoint.py     # 下载 HF 模型、锁定 revision、记录 SHA-256
├── evaluate_robustness.py     # 位置扰动、EMA、评估与统计
├── analyze_paired.py          # 配对种子与探索性 McNemar 统计
├── plot_results.py            # 真实汇总 JSON -> 评估对比图
├── tests/test_noise.py        # 位置噪声和 Mock 环境集成测试
├── tests/test_paired.py       # 配对统计工具回归测试
├── PILOT_REPORT_2026-11-01.md # 35回合预实验
├── EVAL30_REPORT_2026-11-01.md # 210回合独立评估
├── DEMO_REPORT_2026-11-01.md # 真实对比 GIF 与一致性核查
├── RESUME_CASE_STUDY.md       # 有证据的简历与面试材料
├── record_demo.py             # 隔离渲染的动作录制代码
├── RESULTS_TEMPLATE.md        # 真实实验结果报告模板
└── .gitignore                 # 忽略原始实验输出和模型
```

## 下一步验收标准

- [ ] Windows 本机通过基础单元测试
- [x] 云端成功加载 Hugging Face `best_model.zip`（记录 revision 与 SHA-256）并完成 5回合/条件
- [ ] 用相同模型和环境对 clean 基线复核，记录与 Hugging Face 模型卡评估口径的差异
- [x] 在预先冻结的独立30个环境种子上完成对照，保存CSV/JSON和图表
- [x] 明确标出 EMA 各条件下有正/负效果并计算配对结果
- [ ] 解释滤波延迟与失败类型的实际机制，复核启发式标签
- [x] GitHub Linux云端已录制真实 noisy20 / EMA20 对照 GIF
- [ ] Windows 本机录制与人工交互 Demo（云端录像不能代替本机验收）
- [x] 用真实结果更新展示网页和30种子技术报告
- [ ] 完成求职简历与本人可播放演示视频、招聘方可直接访问的公开页面

## 资料来源与授权

- **实际运行的预训练权重**：[hhmm1122/fetch-pickandplace-sac-her](https://huggingface.co/hhmm1122/fetch-pickandplace-sac-her)。模型卡给出 SAC+HER、Python 加载示例和框架版本，但未明确展示授权许可证；本仓库仅记录来源并在执行环境中临时获取模型，**不包含该模型权重**。
- **历史参考而非运行依赖**：[NickJohr/fetch-pick-and-place-rl](https://github.com/NickJohr/fetch-pick-and-place-rl)，此前曾通过网页索引发现，但云端克隆失败，已改用 Hugging Face 来源。
- 仿真环境：[Gymnasium-Robotics FetchPickAndPlace](https://robotics.farama.org/envs/fetch/pick_and_place/)。
- 本目录新增评估代码为独立编写。除了记录来源外，不将任何第三方模型权重/训练代码声称为自己的成果；若模型卡或其他权利人提出使用限制，必须遵守。
