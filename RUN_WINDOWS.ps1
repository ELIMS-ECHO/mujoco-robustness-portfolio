# 在 Windows 上运行有回合数上限的 MuJoCo 真实仿真评估。
# 首次使用：
#   conda create -n mujoco-portfolio python=3.11 -y
#   conda activate mujoco-portfolio
#   .\RUN_WINDOWS.ps1 -InstallDependencies -Episodes 5
#
# 5 episodes 表示每个实验条件 5 个随机种子，共 7 x 5 = 35 回合。
# 冒烟测试通过后，如需本机复核正式评估规模：
#   .\RUN_WINDOWS.ps1 -Episodes 30 -SeedStart 10000
#
# 模型权重来自 Hugging Face 上的 hhmm1122/fetch-pickandplace-sac-her。
# 模型权重不会复制到 Git 仓库中，也不会启动模型训练。

param(
    [ValidateRange(1, 100000)][int]$Episodes = 5,
    [switch]$InstallDependencies,
    [ValidateRange(0, 2147483647)][int]$SeedStart = 2026
)

$ErrorActionPreference = "Stop"
$originalDir = Get-Location

function Assert-ExitCode([string]$Task) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Task 执行失败（退出代码 $LASTEXITCODE）。请检查上方命令输出。"
    }
}

try {
    Set-Location $PSScriptRoot

    Write-Host "1/5 - 检查 Python 版本"
    python --version
    Assert-ExitCode "Python 环境检查"

    $pythonVersion = (& python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    Assert-ExitCode "Python 版本读取"
    if ($pythonVersion -ne "3.11") {
        throw "当前 Python 为 $pythonVersion。请先执行 conda activate mujoco-portfolio，确保使用 Python 3.11。"
    }

    if ($InstallDependencies) {
        Write-Host "2/5 - 安装 CPU 版 PyTorch 与完整锁定依赖"
        python -m pip install --upgrade pip
        Assert-ExitCode "pip 更新"
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        Assert-ExitCode "CPU 版 PyTorch 安装"
        python -m pip install -r requirements.txt
        Assert-ExitCode "项目 requirements.txt 安装"
    } else {
        Write-Host "2/5 - 使用当前 Python 环境"
    }

    python -m pip check
    Assert-ExitCode "Python 依赖一致性检查"

    Write-Host "3/5 - 运行自动化测试并核验第三方模型版本"
    python -m unittest discover -s tests -v
    Assert-ExitCode "单元测试"
    python .\download_checkpoint.py --output-dir .\upstream-model
    Assert-ExitCode "模型权重下载与 SHA-256 核验"

    $model = Join-Path $PSScriptRoot "upstream-model\best_model.zip"
    if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
        throw "下载后仍未找到模型权重文件：$model"
    }

    $out = Join-Path $PSScriptRoot ("results\eval_" + $Episodes + "_seed_" + $SeedStart)

    Write-Host "4/5 - 运行真实 MuJoCo 策略评估（不进行训练）"
    python .\evaluate_robustness.py --model $model --hf-compatible --episodes $Episodes --seed-start $SeedStart --output $out
    Assert-ExitCode "MuJoCo 仿真评估"

    Write-Host "5/5 - 生成图表、配对统计并检查证据文件"
    python .\plot_results.py --summary (Join-Path $out "summary.json") --output (Join-Path $out "success_rate_comparison.png")
    Assert-ExitCode "实验图表生成"
    python .\analyze_paired.py --episodes (Join-Path $out "episodes.csv") --output (Join-Path $out "paired_analysis.json")
    Assert-ExitCode "配对统计分析"

    python -c "import csv,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); n=int(sys.argv[2]); rows=list(csv.DictReader((p/'episodes.csv').open(encoding='utf-8'))); summary=json.loads((p/'summary.json').read_text(encoding='utf-8')); assert len(rows)==7*n, f'预期 {7*n} 行，实际 {len(rows)} 行'; assert len(summary['summary'])==7; assert (p/'success_rate_comparison.png').is_file(); assert (p/'paired_analysis.json').is_file(); print(f'验收通过：{len(rows)} 个回合记录、7 组条件、图表与配对分析文件均已生成。') " $out $Episodes
    Assert-ExitCode "结果文件完整性检查"

    Write-Host ""
    Write-Host "本机仿真评估已完成，结果目录：$out"
    Write-Host "说明：该脚本验证 MuJoCo 物理环境和策略推理；不训练模型，也不要求打开图形 Viewer。"
} finally {
    Set-Location $originalDir
}
