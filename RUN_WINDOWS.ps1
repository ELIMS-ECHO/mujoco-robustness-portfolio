# 在 Windows 上运行有回合数上限的 MuJoCo 真实仿真评估。
# 首先创建并激活 Python 3.11 conda 环境：
#   conda create -n mujoco-portfolio python=3.11 -y
#   conda activate mujoco-portfolio
# 然后执行：
#   .\RUN_WINDOWS.ps1 -InstallDependencies -Episodes 5
#   .\RUN_WINDOWS.ps1 -Episodes 100
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
    Write-Host "1/4 - 检查 Python 环境"
    python --version
    Assert-ExitCode "Python 环境检查"

    if ($InstallDependencies) {
        Write-Host "2/4 - 安装兼容依赖（CPU 版本）"
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        Assert-ExitCode "PyTorch 安装"
        python -m pip install "numpy==2.3.5" "mujoco==3.3.7" "stable-baselines3==2.8.0" "gymnasium-robotics==1.4.2" "huggingface_hub>=0.29,<2"
        Assert-ExitCode "项目依赖安装"
    } else {
        Write-Host "2/4 - 使用当前 Python 环境"
    }

    Write-Host "3/4 - 运行单元测试并核验模型版本"
    python -m unittest discover -s tests -v
    Assert-ExitCode "单元测试"
    python .\download_checkpoint.py --output-dir .\upstream-model
    Assert-ExitCode "模型权重下载"

    $model = Join-Path $PSScriptRoot "upstream-model\best_model.zip"
    if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
        throw "下载后仍未找到模型权重文件：$model"
    }

    Write-Host "4/4 - 运行真实 MuJoCo 策略评估（不进行训练）"
    $out = Join-Path $PSScriptRoot ("results\eval_" + $Episodes + "_seed_" + $SeedStart)
    python .\evaluate_robustness.py --model $model --hf-compatible --episodes $Episodes --seed-start $SeedStart --output $out
    Assert-ExitCode "MuJoCo 仿真评估"

    Write-Host "评估已完成，结果目录：$out"
    Write-Host "请打开 index.html，并导入此目录中的 summary.json"
} finally {
    Set-Location $originalDir
}
