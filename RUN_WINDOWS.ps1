# Run real, bounded MuJoCo evaluation on Windows.
# First activate Python 3.11 conda environment:
#   conda create -n mujoco-portfolio python=3.11 -y
#   conda activate mujoco-portfolio
# Then:
#   .\RUN_WINDOWS.ps1 -InstallDependencies -Episodes 5
#   .\RUN_WINDOWS.ps1 -Episodes 100
#
# Checkpoint is credited to hhmm1122/fetch-pickandplace-sac-her (Hugging Face).
# It is never copied into the Git repository. No training is started.
param(
    [ValidateRange(1, 100000)][int]$Episodes = 5,
    [switch]$InstallDependencies,
    [ValidateRange(0, 2147483647)][int]$SeedStart = 2026
)

$ErrorActionPreference = "Stop"
$originalDir = Get-Location

function Assert-ExitCode([string]$Task) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Task failed (exit code $LASTEXITCODE). Inspect command output above."
    }
}

try {
    Set-Location $PSScriptRoot
    Write-Host "1/4 - Checking Python"
    python --version
    Assert-ExitCode "Python check"

    if ($InstallDependencies) {
        Write-Host "2/4 - Installing compatible packages (CPU)"
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        Assert-ExitCode "PyTorch installation"
        python -m pip install "numpy==2.3.5" "mujoco==3.3.7" "stable-baselines3==2.8.0" "gymnasium-robotics==1.4.2" "huggingface_hub>=0.29,<2"
        Assert-ExitCode "Dependency installation"
    } else {
        Write-Host "2/4 - Reusing existing environment"
    }

    Write-Host "3/4 - Running unit tests and resolving checkpoint revision"
    python -m unittest discover -s tests -v
    Assert-ExitCode "Unit tests"
    python .\download_checkpoint.py --output-dir .\upstream-model
    Assert-ExitCode "Checkpoint download"

    $model = Join-Path $PSScriptRoot "upstream-model\best_model.zip"
    if (-not (Test-Path -LiteralPath $model -PathType Leaf)) {
        throw "Missing checkpoint after download: $model"
    }

    Write-Host "4/4 - Running actual MuJoCo policy evaluation (no training)"
    $out = Join-Path $PSScriptRoot ("results\eval_" + $Episodes + "_seed_" + $SeedStart)
    python .\evaluate_robustness.py --model $model --hf-compatible --episodes $Episodes --seed-start $SeedStart --output $out
    Assert-ExitCode "MuJoCo evaluation"

    Write-Host "Evaluation completed: $out"
    Write-Host "Open index.html and import summary.json from this folder"
} finally {
    Set-Location $originalDir
}
