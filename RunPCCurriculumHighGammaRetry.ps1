param(
    [double[]] $Gammas = @(0.75, 1.0),
    [int] $EpochsPerStage = 100,
    [int] $CheckValEveryNEpoch = 1,
    [double] $LearningRate = 0.00001,
    [int] $Seed = 1234,
    [string] $Python = "C:\Users\jyr03\anaconda3\envs\pc_env\python.exe",
    [string] $SourceRunRoot = "logs\train\curriculum\2026-10-10_20-42-23",
    [string] $OutputRoot = "logs\train\curriculum_retry_lr1e-5"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Format-GammaLabel {
    param([double] $Gamma)
    return ("{0:0.##}" -f $Gamma).Replace(".", "p")
}

function Get-BestCheckpoint {
    param([string] $StageDir)

    $checkpointDir = Join-Path $StageDir "checkpoints"
    $best = Get-ChildItem -LiteralPath $checkpointDir -Filter "*.ckpt" -File |
        Where-Object { $_.Name -ne "last.ckpt" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $best) {
        $lastPath = Join-Path $checkpointDir "last.ckpt"
        if (!(Test-Path -LiteralPath $lastPath)) {
            throw "No checkpoint found in $checkpointDir"
        }
        return (Resolve-Path -LiteralPath $lastPath).Path
    }
    return $best.FullName
}

function Invoke-TrainingStage {
    param(
        [string] $Label,
        [string] $StageDir,
        [string[]] $Arguments
    )

    if (Test-Path -LiteralPath $StageDir) {
        throw "Output directory already exists: $StageDir"
    }

    Write-Host ""
    Write-Host ("=" * 100)
    Write-Host $Label
    Write-Host ("=" * 100)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE`: $Label"
    }
}

if (!(Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}
if (!(Test-Path -LiteralPath $SourceRunRoot)) {
    throw "Source curriculum run not found: $SourceRunRoot"
}

foreach ($requiredN in @(30, 40)) {
    foreach ($split in @("val_seed4321", "test_seed1234")) {
        $dataPath = "data\pc\pc${requiredN}_newdist_${split}.npz"
        if (!(Test-Path -LiteralPath $dataPath)) {
            throw "Missing fixed dataset: $dataPath"
        }
    }
}

$runStamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$activeRunRoot = Join-Path $OutputRoot $runStamp
New-Item -ItemType Directory -Path $activeRunRoot -Force | Out-Null
Set-Content -LiteralPath (Join-Path $OutputRoot "latest_run.txt") -Value $activeRunRoot

Write-Host "PC high-gamma curriculum retry"
Write-Host "Gammas: $($Gammas -join ', ')"
Write-Host "Reuse: Stage 1 best checkpoints from $SourceRunRoot"
Write-Host "Retrain: Stage 2 and Stage 3"
Write-Host "Learning rate: $LearningRate"
Write-Host "Epochs per stage: $EpochsPerStage"
Write-Host "Seed: $Seed"
Write-Host "Output: $activeRunRoot"

foreach ($gamma in $Gammas) {
    $gammaLabel = Format-GammaLabel -Gamma $gamma
    $sourceStage1Dir = Join-Path $SourceRunRoot "gamma_$gammaLabel\stage1_n10-20"
    $sourceStage1Best = Get-BestCheckpoint -StageDir $sourceStage1Dir

    # Keep a copy of the reused Stage 1 checkpoint so the retry run is
    # self-contained for the existing curriculum evaluation scripts.
    $gammaDir = Join-Path $activeRunRoot "gamma_$gammaLabel"
    $stage1CheckpointDir = Join-Path $gammaDir "stage1_n10-20\checkpoints"
    New-Item -ItemType Directory -Path $stage1CheckpointDir -Force | Out-Null
    $stage1Best = Join-Path $stage1CheckpointDir (Split-Path $sourceStage1Best -Leaf)
    Copy-Item -LiteralPath $sourceStage1Best -Destination $stage1Best
    Set-Content -LiteralPath (Join-Path $gammaDir "reused_stage1_checkpoint.txt") -Value $sourceStage1Best

    # Stage 2: keep the original 20%/80% replay ratio.  Only the LR changes.
    $stage2Dir = Join-Path $gammaDir "stage2_n10-30"
    $stage2Args = @(
        "run.py",
        "experiment=pc/am_pc_edge",
        "seed=$Seed",
        "env.modularity_gamma=$gamma",
        "env.generator_params.num_parts=10",
        "env.generator_params.max_num_parts=30",
        "env.generator_params.size_ranges=[[10,20],[21,30]]",
        "env.generator_params.size_weights=[0.2,0.8]",
        "env.val_file=pc30_newdist_val_seed4321.npz",
        "env.test_file=pc30_newdist_test_seed1234.npz",
        "model.optimizer_kwargs.lr=$LearningRate",
        "trainer.max_epochs=$EpochsPerStage",
        "trainer.check_val_every_n_epoch=$CheckValEveryNEpoch",
        "init_from_checkpoint=$stage1Best",
        "logger.tensorboard.name=curriculum_retry_lr1e-5_gamma${gammaLabel}_stage2",
        "hydra.run.dir=$stage2Dir"
    )
    Invoke-TrainingStage -Label "gamma=$gamma, Stage 2 retry, lr=$LearningRate" -StageDir $stage2Dir -Arguments $stage2Args
    $stage2Best = Get-BestCheckpoint -StageDir $stage2Dir

    # Stage 3: keep the original 10%/10%/80% replay ratio.
    $stage3Dir = Join-Path $gammaDir "stage3_n10-40"
    $stage3Args = @(
        "run.py",
        "experiment=pc/am_pc_edge",
        "seed=$Seed",
        "env.modularity_gamma=$gamma",
        "env.generator_params.num_parts=10",
        "env.generator_params.max_num_parts=40",
        "env.generator_params.size_ranges=[[10,20],[21,30],[31,40]]",
        "env.generator_params.size_weights=[0.1,0.1,0.8]",
        "env.val_file=pc40_newdist_val_seed4321.npz",
        "env.test_file=pc40_newdist_test_seed1234.npz",
        "model.optimizer_kwargs.lr=$LearningRate",
        "trainer.max_epochs=$EpochsPerStage",
        "trainer.check_val_every_n_epoch=$CheckValEveryNEpoch",
        "init_from_checkpoint=$stage2Best",
        "logger.tensorboard.name=curriculum_retry_lr1e-5_gamma${gammaLabel}_stage3",
        "hydra.run.dir=$stage3Dir"
    )
    Invoke-TrainingStage -Label "gamma=$gamma, Stage 3 retry, lr=$LearningRate" -StageDir $stage3Dir -Arguments $stage3Args
    $stage3Best = Get-BestCheckpoint -StageDir $stage3Dir

    Set-Content -LiteralPath (Join-Path $gammaDir "final_best_checkpoint.txt") -Value $stage3Best
    Write-Host "Final retry checkpoint for gamma=$gamma`: $stage3Best"
}

Write-Host ""
Write-Host "All high-gamma retries finished: $activeRunRoot"
