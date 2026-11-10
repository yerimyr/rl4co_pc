param(
    [double[]] $Gammas = @(0.0, 0.25, 0.5, 0.75, 1.0),
    [int] $EpochsPerStage = 100,
    [int] $CheckValEveryNEpoch = 1,
    [double] $LearningRate = 0.0001,
    [int] $MaterialTypes = 2,
    [string] $Python = "C:\Users\jyr03\anaconda3\envs\pc_env\python.exe",
    [string] $OutputRoot = "logs\train\curriculum",
    [string] $RunRoot = "",
    [switch] $ArchiveExistingGamma
)

if ($MaterialTypes -lt 1) {
    throw "MaterialTypes must be at least 1."
}

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Format-GammaLabel {
    param([double] $Gamma)
    return ("{0:0.##}" -f $Gamma).Replace(".", "p")
}

function Invoke-TrainingStage {
    param(
        [string] $Label,
        [string] $StageDir,
        [string[]] $Arguments
    )

    if (Test-Path -LiteralPath $StageDir) {
        throw "Output directory already exists: $StageDir. Choose another -OutputRoot or move the old run."
    }

    Write-Host ""
    Write-Host ("=" * 100)
    Write-Host $Label
    Write-Host ("=" * 100)
    Write-Host "$Python $($Arguments -join ' ')"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE`: $Label"
    }
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

$materialSuffix = if ($MaterialTypes -eq 2) { "" } else { "_material$MaterialTypes" }

foreach ($requiredN in @(20, 30, 40)) {
    foreach ($split in @("val_seed4321", "test_seed1234")) {
        $dataPath = "data\pc\pc${requiredN}_newdist${materialSuffix}_${split}.npz"
        if (!(Test-Path -LiteralPath $dataPath)) {
            throw "Missing fixed dataset: $dataPath"
        }
    }
}

$runStamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $activeRunRoot = Join-Path $OutputRoot $runStamp
} else {
    $activeRunRoot = $RunRoot
}

Write-Host "PC curriculum training"
Write-Host "Gammas: $($Gammas -join ', ')"
Write-Host "Epochs per stage: $EpochsPerStage"
Write-Host "Learning rate: $LearningRate"
Write-Host "Maximum material types: $MaterialTypes"
Write-Host "Run root: $activeRunRoot"

foreach ($gamma in $Gammas) {
    $gammaLabel = Format-GammaLabel -Gamma $gamma
    $gammaDir = Join-Path $activeRunRoot "gamma_$gammaLabel"
    if (Test-Path -LiteralPath $gammaDir) {
        if (!$ArchiveExistingGamma) {
            throw "Gamma output already exists: $gammaDir. Use -ArchiveExistingGamma to preserve it under a backup name and retrain from scratch."
        }
        $archiveDir = Join-Path $activeRunRoot "gamma_${gammaLabel}_interrupted_${runStamp}"
        if (Test-Path -LiteralPath $archiveDir) {
            throw "Archive target already exists: $archiveDir"
        }
        Write-Host "Archive existing gamma run: $gammaDir -> $archiveDir"
        Move-Item -LiteralPath $gammaDir -Destination $archiveDir
    }

    # Stage 1: N=10..20, 100%
    $stage1Dir = Join-Path $gammaDir "stage1_n10-20"
    $common1 = @(
        "run.py",
        "experiment=pc/am_pc_edge",
        "env.modularity_gamma=$gamma",
        "env.generator_params.num_parts=10",
        "env.generator_params.max_num_parts=20",
        "env.generator_params.material_types=$MaterialTypes",
        "env.val_file=pc20_newdist${materialSuffix}_val_seed4321.npz",
        "env.test_file=pc20_newdist${materialSuffix}_test_seed1234.npz",
        "model.optimizer_kwargs.lr=$LearningRate",
        "trainer.max_epochs=$EpochsPerStage",
        "trainer.check_val_every_n_epoch=$CheckValEveryNEpoch",
        "logger.tensorboard.name=curriculum_gamma${gammaLabel}_stage1",
        "hydra.run.dir=$stage1Dir"
    )
    Invoke-TrainingStage -Label "gamma=$gamma, stage 1: N=10-20 (100%)" -StageDir $stage1Dir -Arguments $common1
    $stage1Best = Get-BestCheckpoint -StageDir $stage1Dir

    # Stage 2: old N=10..20 20%, new N=21..30 80%
    $stage2Dir = Join-Path $gammaDir "stage2_n10-30"
    $common2 = @(
        "run.py",
        "experiment=pc/am_pc_edge",
        "env.modularity_gamma=$gamma",
        "env.generator_params.num_parts=10",
        "env.generator_params.max_num_parts=30",
        "env.generator_params.size_ranges=[[10,20],[21,30]]",
        "env.generator_params.size_weights=[0.2,0.8]",
        "env.generator_params.material_types=$MaterialTypes",
        "env.val_file=pc30_newdist${materialSuffix}_val_seed4321.npz",
        "env.test_file=pc30_newdist${materialSuffix}_test_seed1234.npz",
        "model.optimizer_kwargs.lr=$LearningRate",
        "trainer.max_epochs=$EpochsPerStage",
        "trainer.check_val_every_n_epoch=$CheckValEveryNEpoch",
        "init_from_checkpoint=$stage1Best",
        "logger.tensorboard.name=curriculum_gamma${gammaLabel}_stage2",
        "hydra.run.dir=$stage2Dir"
    )
    Invoke-TrainingStage -Label "gamma=$gamma, stage 2: N=10-20 (20%) + N=21-30 (80%)" -StageDir $stage2Dir -Arguments $common2
    $stage2Best = Get-BestCheckpoint -StageDir $stage2Dir

    # Stage 3: old bands 10% each, new N=31..40 80%
    $stage3Dir = Join-Path $gammaDir "stage3_n10-40"
    $common3 = @(
        "run.py",
        "experiment=pc/am_pc_edge",
        "env.modularity_gamma=$gamma",
        "env.generator_params.num_parts=10",
        "env.generator_params.max_num_parts=40",
        "env.generator_params.size_ranges=[[10,20],[21,30],[31,40]]",
        "env.generator_params.size_weights=[0.1,0.1,0.8]",
        "env.generator_params.material_types=$MaterialTypes",
        "env.val_file=pc40_newdist${materialSuffix}_val_seed4321.npz",
        "env.test_file=pc40_newdist${materialSuffix}_test_seed1234.npz",
        "model.optimizer_kwargs.lr=$LearningRate",
        "trainer.max_epochs=$EpochsPerStage",
        "trainer.check_val_every_n_epoch=$CheckValEveryNEpoch",
        "init_from_checkpoint=$stage2Best",
        "logger.tensorboard.name=curriculum_gamma${gammaLabel}_stage3",
        "hydra.run.dir=$stage3Dir"
    )
    Invoke-TrainingStage -Label "gamma=$gamma, stage 3: N=10-20 (10%) + N=21-30 (10%) + N=31-40 (80%)" -StageDir $stage3Dir -Arguments $common3
    $stage3Best = Get-BestCheckpoint -StageDir $stage3Dir

    Set-Content -LiteralPath (Join-Path $gammaDir "final_best_checkpoint.txt") -Value $stage3Best
    Write-Host "Final best checkpoint for gamma=$gamma`: $stage3Best"
}

Write-Host ""
Write-Host "All curriculum runs finished: $activeRunRoot"
