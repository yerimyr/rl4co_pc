param(
    [int] $Seed = 1234,
    [int] $TestSize = 30,
    [int] $Limit = 30,
    [int] $Repeats = 1,
    [string] $Device = "cpu",
    [string] $Python = "python",
    [string] $RunRoot = "logs\train\runs",
    [int] $GaPopSize = 100,
    [int] $GaGenerations = 3000,
    [int] $SaIterations = 5000,
    [string] $Algorithms = "cpccd,sa,nco-custom",
    [string] $OutputSuffix = "",
    [bool] $PlotHistory = $true,
    [string] $N50CurrentCkpt = ""
)

$ErrorActionPreference = "Stop"

function Invoke-CheckedCommand {
    param(
        [string] $Label,
        [string] $Command,
        [string[]] $Arguments
    )

    Write-Host ""
    Write-Host ("=" * 100)
    Write-Host $Label
    Write-Host ("=" * 100)
    Write-Host "$Command $($Arguments -join ' ')"
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Label"
    }
}

function Get-LatestCheckpoint {
    param(
        [string] $Pattern,
        [string] $ExplicitPath
    )

    if ($ExplicitPath -and $ExplicitPath.Trim().Length -gt 0) {
        if (!(Test-Path $ExplicitPath)) {
            throw "Explicit checkpoint does not exist: $ExplicitPath"
        }
        return (Resolve-Path $ExplicitPath).Path
    }

    $matches = Get-ChildItem -Path $RunRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            if ($_.Name -like $Pattern) {
                return $true
            }

            $tensorboardDir = Join-Path $_.FullName "tensorboard"
            if (Test-Path $tensorboardDir) {
                $tbMatch = Get-ChildItem -Path $tensorboardDir -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -like $Pattern } |
                    Select-Object -First 1
                if ($tbMatch) {
                    return $true
                }
            }

            return $false
        } |
        Sort-Object LastWriteTime -Descending

    foreach ($run in $matches) {
        $checkpointDir = Join-Path $run.FullName "checkpoints"
        $lastCkpt = Join-Path $checkpointDir "last.ckpt"
        if (Test-Path $lastCkpt) {
            return (Resolve-Path $lastCkpt).Path
        }

        $latestEpochCkpt = Get-ChildItem -Path $checkpointDir -Filter "*.ckpt" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($latestEpochCkpt) {
            return $latestEpochCkpt.FullName
        }
    }

    throw "Could not find checkpoint for pattern '$Pattern' under '$RunRoot'. Pass -N50CurrentCkpt explicitly."
}

function Test-DataFile {
    param([string] $Path)
    if (!(Test-Path $Path)) {
        throw "Missing dataset: $Path"
    }
}

function Invoke-Performance {
    param(
        [string] $Label,
        [int] $NumParts,
        [int] $CheckpointNumParts,
        [string] $DataPath,
        [string] $OutputDir,
        [string] $CurrentCkpt
    )

    Test-DataFile -Path $DataPath

    $args = @(
        "Evaluation\performance.py",
        "--num-parts", "$NumParts",
        "--checkpoint-num-parts", "$CheckpointNumParts",
        "--seed", "$Seed",
        "--test-size", "$TestSize",
        "--limit", "$Limit",
        "--repeats", "$Repeats",
        "--device", "$Device",
        "--data", $DataPath,
        "--output-dir", $OutputDir,
        "--algorithms", $Algorithms,
        "--ga-pop-size", "$GaPopSize",
        "--ga-generations", "$GaGenerations",
        "--sa-iterations", "$SaIterations",
        "--nco-current-ckpt", $CurrentCkpt
    )

    if ($PlotHistory) {
        $args += "--plot-history"
    }

    Invoke-CheckedCommand `
        -Label $Label `
        -Command $Python `
        -Arguments $args
}

Write-Host "PC n=50 custom-only evaluation scenario runner"
Write-Host "Seed: $Seed"
Write-Host "TestSize: $TestSize"
Write-Host "Limit: $Limit"
Write-Host "Repeats: $Repeats"
Write-Host "Device: $Device"
Write-Host "Algorithms: $Algorithms"
Write-Host "OutputSuffix: $OutputSuffix"
Write-Host "PlotHistory: $PlotHistory"

$ckpt50Current = Get-LatestCheckpoint -Pattern "*reinforce_edge_n50*" -ExplicitPath $N50CurrentCkpt

Write-Host ""
Write-Host "Resolved checkpoint:"
Write-Host "n=50 current: $ckpt50Current"

Invoke-Performance `
    -Label "Exp 1: train n=50, test n=50, same distribution" `
    -NumParts 50 `
    -CheckpointNumParts 50 `
    -DataPath "data\pc\pc50_newdist_test_seed${Seed}.npz" `
    -OutputDir "outputs\evaluation\performance\n50_custom_only_seed${Seed}${OutputSuffix}" `
    -CurrentCkpt $ckpt50Current

Invoke-Performance `
    -Label "Exp 2: train n=50, test n=50, shifted distribution" `
    -NumParts 50 `
    -CheckpointNumParts 50 `
    -DataPath "data\pc\pc50_shifted_test_seed${Seed}.npz" `
    -OutputDir "outputs\evaluation\performance\train_n50_test_n50_shifted_custom_only_seed${Seed}${OutputSuffix}" `
    -CurrentCkpt $ckpt50Current

Invoke-Performance `
    -Label "Exp 3: train n=50, test n=70, same distribution" `
    -NumParts 70 `
    -CheckpointNumParts 50 `
    -DataPath "data\pc\pc70_newdist_test_seed${Seed}.npz" `
    -OutputDir "outputs\evaluation\performance\train_n50_test_n70_custom_only_seed${Seed}${OutputSuffix}" `
    -CurrentCkpt $ckpt50Current

Invoke-Performance `
    -Label "Exp 4: train n=50, test n=70, shifted distribution" `
    -NumParts 70 `
    -CheckpointNumParts 50 `
    -DataPath "data\pc\pc70_shifted_test_seed${Seed}.npz" `
    -OutputDir "outputs\evaluation\performance\train_n50_test_n70_shifted_custom_only_seed${Seed}${OutputSuffix}" `
    -CurrentCkpt $ckpt50Current

Write-Host ""
Write-Host "All requested n=50 custom-only evaluation scenarios finished."
