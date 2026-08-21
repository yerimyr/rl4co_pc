param(
    [int] $Seed = 1234,
    [int] $TestSize = 20,
    [int] $Limit = 20,
    [int] $Repeats = 1,
    [string] $Device = "cpu",
    [string] $Python = "python",
    [string] $RunRoot = "logs\train\runs",
    [int] $GaPopSize = 100,
    [int] $GaGenerations = 3000,
    [int] $SaIterations = 4000,
    [string] $N10CurrentCkpt = "",
    [string] $N10MatNetCkpt = "",
    [string] $N10NewCkpt = "",
    [string] $N20CurrentCkpt = "",
    [string] $N20MatNetCkpt = "",
    [string] $N20NewCkpt = "",
    [string] $N30CurrentCkpt = "",
    [string] $N30MatNetCkpt = "",
    [string] $N30NewCkpt = ""
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

    throw "Could not find checkpoint for pattern '$Pattern' under '$RunRoot'. Pass an explicit checkpoint path."
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
        [string] $CurrentCkpt,
        [string] $MatNetCkpt,
        [string] $NewCkpt
    )

    Test-DataFile -Path $DataPath

    Invoke-CheckedCommand `
        -Label $Label `
        -Command $Python `
        -Arguments @(
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
            "--ga-pop-size", "$GaPopSize",
            "--ga-generations", "$GaGenerations",
            "--sa-iterations", "$SaIterations",
            "--nco-current-ckpt", $CurrentCkpt,
            "--nco-matnet-ckpt", $MatNetCkpt,
            "--nco-new-ckpt", $NewCkpt
        )
}

Write-Host "PC evaluation scenario runner"
Write-Host "Seed: $Seed"
Write-Host "TestSize: $TestSize"
Write-Host "Limit: $Limit"
Write-Host "Repeats: $Repeats"
Write-Host "Device: $Device"

$ckpt10Current = Get-LatestCheckpoint -Pattern "*reinforce_edge_n10*" -ExplicitPath $N10CurrentCkpt
$ckpt10MatNet = Get-LatestCheckpoint -Pattern "*reinforce_matnet_n10*" -ExplicitPath $N10MatNetCkpt
$ckpt10New = Get-LatestCheckpoint -Pattern "*reinforce_part_matrix_n10*" -ExplicitPath $N10NewCkpt

$ckpt20Current = Get-LatestCheckpoint -Pattern "*reinforce_edge_n20*" -ExplicitPath $N20CurrentCkpt
$ckpt20MatNet = Get-LatestCheckpoint -Pattern "*reinforce_matnet_n20*" -ExplicitPath $N20MatNetCkpt
$ckpt20New = Get-LatestCheckpoint -Pattern "*reinforce_part_matrix_n20*" -ExplicitPath $N20NewCkpt

$ckpt30Current = Get-LatestCheckpoint -Pattern "*reinforce_edge_n30*" -ExplicitPath $N30CurrentCkpt
$ckpt30MatNet = Get-LatestCheckpoint -Pattern "*reinforce_matnet_n30*" -ExplicitPath $N30MatNetCkpt
$ckpt30New = Get-LatestCheckpoint -Pattern "*reinforce_part_matrix_n30*" -ExplicitPath $N30NewCkpt

Write-Host ""
Write-Host "Resolved checkpoints:"
Write-Host "n=10 current: $ckpt10Current"
Write-Host "n=10 matnet : $ckpt10MatNet"
Write-Host "n=10 new    : $ckpt10New"
Write-Host "n=20 current: $ckpt20Current"
Write-Host "n=20 matnet : $ckpt20MatNet"
Write-Host "n=20 new    : $ckpt20New"
Write-Host "n=30 current: $ckpt30Current"
Write-Host "n=30 matnet : $ckpt30MatNet"
Write-Host "n=30 new    : $ckpt30New"

Invoke-Performance `
    -Label "Exp 1: train n=10, test n=10, same distribution" `
    -NumParts 10 `
    -CheckpointNumParts 10 `
    -DataPath "data\pc\pc10_newdist_test_seed${Seed}.npz" `
    -OutputDir "outputs\evaluation\performance\n10_seed${Seed}" `
    -CurrentCkpt $ckpt10Current `
    -MatNetCkpt $ckpt10MatNet `
    -NewCkpt $ckpt10New

Invoke-Performance `
    -Label "Exp 2: train n=20, test n=20, same distribution" `
    -NumParts 20 `
    -CheckpointNumParts 20 `
    -DataPath "data\pc\pc20_newdist_test_seed${Seed}.npz" `
    -OutputDir "outputs\evaluation\performance\n20_seed${Seed}" `
    -CurrentCkpt $ckpt20Current `
    -MatNetCkpt $ckpt20MatNet `
    -NewCkpt $ckpt20New

Invoke-Performance `
    -Label "Exp 3: train n=30, test n=30, same distribution" `
    -NumParts 30 `
    -CheckpointNumParts 30 `
    -DataPath "data\pc\pc30_newdist_test_seed${Seed}.npz" `
    -OutputDir "outputs\evaluation\performance\n30_seed${Seed}" `
    -CurrentCkpt $ckpt30Current `
    -MatNetCkpt $ckpt30MatNet `
    -NewCkpt $ckpt30New

Invoke-Performance `
    -Label "Exp 4: train n=20, test n=20, shifted distribution" `
    -NumParts 20 `
    -CheckpointNumParts 20 `
    -DataPath "data\pc\pc20_shifted_test_seed${Seed}.npz" `
    -OutputDir "outputs\evaluation\performance\train_n20_test_n20_shifted_seed${Seed}" `
    -CurrentCkpt $ckpt20Current `
    -MatNetCkpt $ckpt20MatNet `
    -NewCkpt $ckpt20New

Invoke-Performance `
    -Label "Exp 5: train n=20, test n=30, same distribution" `
    -NumParts 30 `
    -CheckpointNumParts 20 `
    -DataPath "data\pc\pc30_newdist_test_seed${Seed}.npz" `
    -OutputDir "outputs\evaluation\performance\train_n20_test_n30_seed${Seed}" `
    -CurrentCkpt $ckpt20Current `
    -MatNetCkpt $ckpt20MatNet `
    -NewCkpt $ckpt20New

Invoke-Performance `
    -Label "Exp 6: train n=20, test n=30, shifted distribution" `
    -NumParts 30 `
    -CheckpointNumParts 20 `
    -DataPath "data\pc\pc30_shifted_test_seed${Seed}.npz" `
    -OutputDir "outputs\evaluation\performance\train_n20_test_n30_shifted_seed${Seed}" `
    -CurrentCkpt $ckpt20Current `
    -MatNetCkpt $ckpt20MatNet `
    -NewCkpt $ckpt20New

Write-Host ""
Write-Host "All requested PC evaluation scenarios finished."
