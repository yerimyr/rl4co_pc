param(
    [int[]] $NumParts = @(20, 10, 50),
    [switch] $GenerateData,
    [switch] $OverwriteData,
    [string] $Python = "python"
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

function Test-PCDataset {
    param([int] $N)

    $val = "data\pc\pc${N}_newdist_val_seed4321.npz"
    $test = "data\pc\pc${N}_newdist_test_seed1234.npz"
    if (!(Test-Path $val) -or !(Test-Path $test)) {
        throw "Missing dataset for n=$N. Expected: $val and $test. Run with -GenerateData or run GeneratePCDatasets.py first."
    }
}

Write-Host "PC NCO training scenario runner"
Write-Host "NumParts: $($NumParts -join ', ')"
Write-Host "Python: $Python"

if ($GenerateData) {
    $dataArgs = @("GeneratePCDatasets.py", "--num-parts") + ($NumParts | ForEach-Object { "$_" })
    if ($OverwriteData) {
        $dataArgs += "--overwrite"
    }
    Invoke-CheckedCommand `
        -Label "Generate fixed PC validation/test datasets" `
        -Command $Python `
        -Arguments $dataArgs
}

foreach ($n in $NumParts) {
    Test-PCDataset -N $n
}

foreach ($n in $NumParts) {
    Invoke-CheckedCommand `
        -Label "Train current edge encoder, n=$n" `
        -Command $Python `
        -Arguments @(
            "run.py",
            "experiment=pc/am_pc_edge",
            "env.generator_params.num_parts=$n",
            "logger.tensorboard.name=reinforce_edge_n$n"
        )

    Invoke-CheckedCommand `
        -Label "Train MatNet encoder, n=$n" `
        -Command $Python `
        -Arguments @(
            "run.py",
            "experiment=pc/am_pc_matnet",
            "env.generator_params.num_parts=$n",
            "logger.tensorboard.name=reinforce_matnet_n$n"
        )
}

Write-Host ""
Write-Host "All requested PC training scenarios finished."
