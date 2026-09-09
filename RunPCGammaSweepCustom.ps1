param(
    [int[]] $NumParts = @(30),
    [double[]] $Gammas = @(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
    [int] $MaxEpochs = 15,
    [int] $CheckValEveryNEpoch = 5,
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

function Format-GammaLabel {
    param([double] $Gamma)

    return ("{0:0.##}" -f $Gamma).Replace(".", "p")
}

Write-Host "PC nco-custom gamma sweep training runner"
Write-Host "NumParts: $($NumParts -join ', ')"
Write-Host "Gammas: $($Gammas -join ', ')"
Write-Host "MaxEpochs: $MaxEpochs"
Write-Host "CheckValEveryNEpoch: $CheckValEveryNEpoch"
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
    foreach ($gamma in $Gammas) {
        $gammaLabel = Format-GammaLabel -Gamma $gamma
        $runName = "reinforce_edge_n${n}_gamma${gammaLabel}"

        Invoke-CheckedCommand `
            -Label "Train nco-custom edge encoder, n=$n, gamma=$gamma" `
            -Command $Python `
            -Arguments @(
                "run.py",
                "experiment=pc/am_pc_edge",
                "env.generator_params.num_parts=$n",
                "env.modularity_gamma=$gamma",
                "trainer.max_epochs=$MaxEpochs",
                "trainer.check_val_every_n_epoch=$CheckValEveryNEpoch",
                "logger.tensorboard.name=$runName"
            )
    }
}

Write-Host ""
Write-Host "All requested PC nco-custom gamma sweep runs finished."
