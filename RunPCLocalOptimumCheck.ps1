param(
    [int[]] $Seeds = @(2025, 3456, 5678),
    [string] $Python = "C:\Users\jyr03\anaconda3\envs\pc_env\python.exe"
)

$ErrorActionPreference = "Stop"

$experiments = @(
    @{ NumParts = 50; Gamma = 0.5;  MaxEpochs = 50  },
    @{ NumParts = 50; Gamma = 0.75; MaxEpochs = 100 },
    @{ NumParts = 40; Gamma = 0.5;  MaxEpochs = 50  },
    @{ NumParts = 30; Gamma = 0.5;  MaxEpochs = 50  }
)

function Format-GammaLabel {
    param([double] $Gamma)
    return ("{0:0.##}" -f $Gamma).Replace(".", "p")
}

foreach ($seed in $Seeds) {
    foreach ($experiment in $experiments) {
        $n = $experiment.NumParts
        $gamma = $experiment.Gamma
        $epochs = $experiment.MaxEpochs
        $gammaLabel = Format-GammaLabel -Gamma $gamma
        $runName = "reinforce_edge_n${n}_gamma${gammaLabel}_trainseed${seed}"

        Write-Host ""
        Write-Host ("=" * 100)
        Write-Host "Independent training: n=$n, gamma=$gamma, seed=$seed, epochs=$epochs"
        Write-Host ("=" * 100)

        & $Python run.py `
            experiment=pc/am_pc_edge `
            seed=$seed `
            env.generator_params.num_parts=$n `
            env.modularity_gamma=$gamma `
            trainer.max_epochs=$epochs `
            trainer.check_val_every_n_epoch=1 `
            logger.tensorboard.name=$runName

        if ($LASTEXITCODE -ne 0) {
            throw "Training failed: n=$n, gamma=$gamma, seed=$seed"
        }
    }
}

Write-Host ""
Write-Host "All independent local-optimum check runs completed."
