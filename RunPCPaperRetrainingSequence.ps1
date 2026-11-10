param(
    [string] $Python = "C:\Users\jyr03\anaconda3\envs\pc_env\python.exe"
)

$ErrorActionPreference = "Stop"

$experiments = @(
    @{
        Label = "1/5 N=50 gamma=0.5 train_seed=2345 epochs=50"
        NumParts = 50; Gamma = 0.5; Seed = 2345; MaxEpochs = 50; LearningRate = 1e-4
        RunName = "reinforce_edge_n50_gamma0p5_trainseed2345"
    },
    @{
        Label = "2/5 N=40 gamma=0.5 train_seed=2345 epochs=50"
        NumParts = 40; Gamma = 0.5; Seed = 2345; MaxEpochs = 50; LearningRate = 1e-4
        RunName = "reinforce_edge_n40_gamma0p5_trainseed2345"
    },
    @{
        Label = "3/5 N=40 gamma=0.75 lr=1e-5 epochs=100"
        NumParts = 40; Gamma = 0.75; Seed = 1234; MaxEpochs = 100; LearningRate = 1e-5
        RunName = "reinforce_edge_n40_gamma0p75_lr1e-5_seed1234"
    },
    @{
        Label = "4/5 N=30 gamma=0.5 train_seed=2345 epochs=50"
        NumParts = 30; Gamma = 0.5; Seed = 2345; MaxEpochs = 50; LearningRate = 1e-4
        RunName = "reinforce_edge_n30_gamma0p5_trainseed2345"
    },
    @{
        Label = "5/5 N=30 gamma=1.0 lr=1e-5 epochs=100"
        NumParts = 30; Gamma = 1.0; Seed = 1234; MaxEpochs = 100; LearningRate = 1e-5
        RunName = "reinforce_edge_n30_gamma1p0_lr1e-5_seed1234"
    }
)

foreach ($experiment in $experiments) {
    Write-Host ""
    Write-Host ("=" * 100)
    Write-Host $experiment.Label
    Write-Host ("=" * 100)

    & $Python run.py `
        experiment=pc/am_pc_edge `
        seed=$($experiment.Seed) `
        env.generator_params.num_parts=$($experiment.NumParts) `
        env.modularity_gamma=$($experiment.Gamma) `
        model.optimizer_kwargs.lr=$($experiment.LearningRate) `
        trainer.max_epochs=$($experiment.MaxEpochs) `
        trainer.check_val_every_n_epoch=1 `
        logger.tensorboard.name=$($experiment.RunName)

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $($experiment.Label)"
    }
}

Write-Host ""
Write-Host "All five paper retraining runs completed."
