@echo off
setlocal

cd /d C:\rl4co_pc

"C:\Users\jyr03\anaconda3\envs\pc_env\python.exe" Evaluation\evaluate_ga_scenario_sweep.py ^
  --num-parts "20,30,40" ^
  --gammas "0,0.5,1.0" ^
  --seed 1234 ^
  --limit 10 ^
  --pop-size 100 ^
  --generations 3000 ^
  --elite-size 0 ^
  --tournament-size 2 ^
  --crossover-rate 0.9 ^
  --mutation-rate 0.3 ^
  --strong-mutation-prob 0.5 ^
  --strong-mutation-steps 5 ^
  --init-new-group-bias 0.6 ^
  --init-diverse-fraction 0.7 ^
  --output-dir "outputs\evaluation\genetic_algorithm"

if errorlevel 1 (
  echo.
  echo GA scenario sweep failed with exit code %errorlevel%.
  exit /b %errorlevel%
)

echo.
echo GA scenario sweep completed successfully.
endlocal
