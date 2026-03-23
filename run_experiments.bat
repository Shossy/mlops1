@echo off
REM Lab 1-style manual sweeps are superseded by Lab 3 HPO (src/optimize.py).
REM Use DVC train for a single run from params.yaml, or run HPO manually.

echo.
echo === Option A: DVC pipeline train stage (params.yaml) ===
echo    dvc repro train
echo.
echo === Option B: Lab 3 Optuna + Hydra + MLflow (nested runs) ===
echo    py src\optimize.py
echo    py src\optimize.py hpo=random
echo    py src\optimize.py hpo.n_trials=30
echo    dvc repro optimize
echo.
pause
