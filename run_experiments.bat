@echo off

SET NROWS=2000000
SET AUTHOR=pasha
SET EXPERIMENT=House_Rent_Prediction

echo.
echo Experiment 1/5 - max_depth=3 (Underfitting)
py src\train.py --n_estimators 100 --max_depth 3 --nrows %NROWS% --experiment_name "%EXPERIMENT%" --run_name "RF_depth3_underfitting" --author %AUTHOR%

echo.
echo Experiment 2/5 - max_depth=6
py src\train.py --n_estimators 100 --max_depth 6 --nrows %NROWS% --experiment_name "%EXPERIMENT%" --run_name "RF_depth6" --author %AUTHOR%

echo.
echo Experiment 3/5 - max_depth=10 (Baseline)
py src\train.py --n_estimators 100 --max_depth 10 --nrows %NROWS% --experiment_name "%EXPERIMENT%" --run_name "RF_depth10_baseline" --author %AUTHOR%

echo.
echo Experiment 4/5 - max_depth=20
py src\train.py --n_estimators 100 --max_depth 20 --nrows %NROWS% --experiment_name "%EXPERIMENT%" --run_name "RF_depth20" --author %AUTHOR%

echo.
echo Experiment 5/5 - max_depth=30
py src\train.py --n_estimators 100 --max_depth 30 --nrows %NROWS% --experiment_name "%EXPERIMENT%" --run_name "RF_depthNone_overfitting" --author %AUTHOR%

pause
