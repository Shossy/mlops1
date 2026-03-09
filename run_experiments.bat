@echo off
REM ============================================================
REM run_experiments.bat (Windows версія)
REM Запускає 5 MLflow експериментів для аналізу max_depth
REM
REM Використання:
REM   run_experiments.bat
REM ============================================================

echo ==============================================
echo   Запуск 5 MLflow експериментів
echo   Датасет: House Rent Prediction (India)
echo ==============================================

SET NROWS=2000000
SET AUTHOR=student
SET EXPERIMENT=House_Rent_Prediction

echo.
echo Експеримент 1/5 - max_depth=3 (Underfitting)
py src\train.py --n_estimators 100 --max_depth 3 --nrows %NROWS% --experiment_name "%EXPERIMENT%" --run_name "RF_depth3_underfitting" --author %AUTHOR%

echo.
echo Експеримент 2/5 - max_depth=6
py src\train.py --n_estimators 100 --max_depth 6 --nrows %NROWS% --experiment_name "%EXPERIMENT%" --run_name "RF_depth6" --author %AUTHOR%

echo.
echo Експеримент 3/5 - max_depth=10 (Baseline)
py src\train.py --n_estimators 100 --max_depth 10 --nrows %NROWS% --experiment_name "%EXPERIMENT%" --run_name "RF_depth10_baseline" --author %AUTHOR%

echo.
echo Експеримент 4/5 - max_depth=20
py src\train.py --n_estimators 100 --max_depth 20 --nrows %NROWS% --experiment_name "%EXPERIMENT%" --run_name "RF_depth20" --author %AUTHOR%

echo.
echo Експеримент 5/5 - max_depth=99 (майже необмежено)
py src\train.py --n_estimators 100 --max_depth 35 --nrows %NROWS% --experiment_name "%EXPERIMENT%" --run_name "RF_depthNone_overfitting" --author %AUTHOR%

echo.
echo ==============================================
echo   Всі 5 експериментів завершені!
echo   Запустіть MLflow UI: mlflow ui
echo   Відкрийте: http://127.0.0.1:5000
echo ==============================================
pause
