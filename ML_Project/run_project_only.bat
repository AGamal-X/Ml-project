@echo off
cd /d "C:\Users\youss\Documents\machine learning\ML_Project"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment was not found.
    echo Run build_once_then_run.bat first.
    pause
    exit /b 1
)

if not exist "outputs\trained_models\linear_regression_model.joblib" (
    echo Trained models were not found.
    echo Run build_once_then_run.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo Opening GUI with saved models...
python gui_app.py

pause
