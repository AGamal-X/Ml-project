@echo off
cd /d "C:\Users\youss\Documents\machine learning\ML_Project"

echo Preparing environment...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing requirements...
python -m pip install -r requirements.txt

echo Training models and generating outputs...
python main.py

echo Opening GUI...
python gui_app.py

pause
