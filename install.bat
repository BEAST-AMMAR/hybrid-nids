@echo off
echo =========================================
echo Installing Hybrid NIDS Dependencies
echo =========================================

echo.
echo [1/3] Creating Python Virtual Environment...
python -m venv .venv

echo.
echo [2/3] Installing Backend Dependencies...
call .venv\Scripts\activate
pip install -r src\backend\requirements.txt
pip install websockets "uvicorn[standard]"

echo.
echo [3/3] Installing Frontend Dependencies...
cd src\frontend
call npm install
cd ..\..

echo.
echo =========================================
echo Setup Complete!
echo You can now run the project using run.bat
echo =========================================
pause
