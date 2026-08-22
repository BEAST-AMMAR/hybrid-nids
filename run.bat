@echo off
echo =========================================
echo Starting Hybrid NIDS
echo =========================================

echo.
echo Starting Backend (FastAPI)...
start "Hybrid NIDS Backend" cmd /k "call .venv\Scripts\activate && python -m src.backend.main"

echo.
echo Starting Frontend (React/Vite)...
start "Hybrid NIDS Frontend" cmd /k "cd src\frontend && npm run dev"

echo.
echo Both services are starting in new windows.
echo - Backend: http://127.0.0.1:8000
echo - Frontend: http://localhost:5173
echo.
echo Close this window at any time.
pause
