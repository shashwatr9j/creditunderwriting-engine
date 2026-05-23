@echo off
title Alt-Credit Underwriting Engine
echo.
echo ============================================================
echo   Multi-Agent Alternative Credit Underwriting Engine
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: Create venv if it doesn't exist
if not exist ".venv\" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
)

:: Activate and install
echo [2/3] Installing dependencies...
call .venv\Scripts\activate.bat
pip install -r requirements.txt --quiet

:: Launch
echo [3/3] Starting Streamlit at http://localhost:8501
echo.
echo       Press Ctrl+C to stop the server.
echo.
streamlit run app.py --server.headless false
