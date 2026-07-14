@echo off
REM RevFinder launcher for Windows. Double-click, or run from a command prompt.
REM Creates the virtual environment on first run, installs deps, then serves on the LAN.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv || python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo.
        echo ERROR: Could not create a virtual environment.
        echo Install Python 3.9+ from https://www.python.org/downloads/ ^(check "Add to PATH"^) and retry.
        pause
        exit /b 1
    )
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo.
echo Starting RevFinder on port 8502 (reachable at http://THIS-PC-IP:8502)
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8502 --server.address 0.0.0.0 --server.headless true

pause
