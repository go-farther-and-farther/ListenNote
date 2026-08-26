@echo off
cd /d "%~dp0"
echo ========================================
echo   Installing dependencies
echo ========================================

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] python not found in PATH
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Install failed
    pause
    exit /b 1
)
echo.
echo ========================================
echo   Done! Double-click run-record.bat to start
echo ========================================
pause
