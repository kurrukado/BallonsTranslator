@echo off
title BalloonsTranslator
cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo ===================================================
echo   BalloonsTranslator + Gemini Proxy (Auto-Start)
echo ===================================================
echo.

:: 1. Auto-detect or Auto-create Python Virtual Environment (venv)
if not exist "venv\Scripts\python.exe" (
    echo [!] Virtual environment .\venv not found.
    echo [*] Creating virtual environment [python -m venv venv]...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Please install Python 3.10-3.14.
        pause
        exit /b 1
    )
    echo [*] Installing requirements from requirements.txt...
    ".\venv\Scripts\python.exe" -m pip install --upgrade pip
    ".\venv\Scripts\python.exe" -m pip install -r requirements.txt
)

:: 2. Auto-start Gemini Proxy in background if not already running
netstat -ano | findstr /R ":8080 .*LISTENING" >nul 2>&1
if errorlevel 1 (
    if exist "%~dp0gemini-proxy\gemini-proxy.exe" (
        echo [*] Auto-starting Gemini Proxy on port 8080...
        start "" /min "%~dp0gemini-proxy\gemini-proxy.exe"
        ping 127.0.0.1 -n 2 >nul
    ) else if exist "%~dp0gemini-proxy.exe" (
        echo [*] Auto-starting Gemini Proxy on port 8080...
        start "" /min "%~dp0gemini-proxy.exe"
        ping 127.0.0.1 -n 2 >nul
    ) else if exist "E:\gemini-proxy\gemini-proxy.exe" (
        echo [*] Auto-starting Gemini Proxy on port 8080...
        start "" /min "E:\gemini-proxy\gemini-proxy.exe"
        ping 127.0.0.1 -n 2 >nul
    ) else if exist "D:\gemini-proxy\gemini-proxy.exe" (
        echo [*] Auto-starting Gemini Proxy on port 8080...
        start "" /min "D:\gemini-proxy\gemini-proxy.exe"
        ping 127.0.0.1 -n 2 >nul
    )
)

:: 3. Launch BalloonsTranslator
echo [OK] Launching BalloonsTranslator GUI...
".\venv\Scripts\python.exe" launch.py --frozen %*

echo.
echo ===================================================
echo   BalloonsTranslator has closed.
echo ===================================================
pause
