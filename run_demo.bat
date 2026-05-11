@echo off
chcp 65001 >nul
echo.
echo ============================================================
echo   STOPPING OLD DEMO...
echo ============================================================
taskkill /f /im python.exe 2>nul
timeout /t 2 /nobreak >nul

echo.
echo ============================================================
echo   STARTING TRANSFORMER DEMO...
echo ============================================================
echo.

cd /d "%~dp0"
python -u demo_transformer.py

pause
