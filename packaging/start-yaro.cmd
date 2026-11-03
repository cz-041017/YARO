@echo off
rem YARO one-click launcher (works for both source checkout and offline install).
rem Prefers the repo .venv, then the offline install dir, then the system python.
rem NOTE: ASCII-only on purpose; .cmd files are parsed with the console codepage.
setlocal
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\YARO\.venv\Scripts\python.exe" set "PY=%LOCALAPPDATA%\YARO\.venv\Scripts\python.exe"
if not defined PY set "PY=python"

echo Using interpreter: %PY%
"%PY%" -m yaro app --port 8765
if errorlevel 1 (
  echo.
  echo Window mode failed, falling back to browser mode: http://127.0.0.1:8765/
  "%PY%" -m yaro gui --port 8765
)
if errorlevel 1 pause
