@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Run setup_environment.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u -m femto.capture.scene_capture
if errorlevel 1 pause
