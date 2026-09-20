@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: drag a result folder here, or pass result_folder --spec spec.json
  pause
  exit /b 2
)
"%~dp0.venv\Scripts\python.exe" "%~dp0scene_quality_report.py" %*
if errorlevel 1 pause
