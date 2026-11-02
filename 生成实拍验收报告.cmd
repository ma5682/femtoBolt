@echo off
setlocal
cd /d "%~dp0"
call "%~dp0_resolve_femto_python.cmd"
if errorlevel 1 (
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Usage: drag a result folder here, or pass result_folder --spec spec.json
  pause
  exit /b 2
)
"%FEMTO_PYTHON%" -m femto.reports.scene_quality_report %*
if errorlevel 1 pause
