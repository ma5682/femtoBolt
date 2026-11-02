@echo off
setlocal
cd /d "%~dp0"
call "%~dp0_resolve_femto_python.cmd"
if errorlevel 1 (
  pause
  exit /b 1
)
"%FEMTO_PYTHON%" -m femto.reports.scene_scale_validation %*
