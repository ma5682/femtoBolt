@echo off
chcp 65001 >nul
rem Resolve only this checkout's capture environment.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "VIRTUAL_ENV=%~dp0.venv"
set "FEMTO_PYTHON=%~dp0.venv\Scripts\python.exe"
set "FEMTO_PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%FEMTO_PYTHON%" (
  echo 未找到 Femto Python 环境。请先运行 setup_environment.cmd。
  exit /b 1
)
"%FEMTO_PYTHON%" -I -c "import sys; from pathlib import Path; expected=Path(sys.argv[1]).absolute(); actual=Path(sys.prefix).resolve(); sys.exit(0 if actual==expected and sys.prefix!=sys.base_prefix else 1)" "%~dp0.venv" >nul 2>nul
if errorlevel 1 (
  echo 当前 .venv 不是本工作区独立环境，或环境已损坏。请检查环境归属。
  exit /b 1
)
exit /b 0
