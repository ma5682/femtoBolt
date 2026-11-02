@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
rem The checker uses only stdlib: prefer the base launcher so a broken venv can be diagnosed.
py -3.12 -I -c "import sys" >nul 2>nul
if not errorlevel 1 (
  py -3.12 -I -B -X utf8 tools\check_workspace_runtime.py %*
) else (
  if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -I -B -X utf8 tools\check_workspace_runtime.py %*
  ) else (
    echo 找不到可用于诊断的 Python 3.12。请安装 Python 3.12 后重试。
    exit /b 1
  )
)
set "FEMTO_CHECK_EXIT=%ERRORLEVEL%"
if "%~1"=="" pause
exit /b %FEMTO_CHECK_EXIT%
