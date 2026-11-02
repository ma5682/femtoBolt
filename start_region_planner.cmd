@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
call "%~dp0_resolve_femto_python.cmd"
if errorlevel 1 exit /b 1
set "PYTHONW=%FEMTO_PYTHONW%"
if not exist "%PYTHONW%" (
    echo 未找到 Python 环境，请先运行 setup_environment.cmd。
    pause
    exit /b 1
)
rem The old standalone planning window is now part of the Qt workstation.
start "Femto Workstation" /B "%PYTHONW%" -m femto.app.qt_main --ui-preview %*
endlocal
