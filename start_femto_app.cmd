@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "ROOT=%~dp0"
call "%~dp0_resolve_femto_python.cmd"
if errorlevel 1 exit /b 1
set "PYTHONW=%FEMTO_PYTHONW%"
if not exist "%PYTHONW%" (
    echo 未找到 Python 环境：
    echo   %ROOT%.venv\Scripts\pythonw.exe
    pause
    exit /b 1
)
set "PYTHON=%FEMTO_PYTHON%"
if not exist "%PYTHON%" (
    echo 找到 pythonw，但没有对应的 python.exe：%PYTHON%
    pause
    exit /b 1
)
if not exist "%ROOT%outputs" mkdir "%ROOT%outputs"
"%PYTHON%" -c "import PySide6; import femto.app.qt_main" >nul 2>"%ROOT%outputs\qt_startup_check.log"
if errorlevel 1 (
    echo PySide6 工作台启动检查失败。请查看：
    echo   %ROOT%outputs\qt_startup_check.log
    pause
    exit /b 1
)
rem The Qt workstation keeps the original capture -> region -> route -> save flow.
start "Femto Workstation" /B "%PYTHONW%" -m femto.app.qt_main %*
endlocal
