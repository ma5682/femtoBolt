@echo off
setlocal
cd /d "%~dp0"
call "%~dp0..\_resolve_femto_python.cmd"
if errorlevel 1 exit /b 1
"%FEMTO_PYTHON%" main.py %*
if errorlevel 1 pause
endlocal
