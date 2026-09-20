@echo off
setlocal
cd /d "%~dp0"
start "Femto Control" /B ".venv\Scripts\pythonw.exe" -m femto.app.femto_control_app
endlocal
