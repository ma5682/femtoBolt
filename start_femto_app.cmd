@echo off
setlocal
cd /d "%~dp0"
start "Femto Control" /B ".venv\Scripts\pythonw.exe" femto_control_app.py
endlocal
