@echo off
setlocal
cd /d "%~dp0"
start "Depth Debug" /B ".venv\Scripts\pythonw.exe" depth_debug_viewer.py %*
endlocal
