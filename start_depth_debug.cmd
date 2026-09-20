@echo off
setlocal
cd /d "%~dp0"
start "Depth Debug" /B ".venv\Scripts\pythonw.exe" -m femto.viewer.depth_debug_viewer %*
endlocal
