@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scene_demo_viewer.py
if errorlevel 1 pause
endlocal
