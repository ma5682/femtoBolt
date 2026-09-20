@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m femto.viewer.scene_demo_viewer
if errorlevel 1 pause
endlocal
