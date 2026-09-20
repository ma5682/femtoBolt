@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m femto.pipeline.complete_demo --mode synthetic
if errorlevel 1 pause
endlocal
