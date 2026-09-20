@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" complete_demo.py --mode synthetic
if errorlevel 1 pause
endlocal
