@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m femto.planning.region_project
if errorlevel 1 pause
