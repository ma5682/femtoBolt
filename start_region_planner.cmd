@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" "%~dp0region_project.py"
if errorlevel 1 pause
