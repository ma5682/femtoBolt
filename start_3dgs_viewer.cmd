@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scene_3dgs.py view
if errorlevel 1 pause
