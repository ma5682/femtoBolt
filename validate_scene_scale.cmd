@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" scene_scale_validation.py %*
