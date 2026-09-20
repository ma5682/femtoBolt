@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" -m femto.reports.scene_scale_validation %*
