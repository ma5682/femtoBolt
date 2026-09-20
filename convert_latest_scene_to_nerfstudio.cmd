@echo off
setlocal
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m femto.pipeline.scene_to_nerfstudio
pause
