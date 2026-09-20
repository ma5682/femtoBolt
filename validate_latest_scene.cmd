@echo off
setlocal
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m femto.dataset.scene_dataset_validator
pause
