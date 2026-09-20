@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m femto.gsplat.scene_3dgs status
pause
