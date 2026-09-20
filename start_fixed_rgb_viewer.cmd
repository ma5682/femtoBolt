@echo off
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" -X utf8 -m femto.planning.fixed_rgb_viewer %*
if errorlevel 1 pause
