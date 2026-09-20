@echo off
chcp 65001 >nul
cd /d "%~dp0"
".venv\Scripts\python.exe" -X utf8 fixed_rgb_viewer.py %*
if errorlevel 1 pause
