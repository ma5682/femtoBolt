@echo off
chcp 65001 >nul
if /I "%~1"=="--list" goto list_only
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0清理历史数据.ps1" -Interactive
pause
exit /b
:list_only
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0清理历史数据.ps1" -ListOnly
exit /b
