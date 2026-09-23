@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo 请把区域导出目录或 region_N.json 拖到本文件上（阶段 C 诊断：派生表面候选，不进入规划）。
  pause
  exit /b 2
)
"%~dp0.venv\Scripts\python.exe" -X utf8 -m femto.planning.derived_surface_candidates "%~1"
set "DERIVED_EXIT=%ERRORLEVEL%"
if not "%DERIVED_EXIT%"=="0" echo 候选诊断被拒绝，详情见输出目录中的 derived_surface_blocked.json。
pause
exit /b %DERIVED_EXIT%
