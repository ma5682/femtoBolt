@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo 请把区域导出目录或 region_N.json 拖到本文件上。
  pause
  exit /b 2
)
"%~dp0.venv\Scripts\python.exe" -X utf8 -m femto.planning.mesh_slice_planner "%~1"
set "TRAJECTORY_EXIT=%ERRORLEVEL%"
if not "%TRAJECTORY_EXIT%"=="0" echo 轨迹生成被拒绝，详情见输出目录中的 trajectory_blocked.json。
pause
exit /b %TRAJECTORY_EXIT%
