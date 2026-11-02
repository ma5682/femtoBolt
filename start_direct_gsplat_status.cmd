@echo off
setlocal
cd /d "%~dp0"
call "%~dp0_resolve_gsplat_runtime.cmd"
if errorlevel 1 exit /b 2
if not exist "%FEMTO_GSPLAT_ENV%\Scripts\python.exe" (
  echo Direct gsplat environment is missing. Run setup_direct_gsplat_environment.cmd first.
  exit /b 2
)
"%FEMTO_GSPLAT_ENV%\Scripts\python.exe" -m femto.gsplat.scene_gsplat_trainer status %*
