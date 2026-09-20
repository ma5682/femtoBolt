@echo off
setlocal
cd /d "%~dp0"
if not exist "C:\FemtoGsplatRuntime\Scripts\python.exe" (
  echo Direct gsplat environment is missing. Run setup_direct_gsplat_environment.cmd first.
  exit /b 2
)
"C:\FemtoGsplatRuntime\Scripts\python.exe" scene_gsplat_trainer.py status %*
