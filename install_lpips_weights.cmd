@echo off
setlocal
cd /d "%~dp0"
call "%~dp0_resolve_gsplat_runtime.cmd"
if errorlevel 1 exit /b 2
if not exist "%FEMTO_GSPLAT_ENV%\Scripts\python.exe" (
  echo Direct gsplat environment is missing. Run setup_direct_gsplat_environment.cmd first.
  exit /b 2
)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul
echo Installing the official torchvision AlexNet checkpoint for LPIPS evaluation.
echo Default source: %USERPROFILE%\Downloads\alexnet-owt-7be5be79.pth  (override with --source PATH)
echo The file is verified against the official SHA-256 before it is copied into the torch hub cache.
"%FEMTO_GSPLAT_ENV%\Scripts\python.exe" -m femto.gsplat.scene_gsplat_trainer install-lpips %*
