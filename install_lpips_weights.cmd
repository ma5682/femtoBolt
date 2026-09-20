@echo off
setlocal
cd /d "%~dp0"
if not exist "C:\FemtoGsplatRuntime\Scripts\python.exe" (
  echo Direct gsplat environment is missing. Run setup_direct_gsplat_environment.cmd first.
  exit /b 2
)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul
echo Installing the official torchvision AlexNet checkpoint for LPIPS evaluation.
echo Default source: %USERPROFILE%\Downloads\alexnet-owt-7be5be79.pth  (override with --source PATH)
echo The file is verified against the official SHA-256 before it is copied into the torch hub cache.
"C:\FemtoGsplatRuntime\Scripts\python.exe" scene_gsplat_trainer.py install-lpips %*
