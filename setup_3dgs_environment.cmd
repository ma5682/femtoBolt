@echo off
setlocal
cd /d "%~dp0"

echo DEVELOPMENT TRAINING ENVIRONMENT ONLY.
echo Do not redistribute this Nerfstudio environment with a closed-source product
echo until all transitive licenses, including PyMeshLab GPL-3.0, are reviewed.
echo.

rem Nerfstudio 1.1.5 pins gsplat 1.4.0. On Windows it must compile/JIT its CUDA
rem extension for the RTX 50-series path, so fail before multi-GB downloads when
rem the compiler/toolkit are absent.
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo NVIDIA driver tools were not found. Install a supported NVIDIA driver first.
  goto :failed
)

if exist "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" (
  set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
  set "PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"
)
where nvcc >nul 2>nul
if errorlevel 1 (
  echo CUDA Toolkit 12.8 with nvcc is required before installing the 3DGS runtime.
  goto :failed
)

where cl >nul 2>nul
if errorlevel 1 if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
  for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_INSTALL=%%i"
  if defined VS_INSTALL call "%VS_INSTALL%\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul
)
where cl >nul 2>nul
if errorlevel 1 (
  echo Visual Studio 2022 Build Tools with Desktop development with C++ is required.
  goto :failed
)

py -3.10 -c "import sys; assert sys.version_info[:2] == (3, 10)" >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 x64 is required for the tested Windows 3DGS environment.
  echo The Femto capture environment may remain on Python 3.12.
  goto :failed
)

if exist ".venv-3dgs\Scripts\python.exe" (
  ".venv-3dgs\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] == (3, 10)" >nul 2>nul
  if errorlevel 1 (
    echo Existing .venv-3dgs uses the wrong Python version. Rename or remove it, then rerun setup.
    goto :failed
  )
)
if not exist ".venv-3dgs\Scripts\python.exe" py -3.10 -m venv .venv-3dgs
if errorlevel 1 goto :failed
".venv-3dgs\Scripts\python.exe" -m pip install --upgrade pip "setuptools<82" wheel ninja
if errorlevel 1 goto :failed
".venv-3dgs\Scripts\python.exe" -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed
".venv-3dgs\Scripts\python.exe" -m pip install nerfstudio==1.1.5
if errorlevel 1 goto :failed
set "TORCH_CUDA_ARCH_LIST=12.0"
".venv-3dgs\Scripts\python.exe" -c "import torch, gsplat, nerfstudio; assert torch.cuda.is_available(); print('3DGS runtime ready:', torch.cuda.get_device_name(0))"
if errorlevel 1 goto :failed
echo.
echo 3DGS environment is ready.
pause
exit /b 0
:failed
echo.
echo 3DGS setup failed. Check Log/output and install the Windows C++ and CUDA build prerequisites if gsplat reports a compiler error.
pause
exit /b 1
