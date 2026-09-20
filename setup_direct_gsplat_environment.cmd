@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo DIRECT GSPLAT PRODUCT TRAINING ENVIRONMENT
echo This environment does not install Nerfstudio, PyMeshLab, or Graphdeco code.
echo.

set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if not exist "%CUDA_HOME%\bin\nvcc.exe" (
  echo CUDA Toolkit 12.8 nvcc was not found at:
  echo %CUDA_HOME%\bin\nvcc.exe
  goto :failed
)
set "PATH=%CUDA_HOME%\bin;%PATH%"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo Visual Studio Installer vswhere.exe was not found.
  goto :failed
)
set "VS_INSTALL="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_INSTALL=%%i"
if not defined VS_INSTALL (
  echo Visual Studio 2022 C++ Build Tools were not found.
  goto :failed
)
call "%VS_INSTALL%\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul
where cl >nul 2>nul
if errorlevel 1 (
  echo MSVC cl.exe is unavailable after loading VsDevCmd.bat.
  goto :failed
)

py -3.10 -c "import sys,struct; assert sys.version_info[:2] == (3,10) and struct.calcsize('P') == 8" >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 x64 is required.
  goto :failed
)

if exist ".venv-gsplat\Scripts\python.exe" (
  ".venv-gsplat\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] == (3,10)" >nul 2>nul
  if errorlevel 1 (
    echo Existing .venv-gsplat uses the wrong Python version. Rename it and rerun setup.
    goto :failed
  )
) else (
  py -3.10 -m venv .venv-gsplat
  if errorlevel 1 goto :failed
)

if not exist "C:\FemtoGsplatRuntime\Scripts\python.exe" (
  if exist "C:\FemtoGsplatRuntime" (
    echo C:\FemtoGsplatRuntime already exists and is not this environment.
    goto :failed
  )
  mklink /J "C:\FemtoGsplatRuntime" "%CD%\.venv-gsplat" >nul
  if errorlevel 1 goto :failed
)

".venv-gsplat\Scripts\python.exe" -m pip install --upgrade "pip==25.2" "setuptools<82" wheel ninja
if errorlevel 1 goto :failed

echo.
echo Installing the official PyTorch CUDA 12.8 wheels. This download is several GB.
set "WHEEL_DIR=%USERPROFILE%\Downloads\femto-gsplat-wheels"
set "TORCH_WHEEL=%WHEEL_DIR%\torch-2.7.1+cu128-cp310-cp310-win_amd64.whl"
set "TORCHVISION_WHEEL=%WHEEL_DIR%\torchvision-0.22.1+cu128-cp310-cp310-win_amd64.whl"
set "EXPECTED_TORCH_SHA=5174f02de8ca14df87c8e333c4c39cf3ce93a323c9d470d690301d110a053b3c"
set "EXPECTED_TORCHVISION_SHA=ad48ba3c3ffd48027e3a8de42fcea131a53a524ee9416ca4efb22f9ac6b7328d"
if exist "%TORCH_WHEEL%" if exist "%TORCHVISION_WHEEL%" goto :install_local_torch
echo Local verified wheels were not found. Falling back to the official PyTorch index.
".venv-gsplat\Scripts\python.exe" -m pip install --retries 12 --timeout 120 torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :failed
goto :torch_ready

:install_local_torch
echo Found manually downloaded PyTorch wheels. Verifying official SHA-256 values...
".venv-gsplat\Scripts\python.exe" -c "import os; from pathlib import Path; from download_verified_file import sha256_file; p=Path(os.environ['TORCH_WHEEL']); h=sha256_file(p); print(p.name, h); assert h == os.environ['EXPECTED_TORCH_SHA'], 'torch wheel SHA-256 mismatch'"
if errorlevel 1 goto :failed
".venv-gsplat\Scripts\python.exe" -c "import os; from pathlib import Path; from download_verified_file import sha256_file; p=Path(os.environ['TORCHVISION_WHEEL']); h=sha256_file(p); print(p.name, h); assert h == os.environ['EXPECTED_TORCHVISION_SHA'], 'torchvision wheel SHA-256 mismatch'"
if errorlevel 1 goto :failed
".venv-gsplat\Scripts\python.exe" -m pip install --retries 12 --timeout 120 "%TORCH_WHEEL%" "%TORCHVISION_WHEEL%"
if errorlevel 1 goto :failed

:torch_ready

set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
".venv-gsplat\Scripts\python.exe" -m pip install --retries 12 --timeout 120 -r requirements-gsplat-lock.txt
if errorlevel 1 goto :failed

"C:\FemtoGsplatRuntime\Scripts\python.exe" tools\patch_gsplat_windows.py
if errorlevel 1 goto :failed

".venv-gsplat\Scripts\python.exe" -c "import torch,gsplat; assert torch.cuda.is_available(); print('Direct gsplat runtime ready:', torch.__version__, gsplat.__version__, torch.cuda.get_device_name(0))"
if errorlevel 1 goto :failed

".venv-gsplat\Scripts\python.exe" -m pip freeze > requirements-gsplat-resolved.txt
echo.
echo Direct gsplat environment is ready.
exit /b 0

:failed
echo.
echo Direct gsplat setup failed. Review the output above; the Femto capture environment was not changed.
exit /b 1
