@echo off
setlocal
cd /d "%~dp0"
if not exist "C:\FemtoGsplatRuntime\Scripts\python.exe" (
  echo Direct gsplat environment is missing. Run setup_direct_gsplat_environment.cmd first.
  exit /b 2
)
set "CUDA_HOME=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=C:\FemtoGsplatRuntime\Scripts;%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :run
set "VS_INSTALL="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_INSTALL=%%i"
if not defined VS_INSTALL goto :run
call "%VS_INSTALL%\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul
:run
set "VSLANG=1033"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul
"C:\FemtoGsplatRuntime\Scripts\python.exe" scene_gsplat_trainer.py train %*
