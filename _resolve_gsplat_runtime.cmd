@echo off
rem Resolve the dedicated ASCII runtime and caches for this checkout.
call "%~dp0_resolve_femto_python.cmd"
if errorlevel 1 exit /b 1
set "FEMTO_GSPLAT_ENV="
pushd "%~dp0"
set "FEMTO_RUNTIME_PATHS_FILE=%TEMP%\femto-runtime-%RANDOM%-%RANDOM%.txt"
"%FEMTO_PYTHON%" -c "from femto.paths import ASCII_GSPLAT_RUNTIME_ROOT, workspace_runtime_environment; print('FEMTO_GSPLAT_ENV='+str(ASCII_GSPLAT_RUNTIME_ROOT)); print('FEMTO_ASCII_RUNTIME_ROOT='+str(ASCII_GSPLAT_RUNTIME_ROOT.parent)); [print(k+'='+v) for k,v in workspace_runtime_environment().items()]" > "%FEMTO_RUNTIME_PATHS_FILE%"
if errorlevel 1 (
  del "%FEMTO_RUNTIME_PATHS_FILE%" >nul 2>nul
  popd
  exit /b 1
)
for /f "usebackq tokens=1,* delims==" %%A in ("%FEMTO_RUNTIME_PATHS_FILE%") do set "%%A=%%B"
del "%FEMTO_RUNTIME_PATHS_FILE%" >nul 2>nul
set "FEMTO_RUNTIME_PATHS_FILE="
popd
if not defined FEMTO_GSPLAT_ENV (
  echo Could not resolve this checkout's gsplat runtime. Run setup_environment.cmd first.
  exit /b 1
)
set "VIRTUAL_ENV=%FEMTO_GSPLAT_ENV%"
if exist "%FEMTO_GSPLAT_ENV%\Scripts\python.exe" (
  "%FEMTO_GSPLAT_ENV%\Scripts\python.exe" -I -c "import sys; from pathlib import Path; expected=Path(sys.argv[1]).absolute(); sys.exit(0 if Path(sys.prefix).resolve()==expected and sys.prefix!=sys.base_prefix else 1)" "%FEMTO_GSPLAT_ENV%" >nul 2>nul
  if errorlevel 1 (
    echo 专属 3DGS 环境归属不符或已损坏。
    exit /b 1
  )
)
exit /b 0
