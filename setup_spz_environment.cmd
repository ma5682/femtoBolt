@echo off
setlocal EnableExtensions
cd /d "%~dp0"
call "%~dp0_resolve_gsplat_runtime.cmd"
if errorlevel 1 exit /b 2
if not exist "%FEMTO_GSPLAT_ENV%\Scripts\python.exe" (
  echo Direct gsplat environment is missing. Run setup_direct_gsplat_environment.cmd first.
  exit /b 2
)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo Visual Studio Installer was not found.
  exit /b 2
)
set "VS_INSTALL="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_INSTALL=%%i"
if not defined VS_INSTALL (
  echo Visual Studio 2022 C++ Build Tools were not found.
  exit /b 2
)
call "%VS_INSTALL%\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul
set "SPZ_SOURCE=%FEMTO_ASCII_RUNTIME_ROOT%\spz-deps\spz-src"
set "ZLIB_SOURCE=%FEMTO_ASCII_RUNTIME_ROOT%\spz-deps\zlib-1.3.1"
set "ZSTD_SOURCE=%FEMTO_ASCII_RUNTIME_ROOT%\spz-deps\zstd-src"
set "SPZ_ARCHIVE=%FEMTO_ASCII_RUNTIME_ROOT%\spz-deps\spz-v3.0.0.zip"
set "ZLIB_ARCHIVE=%FEMTO_ASCII_RUNTIME_ROOT%\spz-deps\zlib-1.3.1.tar.gz"
set "ZSTD_ARCHIVE=%FEMTO_ASCII_RUNTIME_ROOT%\spz-deps\zstd-1.5.6.tar.gz"
set "NANOBIND_WHEEL=%FEMTO_ASCII_RUNTIME_ROOT%\spz-deps\nanobind-3.0.1-py3-none-any.whl"
set "SPZ_BUILD_CONSTRAINT=%~dp0spz-build-constraints.txt"
set "SPZ_BUILT_ZLIB=%SPZ_SOURCE%\build\cp310-cp310-win_amd64\_deps\zlib-build\Release\zlib.dll"
set "SPZ_RUNTIME_ZLIB=%FEMTO_GSPLAT_ENV%\Lib\site-packages\spz\zlib.dll"
if not exist "%SPZ_SOURCE%\CMakeLists.txt" (
  echo Local SPZ source cache is missing: %SPZ_SOURCE%
  echo Download the fixed official tag v3.0.0 archive and extract it there, then rerun this script.
  exit /b 3
)
if not exist "%ZLIB_SOURCE%\CMakeLists.txt" (
  echo Local zlib source cache is missing: %ZLIB_SOURCE%
  echo Download zlib-1.3.1.tar.gz with SHA-256 9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23 and extract it there.
  exit /b 3
)
if not exist "%ZSTD_SOURCE%\build\cmake\CMakeLists.txt" (
  echo Local zstd source cache is missing: %ZSTD_SOURCE%
  echo Download zstd-1.5.6.tar.gz with SHA-256 8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1 and extract it there.
  exit /b 3
)
if not exist "%SPZ_ARCHIVE%" (
  echo Verified SPZ archive is missing: %SPZ_ARCHIVE%
  exit /b 3
)
if not exist "%ZLIB_ARCHIVE%" (
  echo Verified zlib archive is missing: %ZLIB_ARCHIVE%
  exit /b 3
)
if not exist "%ZSTD_ARCHIVE%" (
  echo Verified zstd archive is missing: %ZSTD_ARCHIVE%
  exit /b 3
)
if not exist "%NANOBIND_WHEEL%" (
  echo Verified nanobind wheel is missing: %NANOBIND_WHEEL%
  exit /b 3
)
if not exist "%SPZ_BUILD_CONSTRAINT%" (
  echo SPZ build constraint file is missing: %SPZ_BUILD_CONSTRAINT%
  exit /b 3
)
"%FEMTO_GSPLAT_ENV%\Scripts\python.exe" -c "import hashlib, os, pathlib, sys; expected={os.environ['SPZ_ARCHIVE']:'7f0f08ab144ecc940475b4edbfbb967d504aa2c78a1fbed17fd857bef8d95665',os.environ['ZLIB_ARCHIVE']:'9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23',os.environ['ZSTD_ARCHIVE']:'8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1',os.environ['NANOBIND_WHEEL']:'4b49491fe8bf483a5e0342125b8fe56237e490ba642c7e296170b8cd256a7764'}; actual={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in expected}; print(actual); sys.exit(0 if actual == expected else 4)"
if errorlevel 1 (
  echo Source archive SHA-256 verification failed.
  exit /b 4
)
echo Building Niantic SPZ tag v3.0.0 from the verified local MIT source cache...
echo Source commit: 5bf2945de1a003cee07133b1e495fe9c6ffdc7e7
echo zlib SHA-256: 9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
echo zstd SHA-256: 8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1
echo nanobind: 3.0.1, wheel SHA-256: 4b49491fe8bf483a5e0342125b8fe56237e490ba642c7e296170b8cd256a7764
set "CMAKE_ARGS=-DFETCHCONTENT_SOURCE_DIR_ZLIB=%ZLIB_SOURCE:\=/% -DFETCHCONTENT_SOURCE_DIR_ZSTD=%ZSTD_SOURCE:\=/% -DCMAKE_CXX_FLAGS=/utf-8 -DCMAKE_C_FLAGS=/utf-8"
set "PIP_CONSTRAINT=%SPZ_BUILD_CONSTRAINT%"
set "PIP_FIND_LINKS=%FEMTO_ASCII_RUNTIME_ROOT%\spz-deps"
"%FEMTO_GSPLAT_ENV%\Scripts\python.exe" -c "import importlib.metadata as m,sys; actual=m.version('numpy'); print('preinstalled numpy:',actual); sys.exit(0 if actual=='2.2.6' else 5)"
if errorlevel 1 (
  echo The product runtime numpy version is not the locked 2.2.6.
  exit /b 5
)
"%FEMTO_GSPLAT_ENV%\Scripts\python.exe" -m pip install --force-reinstall --no-deps "%SPZ_SOURCE%"
if errorlevel 1 exit /b 1
if not exist "%SPZ_BUILT_ZLIB%" (
  echo Pinned zlib 1.3.1 build output is missing: %SPZ_BUILT_ZLIB%
  exit /b 6
)
"%FEMTO_GSPLAT_ENV%\Scripts\python.exe" -c "import hashlib,pathlib,shutil,sys; source=pathlib.Path(r'%SPZ_BUILT_ZLIB%'); destination=pathlib.Path(r'%SPZ_RUNTIME_ZLIB%'); expected='2f08f389724c8a05c672eeb3dec1ac1ff7f589d133d4011d807e489bee522baa'; actual=hashlib.sha256(source.read_bytes()).hexdigest(); print('built zlib 1.3.1 SHA-256:',actual); sys.exit(6) if actual != expected else None; shutil.copyfile(source,destination); print('SPZ zlib 1.3.1 DLL:',destination)"
if errorlevel 1 exit /b 6
"%FEMTO_GSPLAT_ENV%\Scripts\python.exe" -c "import ctypes,hashlib,pathlib,sys; path=pathlib.Path(r'%SPZ_RUNTIME_ZLIB%'); dll=ctypes.CDLL(str(path)); dll.zlibVersion.restype=ctypes.c_char_p; version=dll.zlibVersion().decode('ascii'); actual=hashlib.sha256(path.read_bytes()).hexdigest(); print('runtime zlib:',version,'SHA-256:',actual); sys.exit(6) if version != '1.3.1' or actual != '2f08f389724c8a05c672eeb3dec1ac1ff7f589d133d4011d807e489bee522baa' else None"
if errorlevel 1 exit /b 6
"%FEMTO_GSPLAT_ENV%\Scripts\python.exe" -c "import importlib.metadata as m,spz; print('Niantic SPZ Python bindings ready:', spz.__file__); print('distribution version:', m.version('spz')); print('source tag: v3.0.0; commit: 5bf2945de1a003cee07133b1e495fe9c6ffdc7e7')"
if errorlevel 1 exit /b 1
