@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install mediapipe==1.0.1 numpy==2.2.6 open3d==0.19.0 opencv-contrib-python==5.0.0.93 pillow==12.3.0 trimesh==5.1.0 setuptools==84.0.0
if errorlevel 1 goto :failed
rem pyorbbecsdk2 declares opencv-python, while MediaPipe declares the contrib
rem build. The contrib wheel is a superset and both distributions share cv2,
rem so install Orbbec without pulling a second conflicting OpenCV wheel.
".venv\Scripts\python.exe" -m pip install --no-deps pyorbbecsdk2==2.1.2
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" download_model.py
if errorlevel 1 goto :failed
echo Environment is ready.
exit /b 0

:failed
echo Setup failed. Review the error above.
pause
exit /b 1
