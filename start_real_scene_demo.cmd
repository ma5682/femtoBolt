@echo off
setlocal
cd /d "%~dp0"
set "before_marker="
if exist "datasets\latest_scene.txt" set /p before_marker=<"datasets\latest_scene.txt"
echo [1/2] 请先调整 Femto 画面，再点 START CAPTURE；围绕场景缓慢移动并保持相邻画面重叠。
".venv\Scripts\python.exe" -m femto.capture.scene_capture --capture-seconds 10 --capture-fps 6 --max-frames 60
if errorlevel 1 goto :error
set "after_marker="
if exist "datasets\latest_scene.txt" set /p after_marker=<"datasets\latest_scene.txt"
if "%after_marker%"=="" goto :nocapture
if "%after_marker%"=="%before_marker%" goto :nocapture
echo [2/2] 正在估算位姿、执行质量门禁并重建最新实拍场景……
".venv\Scripts\python.exe" -m femto.pipeline.complete_demo --mode latest
if errorlevel 1 goto :error
goto :end
:error
echo Demo 已停止：采集、位姿或质量门禁没有通过。
pause
goto :end
:nocapture
echo 没有检测到新的采集数据，本次不会误用旧数据。
pause
:end
endlocal
