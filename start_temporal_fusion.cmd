@echo off
setlocal
cd /d "%~dp0"
echo Temporal Fusion: keep camera AND scene fixed. Close other camera applications first.
echo Default: 5 frames. Use --mode quality for 15 frames, or --frames 10.
".venv\Scripts\python.exe" scene_temporal_fusion.py --capture %*
pause
endlocal
