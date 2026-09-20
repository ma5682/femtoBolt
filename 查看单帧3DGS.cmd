@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -c "from pathlib import Path; from femto.gsplat.scene_single_gs import MARKER; from femto.gsplat.scene_single_gs_viewer import open_viewer; open_viewer(Path(MARKER.read_text(encoding='utf-8-sig').strip()))"
if errorlevel 1 pause
