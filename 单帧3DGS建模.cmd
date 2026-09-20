@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -c "from pathlib import Path; from scene_single_gs import launch,ROOT; from scene_single_gs_viewer import open_viewer; p=Path((ROOT/'outputs/latest_complete_demo.txt').read_text(encoding='utf-8-sig').strip()).parent; open_viewer(launch(p))"
if errorlevel 1 pause
