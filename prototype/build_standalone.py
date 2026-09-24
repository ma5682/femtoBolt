"""Generate the optional self-contained preview from the modular application."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
MODULES = [
    "models/capture_state.py", "models/task_state.py", "models/planning_state.py",
    "app/app_state.py", "mock/mock_patient.py", "mock/mock_capture.py", "mock/mock_planning.py", "mock/plan_store.py",
    "app/navigation.py", "app/controller.py",
    "adapters/project_data.py", "adapters/replay_images.py",
    "widgets/status_panel.py", "widgets/patient_header.py", "widgets/workflow_sidebar.py", "widgets/action_bar.py", "widgets/tool_palette.py",
    "pages/capture_page.py", "pages/region_page.py", "pages/planning_page.py", "pages/confirmation_page.py",
    "widgets/medical_viewport.py", "widgets/viewport.py", "widgets/replay_viewport.py", "app/presentation.py", "app/main_window.py",
]
parts = ['"""Generated standalone mock workstation. Run: python standalone.py"""']
for filename in MODULES:
    source = (ROOT / filename).read_text(encoding="utf-8")
    lines = [line for line in source.splitlines() if not re.match(r"\s*from (app|models|mock|pages|widgets|adapters)\.", line) and not line.startswith("from __future__")]
    parts.append("\n".join(lines))
parts.append("MEDICAL_QSS = " + repr((ROOT / "styles/medical.qss").read_text(encoding="utf-8")))
entry = (ROOT / "main.py").read_text(encoding="utf-8")
entry = entry.replace("from app.main_window import MainWindow", "")
entry = entry.replace('qss = Path(__file__).parent / "styles" / "medical.qss"', 'qss = None')
entry = entry.replace('app.setStyleSheet(qss.read_text(encoding="utf-8"))', 'app.setStyleSheet(MEDICAL_QSS)')
parts.append(entry)
(ROOT / "standalone.py").write_text("\n\n".join(parts), encoding="utf-8")
print("Standalone rebuilt")
