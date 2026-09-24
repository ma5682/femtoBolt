"""Launch with: python main.py"""

import sys
from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("三维采集与轨迹规划工作台 · 原型")
    app.setFont(QFont("Microsoft YaHei", 10))
    qss = Path(__file__).parent / "styles" / "medical.qss"
    app.setStyleSheet(qss.read_text(encoding="utf-8"))
    window = MainWindow(records_dir=Path(__file__).resolve().parent / "demo_records")
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
