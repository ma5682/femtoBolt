"""Launch with: python main.py"""

import sys
import argparse
from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> int:
    parser = argparse.ArgumentParser(description="三维采集与规划工作台 · 独立界面原型")
    parser.add_argument("--replay", action="store_true", help="只读载入已保存项目；默认使用 Mock 演示")
    args, qt_args = parser.parse_known_args()
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("三维采集与轨迹规划工作台 · 原型")
    app.setFont(QFont("Microsoft YaHei", 10))
    qss = Path(__file__).parent / "styles" / "medical.qss"
    app.setStyleSheet(qss.read_text(encoding="utf-8"))
    window = MainWindow(records_dir=Path(__file__).resolve().parent / "demo_records", use_project_replay=args.replay)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
