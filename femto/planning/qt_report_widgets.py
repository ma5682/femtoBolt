"""Qt widgets shared by the planning reports, without an application dependency."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QLabel, QSizePolicy, QTextBrowser, QVBoxLayout


class ReportDialog(QDialog):
    """Modeless report whose lifetime is bound to its source plan and owner."""

    def __init__(self, parent, title, is_current=lambda: True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.plan_is_current = is_current
        self._closed = False
        self._guard = QTimer(self)
        self._guard.setInterval(200)
        self._guard.timeout.connect(self.check_current)
        self._guard.start()
        # Python owns the wrapper while a modeless report is open.
        if parent is not None:
            if not hasattr(parent, '_planning_report_windows'):
                parent._planning_report_windows = []
            parent._planning_report_windows.append(self)

    def check_current(self):
        if self._closed:
            return False
        if not self.plan_is_current():
            self.close()
            return False
        return True

    def stop(self):
        self._closed = True
        self._guard.stop()
        for child in self.findChildren(ReportDialog):
            child.close()
        parent = self.parent()
        if parent is not None:
            windows = getattr(parent, '_planning_report_windows', [])
            if self in windows:
                windows.remove(self)

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)

    def done(self, result):
        self.stop()
        super().done(result)


class ImageView(QLabel):
    """Fit the original image on resize without changing export resolution."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(180, 150)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._original = None

    def set_image(self, picture):
        image = picture.convert('RGB')
        pixels = image.tobytes()
        self._original = QPixmap.fromImage(QImage(
            pixels, image.width, image.height, image.width * 3,
            QImage.Format_RGB888).copy())
        self._fit()

    def _fit(self):
        if self._original is not None:
            self.setPixmap(self._original.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


def text_report(parent, title, text, is_current=lambda: True):
    window = ReportDialog(parent, title, is_current)
    window.resize(1000, 700)
    layout = QVBoxLayout(window)
    content = QTextBrowser(window)
    content.setPlainText(text)
    layout.addWidget(content)
    window.show()
    return window
