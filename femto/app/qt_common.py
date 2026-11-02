"""Qt presentation utilities and a single cancellable worker per task owner.

Backends receive a cancellation event and progress function. No worker touches
widgets; success/error delivery always happens on the Qt application thread.
"""
from __future__ import annotations

import threading
import traceback
import re
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from PySide6.QtCore import QObject, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QImage
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout

from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
from femto.planning.planning_revision import install_allowed
from femto.operation_diagnostics import OperationSummary, record_operation
from femto.compute_scheduling import interactive_task_thread

TaskValue = TypeVar('TaskValue')


def breakable_text(text: str) -> str:
    """Let labels wrap long ASCII tokens without changing the stored source."""
    return re.sub(r'[!-~]{17,}', lambda match: '\u200b'.join(
        match[0][i:i+16] for i in range(0, len(match[0]), 16)), text)


class TaskResultDiscarded(GeometryTaskCancelled):
    """A completed worker no longer belongs to the current calculation inputs."""


class TaskRunner(QObject):
    progress = Signal(str)
    finished = Signal()
    cancel_requested = Signal()
    _delivered = Signal(int, object, object)
    _progressed = Signal(int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.busy = False
        self._number = 0
        self._thread = None
        self._cancel = threading.Event()
        self._on_success = self._on_error = None
        self.last_diagnostics: OperationSummary | None = None
        self._delivery_diagnostics: tuple[int, OperationSummary] | None = None
        self._delivered.connect(self._complete, Qt.QueuedConnection)
        self._progressed.connect(self._report, Qt.QueuedConnection)

    def start(self, function: Callable[[threading.Event, Callable[[str], None]], TaskValue],
              on_success: Callable[[TaskValue], None], on_error: Callable[[Exception], None] | None = None, *,
              revision: Any = None, current_revision: Callable[[], Any] | None = None) -> bool:
        if self.busy:
            return False
        self.busy = True
        self._number += 1
        number = self._number
        self._cancel = cancel = threading.Event()
        self._on_success, self._on_error = on_success, on_error
        self._revision, self._current_revision = revision, current_revision
        self.last_diagnostics = self._delivery_diagnostics = None

        def run():
            value = error = None
            trace = None
            try:
                with interactive_task_thread():
                    with record_operation('qt_worker') as trace:
                        value = function(cancel, lambda text: self._progressed.emit(number, str(text)))
            except Exception as exc:
                error = exc
            if trace is not None:
                self._delivery_diagnostics = (number, trace.summary())
            self._delivered.emit(number, value, error)

        # Saving must be allowed to finish or cancel transactionally on close.
        self._thread = threading.Thread(target=run, name=f'QtTask-{number}', daemon=False)
        self._thread.start()
        return True

    @Slot(int, str)
    def _report(self, number: int, message: str) -> None:
        if number == self._number and self.busy and not self._cancel.is_set():
            self.progress.emit(message)

    @Slot(int, object, object)
    def _complete(self, number: int, value: Any, error: Exception | None) -> None:
        if number != self._number or not self.busy:
            return
        self.busy = False
        delivered = self._delivery_diagnostics
        self.last_diagnostics = delivered[1] if delivered is not None and delivered[0] == number else None
        self._delivery_diagnostics = None
        success, failure = self._on_success, self._on_error
        self._on_success = self._on_error = None
        try:
            current = self._current_revision() if self._current_revision is not None else self._revision
            if not install_allowed(request=number, current_request=self._number,
                                   cancelled=self._cancel.is_set(), revision=self._revision, current_revision=current):
                error = (GeometryTaskCancelled('任务已取消') if self._cancel.is_set() else
                         TaskResultDiscarded('输入或当前模型已变化，旧任务结果未安装；请重新计算。'))
            if error is None:
                success(value)
            elif failure is not None:
                failure(error)
            else:
                self.progress.emit(f'操作失败：{error}')
        except Exception as exc:
            traceback.print_exc()
            if failure is not None:
                failure(exc)
            else:
                self.progress.emit(f'结果处理失败：{exc}')
        finally:
            self.finished.emit()

    @property
    def cancelling(self) -> bool:
        return self.busy and self._cancel.is_set()

    def cancel(self) -> None:
        notify = self.busy and not self._cancel.is_set()
        self._cancel.set()
        if notify:
            self.cancel_requested.emit()

    def shutdown(self, timeout_ms: int = 200) -> bool:
        self.cancel()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(max(0, timeout_ms) / 1000)
        return self._thread is None or not self._thread.is_alive()


def rgb_image(pixels: np.ndarray) -> QImage:
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if array.ndim == 2:
        kind = QImage.Format_Grayscale8
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        kind = QImage.Format_RGB888 if array.shape[2] == 3 else QImage.Format_RGBA8888
    else:
        raise ValueError('显示图像必须是灰度、RGB 或 RGBA 数组')
    height, width = array.shape[:2]
    return QImage(array.data, width, height, array.strides[0], kind).copy()


def show_text(parent, title: str, text: str):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(820, 600)
    layout = QVBoxLayout(dialog)
    content = QTextBrowser(dialog)
    content.setPlainText(str(text))
    layout.addWidget(content)
    buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=dialog)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.exec()


def open_path(path):
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f'文件或目录不存在：{target}')
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
        raise OSError(f'系统未能打开：{target}')
