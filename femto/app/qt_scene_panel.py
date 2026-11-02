"""Lazy 3D inspection beside the pixel-bound reference annotation canvas."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel, QStackedWidget


def create_viewport(parent):
    from femto.app.qt_vtk_viewport import VTKViewport
    return VTKViewport(parent)


class ScenePanel(QWidget):
    mode_changed = Signal(bool)

    def __init__(self, canvas, parent=None, *, readonly=False):
        super().__init__(parent)
        self.canvas = canvas
        self._readonly = readonly
        self.viewport = None
        self._scene = (None, None, None)
        self._loaded_key = None
        self._coverage = (None, 0)
        layout = QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        row = QHBoxLayout()
        self.view_controls = row
        self.mode = QComboBox();self.mode.addItems(['参考图与标记', '三维模型检查'])
        self.mode.setToolTip('参考图用于标记区域；三维视图用于检查模型与路线。')
        row.addWidget(self.mode)
        self.reset = QPushButton('复位三维视角');self.reset.clicked.connect(self.reset_camera);row.addWidget(self.reset)
        self.retry = QPushButton('重试三维显示');self.retry.clicked.connect(self._show_3d);row.addWidget(self.retry)
        row.addStretch();layout.addLayout(row)
        self.stack = QStackedWidget();self.stack.addWidget(canvas);layout.addWidget(self.stack,1)
        self.notice = QLabel();self.notice.setWordWrap(True);layout.addWidget(self.notice)
        self.mode.currentIndexChanged.connect(self._switch)
        self.set_scene(None)
        self._switch(0)

    @property
    def is_3d(self):
        return self.mode.currentIndex() == 1

    def set_scene(self, model, region=None, plan=None):
        changed = any(a is not b for a,b in zip(self._scene,(model,region,plan)))
        if self._scene[0] is not model or self._scene[2] is not plan:self._coverage = (None,0)
        self._scene = (model, region, plan)
        self.mode.setEnabled(model is not None)
        if model is None:self.show_reference()
        elif self.is_3d and changed:self._show_3d()

    def show_reference(self):
        self.mode.setCurrentIndex(0)

    def _switch(self, index):
        self.canvas.finish_stroke();self.canvas.timer.stop()
        self.reset.setVisible(index == 1);self.retry.hide()
        if index == 1:self._show_3d()
        else:
            self.stack.setCurrentWidget(self.canvas)
            self.stack.setToolTip('')
            self.notice.setText('只读回看' if self._readonly else '')
            self.notice.setVisible(self._readonly)
        self.mode_changed.emit(index == 1)

    def _show_3d(self):
        if self._scene[0] is None:return
        key = tuple(id(value) for value in self._scene)
        try:
            if self.viewport is None:
                viewport = create_viewport(self)
                self.viewport = viewport
                self.stack.addWidget(viewport)
                viewport.picked.connect(self._picked)
            if key != self._loaded_key:
                self.viewport.set_scene(*self._scene,
                    reset_camera=self._loaded_key is None or key[0] != self._loaded_key[0])
                if hasattr(self.viewport, 'set_coverage'):self.viewport.set_coverage(*self._coverage)
                self._loaded_key = key
            self.stack.setCurrentWidget(self.viewport)
            self.retry.hide();self.reset.setEnabled(True)
            self.stack.setToolTip('拖动平移 · 滚轮缩放 · 单击拾取坐标。修改区域请返回参考图。')
            self.notice.setText('覆盖显示为近似；未知范围请在参考图核对。' if self._coverage[0] is not None else
                                '绿色：工作区域 · 黄色：路线')
            self.notice.show()
        except Exception as error:
            self._failed(error)

    def _failed(self,error):
        self._loaded_key = None
        self.stack.setCurrentWidget(self.canvas)
        self.retry.show();self.reset.setEnabled(False)
        self.notice.setText('三维显示未完成：'+str(error)+'；可重试或返回参考图继续操作。')
        self.notice.show()

    def reset_camera(self):
        if self.viewport is not None:
            try:self.viewport.reset_camera()
            except Exception as error:self._failed(error)

    def _picked(self, record):
        xyz = record['point_world_m']
        text = f"原始网格面 {record['triangle_index']} · 世界坐标（m）：" + ', '.join(f'{value:.6f}' for value in xyz)
        display = self._coverage[0]
        if display is not None:
            from femto.planning.coverage_display import NAMES
            states = set(display.status[display.source_faces == record['triangle_index']])
            text += ' · 该面的目标单元：'+('、'.join(NAMES[int(state)] for state in sorted(states)) if states else '不在此目标台账中')
        self.notice.setText(text+' · 仅供研发核对')
        self.notice.show()

    def set_coverage(self, display=None, reason=0):
        if self._coverage[0] is display and self._coverage[1] == reason:return
        self._coverage = (display, reason)
        if self.viewport is not None and self.is_3d:
            try:
                self.viewport.set_coverage(display, reason)
                self._show_3d()
            except Exception as error:self._failed(error)

    def focus_coverage(self, reason=0):
        if self.is_3d and self.viewport is not None:
            try:return self.viewport.focus_coverage(reason)
            except Exception as error:self._failed(error)
        return False

    def shutdown(self):
        if self.viewport is not None:
            self.viewport.shutdown()
            self.stack.removeWidget(self.viewport)
            self.viewport.deleteLater();self.viewport=None
            self._loaded_key=None
        self.show_reference()

    def closeEvent(self,event):
        self.shutdown()
        super().closeEvent(event)
