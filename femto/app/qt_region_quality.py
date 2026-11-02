"""Debounced, source-scoped quality feedback shared by planning and read-only review."""
from PySide6.QtCore import QTimer, Signal, Qt, QSize
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

from femto.app.qt_common import TaskRunner, show_text
from femto.app.qt_icons import line_icon
from femto.app.qt_planning_io import clone_model
from femto.app.region_quality import assess_selection, LABELS, LIMITS
from femto.planning.planning_revision import capture, array_record


class RegionQualityPanel(QWidget):
    changed = Signal()
    focus_requested = Signal()
    rescan_requested = Signal()

    def __init__(self, canvas, parent=None, *, compact=False):
        super().__init__(parent)
        self.canvas = canvas
        self._compact=compact
        self._message='尚未加载数据'
        self.setObjectName('regionQualityCard');self.setAttribute(Qt.WA_StyledBackground,True)
        self.result = None
        self._generation = 0
        self._closing = False
        self.runner = TaskRunner(self)
        self.timer = QTimer(self); self.timer.setSingleShot(True)
        self.timer.setInterval(140); self.timer.timeout.connect(self._start)
        layout = QVBoxLayout(self); layout.setContentsMargins(8, 3, 8, 3);layout.setSpacing(5)
        heading = QLabel('选区检查' if compact else '当前选区几何检查')
        heading.setProperty('role', 'panelTitle'); layout.addWidget(heading)
        heading.setObjectName('regionQualityHeading')
        heading.setVisible(not compact)
        self.summary = QLabel('尚未加载数据'); self.summary.setWordWrap(True); layout.addWidget(self.summary)
        self.summary.setObjectName('regionQualitySummary')
        self.legend = QLabel('绿：一致　紫：待检查　橙：缺失 / 冲突\n蓝：选区　红：历史禁区（扣除）' if compact else
                            '蓝：选区　绿：网格与深度一致\n紫：待检查　橙：缺失 / 冲突\n红：历史禁区（扣除）')
        self.legend.setWordWrap(True); self.legend.setProperty('role', 'secondary'); layout.addWidget(self.legend)
        self.legend.setVisible(not compact)
        if not compact:self.legend.setMinimumHeight(self.legend.fontMetrics().lineSpacing()*3+6)
        row = QHBoxLayout()
        self.focus = QPushButton('定位待检查处' if compact else '定位待检查范围'); self.focus.clicked.connect(self.focus_pending); row.addWidget(self.focus)
        self.details = QPushButton('详情' if compact else '检查详情'); self.details.clicked.connect(self.show_details); row.addWidget(self.details)
        layout.addLayout(row)
        self.rescan = QPushButton('重新采集' if compact else '在此任务下重新采集'); self.rescan.clicked.connect(self.rescan_requested); layout.addWidget(self.rescan)
        for button,icon in ((self.focus,'target'),(self.details,'info'),(self.rescan,'refresh')):
            button.setIcon(line_icon(icon));button.setIconSize(QSize(18,18));button.setProperty('regionControl',True)
        self._render()

    def invalidate(self, message='标记已更新，几何支持检查中…'):
        self._generation += 1; self.runner.cancel(); self.timer.stop()
        self.result = None; self.canvas.set_quality_overlay(None)
        self._message=message
        self.summary.setText(message); self.summary.setToolTip('');self._render(); self.changed.emit()

    def request(self):
        self.invalidate()
        if self.canvas.model is not None and not self._closing: self.timer.start()
        elif self.canvas.model is None:
            self._message='尚未加载可检查的数据。';self.summary.setText(self._message);self._render()

    def _start(self):
        if self._closing or self.canvas.model is None: return
        if self.runner.busy:
            self.timer.start(); return
        token = self._generation
        model = self.canvas.model
        revision = self._input_revision()
        snapshot = clone_model(model, freeze_inputs=True)
        work, forbidden = self.canvas.work.copy(), self.canvas.forbidden.copy()
        def done(result):
            if token == self._generation and model is self.canvas.model and not self._closing and not self.runner._cancel.is_set():
                self.install(result)
        def failed(error):
            if token == self._generation and not self._closing:
                self._message='检查未完成：'+str(error)+'。调整选区可重试。'
                self.summary.setText(self._message)
                self._render(); self.changed.emit()
        self.runner.start(lambda cancel, progress: assess_selection(snapshot, work, forbidden, cancel), done, failed,
                          revision=revision, current_revision=self._input_revision)

    def _input_revision(self):
        model = self.canvas.model
        if model is None:return (None, self._generation)
        return (id(model), self._generation, capture(model, None, {})['sha256'],
                array_record(self.canvas.work)['sha256'], array_record(self.canvas.forbidden)['sha256'])

    def install(self, result):
        self.result = result
        self._message=result['message']
        self.canvas.set_quality_overlay(result.get('rgba'))
        if result['status'] == 'checked':
            self.summary.setText(f"网格与参考深度一致：{result['ratio']:.1%}\n"
                f"选区 {result['selected']} 像素 · 待检查 {result['pending']} 像素")
            self.summary.setToolTip(result['message']+'\n'+LIMITS)
        else: self.summary.setText(result['message'])
        self._render(); self.changed.emit()

    def _render(self):
        checked = self.result is not None and self.result['status'] == 'checked'
        if self._compact:
            result=self.result or {}
            known_selection=self.result is not None and result.get('status') in ('checked','empty')
            values=(f"{result['ratio']:.1%}" if checked else '—',
                    str(result['selected']) if known_selection else '—',
                    str(result['pending']) if checked else '—')
            colors=('#06d6a0' if checked else '#7895a9',
                    '#75b8ea' if known_selection else '#7895a9',
                    '#f4b942' if checked and result['pending'] else '#06d6a0' if checked else '#7895a9')
            cells=[]
            for value,color,title in zip(values,colors,('一致比例','选区像素','待检查像素')):
                cells.append('<td align="left" width="33%">'
                    f'<span style="color:{color};font-size:11px">●</span> '
                    f'<span style="color:#eef8ff;font-size:15px;font-weight:700">{value}</span><br>'
                    f'<span style="color:#83abc5;font-size:10px">{title}</span></td>')
            self.summary.setText('<table width="100%" cellspacing="3" cellpadding="0"><tr>'+''.join(cells)+'</tr></table>')
            self.summary.setToolTip(self._message+'\n'+self.legend.text()+'\n'+LIMITS)
            self.summary.setAccessibleName('选区几何检查：'+self._message)
            self.summary.setAccessibleDescription('，'.join(f'{name} {value}' for name,value in
                zip(('一致比例','选区像素','待检查像素'),values)))
            self.summary.setMinimumHeight(43)
        self.focus.setEnabled(bool(checked and self.result['pending']))
        self.details.setEnabled(self.result is not None)

    def focus_pending(self):
        if self.result is None or self.result['status'] != 'checked': return
        classes = self.result['classes']
        self.focus_requested.emit()
        self.canvas.fit_mask(classes > 1)

    def show_details(self):
        if self.result is None: return
        lines = [self.result['message'], LIMITS, '', '当前加载来源：'+str(getattr(self.canvas.model, 'reconstruction', '未知'))]
        lines.extend(f'{LABELS[code]}：{count} 像素' for code, count in self.result['counts'].items())
        lines.append('\n这是当前数据的只读检查，不替换历史保存结论，也不会补点或生成患者几何。')
        show_text(self, '选区几何检查详情', '\n'.join(lines))

    def shutdown(self):
        self._closing = True; self._generation += 1; self.timer.stop()
        return self.runner.shutdown(200)
