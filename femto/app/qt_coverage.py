"""Shared coverage summaries, mutually exclusive overlay and reason navigation."""
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QComboBox, QSizePolicy

from femto.planning.coverage_workflow import KEY, summaries
from femto.planning.coverage_display import NAMES
from femto.app.qt_common import breakable_text
from femto.app.qt_icons import line_icon


class CoveragePanel(QWidget):
    verify_requested = Signal()
    refine_requested = Signal()
    restore_requested = Signal()
    strict_requested = Signal()
    ordinary_requested = Signal()
    variable_requested = Signal()
    variable_restore_requested = Signal()
    layer_changed = Signal(bool, int)
    focus_requested = Signal(int)
    task_requested = Signal(int)
    tasks_export_requested = Signal()

    def __init__(self, parent=None, *, readonly=False, compact=False):
        super().__init__(parent)
        self._display = None
        self._key = None
        self._available = False
        self._readonly = readonly
        self._compact = compact
        self._algorithm = None
        layout = QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(7)
        if compact:
            self.setObjectName('routeCoverageCard')
            self.setAttribute(Qt.WA_StyledBackground, True)
            layout.setContentsMargins(10,10,10,10)
            heading=QLabel('覆盖核验');heading.setObjectName('regionQualityHeading');layout.addWidget(heading)
        self.labels = []
        for title in ('路径', '原目标几何', '模型覆盖'):
            label = QLabel(title);label.setWordWrap(True);label.setMinimumWidth(0)
            layout.addWidget(label);self.labels.append(label)
        if compact:
            for label in self.labels[:2]:label.hide()
            self.labels[2].setProperty('role','secondary')
        self.verify = QPushButton('核验模型覆盖');self.verify.clicked.connect(self.verify_requested)
        self.verify.setToolTip('保留当前路线，以名义作用宽度核验原目标；计算期间可取消。')
        actions = QHBoxLayout();self._actions = actions;actions.addWidget(self.verify)
        self.refine = QPushButton('补充规划');self.refine.clicked.connect(self.refine_requested)
        self.refine.setToolTip('先筛查工作区域与参考质量，再检查每条新增边；保留原确认范围且有新增覆盖时才采用。未通过原因见详情。')
        actions.addWidget(self.refine)
        self.restore = QPushButton('恢复补充前路线');self.restore.clicked.connect(self.restore_requested)
        actions.addWidget(self.restore);layout.addLayout(actions)
        strict_actions = QHBoxLayout();self._strict_actions = strict_actions;self._strict_inline = None
        self.strict = QPushButton('严格重建路径');self.strict.clicked.connect(self.strict_requested)
        self.strict.setToolTip('从当前常规路线构造新的表面路径并重新检查全部约束；保留原路线，完成后需重新核验覆盖。')
        strict_actions.addWidget(self.strict)
        self.ordinary = QPushButton('恢复常规路线');self.ordinary.clicked.connect(self.ordinary_requested)
        self.ordinary.setToolTip('恢复严格重建前的常规路线；当前覆盖失效，需重新核验后保存为新版本。')
        strict_actions.addWidget(self.ordinary);layout.addLayout(strict_actions)
        variable_actions = QHBoxLayout()
        self.variable = QPushButton('变宽作用模型');self.variable.clicked.connect(self.variable_requested)
        self.variable.setToolTip('在严格路线中明确半宽上下界与插值规则；应用后原覆盖失效，须重新核验。')
        self.variable_restore = QPushButton('恢复应用模型前路线');self.variable_restore.clicked.connect(self.variable_restore_requested)
        variable_actions.addWidget(self.variable);variable_actions.addWidget(self.variable_restore);layout.addLayout(variable_actions)
        self.profile_summary = QLabel();self.profile_summary.setWordWrap(True);self.profile_summary.setMinimumWidth(0)
        self.profile_summary.setTextFormat(Qt.PlainText)
        self.profile_summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.profile_summary)
        self.show_layer = QCheckBox('显示模型覆盖');layout.addWidget(self.show_layer)
        self.show_layer.setToolTip('参考图为像素汇总，三维为证明叶单元的显示近似。固定宽度自动模式叠加黄色中心线，便于对照覆盖边缘；未确认状态不等同于已证明漏扫。')
        row = QHBoxLayout();self.reason = QComboBox()
        for code, name in NAMES.items():self.reason.addItem(name, code)
        row.addWidget(self.reason, 1)
        self.focus = QPushButton('定位');row.addWidget(self.focus);layout.addLayout(row)
        self.pixel_reason = QComboBox()
        self.pixel_reason.setToolTip('按原目标台账的像素原因筛选参考图；不改变三维证据。')
        layout.addWidget(self.pixel_reason)
        self.task_summary = QLabel();self.task_summary.setWordWrap(True)
        self.task_summary.setTextFormat(Qt.PlainText);self.task_summary.setMinimumWidth(0)
        layout.addWidget(self.task_summary)
        task_actions = QHBoxLayout()
        self.task_action = QPushButton('建议下一步');task_actions.addWidget(self.task_action)
        self.tasks_export = QPushButton('导出后续任务');task_actions.addWidget(self.tasks_export)
        layout.addLayout(task_actions)
        self.task_action.clicked.connect(lambda:self.task_requested.emit(self.reason.currentData()))
        self.tasks_export.clicked.connect(self.tasks_export_requested)
        self.legend = QLabel('绿：已确认 · 橙：未确认\n紫：几何待核验 · 灰：预算内未处理\n参考图绿色要求整个像素内的目标都已确认。')
        self.legend.setWordWrap(True);self.legend.setProperty('role','secondary');layout.addWidget(self.legend)
        self.show_layer.toggled.connect(self._layer)
        self.reason.currentIndexChanged.connect(self._layer)
        self.reason.currentIndexChanged.connect(self._task_state)
        self.pixel_reason.currentIndexChanged.connect(self._task_state)
        self.pixel_reason.currentIndexChanged.connect(self._layer)
        self.focus.clicked.connect(lambda:self.focus_requested.emit(self.reason.currentData()))
        if compact:
            for button,icon in ((self.verify,'shield'),(self.strict,'refresh'),(self.ordinary,'undo'),
                                (self.refine,'sliders'),(self.restore,'undo'),(self.focus,'target')):
                button.setIcon(line_icon(icon));button.setProperty('regionControl',True)
        self.set_context(None, None)

    @property
    def display(self):return self._display

    def set_context(self, model, plan, *, ready=False, busy=False):
        from femto.planning.target_surface import is_deferred
        deferred=is_deferred(((plan or {}).get('planning_context') or {}).get('target'))
        for label, value in zip(self.labels, summaries(plan)):label.setText(value)
        if plan and not ready and not busy:
            self.labels[2].setText('模型覆盖：当前不可使用；请先核对输入与路线。')
        metadata = (plan or {}).get(KEY)
        if self._compact:self.labels[2].setVisible(metadata is not None or deferred)
        if deferred and ready:self.labels[2].setText('面积与覆盖尚未计算，可按需检查。')
        algorithm = (plan or {}).get('surface_trajectory', {}).get('planner', {}).get('algorithm_version')
        self._algorithm = algorithm
        refined = algorithm in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-refined-surface-v1')
        constructed = algorithm == 'constructed-surface-route-v1'
        variable_refined = algorithm == 'nominal-variable-width-refined-surface-v1'
        variable = algorithm in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1')
        self.variable.setVisible(not self._readonly and (constructed or variable) and not variable_refined)
        self.variable.setText('更改变宽模型' if variable else '变宽作用模型')
        self.variable.setEnabled(bool(not busy and (ready or variable) and model is not None))
        self.variable_restore.setVisible(not self._readonly and variable and not variable_refined)
        self.variable_restore.setEnabled(bool(not busy and model is not None))
        self.profile_summary.setVisible(variable)
        if variable:
            from femto.planning.variable_width_route import summary_text
            self.profile_summary.setText(breakable_text(summary_text(plan['surface_trajectory'])))
        inline = bool(not metadata and not self._readonly and
                      (constructed or algorithm == 'quality-constrained-surface-v3' and (plan or {}).get('planning_context')))
        if inline != self._strict_inline:
            # Before coverage exists, keep the two usable operations together.
            # The unavailable refinement must not push restoration below the fold.
            row = self._actions if inline else self._strict_actions
            for button in (self.strict, self.ordinary):
                self._actions.removeWidget(button);self._strict_actions.removeWidget(button)
                row.addWidget(button)
            self._strict_inline = inline
        self.strict.setVisible(not self._readonly and algorithm == 'quality-constrained-surface-v3')
        self.ordinary.setVisible(not self._readonly and constructed)
        self.strict.setEnabled(bool(ready and not busy and (plan or {}).get('planning_context')))
        self.ordinary.setEnabled(bool(ready and not busy))
        self.refine.setText('变宽补充规划' if variable else '严格补充规划' if constructed else '补充规划')
        self.verify.setVisible(not self._readonly and not refined)
        self.refine.setVisible(not self._readonly and not refined and not inline and (not variable or bool(metadata)))
        self.restore.setVisible(not self._readonly and refined)
        self.refine.setEnabled(bool(ready and not busy and metadata))
        expired=False
        if variable_refined:
            from femto.planning.variable_footprint import require_current
            try:require_current(plan['surface_trajectory']['planner']['parameters']['variable_footprint'])
            except ValueError:expired=True
        self.restore.setEnabled(bool((ready or expired) and not busy and metadata))
        self.show_layer.setVisible(metadata is not None)
        self.reason.setVisible(metadata is not None);self.focus.setVisible(metadata is not None)
        display = getattr(model, 'coverage_display', None)
        if not (metadata and display and display.certificate_sha256 == metadata.get('sha256')):display = None
        key = (id(model), None if metadata is None else metadata['sha256'])
        changed = key != self._key
        if changed:
            self._key = key;self._display = display
            self.show_layer.setChecked(False)
            self.reason.setCurrentIndex(0)
        self._display = display
        self._pixel_reasons(changed)
        self.verify.setText('计算面积与覆盖' if deferred else '重新核验模型覆盖' if metadata else '核验模型覆盖')
        self.verify.setEnabled(bool(ready and not busy and (plan or {}).get('planning_context')))
        available = bool(ready and not busy and display is not None)
        availability_changed = available != self._available
        self._available = available
        self.show_layer.setEnabled(available);self.reason.setEnabled(available)
        self.focus.setEnabled(available)
        self._task_state()
        self.legend.setVisible(bool(available and self.show_layer.isChecked()))
        if changed or availability_changed:self._layer()

    def _task_state(self, *_):
        from femto.planning.coverage_tasks import operation
        tasks = None if self._display is None else self._display.tasks
        visible = tasks is not None
        self.pixel_reason.setVisible(visible and self.reason.currentData() == 8)
        self.pixel_reason.setEnabled(bool(visible and self._available))
        self.task_summary.setVisible(visible)
        self.task_action.setVisible(visible and not self._readonly)
        self.tasks_export.setVisible(visible and not self._readonly)
        self.tasks_export.setEnabled(bool(visible and self._available and not self._readonly))
        self.task_action.setEnabled(False)
        if not visible:return
        reason = self.reason.currentData()
        task = tasks.task(reason)
        if task is None:
            pending = [item for item in tasks.items if item.reason != 1 and item.present]
            self.task_summary.setText(f'后续任务：{len(pending)} 类待处理原因。选择具体原因，查看影响范围和建议操作。'
                                      if pending else '后续任务：未列出可定位的待处理范围；原目标完成状态请核对覆盖摘要。')
            self.task_action.setText('选择具体原因后处理');return
        action, label = operation(task, self._algorithm)
        self.task_action.setText(label)
        self.task_action.setEnabled(bool(self._available and action is not None and not self._readonly))
        scope = (f'{task.pixel_count} 个像素含未定位范围，面积未知' if reason == 8 else
                 f'{task.nominal_area_m2*1e6:.1f} mm²，{task.leaf_count} 个目标叶单元，涉及 {task.pixel_count} 个参考像素')
        if not task.present:
            self.task_summary.setText('当前证据中没有这一类范围。');return
        text = scope+'。'+task.guidance
        if self.reference_reason is not None:
            text = '当前定位：'+self.pixel_reason.currentText()+'；面积未知。'+text
        if task.calculation_limits_reached:text += ' 本次计算达到限额，零未处理面积也不能替代完成状态核验。'
        if action == 'restore':text += ' 当前为补充版本，请恢复原路线与原覆盖后再处理。'
        if not self._available:text += ' 当前输入不可用或正在处理，任务暂不可执行。'
        self.task_summary.setText(text)

    @property
    def reference_reason(self):
        value = self.pixel_reason.currentData()
        return value if self.reason.currentData() == 8 and value else None

    def _pixel_reasons(self, reset):
        from femto.planning.coverage_tasks import PIXEL_REASON_LABELS
        from femto.planning.target_surface import PIXEL_STATES
        previous = None if reset else self.pixel_reason.currentData()
        self.pixel_reason.blockSignals(True)
        try:
            self.pixel_reason.clear();self.pixel_reason.addItem('全部未定位原因', 0)
            task = None if self._display is None else self._display.tasks.task(8)
            counts = {} if task is None else dict(task.unknown_by_reason)
            for code, name in PIXEL_STATES.items():
                count = counts.get(name, 0)
                if count:self.pixel_reason.addItem(f'{PIXEL_REASON_LABELS[name]}：{count} 像素', code)
            index = self.pixel_reason.findData(previous)
            self.pixel_reason.setCurrentIndex(max(0,index))
        finally:self.pixel_reason.blockSignals(False)

    def _layer(self, *_):
        active = self.show_layer.isChecked() and self.show_layer.isEnabled() and self._display is not None
        self.legend.setVisible(active)
        self.layer_changed.emit(active, self.reason.currentData())
