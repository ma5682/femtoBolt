from PySide6.QtCore import Signal, QSignalBlocker
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLineEdit, QRadioButton, QToolButton, QWidget
from widgets.status_panel import StatusPanel, text_label, tone


class PlanningPage(StatusPanel):
    parameter_changed = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.section("扫描行距")
        self.modes = QButtonGroup(self)
        self.direct = QRadioButton("直接指定行距")
        self.width = QRadioButton("按有效作用宽度计算")
        self.modes.addButton(self.direct)
        self.modes.addButton(self.width)
        for button, mode in ((self.direct, "direct"), (self.width, "width")):
            button.setMinimumHeight(30)
            button.toggled.connect(lambda checked, m=mode: self.parameter_changed.emit("mode", m) if checked else None)
            self.column.addWidget(button)
        self.inputs = {}
        for key, name in (("direct_spacing", "行距 / mm"), ("effective_width", "有效作用宽度 / mm"), ("overlap", "重叠率 / %")):
            item = QWidget()
            row = QHBoxLayout(item)
            row.setContentsMargins(0, 2, 0, 2)
            row.addWidget(text_label(name, "rowName"), 1)
            field = QLineEdit()
            field.setObjectName("parameterInput")
            field.setFixedWidth(85)
            field.setMinimumHeight(36)
            field.textEdited.connect(lambda value, k=key: self.parameter_changed.emit(k, value))
            row.addWidget(field)
            self.column.addWidget(item)
            self.inputs[key] = field
        self.spacing = self.note("", "spacingValue")
        self.error = self.note("", "inlineError")
        self.note("演示占位值，非设备参数。宽度 × (1 − 重叠率)；行距范围 0.5–100 mm，重叠率 0–90%。")
        self.divider()
        self.section("规划检查")
        self.region = self.row("规划区")
        self.forbidden = self.row("禁入区")
        self.geometry = self.row("三维几何")
        self.continuity = self.row("扫描线")
        self.route_issue = self.issue("定位路线缺口  →", "route")
        self.edge_issue = self.issue("查看未覆盖边缘  →", "edge")
        self.note("连接段不计入扫描；没有真实有效作用宽度时，不计算治疗覆盖率。")
        self.unused = QToolButton()
        self.unused.setObjectName("detailsButton")
        self.unused.setText("＋ 待用参数")
        self.unused.setCheckable(True)
        self.unused.setMinimumHeight(40)
        self.column.addWidget(self.unused)
        self.unused_text = self.note("治疗深度：暂未参与规划\n外壳尺寸：暂未参与规划\n外壳避让余量：暂未参与规划\n这些参数仅记录，未来用于焦点层、接触与避让检查；不用于当前扫描行距。")
        self.unused_text.hide()
        self.unused.toggled.connect(self.unused_text.setVisible)
        self.finish()

    def update_state(self, s):
        p = s.plan
        for btn, checked in ((self.direct, p.parameters.mode == "direct"), (self.width, p.parameters.mode == "width")):
            with QSignalBlocker(btn):
                btn.setChecked(checked)
            btn.setEnabled(not s.busy and not s.replay_mode)
        for key, field in self.inputs.items():
            value = getattr(p.parameters, key)
            if field.text() != value:
                with QSignalBlocker(field):
                    field.setText(value)
            field.setEnabled(not s.busy and not s.replay_mode and ((key == "direct_spacing") == (p.parameters.mode == "direct")))
        spacing, error = p.parameters.spacing()
        self.spacing.setText("最终行距  —" if error else f"最终行距  {spacing:.2f} mm")
        tone(self.spacing, "bad" if error else "good")
        self.error.setText(error)
        self.error.setVisible(bool(error))
        ready = p.phase in ("ready", "failed")
        stats = s.replay_stats
        self.value(self.region, "● 原始规划记录" if s.replay_mode and ready else "● 已规划" if ready else "等待生成", "good" if ready else "")
        self.value(self.forbidden, "原始禁入区记录" if s.replay_mode and ready else "● 未进入 · 模拟" if ready else "等待检查", "good" if ready else "")
        if s.replay_mode and ready:
            geometry = stats.get("geometry_status", "已记录")
            geometry_label = "部分未确认" if "unconfirmed" in geometry else "已确认"
            self.value(self.geometry, geometry_label, "bad" if "unconfirmed" in geometry else "good")
            gaps = int(stats.get("gap_count", 0))
            continuity = f"局部断开 · {gaps} 个缺口" if gaps else "● 连续"
            self.value(self.continuity, continuity, "bad" if gaps else "good")
        else:
            self.value(self.geometry, "● 有效 · 模拟", "good")
            self.value(self.continuity, "局部断开" if p.phase == "failed" else "● 段内连续 · 模拟" if ready else "等待生成", "bad" if p.phase == "failed" else "good" if ready else "")
        self.route_issue.setVisible(p.phase == "failed")
        self.edge_issue.setVisible(ready and (not s.replay_mode or stats.get("coverage_status") == "partial"))
