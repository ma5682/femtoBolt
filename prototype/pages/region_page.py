from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton
from widgets.status_panel import StatusPanel


class RegionPage(StatusPanel):
    example_requested = Signal()

    def __init__(self):
        super().__init__()
        self.section("当前标记")
        self.work = self.row("规划区")
        self.forbidden = self.row("禁入区")
        self.note("绿色：目标规划区域\n红色：探头或路线不得进入的区域", "panelHint")
        self.divider()
        self.section("三维支持检查")
        self.support = self.row("三维支持")
        self.issue_button = self.issue("定位缺失三维位置  →", "geometry")
        self.hint = self.note("", "panelHint")
        self.divider()
        self.section("标记操作")
        self.note("在画面中按住左键涂抹。擦除工具删除命中的一笔；撤销可恢复上一笔。")
        self.example = QPushButton("载入示例选区")
        self.example.setObjectName("quietButton")
        self.example.setMinimumHeight(44)
        self.example.clicked.connect(self.example_requested.emit)
        self.column.addWidget(self.example)
        self.example_note = self.note("示例用于演示操作；可继续添加或修改。")
        self.finish()

    def update_state(self, s):
        r = s.region
        self.value(self.work, "● 已标记" if r.has_work else "待标记", "good" if r.has_work else "")
        self.value(self.forbidden, "已标记" if r.has_forbidden else "尚未标记")
        names = {
            "unchecked": "待检查", "checking": "检查中…", "valid": "● 已确认",
            "partial": "⚠ 部分未确认", "missing": "阻断 · 局部缺失",
        }
        self.value(self.support, names.get(r.support, "待检查"), "bad" if r.support in ("missing", "partial") else "good" if r.support == "valid" else "")
        self.issue_button.setVisible(r.support == "missing")
        unconfirmed = int(s.replay_stats.get("geometry_unconfirmed_pixel_count", 0) or 0)
        ratio = s.replay_stats.get("geometry_support_ratio")
        ratio_text = f"几何像素支持约 {float(ratio) * 100:.1f}%；" if ratio is not None else ""
        self.hint.setText(
            "该区域缺少可靠三维数据，暂时无法生成轨迹。补采后会保留已画区域。"
            if r.support == "missing" else
            f"{ratio_text}部分像素缺少深度参考或存在冲突（约 {unconfirmed} 个）；当前仅可作为规划预览，不能视为治疗覆盖已确认。"
            if r.support == "partial" else
            "生成轨迹前会自动检查选区的三维支持。"
        )
        self.example.setVisible(not s.replay_mode)
        self.example_note.setVisible(not s.replay_mode)
        self.example.setEnabled(not s.busy and not s.replay_mode)
