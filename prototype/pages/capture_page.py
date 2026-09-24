from PySide6.QtWidgets import QProgressBar
from widgets.status_panel import StatusPanel


class CapturePage(StatusPanel):
    """Compact Inspector for acquisition, quality and spatial data."""

    def __init__(self):
        super().__init__()
        self.section("采集信息")
        self.phase = self.row("采集状态")
        self.resolution = self.row("相机分辨率")
        self.fps = self.row("帧率 FPS")
        self.capture_time = self.row("采集时间")
        self.frames = self.row("有效视角")
        self.divider()

        self.section("数据质量")
        self.coverage = self.row("覆盖率")
        self.coverage_bar = self.metric_bar()
        self.stability = self.row("有效深度")
        self.progress = self.metric_bar()
        self.views_bar = self.metric_bar()
        self.divider()

        self.section("空间数据")
        self.depth = self.row("深度数据")
        self.pose = self.row("相机位姿")
        self.reference = self.row("空间基准")
        self.surface = self.row("表面模型")
        self.problem = self.issue("查看缺失位置  →", "capture")
        self.hint = self.note("", "panelHint")
        self.result_note = self.note("", "panelHint")
        self.hint.setVisible(False)
        self.result_note.setVisible(False)
        self.finish()

    def metric_bar(self):
        bar = QProgressBar()
        bar.setObjectName("inspectorProgress")
        bar.setRange(0, 100)
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        self.column.addWidget(bar)
        return bar

    @staticmethod
    def _percent(value, default=0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return value * 100 if 0 <= value <= 1 else value

    def update_state(self, s):
        c = s.capture
        names = {"idle": "待开始", "capturing": "采集中", "checking": "检查中", "passed": "已就绪", "failed": "需补采"}
        self.value(self.phase, names[c.phase], "bad" if c.phase == "failed" else "good" if c.phase == "passed" else "")
        self.value(self.resolution, "1920 × 1080")
        self.value(self.fps, "30")
        if s.replay_mode:
            created = str(s.replay_stats.get("created_local", ""))
            self.value(self.capture_time, created.replace("T", " ")[:16] or "—")
        else:
            self.value(self.capture_time, "演示采集")
        self.value(self.frames, f"{c.frames} / 3" if c.frames else "0 / 3", "good" if c.frames >= 3 else "")
        self.views_bar.setValue(min(100, int(c.frames / 3 * 100)))

        if s.replay_mode:
            coverage = self._percent(s.replay_stats.get("geometry_support_ratio"), 96)
            depth = self._percent(s.replay_stats.get("valid_depth_ratio_mean"), 75)
        else:
            coverage = c.support
            depth = c.stability if c.reference_ready else 0
        coverage = max(0, min(100, int(coverage)))
        depth = max(0, min(100, int(depth)))
        self.value(self.coverage, f"{coverage}%")
        self.coverage_bar.setValue(coverage)
        self.value(self.stability, f"{depth}%", "warning" if depth < 90 and s.replay_mode else "")
        self.progress.setValue(depth)

        checked = c.phase in ("passed", "failed")
        if s.replay_mode:
            self.value(self.depth, "已获取", "good")
            self.value(self.pose, "已建立", "good")
            self.value(self.reference, "已对齐", "good")
            self.value(self.surface, "仅供校对", "warning")
        else:
            status = "已获取" if checked else "待检查"
            status_tone = "good" if checked else ""
            self.value(self.depth, status, status_tone)
            self.value(self.pose, "已建立" if checked else "待建立", status_tone)
            self.value(self.reference, "已对齐" if c.reference_ready else "待建立", "good" if c.reference_ready else "")
            self.value(self.surface, "已生成" if checked else "待检查", status_tone)
        self.problem.setVisible(c.phase == "failed")
