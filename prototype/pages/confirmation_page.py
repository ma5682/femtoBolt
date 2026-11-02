from widgets.status_panel import StatusPanel


class ConfirmationPage(StatusPanel):
    def __init__(self):
        super().__init__()
        self.section("规划确认")
        self.patient = self.row("患者")
        self.task = self.row("目标区域")
        self.model = self.row("采集模型")
        self.execution = self.row("可执行性")
        self.divider()
        self.section("规划摘要")
        self.spacing = self.row("最终行距")
        self.source = self.row("行距来源")
        self.width = self.row("有效作用宽度")
        self.overlap = self.row("重叠率")
        self.length = self.row("轨迹长度")
        self.version = self.row("规划版本")
        self.time = self.row("生成时间")
        self.divider()
        self.section("检查结果")
        self.checks = self.note("", "checkSummary")
        self.warning = self.issue("未覆盖区域需核对  →", "edge")
        self.note("当前仅为规划预览。治疗深度、工具姿态、接触、碰撞和治疗头覆盖均未参与。", "panelHint")
        self.saved = self.note("", "savedNotice")
        self.finish()

    def update_state(self, s):
        p = s.plan
        spacing, _ = p.parameters.spacing()
        self.value(self.patient, f"{s.patient.name} · {s.patient.identifier}")
        self.value(self.task, "规划区（目标区域）")
        self.value(self.model, f"本次采集 · {s.capture.model_revision:02d}")
        self.value(self.execution, "不可执行 · 规划预览", "bad" if s.replay_mode or not s.plan_current else "")
        self.value(self.spacing, f"{spacing:.2f} mm" if spacing is not None else "—", "good")
        self.value(self.source, "原始记录 · 直接指定" if s.replay_mode else "直接指定" if p.parameters.mode == "direct" else "按宽度计算")
        self.value(self.width, "未配置 · 不用于覆盖率" if s.replay_mode else f"{p.parameters.effective_width} mm · 占位")
        self.value(self.overlap, "未提供" if s.replay_mode else f"{p.parameters.overlap}% · 占位")
        if s.replay_mode:
            total = s.replay_stats.get("total_length_m")
            self.value(self.length, f"{float(total):.2f} m · 原始记录" if total is not None else "—")
        else:
            self.value(self.length, "1.8 m · 模拟值")
        self.value(self.version, f"预览 {p.revision:02d}")
        timestamp = p.generated_at.replace("T", " ")[5:] or "—"
        self.value(self.time, f"{timestamp} · 文件时间" if s.replay_mode and timestamp != "—" else timestamp)
        if s.replay_mode:
            stats = s.replay_stats
            gaps = int(stats.get("gap_count", 0))
            coverage = str(stats.get("coverage_status", "未提供"))
            self.warning.setText(f"存在 {gaps} 个路线缺口，查看轨迹端点  →" if gaps else "查看边缘未覆盖位置  →")
            self.warning.setVisible(bool(gaps or coverage == "partial"))
            self.checks.setText(
                "● 规划区与三维支持记录已载入\n"
                f"• 覆盖状态：{coverage}\n"
                f"• 扫描缺口：{gaps} 个；路线仅供只读核对"
            )
        else:
            self.warning.setText("边缘未覆盖区域需核对  →")
            self.warning.setVisible(True)
            self.checks.setText("● 规划区与三维支持已检查\n● 禁入区未进入（模拟）\n● 扫描段连续（模拟）" if s.plan_current else "当前规划需要重新检查。")
        self.saved.setText(
            "项目回放为只读记录，不保存、不覆盖原始数据。"
            if s.replay_mode else
            "● 本版本已保存，可在历史中查看。" if p.saved_id else
            "保存后保留当前模型来源、区域、参数与检查结果。"
        )
