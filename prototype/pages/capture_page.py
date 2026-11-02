from widgets.status_panel import StatusPanel


class CapturePage(StatusPanel):
    """Acquisition Inspector with one prominent phase and factual data rows."""

    def __init__(self):
        super().__init__()
        self.section('当前状态')
        self.phase = self.note('等待建立参考位置', 'phaseLabel')
        self.hint = self.note('保持相机与目标稳定\n在下方开始模拟采集', 'panelHint')
        self.divider()
        self.section('采集信息')
        self.camera = self.row('相机状态')
        self.rgb = self.row('RGB 图像')
        self.depth = self.row('深度图像')
        self.surface = self.row('三维表面')
        self.reference = self.row('参考位置')
        self.pose = self.reference
        self.divider()
        self.section('当前采集')
        self.frames = self.row('有效视角帧数')
        self.frames.setObjectName('metricValue')
        self.minimum_frames = self.row('建议最少帧数')
        self.value(self.minimum_frames, '3 帧')
        self.problem = self.issue('查看待补采位置  →', 'capture')
        self.problem.hide()
        self.result_note = self.note('补采会保留参考位置。\n完成后检查三维表面质量。', 'panelHint')
        self.finish()

    def update_state(self, s):
        c = s.capture
        names = {'idle': '等待建立参考位置', 'capturing': '正在模拟采集', 'checking': '正在检查模型', 'passed': '三维模型已就绪', 'failed': '质量不足，需补采'}
        hints = {'idle': '保持相机与目标稳定\n在下方开始模拟采集', 'capturing': '多角度补充目标表面\n至少采集 3 个模拟视角', 'checking': '正在检查三维表面\n检查完成后可继续', 'passed': '模型检查已完成\n可继续标记目标区域', 'failed': '发现三维数据缺失\n请定位缺失处后继续补采'}
        phase_tone = 'bad' if c.phase == 'failed' else 'good' if c.phase == 'passed' else 'current' if c.phase in ('capturing', 'checking') else 'idle'
        self.value(self.phase, names.get(c.phase, '等待确认'), phase_tone)
        self.hint.setText(hints.get(c.phase, '请按底部提示继续操作'))
        self.value(self.frames, f'{c.frames} 帧', 'good' if c.frames >= 3 else 'idle')
        if s.replay_mode:
            self.value(self.phase, '查看已保存采集', 'current')
            self.hint.setText('当前为项目数据回放\n相机未连接，不执行采集')
            self.value(self.camera, '○ 未连接', 'idle')
            self.value(self.rgb, '● 回放数据', 'current')
            self.value(self.depth, '● 回放数据', 'current')
            self.value(self.reference, '● 已载入', 'current')
            self.value(self.surface, '仅供校对', 'warning')
            self.result_note.setText('仅显示已保存采集与检查结果。')
        else:
            active = c.phase != 'idle'
            self.value(self.camera, '● 模拟中' if active else '○ 未就绪', 'current' if active else 'idle')
            self.value(self.rgb, '● 模拟画面' if active else '○ 未获取', 'current' if active else 'idle')
            self.value(self.depth, '● 模拟画面' if active else '○ 未获取', 'current' if active else 'idle')
            self.value(self.reference, '● 模拟已建立' if c.reference_ready else '○ 未建立', 'good' if c.reference_ready else 'idle')
            surface = {'idle': '等待采集', 'capturing': '采集中', 'checking': '检查中', 'passed': '● 模拟已生成', 'failed': '● 模拟有缺失'}
            self.value(self.surface, surface.get(c.phase, '等待确认'), phase_tone)
            self.result_note.setText('补采会保留参考位置。\n完成后检查三维表面质量。')
        self.problem.setVisible(c.phase == 'failed')
