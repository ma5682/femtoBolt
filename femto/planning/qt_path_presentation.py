"""Qt route playback with explicit stops at every unplanned transition."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QSlider, QVBoxLayout,
)

from femto.planning.path_presentation import _block_state_label, render_diagram
from femto.planning.qt_report_widgets import ImageView, ReportDialog, text_report
from femto.planning.route_motion import MotionPlayer, statistics


class RoutePresentation(ReportDialog):
    def __init__(self, parent, model, region, plan, is_current):
        super().__init__(parent, '圈选区域 · 扫描路线图', is_current)
        self.resize(1100, 850)
        self.model, self.region, self.plan = model, region, plan
        self.blocks = plan.get('blocks')
        self.motion_player = MotionPlayer(plan)
        self._notice = ''
        self._selecting_block = False
        self._last_tick = 0.
        self.animation = QTimer(self)
        self.animation.setInterval(100)
        self.animation.timeout.connect(self.tick)
        main = QVBoxLayout(self)
        controls = QGridLayout()
        main.addLayout(controls)
        self.show_region = self._checkbox(controls, '显示区域', True, 0, 0)
        self.show_path = self._checkbox(controls, '显示路径', True, 0, 1)
        self.show_coverage = self._checkbox(controls, '覆盖检查', bool(plan.get('coverage')), 0, 2)
        self.show_coverage.setEnabled(bool(plan.get('coverage')))
        self.show_turns = self._checkbox(controls, '急转点', bool(plan.get('route_quality')), 0, 3)
        self.show_turns.setVisible(bool(plan.get('route_quality')))
        self.chain_selection = QComboBox()
        self.chain_selection.addItems(['全部路径段'] + [f'第 {i + 1} 段' for i in range(len(plan['segments']))])
        self.chain_selection.currentIndexChanged.connect(self.select_chain)
        controls.addWidget(self.chain_selection, 0, 4)
        export = QPushButton('导出路线图 PNG')
        export.clicked.connect(self.save)
        controls.addWidget(export, 0, 5)
        self.block_selection = QComboBox()
        self.block_selection.addItem('全部块')
        if self.blocks:
            self.block_selection.addItems([
                f"{item['label']}（{_block_state_label(item, bool(self.blocks.get('selection_blocks')))}）"
                for item in self.blocks['items']])
            self.block_selection.currentIndexChanged.connect(self.select_block)
            controls.addWidget(QLabel('块'), 1, 0)
            controls.addWidget(self.block_selection, 1, 1, 1, 5)
        playback = QHBoxLayout()
        main.addLayout(playback)
        for title, callback in [('播放', self.play), ('暂停', self.pause),
                                ('继续', self.play), ('复位', self.reset)]:
            button = QPushButton(title)
            button.clicked.connect(callback)
            playback.addWidget(button)
        playback.addWidget(QLabel('模拟速度 mm/s'))
        self.speed_box = QComboBox()
        self.speed_box.addItems(['5', '10', '20', '40'])
        self.speed_box.setCurrentText('20')
        self.speed_box.currentTextChanged.connect(self.change_speed)
        playback.addWidget(self.speed_box)
        playback.addStretch()
        report_controls = QHBoxLayout()
        main.addLayout(report_controls)
        report_controls.addStretch()
        for title, callback, enabled in [
            ('查看覆盖报告', self.coverage_report, True),
            ('表面间距报告', self.spacing_report, bool(plan.get('spacing_report'))),
            ('路线质量/急转角', self.quality_report, bool(plan.get('route_quality'))),
        ]:
            if enabled:
                button = QPushButton(title)
                button.clicked.connect(callback)
                report_controls.addWidget(button)
        self.image_view = ImageView()
        main.addWidget(self.image_view, 1)
        self.progress = QSlider(Qt.Horizontal)
        self.progress.setRange(0, 10000)
        self.progress.valueChanged.connect(lambda value: self.seek_progress(value / 100))
        main.addWidget(self.progress)
        self.state_label = QLabel()
        self.state_label.setWordWrap(True)
        self.state_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        main.addWidget(self.state_label)
        self.update_state()
        self.redraw()

    def _checkbox(self, layout, title, checked, row, column):
        box = QCheckBox(title)
        box.setChecked(checked)
        box.toggled.connect(self.redraw)
        layout.addWidget(box, row, column)
        return box

    @staticmethod
    def _set_index(combo, index):
        old = combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(old)

    def chosen_block(self):
        index = self.block_selection.currentIndex()
        return None if not self.blocks or index == 0 else self.blocks['items'][index - 1]

    def update_state(self):
        player = self.motion_player
        pos = self.motion_state = player.position()
        self.progress.blockSignals(True)
        self.progress.setValue(round(pos['progress'] * 10000))
        self.progress.blockSignals(False)
        block = self.chosen_block()
        if block is not None and not self._selecting_block and player.chain not in block['chain_indices']:
            self._set_index(self.block_selection, 0)
        xyz = ', '.join(f'{value:.6f}' for value in pos['xyz_m'])
        direction = ', '.join(f'{value:.3f}' for value in pos['direction_W0'])
        info = statistics(self.plan, player.speed)
        block_info = (f" · 块 {self.blocks['chain_blocks'][player.chain] + 1}/{self.blocks['summary']['block_count']}"
                      if self.blocks else '')
        gap = ''
        if player.at_break:
            breaks = self.plan.get('chain_breaks', [])
            reason = breaks[player.chain]['explanation'] if player.chain < len(breaks) else '不同链之间没有已规划连接'
            if player.chain < len(breaks) and breaks[player.chain].get('inter_block'):
                gap = ('块间停止（选择分离）：' if breaks[player.chain].get('reason') == 'selection_break'
                       else '块间停止：') + reason + '；点击继续仅定位下一块起点，不播放跨块运动。'
            else:
                gap = '断点停止：' + reason + '；点击继续仅定位下一链起点。'
        state = '播放结束' if player.finished else '播放中' if player.running else '已暂停'
        kind = '连接' if pos['motion_kind'] == 'connector' else '扫描'
        self.state_label.setText(
            f'{state} · 链 {player.chain + 1}/{len(player.xyz)}{block_info} · {kind} · '
            f'W0/RDF XYZ（米）({xyz}) · 方向单位向量 ({direction})\n'
            f"进度 {pos['progress'] * 100:.1f}% / 对应恒速时间 {pos['known_time_s']:.1f} s；"
            f"扫描 {info['scan_length_m'] * 1000:.1f} / 连接 {info['connector_length_m'] * 1000:.1f} / "
            f"总长 {info['total_length_m'] * 1000:.1f} mm；{player.speed:g} mm/s 已规划运动 "
            f"{info['known_motion_time_s']:.1f} s（仅链内，不含暂停、加减速或未知跨链移动）。\n"
            + gap + self._notice)

    def redraw(self):
        if not self.check_current():
            return
        index = self.chain_selection.currentIndex() - 1
        block = self.chosen_block()
        self.current_image = render_diagram(
            self.model.rgb, self.model.projection.ids, self.region, self.plan,
            self.show_region.isChecked(), self.show_path.isChecked(), None if index < 0 else index,
            self.show_coverage.isChecked(), self.motion_state['pixel'], self.show_turns.isChecked(),
            None if block is None else block['chain_indices'])
        self.image_view.set_image(self.current_image)

    def tick(self):
        if not self.check_current():
            return
        player = self.motion_player
        if not player.running:
            self.animation.stop()
            return
        now = time.monotonic()
        player.advance(now - self._last_tick)
        self._last_tick = now
        self.update_state()
        self.redraw()
        if not player.running:
            self.animation.stop()

    def play(self):
        if not self.check_current() or self.motion_player.running:
            return
        player = self.motion_player
        previous_chain = player.chain
        if player.play():
            breaks = self.plan.get('chain_breaks', [])
            inter = previous_chain < len(breaks) and breaks[previous_chain].get('inter_block')
            self._notice = ('已定位下一块起点；块间动作未规划，未播放任何跨块运动。' if inter
                            else '已定位下一链起点；跨链动作未规划。')
        else:
            self._notice = ''
        if self.chain_selection.currentIndex():
            self._set_index(self.chain_selection, player.chain + 1)
        self._last_tick = time.monotonic()
        self.update_state()
        self.redraw()
        self.animation.start()

    def pause(self):
        player = self.motion_player
        if player.running and self.animation.isActive():
            player.advance(max(0., time.monotonic() - self._last_tick))
        player.pause()
        self.animation.stop()
        self.update_state()
        self.redraw()

    def reset(self):
        if not self.check_current():
            return
        self.pause()
        self.motion_player.reset()
        self._notice = '已复位。'
        self._set_index(self.chain_selection, 0)
        self.update_state()
        self.redraw()

    def seek_progress(self, value):
        if not self.check_current():
            return
        self.pause()
        self.motion_player.seek(float(value) / 100)
        self._notice = '进度定位；该定位不是已规划运动。'
        if self.chain_selection.currentIndex():
            self._set_index(self.chain_selection, self.motion_player.chain + 1)
        self.update_state()
        self.redraw()

    def select_chain(self, index):
        if not self.check_current():
            return
        self.pause()
        if index > 0:
            self.motion_player.select_chain(index - 1)
        self._notice = '逐链定位；不表示跨链运动。'
        self.update_state()
        self.redraw()

    def select_block(self, _index=None):
        if not self.check_current():
            return
        self._selecting_block = True
        try:
            self.pause()
            block = self.chosen_block()
            if block is None:
                self._notice = '显示全部块；块间只有 gap，没有连线。'
            elif block['chain_indices']:
                self.motion_player.select_chain(block['chain_indices'][0])
                self._set_index(self.chain_selection, 0)
                self._notice = (f"仅显示 {block['label']}（访问顺序第 {block['block_index'] + 1}，"
                                f"{block['segment_count']} 段 / {block['point_count']} 点）；显示过滤不修改保存的轨迹。")
            else:
                self._set_index(self.chain_selection, 0)
                self._notice = f"{block['label']} 未规划（{block.get('reason')}），没有可显示或播放的扫描段；轨迹未覆盖该块。"
            self.update_state()
            self.redraw()
        finally:
            self._selecting_block = False

    def change_speed(self, value):
        if not self.check_current():
            return
        self.pause()
        self.motion_player.set_speed(float(value))
        self._notice = '速度已更新，点击继续。'
        self.update_state()
        self.redraw()

    def locate_spacing(self, row):
        if not self.check_current():
            return
        self.pause()
        location = row.get('player_location')
        if location is not None and location.get('status') != 'unique':
            self._notice = location['explanation']
        else:
            self._locate_point(location or row)
        self.update_state()
        self.redraw()

    def _locate_point(self, row):
        # Select the actual chain first: global progress is ambiguous at a chain end.
        player = self.motion_player
        player.select_chain(row['chain_index'])
        player.distance = float(player.distances[player.chain][row['sample_index']])
        self._set_index(self.chain_selection, 0)
        self._notice = '报告定位；该定位不是已规划运动。'

    def locate_turn(self, turn):
        if not self.check_current():
            return
        self.pause()
        self._locate_point(turn)
        self._notice = (f"急转点：链 {turn['chain_index'] + 1} / 点 {turn['sample_index'] + 1}，"
                        f"方向变化 {turn['deflection_degrees']:.2f}°；未平滑。")
        self.update_state()
        self.redraw()

    def coverage_report(self):
        if not self.check_current():
            return
        self.pause()
        if self.plan.get('surface_trajectory'):
            from femto.planning.surface_workspace import report_text
            text_report(self, '三维规划与覆盖报告',
                        report_text(self.plan, self.model.store.mesh, self.region), self.plan_is_current)
        else:
            from femto.planning.coverage_inspector import show_inspector
            show_inspector(self, self.model, self.plan, self.plan_is_current)

    def spacing_report(self):
        if self.check_current():
            self.pause()
            from femto.planning.surface_spacing_report import show_spacing_report
            show_spacing_report(self, self.plan, self.locate_spacing, self.plan_is_current)

    def quality_report(self):
        if self.check_current():
            self.pause()
            from femto.planning.route_quality import show_quality
            show_quality(self, self.plan, self.locate_turn, self.plan_is_current)

    def save(self):
        if not self.check_current():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '导出路线图 PNG',
            str(Path(getattr(self.model, 'reconstruction', '.')) / '区域扫描路线图.png'), 'PNG (*.png)')
        if path and self.check_current():
            try:
                if not Path(path).suffix:
                    path += '.png'
                self.current_image.save(path)
            except OSError as error:
                QMessageBox.critical(self, '导出失败', str(error))

    def stop(self):
        self.animation.stop()
        self.motion_player.pause()
        super().stop()

    def close_demo(self):
        self.close()
