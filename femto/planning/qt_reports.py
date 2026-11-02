"""Interactive Qt coverage, turn and surface-spacing inspection reports."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QTabWidget, QTextBrowser, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from femto.planning.qt_report_widgets import ImageView, ReportDialog


def _tree(headers):
    tree = QTreeWidget()
    tree.setHeaderLabels(headers)
    tree.setRootIsDecorated(False)
    tree.setSelectionMode(QAbstractItemView.SingleSelection)
    tree.setAlternatingRowColors(True)
    return tree


def _locatable_report(parent, title, text, headers, rows, values, locate, is_current):
    window = ReportDialog(parent, title, is_current)
    window.resize(1000, 730)
    layout = QVBoxLayout(window)
    content = QTextBrowser()
    content.setPlainText(text)
    layout.addWidget(content, 1)
    layout.addWidget(QLabel('点击条目，暂停播放并定位；定位不是已规划跨链运动。'))
    tree = _tree(headers)
    for row in rows:
        item = QTreeWidgetItem(values(row))
        item.setData(0, Qt.UserRole, row)
        tree.addTopLevelItem(item)
    for column in range(len(headers)):
        tree.resizeColumnToContents(column)
    layout.addWidget(tree, 1)

    def selected():
        if window.check_current() and tree.currentItem() is not None:
            locate(tree.currentItem().data(0, Qt.UserRole))

    tree.itemSelectionChanged.connect(selected)
    window.show()
    return window, tree


def quality_report(parent, plan, locate, is_current):
    from femto.planning.report_catalog import quality_text
    window, tree = _locatable_report(
        parent, '路线质量与急转角', quality_text(plan),
        ['路线链', '采样点', '方向变化 / °'], plan['route_quality']['sharp_turns'],
        lambda row: [str(row['chain_index'] + 1), str(row['sample_index'] + 1),
                     f"{row['deflection_degrees']:.2f}"], locate, is_current)
    window.turn_tree = tree
    return window


def spacing_report(parent, plan, locate, is_current):
    from femto.planning.report_catalog import spacing_text
    labels = {'within_bounds': '采样界内', 'outside_bounds': '超差',
              'uncertain_bounds': '无法证实', 'unmeasurable': '无可达前层'}

    def values(row):
        distance, deviation = row['surface_graph_distance_mm'], row['signed_deviation_mm']
        return [str(row['chain_index'] + 1), str(row['sample_index'] + 1),
                '未知' if distance is None else f'{distance:.3f}',
                '未知' if deviation is None else f'{deviation:+.3f}', labels[row['status']]]

    window, tree = _locatable_report(
        parent, '表面间距分布与定位', spacing_text(plan),
        ['链', '点', '网格上界 mm', '上界偏差 mm', '上下界判断'],
        plan['spacing_report']['measurements'], values, locate, is_current)
    window.measurement_tree = tree
    return window


class CoverageInspector(ReportDialog):
    def __init__(self, parent, model, plan, is_current):
        from femto.planning.coverage_inspector import LAYER_NAMES
        from femto.planning.report_catalog import coverage_report_text
        super().__init__(parent, '覆盖检查报告 · 遗漏定位', is_current)
        self.resize(1280, 850)
        self.model, self.plan = model, plan
        self.report = plan.get('coverage_inspection')
        self.selected_omission = None
        self.layer_checks = {}
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        method = QTextBrowser()
        method.setPlainText(coverage_report_text(plan))
        if self.report is None:
            tabs.addTab(method, '旧版报告（无遗漏明细）')
            return
        self.plan_binding = plan['plan_binding']
        self.lookup = {entry['id']: entry for entry in self.report['omissions']}
        view = QWidget()
        tabs.addTab(view, '图层与遗漏定位')
        tabs.addTab(method, '方法、分母与版本')
        main = QVBoxLayout(view)
        summary = QLabel(
            f"采样覆盖 {plan['coverage']['sampled_coverage_percent']:.2f}% / 分母 "
            f"{self.report['denominator']['sampled_allowed_supported_area_m2'] * 1e6:.2f} mm²"
            '（允许、可信采样代表面积）；禁区/无支持不计入，预留计入；无几何面积未知。')
        summary.setWordWrap(True)
        main.addWidget(summary)
        toggles = QGridLayout()
        main.addLayout(toggles)
        for index, (key, name) in enumerate(LAYER_NAMES.items()):
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            self.layer_checks[key] = checkbox
            checkbox.toggled.connect(self.redraw)
            toggles.addWidget(checkbox, index // 4, index % 4)
        overview = QPushButton('返回总览')
        overview.clicked.connect(self.overview)
        toggles.addWidget(overview, 0, 4, 2, 1)
        body = QSplitter(Qt.Horizontal)
        main.addWidget(body, 1)
        self.omission_tree = _tree(['编号', '原因', '采样代表 mm² / 像素'])
        body.addWidget(self.omission_tree)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.feedback = QLabel('选择条目以高亮并放大；无几何条目只有像素位置，不推算XYZ或面积。')
        self.feedback.setWordWrap(True)
        right_layout.addWidget(self.feedback)
        self.image_view = ImageView()
        right_layout.addWidget(self.image_view, 1)
        body.addWidget(right)
        body.setSizes([450, 780])
        for entry in self.report['omissions']:
            amount = (f"{entry['sampled_area_m2'] * 1e6:.3f} mm²"
                      if entry['sampled_area_m2'] is not None
                      else f"{entry['pixel_count']} px；面积未知")
            item = QTreeWidgetItem([entry['id'].split('_')[-1], entry['label'], amount])
            item.setData(0, Qt.UserRole, entry['id'])
            self.omission_tree.addTopLevelItem(item)
        self.omission_tree.itemSelectionChanged.connect(self._selected)
        self.redraw()

    def _selected(self):
        item = self.omission_tree.currentItem()
        if item is not None and item.isSelected():
            self.locate_omission(item.data(0, Qt.UserRole))

    def redraw(self):
        from femto.planning.coverage_inspector import crop_bounds, render_map
        if not self.check_current() or self.report is None:
            return
        entry = self.selected_omission
        image = render_map(self.model.rgb, self.plan,
                           [key for key, box in self.layer_checks.items() if box.isChecked()],
                           entry, zoom=entry is not None, header=False)
        self.image_view.set_image(image)
        self.focus_bbox = crop_bounds(self.report, entry)

    def locate_omission(self, entry_id):
        if not self.check_current():
            return
        entry = self.selected_omission = self.lookup[entry_id]
        position = ('XYZ（米）' + str(entry['focus_xyz_m'])
                    if entry['focus_xyz_m'] is not None else 'XYZ与三维面积未知')
        self.feedback.setText(entry['label'] + '：' + entry['evidence'] +
                              '\n像素范围 ' + str(entry['bbox_pixel']) + '；' + position)
        self.redraw()

    def overview(self):
        self.selected_omission = None
        self.omission_tree.clearSelection()
        self.feedback.setText('总览：面积统计只采用报告中可参与分母的采样点。')
        self.redraw()
