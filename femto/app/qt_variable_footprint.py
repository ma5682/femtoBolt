"""Small P5 model editor; units and interpolation are explicit user inputs."""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QDialogButtonBox, QFileDialog, QComboBox)

from femto.planning import variable_footprint as footprint


class VariableFootprintDialog(QDialog):
    def __init__(self, width_mm: float, profile: dict | None = None, parent=None):
        super().__init__(parent)
        self.profile = None
        self.setWindowTitle('名义变宽作用模型');self.resize(610, 610)
        layout = QVBoxLayout(self)
        note = QLabel(f'节点位置按每条连续路线的长度百分比计算，包含连接段。\n'
                      f'当前路线生成半宽为 {width_mm/2:g} mm；模型上界超过它时需要重新生成路线。\n'
                      '下界用于确认覆盖，上界用于路径检查。设备偏移上界未知，当前仅核验名义几何模型。')
        note.setWordWrap(True);layout.addWidget(note)
        form = QFormLayout();self.name = QLineEdit();self.version = QLineEdit()
        self.start = QLineEdit();self.end = QLineEdit()
        self.start.setPlaceholderText('可留空；YYYY-MM-DD（UTC）');self.end.setPlaceholderText('可留空；YYYY-MM-DD（UTC）')
        self.rule = QComboBox();self.rule.addItem('节点间分段线性', 'piecewise_linear')
        self.rule.addItem('仅有离散样本，整段待核验', 'samples_only')
        for label, widget in (('来源名称', self.name), ('模型版本', self.version), ('插值规则', self.rule),
                              ('生效日期', self.start), ('截止日期', self.end)):form.addRow(label, widget)
        layout.addLayout(form)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['路线位置（%）', '半宽下界（mm）', '半宽上界（mm）'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        add = QPushButton('添加节点');add.clicked.connect(self.add_row);row.addWidget(add)
        remove = QPushButton('删除节点');remove.clicked.connect(self.remove_row);row.addWidget(remove)
        self.import_button = QPushButton('导入模型 JSON');self.import_button.clicked.connect(self.import_model)
        row.addWidget(self.import_button);layout.addLayout(row)
        self.error = QLabel();self.error.setWordWrap(True);self.error.setProperty('role', 'warning');layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('应用并检查')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('取消')
        buttons.accepted.connect(self.accept_model);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        if profile is None:
            profile = footprint.nominal([dict(position=t, lower_half_width_m=width_mm/2000, upper_half_width_m=width_mm/2000)
                                         for t in (0., .5, 1.)], name='研发名义模型', version='1')
        self.set_profile(profile)

    def set_profile(self, profile: dict) -> None:
        footprint.validate(profile)
        self.name.setText(profile['source']['name']);self.version.setText(profile['source']['version'])
        self.start.setText(profile['valid_from'] or '');self.end.setText(profile['valid_until'] or '')
        self.rule.setCurrentIndex(self.rule.findData(profile['interpolation']))
        self.table.setRowCount(len(profile['knots']))
        for i, knot in enumerate(profile['knots']):
            for j, value in enumerate((knot['position']*100, knot['lower_half_width_m']*1000, knot['upper_half_width_m']*1000)):
                self.table.setItem(i, j, QTableWidgetItem(format(value, '.17g')))
        self.error.clear()
        self._loaded_profile = deepcopy(profile)
        self._loaded_controls = self._controls()

    def _controls(self) -> tuple:
        return (self.name.text(), self.version.text(), self.rule.currentData(), self.start.text(), self.end.text(),
                tuple(tuple(None if self.table.item(i,j) is None else self.table.item(i,j).text() for j in range(3))
                      for i in range(self.table.rowCount())))

    def add_row(self) -> None:
        if self.table.rowCount() >= footprint.MAX_KNOTS:
            self.error.setText(f'最多 {footprint.MAX_KNOTS} 个节点。');return
        index = max(1, self.table.currentRow()+1)
        self.table.insertRow(index)
        for j in range(3):self.table.setItem(index, j, QTableWidgetItem(''))
        self.table.setCurrentCell(index, 0)

    def remove_row(self) -> None:
        if self.table.rowCount() > 2 and self.table.currentRow() >= 0:self.table.removeRow(self.table.currentRow())

    def value(self) -> dict:
        if self._controls() == self._loaded_controls:return deepcopy(self._loaded_profile)
        rows = []
        for i in range(self.table.rowCount()):
            try:position, lower, upper = [float(self.table.item(i, j).text()) for j in range(3)]
            except (ValueError, AttributeError) as error:raise ValueError(f'第 {i+1} 行请填写完整数值') from error
            rows.append(dict(position=position/100, lower_half_width_m=lower/1000, upper_half_width_m=upper/1000))
        return footprint.nominal(rows, name=self.name.text().strip(), version=self.version.text().strip(),
            interpolation=self.rule.currentData(), valid_from=self.start.text().strip() or None, valid_until=self.end.text().strip() or None)

    def accept_model(self) -> None:
        try:
            profile = self.value();footprint.require_current(profile)
        except ValueError as error:self.error.setText(str(error));return
        self.profile = profile;self.accept()

    def import_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, '导入名义变宽模型', '', '作用模型 (*.json)')
        if not path:return
        try:self.set_profile(footprint.load(Path(path)))
        except (ValueError, OSError) as error:self.error.setText(str(error))
