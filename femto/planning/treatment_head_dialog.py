"""Qt treatment-head parameter form kept for external planning integrations."""
from __future__ import annotations

from typing import Any

from femto.planning.treatment_head import PLACEHOLDER_DEFAULTS, head_parameters

FIELDS = (
    ("row_spacing_mm", "指定行距 / mm"),
    ("effective_treatment_width_mm", "有效作用宽度 / mm"),
    ("overlap_percent", "治疗重叠率 / %"),
    ("treatment_depth_mm", "治疗深度 / mm（仅记录）"),
    ("head_diameter_mm", "治疗头外壳直径 / mm（仅记录）"),
    ("safety_margin_mm", "探头外壳避让余量 / mm（仅记录）"),
)

def collect(mode: str, texts: dict[str, str]) -> dict[str, Any]:
    values: dict[str, float] = {}
    for key, label in FIELDS:
        if key == "row_spacing_mm" and mode != "direct_row_spacing":
            continue
        try: values[key] = float(str(texts[key]).strip())
        except (KeyError, TypeError, ValueError):
            error_label = "有效治疗宽度 / mm" if key == "effective_treatment_width_mm" else label
            raise ValueError(f"{error_label} 不是有效数字") from None
    return head_parameters(mode, **values)

def ask_treatment_head(parent: Any = None, initial: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Show a modal Qt form and return a validated record, or ``None``."""
    from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                                   QLabel, QLineEdit, QVBoxLayout)
    initial = dict(initial or {}); initial.update(initial.pop("deferred", None) or {})
    dialog = QDialog(parent); dialog.setWindowTitle("治疗头参数")
    layout = QVBoxLayout(dialog)
    note = QLabel("默认值为占位值，不是设备参数。只有行距影响轨迹；其余参数仅写入记录。")
    note.setWordWrap(True); layout.addWidget(note)
    mode = QComboBox(); mode.addItem("直接指定行距", "direct_row_spacing"); mode.addItem("按有效作用宽度计算", "from_effective_width")
    mode.setCurrentIndex(0 if initial.get("mode", "direct_row_spacing") == "direct_row_spacing" else 1); layout.addWidget(mode)
    fields: dict[str, QLineEdit] = {}
    form = QFormLayout(); layout.addLayout(form)
    for key, label in FIELDS:
        edit = QLineEdit(f"{initial.get(key, PLACEHOLDER_DEFAULTS[key]):g}"); fields[key] = edit; form.addRow(label, edit)
    result: list[dict[str, Any] | None] = [None]
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); layout.addWidget(buttons)
    def refresh(*_):
        direct = mode.currentData() == "direct_row_spacing"; fields["row_spacing_mm"].setEnabled(direct); fields["overlap_percent"].setEnabled(not direct)
        try: buttons.button(QDialogButtonBox.Ok).setEnabled(True); dialog.setToolTip(f"最终规划行距：{collect(mode.currentData(), {k: v.text() for k, v in fields.items()})['row_spacing_mm']:g} mm")
        except ValueError: buttons.button(QDialogButtonBox.Ok).setEnabled(False)
    def accept():
        try: result[0] = collect(mode.currentData(), {k: v.text() for k, v in fields.items()})
        except ValueError: return
        dialog.accept()
    mode.currentIndexChanged.connect(refresh)
    for edit in fields.values(): edit.textChanged.connect(refresh)
    buttons.accepted.connect(accept); buttons.rejected.connect(dialog.reject); refresh()
    return result[0] if dialog.exec() == QDialog.Accepted else None

__all__ = ["FIELDS", "ask_treatment_head", "collect"]
