"""Single-step forms for existing research preview planners.

These fields retain the preview planners' bounds and semantics.  Treatment
head parameters for saved trajectories use treatment_head_dialog instead.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable


@dataclass(frozen=True)
class PreviewField:
    key: str
    label: str
    minimum: float
    maximum: float


COVERAGE_FIELDS = (
    PreviewField('width_mm', '模拟扫描宽度 / mm', .2, 100.),
    PreviewField('overlap_percent', '相邻路线重叠率 / %', 0., 90.),
)
COVERAGE_ADVANCED_FIELDS = (
    PreviewField('boundary_margin_mm', '选区边缘内缩 / mm（会减少边缘覆盖）', 0., 50.),
)
SURFACE_FIELDS = (
    PreviewField('spacing_mm', '目标表面层间距 / mm', .1, 100.),
    PreviewField('sample_step_mm', '沿线最大采样步长 / mm', .1, 100.),
    PreviewField('width_mm', '模拟扫描宽度 / mm', .2, 100.),
)
SURFACE_ADVANCED_FIELDS = (
    PreviewField('boundary_margin_mm', '选区边缘内缩 / mm（会减少边缘覆盖）', 0., 50.),
    PreviewField('tolerance_mm', '间距报告容差 / mm', .01, 100.),
)
DIRECTIONS = {'水平往返': 'horizontal', '垂直往返': 'vertical'}
CORNERS = {'左上': 'top_left', '右上': 'top_right', '左下': 'bottom_left', '右下': 'bottom_right'}


def parse_fields(fields: tuple[PreviewField, ...], texts: dict[str, str]) -> dict[str, float]:
    """Validate every visible field before sending anything to a planner."""
    result = {}
    for field in fields:
        try:
            value = float(texts[field.key].strip())
        except (KeyError, ValueError, TypeError):
            raise ValueError(f'{field.label}：请输入数字。') from None
        if not math.isfinite(value) or not field.minimum <= value <= field.maximum:
            raise ValueError(f'{field.label}：范围 {field.minimum:g}–{field.maximum:g}。')
        result[field.key] = value
    return result


class PreviewSettingsDialog:
    """Modal view with inline validation and no writes on cancel."""

    def __init__(self, parent: Any, title: str, fields: tuple[PreviewField, ...],
                 initial: dict[str, Any], summary: Callable[[dict[str, float]], str],
                 choices: dict[str, tuple[str, dict[str, str]]] | None = None,
                 advanced_fields: tuple[PreviewField, ...] = ()) -> None:
        import tkinter as tk
        from tkinter import ttk
        self.fields, self.advanced_fields, self.summary = fields, advanced_fields, summary
        self.all_fields = fields + advanced_fields
        self.choices = choices or {}
        self.result = None
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.transient(parent)
        self.window.resizable(False, False)
        self.window.configure(bg='#161d27')
        body = tk.Frame(self.window, bg='#161d27', padx=22, pady=18)
        body.pack(fill='both', expand=True)
        label_options = dict(bg='#161d27', fg='#dce6ee', font=('Microsoft YaHei UI', 10))
        tk.Label(body, text=title, **{**label_options, 'font': ('Microsoft YaHei UI', 14, 'bold')}).grid(
            row=0, column=0, columnspan=2, sticky='w')
        tk.Label(body, text='占位值，非设备参数。结果仅为几何预览。\n'
                 '探头外壳避让余量不在此处计算；正式治疗头参数中只记录，不参与规划。',
                 wraplength=440, justify='left', **{**label_options, 'fg': '#ffd166'}).grid(
            row=1, column=0, columnspan=2, sticky='w', pady=(8, 14))
        self.texts, self.choice_vars, self.entries = {}, {}, {}
        row = 2
        for key, (label, options) in self.choices.items():
            selected = next((name for name, value in options.items() if value == initial.get(key)), next(iter(options)))
            variable = tk.StringVar(self.window, value=selected)
            self.choice_vars[key] = variable
            tk.Label(body, text=label, **label_options).grid(row=row, column=0, sticky='w', pady=6)
            ttk.Combobox(body, textvariable=variable, values=list(options), state='readonly', width=19).grid(
                row=row, column=1, sticky='e', padx=(18, 0), pady=6)
            row += 1
        for field in fields:
            variable = tk.StringVar(self.window, value=f'{initial.get(field.key, 0.):g}')
            self.texts[field.key] = variable
            tk.Label(body, text=field.label, **label_options).grid(row=row, column=0, sticky='w', pady=6)
            entry = ttk.Entry(body, textvariable=variable, width=21)
            entry.grid(row=row, column=1, sticky='e', padx=(18, 0), pady=6)
            self.entries[field.key] = entry
            row += 1
        self.advanced_enabled = tk.BooleanVar(self.window, value=False)
        self.advanced_frame = None
        if advanced_fields:
            ttk.Checkbutton(body, text='显示高级几何参数', variable=self.advanced_enabled,
                            command=self.toggle_advanced).grid(row=row, column=0, columnspan=2,
                                                                sticky='w', pady=(8, 2))
            row += 1
            self.advanced_frame = tk.Frame(body, bg='#161d27')
            self.advanced_frame.grid(row=row, column=0, columnspan=2, sticky='ew')
            self.advanced_frame.grid_remove()
            for field in advanced_fields:
                variable = tk.StringVar(self.window, value=f'{initial.get(field.key, 0.):g}')
                self.texts[field.key] = variable
                tk.Label(self.advanced_frame, text=field.label, **label_options).grid(
                    row=len(self.entries), column=0, sticky='w', pady=6)
                entry = ttk.Entry(self.advanced_frame, textvariable=variable, width=21)
                entry.grid(row=len(self.entries), column=1, sticky='e', padx=(18, 0), pady=6)
                self.entries[field.key] = entry
                row += 1
        self.feedback = tk.StringVar(self.window)
        self.feedback_label = tk.Label(body, textvariable=self.feedback, wraplength=440, justify='left',
                                       **{**label_options, 'fg': '#06d6a0'})
        self.feedback_label.grid(row=row, column=0, columnspan=2, sticky='w', pady=(14, 8))
        buttons = tk.Frame(body, bg='#161d27')
        buttons.grid(row=row+1, column=0, columnspan=2, sticky='e', pady=(8, 0))
        ttk.Button(buttons, text='取消', command=self.window.destroy).pack(side='left', padx=(0, 8))
        self.ok = ttk.Button(buttons, text='生成路线预览', command=self.accept, style='Medical.Primary.TButton')
        self.ok.pack(side='left')
        for variable in self.texts.values():
            variable.trace_add('write', self.refresh)
        self.window.bind('<Escape>', lambda _: self.window.destroy())
        self.window.bind('<Return>', lambda _: self.accept())
        self.refresh()

    def values(self) -> dict[str, float]:
        return parse_fields(self.all_fields, {key: var.get() for key, var in self.texts.items()})

    def toggle_advanced(self) -> None:
        if self.advanced_frame is None:
            return
        if self.advanced_enabled.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()

    def refresh(self, *_: Any) -> None:
        try:
            values = self.values()
        except ValueError as error:
            self.feedback.set(str(error))
            self.feedback_label.configure(fg='#ffd166')
            self.ok.configure(state='disabled')
        else:
            self.feedback.set(self.summary(values))
            self.feedback_label.configure(fg='#06d6a0')
            self.ok.configure(state='normal')

    def accept(self) -> None:
        try:
            values = self.values()
            selected = {key: options[self.choice_vars[key].get()] for key, (_, options) in self.choices.items()}
        except (ValueError, KeyError):
            self.refresh()
            return
        self.result = dict(values, **selected)
        self.window.destroy()

    def show(self) -> dict[str, Any] | None:
        self.window.wait_visibility()
        self.window.grab_set()
        next(iter(self.entries.values())).focus_set()
        self.window.wait_window()
        return self.result


def ask_coverage_settings(parent: Any, initial: dict[str, Any]) -> dict[str, Any] | None:
    dialog = PreviewSettingsDialog(parent, '固定 RGB 覆盖预览', COVERAGE_FIELDS, initial,
        lambda v: f"预计相邻行距：{v['width_mm'] * (1-v['overlap_percent']/100):g} mm\n"
                  '按固定画面选择起点，实际位置由表面与禁区决定。',
        dict(direction=('往返方向', DIRECTIONS), start_corner=('起点偏好', CORNERS)),
        advanced_fields=COVERAGE_ADVANCED_FIELDS)
    return dialog.show()


def ask_surface_settings(parent: Any, initial: dict[str, Any]) -> dict[str, Any] | None:
    dialog = PreviewSettingsDialog(parent, '三维表面间距预览', SURFACE_FIELDS, initial,
        lambda v: f"目标层间距：{v['spacing_mm']:g} mm · 采样步长：{v['sample_step_mm']:g} mm\n"
                  '网格表面层间距近似；以生成后的间距报告为准。',
        advanced_fields=SURFACE_ADVANCED_FIELDS)
    return dialog.show()
