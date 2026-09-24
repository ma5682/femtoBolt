"""Modal dialog for treatment-head parameters before generating a surface trajectory."""
from __future__ import annotations

from typing import Any

from femto.planning.treatment_head import PLACEHOLDER_DEFAULTS, head_parameters


FIELDS = (
    ("row_spacing_mm", "指定行距 / mm"),
    ("effective_treatment_width_mm", "有效治疗宽度 / mm"),
    ("overlap_percent", "治疗重叠率 / %"),
    ("treatment_depth_mm", "治疗深度 / mm（仅记录）"),
    ("head_diameter_mm", "治疗头直径 / mm（仅记录）"),
    ("safety_margin_mm", "安全边界 / mm（仅记录）"),
)


def collect(mode: str, texts: dict[str, str]) -> dict[str, Any]:
    """Parse entry texts into a validated record; raises ValueError with a readable message."""
    values = {}
    for key, label in FIELDS:
        if key == "row_spacing_mm" and mode != "direct_row_spacing":
            continue  # disabled entry; the row spacing is computed
        try:
            values[key] = float(texts[key].strip())
        except (KeyError, ValueError):
            raise ValueError(f"{label} 不是有效数字") from None
    return head_parameters(mode, **values)


def ask_treatment_head(parent: Any, initial: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return a `head_parameters` record, or None if cancelled."""
    import tkinter as tk
    from tkinter import ttk
    initial = dict(initial or {})
    initial.update(initial.pop("deferred", None) or {})
    window = tk.Toplevel(parent)
    window.title("治疗头参数")
    window.transient(parent)
    window.resizable(False, False)
    body = ttk.Frame(window, padding=12)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="默认值为占位值，不是设备参数。只有行距影响轨迹；深度、直径、安全边界仅写入记录。",
              wraplength=380, justify="left").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
    mode = tk.StringVar(value=initial.get("mode", "direct_row_spacing"))
    ttk.Radiobutton(body, text="直接指定行距", variable=mode, value="direct_row_spacing").grid(row=1, column=0, columnspan=2, sticky="w")
    ttk.Radiobutton(body, text="根据有效治疗宽度自动计算", variable=mode, value="from_effective_width").grid(row=2, column=0, columnspan=2, sticky="w")
    texts: dict[str, tk.StringVar] = {}
    entries: dict[str, Any] = {}
    for index, (key, label) in enumerate(FIELDS, start=3):
        ttk.Label(body, text=label).grid(row=index, column=0, sticky="w", pady=2)
        texts[key] = tk.StringVar(value=f"{initial.get(key, PLACEHOLDER_DEFAULTS[key]):g}")
        entries[key] = ttk.Entry(body, textvariable=texts[key], width=10)
        entries[key].grid(row=index, column=1, sticky="e", pady=2)
    computed = tk.StringVar()
    ttk.Label(body, textvariable=computed, wraplength=380, justify="left").grid(row=len(FIELDS) + 3, column=0, columnspan=2, sticky="w", pady=(8, 0))
    result: list[dict[str, Any] | None] = [None]
    ok = ttk.Button(body, text="生成")

    def refresh(*_: Any) -> None:
        direct = mode.get() == "direct_row_spacing"
        entries["row_spacing_mm"].configure(state="normal" if direct else "disabled")
        entries["overlap_percent"].configure(state="disabled" if direct else "normal")
        try:
            record = collect(mode.get(), {key: value.get() for key, value in texts.items()})
            computed.set(f"计算行距：{record['row_spacing_mm']:g} mm")
            ok.configure(state="normal")
        except ValueError as error:
            computed.set("参数无效：" + str(error))
            ok.configure(state="disabled")

    def accept() -> None:
        try:
            result[0] = collect(mode.get(), {key: value.get() for key, value in texts.items()})
        except ValueError:
            return
        window.destroy()

    mode.trace_add("write", refresh)
    for value in texts.values():
        value.trace_add("write", refresh)
    buttons = ttk.Frame(body)
    buttons.grid(row=len(FIELDS) + 4, column=0, columnspan=2, sticky="e", pady=(10, 0))
    ok.configure(command=accept)
    ok.pack(in_=buttons, side="left")
    ttk.Button(buttons, text="取消", command=window.destroy).pack(side="left", padx=(6, 0))
    window.bind("<Escape>", lambda _: window.destroy())
    refresh()
    window.grab_set()
    window.wait_window()
    return result[0]
