"""Shared Tk/ttk presentation for capture and planning windows."""
from __future__ import annotations

from typing import Any


def apply_workstation_theme(root: Any) -> None:
    """Use readable dark controls, including disabled and keyboard focus states."""
    from tkinter import ttk
    style = ttk.Style(root)
    if style.theme_use() != 'clam':
        style.theme_use('clam')
    background, panel, border = '#0f141c', '#121b25', '#243042'
    text, muted, accent = '#e8eef5', '#8fa2b5', '#00b4d8'
    style.configure('.', font=('Microsoft YaHei UI', 10), background=panel, foreground=text)
    style.configure('TFrame', background=panel)
    style.configure('TLabel', background=panel, foreground=text)
    style.configure('TSeparator', background=border)
    style.configure('TButton', background='#1d2b39', foreground=text, bordercolor=border,
                    lightcolor=border, darkcolor=border, borderwidth=1, padding=(10, 8))
    style.map('TButton', background=[('disabled', '#17212b'), ('active', '#294052')],
              foreground=[('disabled', '#617284')], bordercolor=[('focus', accent)])
    style.configure('Medical.Primary.TButton', background=accent, foreground='#07141b',
                    font=('Microsoft YaHei UI', 10, 'bold'), padding=(12, 9))
    style.map('Medical.Primary.TButton', background=[('disabled', '#1b3945'), ('active', '#0096c7')],
              foreground=[('disabled', '#78939f')])
    for name in ('TCheckbutton', 'TRadiobutton'):
        style.configure(name, background=panel, foreground=text, indicatorbackground='#233444', padding=5)
        style.map(name, background=[('active', '#1d2b39')], foreground=[('disabled', '#617284')],
                  indicatorbackground=[('selected', accent), ('disabled', '#17212b')])
    for name in ('TEntry', 'TCombobox', 'TSpinbox'):
        style.configure(name, fieldbackground='#0b141f', foreground=text, background='#1d2b39',
                        bordercolor=border, insertcolor=text, arrowcolor=muted, padding=6)
        style.map(name, fieldbackground=[('disabled', '#17212b'), ('readonly', '#1d2b39')],
                  foreground=[('disabled', '#617284')], bordercolor=[('focus', accent)])
    style.configure('TNotebook', background=background, borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure('TNotebook.Tab', background='#161d27', foreground=muted, padding=(10, 8),
                    font=('Microsoft YaHei UI', 9), borderwidth=0)
    style.map('TNotebook.Tab', background=[('selected', '#1d3547'), ('active', '#203c50')],
              foreground=[('selected', text), ('active', text)])
    style.configure('Vertical.TScrollbar', background='#294052', troughcolor=panel,
                    borderwidth=0, arrowcolor=muted)
    style.configure('Dark.Horizontal.TProgressbar', troughcolor='#1b2734', background='#06d6a0',
                    bordercolor='#1b2734', lightcolor='#06d6a0', darkcolor='#06d6a0')
    root.option_add('*TCombobox*Listbox.background', '#1d2b39')
    root.option_add('*TCombobox*Listbox.foreground', text)
    root.option_add('*TCombobox*Listbox.selectBackground', '#1d3547')
    root.configure(bg=background)
