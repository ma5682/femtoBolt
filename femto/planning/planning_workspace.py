"""One managed planning window per source in the current GUI process."""
from __future__ import annotations
from typing import Any

from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox


def open_workspace(parent: Any,path: str | Path,language: str='zh',region_index: int | None=None,on_open: Any=None) -> Any:
    root=parent._root()
    windows=getattr(root,'_planning_windows',None)
    if windows is None:windows={};root._planning_windows=windows
    key=(str(Path(path).resolve()).casefold(),region_index)
    existing=windows.get(key)
    if existing is not None and existing.winfo_exists():
        existing.deiconify();existing.lift();existing.focus_force()
        if on_open:on_open()
        return existing
    window=tk.Toplevel(root);windows[key]=window
    window.title('正在打开规划工作台…');window.geometry('480x130')
    from femto.planning.planning_io import TaskControl
    control=TaskControl('准备区域工程');timer=[None]
    label=tk.Label(window,text='正在准备区域工程，请稍候…');label.pack(pady=15)
    cancel_button=tk.Button(window,text='取消打开',command=window.destroy);cancel_button.pack()
    def cleanup(event):
        if event.widget is window:
            control.set()
            if timer[0]:
                try:window.after_cancel(timer[0])
                except tk.TclError:pass
            for alias,value in list(windows.items()):
                if value is window:windows.pop(alias,None)
            if parent.winfo_exists():parent.deiconify();parent.lift()
    window.bind('<Destroy>',cleanup,add='+')
    mailbox=queue.Queue()
    def prepare():
        try:
            source=Path(path)
            if region_index is not None:
                from femto.planning.region_project import project_from_annotations
                source=project_from_annotations(source,region_index,cancel=control,progress=control.progress)
            control.check();mailbox.put((source,None))
        except Exception as error:mailbox.put((None,str(error)))
    def poll():
        if not window.winfo_exists():return
        label.configure(text=f'{control.stage} · {time.monotonic()-control.started:.1f} 秒')
        try:source,error=mailbox.get_nowait()
        except queue.Empty:timer[0]=window.after(100,poll);return
        if error:
            messagebox.showerror('打开规划失败',error,parent=window);window.destroy();return
        label.destroy();cancel_button.destroy();timer[0]=None
        from femto.planning.fixed_rgb_viewer import FixedRGBViewer
        window.viewer=FixedRGBViewer(window,source,language)
        windows[(str(Path(source).resolve()).casefold(),None)]=window
        if on_open:on_open()
    threading.Thread(target=prepare,daemon=True).start();timer[0]=window.after(100,poll)
    return window


def close_workspaces(root: Any,force: bool=False) -> bool:
    """Return False before host shutdown if any workspace declines closing."""
    windows=getattr(root,'_planning_windows',{})
    # Test doubles and partially initialized roots may synthesize attributes.
    # Only the dictionary installed by open_workspace owns planning windows.
    if not isinstance(windows,dict):windows={}
    for window in list(set(windows.values())):
        if not window.winfo_exists():continue
        viewer=getattr(window,'viewer',None)
        if viewer is None:window.destroy()
        else:
            viewer.close(force=force)
            if window.winfo_exists():return False
    return True
