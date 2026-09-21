"""Temporal controls inside FemtoControlApp. All widgets run on the Tk thread."""
from __future__ import annotations
from typing import Any

from femto.paths import REPO_ROOT
from femto.ui_i18n import pick
import json
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox


class TemporalPanel:
    def __init__(self, app):
        self.app=app
        self.window=tk.Toplevel(app.root)
        self.window.title(pick(self.app.language, '时域融合 · 固定相机和固定场景','Temporal Fusion · Static camera and scene'))
        self.window.geometry('680x430'); self.window.minsize(600,400)
        body=ttk.Frame(self.window,padding=16); body.pack(fill='both',expand=True)
        ttk.Label(body,text=pick(self.app.language, '复用主界面相机连接；保持相机和场景静止。','Uses the main camera connection; keep camera and scene fixed.'),wraplength=630).pack(anchor='w')
        controls=ttk.Frame(body); controls.pack(fill='x',pady=14)
        self.count=tk.StringVar(value='5'); self.mode=tk.StringVar(value='fast')
        ttk.Label(controls,text=pick(self.app.language, '帧数','Frames')).grid(row=0,column=0,padx=4)
        self.count_box=ttk.Combobox(controls,textvariable=self.count,values=('5','10','15'),state='readonly',width=5)
        self.count_box.grid(row=0,column=1,padx=4)
        self.mode_box=ttk.Combobox(controls,textvariable=self.mode,values=('fast','quality'),state='readonly',width=9)
        self.mode_box.grid(row=0,column=2,padx=4)
        self.mode_box.bind('<<ComboboxSelected>>',lambda _:self.count.set('5' if self.mode.get()=='fast' else '15'))
        self.start_button=ttk.Button(controls,text=pick(self.app.language, '开始采集并融合','Capture and fuse'),command=self.start)
        self.start_button.grid(row=0,column=3,padx=4)
        self.cancel_button=ttk.Button(controls,text=pick(self.app.language, '停止采集','Stop capture'),command=self.cancel)
        self.cancel_button.grid(row=0,column=4,padx=4)
        self.progress=tk.DoubleVar(value=0)
        ttk.Progressbar(body,variable=self.progress,maximum=100).pack(fill='x')
        self.status=tk.StringVar(value=pick(self.app.language, '就绪；5/10/15帧，仅生成独立派生结果。','Ready: 5/10/15 frames; independent derived output.'))
        ttk.Label(body,textvariable=self.status,wraplength=630,justify='left').pack(fill='x',pady=12)
        actions=ttk.Frame(body); actions.pack(fill='x')
        self.buttons={}
        for i,(name,zh,en) in enumerate((('cloud','查看融合点云','View cloud'),('comparison','前后对照','Compare'),('report','质量报告','Report'),('folder','结果目录','Folder'))):
            button=ttk.Button(actions,text=pick(self.app.language, zh,en),command=lambda n=name:self.open_result(n))
            button.grid(row=0,column=i,padx=3)
            self.buttons[name]=button
        ttk.Label(body,text=pick(self.app.language, '绿/黄置信度是稳定性评分，不是测量精度。运动检查失败不会生成融合点云。\n后台处理时仍可观看主窗口实时预览。',
            'Confidence indicates temporal stability, not measurement accuracy. Motion rejection produces no fused cloud.\nThe main live preview continues during processing.'),wraplength=630,justify='left').pack(fill='x',pady=18)
        self.result=app.latest_temporal
        if self.result is None:
            self.restore_last()
        if self.result:
            self.refresh('temporal_finished',self.result)
        self.update_controls()

    def restore_last(self) -> None:
        from femto.app.femto_control_app import DATASETS_ROOT
        # Legacy standalone runs have no marker. Find the newest temporal raw
        # capture and its latest report without altering reconstruction markers.
        for source in sorted(DATASETS_ROOT.glob('temporal_capture_*'),reverse=True)[:10]:
            for output in sorted((source/'derived').glob('temporal_fusion_*'),reverse=True):
                try:
                    report=json.loads((output/'temporal_report.json').read_text(encoding='utf-8'))
                    self.result=dict(status=report['status'],source=str(source),output=str(output),report=report)
                    self.app.latest_temporal=self.result
                    return
                except (OSError,ValueError,KeyError): continue

    def start(self) -> None:
        if self.app.start_temporal(int(self.count.get()),self.mode.get()):
            self.result=None; self.progress.set(0)
            self.status.set(pick(self.app.language, '等待相机采集新帧…','Waiting for new camera frames…'))

    def cancel(self) -> None:
        if self.app.temporal_collecting:
            self.app.command_queue.put(('temporal_stop',None))
            self.status.set(pick(self.app.language, '正在停止；已采集原始帧将保留。','Stopping; captured raw frames will be retained.'))

    def refresh(self,event: Any,payload: Any) -> None:
        if event=='temporal_progress':
            self.progress.set(100*payload['frames']/payload['total'])
            self.status.set(pick(self.app.language, '采集中','Capturing')+f" {payload['frames']}/{payload['total']}")
        elif event=='temporal_processing':
            self.status.set(pick(self.app.language, '保存原始数据并后台融合…主窗口预览继续。','Saving raw data and fusing in background… Live preview continues.'))
        elif event=='temporal_error':
            self.result=dict(status='error',output=payload.get('source'),error=payload['error'])
            self.status.set(pick(self.app.language, '未完成：','Failed: ')+payload['error'])
        elif event=='temporal_finished':
            self.result=payload
            status=payload['status']; report=payload.get('report',{})
            if status=='error':
                self.status.set(pick(self.app.language, '未完成：','Failed: ')+payload.get('error',''))
            elif status=='completed':
                self.progress.set(100)
                self.status.set(pick(self.app.language, '融合完成','Fusion completed')+f" · {report['frame_count']} frames\n"
                    f"invalid: {report['original_invalid_rate']:.3%} → {report['after_temporal_invalid_rate']:.3%}\n"
                    f"Points: {report['reference_point_count']:,} → {report['point_count']:,}\n"+str(payload['output']))
            else:
                self.progress.set(0)
                self.status.set(pick(self.app.language, '已停止或未通过静止检查；未生成融合模型。','Stopped or static check rejected; no fused model.')+'\n'+str(payload['output']))

    def update_controls(self) -> None:
        if not self.window.winfo_exists(): return
        app=self.app
        available=app.camera_ready and not (app.processing or app.capturing or app.photo_saving)
        self.start_button.configure(state='normal' if available else 'disabled')
        self.cancel_button.configure(state='normal' if app.temporal_collecting else 'disabled')
        for box in (self.mode_box,self.count_box): box.configure(state='disabled' if app.temporal_busy else 'readonly')
        result=self.result or {}; output=Path(result['output']) if result.get('output') else None
        for name,button in self.buttons.items():
            file={'cloud':'fused_pointcloud.ply','comparison':'comparison.png','report':'temporal_report.json'}.get(name)
            enabled=bool(output and output.exists() and (file is None or (output/file).exists()))
            if name in ('cloud','comparison'): enabled=enabled and result.get('status')=='completed'
            button.configure(state='normal' if enabled else 'disabled')
        self.app.root.after(200,self.update_controls)

    def open_result(self,name: str) -> None:
        if not self.result or not self.result.get('output'): return
        output=Path(self.result['output'])
        try:
            if name=='cloud':
                if self.result.get('status')!='completed': return
                subprocess.Popen([sys.executable,'-m','femto.viewer.segmented_cloud_viewer',str(output/'fused_pointcloud.ply')],cwd=REPO_ROOT,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            else:
                target=output if name=='folder' else output/('comparison.png' if name=='comparison' else 'temporal_report.json')
                os.startfile(target)
        except OSError as error:
            messagebox.showerror(pick(self.app.language, '时域融合','Temporal Fusion'),str(error),parent=self.window)
