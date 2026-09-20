"""Independent A/B/C replay controls; worker events stay within this dialog."""
from __future__ import annotations

import os
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
import uuid

from io_utils import load_json_object as read_json


def comparison_source(result_or_dataset):
    """Resolve recorded ancestry, never infer a new origin or overwrite a model."""
    root = Path(result_or_dataset).resolve()
    if root.name == 'reconstruction':
        root = root.parent
    expected = None
    if (root / 'reconstruction/reconstruction_report.json').is_file():
        if (root / 'publication.json').is_file() and read_json(root / 'publication.json').get('status') != 'published':
            raise ValueError('This result was not published; choose a completed result or its original posed dataset')
        context = read_json(root / 'reconstruction/local_geometry.json')
        expected = (context.get('session_id'), context.get('reference_frame_id'))
        root = Path(read_json(root / 'reconstruction/reconstruction_report.json')['source_dataset']).resolve()
    seen = set()
    for _ in range(8):
        if root in seen:
            raise ValueError('Cyclic comparison source ancestry')
        seen.add(root)
        manifest = read_json(root / 'manifest.json')
        local = manifest.get('local_capture', {})
        identity = (local.get('session_id'), local.get('reference_frame_id'))
        if not local.get('measured_only') or not identity[0] or type(identity[1]) is not int:
            raise ValueError('Local session/reference evidence is missing')
        if expected is None:
            expected = identity
        if identity != expected:
            raise ValueError('Comparison source crosses a session or initial reference')
        if local.get('temporal_support_report'):
            root = Path(read_json(local['temporal_support_report'])['source_dataset']).resolve()
        elif manifest.get('refinement_source_dataset'):
            root = Path(manifest['refinement_source_dataset']).resolve()
        elif manifest.get('pose', {}).get('source') == 'online_rgbd_research_grade':
            return root
        else:
            raise ValueError('Original online local poses are unavailable; no fallback origin is assumed')
    raise ValueError('Comparison source ancestry exceeds the eight-level budget')


def progress_text(report, language='zh'):
    zh = language == 'zh'
    choose = lambda a, b: a if zh else b
    labels = {'running': choose('执行中', 'Running'), 'complete': choose('已完成', 'Complete'),
              'failed': choose('失败', 'Failed'), 'cancelled': choose('已取消', 'Cancelled'),
              'adopted': choose('已采用', 'Adopted'), 'not_adopted': choose('未采用', 'Not adopted'),
              'baseline': choose('原始基线', 'Original baseline')}
    lines = [choose('对照状态：', 'Comparison state: ') + labels.get(report.get('state'), str(report.get('state', '—')))]
    if report.get('session_id'):
        lines.append(choose('会话：', 'Session: ') + report['session_id'])
    if report.get('source_frame_count') is not None:
        lines.append(choose('原始帧 / 接纳视角：', 'Source frames / accepted views: ') +
                     f"{report['source_frame_count']} / {len(report.get('accepted_frame_ids', []))}")
    operations = {'validate': choose('来源与门禁检查', 'Source and gate checks'),
                  'refine': choose('精配准', 'Refinement'), 'temporal': choose('五帧二值筛选', 'Five-frame binary filter'),
                  'reconstruct': choose('几何重建与固定视图', 'Reconstruction and fixed projection')}
    for stage in report.get('stages', []):
        label = operations.get(stage.get('operation'), stage.get('operation', '—'))
        line = f"{stage.get('name', '')}: {label} — {labels.get(stage.get('state'), stage.get('state', '—'))}"
        if stage.get('elapsed_s') is not None:
            line += f" · {stage['elapsed_s']:.1f} s"
        lines.append(line)
    for variant in report.get('variants', []):
        status = variant.get('optimization', {}).get('status')
        line = variant['name'] + ': ' + labels.get(status, str(status))
        if variant.get('same_as'):
            line += choose('；几何与 ', '; geometry identical to ') + variant['same_as']
        lines.append(line)
    if report.get('error'):
        lines.append(choose('原因：', 'Reason: ') + str(report['error']))
    if report.get('state') != 'running':
        for key, title in [('source_unchanged', choose('原始文件哈希保持', 'Source hashes unchanged')),
                           ('latest_unchanged', choose('业务最新索引保持', 'Business latest unchanged'))]:
            value = report.get(key)
            lines.append(title + ': ' + (choose('未记录', 'Not recorded') if value is None else str(value)))
    return '\n'.join(lines)


class ComparisonWindow:
    def __init__(self, parent, result_or_dataset, outputs, language='zh'):
        self.language, self.source_input, self.outputs = language, Path(result_or_dataset), Path(outputs)
        self.window = tk.Toplevel(parent)
        self.window.title(self.t('Femto 同源 A/B/C 对照', 'Femto same-source A/B/C comparison'))
        self.window.geometry('900x590')
        self.window.minsize(600, 400)
        self.closed, self.running, self.generation = False, False, 0
        self.cancel_event = threading.Event()
        self.mailbox = queue.Queue(maxsize=1)
        self.result = None
        self.report = None
        header = ttk.Label(self.window, text=self.t(
            '同一原始 RGB-D：A 在线位姿；B 通过门禁的精配准；C 精配准与五帧二值筛选。\n'
            'W0、帧来源、观察视角和尺度固定。未采用时明确复用前一版本，原模型与最新结果索引保留。',
            'Same raw RGB-D: A online poses; B accepted refinement; C refinement with five-frame binary filtering.\n'
            'Fixed W0, source frames, projection and scale. Rejected stages reuse prior geometry; existing models and latest indices stay intact.'),
            wraplength=850, justify='left', padding=12)
        header.pack(fill='x')
        budget_label=ttk.Label(self.window, text=self.t(
            '每阶段上限 300 秒 / 3072 MiB 任务组提交内存；最多 20 个接纳视角。可取消，不读取相机。\n'
            '内部一致性不是实物误差，点数增长不是补洞覆盖率。',
            'Per-stage budget: 300 s / 3072 MiB job committed memory; at most 20 accepted views. Cancellable; no camera access.\n'
            'Internal consistency is not physical error; point growth is not hole-fill coverage.'),
            wraplength=850, justify='left', padding=(12, 0, 12, 10))
        budget_label.pack(fill='x')
        self.status = tk.StringVar(value=self.t('准备执行；详细进度来自实际任务状态。', 'Ready; progress comes from actual task states.'))
        status_label=ttk.Label(self.window, textvariable=self.status, padding=12, wraplength=850)
        status_label.pack(fill='x')
        body = ttk.Frame(self.window, padding=(12, 0, 12, 8))
        body.pack(fill='both', expand=True)
        scroll = ttk.Scrollbar(body, orient='vertical')
        scroll.pack(side='right', fill='y')
        self.details = tk.Text(body, wrap='word', width=1, height=6, state='disabled', yscrollcommand=scroll.set)
        self.details.pack(fill='both', expand=True)
        scroll.configure(command=self.details.yview)
        self._text(str(self.source_input))
        actions = ttk.Frame(self.window, padding=12)
        actions.pack(fill='x')
        self.buttons = {}
        for index, (key, zh, en, command) in enumerate([
            ('start', '开始独立对照', 'Start comparison', self.start),
            ('cancel', '取消任务', 'Cancel task', self.cancel),
            ('report', '打开对照报告', 'Open comparison', self.open_report),
            ('folder', '打开结果目录', 'Open output folder', self.open_folder)]):
            button = ttk.Button(actions, text=self.t(zh, en), command=command)
            button.grid(row=index // 2, column=index % 2, sticky='ew', padx=3, pady=3)
            actions.columnconfigure(index % 2, weight=1)
            self.buttons[key] = button
        self._buttons()
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        def resize(event):
            if event.widget is self.window:
                for label in (header,budget_label,status_label):
                    label.configure(wraplength=max(300,self.window.winfo_width()-35))
        self.window.bind('<Configure>', resize)
        self.window.bind('<Destroy>', self._destroyed)
        self.timer = self.window.after(100, self.poll)

    def t(self, zh, en):
        return zh if self.language == 'zh' else en

    def _text(self, value):
        self.details.configure(state='normal')
        self.details.delete('1.0', 'end')
        self.details.insert('1.0', value)
        self.details.configure(state='disabled')

    def _buttons(self):
        self.buttons['start'].configure(state='disabled' if self.running else 'normal')
        self.buttons['cancel'].configure(state='normal' if self.running and not self.cancel_event.is_set() else 'disabled')
        self.buttons['report'].configure(state='normal' if self.result and self.report is not None and not self.running else 'disabled')
        self.buttons['folder'].configure(state='normal' if self.result and not self.running else 'disabled')

    def _put(self, generation, report, terminal=False):
        try:
            self.mailbox.get_nowait()
        except queue.Empty:
            pass
        self.mailbox.put_nowait((generation, report, terminal))

    def start(self):
        if self.closed or self.running:
            return
        self.generation += 1
        generation = self.generation
        self.cancel_event = threading.Event()
        cancellation = self.cancel_event
        self.running, self.report = True, None
        self.result = self.outputs / 'local_comparisons' / ('comparison_' + time.strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8])
        result = self.result
        source_input = self.source_input
        self.status.set(self.t('正在核对原始位姿来源…', 'Checking original pose ancestry…'))
        self._buttons()
        def worker():
            try:
                from local_comparison import run_comparison
                source = comparison_source(source_input)
                result.parent.mkdir(parents=True, exist_ok=True)
                report = run_comparison(source, result, cancelled=cancellation.is_set,
                                        progress=lambda value: self._put(generation, value))
            except Exception as error:
                report = dict(state='cancelled' if cancellation.is_set() else 'failed', error=str(error))
            self._put(generation, report, terminal=True)
        self.thread = threading.Thread(target=worker, name='FemtoComparison', daemon=False)
        self.thread.start()

    def cancel(self):
        if self.running:
            self.cancel_event.set()
            self.status.set(self.t('正在取消本次对照；等待任务确认终止。', 'Cancelling this comparison; waiting for task termination.'))
            self._buttons()

    def poll(self):
        # A direct refresh must consume the old token too, otherwise it creates
        # another timer whose callback outlives this dialog after close.
        self._cancel_poll()
        if self.closed:
            return
        try:
            generation, report, terminal = self.mailbox.get_nowait()
        except queue.Empty:
            pass
        else:
            if generation == self.generation and self.running:
                self.report = report
                self.running = not terminal
                self._text(progress_text(report, self.language))
                if not self.running:
                    self.status.set(self.t('任务已结束；结果与未采用原因见下方及详细报告。', 'Task ended; see results and rejection reasons below and in the report.'))
                self._buttons()
        self.timer = self.window.after(100, self.poll)

    def open_report(self):
        path = self.result / 'comparison_report.html' if self.result else None
        if path and path.is_file():
            os.startfile(path)
        else:
            self.status.set(self.t('对照报告尚未生成或文件缺失。', 'Comparison report is not generated or is missing.'))

    def open_folder(self):
        if self.result and self.result.is_dir():
            os.startfile(self.result)

    def _cancel_poll(self):
        timer = getattr(self, 'timer', None)
        self.timer = None
        if timer is not None:
            try:
                self.window.after_cancel(timer)
            except tk.TclError:
                # Parent destruction may already have removed the Tcl command.
                pass

    def close(self):
        self._cancel_poll()
        if self.closed:return
        self.cancel_event.set()
        self.closed = True
        self.generation += 1
        self.window.destroy()

    def _destroyed(self,event):
        if event.widget is self.window:
            self._cancel_poll()
            if not self.closed:
                self.cancel_event.set()
                self.closed=True
                self.generation+=1
