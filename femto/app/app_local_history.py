"""Independent history window: browse completed models and unsuccessful sessions."""
from femto.paths import REPO_ROOT
from datetime import datetime
from pathlib import Path
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk,messagebox
from femto.annotation.local_history import scan_history,existing_actions,set_records_deleted
from femto.pipeline.local_processing import format_quality,empty_report

# These secondary tools are temporarily kept out of the history UI.  Their
# implementation and recorded evidence stay intact so the entries can be
# restored without changing historical data.
HIDDEN_HISTORY_FEATURES=frozenset({'quality','compare','report'})


class LocalHistoryWindow:
    def __init__(self,parent,datasets,outputs,language='zh'):
        self.language=language;self.datasets=datasets;self.outputs=outputs
        self.closed=False;self.rows=[];self.mailbox=queue.Queue(maxsize=1);self.loading=False
        self.generation=0;self.cancel=threading.Event();self.selected_key=None
        self.deleted_stack=[]
        self.window=tk.Toplevel(parent);self.window.title(self.t('局部采集历史记录','Local capture history'))
        self.window.geometry('960x620');self.window.minsize(760,520)
        self.window.protocol('WM_DELETE_WINDOW',self.close)
        top=ttk.Frame(self.window,padding=12);top.pack(fill='x')
        self.refresh_button=ttk.Button(top,text=self.t('刷新历史记录','Refresh'),command=self.refresh);self.refresh_button.pack(side='right')
        self.undo_delete_button=ttk.Button(top,text=self.t('撤销删除','Undo delete'),command=self.undo_delete,state='disabled')
        self.undo_delete_button.pack(side='right',padx=6)
        ttk.Label(top,text=self.t('先选中一条或多条记录，再执行查看或删除；历史浏览不会改变当前采集的原点或最近结果。',
                                 'Select one or more records before viewing or deleting; browsing history does not change the current origin or latest result.')).pack(side='left')
        table=ttk.Frame(self.window,padding=(12,0));table.pack(fill='both',expand=True)
        # Six requested rows leave room for both action rows at 760 x 520;
        # the table still expands to fill taller windows and keeps its scrollbar.
        self.tree=ttk.Treeview(table,columns=('time','state','count','name'),show='headings',selectmode='extended',height=6)
        for key,zh,en,width in [('time','时间','Time',150),('state','状态','Status',140),('count','接纳帧','Views',65),('name','记录目录','Record',420)]:
            self.tree.heading(key,text=self.t(zh,en));self.tree.column(key,width=width,minwidth=50,stretch=key=='name')
        scroll=ttk.Scrollbar(table,orient='vertical',command=self.tree.yview);scroll.pack(side='right',fill='y')
        self.tree.configure(yscrollcommand=scroll.set);self.tree.pack(fill='both',expand=True)
        self.tree.bind('<<TreeviewSelect>>',lambda e:self.selected())
        self.tree.bind('<Delete>',self.delete_key)
        self.details=tk.StringVar();self.status=tk.StringVar()
        ttk.Label(self.window,textvariable=self.details,wraplength=880,justify='left',padding=12).pack(fill='x')
        quality_frame=ttk.Frame(self.window,padding=(12,0))
        if 'quality' not in HIDDEN_HISTORY_FEATURES:quality_frame.pack(fill='x')
        quality_scroll=ttk.Scrollbar(quality_frame,orient='vertical');quality_scroll.pack(side='right',fill='y')
        self.quality_text=tk.Text(quality_frame,height=8,wrap='word',state='disabled',
                                 yscrollcommand=quality_scroll.set)
        self.quality_text.pack(fill='both',expand=True);quality_scroll.configure(command=self.quality_text.yview)
        self.window.bind('<Configure>',lambda e:self.resize(e))
        actions=ttk.Frame(self.window,padding=12);actions.pack(fill='x');self.buttons={}
        visible_actions=[item for item in [('fixed','固定 RGB 选区','Fixed RGB annotation'),('mesh','高级三维查看','Advanced 3D'),
                ('compare','同源对照','A/B/C comparison'),('report','查看报告','Report'),('folder','打开数据目录','Open folder'),('delete','删除选中记录','Delete selected')]
            if item[0] not in HIDDEN_HISTORY_FEATURES]
        columns=2
        for column in range(columns):actions.columnconfigure(column,weight=1)
        for n,(key,zh,en) in enumerate(visible_actions):
            b=ttk.Button(actions,text=self.t(zh,en),command=lambda k=key:self.open(k),state='disabled')
            b.grid(row=n//columns,column=n%columns,sticky='ew',padx=4,pady=3);self.buttons[key]=b
        ttk.Label(self.window,textvariable=self.status,padding=(12,0,12,12),wraplength=880).pack(fill='x')
        self.refresh();self.timer=self.window.after(100,self.poll)

    def t(self,zh,en):return zh if self.language=='zh' else en
    def resize(self,event):
        if event.widget is self.window:
            for w in self.window.winfo_children():
                if isinstance(w,ttk.Label):w.configure(wraplength=max(300,event.width-30))
    def refresh(self):
        if self.loading or self.closed:return
        current=self.current()
        self.selected_key=current['key'] if current else self.selected_key
        self.generation+=1;generation=self.generation
        self.cancel=threading.Event();cancel=self.cancel
        self.loading=True;self.refresh_button.configure(state='disabled')
        self.buttons['delete'].configure(state='disabled');self.undo_delete_button.configure(state='disabled')
        self.status.set(self.t('正在读取历史记录…','Loading history…'))
        def worker():
            try:rows,error=scan_history(self.datasets,self.outputs,cancel=cancel),None
            except Exception as exc:rows,error=[],str(exc)
            if not cancel.is_set():
                try:self.mailbox.put_nowait((generation,rows,error))
                except queue.Full:pass
        threading.Thread(target=worker,name='LocalHistory',daemon=True).start()
    def poll(self):
        if self.closed:return
        try:
            generation,rows,error=self.mailbox.get_nowait()
            if generation!=self.generation:
                self.timer=self.window.after(100,self.poll);return
            self.loading=False;self.refresh_button.configure(state='normal')
            self.undo_delete_button.configure(state='normal' if self.deleted_stack else 'disabled')
            if error:
                self.selected()
                self.status.set(self.t('读取失败，保留当前列表：','Read failed; current list retained: ')+error)
                self.timer=self.window.after(100,self.poll);return
            self.rows=rows;self.tree.delete(*self.tree.get_children());self.details.set('')
            states={'model':self.t('模型已生成','Model ready'),'failed':self.t('失败 / 未生成模型','Failed / no model'),
                    'lost':self.t('跟踪丢失','Tracking lost'),'recorded':self.t('原始记录 / 无模型','Raw record / no model'),
                    'incomplete':self.t('文件缺失或损坏','Missing / invalid files'),
                    'unpublished':self.t('结果未发布','Unpublished result')}
            for i,row in enumerate(rows):
                self.tree.insert('', 'end', iid=str(i),values=(datetime.fromtimestamp(row['time']).strftime('%Y-%m-%d %H:%M:%S'),states[row['state']],row['count'] if row['count'] is not None else '—',row['name']))
            self.status.set(error or self.t(f'找到 {len(rows)} 条记录。',f'{len(rows)} records found.'))
            if rows:
                selected=next((i for i,row in enumerate(rows) if row['key']==self.selected_key),0)
                self.tree.selection_set(str(selected));self.tree.see(str(selected))
            self.selected()
        except queue.Empty:pass
        self.timer=self.window.after(100,self.poll)
    def current(self):
        selection=self.tree.selection()
        return self.rows[int(selection[0])] if selection and int(selection[0])<len(self.rows) else None
    def selected_rows(self):
        """Return all valid selected records in display order."""
        result=[]
        for item in self.tree.selection():
            try:index=int(item)
            except (TypeError,ValueError):continue
            if 0<=index<len(self.rows):result.append(self.rows[index])
        return result
    def selected(self):
        selected_rows=self.selected_rows();row=selected_rows[0] if selected_rows else None
        enabled=existing_actions(row) if row else {}
        enabled['delete']=bool(selected_rows and not self.loading)
        for key,button in self.buttons.items():button.configure(state='normal' if enabled.get(key) else 'disabled')
        self.quality_text.configure(state='normal')
        self.quality_text.delete('1.0','end')
        if row:self.quality_text.insert('end',format_quality(row.get('quality',empty_report()),self.language))
        self.quality_text.configure(state='disabled')
        if row:
            self.selected_key=row['key']
            text=''
            if len(selected_rows)>1:
                text+=self.t(f'已选中 {len(selected_rows)} 条记录；查看按钮作用于第一条选中记录。\n',
                              f'{len(selected_rows)} records selected; view buttons use the first selected record.\n')
            text+=self.t('模型：','Model: ')+(row['result'] or '—')+'\n'+self.t('原始会话：','Raw session: ')+(row['source'] or '—')
            if not enabled.get('fixed') and row['mesh']:
                reasons={
                    'origin_unverified':self.t('初始原点来源未通过核对。','Initial-origin provenance could not be verified.'),
                    'invalid_calibration':self.t('参考标定无效。','Reference calibration is invalid.'),
                    'source_budget':self.t('来源超出历史预检预算。','Source exceeds the history preflight budget.'),
                    'unpublished_result':self.t('该任务未发布为成功结果（处理中、已取消、失败或缺少发布证明）。',
                                               'This task was not published as a successful result (pending, cancelled, failed, or missing publication evidence).'),
                }
                publication_reasons={
                    'pending':self.t('该任务仍在处理或等待发布，不是已发布成功结果。','This task is processing or awaiting publication; it is not a published result.'),
                    'cancelled':self.t('该任务已取消，没有发布成功结果。','This task was cancelled and did not publish a successful result.'),
                    'failed':self.t('该任务失败，没有发布成功结果。','This task failed and did not publish a successful result.'),
                    'missing':self.t('新结果缺少必需的发布证明。','The required publication receipt is missing.'),
                    'invalid':self.t('发布证明与模型或会话不匹配。','The publication receipt does not match this model or session.'),
                }
                if row.get('fixed_reason')=='unpublished_result':
                    reasons['unpublished_result']=publication_reasons.get(row.get('publication_status'),reasons['unpublished_result'])
                text+='\n'+reasons.get(row.get('fixed_reason'),self.t('参考 RGB 或来源文件缺失/已变化。','Reference RGB/source files are missing or changed.'))
                text+=' '+self.t('可使用高级三维查看。','Advanced 3D inspection remains available.')
            if row['error']:text+='\n'+str(row['error'])
            self.details.set(text)
        else:
            self.details.set('')
            if not self.loading:self.status.set(self.t('请先选中一条或多条历史记录。','Select one or more history records first.'))
    def open(self,key):
        if key=='delete':self.delete_record();return
        row=self.current()
        if not row or not existing_actions(row).get(key):
            self.selected();self.status.set(self.t('所需文件不可用，请刷新历史记录。','Required files are unavailable; refresh history.'));return
        try:
            if key=='compare':
                from femto.app.app_local_comparison import ComparisonWindow
                ComparisonWindow(self.window,row['result'],self.outputs,self.language)
            elif key in ('fixed','mesh'):
                module='femto.planning.fixed_rgb_viewer' if key=='fixed' else 'femto.viewer.scene_demo_viewer'
                subprocess.Popen([sys.executable,'-m',module,str(Path(row['result'])/'reconstruction'),'--lang',self.language],cwd=REPO_ROOT)
            else:os.startfile(row['report'] if key=='report' else row['result'] or row['source'])
        except (OSError,ValueError,RuntimeError,ImportError) as exc:
            self.status.set(self.t('无法打开：','Could not open: ')+str(exc));self.selected()
    def delete_key(self,event=None):
        self.delete_record()
        return 'break'
    def delete_record(self):
        if self.loading:return
        rows=self.selected_rows()
        if not rows:
            self.status.set(self.t('请先在列表中选中一条或多条记录。','Select one or more records in the list first.'))
            return
        names='\n'.join(f'• {row["name"]}' for row in rows)
        if not messagebox.askyesno(self.t('删除历史记录','Delete history record'),
                self.t(f"从历史列表删除以下 {len(rows)} 条记录？\n{names}\n\n原始 RGB-D、模型和标记文件保留，不释放磁盘空间。可在本窗口整批撤销删除。",
                       f"Remove these {len(rows)} records from history?\n{names}\n\nRGB-D, models and annotations are retained; disk space is not freed. You can undo the batch in this window."),
                parent=self.window,default='no'):return
        try:
            set_records_deleted(rows,self.datasets,self.outputs)
            self.deleted_stack.append([dict(row) for row in rows]);self.selected_key=None;self.refresh()
        except (OSError,ValueError) as exc:self.status.set(self.t('删除失败：','Delete failed: ')+str(exc))

    def undo_delete(self):
        if self.loading or not self.deleted_stack:return
        rows=self.deleted_stack[-1]
        try:
            set_records_deleted(rows,self.datasets,self.outputs,False)
            self.deleted_stack.pop();self.tree.selection_remove(*self.tree.selection())
            self.selected_key=rows[0]['key'] if rows else None;self.refresh()
        except (OSError,ValueError) as exc:self.status.set(self.t('恢复失败：','Restore failed: ')+str(exc))

    def close(self):
        self.closed=True;self.generation+=1;self.cancel.set()
        if hasattr(self,'timer'):self.window.after_cancel(self.timer)
        self.window.destroy()
