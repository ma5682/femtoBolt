"""Local annotation on a locked reference photograph; geometry stays in W0."""
import argparse
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
from PIL import Image,ImageTk
from fixed_rgb_geometry import FixedRGBModel, GeometryBudget, MAX_POLYGON_CORNERS
from scene_annotations import point_readout_mm,distance_m

# Keep non-product workflows implemented but out of the current product demo.
PRODUCT_HIDDEN_FEATURES=frozenset({
    'annotation_load','measurement_tools','project_management','legacy_exclusion_editor','point_tools'
})
PRODUCT_SINGLE_REGION=True


class FixedRGBViewer:
    def __init__(self,root,result,language='zh'):
        self.root,self.result,self.language=root,Path(result),language
        self.project_path=self.result.resolve() if self.result.is_file() else None
        self._plans={};self._requests={};self._plan_errors={};self._saved_plans={};self._saved_requests={};self._plan_buttons={}
        self._project_region_id=None
        from planning_drafts import Journal
        self.draft_journal=Journal(self.result);self._draft_ready=False;self._draft_due=None;self._draft_signature=None
        self.model=None;self.busy=False;self.closed=False;self.polygon=[];self.focus=None
        self._brush_history=[];self._brush_mask=None;self._brush_last=None;self._brush_cursor=None
        self._selection_polygons={'work':[],'forbidden':[]}
        self.generation=0;self.task_cancel=threading.Event();self.operation='load_model'
        self._task_lock=threading.Lock();self._task_local=threading.local()
        self._annotations_dirty=False;self._syncing_fields=False;self.active_mode='region'
        self.zoom=1.;self.pan=np.zeros(2);self.scale=1.;self.offset=np.zeros(2)
        self.mailbox=queue.Queue(maxsize=1);self.widgets=[]
        root.title(self.t('Femto · 固定 RGB 选区与三维坐标','Femto · Fixed RGB annotation'))
        root.geometry('1280x820');root.minsize(1000,720);root.configure(bg='#101b26')
        root.protocol('WM_DELETE_WINDOW',self.close)
        main=tk.Frame(root,bg='#101b26');main.pack(side='left',fill='both',expand=True)
        tk.Label(main,text=self.t('固定初始相机视角 · 图像选区 / 三维映射','Locked initial camera · RGB selection / 3D mapping'),
            fg='#e5f1f8',bg='#101b26',font=('Microsoft YaHei UI',16,'bold'),pady=12).pack(fill='x')
        self.status=tk.StringVar(value=self.t('正在读取参考照片与三维表面…','Loading reference RGB and surface…'))
        tk.Label(main,textvariable=self.status,fg='#9ae1cf',bg='#18303c',wraplength=680,justify='left',pady=8).pack(fill='x')
        self.canvas=tk.Canvas(main,bg='#08121c',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.draw())
        self.canvas.bind('<Button-1>',self.click)
        self.canvas.bind('<B1-Motion>',self.brush_move)
        self.canvas.bind('<ButtonRelease-1>',self.brush_end)
        self.canvas.bind('<Motion>',self.brush_hover)
        self.canvas.bind('<Leave>',lambda e:self.canvas.delete('brush_cursor'))
        self.canvas.bind('<MouseWheel>',self.wheel)
        self.canvas.bind('<ButtonPress-3>',self.pan_start)
        self.canvas.bind('<B3-Motion>',self.pan_move)
        tk.Label(main,text=self.t('按住左键涂抹 · 滚轮缩放 · 右键平移','Left drag: paint · Wheel: zoom · Right drag: pan'),
            bg='#101b26',fg='#b6c9d9',pady=8).pack(fill='x')
        sidebar=tk.Frame(root,bg='#142330');sidebar.pack(side='right',fill='y')
        scroll=ttk.Scrollbar(sidebar,orient='vertical');scroll.pack(side='right',fill='y')
        self.side_canvas=tk.Canvas(sidebar,width=380 if self.result.is_file() else 320,bg='#142330',highlightthickness=0,yscrollcommand=scroll.set)
        self.side_canvas.pack(side='left',fill='y');scroll.configure(command=self.side_canvas.yview)
        side=tk.Frame(self.side_canvas,bg='#142330',padx=14,pady=14)
        self.side_canvas.create_window((0,0),window=side,anchor='nw',width=380 if self.result.is_file() else 320)
        side.bind('<Configure>',lambda e:self.side_canvas.configure(scrollregion=self.side_canvas.bbox('all')))
        self.save_state=tk.StringVar(value=self.t('没有未保存的更改','No unsaved changes'))
        tk.Label(side,textvariable=self.save_state,bg='#142330',fg='#e7c27d',wraplength=285,justify='left').pack(fill='x',pady=4)
        self.task_feedback=tk.StringVar(value='')
        tk.Label(side,textvariable=self.task_feedback,bg='#142330',fg='#9ae1cf',wraplength=285,justify='left').pack(fill='x')
        self.draft_feedback=tk.StringVar(value='草稿：等待来源加载')
        tk.Label(side,textvariable=self.draft_feedback,bg='#142330',fg='#a4bdcf',wraplength=285,justify='left').pack(fill='x')
        self.cancel_task_button=ttk.Button(side,text='取消当前任务',command=self.cancel_planning)
        self.cancel_task_button.pack(fill='x',pady=3)
        self.workspace_tabs=None
        advanced_side=side;save_side=side;plan_side=side
        if self.result.is_file():
            self.workspace_tabs=ttk.Notebook(side);self.workspace_tabs.pack(fill='both',expand=True)
            tabs=[]
            for title in ('1 区域与禁区','2 路线演示','3 保存分享','高级'):
                tab=tk.Frame(self.workspace_tabs,bg='#142330');self.workspace_tabs.add(tab,text=title);tabs.append(tab)
            side,plan_side,save_side,advanced_side=tabs
        self.info=tk.StringVar(value='—')
        tk.Label(advanced_side,textvariable=self.info,bg='#142330',fg='#e3eef6',wraplength=285,justify='left',anchor='w').pack(fill='x',pady=6)
        self.readout=tk.StringVar(value='X —  Y —  Z —')
        tk.Label(advanced_side,textvariable=self.readout,bg='#142330',fg='#70e1c3',font=('Consolas',13),wraplength=285,justify='left',anchor='w').pack(fill='x',pady=10)
        self.mode=tk.StringVar(value='region')
        row=tk.Frame(side,bg='#142330');row.pack(fill='x')
        if 'point_tools' in PRODUCT_HIDDEN_FEATURES:
            tk.Label(row,text=self.t('方形涂抹','Square brush'),bg='#142330',fg='#e3eef6').pack(side='left')
        else:
            for value,zh,en in [('point','选点 / 测距','Point / distance'),('region','方形涂抹','Square brush')]:
                w=ttk.Radiobutton(row,text=self.t(zh,en),value=value,variable=self.mode,command=self.change_mode)
                w.pack(side='left');self.widgets.append(w)
        self.selection_method=tk.StringVar(value='work');self.active_selection_method='work'
        tk.Label(side,text=self.t('方形涂抹类型（结果自动判断）','Square paint type (result is automatic)'),bg='#142330',fg='#c4d8e8').pack(anchor='w',pady=(6,0))
        method_row=tk.Frame(side,bg='#142330');method_row.pack(fill='x')
        for value,zh,en in [('work','工作区域','Work area'),('forbidden','禁止区域','Forbidden area')]:
            w=ttk.Radiobutton(method_row,text=self.t(zh,en),value=value,variable=self.selection_method,
                command=self.change_selection_method)
            w.pack(anchor='w');self.widgets.append(w)
        self.brush_size=tk.IntVar(value=40)
        self.brush_erase=tk.BooleanVar(value=False)
        brush_row=ttk.Frame(side);brush_row.pack(fill='x',pady=4)
        ttk.Label(brush_row,text=self.t('方块大小（屏幕像素）','Square size (screen px)')).pack(side='left')
        self.brush_size_label=ttk.Label(brush_row,text='40');self.brush_size_label.pack(side='right')
        brush_scale=tk.Scale(side,from_=8,to=160,orient='horizontal',variable=self.brush_size,
            showvalue=False,command=self.brush_resize,bg='#142330',fg='#dae8ef',highlightthickness=0)
        brush_scale.pack(fill='x');self.widgets.append(brush_scale)
        w=ttk.Checkbutton(side,text=self.t('擦除已涂部分','Erase painted selection'),variable=self.brush_erase,
            command=lambda:self.draw_brush_cursor())
        w.pack(anchor='w');self.widgets.append(w)
        self.overlay=tk.BooleanVar(value=False)
        self.display_mode=tk.StringVar(value='rgb')
        display_row=ttk.Frame(side);display_row.pack(fill='x',pady=5)
        for value,zh,en in [('rgb','RGB','RGB'),('cloud','纯点云','Cloud'),('distance','距原点着色','W0 distance')]:
            ttk.Radiobutton(display_row,text=self.t(zh,en),value=value,variable=self.display_mode,command=self.draw).pack(side='left')
        ttk.Checkbutton(side,text=self.t('显示三维支持范围','Show geometry support'),variable=self.overlay,command=self.draw).pack(anchor='w',pady=5)
        self.show_axes=tk.BooleanVar(value=True)
        self.show_optical_axis=tk.BooleanVar(value=False)
        ttk.Checkbutton(side,text=self.t('显示坐标方向（W0）','Show W0 directions'),variable=self.show_axes,command=self.draw).pack(anchor='w',pady=3)
        ttk.Checkbutton(side,text=self.t('显示光轴参考线','Show optical-axis guides'),variable=self.show_optical_axis,command=self.draw).pack(anchor='w',pady=3)
        self.action(side,'完成区域选择','Finish area selection',self.finish_polygon)
        self.exclusion_target=None
        if 'legacy_exclusion_editor' not in PRODUCT_HIDDEN_FEATURES:
            self.action(side,'添加禁止区域（先选主区域）','Draw forbidden area in selected region',self.begin_exclusion)
            self.action(side,'清除所选区域禁区 / 可撤销','Clear exclusions / undoable',self.clear_exclusions)
        tk.Label(side,text=self.t('名称 / 备注','Name / note'),bg='#142330',fg='#c4d8e8').pack(anchor='w')
        self.name=tk.StringVar(value=self.t('标记 01','Marker 01'));self.note=tk.StringVar()
        for var in (self.name,self.note):
            w=ttk.Entry(side,textvariable=var,width=32);w.pack(fill='x',pady=3);self.widgets.append(w)
        for var in (self.name,self.note):var.trace_add('write',lambda *args:self.update_save_state())
        self.items=tk.Listbox(side,height=5,selectmode='extended',exportselection=False,bg='#0e1923',fg='#dae8ef')
        self.items.pack(fill='x',pady=6);self.items.bind('<<ListboxSelect>>',self.select);self.widgets.append(self.items)
        self.items.bind('<Control-a>',self.select_all)
        self.action(side,'全选标记','Select all annotations',self.select_all)
        tk.Label(side,text=self.t('Ctrl 多选 · Shift 连选 · Ctrl+A 全选','Ctrl: multi-select · Shift: range · Ctrl+A: all'),bg='#142330',fg='#a4bdcf').pack(anchor='w')
        if not PRODUCT_SINGLE_REGION:
            for zh,en,fn in [('应用名称备注','Apply edits',self.edit),('删除所选 / 可撤销','Delete selected',self.delete),
                ('清除当前选中','Clear selection',self.clear_selection)]:
                self.action(side,zh,en,fn)
        self.action(side,'撤销','Undo',self.undo)
        self.action(save_side,'保存并导出当前结果 GLB / PLY','Save and export current result GLB / PLY',self.save)
        if 'annotation_load' not in PRODUCT_HIDDEN_FEATURES:
            self.action(advanced_side,'回读标记','Load annotations',self.load)
        self.action(side,'复位画面','Fit image',self.fit)
        if 'measurement_tools' not in PRODUCT_HIDDEN_FEATURES:
            self.action(advanced_side,'记录实测距离（选中两点）','Record reference distance (select 2 points)',self.record_distance_test)
            self.action(advanced_side,'同位置几何诊断（选中一点）','Local geometry evidence (select one point)',self.local_geometry_diagnostics)
            self.action(advanced_side,'数据质量 / 精度测试报告','Data quality / distance test report',self.measurement_report)
            self.action(advanced_side,'打开精度测试记录','Open distance test records',self.open_measurement_records)
            tk.Label(advanced_side,text=self.t('测量：派生几何 · 展示：RGB / GLB\n绝对 XYZ 精度未验证；实测距离不等于治疗定位验收。',
                'Measurement: derived geometry; display: RGB / GLB\nAbsolute XYZ accuracy unverified; distance tests are not clinical validation.'),
                bg='#142330',fg='#e7c27d',wraplength=285,justify='left').pack(fill='x',pady=6)
        self.action(save_side,'历史区域导出记录','Region export history',self.export_history)
        if 'project_management' not in PRODUCT_HIDDEN_FEATURES:
            self.action(save_side,'保存方案','Save plan',self.save_region_project)
            self._plan_buttons['save_region_project_as']=self.action(save_side,'方案另存为…','Save plan as…',lambda:self.save_region_project(save_as=True))
            self.action(save_side,'恢复或放弃草稿…','Recover or discard draft…',self.offer_recovery)
            self.action(save_side,'打开已有方案…','Open portable region workspace',self.open_region_project)
        self.planning_tools=tk.Frame(plan_side,bg='#142330')
        if self.result.is_file():self.planning_tools.pack(fill='x')
        self.action(advanced_side if self.workspace_tabs else self.planning_tools,'像素扫描路径（预览）','Pixel raster path preview',self.plan_region_path)
        self.action(advanced_side if self.workspace_tabs else self.planning_tools,'毫米曲面路径（近似）','Metric surface contours (approx.)',self.plan_metric_path)
        self.cancel_plan_button=ttk.Button(self.planning_tools,text='取消当前规划任务',command=self.cancel_planning)
        self.cancel_plan_button.pack(fill='x',pady=3)
        self.action(self.planning_tools,'固定RGB蛇形覆盖','Fixed RGB serpentine coverage',self.plan_coverage_path)
        self.action(self.planning_tools,'三维表面间距规划','3D surface spacing',self.plan_surface_spacing)
        self.action(self.planning_tools,'播放路线 / 检查覆盖','Route diagram / export PNG',self.present_region_path)
        self.action(save_side if self.workspace_tabs else self.planning_tools,'导出分享包（模型＋路线）','Export region + path / diagram / ZIP',self.export_region_path)
        self.action(advanced_side if self.workspace_tabs else self.planning_tools,'回读并核验路径 JSON','Load / validate path JSON',self.load_region_path)
        self.action(advanced_side if self.workspace_tabs else self.planning_tools,'清除路径预览','Clear path preview',self.clear_path_preview)
        tk.Label(side,text=self.t('RGB 照片不代表每个像素有深度。\n绿色：几何支持；青色：首帧缺深度处获得几何支持。\n坐标为估计值，未做实物精度验证；仅用于研发。',
            'RGB visibility does not guarantee depth.\nGreen: geometry support; cyan: support where reference depth is missing.\nEstimated coordinates; physical accuracy unverified. R&D only.'),
            bg='#142330',fg='#a4bdcf',wraplength=285,justify='left').pack(fill='x',pady=10)
        def scroll_side(e):
            self.side_canvas.yview_scroll(-1 if e.delta>0 else 1,'units');return 'break'
        def bind_scroll(widget):
            widget.bind('<MouseWheel>',scroll_side)
            for child in widget.winfo_children():bind_scroll(child)
        bind_scroll(side);self.side_canvas.bind('<MouseWheel>',scroll_side)
        from region_project import load_source
        self.work(lambda:load_source(result,cancel=self.operation_cancel(),progress=self.operation_progress),self.loaded)
        self.timer=root.after(80,self.poll)

    def t(self,zh,en):return zh if self.language=='zh' else en
    def error_text(self,error):
        translations={
            'Geometry task cancelled':'几何任务已取消',
            'Geometry operation exceeded its 120 s budget':'几何操作超过 120 秒预算',
            'Fixed RGB mapping exceeds the image / mesh memory budget':'固定 RGB 映射超出图像或网格内存预算',
            'Fixed RGB mesh exceeds the 4000000 triangle memory budget':'固定 RGB 网格超出 400 万三角面的内存预算',
            'Mesh file exceeds the 512 MiB input budget':'网格文件超过 512 MiB 回读预算',
            'A bounded PLY header is required for fixed RGB mapping':'固定 RGB 映射需要完整且不超过 64 KiB 的 PLY 文件头',
            'Invalid reference depth range':'参考深度范围无效',
            'Annotation count exceeds the 2000 item memory budget':'标记数量超过 2000 项的内存预算',
            'Region boundary exceeds the 300000 pixel display-sample budget':'区域轮廓超过 30 万显示采样点预算，请缩小区域或使用高级三维检查',
            'Annotation overlays exceed the 64 MiB display memory budget':'标记叠加超出 64 MiB 显示内存预算',
            'Annotation file exceeds the 64 MiB input budget':'标记文件超过 64 MiB 回读预算',
            'Region image coordinates / units mismatch':'区域的图像坐标系或像素单位不匹配',
            'Invalid annotation document':'标记文档格式无效',
            'Reference RGB evidence mismatch':'参考 RGB / Depth 来源证据不匹配',
            'Source model / coordinate frame mismatch':'模型来源、几何指纹、坐标系或单位不匹配',
            'Capture session / reference frame mismatch':'采集会话或初始参考帧不匹配',
            'Origin reference provenance mismatch':'初始原点来源证据不匹配',
            'Initial color-camera origin cannot be verified; use the advanced viewer for this model.':'无法验证初始彩色相机原点，请使用高级三维查看器检查此模型。',
        }
        return translations.get(error,error) if self.language=='zh' else error
    def action(self,parent,zh,en,fn):
        b=ttk.Button(parent,text=self.t(zh,en),command=fn);b.pack(fill='x',pady=3);self.widgets.append(b)
        name=getattr(fn,'__name__','')
        if name in ('plan_region_path','plan_metric_path','plan_surface_spacing','plan_coverage_path','present_region_path','export_region_path','load_region_path','clear_path_preview','save_region_project'):
            self._plan_buttons[name]=b
        return b
    def work(self,fn,done):
        if self.busy or self.closed:return
        if getattr(self,'_draft_ready',False):
            self._draft_due=0;self.poll_draft()
        with self._task_lock:
            self.generation+=1;generation=self.generation
            from planning_io import TaskControl
            self.task_cancel=TaskControl(self.operation);cancel=self.task_cancel
            while not self.mailbox.empty():
                try:self.mailbox.get_nowait()
                except queue.Empty:break
        self.task_started=time.monotonic();self.task_timed_out=False
        labels={'load_model':'加载工程','path_preview':'生成/核验路径','save':'保存方案','export':'导出分享包','restore_draft':'恢复草稿'}
        self.task_cancel.stage=labels.get(self.operation,'处理几何/标记')
        self.busy=True
        if self.operation=='path_preview':
            from planning_state import region_signature
            import copy
            self._task_region_id=self.planning_region_id()
            region=self.planning_region()
            self._task_region_signature=region_signature(region) if region else None
            self._task_request=copy.deepcopy(self._requests.get(self._task_region_id))
            self._plan_errors.pop(self._task_region_id,None)
            self.status.set(self.planning_state()['reason'])
        self.update_save_state()
        for w in self.widgets:w.configure(state='disabled')
        def run():
            self._task_local.cancel=cancel
            try:value,error=fn(),None
            except Exception as exc:value,error=None,str(exc)
            with self._task_lock:
                if not self.closed and generation==self.generation:
                    try:self.mailbox.put_nowait((generation,done,value,error))
                    except queue.Full:pass
        # Reading/raycasting can exit cooperatively; a save is allowed to finish
        # its current file write before exiting, and publishes JSON only last.
        self.worker=threading.Thread(target=run,daemon=self.operation not in ('save','export'),name='FixedRGBGeometry')
        self.worker.start()
    def operation_progress(self,stage):
        control=self.operation_cancel()
        if hasattr(control,'progress'):control.progress(stage)

    def operation_commit(self,action):
        control=self.operation_cancel()
        if hasattr(control,'publish'):return control.publish(action)
        return action()

    def operation_cancel(self):
        return getattr(self._task_local,'cancel',self.task_cancel)
    def poll(self):
        if self.closed:return
        if self.busy:
            self.task_feedback.set(f'{getattr(self.task_cancel,"stage",self.operation)} · {time.monotonic()-self.task_started:.1f} 秒'+(' · 正在取消' if self.task_cancel.is_set() else ''))
        self.poll_draft()
        if self.busy and not getattr(self.task_cancel,'committed',False) and not self.task_timed_out and time.monotonic()-self.task_started>120:
            self.task_timed_out=True;self.task_cancel.set()
            self.status.set(self.t('操作超过 120 秒预算，正在取消；可关闭此窗口。','Operation exceeded the 120 s budget; cancelling. This window can be closed.'))
        try:
            generation,done,value,error=self.mailbox.get_nowait()
            if generation!=self.generation:
                self.timer=self.root.after(80,self.poll);return
            self.busy=False
            if self.operation=='path_preview' and not error:
                from planning_state import region_signature
                region=self.planning_region()
                if (not region or region_signature(region)!=self._task_region_signature or
                        self._requests.get(self._task_region_id)!=self._task_request):
                    error='规划期间区域或参数已改变，未采用过期结果'
            if self.operation=='path_preview' and (error or self.task_timed_out):
                self._plan_errors[getattr(self,'_task_region_id',None)]=error or '操作超时/已取消'
            for w in self.widgets:w.configure(state='normal')
            if self.task_cancel.is_set() and not error and not getattr(self.task_cancel,'committed',False):
                error='几何任务已取消'
                self._plan_errors[getattr(self,'_task_region_id',None)]=error
            if self.task_timed_out:
                self.status.set(self.t('操作已超时；未采用迟到界面结果。','Operation timed out; late UI result was not applied.'))
            elif error:self.status.set(self.t('操作失败，当前标记与草稿保留：','Operation failed; current annotations and draft retained: ')+self.error_text(error))
            else:
                try:done(value)
                except Exception as exc:
                    error=str(exc)
                    if self.operation=='path_preview':self._plan_errors[getattr(self,'_task_region_id',None)]=str(exc)
                    self.status.set('结果未采用：'+str(exc))
            if self.model is not None:self._annotations_dirty=self.model.store.dirty
            self.task_feedback.set(('已取消' if self.task_cancel.is_set() else '失败' if error or self.task_timed_out else '完成')+f' · {time.monotonic()-self.task_started:.1f} 秒')
            self.update_save_state()
        except queue.Empty:pass
        self.timer=self.root.after(80,self.poll)
    def loaded(self,model):
        if (model.store.local_context or {}).get('capture_scope') == 'interrupted_prefix':
            self.root.title(self.t('Femto · 中断前数据建模 · 固定 RGB 选区',
                                   'Femto · Interrupted-prefix reconstruction · Fixed RGB'))
        if self.model is not None and self.model is not model:
            from planning_drafts import Journal
            self.draft_journal=Journal(getattr(model,'project_source_path',model.reconstruction))
            self._draft_ready=False;self._draft_due=None;self._draft_signature=None
            self.project_path=getattr(model,'project_source_path',None)
            self._project_region_id=None
            with self._task_lock:
                self.generation+=1;self.task_cancel.set()
            self.busy=False
            for w in self.widgets:w.configure(state='normal')
        self.polygon=[];self.reset_brush();self.clear_readout()
        self.model=model
        self._plans={};self._requests={};self._plan_errors={};self._saved_plans={};self._saved_requests={}
        self.base=Image.fromarray(model.rgb)
        from fixed_rgb_geometry import vertex_cloud_view
        self.cloud_view=vertex_cloud_view(model.store.mesh,model.projection)
        if hasattr(model,'support_rgb'):self.support_image=Image.fromarray(model.support_rgb)
        else:
            tint=model.rgb.astype(float).copy();valid=model.projection.valid;new=model.projection.new_support
            tint[valid]=tint[valid]*.6+np.array([40,220,90])*.4
            tint[new]=model.rgb[new]*.5+np.array([40,210,240])*.5
            self.support_image=Image.fromarray(tint.astype(np.uint8))
        self.info.set(self.t('固定初始彩色相机 W0','Initial color camera W0')+
            f"\n{model.reconstruction.parent.name}\n"+self.t('参考帧：','Reference frame: ')+str(model.view_reference['reference_frame_id'])+
            '\nRDF · X '+self.t('右 / Y 下 / Z 前','right / Y down / Z forward')+
            self.t('\n本模型接纳位姿最大半径：','\nMax accepted pose radius: ')+f'{model.max_radius_mm:.1f} mm')
        self.status.set(self.t('画面已锁定。请选择有三维支持的表面。','View locked. Select a supported surface.')+
            (self.t(' 此历史模型超出当前 30 mm 补采半径。',' This historical model exceeds the current 30 mm capture radius.') if model.max_radius_mm>30 else ''))
        self.fit()
        self.refresh_items()
        if hasattr(model,'project_id'):
            self.planning_tools.pack(fill='x')
            self.root.title('Femto · 独立区域规划工作台（无需相机）')
            region=model.store.regions[0]
            self._project_region_id=region['annotation_id']
            self.refresh_items(('region',region['annotation_id']))
            from region_path import signature
            if model.saved_plan is not None:
                self.show_region_path(model,model.saved_plan)
                self._saved_plans[region['annotation_id']]=signature(model.saved_plan)
            if getattr(model,'planning_parameters',None) is not None:
                self._requests[region['annotation_id']]=model.planning_parameters
            import copy
            self._saved_requests=copy.deepcopy(self._requests)
            self.update_save_state()
            self.status.set(getattr(model,'plan_warning',None) or '区域工程已打开，可编辑禁区、重新规划并保存方案；不依赖原采集目录。')
        self._draft_ready=True
        if 'project_management' not in PRODUCT_HIDDEN_FEATURES:self.offer_recovery()

    def fit(self):
        self.zoom=1.;self.pan=np.zeros(2);self.draw()
    def draw(self):
        if self.model is None:return
        selected_items=set(self.selected_annotations())
        if not selected_items and self.focus is not None:selected_items.add(self.focus)
        selected_regions=[r for i,r in enumerate(self.model.store.regions) if ('region',i) in selected_items]
        w,h=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        iw,ih=self.base.size;self.scale=min(w/iw,h/ih)*self.zoom
        size=(max(1,round(iw*self.scale)),max(1,round(ih*self.scale)))
        self.offset=(np.array([w,h])-size)/2+self.pan
        image=self.support_image if self.overlay.get() else self.base
        # Render only the visible crop; zoom must not allocate giant full images.
        affine=(1/self.scale,0,-self.offset[0]/self.scale,0,1/self.scale,-self.offset[1]/self.scale)
        rendered=image.transform((w,h),Image.Transform.AFFINE,affine,Image.Resampling.BILINEAR)
        cloud_mask=None
        if self.display_mode.get()!='rgb':
            from fixed_rgb_geometry import render_vertex_cloud
            pixels,cloud_mask=render_vertex_cloud(self.cloud_view,w,h,self.scale,self.offset,self.display_mode.get()=='distance')
            rendered=Image.fromarray(pixels)
        if selected_regions:
            from fixed_rgb_geometry import selected_surface_mask
            triangle_ids=sorted({tid for region in selected_regions for tid in region['triangle_indices']})
            mask=selected_surface_mask(self.model.projection,triangle_ids,w,h,self.scale,self.offset)
            if cloud_mask is not None:mask &= cloud_mask
            pixels=np.asarray(rendered).copy()
            pixels[mask]=(pixels[mask]*.55+np.array([255,102,127])*.45).astype(np.uint8)
            rendered=Image.fromarray(pixels)
        self.photo=ImageTk.PhotoImage(rendered)
        self.canvas.delete('all');self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        self.draw_coordinate_guides(w,h)
        plan=self.current_region_path()
        if plan:
            for segment in plan['segments']:
                uv=np.asarray([sample['pixel'] for sample in segment])*self.scale+self.offset
                self.canvas.create_line(*uv.ravel().tolist(),fill='#ffdb4d',width=2,tags='path_preview')
        if self.display_mode.get()!='rgb':
            label=self.t('网格顶点点云 · 固定视角 · 无 RGB 背景','Mesh vertex cloud · fixed view · no RGB background')
            if self.display_mode.get()=='distance':
                low,high=self.cloud_view['limits']
                label+=self.t(f'\n距 W0：蓝 {low:.1f} → 红 {high:.1f} mm',f'\nW0 distance: blue {low:.1f} → red {high:.1f} mm')
            self.canvas.create_text(12,h-12,anchor='sw',text=label,fill='#ffffff',tags='cloud_legend')
        for i,p in enumerate(self.model.store.points):
            uv=self.model.display_for('point',p)
            if uv is None:continue
            x,y=np.asarray(uv)*self.scale+self.offset
            selected=self.focus==('point',i) or ('point',i) in self.selected_annotations();radius=7 if selected else 4
            self.canvas.create_oval(x-radius,y-radius,x+radius,y+radius,
                fill='#ff667f' if selected else '#ffb431',outline='white',width=2 if selected else 1,
                tags=('annotation','selected' if selected else 'ordinary','point',p['annotation_id']))
        for i,region in enumerate(self.model.store.regions):
            if ('region',i) not in selected_items:continue
            for polygon in region.get('exclusion_polygons',[]):
                xy=np.asarray(polygon)*self.scale+self.offset
                self.canvas.create_polygon(*xy.ravel(),fill='#e34141',stipple='gray50',outline='#ff4242',width=3,tags='forbidden')
            # Cached display-only segments were clipped against nearest hits in
            # the worker. Neither the world boundary nor triangle IDs are moved.
            selected=True
            segments=self.model.display_for('region',region)
            for segment in segments if segments is not None else []:
                xy=np.asarray(segment)*self.scale+self.offset
                self.canvas.create_line(*xy.ravel(),fill='#ff667f' if selected else '#50e69f',
                    width=4 if selected else 2,tags=('annotation','selected' if selected else 'ordinary','region',region['annotation_id']))
        if not selected_regions:self.draw_brush_overlay(affine,w,h)
        self.draw_brush_cursor()

    def draw_coordinate_guides(self,w,h):
        # W0 is the reference camera centre (Z=0), which cannot be projected.
        # The crosshair is the calibrated principal point, never a surface origin.
        if self.show_optical_axis.get():
            k=self.model.projection.k
            x,y=np.array([k[0,2],k[1,2]])*self.scale+self.offset
            iw,ih=self.base.size
            left,top=np.maximum(self.offset,[0,0])
            right,bottom=np.minimum(self.offset+np.array([iw,ih])*self.scale,[w,h])
            if left<right and top<bottom:
                if top<=y<=bottom:self.canvas.create_line(left,y,right,y,fill='#ff7777',dash=(6,4),tags='coordinate_guide')
                if left<=x<=right:self.canvas.create_line(x,top,x,bottom,fill='#70e1a2',dash=(6,4),tags='coordinate_guide')
                if left<=x<=right and top<=y<=bottom:
                    self.canvas.create_oval(x-5,y-5,x+5,y+5,outline='#7fbdff',width=2,tags='coordinate_guide')
                    self.canvas.create_text(x+9,y+9,anchor='nw',text=self.t('光轴中心（非原点）','Optical axis (not W0)'),fill='#ffffff',tags='coordinate_guide')
        if self.show_axes.get():
            c=self.canvas;tag='coordinate_guide'
            c.create_rectangle(12,12,268,142,fill='#101b26',outline='#395268',tags=tag)
            c.create_text(24,24,anchor='nw',text=self.t('W0 坐标方向示意 · 无比例','W0 directions · not to scale'),fill='#dceaf4',tags=tag)
            x,y=40,65
            c.create_line(x,y,x+65,y,arrow='last',fill='#ff7777',width=2,tags=tag)
            c.create_text(x+70,y,text='+X',anchor='w',fill='#ff7777',tags=tag)
            c.create_line(x,y,x,y+38,arrow='last',fill='#70e1a2',width=2,tags=tag)
            c.create_text(x+8,y+37,text='+Y',anchor='w',fill='#70e1a2',tags=tag)
            c.create_oval(x-5,y-5,x+5,y+5,outline='#7fbdff',tags=tag)
            c.create_line(x-3,y-3,x+3,y+3,fill='#7fbdff',tags=tag)
            c.create_line(x-3,y+3,x+3,y-3,fill='#7fbdff',tags=tag)
            c.create_text(150,80,text=self.t('+Z 向前\n进入画面','+Z forward\ninto image'),anchor='w',fill='#7fbdff',tags=tag)
            c.create_text(24,120,anchor='nw',text=self.t('原点在初始相机光心','W0: initial camera optical centre'),fill='#a4bdcf',tags=tag)

    def image_pixel(self,x,y):
        return (np.array([x,y])-self.offset)/self.scale
    def wheel(self,e):
        self.brush_end()
        if self.model is None:return
        pixel=self.image_pixel(e.x,e.y);old=self.zoom
        self.zoom=float(np.clip(self.zoom*(1.2 if e.delta>0 else 1/1.2),1,6))
        ratio=self.zoom/old
        center=np.array([self.canvas.winfo_width(),self.canvas.winfo_height()])/2
        self.pan=np.array([e.x,e.y])-center-(self.offset+pixel*self.scale-center-self.pan)*ratio
        self.draw()
    def pan_start(self,e):
        self.brush_end();self.drag=np.array([e.x,e.y])
    def pan_move(self,e):
        here=np.array([e.x,e.y]);self.pan+=here-getattr(self,'drag',here);self.drag=here;self.draw()
    def pending_edits(self):
        if self.model is None or self.focus is None:return False
        kind,i=self.focus
        items=self.model.store.points if kind=='point' else self.model.store.regions
        if i>=len(items):return False
        item=items[i]
        return self.name.get().strip()!=item.get('name','') or self.note.get().strip()!=item.get('note','')
    def cancel_planning(self):
        if self.busy:
            self.task_cancel.set()
            self.status.set('结果已完整发布，取消已不适用。' if getattr(self.task_cancel,'committed',False) else '正在取消任务；保留原有有效文件和未保存内容。')

    def poll_draft(self):
        if not getattr(self,'_draft_ready',False) or self.model is None:return
        journal=self.draft_journal
        if journal.error:
            self.draft_feedback.set('草稿写入失败（正式方案不受影响）：'+journal.error)
            self._draft_signature=None
            if self._draft_due is None:self._draft_due=time.monotonic()+5
        elif journal.saved_at:self.draft_feedback.set('可恢复草稿已写入 '+time.strftime('%H:%M:%S',time.localtime(journal.saved_at)))
        if self.busy or self._draft_due is None or time.monotonic()<self._draft_due:return
        self._draft_due=None
        if not (self.model.store.dirty or self.plan_unsaved() or self.pending_edits() or self.has_selection_draft()):
            try:journal.discard();self._draft_signature=None;self.draft_feedback.set('当前内容已保存，无恢复草稿')
            except OSError as error:self.draft_feedback.set('草稿清理失败：'+str(error))
            return
        from planning_drafts import snapshot
        from region_path import signature
        data=snapshot(self);content=dict(data);content.pop('created_at')
        digest=signature(content)
        if digest!=self._draft_signature:
            self._draft_signature=digest;journal.submit(data);self.draft_feedback.set('正在写入可恢复草稿…')

    def offer_recovery(self):
        if self.model is None or self.busy:return
        if not self.draft_journal.path.exists():
            self.draft_feedback.set('草稿：修改后自动保存');return
        self._draft_ready=False
        from planning_drafts import read
        try:data=read(self.draft_journal.path,self.model)
        except Exception as error:
            self.draft_feedback.set('草稿未恢复：'+str(error));return
        answer=messagebox.askyesnocancel('发现可恢复草稿',
            '发现未提交的区域、禁区和参数草稿。恢复吗？\n是：恢复草稿，路径需重新生成；否：放弃草稿；取消：保留草稿，暂不恢复。',parent=self.root)
        if answer is None:return
        if not answer:
            try:self.draft_journal.discard()
            except OSError as error:self.draft_feedback.set('无法放弃草稿：'+str(error));return
            self._draft_ready=True;self.draft_feedback.set('已放弃恢复草稿');return
        model=self.model
        def restore():
            import copy
            candidate=copy.copy(model);candidate.store=copy.copy(model.store);candidate._overlay_cache={}
            candidate.store.regions=data['regions'];candidate.store.points=data['points']
            candidate.prepare_display(cancel=self.operation_cancel())
            GeometryBudget(self.operation_cancel()).check()
            return candidate
        def done(candidate):
            model.store.regions=candidate.store.regions;model.store.points=candidate.store.points
            model.store._history=[];model._overlay_cache=candidate._overlay_cache
            self._plans={};self._requests=data['parameters'];self._plan_errors={}
            self.refresh_items(('region',data['focus']) if data.get('focus') else None)
            self.reset_brush();self.exclusion_target=data.get('exclusion_target')
            legacy_role={'include':'work','exclude':'forbidden'}.get(data.get('selection_method'),'work')
            role=data.get('paint_role',legacy_role)
            selections=data.get('selection_polygons') or {'work':[],'forbidden':[]}
            if 'selection_polygons' not in data:selections[role]=data.get('polygon',[])
            self._selection_polygons={key:[p[:] for p in selections[key]] for key in ('work','forbidden')}
            self.selection_method.set(role);self.active_selection_method=role
            self.polygon=[p[:] for p in self._selection_polygons[role]]
            self.mode.set(data['mode']);self.active_mode=data['mode']
            self._syncing_fields=True;self.name.set(data['name']);self.note.set(data['note']);self._syncing_fields=False
            self._draft_ready=True;self.update_save_state();self.draw()
            self.draft_feedback.set('已恢复草稿，尚未保存正式方案；路径需重新生成')
        self.operation='restore_draft';self.work(restore,done)

    def planning_region(self):
        if self.model is None or not self.focus or self.focus[0]!='region':return None
        if not 0<=self.focus[1]<len(self.model.store.regions):return None
        return self.model.store.regions[self.focus[1]]

    def planning_region_id(self):
        region=self.planning_region()
        return region['annotation_id'] if region else None

    def planning_state(self):
        from planning_state import state
        region=self.planning_region();key=self.planning_region_id()
        return state(region,self._plans.get(key),dirty=bool(self.model and self.model.store.dirty) or self.plan_unsaved(),
            pending=self.pending_edits() or self.has_selection_draft(),busy=self.busy,
            error=self._plan_errors.get(key),request=self._requests.get(key),
            geometry_digest=self.model.store.digest if self.model else None,draft=self.has_selection_draft())

    def plan_unsaved(self):
        from region_path import signature
        if self._requests!=self._saved_requests:return True
        return any(signature(plan)!=self._saved_plans.get(key) for key,plan in self._plans.items()) or any(
            key not in self._plans for key in self._saved_plans)

    def update_save_state(self):
        if self._syncing_fields or not hasattr(self,'save_state'):return
        state=self.planning_state()
        labels={'saved':'没有未保存的更改','dirty':'区域/方案更改尚未保存','pending':'区域待保存：请先完成涂抹或应用名称 / 备注'}
        states=[labels[state['save_state']],state['reason']]
        if state['phase']=='failed' and state['valid']:states.append('原有同参数路径仍有效，已保留')
        if not state['can_use']:states.append('播放、报告与导出不可用：'+state['reason'])
        self.save_state.set('\n'.join(states))
        if hasattr(self,'cancel_plan_button'):
            self.cancel_plan_button.configure(state='normal' if self.busy and self.operation=='path_preview' else 'disabled')
        for name,button in self._plan_buttons.items():
            enabled=(state['can_save'] if name.startswith('save_region_project') else
                     state['can_use'] if name in ('present_region_path','export_region_path') else state['can_generate'])
            button.configure(state='normal' if enabled else 'disabled')
        if hasattr(self,'cancel_task_button'):self.cancel_task_button.configure(state='normal' if self.busy and not getattr(self.task_cancel,'committed',False) else 'disabled')
        if getattr(self,'_draft_ready',False) and not self.busy:self._draft_due=time.monotonic()+.75
        demo=getattr(self.root,'_route_demo_window',None)
        if demo is not None and demo.winfo_exists() and not demo.plan_is_current():demo.close_demo()

    def confirm_pending_edits(self):
        if not self.pending_edits():return True
        kind,i=self.focus
        if PRODUCT_SINGLE_REGION and kind=='region':
            self.model.store.update_annotation(kind,i,name=self.name.get().strip(),note=self.note.get().strip())
            self._annotations_dirty=self.model.store.dirty;self.update_save_state()
            return True
        answer=messagebox.askyesnocancel(self.t('名称与备注尚未应用','Unapplied name / note'),
            self.t('先将名称和备注应用到当前标记？\n是：应用；否：放弃字段编辑；取消：保留当前操作。',
                   'Apply the name and note to the current annotation first?\nYes: apply. No: discard field edits. Cancel: stay here.'),parent=self.root)
        if answer is None:return False
        if answer:self.model.store.update_annotation(kind,i,name=self.name.get().strip(),note=self.note.get().strip())
        item=(self.model.store.points if kind=='point' else self.model.store.regions)[i]
        self._syncing_fields=True;self.name.set(item.get('name',''));self.note.set(item.get('note',''));self._syncing_fields=False
        self._annotations_dirty=self.model.store.dirty;self.update_save_state()
        return True
    def confirm_discard(self,action):
        if self.model is not None:self._annotations_dirty=self.model.store.dirty
        self.update_save_state()
        if not (self._annotations_dirty or self.plan_unsaved() or self.pending_edits() or self.has_selection_draft()):return True
        return messagebox.askokcancel(self.t('尚有未保存内容','Unsaved work'),
            self.save_state.get()+'\n\n'+self.t(f'{action}会放弃这些未保存内容。继续？',f'{action} will discard this unsaved work. Continue?'),parent=self.root)
    def change_mode(self):
        if self.has_selection_draft() and not messagebox.askokcancel(self.t('涂抹尚未完成','Unfinished painting'),
            self.t('切换模式将放弃当前涂抹草稿。继续？','Changing mode discards the current polygon draft. Continue?'),parent=self.root):
            self.mode.set(self.active_mode);return
        if not self.confirm_pending_edits():self.mode.set(self.active_mode);return
        self.exclusion_target=None
        self.active_mode=self.mode.get();self.polygon=[];self.reset_brush();self.clear_readout();self.update_save_state();self.draw()
    def change_selection_method(self):
        self.brush_end();method=self.selection_method.get()
        self._selection_polygons[self.active_selection_method]=[p[:] for p in self.polygon]
        self.active_selection_method=method;self.exclusion_target=None
        self.mode.set('region');self.active_mode='region'
        self.polygon=[p[:] for p in self._selection_polygons[method]]
        self._brush_mask=None;self._brush_last=None;self.clear_readout()
        self.brush_erase.set(False);self.update_save_state();self.draw()
        self.status.set(self.t(
            '当前涂抹：工作区域（橙色）。可以随时切换到禁止区域，已有标记会保留。' if method=='work' else
            '当前涂抹：禁止区域（红色）。可以随时切换到工作区域，已有标记会保留。',
            'Painting work area (orange). You may switch roles without losing marks.' if method=='work' else
            'Painting forbidden area (red). You may switch roles without losing marks.'))
    def clear_readout(self):
        self.focus=None;self.readout.set('X —  Y —  Z —');self.items.selection_clear(0,'end')
    def clear_selection(self):
        if self.busy or not self.confirm_pending_edits():return
        self.clear_readout();self.update_save_state();self.draw()
        self.status.set(self.t('已清除当前选中项；已有标记保留。','Current selection cleared; annotations retained.'))
    def click(self,e):
        if self.model is None or self.busy:return
        if self.show_axes.get() and 12<=e.x<=268 and 12<=e.y<=142:return
        if not self.confirm_pending_edits():return
        self.clear_readout();x,y=self.image_pixel(e.x,e.y)
        if not (0<=x<self.base.width and 0<=y<self.base.height):
            self.status.set(self.t('点击位于参考图像外，未新增标记。','Click is outside the reference image; no annotation added.'))
            self.update_save_state();self.draw();return
        if self.mode.get()=='region':
            self.brush_start(e);return
        try:
            item=self.model.pick(x,y,self.name.get().strip(),self.note.get().strip())
            if item is None:
                messages={
                    'depth_conflict':self.t('此处网格与参考深度不一致，未新增标记。','Mesh and reference depth disagree here; no annotation added.'),
                    'no_source_pixel':self.t('此处去畸变后没有有效原始图像像素，未新增标记。','This undistorted pixel has no valid source image pixel; no annotation added.'),
                    'outside_depth_range':self.t('此处表面超出有效深度范围，未新增标记。','Surface is outside the valid depth range; no annotation added.'),
                    'outside_image':self.t('点击位于参考图像外，未新增标记。','Click is outside the reference image; no annotation added.'),
                }
                self.status.set(messages.get(self.model.projection.pixel_reason(x,y),self.t('此处无有效三维表面，未新增标记。','No supported 3D surface here; no annotation added.')))
            else:self.refresh_items(('point',item['annotation_id']));self.status.set(self.t('已映射到三维表面。','Mapped to 3D surface.'))
        except Exception as exc:self.status.set(self.error_text(str(exc)))
        self.update_save_state();self.draw()
    def reset_brush(self):
        self.cancel_brush_frame()
        self._brush_history=[];self._brush_mask=None;self._brush_last=None
        self._selection_polygons={'work':[],'forbidden':[]}
        self._brush_mask_cache={}

    def cancel_brush_frame(self):
        frame=getattr(self,'_brush_frame',None)
        if frame is not None:
            self.root.after_cancel(frame);self._brush_frame=None

    def cached_brush_mask(self,role,polygon):
        from square_brush import polygon_mask
        if not hasattr(self,'_brush_mask_cache'):self._brush_mask_cache={}
        key=(self.base.size,tuple(map(tuple,polygon)))
        cached=self._brush_mask_cache.get(role)
        if cached is None or cached[0]!=key:
            cached=(key,polygon_mask(polygon,self.base.size));self._brush_mask_cache[role]=cached
        return cached[1]

    def selection_polygons(self):
        result={key:[p[:] for p in value] for key,value in self._selection_polygons.items()}
        role=self.selection_method.get() if hasattr(self,'selection_method') else 'work'
        result[role]=[p[:] for p in self.polygon]
        return result

    def has_selection_draft(self):
        return any(self.selection_polygons().values())

    def brush_resize(self,value):
        self.brush_size_label.configure(text=str(round(float(value))))
        self.draw_brush_cursor()

    def brush_hover(self,e):
        self._brush_cursor=(e.x,e.y);self.draw_brush_cursor()

    def draw_brush_cursor(self):
        self.canvas.delete('brush_cursor')
        cursor=getattr(self,'_brush_cursor',None)
        if not cursor or self.busy or self.mode.get()!='region':return
        x,y=cursor;u,v=self.image_pixel(x,y)
        if self.model is None or not (0<=u<self.base.width and 0<=v<self.base.height):return
        half=self.brush_size.get()/2
        forbidden=self.selection_method.get()=='forbidden' or self.exclusion_target is not None
        color='#7ce6c3' if self.brush_erase.get() else ('#ff7373' if forbidden else '#ffcf58')
        # A quiet square edge shows the exact footprint; four coloured corners
        # remain legible on photography without a heavy double frame.
        self.canvas.create_rectangle(x-half,y-half,x+half,y+half,
            outline='#dce6ec',width=1,tags=('brush_cursor','brush_bounds'))
        arm=min(10.,half*.45)
        for sx,sy in ((-1,-1),(1,-1),(1,1),(-1,1)):
            cx,cy=x+sx*half,y+sy*half
            points=(cx-sx*arm,cy,cx,cy,cx,cy-sy*arm)
            self.canvas.create_line(*points,fill='#243340',width=3,
                capstyle='round',joinstyle='round',tags='brush_cursor')
            self.canvas.create_line(*points,fill=color,width=2,
                capstyle='round',joinstyle='round',tags='brush_cursor')
        if half>=8:
            self.canvas.create_oval(x-1.5,y-1.5,x+1.5,y+1.5,fill=color,
                outline='#243340',width=1,tags='brush_cursor')

    def draw_brush_overlay(self,affine,w,h,only_role=None):
        active=self.selection_method.get();polygons=self.selection_polygons()
        if only_role is None:self._brush_photos={}
        for role,color in [('work','#ffc14d'),('forbidden','#ff5353')]:
            if only_role is not None and only_role!=role:continue
            tag='brush_layer_'+role
            self.canvas.delete(tag)
            mask=self._brush_mask if role==active and self._brush_mask is not None else None
            if mask is None:
                polygon=polygons[role]
                if not polygon:continue
                mask=self.cached_brush_mask(role,polygon)
            if not mask.any():continue
            alpha=Image.fromarray(mask*90).transform((w,h),Image.Transform.AFFINE,affine,Image.Resampling.NEAREST)
            layer=Image.new('RGBA',(w,h),color);layer.putalpha(alpha)
            photo=ImageTk.PhotoImage(layer);self._brush_photos[role]=photo
            self.canvas.create_image(0,0,image=photo,anchor='nw',tags=('brush_draft',role,tag))
        self.canvas.tag_raise('brush_layer_forbidden')

    def paint_brush_frame(self):
        self._brush_frame=None
        if self.closed or self._brush_last is None:return
        w,h=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        affine=(1/self.scale,0,-self.offset[0]/self.scale,0,1/self.scale,-self.offset[1]/self.scale)
        self.draw_brush_overlay(affine,w,h,only_role=self.selection_method.get())
        self.draw_brush_cursor()

    def brush_start(self,e):
        self.brush_end()
        self._brush_selection_before=self.selection_polygons()
        self._brush_before=[p[:] for p in self.polygon]
        self._brush_mask=self.cached_brush_mask(self.selection_method.get(),self.polygon).copy()
        self._brush_last=self.image_pixel(e.x,e.y)
        self._brush_stroke_size=self.brush_size.get()/self.scale
        self._brush_stroke_erase=self.brush_erase.get()
        self.draw()
        self.brush_move(e)

    def brush_move(self,e):
        self._brush_cursor=(e.x,e.y)
        if self.busy or self._brush_last is None:return
        from square_brush import stroke
        pixel=self.image_pixel(e.x,e.y)
        # Clip extreme off-canvas coordinates before passing them to OpenCV.
        pixel=np.clip(pixel,[-self.base.width,-self.base.height],[2*self.base.width,2*self.base.height])
        stroke(self._brush_mask,self._brush_last,pixel,self._brush_stroke_size,self._brush_stroke_erase)
        self._brush_last=pixel
        # Keep every stroke segment, but coalesce paint events to at most 30 fps.
        # The RGB image, cloud and unchanged selection layer remain on canvas.
        self.draw_brush_cursor()
        if getattr(self,'_brush_frame',None) is None:
            self._brush_frame=self.root.after(33,self.paint_brush_frame)

    def brush_end(self,e=None):
        if getattr(self,'_brush_last',None) is None:return
        if e is not None:self.brush_move(e)
        self.cancel_brush_frame()
        from square_brush import mask_polygon
        self._brush_last=None
        try:
            polygon=mask_polygon(self._brush_mask,MAX_POLYGON_CORNERS)
            if polygon!=self._brush_before:
                self._brush_history.append(self._brush_selection_before)
                self._brush_history=self._brush_history[-100:]
            self.polygon=polygon
            self._selection_polygons[self.selection_method.get()]=[p[:] for p in polygon]
            self.status.set(self.t('继续涂抹，或切换工作/禁止区域；系统会在完成时自动组合。',
                'Continue painting or switch between work/forbidden areas; the final result is combined automatically.'))
        except ValueError as error:
            self.polygon=self._brush_before;self.status.set(str(error))
        self._brush_mask=None;self.update_save_state();self.draw()

    def finish_polygon(self):
        self.brush_end()
        if self.model is None or self.busy:return
        selections=self.selection_polygons()
        if not any(selections.values()):
            self.status.set(self.t('请先涂抹工作区域或禁止区域。','Paint a work area or forbidden area first.'));return
        if not self.confirm_pending_edits():return
        if getattr(self,'exclusion_target',None) is not None:
            if len(self.polygon)<3:
                self.status.set('请先涂抹禁止区域。');return
            target=self.exclusion_target
            index=next((i for i,r in enumerate(self.model.store.regions) if r['annotation_id']==target),None)
            if index is None:self.status.set('原圈选区域已不存在，请重新选择。');return
            region=self.model.store.regions[index]
            try:self.model.store.set_exclusions(index,region.get('exclusion_polygons',[])+[[p[:] for p in self.polygon]])
            except ValueError as exc:self.status.set(str(exc));return
            self.polygon=[];self.reset_brush();self.exclusion_target=None
            self.refresh_items(('region',target));self.draw()
            self.status.set('禁止区域已添加（红色）；旧路径失效，请重新生成区域覆盖规划。');return
        model=self.model
        from square_brush import polygon_mask,mask_polygon
        work_polygon=selections['work'];forbidden_polygon=selections['forbidden']
        work_mask=polygon_mask(work_polygon,self.base.size) if work_polygon else None
        forbidden_mask=polygon_mask(forbidden_polygon,self.base.size) if forbidden_polygon else None
        if work_mask is not None:
            final_mask=work_mask.copy();method='work_minus_forbidden' if forbidden_mask is not None else 'work_only'
        else:
            final_mask=np.ones((self.base.height,self.base.width),np.uint8);method='full_minus_forbidden'
        if forbidden_mask is not None:final_mask[forbidden_mask.astype(bool)]=0
        if not final_mask.any():
            self.status.set(self.t('禁止区域覆盖了全部可选范围，没有可保存内容。',
                'The forbidden area removes the entire selectable range; nothing remains to save.'));return
        try:polygon=mask_polygon(final_mask,MAX_POLYGON_CORNERS)
        except ValueError as error:self.status.set(str(error));return
        name,note=self.name.get().strip(),self.note.get().strip()
        self.status.set(self.t('正在提取可见三维区域…','Extracting visible 3D region…'))
        def done(item):
            if item is None:self.status.set(self.t(
                '自动组合后的范围内没有有效三维内容。','No valid 3D content exists in the combined selection.'))
            else:
                item['selection_method']=method
                item['work_screen_polygon']=work_polygon
                item['forbidden_screen_polygon']=forbidden_polygon
                self.polygon=[];self.reset_brush();self.refresh_items(('region',item['annotation_id']))
                messages={
                    'work_only':'已提交单次结果：仅工作区域。粉色部分为将要保存和导出的内容。',
                    'full_minus_forbidden':'已提交单次结果：仅禁止区域。粉色部分为排除红色范围后的其余内容。',
                    'work_minus_forbidden':'已提交单次结果：工作区域减去禁止区域。工作区外的禁止标记已忽略。'}
                self.status.set(self.t(messages[method],{
                    'work_only':'Single result submitted: work area only. Pink shows the result to save and export.',
                    'full_minus_forbidden':'Single result submitted: forbidden area only. Pink shows all remaining content.',
                    'work_minus_forbidden':'Single result submitted: work minus forbidden areas. Forbidden marks outside work are ignored.'}[method]))
            self.draw()
        self.operation='region'
        self.work(lambda:model.region(polygon,name,note,cancel=self.operation_cancel(),
            replace_existing=PRODUCT_SINGLE_REGION),done)

    def begin_exclusion(self):
        if self.busy or self.model is None:return
        if not self.focus or self.focus[0]!='region' or len(self.selected_annotations())!=1:
            self.status.set('请先在列表中单选一个已完成的主圈选区域。');return
        if not self.confirm_pending_edits():return
        self.exclusion_target=self.model.store.regions[self.focus[1]]['annotation_id']
        self.mode.set('region');self.active_mode='region';self.polygon=[];self.reset_brush();self.brush_erase.set(False)
        self.status.set('按住左键涂抹禁区，随后点击“完成涂抹”；可勾选擦除修正，禁区只作用于选定主区域。')
        self.draw()

    def clear_exclusions(self):
        if self.busy or self.model is None:return
        if not self.focus or self.focus[0]!='region' or len(self.selected_annotations())!=1:return
        i=self.focus[1];item=self.model.store.set_exclusions(i,[])
        self.exclusion_target=None
        self.refresh_items(('region',item['annotation_id']));self.draw()
        self.status.set('已清除该区域全部禁区，可撤销；请重新生成路径。')
    def refresh_items(self,focus=None):
        self.items.delete(0,'end');self.index=[]
        for kind,items in [('point',self.model.store.points),('region',self.model.store.regions)]:
            for i,item in enumerate(items):
                self.items.insert('end',self.t('点' if kind=='point' else '区域',kind)+f' {i+1} · '+item.get('name',''))
                self.index.append((kind,i))
        self.clear_readout()
        self._annotations_dirty=self.model.store.dirty
        if focus:
            for n,(kind,i) in enumerate(self.index):
                item=(self.model.store.points if kind=='point' else self.model.store.regions)[i]
                if (kind,item['annotation_id'])==focus:self.items.selection_set(n);self.select();break
        self.update_save_state()
    def selected_annotations(self):
        return [self.index[n] for n in self.items.curselection() if n<len(getattr(self,'index',[]))]

    def record_distance_test(self):
        if self.model is None or self.busy:return
        selected=self.selected_annotations()
        if len(selected)!=2 or any(kind!='point' for kind,_ in selected):
            self.status.set(self.t('请按 Ctrl 在标记列表中选中两个表面点，再记录尺量两点之间的直线距离。',
                'Ctrl-select two surface points; enter their independently measured straight-line distance.'));return
        points=[self.model.store.points[i] for _,i in selected]
        window=tk.Toplevel(self.root);window.title(self.t('记录两点实测距离','Reference distance experiment'))
        window.transient(self.root);window.grab_set()
        ttk.Label(window,text=self.t('只验证所选两点间直线距离，不验证相机外壳距离或绝对 XYZ。\n误差限仅用于本次实验，不改变融合门限。',
            'Two-point straight-line distance only, not housing distance or absolute XYZ.\nThe entered limit does not change fusion gates.'),wraplength=490).grid(row=0,column=0,columnspan=2,padx=12,pady=10)
        fields={}
        specs=[('test_id','测试编号（同目标重复使用）','Test ID (same target)', ''),
            ('reference_mm','尺量参考距离 / mm','Reference distance / mm',''),
            ('reference_uncertainty_mm','参考测量误差界限 / mm','Reference uncertainty bound / mm',''),
            ('tolerance_mm','本次实验允许偏差 / mm','Experiment tolerance / mm',''),
            ('instrument','量具与读数方法','Instrument and method',''),
            ('conditions','目标 / 距离 / 角度 / 光照','Target / distance / angle / lighting','')]
        for row,(key,zh,en,default) in enumerate(specs,1):
            ttk.Label(window,text=self.t(zh,en)).grid(row=row,column=0,sticky='w',padx=12,pady=5)
            fields[key]=tk.StringVar(value=default)
            ttk.Entry(window,textvariable=fields[key],width=36).grid(row=row,column=1,padx=12,pady=5)
        feedback=tk.StringVar()
        ttk.Label(window,textvariable=feedback,wraplength=490).grid(row=7,column=0,columnspan=2,padx=12)
        def commit():
            from measurement_audit import distance_record,save_record
            try:
                values={key:value.get() for key,value in fields.items()}
                for key in ('reference_mm','reference_uncertainty_mm','tolerance_mm'):values[key]=float(values[key])
                record=distance_record(self.model,points,**values)
                path=save_record(self.model,record)
            except (ValueError,OSError) as exc:
                feedback.set(self.t('未保存，请检查输入：','Not saved; check input: ')+str(exc));return
            self.status.set(self.t('距离测试已记录：','Distance test saved: ')+
                f"{record['measured_mm']:.2f} mm; "+self.t('偏差 ','error ')+f"{record['signed_error_mm']:+.2f} mm\n{path}")
            window.destroy()
        ttk.Button(window,text=self.t('保存实验记录','Save experiment'),command=commit).grid(row=8,column=0,padx=12,pady=12)
        ttk.Button(window,text=self.t('取消','Cancel'),command=window.destroy).grid(row=8,column=1,pady=12)

    def open_measurement_records(self):
        if self.model is None or self.busy:return
        import os
        folder=self.model.reconstruction/'measurement_validation'
        if not folder.exists():
            self.status.set(self.t('尚无精度测试记录，请先记录两点实测距离或生成报告。','No records yet; record a distance or generate a report.'));return
        try:os.startfile(str(folder))
        except OSError as exc:self.status.set(str(exc))

    def local_geometry_diagnostics(self):
        if self.model is None or self.busy:return
        points=[self.model.store.points[i] for kind,i in self.selected_annotations() if kind=='point']
        if len(points)!=1:
            self.status.set('请单选一个已有三维点标记，再进行同位置几何诊断。');return
        from local_geometry_diagnostics import diagnose,show_diagnostics
        model=self.model;point=points[0];self.operation='geometry_diagnostics'
        def done(report):
            if self.model is not model:return
            show_diagnostics(self.root,model,report)
            self.status.set('局部诊断完成：'+report['support_kind']+'；支持帧数不是精度，未修改模型或融合参数。')
        self.work(lambda:diagnose(model,point,check=GeometryBudget(self.operation_cancel()).check,progress=self.operation_progress),done)

    def measurement_report(self):
        if self.model is None or self.busy:return
        from measurement_audit import save_report
        model=self.model;self.operation='measurement_audit'
        def done(folder):
            window=tk.Toplevel(self.root);window.title(self.t('数据质量与距离测试报告','Data quality and distance test report'))
            text=tk.Text(window,width=95,height=28,wrap='word');text.pack(fill='both',expand=True)
            text.insert('1.0',(folder/'报告.md').read_text(encoding='utf-8'));text.configure(state='disabled')
            import os
            ttk.Button(window,text=self.t('打开报告目录（JSON / CSV）','Open report folder (JSON / CSV)'),
                command=lambda:os.startfile(str(folder))).pack(pady=8)
            self.status.set(self.t('报告已保存：','Report saved: ')+str(folder))
        self.work(lambda:save_report(model),done)

    def select_all(self,event=None):
        if self.model is not None and not self.busy:
            self.items.selection_set(0,'end');self.select()
        return 'break'

    def select(self,event=None):
        if self.busy or self.model is None or not self.items.curselection():return
        selection=self.items.curselection()
        if len(selection)>1:
            previous=self.focus
            if not self.confirm_pending_edits():
                self.items.selection_clear(0,'end')
                if previous in self.index:self.items.selection_set(self.index.index(previous))
                return
            self.items.selection_clear(0,'end')
            for n in selection:self.items.selection_set(n)
            self.focus=None
            self._syncing_fields=True;self.name.set('');self.note.set('');self._syncing_fields=False
            self.readout.set(self.t(f'已选 {len(selection)} 项，可批量删除。\n编辑名称备注请单选。',f'{len(selection)} annotations selected. Batch delete available.\nSelect one to edit name/note.'))
            self.update_save_state();self.draw();return
        kind,i=self.index[self.items.curselection()[0]]
        if self.focus==(kind,i):return
        previous=self.focus
        if not self.confirm_pending_edits():
            self.items.selection_clear(0,'end')
            if previous in self.index:self.items.selection_set(self.index.index(previous))
            return
        self.focus=(kind,i)
        item=(self.model.store.points if kind=='point' else self.model.store.regions)[i]
        self._syncing_fields=True;self.name.set(item.get('name',''));self.note.set(item.get('note',''));self._syncing_fields=False
        if kind=='point':
            xyz,d=point_readout_mm(item['point_world_m'])
            text='\n'.join(f'{axis}: {value:.1f} mm' for axis,value in zip('XYZ',xyz))
            text+='\n'+self.t('距初始原点：','Distance to W0: ')+f'{d:.1f} mm'
            if len(self.model.store.points)>=2:
                d2=distance_m(*self.model.store.points[-2:])*1000
                text+='\n'+self.t('末两点距离：','Last pair: ')+f'{d2:.1f} mm'
            self.readout.set(text)
        else:
            text=self.t('区域三角面：','Region triangles: ')+str(len(item['triangle_indices']))
            from fixed_rgb_geometry import region_vertex_summary
            summary=region_vertex_summary(self.model.store.mesh,item['triangle_indices'])
            text+='\n'+self.t('区域顶点点云：','Region mesh vertices: ')+str(summary['vertex_count'])
            text+='\n'+self.t('顶点距原点范围：','Vertex distance to W0: ')+f"{summary['minimum_distance_mm']:.1f}–{summary['maximum_distance_mm']:.1f} mm"
            support=item.get('geometry_support')
            if support:
                text+='\n'+self.t('图像几何支持：','Image geometry support: ')+f"{support['geometry_supported_pixels']} / {support['requested_pixels']} px"
                ratio=support.get('support_ratio')
                text+='\n'+self.t('支持比例：','Supported ratio: ')+(f'{100*ratio:.1f}%' if ratio is not None else self.t('未记录','Not recorded'))
                text+='\n'+self.t('首帧缺深度处有支持：','Supported where reference depth is missing: ')+str(support['reference_depth_missing_supported_pixels'])+' px'
                text+='\n'+self.t('图像像素统计，不是曲面面积。','Image pixel counts, not surface area.')
            else:text+='\n'+self.t('图像几何支持：未记录（旧区域）','Image geometry support: not recorded (legacy region)')
            self.readout.set(text+'\nX —  Y —  Z —')
        self.update_save_state();self.draw()
    def edit(self):
        if self.focus is None or self.busy:return
        kind,i=self.focus
        item=self.model.store.update_annotation(kind,i,name=self.name.get().strip(),note=self.note.get().strip())
        self.refresh_items((kind,item['annotation_id']))
        self.status.set(self.t('名称与备注已应用；请保存标记。','Name and note applied; save the annotations.'))
    def delete(self):
        if self.model is None or self.busy:return
        selection=self.selected_annotations() or ([self.focus] if self.focus else [])
        if not selection or not self.confirm_pending_edits():return
        self.model.store.delete_annotations(selection);self.refresh_items();self.draw()
        self.status.set(self.t(f'已删除 {len(selection)} 项，一次撤销可恢复整批；请保存更改。',f'Deleted {len(selection)} annotations. Undo restores the batch; save changes.'))
    def undo(self):
        if self.model is None or self.busy or not self.confirm_pending_edits():return
        if getattr(self,'_brush_history',[]):
            self._selection_polygons=self._brush_history.pop()
            self.polygon=[p[:] for p in self._selection_polygons[self.selection_method.get()]]
            self._brush_mask=None;self._brush_last=None
        elif self.has_selection_draft():
            self.polygon=[];self._selection_polygons[self.selection_method.get()]=[]
        else:
            model=self.model;focus=model.store.undo()
            if any(model.display_for('region',item) is None for item in model.store.regions):
                self.clear_readout();self.operation='restore_display'
                self.status.set(self.t('正在恢复可见区域高亮…','Restoring visible region highlighting…'))
                def done(_):self.refresh_items(focus);self.draw()
                self.work(lambda:model.prepare_display(cancel=self.operation_cancel()),done)
            else:self.refresh_items(focus)
        self.update_save_state();self.draw()
    def current_region_path(self):
        return self._plans.get(self.planning_region_id()) if self.planning_state()['can_use'] else None

    def clear_path_preview(self):
        key=self.planning_region_id()
        self._plans.pop(key,None);self._requests.pop(key,None);self._plan_errors.pop(key,None)
        self.update_save_state();self.draw()

    def export_history(self):
        if self.model is None or self.busy:return
        from export_history import show_history
        show_history(self.root,self.model.reconstruction)

    def save_region_project(self,save_as=False):
        if self.model is None or self.busy:return
        if not self.focus or self.focus[0]!='region' or len(self.selected_annotations())!=1:
            self.status.set('请先单选需要保存的主圈选区域。');return
        if not self.confirm_pending_edits():return
        if self.has_selection_draft():self.status.set('请先完成或撤销涂抹。');return
        selected_id=self.model.store.regions[self.focus[1]]['annotation_id']
        path=self.project_path if not save_as and selected_id==self._project_region_id else None
        if path is None:path=filedialog.asksaveasfilename(parent=self.root,title='保存区域方案（可继续编辑）',
            defaultextension='.region.zip',initialfile='区域方案.region.zip',filetypes=[('Region project','*.zip')])
        if not path:return
        from region_project import save_project
        import copy
        model=self.model;region=copy.deepcopy(model.store.regions[self.focus[1]]);plan=copy.deepcopy(self.current_region_path())
        request=copy.deepcopy(self._requests.get(region['annotation_id']))
        expected=getattr(model,'project_file_digest',None) if Path(path).resolve()==getattr(model,'project_source_path',None) else None
        self.operation='save'
        self.work(lambda:save_project(model,region,path,plan,GeometryBudget(self.operation_cancel()).check,
            planning_parameters=request,progress=self.operation_progress,commit=self.operation_commit,expected_digest=expected),
            lambda saved:self.project_saved(saved,region,plan))

    def project_saved(self,path,region,plan):
        from region_path import signature
        self._project_region_id=region['annotation_id']
        previous_path=self.project_path
        self.project_path=Path(path).resolve()
        windows=getattr(self.root._root(),'_planning_windows',{})
        if hasattr(self.model,'project_id'):
            if previous_path and windows.get((str(previous_path).casefold(),None)) is self.root:
                windows.pop((str(previous_path).casefold(),None),None)
            windows[(str(self.project_path).casefold(),None)]=self.root
        # A portable project contains one region, not all annotations.
        if hasattr(self.model,'project_id') and len(self.model.store.regions)==1 and not self.model.store.points and self.model.store.regions[0]==region:
            self.model.store.mark_saved();self._annotations_dirty=False
        key=region['annotation_id']
        import copy
        if key in self._requests:self._saved_requests[key]=copy.deepcopy(self._requests[key])
        else:self._saved_requests.pop(key,None)
        if plan:self._saved_plans[key]=signature(plan)
        else:
            self._saved_plans.pop(key,None)
            self._plans.pop(key,None)
        self.update_save_state()
        from region_project import digest
        self.model.project_source_path=Path(path).resolve();self.model.project_file_digest=digest(path)
        if hasattr(self.model,'project_id') and previous_path!=self.project_path:
            from planning_drafts import Journal
            if not (self.model.store.dirty or self.plan_unsaved() or self.pending_edits() or self.has_selection_draft()):
                try:self.draft_journal.discard()
                except OSError as error:self.draft_feedback.set('旧入口草稿清理失败：'+str(error))
            self.draft_journal=Journal(self.project_path);self._draft_signature=None;self._draft_due=0
        self.status.set(getattr(self.model,'last_save_warning',None) or '方案已保存（参考图、几何、禁区和有效路线）：'+str(path))

    def open_region_project(self):
        if self.busy:return
        path=filedialog.askopenfilename(parent=self.root,title='打开已有方案',filetypes=[('Region project','*.zip')])
        if path:
            from planning_workspace import open_workspace
            open_workspace(self.root,path,language=self.language)

    def parameter_default(self,kind,key,default):
        request=self._requests.get(self.planning_region_id(),{})
        return request.get('values',{}).get(key,default) if request.get('kind')==kind else default

    def plan_region_path(self):
        if self.busy or self.model is None:return
        if not self.focus or self.focus[0]!='region' or len(self.selected_annotations())!=1:
            self.status.set(self.t('请单选一个已完成的圈选区域。','Select one completed region.'));return
        spacing=simpledialog.askinteger(self.t('扫描行间距','Scan row spacing'),
            self.t('行间距（图像像素，非表面毫米），默认10：','Row spacing in image pixels (not surface mm), default 10:'),
            initialvalue=self.parameter_default('raster','row_step_px',10),minvalue=1,maxvalue=500,parent=self.root)
        if spacing is None:return
        max_step=simpledialog.askfloat(self.t('路径断段阈值','Path split threshold'),
            self.t('相邻采样点超过多少毫米时断段（不是路径点间距），默认5：',
                   'Split when adjacent samples exceed this distance in mm (not sample spacing), default 5:'),
            initialvalue=self.parameter_default('raster','max_step_mm',5.),minvalue=.01,maxvalue=100.,parent=self.root)
        if max_step is None:return
        from region_path import raster_path
        model=self.model;region=model.store.regions[self.focus[1]]
        def done(plan):
            if self.model is not model:return
            self.show_region_path(model,plan)
        self._requests[region['annotation_id']]=dict(kind='raster',values=dict(row_step_px=spacing,max_step_mm=max_step))
        self.operation='path_preview'
        self.work(lambda:raster_path(model.projection,model.store.mesh,region,spacing,max_step,
            check=GeometryBudget(self.operation_cancel()).check),done)

    def show_region_path(self,model,plan):
        if self.model is not model:return
        from region_path import path_summary
        summary=path_summary(plan)
        from planning_state import validate,parameters
        region=next((r for r in model.store.regions if r['annotation_id']==plan['region_id']),None)
        if region is None:return
        validate(plan,model.store.mesh,region)
        self._plans[plan['region_id']]=plan;self._requests[plan['region_id']]=parameters(plan)
        self._plan_errors.pop(plan['region_id'],None)
        self.update_save_state();self.draw()
        count=summary['segment_count'];points=summary['sample_count']
        length=summary['total_segment_length_m']*1000
        self.status.set(self.t(f'路径：{count}段 / {points}点 / 段内折线总长{length:.1f} mm。连续链内含换行连接；断链不跨越，仅预览。',
            f'Path: {count} segments / {points} samples / {length:.1f} mm chord length. No transitions; preview only.'))
        if plan['schema']=='femto-region-metric-path-v1':
            self.status.set(self.t(f"网格距离近似间距 {plan['spacing_mm']:g} mm；", 
                f"Approximate mesh-distance spacing {plan['spacing_mm']:g} mm; ")+self.status.get())

    def plan_metric_path(self,after=None):
        if self.busy or self.model is None:return
        if not self.focus or self.focus[0]!='region' or len(self.selected_annotations())!=1:
            self.status.set(self.t('请单选一个已完成的圈选区域。','Select one completed region.'));return
        spacing=simpledialog.askfloat(self.t('曲面路径间距','Surface path spacing'),
            self.t('路径间距（mm，网格最短距离近似），默认5：','Contour spacing (mm, approximate mesh-edge distance), default 5:'),
            initialvalue=self.parameter_default('metric','spacing_mm',5.),minvalue=.1,maxvalue=100.,parent=self.root)
        if spacing is None:return
        step=simpledialog.askfloat(self.t('最大采样间距','Maximum sample spacing'),
            self.t('沿路径最大采样间距（mm），默认1：','Maximum distance between samples along path (mm), default 1:'),
            initialvalue=self.parameter_default('metric','sample_step_mm',1.),minvalue=.1,maxvalue=100.,parent=self.root)
        if step is None:return
        from metric_region_path import metric_path
        model=self.model;region=model.store.regions[self.focus[1]];self.operation='path_preview'
        def done(plan):
            self.show_region_path(model,plan)
            if after is not None and self.current_region_path() is not None:after()
        self._requests[region['annotation_id']]=dict(kind='metric',values=dict(spacing_mm=spacing,sample_step_mm=step))
        self.work(lambda:metric_path(model.projection,model.store.mesh,region,spacing,step,
            check=GeometryBudget(self.operation_cancel()).check),done)

    def plan_surface_spacing(self):
        if self.busy or self.model is None:return
        if not self.focus or self.focus[0]!='region' or len(self.selected_annotations())!=1:
            self.status.set('请单选一个已完成区域。');return
        values={}
        for key,label,default,minimum,maximum in [('spacing_mm','目标网格表面层间距 / mm（非精确测地等距）',5.,.1,100.),
                ('sample_step_mm','沿扫描线最大采样步长 / mm',1.,.1,100.),('width_mm','模拟扫描宽度 / mm',4.,.2,100.),
                ('boundary_margin_mm','额外边界预留 / mm',1.,0.,50.),('tolerance_mm','间距报告容差 / mm（超差会明确显示）',1.,.01,100.)]:
            value=simpledialog.askfloat('三维表面间距模式',label,initialvalue=self.parameter_default('surface_spacing',key,default),minvalue=minimum,maxvalue=maximum,parent=self.root)
            if value is None:return
            values[key]=value
        from surface_spacing import surface_spacing_plan
        model=self.model;region=model.store.regions[self.focus[1]];self.operation='path_preview'
        self._requests[region['annotation_id']]=dict(kind='surface_spacing',values=dict(pattern='surface_spacing_v1',**values))
        def done(plan):
            self.show_region_path(model,plan);self.present_region_path()
            self.status.set('三维表面模式已生成；'+('采样间距上下界满足容差。' if plan['spacing_report']['status']=='within_sampled_bounds' else '间距未完全满足或无法证实，请查看表面间距报告；未切换RGB模式。'))
        self.work(lambda:surface_spacing_plan(model.projection,model.store.mesh,region,check=GeometryBudget(self.operation_cancel()).check,progress=self.operation_progress,**values),done)

    def route_preferences(self):
        window=tk.Toplevel(self.root);window.title('往返方向与起点偏好');window.transient(self.root);window.grab_set()
        directions={'水平往返':'horizontal','垂直往返':'vertical'}
        corners={'左上':'top_left','右上':'top_right','左下':'bottom_left','右下':'bottom_right'}
        saved=self._requests.get(self.planning_region_id(),{}).get('values',{})
        direction=tk.StringVar(value=next((k for k,v in directions.items() if v==saved.get('direction')),'水平往返'))
        corner=tk.StringVar(value=next((k for k,v in corners.items() if v==saved.get('start_corner')),'左上'))
        for title,var,choices in [('往返方向',direction,directions),('起点偏好',corner,corners)]:
            ttk.Label(window,text=title).pack(padx=15,pady=5)
            ttk.Combobox(window,textvariable=var,values=list(choices),state='readonly').pack(padx=15)
        ttk.Label(window,text='以固定RGB画面为准；实际起点受表面、禁区和边界余量限制。',wraplength=320).pack(padx=15,pady=10)
        result=[]
        def accept():result.append((directions[direction.get()],corners[corner.get()]));window.destroy()
        ttk.Button(window,text='继续设置覆盖参数',command=accept).pack(pady=5)
        ttk.Button(window,text='取消',command=window.destroy).pack(pady=5)
        self.root.wait_window(window)
        return result[0] if result else None

    def plan_coverage_path(self,after=None):
        if self.busy or self.model is None:return
        if not self.focus or self.focus[0]!='region' or len(self.selected_annotations())!=1:
            self.status.set(self.t('请单选一个已完成的圈选区域。','Select one completed region.'));return
        choice=self.route_preferences()
        if choice is None:return
        direction,start_corner=choice
        width=simpledialog.askfloat('模拟扫描宽度','扫描带宽度（mm，模拟参数）：',
            initialvalue=self.parameter_default('serpentine_v4','width_mm',self.parameter_default('serpentine_v3','width_mm',self.parameter_default('serpentine_v2','width_mm',self.parameter_default('serpentine','width_mm',10.)))),minvalue=.2,maxvalue=100.,parent=self.root)
        if width is None:return
        overlap=simpledialog.askfloat('相邻路线重叠率','重叠率（%，用于推算路线间距）：',
            initialvalue=self.parameter_default('serpentine_v4','overlap_percent',self.parameter_default('serpentine_v3','overlap_percent',self.parameter_default('serpentine_v2','overlap_percent',self.parameter_default('serpentine','overlap_percent',20.)))),minvalue=0.,maxvalue=90.,parent=self.root)
        if overlap is None:return
        margin=simpledialog.askfloat('边界预留距离','除模拟扫描半径外，再向圈内预留多少毫米（研发参数）：',
            initialvalue=self.parameter_default('serpentine_v4','boundary_margin_mm',self.parameter_default('serpentine_v3','boundary_margin_mm',self.parameter_default('serpentine_v2','boundary_margin_mm',self.parameter_default('serpentine','boundary_margin_mm',1.)))),minvalue=0.,maxvalue=50.,parent=self.root)
        if margin is None:return
        from route_optimization import optimized_serpentine_plan as coverage_plan
        model=self.model;region=model.store.regions[self.focus[1]];self.operation='path_preview'
        def done(plan):
            self.show_region_path(model,plan)
            if after is not None:after()
            else:self.present_region_path()
        self._requests[region['annotation_id']]=dict(kind='serpentine_v4',values=dict(width_mm=width,overlap_percent=overlap,boundary_margin_mm=margin,pattern='serpentine_v4',direction=direction,start_corner=start_corner))
        self.work(lambda:coverage_plan(model.projection,model.store.mesh,region,width,overlap,
            check=GeometryBudget(self.operation_cancel()).check,boundary_margin_mm=margin,direction=direction,start_corner=start_corner),done)

    def load_region_path(self):
        if self.busy or self.model is None:return
        if not self.focus or self.focus[0]!='region' or len(self.selected_annotations())!=1:
            self.status.set(self.t('请单选要回读路径的圈选区域。','Select one region to load its path.'));return
        path=filedialog.askopenfilename(parent=self.root,title=self.t('回读路径','Load path'),
            initialdir=str(self.model.reconstruction/'region_paths'),filetypes=[('Path JSON','*.json')])
        if not path:return
        from region_path import load_path
        model=self.model;region=model.store.regions[self.focus[1]]
        def accepted(plan):
            self.show_region_path(model,plan)
            import json
            if 'plan_binding' not in json.loads(Path(path).read_text(encoding='utf-8')):
                self.status.set('旧路径没有算法版本关联；已完整重算核验，请保存新版方案。')
        self.operation='path_preview'
        self.work(lambda:load_path(path,model.store,model.projection,region,
            check=GeometryBudget(self.operation_cancel()).check),accepted)

    def export_region_path(self):
        if self.busy:return
        plan=self.current_region_path()
        if plan is None:
            self.status.set(self.planning_state()['reason']);return
        from region_path import export_path
        model=self.model
        region=model.store.regions[self.focus[1]]
        self.operation='export'
        self.work(lambda:export_path(plan,model.store,model.reconstruction/'region_paths',region,
            presentation=(model.rgb,model.projection.ids),check=GeometryBudget(self.operation_cancel()).check,
            progress=self.operation_progress,commit=self.operation_commit),
            lambda folder:self.status.set(self.t('路径已导出：','Path exported: ')+str(folder)))

    def present_region_path(self):
        if self.busy:return
        plan=self.current_region_path()
        if plan is None:
            self.status.set(self.planning_state()['reason']);return
        from path_presentation import show_presentation
        import copy
        show_presentation(self.root,self.model,copy.deepcopy(self.model.store.regions[self.focus[1]]),copy.deepcopy(plan),
            is_current=lambda:self.current_region_path() is plan)

    def save(self):
        if self.model is None or self.busy:return
        if self.has_selection_draft():self.status.set(self.t('请先完成或撤销涂抹。','Finish or undo the polygon first.'));return
        if not self.confirm_pending_edits():return
        model=self.model;store=model.store;folder=model.reconstruction/'annotations'
        focus=None
        if self.focus:
            kind,i=self.focus;item=(store.points if kind=='point' else store.regions)[i]
            focus=(kind,item['annotation_id'])
        def saved(path):
            if self.model is not model:return
            self.refresh_items(focus);self.draw()
            self.status.set(self.t('已保存：','Saved: ')+str(path))
        self.operation='save';self.status.set(self.t('正在保存标记与区域…','Saving annotations and regions…'))
        self.work(lambda:store.save(folder,check=GeometryBudget(self.operation_cancel()).check),saved)
    def load(self):
        if self.model is None or self.busy:return
        path=filedialog.askopenfilename(parent=self.root,initialdir=self.model.reconstruction,filetypes=[('Annotations','*.json')])
        if not path:return
        if not self.confirm_discard(self.t('回读标记','Loading annotations')):return
        model=self.model
        def done(_):
            if self.model is not model:return
            self.polygon=[];self.reset_brush()
            focus=('point',model.store.points[-1]['annotation_id']) if model.store.points else (
                ('region',model.store.regions[-1]['annotation_id']) if model.store.regions else None)
            self.refresh_items(focus);self.draw()
            self.status.set(self.t('已回读同模型标记。','Loaded annotations for this model.'))
        self.operation='load_annotations'
        self.status.set(self.t('正在核对模型来源并回读；失败时保留当前内容…','Validating model provenance and loading; current work is retained on failure…'))
        self.work(lambda:model.load(path,cancel=self.operation_cancel()),done)
    def advanced(self):
        subprocess.Popen([sys.executable,str(Path(__file__).with_name('scene_demo_viewer.py')),str(self.result),'--lang',self.language])
    def close(self,force=False):
        if self.closed:return
        if not force:
            if self.busy:
                self.update_save_state()
                if not messagebox.askokcancel(self.t('任务仍在运行','Task still running'),
                    self.save_state.get()+'\n\n'+self.t('关闭会取消未完成任务并放弃未保存内容；已开始的独立文件写入将安全退出。继续？',
                        'Closing cancels unfinished work and discards unsaved changes. Any current new-file write will exit safely. Continue?'),parent=self.root):return
            elif not self.confirm_discard(self.t('关闭窗口','Closing the window')):return
        if self.root is self.root._root():
            # Managed workspaces share the Tk process. Respect each one's
            # unsaved-work prompt before destroying their application root.
            windows=list(set(getattr(self.root,'_planning_windows',{}).values()))
            for window in windows:
                if not window.winfo_exists():continue
                viewer=getattr(window,'viewer',None)
                if viewer is not None:
                    viewer.close(force=force)
                    if window.winfo_exists():return
                else:window.destroy()
        demo=getattr(self.root,'_route_demo_window',None)
        if demo is not None and demo.winfo_exists():demo.close_demo()
        if getattr(self,'_draft_ready',False):
            try:self.draft_journal.discard()
            except OSError as error:
                if not force:self.status.set('草稿清理失败，未关闭：'+str(error));return
        with self._task_lock:
            self.cancel_brush_frame()
            self.closed=True;self.generation+=1;self.task_cancel.set()
        if hasattr(self,'timer'):self.root.after_cancel(self.timer)
        self.root.destroy()


if __name__=='__main__':
    from scene_scale_validation import resolve_reconstruction
    p=argparse.ArgumentParser();p.add_argument('result',nargs='?',type=Path);p.add_argument('--lang',choices=['zh','en'],default='zh')
    a=p.parse_args();result=a.result if a.result and a.result.is_file() else resolve_reconstruction(a.result)
    root=tk.Tk();FixedRGBViewer(root,result,a.lang);root.mainloop()
