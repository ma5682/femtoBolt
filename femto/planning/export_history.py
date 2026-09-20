"""Read-only export catalog and standalone preview entry points."""
from femto.paths import REPO_ROOT
from datetime import datetime
import json
from pathlib import Path


def register_project(reconstruction,path,check=lambda:None):
    """Index a portable project without unpacking or modifying it."""
    import zipfile
    import hashlib
    path=Path(path).resolve()
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo('project.json').file_size>100*1024**2:raise ValueError('Project metadata too large')
        data=json.loads(archive.read('project.json'))
    if data.get('schema')!='femto-region-project-v1':raise ValueError('请选择区域工程 ZIP，不是展示分享包。')
    folder=Path(reconstruction)/'region_project_history';folder.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(str(path).casefold().encode()).hexdigest()
    from femto.planning.region_library import digest as library_digest
    record=dict(project_file=str(path),name=data.get('region',{}).get('name') or '未命名区域',
        project_id=data['project_id'],revision_id=data['revision_id'],has_plan=bool(data.get('plan')),sha256=library_digest(path,check))
    temporary=folder/(key+'.tmp')
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    check();temporary.replace(folder/(key+'.json'))
    return path


def local_asset(folder, name):
    if not isinstance(name,str) or not name:return None
    path=(folder/name).resolve()
    if not path.is_relative_to(folder.resolve()):return None
    return path if path.is_file() else None


def scan_region_exports(reconstruction, check=lambda:None):
    """Read only concrete region exports; ignore projects, routes and library index."""
    root=Path(reconstruction);rows=[]
    for source in (root/'annotations').glob('*/annotations.json'):
        check()
        try:
            stamp=source.stat().st_mtime;data=json.loads(source.read_text(encoding='utf-8-sig'))
            for index,region in enumerate(data.get('regions',[])):
                asset=local_asset(source.parent,region.get('glb_file')) or local_asset(source.parent,region.get('mesh_file'))
                rows.append(dict(time=datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S'),stamp=stamp,
                    kind='区域',name=region.get('name') or region.get('annotation_id') or f'区域 {index+1}',
                    folder=source.parent,source=source,asset=asset,region_index=index,
                    status='可预览' if asset else '区域模型文件缺失'))
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as error:
            rows.append(dict(time='',stamp=0,kind='异常记录',name=source.parent.name,
                folder=source.parent,source=source,asset=None,status='读取失败：'+str(error)))
    return sorted(rows,key=lambda row:row['stamp'],reverse=True)


def scan_exports(reconstruction, include_hidden=False, check=lambda:None):
    root=Path(reconstruction);rows=[]
    sources=list((root/'annotations').glob('*/annotations.json'))+list((root/'region_paths').glob('*/path.json'))
    for source in sources:
        check()
        try:
            stamp=source.stat().st_mtime
            data=json.loads(source.read_text(encoding='utf-8-sig'))
            if source.name=='annotations.json':
                items=[('区域',r.get('name') or r.get('annotation_id',''),
                        r.get('glb_file'),r.get('mesh_file')) for r in data['regions']]
                if not items:items=[('标记','仅点标记',None,None)]
            else:
                items=[('区域＋路径',data.get('region',{}).get('name') or data.get('region_id',''),data.get('visualization',{}).get('file'),None)]
            for region_index,(kind,name,glb,ply) in enumerate(items):
                asset=local_asset(source.parent,glb) or local_asset(source.parent,ply)
                rows.append(dict(time=datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S'),
                    stamp=stamp,kind=kind,name=name,folder=source.parent,source=source,asset=asset,
                    diagram=local_asset(source.parent,data.get('route_diagram')),
                    archive=local_asset(source.parent,data.get('share_archive')),
                    region_index=region_index if kind=='区域' else None,
                    status='可预览' if asset else '无三维文件 / 文件缺失'))
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as error:
            rows.append(dict(time='',stamp=0,kind='异常记录',name=source.parent.name,
                folder=source.parent,source=source,asset=None,status='读取失败：'+str(error)))
    for source in (root/'region_project_history').glob('*.json'):
        check()
        try:
            data=json.loads(source.read_text(encoding='utf-8'));path=Path(data['project_file'])
            exists=path.is_file();stamp=path.stat().st_mtime if exists else source.stat().st_mtime
            rows.append(dict(time=datetime.fromtimestamp(stamp).strftime('%Y-%m-%d %H:%M:%S'),stamp=stamp,
                kind='区域工程 / 方案',name=data['name'],folder=path.parent,source=source,asset=None,
                project=path if exists else None,archive=path if exists else None,
                status='可打开继续规划（有保存路径）' if exists and data.get('has_plan') else
                       '可打开继续规划' if exists else '工程已移动或删除，请重新导入'))
        except (OSError,ValueError,KeyError,TypeError):
            rows.append(dict(time='',stamp=0,kind='异常工程记录',name=source.name,folder=source.parent,
                source=source,asset=None,status='工程索引读取失败'))
    from femto.planning.region_library import enrich,read_index
    known={str(r['source'].resolve()).casefold() for r in rows}
    for entry in read_index(reconstruction)['records'].values():
        snapshot=entry.get('snapshot')
        if snapshot and str(Path(snapshot['source']).resolve()).casefold() not in known:
            row=dict(snapshot)
            for k in ('source','folder','asset','diagram','archive'):
                if row.get(k):row[k]=Path(row[k])
            row['status']='文件丢失';rows.append(row)
    rows=enrich(rows,reconstruction,check)
    return sorted([r for r in rows if include_hidden or not r['hidden']],key=lambda row:row['stamp'],reverse=True)


def _build_region_library_table(window):
    import queue
    import tkinter as tk
    from tkinter import ttk
    from femto.planning.planning_io import TaskControl

    thumbnail=ttk.Label(window,text='无缩略图');thumbnail.pack(side='bottom')
    frame=ttk.Frame(window);frame.pack(fill='both',expand=True,padx=12)
    tree=ttk.Treeview(frame,columns=('time','kind','name','status'),show='tree headings',selectmode='extended')
    for key,title,width in [('time','导出时间',170),('kind','类型',100),('name','区域名称 / ID',250),('status','文件状态',280)]:
        tree.heading(key,text=title);tree.column(key,width=width)
    tree.heading('#0',text='区域 / 方案版本');tree.column('#0',width=190)
    scroll=ttk.Scrollbar(frame,command=tree.yview);tree.configure(yscrollcommand=scroll.set)
    scroll.pack(side='right',fill='y');tree.pack(fill='both',expand=True)
    detail=tk.StringVar(value='正在读取导出记录…')
    ttk.Label(window,textvariable=detail,wraplength=1000).pack(fill='x',padx=12,pady=10)
    actions=ttk.Frame(window);actions.pack(fill='x',padx=12,pady=10)
    project_actions=ttk.Frame(window);project_actions.pack(fill='x',padx=12,pady=4)
    return thumbnail,tree,detail,actions,project_actions,[],queue.Queue(),False,[''],TaskControl('读取区域方案库')


def _update_library_thumbnail(row,thumbnail,window):
    thumbnail.configure(image='',text='无缩略图');thumbnail.image=None
    if row and (row.get('diagram') or row.get('project')):
        try:
            from PIL import Image,ImageTk
            import io,zipfile
            image_source=row.get('diagram')
            if not image_source:
                with zipfile.ZipFile(row['project']) as z:
                    if z.getinfo('rgb.png').file_size>20*1024**2:raise ValueError('缩略图超过预算')
                    image_source=io.BytesIO(z.read('rgb.png'))
            with Image.open(image_source) as im:
                if im.width*im.height>20_000_000:raise ValueError('缩略图尺寸超限')
                im.thumbnail((280,140));picture=ImageTk.PhotoImage(im.copy(),master=window)
            thumbnail.configure(image=picture,text='');thumbnail.image=picture
        except Exception:thumbnail.configure(text='缩略图不可读取')


def _render_library_rows(tree,rows,show_hidden,search,kind_filter):
    tree.delete(*tree.get_children());groups=set();versions=set()
    for i,row in enumerate(rows):
        if row.get('hidden') and not show_hidden.get():continue
        if search.get().casefold() not in row['name'].casefold():continue
        if kind_filter.get()!='全部' and row['kind']!=kind_filter.get():continue
        group='g'+row['region_key'];version=group+'v'+row['version_key']
        if group not in groups:
            tree.insert('','end',iid=group,text=row['name'],open=True);groups.add(group)
        if version not in versions:
            tree.insert(group,'end',iid=version,text='方案 '+row['version_key'][:12],open=True);versions.add(version)
        tree.insert(version,'end',iid=str(i),text='已移除' if row.get('hidden') else '记录',values=[row[k] for k in ('time','kind','name','status')])


def show_region_library(parent,reconstruction):
    import os
    import queue
    import subprocess
    import sys
    import threading
    import tkinter as tk
    from tkinter import ttk,messagebox,filedialog,simpledialog
    existing=getattr(parent,'_export_history_window',None)
    if existing is not None and existing.winfo_exists():existing.deiconify();existing.lift();return existing
    window=tk.Toplevel(parent);parent._export_history_window=window;window.title('当前模型 · 区域方案库');window.geometry('1200x740')
    ttk.Label(window,text='按来源区域与方案结果分组。工程用于继续编辑；分享包用于展示。移除索引不删除任何实际文件。').pack(anchor='w',padx=12,pady=10)
    filters=ttk.Frame(window);filters.pack(fill='x',padx=12)
    search=tk.StringVar();kind_filter=tk.StringVar(value='全部');show_hidden=tk.BooleanVar()
    ttk.Label(filters,text='名称搜索').pack(side='left')
    ttk.Entry(filters,textvariable=search).pack(side='left')
    ttk.Combobox(filters,textvariable=kind_filter,values=['全部','区域','区域＋路径','区域工程 / 方案','标记'],state='readonly',width=20).pack(side='left')
    ttk.Checkbutton(filters,text='显示已移除索引',variable=show_hidden,command=lambda:render()).pack(side='left')
    import time
    thumbnail,tree,detail,actions,project_actions,rows,mailbox,busy,outcome,control=_build_region_library_table(window)
    from femto.planning.planning_io import TaskControl
    def close():
        control.set();window.destroy()
    window.protocol('WM_DELETE_WINDOW',close)
    def selected():
        selection=tree.selection()
        return rows[int(selection[0])] if selection and selection[0].isdigit() else None
    def update(_=None):
        row=selected();preview.configure(state='normal' if row and row['asset'] else 'disabled')
        folder_button.configure(state='normal' if row else 'disabled')
        diagram_button.configure(state='normal' if row and row.get('diagram') else 'disabled')
        archive_button.configure(state='normal' if row and row.get('archive') else 'disabled')
        planning_button.configure(state='normal' if row and row.get('file_status') not in ('损坏','文件丢失') and (row.get('project') or row.get('region_index') is not None) else 'disabled')
        if row:
            detail.set(str(row['folder'])+'\n'+row['status']+'\n版本备注：'+row.get('note',''))
            archive_button.configure(text='打开工程 ZIP' if row.get('project') else '打开分享包 ZIP')
        _update_library_thumbnail(row,thumbnail,window)
    def open_folder():
        row=selected()
        if row:
            try:os.startfile(str(row['folder']))
            except OSError as error:messagebox.showerror('打开失败',str(error),parent=window)
    def open_asset(key):
        row=selected()
        if row and row.get(key):
            try:os.startfile(str(row[key]))
            except OSError as error:messagebox.showerror('打开失败',str(error),parent=window)
    def open_preview(_=None):
        row=selected()
        if _ is not None and row and (row.get('project') or row.get('region_index') is not None):plan_selected();return
        if row and row['asset']:
            if not row['asset'].is_file():
                messagebox.showerror('文件缺失','导出文件已移动或删除，请刷新记录。',parent=window);return
            subprocess.Popen([sys.executable,'-m','femto.viewer.export_preview',str(row['asset'])],cwd=REPO_ROOT)
    def import_project():
        path=filedialog.askopenfilename(parent=window,title='导入区域工程到历史记录',filetypes=[('Region project','*.zip')])
        if not path:return
        metadata_action(lambda:register_project(reconstruction,path,check=control.check))
    def plan_selected():
        row=selected()
        if not row or row.get('file_status') in ('损坏','文件丢失'):return
        from femto.planning.planning_workspace import open_workspace
        if row.get('project'):
            if not row['project'].is_file():
                messagebox.showerror('文件缺失','工程已移动或删除，请重新导入。',parent=window);return
            open_workspace(window,row['project'],on_open=window.withdraw)
        elif row.get('region_index') is not None:
            open_workspace(window,row.get('target',row['source']),region_index=row['region_index'],on_open=window.withdraw)
    def refresh(action=None):
        nonlocal busy,control
        if busy:return
        control=TaskControl('读取区域与资产指纹')
        busy=True;outcome[0]='';refresh_button.configure(state='disabled');detail.set('正在读取导出记录…')
        def run():
            try:
                if action:
                    result=action()
                    if isinstance(result,dict) and 'found' in result:
                        outcome[0]=f"；找到 {len(result['found'])} 工程，跳过 {len(result['skipped'])} 个非工程/损坏 ZIP"
                        if result['skipped']:outcome[0]+='；首个原因：'+result['skipped'][0]['reason']
                from femto.planning.region_library import rebuild
                mailbox.put((rebuild(reconstruction,check=control.check,commit=control.publish),None))
            except Exception as error:mailbox.put(([],str(error)))
        threading.Thread(target=run,daemon=True).start()
        window.after(100,poll)
    def poll():
        nonlocal busy,rows
        if not window.winfo_exists():return
        detail.set(f'{control.stage} · {time.monotonic()-control.started:.1f} 秒（可取消）')
        try:result,error=mailbox.get_nowait()
        except queue.Empty:window.after(100,poll);return
        busy=False;refresh_button.configure(state='normal')
        if not error:rows=result
        render()
        detail.set(error or f'共 {len(rows)} 条记录 · {time.monotonic()-control.started:.1f} 秒{outcome[0]}。双击区域/工程进入规划；路径导出只作预览。')
        update()
    def render(*_):
        _render_library_rows(tree,rows,show_hidden,search,kind_filter)
        update()
    search.trace_add('write',render);kind_filter.trace_add('write',render)
    def picked():return [rows[int(i)] for i in tree.selection() if i.isdigit()]
    def metadata_action(action):
        if busy:return
        refresh(action)
    def hide(value):
        from femto.planning.region_library import edit_records
        chosen=picked()
        if chosen:metadata_action(lambda:edit_records(reconstruction,chosen,hidden=value,check=control.check,commit=control.publish))
    def note():
        from femto.planning.region_library import edit_records
        chosen=picked()
        if not chosen:return
        value=simpledialog.askstring('版本备注','仅修改所选记录的库备注，不改工程内容：',initialvalue=chosen[0].get('note',''),parent=window)
        if value is not None:metadata_action(lambda:edit_records(reconstruction,chosen,note=value,check=control.check,commit=control.publish))
    def details():
        row=selected()
        if not row:return
        dialog=tk.Toplevel(window);dialog.title('来源与版本详情');text=tk.Text(dialog,width=100,height=26);text.pack(fill='both',expand=True)
        text.insert('1.0',json.dumps(row['details'],ensure_ascii=False,indent=2));text.configure(state='disabled')
    def relink():
        from femto.planning.region_library import relocate
        row=selected()
        if not row:return
        path=filedialog.askopenfilename(parent=window,title='选择移动后的原工程 ZIP 或原导出清单 JSON（必须内容一致）',filetypes=[('工程或清单','*.zip *.json')])
        if path:metadata_action(lambda:relocate(reconstruction,row,path,check=control.check,commit=control.publish))
    def discover():
        from femto.planning.region_library import discover_projects
        directory=filedialog.askdirectory(parent=window,title='选择工程所在目录（最多扫描 20000 项）')
        if directory:metadata_action(lambda:discover_projects(reconstruction,directory,check=control.check))
    def rebuild_library():
        from femto.planning.region_library import rebuild,repair_index
        def action():
            control.check();repair_index(reconstruction)
            rebuild(reconstruction,check=control.check,commit=control.publish)
        metadata_action(action)
    library_actions=ttk.Frame(window);library_actions.pack(fill='x',padx=12,pady=4)
    for title,command in [('移除所选索引（不删文件）',lambda:hide(True)),('恢复所选索引',lambda:hide(False)),('版本备注',note),('来源详情',details),('重新定位文件',relink),('扫描目录找工程',discover),('重建/修复索引',rebuild_library),('取消读取',lambda:control.set())]:
        ttk.Button(library_actions,text=title,command=command).pack(side='left',padx=3)
    preview=ttk.Button(actions,text='预览选中区域 / 路径',command=open_preview,state='disabled');preview.pack(side='left',padx=4)
    folder_button=ttk.Button(actions,text='打开导出文件夹',command=open_folder,state='disabled');folder_button.pack(side='left',padx=4)
    diagram_button=ttk.Button(actions,text='查看路线图 PNG',command=lambda:open_asset('diagram'),state='disabled');diagram_button.pack(side='left',padx=4)
    archive_button=ttk.Button(actions,text='打开分享包 ZIP',command=lambda:open_asset('archive'),state='disabled');archive_button.pack(side='left',padx=4)
    refresh_button=ttk.Button(actions,text='刷新记录',command=refresh);refresh_button.pack(side='right',padx=4)
    planning_button=ttk.Button(project_actions,text='打开规划工作台',command=plan_selected,state='disabled');planning_button.pack(side='left',padx=4)
    ttk.Button(project_actions,text='导入已有区域工程',command=import_project).pack(side='left',padx=4)
    tree.bind('<<TreeviewSelect>>',update);tree.bind('<Double-1>',open_preview);refresh()
    return window


def show_history(parent,reconstruction):
    """Product view: browse actual exported region assets only."""
    import os
    import queue
    import subprocess
    import sys
    import threading
    import time
    import tkinter as tk
    from tkinter import ttk,messagebox
    existing=getattr(parent,'_export_history_window',None)
    if existing is not None and existing.winfo_exists():existing.deiconify();existing.lift();return existing
    window=tk.Toplevel(parent);parent._export_history_window=window
    window.title('当前模型 · 历史区域导出记录');window.geometry('1000x620');window.minsize(760,480)
    ttk.Label(window,text='仅显示已经导出的区域数据。双击可预览区域模型；打开文件夹可查看 GLB、PLY、CSV 和 JSON。').pack(anchor='w',padx=12,pady=10)
    filters=ttk.Frame(window);filters.pack(fill='x',padx=12)
    search=tk.StringVar();ttk.Label(filters,text='区域名称搜索').pack(side='left')
    ttk.Entry(filters,textvariable=search).pack(side='left',fill='x',expand=True,padx=(6,0))
    frame=ttk.Frame(window);frame.pack(fill='both',expand=True,padx=12,pady=(8,0))
    tree=ttk.Treeview(frame,columns=('time','name','status'),show='headings',selectmode='browse')
    for key,title,width,stretch in [('time','导出时间',180,False),('name','区域名称 / ID',260,True),('status','文件状态',360,True)]:
        tree.heading(key,text=title);tree.column(key,width=width,minwidth=100,stretch=stretch)
    scroll=ttk.Scrollbar(frame,command=tree.yview);tree.configure(yscrollcommand=scroll.set)
    scroll.pack(side='right',fill='y');tree.pack(fill='both',expand=True)
    detail=tk.StringVar(value='正在读取历史区域导出记录…')
    ttk.Label(window,textvariable=detail,wraplength=940,justify='left').pack(fill='x',padx=12,pady=10)
    actions=ttk.Frame(window);actions.pack(fill='x',padx=12,pady=(0,12))
    preview=ttk.Button(actions,text='预览区域模型',state='disabled');preview.pack(side='left',padx=(0,6))
    folder_button=ttk.Button(actions,text='打开导出文件夹',state='disabled');folder_button.pack(side='left')
    refresh_button=ttk.Button(actions,text='刷新记录');refresh_button.pack(side='right')
    from femto.planning.planning_io import TaskControl
    rows=[];mailbox=queue.Queue(maxsize=1);busy=False;control=TaskControl('读取历史区域导出')
    def close():
        control.set();window.destroy()
    window.protocol('WM_DELETE_WINDOW',close)
    def selected():
        selection=tree.selection()
        return rows[int(selection[0])] if selection and selection[0].isdigit() and int(selection[0])<len(rows) else None
    def update(_=None):
        row=selected();asset=row.get('asset') if row else None
        preview.configure(state='normal' if asset and asset.is_file() else 'disabled')
        folder_button.configure(state='normal' if row and row['folder'].is_dir() else 'disabled')
        if row:
            files='区域模型：'+(asset.name if asset else '文件缺失')
            detail.set(str(row['folder'])+'\n'+files+'\n'+row['status'])
        elif not busy:detail.set('请选择一条历史区域导出记录。')
    def render(*_):
        tree.delete(*tree.get_children())
        query=search.get().casefold()
        for i,row in enumerate(rows):
            if query and query not in row['name'].casefold():continue
            tree.insert('','end',iid=str(i),values=(row['time'],row['name'],row['status']))
        update()
    def open_preview(_=None):
        row=selected()
        if not row:return
        asset=row.get('asset')
        if not asset or not asset.is_file():
            messagebox.showerror('文件缺失','区域模型文件已移动或删除，请刷新记录。',parent=window);return
        subprocess.Popen([sys.executable,'-m','femto.viewer.export_preview',str(asset)],cwd=REPO_ROOT)
    def open_folder():
        row=selected()
        if row:
            try:os.startfile(str(row['folder']))
            except OSError as error:messagebox.showerror('打开失败',str(error),parent=window)
    def refresh():
        nonlocal busy,control
        if busy:return
        control=TaskControl('读取历史区域导出');busy=True;refresh_button.configure(state='disabled')
        detail.set('正在读取历史区域导出记录…')
        def run():
            try:
                result=[row for row in scan_region_exports(reconstruction,check=control.check) if row['kind']=='区域']
                mailbox.put_nowait((result,None))
            except Exception as error:
                try:mailbox.put_nowait((None,str(error)))
                except queue.Full:pass
        threading.Thread(target=run,daemon=True,name='RegionExportHistory').start();window.after(80,poll)
    def poll():
        nonlocal busy,rows
        if not window.winfo_exists():return
        try:result,error=mailbox.get_nowait()
        except queue.Empty:
            detail.set(f'{control.stage} · {time.monotonic()-control.started:.1f} 秒');window.after(80,poll);return
        busy=False;refresh_button.configure(state='normal')
        if error:detail.set('读取失败：'+error)
        else:
            rows=result;render();detail.set(f'共找到 {len(rows)} 条历史区域导出记录。')
            if rows:tree.selection_set('0');tree.see('0');update()
    search.trace_add('write',render)
    preview.configure(command=open_preview);folder_button.configure(command=open_folder);refresh_button.configure(command=refresh)
    tree.bind('<<TreeviewSelect>>',update);tree.bind('<Double-1>',open_preview)
    refresh();return window

