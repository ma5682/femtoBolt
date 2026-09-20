"""Fixed-reference route diagrams for product demonstration, not execution."""
from __future__ import annotations
from typing import Any, Callable

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from femto.planning.region_path import path_summary
from femto.planning.route_motion import MotionPlayer, statistics


def render_diagram(rgb: Any, ids: Any, region: Any, plan: Any, show_region: bool=True, show_path: bool=True, segment_index: int | None=None,
                   show_coverage: bool=False, cursor: Any=None, show_turns: bool=True) -> Any:
    height,width=rgb.shape[:2];scale=min(8. if plan.get('route_quality') or plan.get('surface_mode') else 1.,1400/width,900/height)
    size=(max(1,int(width*scale)),max(1,int(height*scale)))
    mask=np.isin(ids,region['triangle_indices'])
    base=np.full_like(rgb,24)
    if show_region:base[mask]=rgb[mask]
    coverage=plan.get('coverage')
    if show_coverage and plan.get('coverage_inspection'):
        from femto.planning.coverage_inspector import render_map
        base=np.asarray(render_map(rgb if show_region else np.zeros_like(rgb),plan,header=False)).copy()
    elif show_coverage and coverage:
        for counts,color in [([3],[35,190,110]),([0,1,2],[235,110,45])]:
            triangles=[tid for tid,n in zip(coverage['triangle_indices'],coverage['covered_samples_per_face']) if n in counts]
            selected=np.isin(ids,triangles)
            base[selected]=(base[selected]*.35+np.asarray(color)*.65).astype(np.uint8)
        polygon=region.get('screen_polygon')
        if polygon:
            polygon_mask=Image.new('1',(width,height));ImageDraw.Draw(polygon_mask).polygon([tuple(p) for p in polygon],fill=1)
            missing=np.asarray(polygon_mask)&(ids<0)
            base[missing]=[180,70,200]
    picture=Image.fromarray(base).resize(size,Image.Resampling.LANCZOS)
    header=(293 if plan.get('route_quality') else 265) if plan.get('start_selection') else 125
    if plan.get('surface_mode'):header=237
    canvas=Image.new('RGB',(max(size[0],1100) if plan.get('route_quality') or plan.get('surface_mode') else size[0],size[1]+header),(18,28,38));canvas.paste(picture,(0,header))
    draw=ImageDraw.Draw(canvas)
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(font_path),17) if font_path.exists() else ImageFont.load_default()
    summary=path_summary(plan)
    spacing=(f"近似曲面间距 {plan['spacing_mm']:g} mm" if 'spacing_mm' in plan else f"图像行间距 {plan['row_step_px']} px")
    lines=['圈选区域扫描路线图 · 固定初始视角',
           f"{spacing} | {summary['segment_count']} 段 | {summary['sample_count']} 点 | 段内总长 {summary['total_segment_length_m']*1000:.1f} mm",
           '绿点：每段起点  红点：每段终点  黄箭头：段内方向；段间无过渡路线',
           '研发预览 · 非机械臂执行轨迹'+(' | 当前仅显示第 '+str(segment_index+1)+' 段' if segment_index is not None else '')]
    if show_coverage and coverage:
        lines[2]=f"采样覆盖估计 {coverage['sampled_coverage_percent']:.1f}% | 绿：3/3采样覆盖 橙：有遗漏 紫：无几何"
        lines[3]='离散面积加权估计，不保证全表面覆盖；缺失几何不计入分母。段间无过渡。'
    if plan.get('coverage_parameters',{}).get('pattern')in ('serpentine_v1','serpentine_v2','serpentine_v3','serpentine_v4'):
        lines[1]=f"水平往返 | 行距 {plan['row_step_px']} px | {summary['segment_count']} 条连续链 | 总长 {summary['total_segment_length_m']*1000:.1f} mm"
        if not show_coverage:lines[2]='绿点：总起点；红点：总终点；箭头：运动方向。'
        lines[3]='黄线：扫描；橙线：换行/绕行连接。断链无过渡；研发演示，非设备执行轨迹。'
    if plan.get('start_selection'):
        params=plan['coverage_parameters'];start=plan['start_selection']
        direction_name={'horizontal':'水平往返','vertical':'垂直往返'}[params['direction']]
        corner_name={'top_left':'左上','top_right':'右上','bottom_left':'左下','bottom_right':'右下'}[params['start_corner']]
        lines[1]=f"{direction_name} / {corner_name} | {summary['chain_count']} 条链 | 总长 {summary['total_length_m']*1000:.1f} mm"
        lines += [f"扫描 {summary['scan_length_m']*1000:.1f} mm | 连接 {summary['connector_length_m']*1000:.1f} mm | 20 mm/s 已规划运动 {summary['known_motion_time_s']:.1f} s",
                  f"实际起点像素 {start['actual_pixel']}；{start['adjustment_cause'] if start['adjusted'] else '偏好起点可用'}",
                  f"实际起点 W0/RDF XYZ（米）{[round(float(v),6) for v in start['actual_xyz_m']]}；按扫掠顺序选择，不保证最近点。",
                  '断链处停止；跨链移动未规划、耗时未知，不插值。进度定位不代表运动。',
                  '时间仅含链内扫描和连接；不含停顿、加减速及跨链移动。']
    if show_coverage and plan.get('coverage_inspection'):
        lines[2]='绿/橙仅为覆盖/未覆盖采样点；青：有效几何；红：禁区；蓝：预留；紫：无几何。'
        lines[3]='分母为圈内允许且可信的采样代表面积；无几何面积未知。完整解释及定位见覆盖报告。'
    if plan.get('route_quality') and len(plan['segments'])>12 and not show_coverage:
        lines[2]='绿/红点：各链起/终点，不画跨链连接；链多时总览省略文字，编号请逐链查看。'
    if plan.get('surface_mode'):
        sr=plan['spacing_report'];dist=sr['distribution']
        lines[0]='三维表面间距模式 · W0/RDF · 独立扫描链，无跨链连接'
        lines[1]=f"目标表面层间距 {sr['target_spacing_mm']:g} mm | {summary['segment_count']} 链 | 非精确测地等距"
        lines += ['实际网格距离上界分布 mm：'+(f"P05 {dist['p05_mm']:.3f} / 中位 {dist['median_mm']:.3f} / P95 {dist['p95_mm']:.3f}" if dist else '无可测样本'),
            '间距状态：'+('采样上下界在容差内' if sr['status']=='within_sampled_bounds' else '未完全满足或无法证实；详见间距报告'),
            '网格方向、粗细、孔洞/禁区及边界裁剪可造成偏差；不保证真实治疗覆盖精度。']
    if plan.get('boundary_check'):
        guard=plan['boundary_check']
        lines[0]+=f" · 模拟宽度 {guard['width_mm']:g} mm / 边界余量 {guard['margin_mm']:g} mm"
    for i,line in enumerate(lines):draw.text((12,8+28*i),line,font=font,fill='#e4edf4')
    polygon=region.get('screen_polygon')
    if polygon:
        outline=np.asarray(polygon)*scale+[0,header]
        draw.line([tuple(p) for p in np.vstack([outline,outline[0]])],fill='#ee80ff',width=2)
    for forbidden in region.get('exclusion_polygons',[]):
        outline=np.asarray(forbidden)*scale+[0,header]
        draw.polygon([tuple(p) for p in outline],fill='#702a30',outline='#ff4242',width=3)
        draw.text(tuple(outline.mean(axis=0)), '禁止',font=font,fill='#ffffff')
    if show_path:
        selected=enumerate(plan['segments']) if segment_index is None else [(segment_index,plan['segments'][segment_index])]
        for index,segment in selected:
            points=np.asarray([sample['pixel'] for sample in segment])*scale+[0,header]
            draw.line([tuple(p) for p in points],fill='#ffd52b',width=2)
            for j in range(1,len(segment)):
                if segment[j].get('motion_kind')=='connector':
                    draw.line([tuple(points[j-1]),tuple(points[j])],fill='#ff9138',width=2)
            # Regular direction markers along continuous chains.
            lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
            valid=np.flatnonzero(lengths>1e-8)
            targets=np.arange(30,lengths.sum(),70) if plan.get('coverage_parameters',{}).get('pattern')in ('serpentine_v1','serpentine_v2','serpentine_v3','serpentine_v4') else [lengths.sum()/2]
            for target in targets if len(valid) else []:
                edge=min(len(lengths)-1,int(np.searchsorted(np.cumsum(lengths),target)))
                direction=(points[edge+1]-points[edge])/lengths[edge]
                tip=(points[edge]+points[edge+1])/2;back=tip-direction*9
                normal=np.array([-direction[1],direction[0]])*4
                draw.polygon([tuple(tip),tuple(back+normal),tuple(back-normal)],fill='#ffd52b')
            for p,color in ((points[0],'#35ea87'),(points[-1],'#ff5765')):
                draw.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill=color)
            if segment_index is not None or (len(plan['segments'])>1 and (not (plan.get('route_quality') or plan.get('surface_mode')) or len(plan['segments'])<=12)):
                draw.text(tuple(points[0]+[5,4]),'链 '+str(index+1)+' 起',font=font,fill='#35ea87')
                draw.text(tuple(points[-1]+[5,4]),'链 '+str(index+1)+' 终/断点',font=font,fill='#ff5765')
    if show_path and show_turns and plan.get('route_quality'):
        for turn in plan['route_quality']['sharp_turns']:
            if segment_index is not None and turn['chain_index']!=segment_index:continue
            x,y=np.asarray(turn['pixel'])*scale+[0,header]
            draw.polygon([(x,y-4),(x+4,y),(x,y+4),(x-4,y)],outline='#ff70ee')
        draw.text((12,header-24),'粉色菱形：方向变化≥60°的折线急转点；未平滑，非动力学结论。',font=font,fill='#ff70ee')
    if cursor is not None:
        p=np.asarray(cursor)*scale+[0,header]
        draw.ellipse((p[0]-7,p[1]-7,p[0]+7,p[1]+7),fill='#20dfff',outline='white',width=2)
    if show_path and plan['segments']:
        for sample,label,color in [(plan['segments'][0][0],'总起点','#35ea87'),(plan['segments'][-1][-1],'总终点','#ff5765')]:
            p=np.asarray(sample['pixel'])*scale+[0,header]
            draw.text(tuple(p+[8,0]),label,font=font,fill=color)
    return canvas


def motion_position(plan: Any,elapsed: float,speed_mm_s: float=20.,gap_seconds: float=.5) -> tuple[Any, Any, str]:
    """Legacy stateless API only; the interactive player uses MotionPlayer and stops at breaks."""
    for index,segment in enumerate(plan['segments']):
        xyz=np.asarray([p['xyz_m'] for p in segment]);uv=np.asarray([p['pixel'] for p in segment])
        distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
        duration=distance[-1]*1000/speed_mm_s
        if elapsed<=duration:
            target=elapsed*speed_mm_s/1000
            edge=min(max(0,int(np.searchsorted(distance,target,side='right'))-1),len(xyz)-2)
            t=np.clip((target-distance[edge])/max(distance[edge+1]-distance[edge],1e-15),0,1)
            return index,(uv[edge]*(1-t)+uv[edge+1]*t).tolist(),'moving'
        elapsed-=duration
        if index<len(plan['segments'])-1:
            if elapsed<gap_seconds:return index,None,'gap'
            elapsed-=gap_seconds
    return len(plan['segments'])-1,None,'finished'


def show_presentation(parent: Any,model: Any,region: Any,plan: Any,is_current: Callable[..., Any]=lambda:True) -> None:
    import tkinter as tk
    import time
    from tkinter import ttk,filedialog,messagebox
    previous=getattr(parent,'_route_demo_window',None)
    if previous is not None and previous.winfo_exists():previous.close_demo()
    window=tk.Toplevel(parent);window.title('圈选区域 · 扫描路线图');window.geometry('1100x850')
    parent._route_demo_window=window
    window.plan_is_current=is_current
    controls=ttk.Frame(window);controls.pack(fill='x',padx=10,pady=8)
    playback=ttk.Frame(window);playback.pack(fill='x',padx=10,pady=2)
    show_region=tk.BooleanVar(value=True);show_path=tk.BooleanVar(value=True)
    show_coverage=tk.BooleanVar(value=bool(plan.get('coverage')))
    show_turns=tk.BooleanVar(value=bool(plan.get('route_quality')))
    selection=tk.StringVar(value='全部路径段')
    label=ttk.Label(window);label.pack(fill='both',expand=True)
    current=[None];timer=[None];animation=[None];cursor=[None];last_tick=[0.]
    player=MotionPlayer(plan);window.motion_player=player
    state=tk.StringVar();speed=tk.DoubleVar(value=20.);progress=tk.DoubleVar(value=0.)
    updating=[False];notice=['']
    ttk.Label(window,textvariable=state,wraplength=1050).pack(side='bottom',fill='x')
    def update_state():
        pos=player.position();window.motion_state=pos;cursor[0]=pos['pixel']
        updating[0]=True;progress.set(pos['progress']*100);updating[0]=False
        xyz=', '.join(f'{v:.6f}' for v in pos['xyz_m']);direction=', '.join(f'{v:.3f}' for v in pos['direction_W0'])
        info=statistics(plan,player.speed)
        gap=''
        if player.at_break:
            breaks=plan.get('chain_breaks',[])
            reason=breaks[player.chain]['explanation'] if player.chain<len(breaks) else '不同链之间没有已规划连接'
            gap='断点停止：'+reason+'；点击继续仅定位下一链起点。'
        state.set(f"{'播放结束' if player.finished else '播放中' if player.running else '已暂停'} · 链 {player.chain+1}/{len(player.xyz)} · {'连接' if pos['motion_kind']=='connector' else '扫描'} · W0/RDF XYZ（米）({xyz}) · 方向单位向量 ({direction})\n"
            f"进度 {pos['progress']*100:.1f}% / 对应恒速时间 {pos['known_time_s']:.1f} s；扫描 {info['scan_length_m']*1000:.1f} / 连接 {info['connector_length_m']*1000:.1f} / 总长 {info['total_length_m']*1000:.1f} mm；{player.speed:g} mm/s 已规划运动 {info['known_motion_time_s']:.1f} s（仅链内，不含暂停、加减速或未知跨链移动）。\n"+gap+notice[0])
    def seek(value):
        if updating[0]:return
        pause();player.seek(float(value)/100);notice[0]='进度定位；该定位不是已规划运动。'
        if selection.get()!='全部路径段':selection.set(f'第 {player.chain+1} 段')
        update_state();redraw()
    window.seek_progress=seek
    ttk.Scale(window,variable=progress,from_=0,to=100,command=seek).pack(side='bottom',fill='x',padx=10)
    def redraw():
        if not is_current():closed();return
        value=selection.get();index=None if value=='全部路径段' else int(value.split()[1])-1
        current[0]=render_diagram(model.rgb,model.projection.ids,region,plan,show_region.get(),show_path.get(),index,
            show_coverage.get(),cursor[0],show_turns.get())
        display=current[0].copy();display.thumbnail((max(200,window.winfo_width()-20),max(200,window.winfo_height()-70)))
        label.image=ImageTk.PhotoImage(display);label.configure(image=label.image)
    def resized(event):
        if event.widget is not window:return
        if timer[0]:window.after_cancel(timer[0])
        timer[0]=window.after(100,redraw)
    def save():
        if not is_current():closed();return
        path=filedialog.asksaveasfilename(parent=window,title='导出路线图 PNG',defaultextension='.png',
            initialdir=str(model.reconstruction),initialfile='区域扫描路线图.png',filetypes=[('PNG','*.png')])
        if path and is_current():
            try:current[0].save(path)
            except OSError as error:messagebox.showerror('导出失败',str(error),parent=window)
    ttk.Checkbutton(controls,text='显示区域',variable=show_region,command=redraw).pack(side='left')
    ttk.Checkbutton(controls,text='显示路径',variable=show_path,command=redraw).pack(side='left')
    ttk.Checkbutton(controls,text='覆盖检查',variable=show_coverage,command=redraw,
        state='normal' if plan.get('coverage') else 'disabled').pack(side='left')
    def tick():
        animation[0]=None
        if not is_current():closed();return
        if not player.running:return
        now=time.monotonic();player.advance(now-last_tick[0]);last_tick[0]=now
        update_state();redraw()
        if player.running:animation[0]=window.after(100,tick)
    def play():
        if not is_current():closed();return
        if player.running:return
        switched=player.play();notice[0]='已定位下一链起点；跨链动作未规划。' if switched else ''
        if selection.get()!='全部路径段':selection.set(f'第 {player.chain+1} 段')
        last_tick[0]=time.monotonic();tick()
    def pause():
        if player.running:player.advance(time.monotonic()-last_tick[0])
        player.pause()
        if animation[0]:window.after_cancel(animation[0]);animation[0]=None
        update_state()
    def reset():
        pause();player.reset();notice[0]='已复位。';selection.set('全部路径段');update_state();redraw()
    for text,command in [('播放',play),('暂停',lambda:(pause(),redraw())),('继续',play),('复位',reset)]:
        ttk.Button(playback,text=text,command=command,width=5).pack(side='left')
    ttk.Label(playback,text='模拟速度 mm/s').pack(side='left',padx=8)
    speed_box=ttk.Combobox(playback,textvariable=speed,values=[5,10,20,40],state='readonly',width=5)
    def change_speed(_):
        pause();player.set_speed(speed.get());notice[0]='速度已更新，点击继续。';update_state();redraw()
    speed_box.pack(side='left');speed_box.bind('<<ComboboxSelected>>',change_speed)
    def report():
        if not is_current():closed();return
        from femto.planning.coverage_inspector import show_inspector
        pause()
        show_inspector(window,model,plan,is_current)
    ttk.Button(playback,text='查看覆盖报告',command=report).pack(side='right')
    if plan.get('spacing_report'):
        def locate_spacing(row):
            if not is_current():closed();return
            ci=row['chain_index'];index=row['sample_index']
            selection.set('全部路径段')
            seek(100*(player.offsets[ci]+player.distances[ci][index])/max(player.offsets[-1],1e-15))
        window.locate_spacing=locate_spacing
        def spacing_dialog():
            if not is_current():closed();return
            pause()
            from femto.planning.surface_spacing_report import show_spacing_report
            show_spacing_report(window,plan,locate_spacing,is_current)
        ttk.Button(playback,text='表面间距报告',command=spacing_dialog).pack(side='right')

    if plan.get('route_quality'):
        ttk.Checkbutton(controls,text='急转点',variable=show_turns,command=redraw).pack(side='left')
        def locate_turn(turn):
            if not is_current():closed();return
            ci=turn['chain_index'];index=turn['sample_index']
            selection.set('全部路径段')
            seek(100*(player.offsets[ci]+player.distances[ci][index])/max(player.offsets[-1],1e-15))
            notice[0]=f"急转点：链 {ci+1} / 点 {index+1}，方向变化 {turn['deflection_degrees']:.2f}°；未平滑。"
            update_state();redraw()
        window.locate_turn=locate_turn
        def inspect_quality():
            if not is_current():closed();return
            from femto.planning.route_quality import show_quality
            pause();show_quality(window,plan,locate_turn,is_current)
        ttk.Button(playback,text='路线质量/急转角',command=inspect_quality).pack(side='right')

    combo=ttk.Combobox(controls,textvariable=selection,state='readonly',width=18,
        values=['全部路径段']+[f'第 {i+1} 段' for i in range(len(plan['segments']))])
    def select(_):
        pause()
        if selection.get()!='全部路径段':player.select_chain(int(selection.get().split()[1])-1)
        notice[0]='逐链定位；不表示跨链运动。';update_state();redraw()
    combo.pack(side='left',padx=10);combo.bind('<<ComboboxSelected>>',select)
    ttk.Button(controls,text='导出路线图 PNG',command=save).pack(side='right')
    def closed():
        pause()
        if timer[0]:window.after_cancel(timer[0])
        window.destroy()
    update_state()
    window.close_demo=closed
    window.protocol('WM_DELETE_WINDOW',closed)
    window.bind('<Configure>',resized);timer[0]=window.after(50,redraw)
