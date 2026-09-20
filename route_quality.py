"""Quantitative polyline diagnostics; exact retraversal, not physical swath overlap."""
import numpy as np
from route_motion import statistics

def quality(plan,check=lambda:None):
    result=statistics(plan);seen=set();repeat=0.;repeated_edges=0;turns=[]
    for ci,chain in enumerate(plan['segments']):
        check();xyz=np.asarray([p['xyz_m'] for p in chain]);delta=np.diff(xyz,axis=0);length=np.linalg.norm(delta,axis=1)
        for ordinal,(a,b,d) in enumerate(zip(xyz[:-1],xyz[1:],length)):
            if ordinal%1024==0:check()
            edge=tuple(sorted((tuple(a),tuple(b))))
            if edge in seen:repeat+=float(d);repeated_edges+=1
            seen.add(edge)
        for i in range(1,len(chain)-1):
            if i%1024==0:check()
            if min(length[i-1],length[i])<=1e-12:continue
            angle=float(np.degrees(np.arccos(np.clip(delta[i-1]@delta[i]/(length[i-1]*length[i]),-1,1))))
            if angle>=60-1e-9:
                turns.append(dict(chain_index=ci,sample_index=i,pixel=chain[i]['pixel'],xyz_m=chain[i]['xyz_m'],deflection_degrees=angle))
    result.update(repeated_edge_length_m=repeat,repeated_edge_count=repeated_edges,
        repeat_semantics='extra_traversals_of_exact_undirected_xyz_edges_not_nearby_swath_overlap',
        disconnected_break_count=max(0,len(plan['segments'])-1),sharp_turn_threshold_degrees=60.,sharp_turn_count=len(turns),
        sampled_coverage_percent=plan.get('coverage',{}).get('sampled_coverage_percent'),
        coverage_denominator_m2=plan.get('coverage',{}).get('target_mesh_area_m2'))
    return result,turns

def quality_text(plan):
    q=plan.get('route_quality')
    if not q:return '旧版无路线质量对照；重新生成新版可获得质量与急转角报告。'
    before=q['before'];after=q['after']
    lines=['路线质量对照（同一数据和参数，基线 serpentine-v3）',
        '结果：'+('量化改善并通过约束/覆盖核验' if q['status']=='improved' else '未获得通过全部门禁的改善，保留原折线')]
    for label,key in [('扫描长度','scan_length_m'),('连接长度','connector_length_m'),('总长','total_length_m'),('精确边重复长度','repeated_edge_length_m')]:
        lines.append(f"{label}：{before[key]*1000:.3f} → {after[key]*1000:.3f} mm")
    for label,key in [('断链数','disconnected_break_count'),('急转点数','sharp_turn_count'),('采样覆盖比例%','sampled_coverage_percent'),('20mm/s已规划时间s','known_motion_time_s')]:
        lines.append(f"{label}：{before[key]} → {after[key]}")
    lines += ['重复长度只计精确相同无向XYZ边的额外经过，不是扫描带重叠面积。',
        '急转点阈值：相邻三维弦段方向变化≥60°，用于研发标注，不是速度/加速度或接触姿态结论。',
        '每个原已覆盖采样点都必须保留，分母和链数不变；没有以覆盖下降换取更短连接。',
        '算法限原认证边图内三种确定性候选、累计200万展开、每链512扫描段；不新增穿越表面的捷径。',
        '平滑：'+q['smoothing']['reason'],
        '墙钟生成耗时在同源对照记录中测量，不写入可复现结果指纹。']
    labels={'not_improved':'未满足长度/重复非退化门禁','coverage_regression_rejected':'覆盖退化，已拒绝','validated':'核验通过','budget_exceeded':'预算超限，停止搜索','validation_failed':'核验失败','footprint_rejected':'模拟包络未通过'}
    for attempt in q['attempts']:
        lines.append(attempt['method']+'：'+labels.get(attempt['status'],attempt['status'])+' '+attempt.get('reason','')+('；丢失原覆盖采样 '+str(attempt['lost_covered_samples'])+' 个' if 'lost_covered_samples' in attempt else ''))
    return '\n'.join(lines)


def show_quality(parent,plan,locate,is_current):
    import tkinter as tk
    from tkinter import ttk
    w=tk.Toplevel(parent);w.title('路线质量与急转角');w.geometry('880x720')
    text=tk.Text(w,height=20,wrap='word');text.pack(fill='x');text.insert('1.0',quality_text(plan));text.configure(state='disabled')
    ttk.Label(w,text='点击急转点，暂停播放并定位；定位不是已规划跨链运动。').pack(fill='x')
    frame=ttk.Frame(w);frame.pack(fill='both',expand=True)
    tree=ttk.Treeview(frame,columns=('chain','point','angle'),show='headings',selectmode='browse')
    for key,label in [('chain','路线链'),('point','采样点'),('angle','方向变化 / °')]:tree.heading(key,text=label)
    scrollbar=ttk.Scrollbar(frame,command=tree.yview);tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right',fill='y');tree.pack(fill='both',expand=True)
    turns=plan['route_quality']['sharp_turns']
    for i,t in enumerate(turns):tree.insert('', 'end',iid=str(i),values=(t['chain_index']+1,t['sample_index']+1,f"{t['deflection_degrees']:.2f}"))
    def selected(_):
        if not is_current():w.destroy();return
        if tree.selection():locate(turns[int(tree.selection()[0])])
    tree.bind('<<TreeviewSelect>>',selected);w.turn_tree=tree
    return w
