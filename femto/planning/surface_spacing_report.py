"""Measurements to preceding scan levels on the allowed mesh, not image spacing."""
import heapq
import numpy as np
from femto.planning.region_coverage import point_segment_distance


def spacing_distribution(values,target_mm):
    """Summary of measured spacing samples; None when nothing was measurable."""
    values=np.asarray(values,dtype=float);errors=values-target_mm
    return None if not len(values) else dict(min_mm=float(values.min()),p05_mm=float(np.percentile(values,5)),median_mm=float(np.median(values)),p95_mm=float(np.percentile(values,95)),max_mm=float(values.max()),mean_mm=float(values.mean()),signed_mean_deviation_mm=float(errors.mean()),rmse_mm=float(np.sqrt(np.mean(errors**2))),max_abs_deviation_mm=float(np.max(np.abs(errors))))


def spacing_report(mesh,region,plan,target_mm,tolerance_mm,check=lambda:None,budget=None, *, target_limited=False):
    """Share expansion budgets; v4.4 may stop once every queried vertex is settled.

    Target-limited Dijkstra preserves the exact measurements, source field,
    sampling and all caps. Older algorithms retain full-field expansion and its
    saved budget statistics for exact reproduction.
    """
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles);ids=region['triangle_indices']
    graph={int(v):{} for v in np.unique(faces[ids])}
    for ordinal,face in enumerate(faces[ids]):
        if ordinal%1024==0:check()
        for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
            graph[int(a)][int(b)]=graph[int(b)][int(a)]=float(np.linalg.norm(vertices[a]-vertices[b]))
    levels=sorted({p['surface_level_m'] for c in plan['segments'] for p in c})
    state=budget if budget is not None else dict(expansions=0,distance_pairs=0)
    rows=[];max_expanded=2000000
    if len(levels)*len(graph)>10000000:raise ValueError('表面间距测量预算超过1000万层/顶点组合，请增大间距或缩小区域')
    for level in levels:
        check();previous=level-target_mm/1000
        source=[c for c in plan['segments'] if abs(c[0]['surface_level_m']-previous)<1e-9]
        targets=[(i,c) for i,c in enumerate(plan['segments']) if c[0]['surface_level_m']==level]
        # The initial field level has no preceding scan by definition.
        if abs(level-target_mm/2000)<1e-9:continue
        distances={v:float('inf') for v in graph};by_face={};source_points=[];max_edge=0.
        for chain in source:
            source_points.extend(p['xyz_m'] for p in chain)
            for a,b in zip(chain,chain[1:]):
                tid=b['triangle_index'];pa=np.asarray(a['xyz_m']);pb=np.asarray(b['xyz_m'])
                max_edge=max(max_edge,float(np.linalg.norm(pb-pa)));by_face.setdefault(tid,[]).append((pa,pb))
                for v,d in zip(faces[tid],point_segment_distance(vertices[faces[tid]],pa,pb)):
                    distances[int(v)]=min(distances[int(v)],float(d))
        pending=None
        if target_limited:
            pending=set()
            for _,chain in targets:
                check()
                for index in np.unique(np.linspace(0,len(chain)-1,min(32,len(chain))).astype(int)):
                    pending.update(int(v) for v in faces[chain[index]['triangle_index']])
        queue=[(d,v) for v,d in distances.items() if np.isfinite(d)];heapq.heapify(queue)
        while queue:
            d,v=heapq.heappop(queue)
            if d!=distances[v]:continue
            state['expansions']+=1
            if state['expansions']%512==0:check()
            if state['expansions']>max_expanded:raise ValueError('表面间距测量超过200万节点展开预算')
            if pending is not None:
                pending.discard(v)
                if not pending:break
            for other,w in graph[v].items():
                if d+w<distances[other]:distances[other]=d+w;heapq.heappush(queue,(d+w,other))
        # A sample-to-node distance minus the longest source edge is a lower bound to all source segments.
        source_points=np.asarray(source_points)
        for ci,chain in targets:
            for index in np.unique(np.linspace(0,len(chain)-1,min(32,len(chain))).astype(int)):
                check();state['distance_pairs']+=len(source_points)+len(by_face.get(chain[index]['triangle_index'],[]))
                if state['distance_pairs']>100000000:raise ValueError('表面间距测量超过1亿距离对预算，请增大间距/步长或缩小区域')
                p=chain[index];xyz=np.asarray(p['xyz_m']);face=faces[p['triangle_index']]
                upper=min(distances[int(v)]+float(np.linalg.norm(xyz-vertices[v])) for v in face)
                for a,b in by_face.get(p['triangle_index'],[]):upper=min(upper,float(point_segment_distance(xyz[None],a,b)[0]))
                lower=max(0.,float(np.linalg.norm(source_points-xyz,axis=1).min())-max_edge) if len(source_points) else None
                measured=float(upper*1000) if np.isfinite(upper) else None
                lower_mm=lower*1000 if lower is not None and measured is not None else None
                status=('unmeasurable' if measured is None else 'within_bounds' if lower_mm>=target_mm-tolerance_mm and measured<=target_mm+tolerance_mm else 'outside_bounds' if measured<target_mm-tolerance_mm or lower_mm>target_mm+tolerance_mm else 'uncertain_bounds')
                rows.append(dict(chain_index=ci,sample_index=int(index),triangle_index=p['triangle_index'],xyz_m=p['xyz_m'],pixel=p['pixel'],
                    level_m=level,preceding_level_m=previous,surface_graph_distance_mm=measured,lower_bound_mm=lower_mm,
                    signed_deviation_mm=None if measured is None else measured-target_mm,status=status))
    values=np.asarray([r['surface_graph_distance_mm'] for r in rows if r['surface_graph_distance_mm'] is not None])
    distribution=spacing_distribution(values,target_mm)
    counts={status:sum(r['status']==status for r in rows) for status in ('within_bounds','outside_bounds','uncertain_bounds','unmeasurable')}
    return dict(schema='femto-surface-spacing-report-v1',target_spacing_mm=target_mm,tolerance_mm=tolerance_mm,
        method='mesh_edge_shortest_path_upper_bound_with_face_local_endpoint_segments; euclidean_node_minus_max_edge_lower_bound',
        direction='from_each_retained_level_to_immediately_preceding_target_level_not_nearest_arbitrary_chain',
        sampling='up_to_32_uniform_sample_indices_per_chain_not_uniform_area; first_level_excluded',
        sample_count=len(rows),status='within_sampled_bounds' if rows and counts['within_bounds']==len(rows) else 'not_fully_verified',
        counts=counts,distribution=distribution,measurements=rows,budget=dict(expansions=state['expansions'],max_expansions=max_expanded,max_level_vertex_pairs=10000000,distance_pairs=state['distance_pairs'],max_distance_pairs=100000000),
        limitations=['测量为实际保留路线之间的网格可行路径上界，不是精确连续测地距离；图方向、粗网格和障碍会扩大误差。',
            '欧氏下界可能跨空缺，仅作数学下界，不作为连接或运动；不可达不报告有限表面间距。',
            '前一目标层缺失或分量不连通时记录无法测量；不跳过缺层冒称目标等距。',
            '每条链最多32个索引采样，非全表面保证；边界裁剪可能造成大间隙。',
            '无实物治疗覆盖精度验证。'])


def spacing_text(plan):
    r=plan['spacing_report'];lines=['三维表面间距检查（研发）',f"目标 {r['target_spacing_mm']:g} mm；判断容差 ±{r['tolerance_mm']:g} mm。",
      '状态：'+('采样上下界均在容差内' if r['status']=='within_sampled_bounds' else '未完全满足/无法证实：请检查偏差、缺层或上下界不确定，未切换模式'),
      str(r['counts']),'测得网格距离分布（mm）：'+str(r['distribution']),r['method'],r['sampling'],*r['limitations']]
    return '\n'.join(lines)

def show_spacing_report(parent,plan,locate,is_current):
    import tkinter as tk
    from tkinter import ttk
    w=tk.Toplevel(parent);w.title('表面间距分布与定位');w.geometry('1000x730')
    text=tk.Text(w,height=16,wrap='word');text.pack(fill='x');text.insert('1.0',spacing_text(plan));text.configure(state='disabled')
    frame=ttk.Frame(w);frame.pack(fill='both',expand=True)
    keys=('chain','sample','distance','deviation','status');tree=ttk.Treeview(frame,columns=keys,show='headings')
    for key,label in zip(keys,('链','点','网格上界 mm','上界偏差 mm','上下界判断')):tree.heading(key,text=label)
    scroll=ttk.Scrollbar(frame,command=tree.yview);tree.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');tree.pack(fill='both',expand=True)
    labels={'within_bounds':'采样界内','outside_bounds':'超差','uncertain_bounds':'无法证实','unmeasurable':'无可达前层'}
    rows=plan['spacing_report']['measurements']
    for i,row in enumerate(rows):tree.insert('', 'end',iid=str(i),values=(row['chain_index']+1,row['sample_index']+1,
        '未知' if row['surface_graph_distance_mm'] is None else f"{row['surface_graph_distance_mm']:.3f}",
        '未知' if row['signed_deviation_mm'] is None else f"{row['signed_deviation_mm']:+.3f}",labels[row['status']]))
    def select(_):
        if not is_current():w.destroy();return
        if tree.selection():locate(rows[int(tree.selection()[0])])
    tree.bind('<<TreeviewSelect>>',select);w.measurement_tree=tree
    return w
