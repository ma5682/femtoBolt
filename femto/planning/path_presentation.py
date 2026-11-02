"""Fixed-reference route diagrams for product demonstration, not execution."""
from __future__ import annotations
from typing import Any, Callable

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from femto.planning.region_path import path_summary
from femto.planning.route_motion import MotionPlayer, statistics


def render_diagram(rgb: Any, ids: Any, region: Any, plan: Any, show_region: bool=True, show_path: bool=True, segment_index: int | None=None,
                   show_coverage: bool=False, cursor: Any=None, show_turns: bool=True, chain_indices: Any=None) -> Any:
    """`chain_indices` is a display filter (e.g. one mesh block); it never reorders or edits the plan."""
    height,width=rgb.shape[:2];scale=min(8. if plan.get('route_quality') or plan.get('surface_mode') else 1.,1400/width,900/height)
    size=(max(1,int(width*scale)),max(1,int(height*scale)))
    blocks=plan.get('blocks')
    selection_blocks=bool(blocks and blocks.get('selection_blocks'))
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
        if polygon and not selection_blocks:
            polygon_mask=Image.new('1',(width,height));ImageDraw.Draw(polygon_mask).polygon([tuple(p) for p in polygon],fill=1)
            missing=np.asarray(polygon_mask)&(ids<0)
            base[missing]=[180,70,200]
    picture=Image.fromarray(base).resize(size,Image.Resampling.LANCZOS)
    header=(293 if plan.get('route_quality') else 265) if plan.get('start_selection') else 125
    if plan.get('surface_mode'):header=237
    if blocks:header+=28
    selection_lines=[]
    if selection_blocks:
        from femto.planning.trajectory_diagram import draw_selection_layers
        tinted,selection_lines=draw_selection_layers(None,base,region,scale,0,None)
        if selection_lines:
            base=tinted;picture=Image.fromarray(base).resize(size,Image.Resampling.LANCZOS)
        header+=28*(1+len(selection_lines))
    kept_block_labels={};skipped_block_labels=[]
    if blocks:
        # Display only: labels sit at the first displayed chain of each planned block; a label
        # whose box would overlap an earlier one is skipped and listed, never a block dropped.
        from femto.planning.surface_trajectory import non_overlapping_labels
        visible=({segment_index} if segment_index is not None else set(chain_indices) if chain_indices is not None
                 else set(range(len(plan['segments']))))
        boxes=[]
        for item in blocks['items']:
            first=next((i for i in item['chain_indices'] if i in visible and 0<=i<len(plan['segments'])),None)
            if first is None:continue
            p=np.asarray(plan['segments'][first][0]['pixel'])*scale
            boxes.append(((item['label'],first),(p[0]-2,p[1]-22,p[0]+13*len(item['label'])+6,p[1]-4)))
        kept,skipped=non_overlapping_labels(boxes)
        kept_block_labels=dict(kept);skipped_block_labels=[label for label,_ in skipped]
        if skipped_block_labels:header+=28
    if plan.get('algorithm_notice'):header+=28
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
        if plan['surface_mode'].get('algorithm', '').startswith('mesh-slice'):
            lines[0]='三维蛇形扫描 · 块内含换行与绕行连接'
            lines[1]=f"切片平面行距 {plan['spacing_mm']:g} mm | {summary['segment_count']} 条连续链 | 非曲面等距"
            lines[2]='黄线：扫描；青线：换行/绕行连接；绿/红点：每条链起点与终点'
    if plan.get('boundary_check'):
        guard=plan['boundary_check']
        lines[0]+=f" · 模拟宽度 {guard['width_mm']:g} mm / 边界余量 {guard['margin_mm']:g} mm"
    if blocks:
        summary=blocks['summary'];shown=('' if chain_indices is None or segment_index is not None else
            ' | 当前仅显示 '+'、'.join(item['label'] for item in blocks['items'] if item['chain_indices'] and set(item['chain_indices'])<=set(chain_indices)))
        if selection_blocks:
            lines.append(f"工作块（医师选择组件）：{summary['block_count']} 块 | 单链完成 {summary['block_count']-summary['incomplete_block_count']-summary['unplannable_block_count']} | 未完成 {summary['incomplete_block_count']} | 无法规划 {summary['unplannable_block_count']}"
                         f" | 覆盖 {'完整' if summary['coverage_status']=='complete' else '部分（未覆盖碎片 '+str(summary['uncovered_fragment_count'])+'）'} | 块间只停不连线"+shown)
            revisit=summary.get('revisit_connector_length_m') or 0
            lines.append('洋红：重复经过其他已扫描行的连接段（按名义半行距判定，非治疗头宽度）'+(f'，共 {revisit*1000:.1f} mm' if revisit else '：无')+'；连接段不推定治疗能量开启')
            lines.extend(selection_lines)
        else:
            lines.append(f"分块：{summary['block_count']} 块 | 可规划 {summary['plannable_block_count']} | 未规划 {summary['unplannable_block_count']}"
                         f"（{summary['unplannable_face_count']} 面） | {'全部块已规划' if summary['coverage_status']=='complete' else '部分覆盖'} | 块间只停不连线"+shown)
        if skipped_block_labels:
            lines.append(f"{len(skipped_block_labels)} 个块编号标签因重叠未显示（"+'、'.join(skipped_block_labels[:12])
                         +('…' if len(skipped_block_labels)>12 else '')+'）；用“块”下拉逐块查看')
    if plan.get('algorithm_notice'):lines.append(plan['algorithm_notice'])
    for i,line in enumerate(lines):draw.text((12,8+28*i),line,font=font,fill='#e4edf4')
    if selection_blocks:
        from femto.planning.trajectory_diagram import _draw_selection_outlines
        _draw_selection_outlines(draw,region,rgb.shape[:2],scale,header,font)
    polygon=region.get('screen_polygon')
    # Brush polygons can contain doubled bridges between independent components.
    # Selection-block plans use the authoritative mask contours above, not those bridges.
    if polygon and not selection_blocks:
        outline=np.asarray(polygon)*scale+[0,header]
        draw.line([tuple(p) for p in np.vstack([outline,outline[0]])],fill='#ee80ff',width=2)
    for forbidden in region.get('exclusion_polygons',[]):
        outline=np.asarray(forbidden)*scale+[0,header]
        draw.polygon([tuple(p) for p in outline],fill='#702a30',outline='#ff4242',width=3)
        draw.text(tuple(outline.mean(axis=0)), '禁止',font=font,fill='#ffffff')
    if show_path:
        if segment_index is not None:selected=[(segment_index,plan['segments'][segment_index])]
        elif chain_indices is not None:selected=[(i,plan['segments'][i]) for i in sorted(set(chain_indices)) if 0<=i<len(plan['segments'])]
        else:selected=list(enumerate(plan['segments']))
        shown_chains={index for index,_ in selected}
        for index,segment in selected:
            points=np.asarray([sample['pixel'] for sample in segment])*scale+[0,header]
            draw.line([tuple(p) for p in points],fill='#ffd52b',width=2)
            for j in range(1,len(segment)):
                if segment[j].get('motion_kind')=='connector':
                    connector_color=('#48cae4' if plan.get('surface_mode',{}).get('algorithm','').startswith('mesh-slice') else '#ff9138')
                    draw.line([tuple(points[j-1]),tuple(points[j])],fill='#ff4fd8' if segment[j].get('revisit') else connector_color,width=2)
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
        if blocks:
            # Block labels sit at the first displayed chain of each planned block; no line joins blocks.
            for item in blocks['items']:
                first=kept_block_labels.get(item['label'])
                if first is None or first not in shown_chains:continue
                p=np.asarray(plan['segments'][first][0]['pixel'])*scale+[0,header]
                draw.rectangle((p[0]-2,p[1]-22,p[0]+13*len(item['label'])+6,p[1]-4),fill='#20344a')
                draw.text((p[0],p[1]-22),item['label'],font=font,fill='#7fd8ff')
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
            if chain_indices is not None and segment_index is None and (0 if label=='总起点' else len(plan['segments'])-1) not in shown_chains:continue
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


def _block_state_label(item: dict[str, Any], selection_blocks: bool) -> str:
    """Display validated continuity counts without calling every route chain a break."""
    if item['status']=='planned':return '单链完成' if selection_blocks else '已规划'
    if item['status']=='incomplete':
        continuity=item.get('continuity') or {}
        chains=continuity.get('chain_count')
        if chains is None:chains=len(item.get('chain_indices', []))
        breaks=continuity.get('internal_gap_count')
        if breaks is None:breaks=max(chains-1,0)
        return f'未完成：{chains} 段路线 / {breaks} 处断开'
    return '未规划：'+str(item.get('reason'))


def show_presentation(parent: Any, model: Any, region: Any, plan: Any, is_current: Callable[..., Any] = lambda: True) -> Any:
    """Open the Qt route player; kept as a compatibility entry point."""
    from femto.planning.qt_path_presentation import RoutePresentation
    return RoutePresentation(parent, model, region, plan, is_current)
