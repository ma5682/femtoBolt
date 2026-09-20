"""Conservative boundary clearance for demo scan footprints; not contact validation."""
import numpy as np
from femto.planning.region_coverage import point_segment_distance


def constrain_path(projection, mesh, region, plan, width_mm, margin_mm, check=lambda:None):
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)[region['triangle_indices']]
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    edges,counts=np.unique(edges,axis=0,return_counts=True)
    if np.any(counts>2):raise ValueError('Non-manifold region boundary')
    boundary=vertices[edges[counts==1]]
    if not len(boundary):raise ValueError('Region has no checkable boundary')
    radius=(width_mm/2+margin_mm)/1000
    # Distance to all boundary segments is conservative on folded surfaces.
    # Midpoint clearance minus half chord bounds clearance for the whole chord.
    pairs=[(left,right) for segment in plan['segments'] for left,right in zip(segment,segment[1:])]
    if len(pairs)*len(boundary)>100_000_000:raise ValueError('Boundary check budget exceeded; reduce region or increase path spacing')
    mids=np.asarray([(np.asarray(a['xyz_m'])+b['xyz_m'])/2 for a,b in pairs]).reshape(-1,3)
    half=np.asarray([np.linalg.norm(np.asarray(a['xyz_m'])-b['xyz_m'])/2 for a,b in pairs])
    clearance=np.full(len(mids),np.inf)
    for start in range(0,len(boundary),128):
        check()
        for a,b in boundary[start:start+128]:clearance=np.minimum(clearance,point_segment_distance(mids,a,b))
    accepted=clearance-half>=radius+1e-9
    visibility_checked=hasattr(projection,'visible')
    if visibility_checked and len(mids):
        accepted &= projection.visible(mids,check=check)
        endpoints=np.asarray([p['xyz_m'] for pair in pairs for p in pair])
        accepted &= projection.visible(endpoints,check=check).reshape(-1,2).all(axis=1)
    polygon=region.get('screen_polygon');polygon_checked=False
    if polygon and len(mids):
        from femto.annotation.fixed_rgb_geometry import polygon_contains
        # Entire 3D ball containing a swept footprint is bounded in image space.
        if not hasattr(projection,'k'):raise ValueError('Camera intrinsics required for polygon footprint constraint')
        uv=projection.project(mids);r=radius+half;z=mids[:,2]
        valid=z>r
        bound=np.full(len(mids),np.inf)
        bound[valid]=max(projection.k[0,0],projection.k[1,1])*r[valid]/(z[valid]-r[valid])*(1+np.linalg.norm(mids[valid,:2],axis=1)/z[valid])
        polygon=np.asarray(polygon,float);distance=np.full(len(mids),np.inf)
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            distance=np.minimum(distance,point_segment_distance(uv,a,b))
        accepted &= valid & polygon_contains(uv,polygon) & (distance>=bound)
        for forbidden in region.get('exclusion_polygons',[]):
            forbidden=np.asarray(forbidden,float);distance=np.full(len(mids),np.inf)
            for a,b in zip(forbidden,np.roll(forbidden,-1,axis=0)):
                distance=np.minimum(distance,point_segment_distance(uv,a,b))
            accepted &= ~polygon_contains(uv,forbidden) & (distance>=bound)
        polygon_checked=True
    output=[];index=0
    for segment in plan['segments']:
        current=[]
        for left,right in zip(segment,segment[1:]):
            if accepted[index]:
                if not current:current=[left]
                current.append(right)
            elif current:output.append(current);current=[]
            index+=1
        if current:output.append(current)
    plan['segments']=output
    return dict(schema='femto-footprint-boundary-v1',width_mm=width_mm,margin_mm=margin_mm,
        candidate_edges=len(pairs),retained_edges=int(accepted.sum()),rejected_edges=int((~accepted).sum()),
        boundary_edge_count=len(boundary),polygon_footprint_checked=polygon_checked,
        visibility_samples_checked=visibility_checked,
        method='euclidean_boundary_lower_bound_for_entire_chord_and_projected_enclosing_ball',
        contact_footprint_verified=False,transitions_planned=False,
        limitations='Conservative simulated radius; not real tool contact geometry. Visibility sampled at endpoints and midpoints, not continuously certified.')


def coverage_report_text(plan):
    from femto.planning.region_path import path_summary
    summary=path_summary(plan);coverage=plan.get('coverage',{});guard=plan.get('boundary_check')
    support=plan.get('image_support',{})
    lines=['区域覆盖规划检查报告（研发演示）',
        f"路径 {summary['segment_count']} 段 / {summary['sample_count']} 点；段内总长 {summary['total_segment_length_m']*1000:.1f} mm。",
        '各段独立；段间动作未规划。此文件不能直接执行机械臂运动。']
    if plan.get('coverage_parameters',{}).get('pattern')=='serpentine_v1':
        lines[2]='水平蛇形往返；链内含换行与绕行连接，不同链之间无过渡规划。非机械臂执行轨迹。'
        lines.append('行距在参考深度换算为像素，不保证曲面等距；跨面连接仅用端点估计覆盖。')
    lines += [f"扫描长度 {summary['scan_length_m']*1000:.3f} mm；连接长度 {summary['connector_length_m']*1000:.3f} mm；连续链 {summary['chain_count']}。",
        f"20 mm/s 恒速模拟：已规划链内扫描和连接 {summary['known_motion_time_s']:.3f} s；不含暂停和加减速。",
        '跨链动作未规划，跨链耗时未知。' if summary['chain_count']>1 else '单链，无跨链移动。']
    if plan.get('start_selection'):
        import json
        lines.append('方向与起点：'+json.dumps(plan['coverage_parameters'],ensure_ascii=False))
        lines.append('起点选择：'+json.dumps(plan['start_selection'],ensure_ascii=False))
        for gap in plan.get('chain_breaks',[]):
            lines.append(f"断点：链 {gap['from_chain']+1} → {gap['to_chain']+1}：{gap['explanation']}；起止 XYZ（米）{gap['from_xyz_m']} → {gap['to_xyz_m']}。")
    if guard:
        lines += [f"模拟扫描宽度 {guard['width_mm']:g} mm；额外边界预留 {guard['margin_mm']:g} mm。",
            f"候选线段 {guard['candidate_edges']}；保留 {guard['retained_edges']}；边界/可见性筛除 {guard['rejected_edges']}。",
            '边界约束针对模拟包络，包含选区外沿及孔洞边界；不是接触力或真实治疗头碰撞验证。']
    else:lines.append('本路径未执行扫描宽度边界约束；建议重新生成区域覆盖规划。')
    if coverage:
        lines += [f"有效表面采样分母 {coverage['target_mesh_area_m2']*1e6:.1f} mm²；采样覆盖估计 {coverage['sampled_coverage_percent']:.2f}%。",
            f"估计未覆盖面积 {max(0.,coverage['target_mesh_area_m2']-coverage['sampled_covered_area_m2'])*1e6:.1f} mm²。",
            '每面3个采样点的面积加权估计；不能保证全表面覆盖。约束收缩后可能留下边缘遗漏，不自动越界补齐。']
    if support:
        lines += [f"圈选图像像素 {support.get('requested_pixels')}；有几何支持 {support.get('geometry_supported_pixels')}；无支持 {support.get('unsupported_pixels')}。",
            '无几何区域不计入曲面覆盖率分母；像素数不能换算成未知曲面面积。']
    else:lines.append('原始圈选图像支持统计不可用；不应把网格覆盖率理解为整个圈选范围的覆盖率。')
    if plan.get('spacing_report'):
        from femto.planning.surface_spacing_report import spacing_text
        lines.append(spacing_text(plan))
    if plan.get('route_quality'):
        from femto.planning.route_quality import quality_text
        lines.append(quality_text(plan))
    inspection=plan.get('coverage_inspection')
    if inspection:
        import json
        lines += ['新版分母：允许、圈内且可信可见的采样代表面积（每面面积/3），含边界预留；排除禁区和无支持采样。并非精确裁剪曲面面积。',
            '无几何支持的三维面积未知，不编造面积，不计入有效表面覆盖率。',
            '方法：'+inspection['method'], '采样密度：'+json.dumps(inspection['sampling'],ensure_ascii=False),
            '报告版本：'+inspection['schema'], *inspection['limitations'],
            '遗漏/排除列表（同类共边面分组；可在报告图层页点击定位）：']
        for item in inspection['omissions']:
            amount=f"采样代表面积 {item['sampled_area_m2']*1e6:.3f} mm²" if item['sampled_area_m2'] is not None else f"{item['pixel_count']}像素，三维面积未知"
            lines.append(f"{item['id']} · {item['label']} · {amount} · 像素{item['bbox_pixel']} · {item['evidence']}")
    else:lines.append('旧版报告没有可定位的采样明细和原因分类；未验证新版模拟包络。请重新生成新版覆盖路线以使用遗漏定位。')
    lines.append('W0 / RDF / 米；几何误差、人体运动、接触形变均未独立验证。')
    if plan.get('plan_binding'):
        import json
        lines.append('方案版本关联：'+json.dumps(plan['plan_binding'],ensure_ascii=False,sort_keys=True))
    if plan.get('exclusions'):
        lines.append(f"禁止区域 {len(plan['exclusions']['polygons'])} 个；剔除相交网格面 {len(plan['exclusions']['removed_triangle_indices'])} 个。覆盖分母仅为允许表面。")
    return '\n'.join(lines)
