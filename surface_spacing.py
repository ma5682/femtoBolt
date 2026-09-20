"""Independent surface-distance mode. Retains mesh contours as separate scan chains."""
import numpy as np
from planning_state import versioned
from metric_region_path import metric_path
from region_exclusions import allowed_region
from region_path_guard import constrain_path
from coverage_inspection import inspect_coverage,check_document_budget
from surface_spacing_report import spacing_report

@versioned
def surface_spacing_plan(projection,mesh,region,spacing_mm=5.,sample_step_mm=1.,width_mm=4.,boundary_margin_mm=1.,tolerance_mm=1.,check=lambda:None,progress=lambda stage:None):
    for value,lo,hi in ((spacing_mm,.1,100),(sample_step_mm,.1,100),(width_mm,.2,100),(boundary_margin_mm,0,50),(tolerance_mm,.01,100)):
        if type(value) not in (int,float) or not np.isfinite(value) or not lo<=value<=hi:raise ValueError('三维表面间距参数越界')
    check();progress('筛选禁区与可信网格面')
    allowed,removed=allowed_region(projection,mesh,region,check)
    ids=np.asarray(allowed['triangle_indices'],int);tri=np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[ids]]
    visible=projection.visible(tri.reshape(-1,3),check=check).reshape(-1,3).all(axis=1)
    visible &= projection.visible(tri.mean(axis=1),check=check)
    trusted=ids[visible].tolist()
    if not trusted:raise ValueError('没有顶点和面心均可信可见的规划面，三维模式未生成；不会回退RGB模式')
    effective=dict(allowed,triangle_indices=trusted,exclusion_polygons=[])
    check();progress('计算表面距离场与等值扫描线')
    plan=metric_path(projection,mesh,effective,spacing_mm,sample_step_mm,check,include_levels=True)
    effective['exclusion_polygons']=region.get('exclusion_polygons',[])
    check();progress('检查整段模拟包络与边界')
    guard=constrain_path(projection,mesh,effective,plan,width_mm,boundary_margin_mm,check)
    if not plan['segments']:raise ValueError('三维路线在模拟包络/边界/可见性约束后全部失效；不会回退RGB模式')
    plan['coverage_parameters']=dict(pattern='surface_spacing_v1',spacing_mm=spacing_mm,sample_step_mm=sample_step_mm,
        width_mm=width_mm,boundary_margin_mm=boundary_margin_mm,tolerance_mm=tolerance_mm)
    plan['boundary_check']=dict(guard,accepted_footprint_checked=True)
    plan['chain_breaks']=[dict(from_chain=i,to_chain=i+1,from_pixel=a[-1]['pixel'],to_pixel=b[0]['pixel'],
        from_xyz_m=a[-1]['xyz_m'],to_xyz_m=b[0]['xyz_m'],reason='independent_surface_contours',
        explanation='独立表面等值扫描链或约束裁剪分段；未规划跨链连接，不允许空间直线跳越',transition_planned=False,transition_time_s=None)
        for i,(a,b) in enumerate(zip(plan['segments'],plan['segments'][1:]))]
    plan['route_order']=dict(mode='surface_distance_levels_then_mesh_piece_order',continuous_chains=len(plan['segments']),disconnected_breaks=len(plan['chain_breaks']),connector_edges=0)
    plan['motion_demo']=dict(speed_mm_s=20.,cross_chain_time_s=None if plan['chain_breaks'] else 0.,gap_policy='stop_require_explicit_continue_no_interpolated_transition',execution_ready=False)
    plan['spacing_semantics']='mesh_edge_distance_field_isoline_increment_not_guaranteed_continuous_geodesic_spacing'
    plan['surface_mode']=dict(schema='femto-surface-mode-v1',algorithm='mesh-surface-spacing-v1',
        seed_rule=plan['seed_rule'],trusted_face_count=len(trusted),untrusted_face_count=int((~visible).sum()),
        trusted_triangle_indices=trusted,forbidden_removed_triangle_indices=removed,
        connection_policy='scan_only; separate_chains; no_silent_air_or_surface_shortcuts',
        fallback_mode=None,physical_accuracy_verified=False)
    if removed:plan['exclusions']=dict(polygons=region['exclusion_polygons'],removed_triangle_indices=removed,allowed_triangle_indices=trusted,
        rule='remove_whole_triangles_touching_dilated_reference_mask')
    check();progress('生成覆盖采样与遗漏报告')
    plan['coverage'],plan['coverage_inspection']=inspect_coverage(projection,mesh,region,effective,plan,check)
    check();progress('测量实际保留路线的表面间距')
    plan['spacing_report']=spacing_report(mesh,effective,plan,spacing_mm,tolerance_mm,check)
    if region.get('screen_polygon'):plan['image_support']=projection.region_support(region['screen_polygon'],check=check)
    check();progress('核验方案大小与版本关联')
    check_document_budget(plan,check)
    return plan
