"""Reproducible, geometry-only interior corrections anchored to a saved route.

The source and base remain embedded and independently validated. This format
never imports a coverage conclusion from either route or a research proof.
"""
from copy import deepcopy
from dataclasses import replace
import numpy as np

from femto.planning import planning_revision as revisions, quality_constrained as quality
from femto.planning.surface_trajectory import SCHEMA, ScanSegment, SurfaceTrajectory

VERSION = 'source-anchored-surface-v1'
POLICY = 'two-interior-source-points-v1'
KEY = 'source_adjustment'
MAX_POINTS = 200_000
ANCHOR_KEYS = {'source_segment', 'source_point', 'base_segment', 'base_point'}


def semantic(value):
    """Numerical route identity, independent of relocated snapshot paths."""
    if isinstance(value, list):return [semantic(v) for v in value]
    if not isinstance(value, dict):return value
    result = {k:semantic(v) for k,v in value.items()}
    if value.get('schema') == SCHEMA:
        result.pop('route_diagram_status', None);result.pop('data_layers', None)
        result['source'] = {k:value['source'][k] for k in ('geometry_sha256','region_id','region_signature')}
    return result


def checked_anchors(anchors):
    if (not isinstance(anchors,list) or len(anchors)!=2
            or any(not isinstance(a,dict) or set(a)!=ANCHOR_KEYS
                   or any(type(v) is not int or v<0 for v in a.values()) for a in anchors)
            or len({(a['base_segment'],a['base_point']) for a in anchors})!=2
            or len({(a['source_segment'],a['source_point']) for a in anchors})!=2):
        raise ValueError('来源修正必须指定两个不同的原路线点和内部目标点')
    return deepcopy(anchors)


def checked_algorithms(algorithms):
    if not isinstance(algorithms,dict) or algorithms.get('route')!=VERSION:return False
    policy=algorithms.get(KEY)
    if (not isinstance(policy,dict) or set(policy)!={'schema','base_sha256','source_sha256','anchors'}
            or policy['schema']!=POLICY):return False
    try:checked_anchors(policy['anchors'])
    except ValueError:return False
    for key in ('base_sha256','source_sha256'):
        v=policy[key]
        if not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v):return False
    base={k:v for k,v in algorithms.items() if k!=KEY};base['route']='quality-constrained-surface-v3'
    normalized=dict(base,target=revisions.LEGACY_ALGORITHMS['target'])
    return (base.get('target') in revisions.TARGET_ALGORITHMS
            and {k:v for k,v in normalized.items() if k!='row_shifts_sha256'}==revisions.RESEARCH_SHIFT_DEEP_RELOCATION_ALGORITHMS
            and revisions.supported_target_algorithms(base))


def _policy(base, source, anchors):
    return dict(schema=POLICY,base_sha256=revisions.digest(semantic(base.to_dict())),
        source_sha256=revisions.digest(semantic(source.to_dict())),anchors=checked_anchors(anchors))


def _context(base, policy):
    from femto.planning.target_paging import rebind_header_records
    inputs=deepcopy(base.planning_context['revision']['inputs'])
    inputs['algorithms'].update(route=VERSION, **{KEY:policy})
    revision=revisions.create(inputs);target=deepcopy(base.planning_context['target'])
    target['computation_sha256']=revision['sha256']
    if 'construction_pages' in target:target['construction_pages']=rebind_header_records(target,revision)
    target['sha256']=revisions.digest({k:v for k,v in target.items() if k!='sha256'})
    return dict(schema=revisions.CONTEXT_SCHEMA,revision=revision,target=target)


def _sources(base, source):
    for item in (base,source):
        if item.planner['algorithm_version']!='quality-constrained-surface-v3' or item.planning_context is None:
            raise ValueError('来源修正只接受绑定原目标的 v3 输入，禁止嵌套修正')
    a,b=base.planning_context['revision'],source.planning_context['revision']
    if revisions.changed(a,b)!=['algorithms']:
        raise ValueError('来源修正的几何、选区、参考、宽度或其他输入不一致')
    normalized=dict(b['inputs']['algorithms'],target=revisions.LEGACY_ALGORITHMS['target'])
    if normalized!=revisions.DEEP_RELOCATION_PATCH_LAYOUT_ALGORITHMS:
        raise ValueError('来源修正需要原局部方向 v12 路线')
    if a['inputs']['algorithms']['target']!=b['inputs']['algorithms']['target']:
        raise ValueError('来源修正不能改变原目标算法')
    if base.origin_reference!=source.origin_reference:
        raise ValueError('来源修正的坐标参考不同')


def _segments(base, source, anchors):
    original=[ScanSegment.from_dict(v) for v in base.gap_report['reconnection']['base_items'] if v['kind']=='scan_segment']
    source_items={s.segment_index:s for s in source.ordered_items if isinstance(s,ScanSegment)}
    if sum(len(s.points) for s in original)>MAX_POINTS:raise ValueError('来源修正超过点数预算')
    segments=deepcopy(original)
    for anchor in checked_anchors(anchors):
        i,j=anchor['base_segment'],anchor['base_point'];s=source_items.get(anchor['source_segment'])
        p=anchor['source_point']
        if not 0<=i<len(segments) or not 0<j<len(segments[i].points)-1 or s is None or not 0<=p<len(s.points):
            raise ValueError('来源修正点索引越界或修改了端点')
        row=segments[i];left,old,right=row.points[j-1:j+2];point=s.points[p]
        if (s.block_index!=row.block_index or point['row_index']!=old['row_index']
                or any(v['row_index']!=old['row_index'] for v in (left,right))
                or any(v.get('role','scan')!='scan' for v in (old,right,point))):
            raise ValueError('来源修正不能跨工作块、扫描行或修改连接点')
        distance=float(np.linalg.norm(np.asarray(point['xyz_m'])-np.asarray(old['xyz_m'])))
        if not np.isfinite(distance) or not 0<distance<=.003:
            raise ValueError('来源修正位移必须在 0 到 3 mm 内')
        replacement=deepcopy(point);replacement.update(row_index=old['row_index'],route_index=old.get('route_index'),role='scan')
        replacement.pop('revisit',None);row.points[j]=replacement
    return segments


def _route(base,source,anchors,check):
    from femto.planning.scan_visit_order import reconstruct
    from femto.planning.selection_blocks import _mark_revisits
    _sources(base,source);policy=_policy(base,source,anchors);context=_context(base,policy)
    if not checked_algorithms(context['revision']['inputs']['algorithms']):raise ValueError('来源修正算法组合不受支持')
    segments=_segments(base,source,anchors);reconnect=base.gap_report['reconnection'];ordering=reconnect['ordering']
    visits=ordering.get('visits');improvement=ordering.get('improvement') or {}
    if improvement.get('adopted') is True:visits=improvement['visits']
    for key in ('preserved_detour','joint_reorder','relocation','relocation_deep'):
        if (ordering.get(key) or {}).get('adopted') is True:visits=ordering[key]['visits']
    chains,_,objective=reconstruct(segments,visits,reconnect['connections'],check,preserve_endpoint_faces=True,budget=MAX_POINTS)
    if objective[0]:raise ValueError('来源修正不接受工作块内断点')
    _mark_revisits([p for _,p in chains],base.planner['parameters']['row_spacing_mm']/1000,check)
    items=quality._items(chains);classes,_=quality.checked_evidence(base.planner['parameters']['quality_evidence'])
    stats,blocks=quality._summaries(items,base.blocks['items'],classes)
    if any(stats[k]>source.statistics[k]+1e-12 for k in
           ('total_length_m','connector_length_m','connected_detours','revisit_connector_length_m')):
        raise ValueError('来源修正的路线长度、绕行或重复连接变差')
    params=dict(deepcopy(base.planner['parameters']),**{KEY:policy})
    report=dict(schema=POLICY,base_trajectory=deepcopy(base.to_dict()),source_trajectory=deepcopy(source.to_dict()),
        anchors=deepcopy(anchors),coverage_status='unverified_new_revision',execution_ready=False,physical_accuracy_verified=False)
    # Slice reports describe the unchanged base layout, not final-point spacing.
    spacing=deepcopy(base.spacing_report);spacing['measurement_scope']='base_layout_before_source_adjustment'
    return replace(base,planner=dict(deepcopy(base.planner),kind='source_anchored_surface',algorithm_version=VERSION,parameters=params),
        ordered_items=items,statistics=stats,blocks=blocks,gap_report={KEY:report},spacing_report=spacing,
        planning_context=context,route_diagram_status=None)


def _check_points(trajectory,mesh,check):
    from femto.geometry.triangle_support import prepare_triangle_faces
    from femto.planning.surface_trajectory import _validate_scan_points,CONNECTED_ALGORITHM
    vertices=faces=prepared=None
    if mesh is not None:vertices,faces,prepared=prepare_triangle_faces(mesh.vertices,mesh.faces)
    expected=0
    for item in trajectory.ordered_items:
        check()
        if not isinstance(item,ScanSegment):continue
        points=[dict({k:v for k,v in p.items() if k!='revisit'},role='scan') for p in item.points]
        shadow=replace(item,points=points,row_index=min(p['row_index'] for p in points),block_index=None)
        expected=_validate_scan_points(shadow,CONNECTED_ALGORITHM,True,False,expected,
            trajectory.planner['parameters']['sample_step_mm']/1000,vertices,faces,_prepared_faces=prepared)


def _geometry(trajectory,mesh,region,check):
    from femto.planning import effective_domain
    from femto.planning.selection_blocks import selection_face_assignment,check_projection_inside_selection
    _check_points(trajectory,mesh,check)
    if mesh is None:return
    params=trajectory.planner['parameters'];record=params['quality_evidence'];width=params['effective_width_mm']
    allowed=effective_domain.validate(params['effective_domain'],record,width,mesh,region,check)
    scan=[s for s in trajectory.ordered_items if isinstance(s,ScanSegment)]
    accepted,_,_=quality.edge_checks(scan,mesh,record,width,check,allowed_faces=allowed)
    if not accepted.all():raise ValueError('来源修正未通过原整边几何检查')
    if region is not None:
        _,_,labels,_=selection_face_assignment(region,np.asarray(mesh.vertices),np.asarray(mesh.faces))
        check_projection_inside_selection(scan,trajectory.blocks['items'],labels,
            np.asarray(region['view_reference']['undistorted_intrinsic']),np.asarray(region['view_reference']['image_size']))


def validate(trajectory,mesh=None,region=None,check=lambda:None):
    check();report=trajectory.gap_report.get(KEY) or {}
    # Preflight before recursive loading: nested adjusted routes are forbidden.
    for key in ('base_trajectory','source_trajectory'):
        if report.get(key,{}).get('planner',{}).get('algorithm_version')!='quality-constrained-surface-v3':
            raise ValueError('来源修正缺少独立原路线或包含嵌套修正')
    base=SurfaceTrajectory.from_dict(report['base_trajectory'],mesh=mesh,region=region,check=check)
    source=SurfaceTrajectory.from_dict(report['source_trajectory'],mesh=mesh,region=region,check=check)
    expected=_route(base,source,report.get('anchors'),check)
    if semantic(expected.to_dict())!=semantic(trajectory.to_dict()):
        raise ValueError('来源修正不能从绑定原路线重现，或混用了旧覆盖状态')
    _geometry(trajectory,mesh,region,check);check()


def build(base,source,anchors,mesh,region,check=lambda:None):
    from femto.planning.surface_trajectory import validate_trajectory
    before=revisions.digest([base.to_dict(),source.to_dict(),anchors,region])
    validate_trajectory(base,mesh=mesh,region=region,check=check)
    validate_trajectory(source,mesh=mesh,region=region,check=check)
    result=_route(base,source,anchors,check);_geometry(result,mesh,region,check)
    check()
    if revisions.digest([base.to_dict(),source.to_dict(),anchors,region])!=before:raise ValueError('来源修正期间输入发生变化')
    return result


def reproduce_sources(trajectory,manifest_path,mesh,region,check=lambda:None):
    """Regenerate both original v3 routes from portable snapshot inputs."""
    from femto.planning.continuous_quality import candidate,replan
    from femto.planning.repair_workflow import generate as repair
    for key in ('base_trajectory','source_trajectory'):
        saved=SurfaceTrajectory.from_dict(trajectory.gap_report[KEY][key],check=check)
        params=deepcopy(saved.planner['parameters']);record=params.pop('quality_evidence')
        width=params.pop('effective_width_mm');domain=params.pop('effective_domain')
        routing=params.pop('continuous_routing');params.pop('preserve_face_crossings')
        layout=params['fixed_width_layout']
        if layout['request']['version']=='fixed-width-local-direction-budgeted-v4':
            params.pop('fixed_width_layout')
            fresh=repair(manifest_path,mesh,region,record,width,domain,layout['request'],check=check,routing=routing,**params)
        else:
            fresh=replan(candidate(manifest_path,check=check,layout_evidence=record,**params),mesh,region,record,width,
                         domain=domain,check=check,routing=routing)
        fresh=replace(fresh,planning_context=saved.planning_context)
        if semantic(saved.to_dict())!=semantic(fresh.to_dict()):raise ValueError('来源修正的原路线无法从输入重算')
