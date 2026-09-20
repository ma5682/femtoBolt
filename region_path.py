"""Research-only fixed-view raster paths on supported mesh samples."""
import csv
import hashlib
import json
import uuid
from pathlib import Path
import numpy as np


def raster_path(projection, mesh, region, row_step_px=10, max_step_mm=5., check=lambda: None):
    """Sample every pixel along scan rows; never join separate rows or holes.

    Row spacing is image pixels, NOT constant surface spacing. Samples are
    ray/mesh intersections, not independent sensor measurements. Adjacent
    samples must lie on the same face or edge-adjacent faces.
    """
    if region.get('exclusion_polygons'):raise ValueError('存在禁止区域，请使用区域覆盖规划。')
    if type(row_step_px) is not int or not 1 <= row_step_px <= 500:
        raise ValueError('Row spacing must be 1–500 image pixels')
    if type(max_step_mm) not in (int, float) or not np.isfinite(max_step_mm) or not 0 < max_step_mm <= 100:
        raise ValueError('Invalid maximum step')
    mask=np.isin(projection.ids,region['triangle_indices']) & projection.valid
    ys,xs=np.nonzero(mask)
    if not len(ys):raise ValueError('Selected region has no supported visible samples')
    faces=np.asarray(mesh.triangles);vertices=np.asarray(mesh.vertices)
    normals=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
    segments=[];total=0
    for ordinal,y in enumerate(range(int(ys.min()),int(ys.max())+1,row_step_px)):
        check();current=[];previous_face=None
        columns=range(int(xs.min()),int(xs.max())+1)
        if ordinal%2:columns=reversed(columns)
        for x in columns:
            if x % 256 == 0:check()
            sample=projection.pixel(x,y) if mask[y,x] else None
            if sample is None:
                if len(current)>1:segments.append(current)
                current=[];previous_face=None;continue
            face=sample['triangle_index'];point=np.asarray(sample['point_world_m'])
            if current and (np.linalg.norm(point-np.asarray(current[-1]['xyz_m']))*1000>max_step_mm+1e-9 or
                    (face!=previous_face and len(set(faces[face]) & set(faces[previous_face]))<2)):
                if len(current)>1:segments.append(current)
                current=[]
            current.append(dict(xyz_m=point.tolist(),normal=normals[face].tolist(),
                                pixel=[x,y],triangle_index=int(face)))
            previous_face=face;total+=1
            if total>200000:raise ValueError('Path exceeds 200000 samples; increase row spacing')
        if len(current)>1:segments.append(current)
    if not segments:raise ValueError('No continuous supported path segment')
    return dict(schema='femto-region-raster-path-v1',coordinate_frame='world_rdf',unit='metre',
        region_id=region['annotation_id'],region_signature=signature(region),
        row_step_px=row_step_px,max_step_mm=max_step_mm,segments=segments,
        spacing_semantics='image_pixels_not_constant_surface_distance',
        connection_policy='separate_segments_no_transition_moves',
        normal_semantics='mesh_winding_normal_not_tool_orientation',
        surface_constraint='discrete_ray_mesh_samples_not_continuous_collision_checked_path',
        execution_ready=False,physical_accuracy_verified=False)


def signature(region):
    return hashlib.sha256(json.dumps(region,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def path_summary(plan):
    lengths = [float(np.linalg.norm(np.diff(np.asarray([p['xyz_m'] for p in segment]), axis=0), axis=1).sum())
               for segment in plan['segments']]
    from route_motion import statistics
    return dict(statistics(plan),segment_count=len(lengths), sample_count=sum(map(len, plan['segments'])),
                total_segment_length_m=sum(lengths),
                length_semantics='sum_of_sample_chords_excludes_transition_moves')


def load_path(path, store, projection, region, check=lambda: None):
    """Reproduce a saved plan against current geometry before accepting it."""
    path = Path(path)
    if path.stat().st_size > 100_000_000:raise ValueError('Path file exceeds 100 MB')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):raise ValueError('Invalid path document')
    from measurement_audit import layer_contract
    if 'data_layers' in payload and payload['data_layers'] != layer_contract(store):
        raise ValueError('Data layer / source model mismatch')
    schema=payload.get('schema')
    if schema not in ('femto-region-raster-path-v1','femto-region-metric-path-v1'):
        raise ValueError('Unknown path schema')
    from planning_state import region_signature, validate
    modern = 'plan_binding' in payload
    if modern:validate(payload, store.mesh, region)
    expected = dict(coordinate_frame='world_rdf', unit='metre',
                    geometry_sha256=store.digest, origin_reference=store.origin_reference,
                    local_context=store.local_context, region_id=region['annotation_id'],
                    region_signature=region_signature(region) if modern else signature(region), execution_ready=False, physical_accuracy_verified=False)
    for key, value in expected.items():
        if key not in payload or payload[key] != value:raise ValueError('Path source/metadata mismatch: '+key)
    if schema=='femto-region-metric-path-v1':
        if 'coverage_parameters' in payload:
            from region_coverage import coverage_plan
            params=payload['coverage_parameters']
            if params.get('pattern')=='surface_spacing_v1':
                from surface_spacing import surface_spacing_plan
                plan=surface_spacing_plan(projection,store.mesh,region,check=check,
                    **{key:params[key] for key in ('spacing_mm','sample_step_mm','width_mm','boundary_margin_mm','tolerance_mm')})
            else:
                if params.get('pattern') in ('serpentine_v1','serpentine_v2','serpentine_v3','serpentine_v4'):
                    from serpentine_path import serpentine_plan as coverage_plan
                    if params.get('pattern')=='serpentine_v3':
                        from coverage_inspection import inspected_serpentine_plan as coverage_plan
                    if params.get('pattern')=='serpentine_v4':
                        from route_optimization import optimized_serpentine_plan as coverage_plan
                elif params.get('pattern') is not None:raise ValueError('Unknown coverage pattern')
                plan=coverage_plan(projection,store.mesh,region,params['width_mm'],params['overlap_percent'],check,
                    boundary_margin_mm=params.get('boundary_margin_mm'),
                    **({'direction':params['direction'],'start_corner':params['start_corner']} if params.get('pattern') in ('serpentine_v2','serpentine_v3','serpentine_v4') else {}))
        else:
            from metric_region_path import metric_path
            plan=metric_path(projection,store.mesh,region,payload.get('spacing_mm'),payload.get('sample_step_mm'),check)
    else:
        plan = raster_path(projection, store.mesh, region, payload.get('row_step_px'),
                           payload.get('max_step_mm'), check=check)
    if payload.get('segments') != plan['segments']:
        raise ValueError('Path samples do not match current supported geometry')
    for key in ('boundary_check','coverage','image_support','exclusions','chain_breaks','start_selection','route_order','motion_demo','coverage_inspection','route_quality','surface_mode','spacing_report'):
        if key in plan and payload.get(key)!=plan[key]:raise ValueError('Path verification report mismatch: '+key)
    if not modern:
        import warnings
        warnings.warn('旧路径无方案版本关联；已按原始来源和参数完整重算验证。请另存新版。', UserWarning)
    return plan


def _export_path(plan, store, parent, region=None, presentation=None,check=lambda:None,progress=lambda stage:None):
    from measurement_audit import layer_contract
    legacy_binding_upgrade=False
    if region is not None:
        from planning_state import validate,binding,region_signature
        if plan.get('plan_binding') is None:
            # v1 raster files predate plan_binding/max_step_mm. Keep their
            # samples, but only after validating their original region link,
            # then label the explicit compatibility upgrade in the export.
            if (plan.get('schema')!='femto-region-raster-path-v1'
                    or plan.get('region_id')!=region.get('annotation_id')
                    or plan.get('region_signature')!=signature(region)):
                raise ValueError('旧路径缺少可验证的区域关联，禁止导出为当前方案')
            plan=dict(plan,max_step_mm=plan.get('max_step_mm',5.),
                      region_signature=region_signature(region),
                      legacy_binding_upgrade='existing_samples_bound_without_replanning')
            plan['plan_binding']=binding(store.mesh,region,plan)
            legacy_binding_upgrade=True
        validate(plan, store.mesh, region)
    progress('写入路径坐标')
    check()
    folder=Path(parent)/('path_'+uuid.uuid4().hex[:12]);folder.mkdir(parents=True,exist_ok=False)
    payload=dict(plan,source_mesh=store.source,geometry_sha256=store.digest,
                 origin_reference=store.origin_reference,local_context=store.local_context,
                 summary=path_summary(plan), data_layers=layer_contract(store),
                 csv_schema='femto-path-csv-v2',csv_semantics='ordered_chain_then_sample; motion_kind_labels_incoming_edge; chain_start_has_no_incoming_motion; cumulative_distance_and_time_exclude_unplanned_cross_chain_moves')
    with (folder/'path.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['segment','index','x_m','y_m','z_m','nx','ny','nz','segment_length_m','motion_kind','chain_start','route_distance_m','known_time_s_at_20mm_s'])
        route_distance=0.
        for segment_id,segment in enumerate(plan['segments']):
            check()
            length=0.
            for i,sample in enumerate(segment):
                if i:length+=float(np.linalg.norm(np.asarray(sample['xyz_m'])-segment[i-1]['xyz_m']))
                writer.writerow([segment_id,i,*sample['xyz_m'],*sample['normal'],length,sample.get('motion_kind','scan'),i==0,route_distance+length,(route_distance+length)*1000/20])
            route_distance+=length
    if region is not None:
        progress("生成展示模型")
        from region_path_glb import export_scene
        payload['visualization']=export_scene(folder,store.mesh,region,plan,store.source)
        payload['region']=region
    if presentation is not None and region is not None:
        progress("生成路线图和覆盖图")
        from path_presentation import render_diagram
        render_diagram(*presentation,region,plan).save(folder/'route_diagram.png')
        payload['route_diagram']='route_diagram.png'
        if 'coverage' in plan:
            if plan.get('coverage_inspection'):
                from coverage_inspector import render_map
                render_map(presentation[0],plan).save(folder/'coverage_diagram.png')
            else:render_diagram(*presentation,region,plan,show_path=False,show_coverage=True).save(folder/'coverage_diagram.png')
            payload['coverage_diagram']='coverage_diagram.png'
    if plan.get('spacing_report'):
        from surface_spacing_report import spacing_text
        (folder/'surface_spacing_report.json').write_text(json.dumps(dict(plan_binding=plan['plan_binding'],report=plan['spacing_report']),ensure_ascii=False,indent=2),encoding='utf-8')
        (folder/'surface_spacing_report.txt').write_text(spacing_text(plan)+'\n方案关联：'+json.dumps(plan['plan_binding'],ensure_ascii=False),encoding='utf-8')
        with (folder/'surface_spacing_samples.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['result_sha256','chain','sample','surface_graph_distance_mm','lower_bound_mm','signed_deviation_mm','status','x_m','y_m','z_m'])
            for row in plan['spacing_report']['measurements']:writer.writerow([plan['plan_binding']['result_sha256'],row['chain_index'],row['sample_index'],row['surface_graph_distance_mm'],row['lower_bound_mm'],row['signed_deviation_mm'],row['status'],*row['xyz_m']])
    if plan.get('route_quality'):
        from route_quality import quality_text
        (folder/'route_quality.txt').write_text(quality_text(plan)+'\n方案关联：'+json.dumps(plan['plan_binding'],ensure_ascii=False),encoding='utf-8')
        with (folder/'sharp_turns.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['result_sha256','chain_index','sample_index','deflection_degrees','x_m','y_m','z_m'])
            for turn in plan['route_quality']['sharp_turns']:writer.writerow([plan['plan_binding']['result_sha256'],turn['chain_index'],turn['sample_index'],turn['deflection_degrees'],*turn['xyz_m']])
        (folder/'route_quality.json').write_text(json.dumps(dict(plan_binding=plan['plan_binding'],report=plan['route_quality']),ensure_ascii=False,indent=2),encoding='utf-8')
    if plan.get('coverage_inspection'):
        check();progress('写入覆盖采样与遗漏定位')
        (folder/'coverage_inspection.json').write_text(json.dumps(dict(plan_binding=plan['plan_binding'],report=plan['coverage_inspection']),ensure_ascii=False,indent=2),encoding='utf-8')
        with (folder/'coverage_omissions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['schema','result_sha256','id','reason','in_denominator','sampled_area_m2','pixel_count','bbox_pixel','focus_xyz_m','evidence'])
            for item in plan['coverage_inspection']['omissions']:
                writer.writerow(['femto-coverage-omissions-v1',plan['plan_binding']['result_sha256'],item['id'],item['reason'],item['in_denominator'],item['sampled_area_m2'],item['pixel_count'],json.dumps(item['bbox_pixel']),json.dumps(item['focus_xyz_m']),item['evidence']])
    (folder/'README.txt').write_text(
        '区域与路线分享包 / Region and route package\n'
        'region_with_path.glb：区域、黄色扫描路径、橙色换行/绕行（如有）、绿色起点、红色终点及方向箭头。各对象可分别隐藏。\n'
        'route_diagram.png（如有）：固定初始照片视角路线图，含参数及方向，可直接分享。\n'
        'coverage_diagram.png：新版覆盖报告绿/橙仅画已覆盖/未覆盖采样点，青为有效几何，红为禁区、蓝为预留、紫为无几何；旧版才按整面采样计数染色。\n'
        'coverage_inspection.json / coverage_omissions.csv（新版）：采样明细、排除范围与遗漏位置，绑定同一结果指纹。\n'
        '覆盖比例为离散面积加权模拟估计，不保证整个曲面覆盖；缺失几何不计入分母。\n'
        '运动播放为本软件演示功能，GLB和PNG为静态展示，不含机械臂运动指令。\n'
        'path.json / path.csv：W0 RDF米制坐标、段号与点序。CSV段号从0开始。\n'
        'PLY仍由区域模型导出入口保存，不混入路径显示几何。\n'
        '蛇形方案链内含已规划连接；旧分段方案及不同链之间无过渡规划；标记尺寸仅为显示，不是工具尺寸。\n'
        '路径为研发预览，非机械臂执行轨迹；蛇形行距为参考深度换算，旧曲面间距为网格距离近似。\n',encoding='utf-8-sig')
    progress('生成报告与文件关联')
    payload['share_archive']='route_package.zip'
    if not legacy_binding_upgrade:
        (folder/'plan_binding.json').write_text(json.dumps(plan.get('plan_binding'),ensure_ascii=False,indent=2),encoding='utf-8')
    if 'coverage' in plan:
        from region_path_guard import coverage_report_text
        payload['coverage_report']='coverage_report.txt'
        (folder/'coverage_report.txt').write_text(coverage_report_text(plan),encoding='utf-8-sig')
    payload['files']={asset.name:hashlib.sha256(asset.read_bytes()).hexdigest() for asset in folder.iterdir() if asset.is_file()}
    (folder/'path.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    progress('压缩完整分享包')
    import zipfile
    with zipfile.ZipFile(folder/'route_package.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for asset in sorted(folder.iterdir()):
            if asset.is_file() and asset.suffix!='.zip':check();archive.write(asset,arcname=asset.name)
    return folder


def export_path(plan,store,parent,region=None,presentation=None,check=lambda:None,progress=lambda stage:None,commit=None):
    from planning_io import publish,durable,windows_file_retry,staging_directory
    # Staging is a sibling of the history folder, never an apparent path_* result.
    parent=Path(parent);parent.parent.mkdir(parents=True,exist_ok=True)
    with staging_directory(parent.parent,prefix='.pending_export_') as temporary:
        folder=_export_path(plan,store,temporary,region,presentation,check,progress)
        durable(folder/'route_package.zip')
        destination=parent/folder.name
        def finish():
            parent.mkdir(parents=True,exist_ok=True)
            windows_file_retry(lambda:folder.rename(destination))
        publish(finish,check,commit)
        return destination


from planning_state import versioned
raster_path = versioned(raster_path)
