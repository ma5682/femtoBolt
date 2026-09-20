"""Portable region/planning workspace. ZIP assets are verified before extraction."""
from femto.paths import OUTPUTS_ROOT
import hashlib
import json
import shutil
import uuid
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from femto.io_utils import write_json, write_json_strict

SCHEMA='femto-region-project-v1'
ASSETS={'scene_tsdf_mesh_m.ply','rgb.png','depth.npy','image_valid.npy',
        'mesh_triangle_uv.npy','texture_multiview.png','project.json'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_project(model,region,path,plan=None,check=lambda:None,*,planning_parameters=None,progress=lambda stage:None,commit=None,expected_digest=None):
    """Keep full source geometry as occlusion context; select only one editable ROI."""
    from femto.annotation.scene_annotations import mesh_digest
    from femto.planning.region_path import signature
    from femto.planning.planning_io import publish,durable,copy_stream,windows_file_retry,staging_directory
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    progress('核验保存目标与方案来源')
    baseline=digest(path) if path.exists() else None
    if expected_digest is not None and baseline!=expected_digest:raise ValueError('方案文件已被其他窗口或工具修改，请另存为或重新打开')
    if path.exists():
        with zipfile.ZipFile(path) as existing:
            previous=json.loads(existing.read('project.json'))
        if previous.get('schema')!=SCHEMA or previous.get('region',{}).get('annotation_id')!=region['annotation_id']:
            raise ValueError('目标文件属于另一个区域，禁止覆盖；请选择新的另存为路径')
    from femto.planning.planning_state import parameters
    if planning_parameters is None and plan is not None:planning_parameters=parameters(plan)
    if plan is not None and planning_parameters!=parameters(plan):raise ValueError('Current parameters differ from path')
    region=model.store.normalized_region(region)
    if region.get('view_reference')!=model.view_reference:raise ValueError('区域与参考图像来源不匹配，未保存方案')
    if plan is not None:
        from femto.planning.planning_state import validate
        validate(plan, model.store.mesh, region)
    with staging_directory(path.parent) as root:
        check()
        # Write current bound geometry, never depend on a relocated source filename.
        progress('复制参考网格与图像资产')
        with open(model.store.source,'rb') as source,(root/'scene_tsdf_mesh_m.ply').open('wb') as target:copy_stream(source,target,check)
        from femto.geometry.fusion import read_geometry_safe
        if mesh_digest(read_geometry_safe(root/'scene_tsdf_mesh_m.ply',mesh=True))!=mesh_digest(model.store.mesh):
            raise ValueError('参考网格文件已改变，未保存方案；请重新加载正确来源')
        check()
        Image.fromarray(model.rgb).save(root/'rgb.png')
        np.save(root/'depth.npy',model.projection.depth,allow_pickle=False)
        np.save(root/'image_valid.npy',getattr(model.projection,'image_valid',np.ones(model.rgb.shape[:2],bool)),allow_pickle=False)
        for name in ('mesh_triangle_uv.npy','texture_multiview.png'):
            source=Path(model.store.source).parent/name
            if source.is_file():shutil.copyfile(source,root/name)
        if (root/'mesh_triangle_uv.npy').exists()!=(root/'texture_multiview.png').exists():raise ValueError('Incomplete texture assets')
        payload=dict(schema=SCHEMA,project_id=getattr(model,'project_id',uuid.uuid4().hex),revision_id=uuid.uuid4().hex,
            coordinate_frame='world_rdf',unit='metre',geometry_sha256=mesh_digest(model.store.mesh),
            origin_reference=model.store.origin_reference,local_context=model.store.local_context,
            source_lineage=getattr(model,'source_lineage',dict(source_mesh=model.store.source,geometry_sha256=model.store.digest)),
            view_reference=model.view_reference,intrinsic=model.projection.k.tolist(),
            depth_min_m=model.projection.minimum,depth_max_m=model.projection.maximum,max_radius_mm=model.max_radius_mm,
            region=region,plan=plan,planning_state=dict(schema='femto-planning-state-v1',parameters=planning_parameters),
            physical_accuracy_verified=False,execution_ready=False,
            scope='single_editable_region_with_full_reference_geometry_and_image_context',
            files={p.name:digest(p) for p in root.iterdir()})
        write_json_strict(root/'project.json', payload)
        progress('写入方案清单与压缩包')
        archive=root/'bundle.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for file in root.iterdir():
                if file.name!='bundle.zip':
                    with file.open('rb') as source,z.open(file.name,'w') as target:copy_stream(source,target,check)
        durable(archive)
        def replace():
            if (digest(path) if path.exists() else None)!=baseline:raise ValueError('保存期间目标方案已改变，未覆盖')
            windows_file_retry(lambda:archive.replace(path))
        publish(replace,check,commit)
    from femto.planning.export_history import register_project
    model.last_save_warning=None
    try:register_project(getattr(model,'reconstruction',Path(model.store.source).parent),path)
    except OSError as error:model.last_save_warning='方案已保存，但历史索引更新失败：'+str(error)
    return path


def _open_project(path,workspace=None,cancel=None,progress=lambda stage:None):
    from femto.annotation.fixed_rgb_geometry import FixedRGBModel,FixedProjection,GeometryBudget,_mesh_budget_before_load,MAX_IMAGE_PIXELS
    from femto.annotation.scene_annotations import AnnotationStore,mesh_digest
    from femto.geometry.fusion import read_geometry_safe
    budget=GeometryBudget(cancel);budget.check()
    progress('检查工程文件和必要资产')
    path=Path(path)
    source_digest=digest(path)
    with zipfile.ZipFile(path) as z:
        entries=z.infolist();names=[i.filename for i in entries]
        if len(names)!=len(set(names)) or not set(names)<=ASSETS or 'project.json' not in names:raise ValueError('Invalid project archive entries')
        if sum(i.file_size for i in entries)>1024**3 or any(i.file_size>512*1024**2 for i in entries):raise ValueError('Project exceeds 1 GiB budget')
        if z.getinfo('project.json').file_size>100*1024**2:raise ValueError('Project metadata too large')
        data=json.loads(z.read('project.json'))
        if data.get('schema')!=SCHEMA or data.get('unit')!='metre' or data.get('coordinate_frame')!='world_rdf':raise ValueError('Unsupported project coordinate contract')
        fields={'files','geometry_sha256','origin_reference','local_context','view_reference','intrinsic','depth_min_m','depth_max_m','max_radius_mm','region','project_id','source_lineage'}
        missing_fields=fields-set(data)
        if missing_fields:raise ValueError('工程缺少来源或必要字段：'+', '.join(sorted(missing_fields)))
        required={'scene_tsdf_mesh_m.ply','rgb.png','depth.npy','image_valid.npy'}
        missing=required-set(names) | (set(data.get('files',{}))-set(names))
        if missing:raise ValueError('工程缺少必要资产：'+', '.join(sorted(missing)))
        if not required<=set(data['files']) or set(data['files'])!=set(names)-{'project.json'}:raise ValueError('工程资产清单不完整或存在未登记文件')
        if ('mesh_triangle_uv.npy' in names)!=('texture_multiview.png' in names):raise ValueError('工程纹理资产不完整：需要UV与纹理成对存在')
        if data.get('region',{}).get('view_reference')!=data.get('view_reference'):raise ValueError('区域与参考图像来源不匹配')
        parent=Path(workspace) if workspace else OUTPUTS_ROOT/'region_workspaces'
        root=parent
        root.mkdir(parents=True,exist_ok=True)
        progress('解包并核对资产指纹')
        for name,expected in data['files'].items():
            budget.check()
            with z.open(name) as source,(root/name).open('wb') as target:
                from femto.planning.planning_io import copy_stream
                copy_stream(source,target,budget.check)
            if digest(root/name)!=expected:raise ValueError('Project asset checksum mismatch: '+name)
        write_json(root/'project.json', data)
    progress('加载参考几何与图像')
    mesh_path=root/'scene_tsdf_mesh_m.ply';_mesh_budget_before_load(mesh_path)
    mesh=read_geometry_safe(mesh_path,mesh=True)
    if mesh_digest(mesh)!=data['geometry_sha256']:raise ValueError('Project geometry fingerprint mismatch')
    with Image.open(root/'rgb.png') as im:
        if im.width*im.height>MAX_IMAGE_PIXELS:raise ValueError('Project image too large')
        rgb=np.asarray(im.convert('RGB')).copy()
    depth=np.load(root/'depth.npy',allow_pickle=False);valid=np.load(root/'image_valid.npy',allow_pickle=False)
    k=np.asarray(data['intrinsic'],float)
    if depth.shape!=rgb.shape[:2] or depth.dtype!=np.uint16 or valid.shape!=depth.shape or valid.dtype!=bool:raise ValueError('Project image/depth shape or type mismatch')
    if model_view:=data.get('view_reference'):
        if 'undistorted_intrinsic' in model_view and model_view['undistorted_intrinsic']!=k.tolist():raise ValueError('参考图像内参与工程内参不一致')
        if 'image_size' in model_view and model_view['image_size']!=[rgb.shape[1],rgb.shape[0]]:raise ValueError('参考图像尺寸与来源不一致')
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:raise ValueError('Invalid project intrinsic')
    model=PortableRegionModel.__new__(PortableRegionModel)
    model.reconstruction=root;model.mesh=mesh;model.rgb=rgb
    model.store=AnnotationStore(mesh,mesh_path)
    model.store.origin_reference=data['origin_reference'];model.store.local_context=data['local_context']
    model.view_reference=data['view_reference'];model.max_radius_mm=float(data['max_radius_mm'])
    model.project_id=data['project_id'];model.source_lineage=data['source_lineage'];model._overlay_cache={}
    progress('重建投影与遮挡支持')
    model.projection=FixedProjection(mesh,k,depth,data['depth_min_m'],data['depth_max_m'],check=budget.check)
    p=model.projection;p.image_valid=valid;p.valid[~valid]=False;p.new_support[~valid]=False;p.ids[~valid]=-1;p.z[~valid]=np.nan;p.reasons[~valid]=4
    region=dict(data['region'],source_mesh=str(mesh_path.resolve()))
    model.store.regions=[model.store.normalized_region(region)];model.store.mark_saved()
    model.prepare_display(cancel=cancel)
    model.saved_plan=None
    model.plan_warning=None
    settings=data.get('planning_state',{})
    if settings and settings.get('schema')!='femto-planning-state-v1':raise ValueError('Unknown planning state version')
    model.planning_parameters=settings.get('parameters')
    if data.get('plan'):
        progress('重算核验路径和报告')
        # Plans store coordinates, never trust imported samples without regeneration.
        from femto.planning.region_path import signature,load_path
        from femto.planning.planning_state import validate
        original=data['plan']
        if 'plan_binding' in original:
            validate(original, mesh, data['region'])
        elif original.get('region_signature') != signature(data['region']):
            raise ValueError('Legacy project region association cannot be verified')
        else:model.plan_warning='旧方案缺少算法版本关联；已按原始来源和参数重算验证，请另存新版。'
        plan=dict(data['plan'],source_mesh=model.store.source,geometry_sha256=model.store.digest,
            local_context=model.store.local_context,origin_reference=model.store.origin_reference,
            region_signature=original['region_signature'] if 'plan_binding' in original else signature(model.store.regions[0]))
        from femto.annotation.measurement_audit import layer_contract
        plan['data_layers']=layer_contract(model.store)
        candidate=root/'candidate_path.json';write_json(candidate, plan, indent=None, ensure_ascii=True)
        model.saved_plan=load_path(candidate,model.store,p,model.store.regions[0],budget.check)
        from femto.planning.planning_state import parameters
        if model.planning_parameters is not None and model.planning_parameters!=parameters(model.saved_plan):
            raise ValueError('Project parameters do not match saved path')
        model.planning_parameters=parameters(model.saved_plan)
    budget.check()
    if digest(path)!=source_digest:raise ValueError('加载期间工程文件已改变，请重新打开')
    model.project_source_path=path.resolve();model.project_file_digest=source_digest
    return model


def open_project(path,workspace=None,cancel=None,progress=lambda stage:None):
    parent=Path(workspace) if workspace else OUTPUTS_ROOT/'region_workspaces'
    parent.mkdir(parents=True,exist_ok=True)
    root=parent/uuid.uuid4().hex
    root.mkdir()
    (root/'.incomplete').touch()
    try:
        model=_open_project(path,root,cancel,progress)
        (root/'.incomplete').unlink()
        return model
    except BaseException:
        # Only remove this call's freshly allocated, checked workspace.
        if root.resolve().parent==parent.resolve():shutil.rmtree(root)
        raise



from femto.annotation.fixed_rgb_geometry import FixedRGBModel

class PortableRegionModel(FixedRGBModel):
    def load(self,path,*,cancel=None):
        # Existing annotations exported inside this workspace retain its source binding.
        old_state=self.store.__dict__.copy();old_cache=self._overlay_cache.copy()
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
            for item in data.get('points',[])+data.get('regions',[]):
                if item.get('view_reference')!=self.view_reference:raise ValueError('Reference RGB evidence mismatch')
            result=self.store.load(path);self.prepare_display(cancel=cancel);return result
        except Exception:
            self.store.__dict__.clear();self.store.__dict__.update(old_state);self._overlay_cache=old_cache
            raise


def load_source(path,cancel=None,progress=lambda stage:None):
    progress('加载参考来源')
    return open_project(path,cancel=cancel,progress=progress) if Path(path).is_file() else FixedRGBModel(path,cancel=cancel)


def project_from_annotations(source,index,cancel=None,progress=lambda stage:None):
    source=Path(source).resolve()
    if source.name!='annotations.json' or source.parent.parent.name!='annotations':
        raise ValueError('请选择当前模型的历史区域标记记录。')
    model=FixedRGBModel(source.parent.parent.parent,cancel=cancel)
    model.load(source,cancel=cancel)
    if type(index) is not int or not 0<=index<len(model.store.regions):raise ValueError('区域记录不存在')
    destination=model.reconstruction/'region_projects'/('region_'+uuid.uuid4().hex+'.region.zip')
    from femto.annotation.fixed_rgb_geometry import GeometryBudget
    return save_project(model,model.store.regions[index],destination,check=GeometryBudget(cancel).check,progress=progress,
        commit=getattr(cancel,"publish",None))


if __name__=='__main__':
    import tkinter as tk
    from tkinter import filedialog,messagebox
    import argparse
    from femto.planning.fixed_rgb_viewer import FixedRGBViewer
    root=tk.Tk();root.withdraw()
    parser=argparse.ArgumentParser();parser.add_argument('--annotations',type=Path);parser.add_argument('--region-index',type=int,default=0)
    args=parser.parse_args()
    if args.annotations:
        try:path=project_from_annotations(args.annotations,args.region_index)
        except Exception as error:
            messagebox.showerror('无法进入规划',str(error)+'\n旧记录需要原参考图片和模型存在；可改为导入独立区域工程。',parent=root)
            root.destroy();raise SystemExit(1)
    else:path=filedialog.askopenfilename(title='打开区域工程 / 规划方案',filetypes=[('Region project','*.zip')])
    if path:root.deiconify();FixedRGBViewer(root,Path(path));root.mainloop()
    else:root.destroy()
