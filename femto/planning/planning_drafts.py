"""Recoverable metadata drafts. Never adopt cached paths or substitute source geometry."""
from __future__ import annotations
from typing import Any

from femto.paths import OUTPUTS_ROOT
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time

SCHEMA='femto-planning-draft-v1'
ROOT=OUTPUTS_ROOT/'planning_drafts'


def source_binding(model: Any) -> dict[str, Any]:
    return dict(geometry_sha256=model.store.digest,view_reference=model.view_reference,
                origin_reference=model.store.origin_reference,local_context=model.store.local_context,
                coordinate_frame='world_rdf',unit='metre')


def snapshot(viewer: Any) -> Any:
    return copy.deepcopy(dict(schema=SCHEMA,created_at=time.time(),source=source_binding(viewer.model),
        regions=viewer.model.store.regions,points=viewer.model.store.points,parameters=viewer._requests,
        polygon=viewer.polygon,exclusion_target=viewer.exclusion_target,
        selection_polygons=viewer.selection_polygons(),paint_role=viewer.selection_method.get(),
        focus=viewer.planning_region_id(),mode=viewer.mode.get(),name=viewer.name.get(),note=viewer.note.get(),
        path_policy='not_restored_regenerate_required'))


def read(path: str | Path,model: Any) -> dict[Any, Any]:
    path=Path(path)
    if path.stat().st_size>50*1024*1024:raise ValueError('草稿超过50MB预算')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema')!=SCHEMA or data.get('source')!=source_binding(model):
        raise ValueError('草稿来源/几何/参考图像不匹配；保留原草稿，请恢复正确源工程')
    store=model.store
    for item in data['regions']+data['points']:
        if item.get('view_reference')!=model.view_reference:raise ValueError('草稿标记与参考图像来源不匹配')
    regions=[store.normalized_region(dict(item,source_mesh=store.source)) for item in data['regions']]
    points=[store.normalized_point(dict(item,source_mesh=store.source)) for item in data['points']]
    ids=[r['annotation_id'] for r in regions+points]
    if len(ids)!=len(set(ids)):raise ValueError('草稿标记ID重复')
    from femto.annotation.fixed_rgb_geometry import MAX_ANNOTATIONS,MAX_POLYGON_CORNERS
    import numpy as np
    polygon=np.asarray(data.get('polygon',[]),float)
    if len(regions)+len(points)>MAX_ANNOTATIONS or len(polygon)>MAX_POLYGON_CORNERS:raise ValueError('草稿超出区域/角点预算')
    if polygon.size and (polygon.ndim!=2 or polygon.shape[1]!=2 or not np.isfinite(polygon).all()):raise ValueError('草稿角点无效')
    selections=data.get('selection_polygons')
    if selections is not None:
        if not isinstance(selections,dict) or set(selections)!=set(('work','forbidden')):raise ValueError('草稿区域类型无效')
        for value in selections.values():
            selected=np.asarray(value,float)
            if len(selected)>MAX_POLYGON_CORNERS or (selected.size and (selected.ndim!=2 or selected.shape[1]!=2 or not np.isfinite(selected).all())):
                raise ValueError('草稿区域轮廓无效')
    if not isinstance(data.get('parameters'),dict):raise ValueError('草稿参数无效')
    if not set(data['parameters'])<=set(ids):
        data['parameters']={key:value for key,value in data['parameters'].items() if key in ids}
    for request in data['parameters'].values():
        if not isinstance(request,dict) or request.get('kind') not in ('raster','metric','coverage','serpentine','serpentine_v2','serpentine_v3','serpentine_v4','surface_spacing') or not isinstance(request.get('values'),dict):raise ValueError('草稿规划参数类型无效')
        kind=request['kind'];values=request['values']
        specs=({'row_step_px':(1,500),'max_step_mm':(.01,100)} if kind=='raster' else
               {'spacing_mm':(.1,100),'sample_step_mm':(.1,100)} if kind=='metric' else
               {'width_mm':(.2,100),'overlap_percent':(0,90)})
        if kind=='surface_spacing':specs={'spacing_mm':(.1,100),'sample_step_mm':(.1,100),'width_mm':(.2,100),'tolerance_mm':(.01,100),'boundary_margin_mm':(0,50)}
        if kind=='surface_spacing' and values.get('pattern')!='surface_spacing_v1':raise ValueError('草稿三维表面模式版本无效')
        if 'boundary_margin_mm' in values and values['boundary_margin_mm'] is not None:specs['boundary_margin_mm']=(0,50)
        for key,(minimum,maximum) in specs.items():
            value=values.get(key)
            if type(value) not in (int,float) or not np.isfinite(value) or not minimum<=value<=maximum:raise ValueError('草稿参数越界：'+key)
        if kind=='raster' and type(values['row_step_px']) is not int:raise ValueError('草稿像素间距须为整数')
        if kind in ('serpentine_v2','serpentine_v3','serpentine_v4') and (values.get('pattern')!=kind or values.get('direction') not in ('horizontal','vertical') or values.get('start_corner') not in ('top_left','top_right','bottom_left','bottom_right')):raise ValueError('草稿方向或起点配置无效')
        if kind=='serpentine' and values.get('pattern')!='serpentine_v1':raise ValueError('草稿算法标识无效')
    if data.get('mode') not in ('point','region'):raise ValueError('草稿模式无效')
    if data.get('paint_role',data.get('selection_method','include')) not in ('work','forbidden','include','exclude'):
        raise ValueError('草稿涂抹类型无效')
    if data.get('exclusion_target') is not None and data['exclusion_target'] not in [r['annotation_id'] for r in regions]:raise ValueError('草稿禁区主区域缺失')
    if not isinstance(data.get('name'),str) or not isinstance(data.get('note'),str):raise ValueError('草稿名称备注无效')
    return dict(data,regions=regions,points=points)


class Journal:
    """At most one writer; latest snapshot wins, tombstones prevent resurrection."""
    def __init__(self,source,root=None):
        key=hashlib.sha256(str(Path(source).resolve()).casefold().encode()).hexdigest()
        self.path=(Path(root) if root else ROOT)/(key+'.draft.json')
        self.lock=threading.RLock();self.sequence=0;self.pending=None;self.worker=None
        self.error=None;self.saved_at=None

    def submit(self,data: Any) -> None:
        with self.lock:
            self.sequence+=1;self.pending=(self.sequence,copy.deepcopy(data))
            if self.worker is None or not self.worker.is_alive():
                self.worker=threading.Thread(target=self._run,daemon=True,name='PlanningDraft')
                self.worker.start()

    def _run(self):
        while True:
            with self.lock:
                if self.pending is None:
                    self.worker=None;return
                sequence,data=self.pending;self.pending=None
            temporary=None
            try:
                self.path.parent.mkdir(parents=True,exist_ok=True)
                with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',prefix='.draft_',suffix='.tmp',dir=self.path.parent,delete=False) as f:
                    temporary=Path(f.name);json.dump(data,f,ensure_ascii=False,allow_nan=False)
                    f.flush();os.fsync(f.fileno())
                with self.lock:
                    if sequence==self.sequence:
                        temporary.replace(self.path);self.saved_at=time.time();self.error=None
            except Exception as exc:
                with self.lock:self.error=str(exc)
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)

    def discard(self) -> None:
        with self.lock:
            self.sequence+=1;self.pending=None
            self.path.unlink(missing_ok=True)

    def wait(self) -> None:
        worker=self.worker
        if worker:worker.join(5)
