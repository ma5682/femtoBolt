"""Derived surface annotations, bound to the exact source mesh (metres/RDF)."""
from __future__ import annotations
from typing import Any

import hashlib
import json
import math
from pathlib import Path
from datetime import datetime
import uuid
import numpy as np
import open3d as o3d
import trimesh
from femto.geometry.fusion import write_point_cloud_safe
import cv2
from femto.io_utils import write_json_strict

STATUS = 'not_physically_validated'


def world_point_m(xyz: Any) -> np.ndarray:
    """Validate stored model/world coordinates; never apply a camera transform."""
    try:
        point = np.asarray(xyz, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid surface coordinates') from exc
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Surface coordinates must be three finite values')
    return point


def distance_to_origin_m(xyz: Any) -> Any:
    """Euclidean distance to the origin of the point's existing coordinate frame."""
    distance = math.hypot(*world_point_m(xyz))
    if not math.isfinite(distance):
        raise ValueError('Non-finite distance to origin')
    return distance


def point_readout_mm(xyz: Any) -> tuple[Any, Any]:
    point = world_point_m(xyz)
    with np.errstate(over='ignore'):
        point_mm = point * 1000.0
    distance_mm = distance_to_origin_m(point) * 1000.0
    if not np.isfinite(point_mm).all() or not math.isfinite(distance_mm):
        raise ValueError('Coordinates exceed the millimetre display range')
    return point_mm.tolist(), distance_mm


def mesh_digest(mesh: Any) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(mesh.vertices, dtype='<f8').tobytes())
    h.update(np.asarray(mesh.triangles, dtype='<i8').tobytes())
    return h.hexdigest()


def distance_m(a: Any, b: Any) -> Any:
    distance = math.dist(world_point_m(a['point_world_m']), world_point_m(b['point_world_m']))
    if not math.isfinite(distance):
        raise ValueError('Non-finite distance between points')
    return distance


def visible_region(mesh: Any, polygon: Any, view: Any, projection: Any, width: int, height: int) -> Any:
    """Select whole triangles whose visible centroid is inside the screen polygon."""
    polygon = np.asarray(polygon, float)
    if len(polygon) < 3:
        raise ValueError('At least three polygon corners are required')
    vertices = np.asarray(mesh.vertices)
    centers = vertices[np.asarray(mesh.triangles)].mean(axis=1)
    matrix = projection @ view
    clip = np.c_[centers, np.ones(len(centers))] @ matrix.T
    valid = (clip[:, 3] > 0) & (np.abs(clip[:, :3]) <= clip[:, 3:4]).all(axis=1)
    ndc = clip[:, :3] / np.where(clip[:, 3:4] == 0, 1, clip[:, 3:4])
    x, y = (ndc[:, 0] + 1) * width / 2, (1 - ndc[:, 1]) * height / 2
    inside = np.zeros(len(centers), bool)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if a[1] != b[1]:
            inside ^= ((a[1] > y) != (b[1] > y)) & (x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
    ids = np.flatnonzero(valid & inside)
    if not len(ids):
        return []
    inv = np.linalg.inv(matrix)
    near = np.c_[ndc[ids, :2], -np.ones(len(ids)), np.ones(len(ids))] @ inv.T
    near = near[:, :3] / near[:, 3:4]
    direction = centers[ids] - near
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    hits = scene.cast_rays(o3d.core.Tensor(np.c_[near, direction].astype(np.float32)))
    return ids[hits['primitive_ids'].numpy() == ids].tolist()


def region_mesh(mesh: Any, ids: Any) -> Any:
    result = o3d.geometry.TriangleMesh(mesh)
    keep = np.zeros(len(mesh.triangles), bool)
    keep[ids] = True
    result.remove_triangles_by_mask(~keep)
    result.remove_unreferenced_vertices()
    return result


def region_boundary(mesh: Any, ids: Any) -> Any:
    """Boundary segments of selected triangles, including holes; never bridge gaps."""
    triangles = np.asarray(mesh.triangles)[ids]
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return np.asarray(mesh.vertices)[unique[counts == 1]].tolist()


def _new_item_metadata():
    now = datetime.now().astimezone().isoformat(timespec='microseconds')
    return dict(annotation_id=uuid.uuid4().hex, created_at=now, updated_at=now)


def _item_metadata(item):
    """Import old records without inventing their original creation time."""
    identifier = item.get('annotation_id', uuid.uuid4().hex)
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError('Invalid annotation ID')
    for key in ('name', 'note'):
        if not isinstance(item.get(key, ''), str):
            raise ValueError('Invalid annotation name / note')
    created, updated = item.get('created_at'), item.get('updated_at', item.get('created_at'))
    for value in (created, updated):
        if value is not None:
            if not isinstance(value, str):
                raise ValueError('Invalid annotation timestamp')
            datetime.fromisoformat(value)
    return dict(annotation_id=identifier, name=item.get('name', ''), note=item.get('note', ''),
                created_at=created, updated_at=updated)


def _selection_records(region: dict[str, Any]) -> dict[str, Any]:
    """Validate physician intent and its geometry-support diagnosis, both kept apart from triangles."""
    from femto.annotation.selection_intent import (
        intent_mask, validate_selection_intent, validate_selection_support)
    if region.get('selection_intent') is None:
        if region.get('selection_support') is not None:
            raise ValueError('Selection support without selection intent')
        return {}
    view = region.get('view_reference')
    size = view.get('image_size') if isinstance(view, dict) else None
    intent = validate_selection_intent(region['selection_intent'], size)
    result = dict(selection_intent=intent)
    if region.get('selection_support') is not None:
        result['selection_support'] = validate_selection_support(region['selection_support'], intent_mask(intent))
    return result


class AnnotationStore:
    def __init__(self, mesh, source):
        self.mesh = mesh
        self.source = str(Path(source).resolve())
        self.digest = mesh_digest(mesh)
        self.points, self.regions = [], []
        self._history = []
        self.local_context = None
        context = {}
        context_path = Path(self.source).parent / 'local_geometry.json'
        if context_path.exists():
            context = json.loads(context_path.read_text(encoding='utf-8-sig'))
            if not isinstance(context, dict):
                raise ValueError('Invalid model coordinate metadata')
            if (context.get('coordinate_frame', 'world_rdf') != 'world_rdf'
                    or context.get('unit', 'metre') != 'metre'):
                raise ValueError('Model coordinate frame / unit mismatch')
            self.local_context = {key: context.get(key) for key in ('session_id', 'reference_frame_id', 'coordinate_frame', 'unit')}
            self.local_context['geometry_source'] = 'raw_depth_supported_tsdf'
            self.local_context['support'] = context.get('support')
            if 'reference_provenance' in context:
                self.local_context['reference_provenance'] = context['reference_provenance']
            if context.get('capture_scope') == 'interrupted_prefix':
                self.local_context['capture_scope'] = 'interrupted_prefix'
                self.local_context['interrupted_prefix'] = context.get('interrupted_prefix')
        self.origin_reference = self.source_origin_reference(self.source, context)
        self.mark_saved()

    @staticmethod
    def source_origin_reference(source: str | Path, context: Any) -> Any:
        """Require matching local-capture evidence, not just an RDF/identity pose.

        Older/standalone models remain usable, but their origin is not asserted
        to be the first camera. All evidence is read-only; model XYZ is untouched.
        """
        unknown = dict(kind='model_coordinate_origin', provenance='unknown')
        session = context.get('session_id')
        reference = context.get('reference_frame_id')
        if (not isinstance(session, str) or not session
                or type(reference) is not int or reference < 0
                or context.get('coordinate_frame') != 'world_rdf'
                or context.get('unit') != 'metre'
                or context.get('transform_direction') != 'T_W0_Ct'):
            return unknown
        try:
            def read(path):
                record = json.loads(Path(path).read_text(encoding='utf-8-sig'))
                if not isinstance(record, dict):
                    raise ValueError('Invalid origin reference evidence')
                return record

            def same_reference(record):
                return (isinstance(record, dict) and record.get('session_id') == session
                        and type(record.get('reference_frame_id')) is int
                        and record['reference_frame_id'] == reference)

            report = read(Path(source).parent / 'reconstruction_report.json')
            if (Path(report['outputs']['mesh']['ply']).resolve() != Path(source).resolve()
                    or not same_reference(report['local_capture'])):
                return unknown
            dataset = Path(report['source_dataset']).resolve()
            manifest = read(dataset / 'manifest.json')
            if not same_reference(manifest['local_capture']):
                return unknown
            tracking = read(context['tracking_report'])
            if (tracking.get('schema') != 'femto-local-tracking-v1'
                    or not same_reference(tracking)
                    or tracking.get('coordinate_frame') != 'world_rdf'
                    or tracking.get('unit') != 'metre'
                    or tracking.get('transform_direction') != 'T_W0_Ct'
                    or tracking.get('pose_source') != 'online_rgbd_research_grade'
                    or tracking.get('accepted_frames', [None])[0] != reference):
                return unknown
            with (dataset / 'frames.jsonl').open(encoding='utf-8-sig') as stream:
                frame = json.loads(next(line for line in stream if line.strip()))
            if not isinstance(frame, dict):
                return unknown
            pose_path = (dataset / frame['pose']).resolve()
            pose_bytes = pose_path.read_bytes()
            pose = json.loads(pose_bytes.decode('utf-8-sig'))
            if not isinstance(pose, dict):
                return unknown
            transform = np.asarray(pose['T_WC'], dtype=float)
            if (frame.get('frame') != reference or frame.get('pose_valid') is not True
                    or pose.get('pose_valid') is not True or not same_reference(pose)
                    or pose.get('pose_source') != 'online_rgbd_research_grade'
                    or transform.shape != (4, 4) or not np.isfinite(transform).all()
                    or not np.allclose(transform, np.eye(4), rtol=0, atol=1e-9)):
                return unknown
            extra = {}
            if 'reference_provenance' in context:
                proof = context['reference_provenance']
                if any(record.get('reference_provenance') != proof
                       for record in (manifest['local_capture'], report['local_capture'], tracking)):
                    return unknown
                from femto.dataset.local_reference import validate_reference_provenance
                validate_reference_provenance(proof, dataset, frame, pose)
                extra = dict(calibration_sha256=proof['calibration']['sha256'],
                             reference_capture_timestamp_us=proof['capture_timestamp_us'])
            return dict(kind='initial_color_camera', provenance='local_capture_reference_pose',
                        session_id=session, reference_frame_id=reference,
                        coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct',
                        source_dataset=str(dataset), reference_pose=str(pose_path),
                        reference_pose_sha256=hashlib.sha256(pose_bytes).hexdigest(), **extra)
        except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration):
            return unknown

    def normalized_point(self, point: Any) -> dict[Any, Any]:
        """Copy a surface point and recompute derived values from its stored XYZ."""
        if not isinstance(point, dict):
            raise ValueError('Invalid surface point')
        xyz = world_point_m(point['point_world_m'])
        tid = point['triangle_index']
        if type(tid) is not int or not 0 <= tid < len(self.mesh.triangles):
            raise ValueError('Invalid surface triangle')
        if (point.get('coordinate_frame', 'world_rdf') != 'world_rdf'
                or point.get('unit', 'metre') != 'metre'
                or point.get('scale_validation', STATUS) != STATUS):
            raise ValueError('Surface point coordinate frame / unit mismatch')
        if point.get('source_mesh') and Path(point['source_mesh']).resolve() != Path(self.source):
            raise ValueError('Surface point source model mismatch')
        # A valid index alone cannot authorize arbitrary coordinates on load.
        a, b, c = np.asarray(self.mesh.vertices)[np.asarray(self.mesh.triangles)[tid]]
        ab, ac, ap = b - a, c - a, xyz - a
        normal = np.cross(ab, ac)
        normal_length = np.linalg.norm(normal)
        if not np.isfinite(normal_length) or normal_length <= 0:
            raise ValueError('Invalid surface triangle')
        tolerance_m = 1e-5  # Numerical ray/PLY roundoff, not a physical accuracy claim.
        if abs(np.dot(ap, normal) / normal_length) > tolerance_m:
            raise ValueError('Point is outside its source surface')
        uv, _, _, _ = np.linalg.lstsq(np.column_stack((ab, ac)), ap, rcond=None)
        closest = a + ab * uv[0] + ac * uv[1]
        barycentric_tolerance = tolerance_m / max(min(np.linalg.norm(ab), np.linalg.norm(ac)), tolerance_m)
        if (min(uv) < -barycentric_tolerance or sum(uv) > 1 + barycentric_tolerance
                or np.linalg.norm(closest - xyz) > tolerance_m):
            raise ValueError('Point is outside its source surface')
        point_readout_mm(xyz)
        return dict(point, **_item_metadata(point), point_world_m=xyz.tolist(),
                    distance_to_origin_m=distance_to_origin_m(xyz),
                    coordinate_frame='world_rdf', unit='metre', scale_validation=STATUS)

    def add_point(self, hit: Any, *, name: str='', note: str='') -> Any:
        point = self.normalized_point(dict(hit, **_new_item_metadata(), name=name, note=note,
                                          source_mesh=self.source))
        focus = ('point', self.points[-1]['annotation_id']) if self.points else None
        self._commit(self.points + [point], self.regions, focus)
        return point

    def normalized_region(self, region: Any) -> dict[Any, Any]:
        if not isinstance(region, dict):
            raise ValueError('Invalid surface region')
        validate_exclusions(region.get('exclusion_polygons',[]))
        ids = region.get('triangle_indices')
        if (not isinstance(ids, list) or not ids
                or any(type(i) is not int or not 0 <= i < len(self.mesh.triangles) for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Invalid triangle indices')
        if (region.get('coordinate_frame', 'world_rdf') != 'world_rdf'
                or region.get('unit', 'metre') != 'metre'
                or region.get('scale_validation', STATUS) != STATUS):
            raise ValueError('Region coordinate frame / unit mismatch')
        if region.get('source_mesh') and Path(region['source_mesh']).resolve() != Path(self.source):
            raise ValueError('Region source model mismatch')
        ids = sorted(ids)
        selection = _selection_records(region)
        corners = np.asarray(self.mesh.vertices)[np.asarray(self.mesh.triangles)[ids]]
        if (not np.isfinite(corners).all()
                or np.any(np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                                  corners[:, 2] - corners[:, 0]), axis=1) <= 0)):
            raise ValueError('Invalid region surface')
        if 'boundary_world_m' in region:
            boundary = np.asarray(region['boundary_world_m'], dtype=float)
            if boundary.size and (boundary.ndim != 3 or boundary.shape[1:] != (2, 3)
                                  or not np.isfinite(boundary).all()):
                raise ValueError('Invalid region boundary coordinates')
        # Saved boundaries are derived. Triangle IDs in the bound mesh are authoritative for
        # geometry; the selection intent mask (when present) is authoritative for what was asked.
        return dict(region, **_item_metadata(region), **selection, triangle_indices=ids,
                    boundary_world_m=region_boundary(self.mesh, ids),
                    source_mesh=self.source, coordinate_frame='world_rdf', unit='metre',
                    scale_validation=STATUS)

    def add_region(self, region: Any, *, replace_existing: bool=False) -> Any:
        region = self.normalized_region(dict(region, **_new_item_metadata()))
        focus = ('region', self.regions[-1]['annotation_id']) if self.regions else None
        self._commit(self.points, [region] if replace_existing else self.regions + [region], focus)
        return region

    def set_exclusions(self,index: int,polygons: Any) -> Any:
        import copy
        current=self.regions[index]
        updated=dict(current,exclusion_polygons=copy.deepcopy(polygons),
            updated_at=datetime.now().astimezone().isoformat(timespec='microseconds'))
        if current.get('selection_intent') is not None:
            # A later exclusion is more unselected intent: subtract it from the authoritative mask.
            from femto.annotation.selection_intent import restrict_selection
            from femto.annotation.square_brush import polygon_mask
            size=current['selection_intent']['image_size']
            updated=restrict_selection(updated,[polygon_mask(polygon,size).astype(bool) for polygon in validate_exclusions(polygons)])
        region=self.normalized_region(updated)
        regions=self.regions[:];regions[index]=region
        self._commit(self.points,regions,('region',region['annotation_id']))
        return region

    def _validated_regions(self, regions):
        return [self.normalized_region(region) for region in regions]

    def _validated_items(self, points, regions):
        if not isinstance(points, list) or not isinstance(regions, list):
            raise ValueError('Invalid annotation collections')
        points = [self.normalized_point(point) for point in points]
        regions = self._validated_regions(regions)
        ids = [item['annotation_id'] for item in points + regions]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate annotation IDs')
        return points, regions

    def _commit(self, points, regions, focus=None):
        # Copy-on-write items keep undo bounded without copying mesh/region data each time.
        self._history.append((self.points, self.regions, focus))
        self._history = self._history[-100:]
        self.points, self.regions = points, regions

    def content_signature(self) -> str:
        """Fingerprint annotation content, not undo depth or a file's existence.

        Derived fields are included because a successful import/save normalizes
        them first. Undoing back to the saved content therefore becomes clean.
        This does not hash or alter the source mesh, RGB-D, or poses.
        """
        payload = json.dumps([self.points, self.regions], ensure_ascii=False,
                             sort_keys=True, allow_nan=False, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def mark_saved(self) -> None:
        self._saved_signature = self.content_signature()

    @property
    def dirty(self) -> Any:
        return self.content_signature() != self._saved_signature

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    def undo(self) -> Any:
        if not self._history:
            return None
        self.points, self.regions, focus = self._history.pop()
        return focus

    def _items(self, kind, index):
        if kind not in ('point', 'region'):
            raise ValueError('Invalid annotation kind')
        items = self.points if kind == 'point' else self.regions
        if type(index) is not int or not 0 <= index < len(items):
            raise ValueError('Invalid annotation selection')
        return items

    def update_annotation(self, kind: str, index: int, *, name: str, note: str) -> Any:
        items = self._items(kind, index)
        item = items[index]
        if not isinstance(name, str) or not isinstance(note, str):
            raise ValueError('Invalid annotation name / note')
        if item.get('name', '') == name and item.get('note', '') == note:
            return item
        updated = dict(item, name=name, note=note,
                       updated_at=datetime.now().astimezone().isoformat(timespec='microseconds'))
        changed = items[:index] + [updated] + items[index + 1:]
        self._commit(changed if kind == 'point' else self.points,
                     changed if kind == 'region' else self.regions, (kind, item['annotation_id']))
        return updated

    def delete_annotation(self, kind: str, index: int) -> None:
        items = self._items(kind, index)
        changed = items[:index] + items[index + 1:]
        self._commit(changed if kind == 'point' else self.points,
                     changed if kind == 'region' else self.regions, (kind, items[index]['annotation_id']))

    def delete_annotations(self, selection: Any) -> None:
        """Validate the entire batch, then record one undo step."""
        chosen=set(selection)
        for kind,index in chosen:self._items(kind,index)
        if not chosen:return
        kind,index=sorted(chosen)[0]
        focus=(kind,self._items(kind,index)[index]['annotation_id'])
        self._commit([p for i,p in enumerate(self.points) if ('point',i) not in chosen],
                     [r for i,r in enumerate(self.regions) if ('region',i) not in chosen],focus)

    def clear_annotations(self, focus: Any=None) -> None:
        if self.points or self.regions:
            self._commit([], [], focus)

    def save(self, parent: str | Path, *, check: Any=None) -> Any:
        from femto.annotation.measurement_audit import layer_contract
        check = check or (lambda: None)
        check()
        points, validated_regions = self._validated_items(self.points, self.regions)
        folder = Path(parent) / (datetime.now().strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8])
        folder.mkdir(parents=True)
        regions = []
        for i, region in enumerate(validated_regions):
            check()
            mesh = region_mesh(self.mesh, region['triangle_indices'])
            mesh_name, cloud_name = f'region_{i+1}_mesh_m.ply', f'region_{i+1}_vertices_m.ply'
            colors = (np.asarray(mesh.vertex_colors)*255).astype(np.uint8) if mesh.has_vertex_colors() else None
            exported = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles), vertex_colors=colors, process=False)
            (folder / mesh_name).write_bytes(exported.export(file_type='ply'))
            from femto.annotation.region_texture import textured_region
            textured,texture_metadata=textured_region(self.mesh,region['triangle_indices'],self.source)
            if textured is not None:exported=textured
            glb_name = f'region_{i+1}_mesh_m.glb'
            # Explicit scene-node conversion: W0 RDF -> glTF right/up/back.
            # Local vertices stay in W0 so the inverse mapping is unambiguous.
            rdf_to_gltf = np.diag([1., -1., -1., 1.])
            scene = trimesh.Scene()
            scene.add_geometry(exported, node_name='region_W0_RDF',
                               transform=rdf_to_gltf)
            (folder / glb_name).write_bytes(scene.export(file_type='glb'))
            boundary_name = f'region_{i+1}_boundary_m.csv'
            import csv
            with (folder / boundary_name).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['edge_id','ax_m','ay_m','az_m','bx_m','by_m','bz_m'])
                for edge_id, edge in enumerate(region['boundary_world_m']):
                    writer.writerow([edge_id, *edge[0], *edge[1]])
            manifest_name = f'region_{i+1}.json'
            assets = dict(mesh_file=mesh_name, glb_file=glb_name,
                          point_cloud_file=cloud_name, boundary_file=boundary_name)
            cloud = o3d.geometry.PointCloud(mesh.vertices)
            if mesh.has_vertex_colors():
                cloud.colors = mesh.vertex_colors
            if not write_point_cloud_safe(folder / cloud_name, cloud):
                raise OSError('Point cloud export failed')
            manifest = dict(schema='femto-region-export-v1', region=region,
                source_mesh=self.source, geometry_sha256=self.digest,
                coordinate_frame='world_rdf', unit='metre',
                origin_reference=self.origin_reference, local_context=self.local_context,
                glb_scene_from_W0=rdf_to_gltf.tolist(),
                W0_from_glb_scene=rdf_to_gltf.tolist(),
                selection='existing_selected_whole_triangles_no_boundary_interpolation',
                appearance=texture_metadata['appearance'],texture_metadata=texture_metadata,
                boundary_semantics='unordered_surface_edges_including_holes_not_a_tool_path',
                physical_accuracy_verified=False, data_layers=layer_contract(self), files=assets,
                file_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                             for name in assets.values()})
            write_json_strict(folder / manifest_name, manifest)
            regions.append(dict(region, **assets, region_manifest_file=manifest_name))
        data = dict(version=1, source_mesh=self.source, geometry_sha256=self.digest,
                    coordinate_frame='world_rdf', unit='metre', scale_validation=STATUS,
                    local_context=self.local_context, created_at=datetime.now().isoformat(),
                    origin_reference=self.origin_reference,
                    points=points, regions=regions,
                    measurements=self._measurements(points), data_layers=layer_contract(self))
        path = folder / 'annotations.json'
        (folder/'导入说明.txt').write_text(
            '外观预览优先打开 GLB：RGB纹理内嵌。PLY仅顶点颜色，部分查看器需开启顶点色。\n'
            '所有坐标数值的长度单位为米。PLY不具备通用强制单位声明，导入时请选择米，不能按毫米解释。\n'
            '例如0.547米应显示为547毫米。GLB节点含RDF到glTF轴转换，原点信息见JSON。\n'
            '局部区域为开放表面，不是封闭实体；请勿为了消除提示直接覆盖原模型封孔。\n'
            '显示平滑法线不改变顶点坐标；几何毛刺、孔洞和原纹理接缝可能仍存在。\n',encoding='utf-8-sig')
        check()
        write_json_strict(path, data)
        self.points, self.regions = points, validated_regions
        self.mark_saved()
        return path

    def load(self, path: str | Path) -> Any:
        from femto.annotation.measurement_audit import layer_contract
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if isinstance(data, dict) and 'data_layers' in data and data['data_layers'] != layer_contract(self):
            raise ValueError('Data layer / source model mismatch')
        if (not isinstance(data, dict) or data.get('version') != 1 or data.get('geometry_sha256') != self.digest
                or Path(data.get('source_mesh', '')).resolve() != Path(self.source)
                or data.get('unit') != 'metre' or data.get('coordinate_frame') != 'world_rdf'
                or data.get('scale_validation') != STATUS):
            raise ValueError('Source model / coordinate frame mismatch')
        loaded_context = data.get('local_context')
        if (loaded_context != self.local_context or
                (self.local_context is not None and
                 type(loaded_context.get('reference_frame_id')) is not type(self.local_context['reference_frame_id']))):
            raise ValueError('Capture session / reference frame mismatch')
        if 'origin_reference' in data and data['origin_reference'] != self.origin_reference:
            raise ValueError('Origin reference provenance mismatch')
        points, regions = self._validated_items(data['points'], data['regions'])
        # Never trust cached distances, including legacy two-point measurements.
        data.update(points=points, regions=regions, measurements=self._measurements(points),
                    origin_reference=self.origin_reference)
        self.points, self.regions = points, regions
        self._history = []
        self.mark_saved()
        return data

    @staticmethod
    def _measurements(points):
        return [dict(point_indices=[i, i + 1], distance_m=distance_m(points[i], points[i + 1]),
                     scale_validation=STATUS) for i in range(len(points) - 1)]


def validate_exclusions(polygons: Any) -> Any:
    if not isinstance(polygons,list) or len(polygons)>32:raise ValueError('最多支持32个禁止区域')
    for polygon in polygons:
        p=np.asarray(polygon,float)
        if p.ndim!=2 or p.shape[1]!=2 or not 3<=len(p)<=512 or not np.isfinite(p).all():
            raise ValueError('禁止区域轮廓无效')
        if np.max(np.abs(p))>100000 or abs(cv2.contourArea(p.astype(np.float32)))<1:
            raise ValueError('禁止区域面积过小或坐标无效')
    return polygons
