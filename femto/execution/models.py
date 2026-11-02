"""Deeply immutable simulation snapshots; geometry remains a read-only reference."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

from femto.dataset.provenance import value_hash
from femto.dataset.scene_data_kind import classify_dataset_data_kind, load_dataset_manifest_chain
from femto.hardware.interface_contract import SimulationStep
from femto.io_utils import sha256_file
from femto.operation_diagnostics import record_operation
from femto.planning.saved_source import load_source
from femto.planning.surface_trajectory import ordered_items_sha256


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def gap_summary(report):
    """Keep recorded conclusions/counts for display, without copying proof arrays."""
    if not isinstance(report, dict):
        return report
    result={}
    for key,value in report.items():
        if value is None or type(value) in (str,bool,int,float):
            result[key]=value
        elif isinstance(value,list):
            result[key+'_count']=len(value)
        elif isinstance(value,dict):
            result[key]=gap_summary(value)
    result['display_semantics']='recorded_summary_only_full_evidence_in_frozen_snapshot_and_export'
    return result


def plan_summary(frozen):
    """Bounded, recorded metadata for candidate, live and historical views."""
    saved = frozen.public_plan
    task = saved.get('task_context') or {}
    return dict(plan_source=frozen.source_kind, source_path=saved.get('source_path'),
                revision_id=saved.get('revision_id'), project_id=saved.get('project_id'),
                result_sha256=saved.get('result_sha256'), task_context=task,
                region=saved.get('region'), gap_report=gap_summary(saved.get('gap_report')),
                coverage_status=saved.get('coverage_status', 'unknown'),
                treatment_head=saved.get('treatment_head'), orientation_fields=None,
                software_steps=len(frozen.steps))


@dataclass(frozen=True)
class AssetReference:
    path: str
    sha256: str

    def __post_init__(self):
        if (not isinstance(self.path, str) or not Path(self.path).is_absolute()
                or not isinstance(self.sha256, str) or len(self.sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.sha256)):
            raise ValueError('Asset reference requires an absolute path and SHA-256')

    def signature(self):
        info = Path(self.path).stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def verify(self, check=None):
        before = self.signature()
        if (not Path(self.path).is_file() or sha256_file(self.path, check) != self.sha256
                or self.signature() != before):
            raise ValueError('当前冻结任务引用的资产已改变或缺失：'+self.path)


@dataclass(frozen=True)
class FrozenPlan:
    data_json: str
    steps: tuple[SimulationStep, ...]
    assets: tuple[AssetReference, ...]
    source_kind: str

    def __post_init__(self):
        data = json.loads(self.data_json)
        if (not isinstance(data, dict) or data.get('execution_ready') is not False or data.get('physical_accuracy_verified') is not False
                or self.source_kind not in ('synthetic_test', 'real_femto', 'unknown')):
            raise ValueError('Simulation plan cannot grant real execution or physical accuracy')
        if not self.steps or len(self.steps) > 200000 or any(not isinstance(s, SimulationStep) for s in self.steps):
            raise ValueError('Empty or unbounded simulation step list')
        if (data.get('coordinate_frame') != 'world_rdf' or data.get('unit') != 'metre'
                or data.get('orientation_fields') is not None or data.get('normals_are_not_tool_pose') is not True
                or data.get('plan_source_kind', self.source_kind) != self.source_kind):
            raise ValueError('Simulation snapshot coordinate/source/pose contract mismatch')
        if (not isinstance(data.get('ordered_items'), list)
                or ordered_items_sha256(data['ordered_items']) != data.get('result_sha256')
                or tuple(self.steps) != _steps(data['ordered_items'])):
            raise ValueError('Simulation steps must match the frozen route and contiguous indices')
        assets = tuple(self.assets)
        if (any(not isinstance(a, AssetReference) for a in assets)
                or len({a.path for a in assets}) != len(assets)):
            raise ValueError('Invalid or duplicate frozen asset references')
        object.__setattr__(self, 'data_json', canonical(data))
        object.__setattr__(self, 'steps', tuple(self.steps))
        object.__setattr__(self, 'assets', assets)

    @property
    def public_plan(self):
        return json.loads(self.data_json)

    def to_dict(self):
        return dict(schema='femto-simulation-plan-v1', version=1, run_source='simulation',
                    plan_source=self.source_kind, plan=json.loads(self.data_json),
                    steps=[dict(index=s.index, reference_sha256=s.reference_sha256, role=s.role,
                                reason=s.reason, schema=s.schema) for s in self.steps],
                    assets=[dict(path=a.path, sha256=a.sha256) for a in self.assets],
                    execution_ready=False, physical_accuracy_verified=False)

    def verify_assets(self, check=None):
        for asset in self.assets:
            asset.verify(check)


def _steps(items):
    result = []
    for item in items:
        if item['kind'] == 'gap':
            result.append(SimulationStep(len(result), value_hash(item), 'gap', item['reason']))
        elif item['kind'] == 'scan_segment':
            points = item['points']
            if not isinstance(points, list) or len(points) < 2:
                raise ValueError('Simulation scan segment requires an original edge')
            # Preserve role transitions in each segment, never bridge gaps.
            run_start, previous_role = 0, None
            for index, (first, second) in enumerate(zip(points, points[1:])):
                # Existing surface_trajectory assigns an edge from its ending
                # point and defaults an absent role to scan. Do not invent a
                # stronger format or carry connector forward to the next edge.
                role = second.get('role', 'scan')
                if role not in ('scan', 'connector'):
                    raise ValueError('Trajectory point lacks a known scan/connector role')
                if previous_role is not None and role != previous_role:
                    result.append(SimulationStep(len(result), value_hash(dict(segment=item['segment_index'],
                        point_range=[run_start,index], points=points[run_start:index+1])), previous_role))
                    run_start=index
                previous_role=role
            result.append(SimulationStep(len(result), value_hash(dict(segment=item['segment_index'],
                point_range=[run_start,len(points)-1], points=points[run_start:])), previous_role))
        else:
            raise ValueError('Unknown ordered trajectory item')
    return tuple(result)


def synthetic_plan():
    """Explicitly selected standalone test; never a fallback for failed sources."""
    items = [dict(kind='scan_segment', segment_index=0,
                  points=[dict(xyz_m=[0, 0, .2], role='scan'), dict(xyz_m=[.01, 0, .2], role='scan'),
                          dict(xyz_m=[.01, .01, .2], role='connector')]),
             dict(kind='gap', reason='selection_break'),
             dict(kind='scan_segment', segment_index=1,
                  points=[dict(xyz_m=[.02, .01, .2], role='scan'), dict(xyz_m=[.03, .01, .2], role='scan')])]
    data = dict(source='synthetic_test', revision_id='synthetic-plan-v1', coordinate_frame='world_rdf',
                axes='+X right, +Y down, +Z forward', unit='metre', ordered_items=items,
                result_sha256=ordered_items_sha256(items), algorithm_version='synthetic-step-fixture-v1',
                origin_reference=dict(kind='synthetic_test'), source_model=None, region=None,
                orientation_fields=None, normals_are_not_tool_pose=True, coverage_status='unknown',
                treatment_head=dict(values_status='placeholder_not_device_parameters'),
                used_calibrations=[], execution_ready=False, physical_accuracy_verified=False)
    return FrozenPlan(canonical(data), _steps(items), (), 'synthetic_test')


def load_saved_plan(path, *, cancel=None, progress=lambda text: None):
    """Read the same complete verification chain as Qt, then release its arrays.

    ZIP extraction and diagnostics live in a private temporary workspace. Source
    path/digest and bound external assets, never temporary paths, are frozen.
    """
    def check():
        if cancel is not None and cancel.is_set():
            raise RuntimeError('方案读取已取消')
    check()
    selected = Path(path).resolve()
    if selected.is_dir() and (selected/'planning.region.zip').is_file():
        selected = selected/'planning.region.zip'
    if selected.name == 'trajectory.json' and (selected.parent/'planning.region.zip').is_file():
        selected = selected.parent/'planning.region.zip'
    if not selected.is_file():
        raise ValueError('请选择已保存方案；未保存草稿需先使用现有保存流程')
    before = sha256_file(selected, check)
    with tempfile.TemporaryDirectory(prefix='femto-simulation-plan-') as temporary:
        root = Path(temporary)
        with record_operation('simulation.plan_verify', root=root/'diagnostics'):
            model = load_source(selected, cancel, progress, workspace=root/'opened')
            plan = getattr(model, 'saved_plan', None)
            if not plan or not plan.get('surface_trajectory'):
                raise ValueError('保存方案没有已核验的三维表面路线；不会改用合成场景')
            trajectory = plan['surface_trajectory']
            if (trajectory.get('execution_ready') is not False or trajectory.get('physical_accuracy_verified') is not False
                    or trajectory.get('orientation_fields') is not None or trajectory.get('normals_are_not_tool_pose') is not True):
                raise ValueError('Saved trajectory violates development-only geometry contract')
            data = json.loads(canonical(trajectory))
            data.update(source_path=str(selected), source_file_sha256=before,
                        simulation_step_algorithm='contiguous-role-runs-v1',
                        revision_id=getattr(model, 'revision_id', None), project_id=getattr(model, 'project_id', None),
                        source_model=dict(geometry_sha256=model.store.digest,
                                          lineage=getattr(model, 'source_lineage', None)),
                        region=dict(annotation_id=model.store.regions[0]['annotation_id'],
                                    signature=plan.get('region_signature')),
                        plan_binding=plan.get('plan_binding'), task_context=getattr(model, 'task_context', None),
                        treatment_head=plan.get('treatment_head'),
                        acquisition_completion=getattr(model, 'acquisition_completion', None),
                        used_calibrations=[], source_verified=True)
            references = [AssetReference(str(selected), before)]
            if selected.suffix.lower() != '.zip':
                # Only registered outputs are part of a native frozen plan.
                # Neighbouring drafts are independent edits for future tasks.
                from femto.planning.trajectory_export import FILES_SCHEMA
                inventory_path = selected.parent/'trajectory_files.json'
                inventory = json.loads(inventory_path.read_text(encoding='utf-8-sig'))
                files = inventory.get('files')
                if (inventory.get('schema') != FILES_SCHEMA or inventory.get('result_sha256') != data['result_sha256']
                        or not isinstance(files, dict) or files.get(selected.name) != before):
                    raise ValueError('原生方案登记资产与已核验路线不一致')
                references.append(AssetReference(str(inventory_path), sha256_file(inventory_path, check)))
                for name, digest in files.items():
                    check()
                    target = (selected.parent/name).resolve()
                    if not target.is_relative_to(selected.parent) or not target.is_file():
                        raise ValueError('原生方案登记资产越界或缺失：'+str(name))
                    if target != selected:
                        references.append(AssetReference(str(target), digest))
                candidates = set()
                source = trajectory.get('source', {})
                for name in ('region_manifest', 'source_mesh'):
                    if source.get(name):
                        candidates.add(Path(source[name]).resolve())
                candidates.add(Path(model.store.source).resolve())
                # The bound region manifest also registers its actual mesh and
                # evidence outputs. Keep the original source mesh separately.
                manifest_path = source.get('region_manifest')
                if manifest_path:
                    manifest_path = Path(manifest_path).resolve()
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
                    for name in manifest.get('files', {}).values():
                        target = (manifest_path.parent/name).resolve()
                        if not target.is_relative_to(manifest_path.parent):
                            raise ValueError('区域来源登记资产越界')
                        candidates.add(target)
                references.extend(AssetReference(str(p), sha256_file(p, check)) for p in sorted(candidates)
                                  if not p.is_relative_to(root) and str(p) not in {r.path for r in references})
            capture_source = getattr(model, 'capture_source', None)
            kind = 'unknown'
            if capture_source:
                report_path = Path(capture_source)/'reconstruction_report.json'
                report = json.loads(report_path.read_text(encoding='utf-8-sig'))
                dataset = report.get('source_dataset')
                kind = classify_dataset_data_kind(Path(dataset)) if dataset else 'unknown'
                evidence_paths = {report_path, Path(model.store.source).resolve()}
                source_root = Path(capture_source).parent
                for name in ('acquisition_completion.json','demo_summary.json','task_context.json'):
                    path=source_root/name
                    if path.is_file():evidence_paths.add(path)
                if dataset:
                    chain,complete=load_dataset_manifest_chain(Path(dataset))
                    if complete:
                        for directory,_ in chain:evidence_paths.add(directory/'manifest.json')
                references.extend(AssetReference(str(p),sha256_file(p, check)) for p in sorted(evidence_paths)
                                  if p.is_file() and not p.is_relative_to(root)
                                  and str(p) not in {r.path for r in references})
            # Portable archives can retain a valid offline source with unavailable
            # original acquisition. Its declared label never certifies real input.
            data['plan_source_kind'] = 'synthetic_test' if kind=='synthetic' else kind
            data['capture_source_verified'] = bool(capture_source)
            steps = _steps(data['ordered_items'])
            if sha256_file(selected, check) != before:
                raise ValueError('方案在核验期间被修改，未冻结模拟任务')
            frozen = FrozenPlan(canonical(data), steps, tuple(references), data['plan_source_kind'])
            frozen.verify_assets(check)
            check()
            return frozen
