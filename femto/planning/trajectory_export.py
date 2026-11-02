"""Atomic JSON/CSV export and reproducible reload for SurfaceTrajectory."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import csv
import json
import os
import uuid

import trimesh
from femto.io_utils import sha256_file
from femto.checksum_batch import sha256_files

from femto.planning.planning_io import publish, staging_directory
from femto.planning.fixed_width_layout import ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION
from femto.planning.surface_trajectory import Gap, ScanSegment, SurfaceTrajectory, canonical_json_bytes, validate_trajectory
from femto.operation_diagnostics import diagnosed
from femto.planning.nominal_revisit_metrics import operation as metric_operation


FILES_SCHEMA = "femto-surface-trajectory-files-v1"


def _sha256(path: Path, check: Callable[[], None] | None = None) -> str:
    return sha256_file(path, check)


def _write_csv(path: Path, rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(rows)


def _write_core(folder: Path, trajectory: SurfaceTrajectory, check: Callable[[], None]) -> None:
    payload = trajectory.to_dict()
    (folder / "trajectory.json").write_bytes(canonical_json_bytes(payload))
    blocks = trajectory.blocks
    selection = bool(blocks) and blocks.get("schema") == "femto-surface-trajectory-blocks-v2"
    from femto.planning.surface_trajectory import ORDERED_SELECTION_ALGORITHM
    ordered_visits = trajectory.planner.get("algorithm_version") in (ORDERED_SELECTION_ALGORITHM, 'quality-constrained-surface-v1', 'quality-constrained-surface-v2', 'quality-constrained-surface-v3', 'source-anchored-surface-v1', 'coverage-refined-surface-v1', 'constructed-surface-route-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1')
    from femto.planning.strict_path import is_strict
    constructed = is_strict(trajectory)
    point_rows = [["route_index", "segment_index", "row_index", "point_index",
                   "x_m", "y_m", "z_m", "triangle_index", "geometry_support"]
                  + (["block_index"] if blocks else []) + (["role", "revisit"] if selection else [])
                  + (["surface_anchor_json"] if constructed else [])]
    gap_rows = [["gap_index", "reason", "inferred_cause", "row_index_from", "row_index_to",
                 "previous_route_index", "next_route_index", "prev_x_m", "prev_y_m", "prev_z_m",
                 "next_x_m", "next_y_m", "next_z_m", "chord_length_m",
                 "scan_dir_x", "scan_dir_y", "scan_dir_z"]
                + (["block_from", "block_to", "row_turn_failure_reason"] if blocks else [])
                + (["link_failure_reason"] if selection else [])]
    previous_block = None
    for item in trajectory.ordered_items:
        check()
        if isinstance(item, ScanSegment):
            for point_index, point in enumerate(item.points):
                point_rows.append([point["route_index"], item.segment_index,
                                   point["row_index"] if ordered_visits else item.row_index, point_index,
                                   *point["xyz_m"], point["triangle_index"], point["geometry_support"]]
                                  + ([item.block_index] if blocks else [])
                                  + ([point.get("role", "scan"), int(point.get("revisit") is True)] if selection else [])
                                  + ([json.dumps(point['surface_anchor'], sort_keys=True)] if constructed else []))
            previous_block = item.block_index
        else:
            gap_rows.append([item.gap_index, item.reason, item.detail["inferred_cause"],
                             item.row_index_from, item.row_index_to,
                             item.previous_point["route_index"], item.next_point["route_index"],
                             *item.previous_point["xyz_m"], *item.next_point["xyz_m"],
                             item.chord_length_m, *item.scan_direction_w0])
            if blocks:
                gap_rows[-1].extend([previous_block, _next_block(trajectory, item.gap_index),
                                     item.detail.get("row_turn_failure_reason", "")])
            if selection:
                gap_rows[-1].append(item.detail.get("link_failure_reason", ""))
    _write_csv(folder / "trajectory.csv", point_rows)
    _write_csv(folder / "gaps.csv", gap_rows)
    if constructed:
        (folder/'surface_support_report.json').write_bytes(canonical_json_bytes(trajectory.gap_report['surface_support']))
    if trajectory.planner['algorithm_version'] in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        (folder/'variable_footprint.json').write_bytes(canonical_json_bytes(trajectory.planner['parameters']['variable_footprint']))
        (folder/'variable_width_report.json').write_bytes(canonical_json_bytes(trajectory.gap_report['variable_width']))
    if blocks:
        # One row per block in visit order; the same authority as trajectory.json.
        block_rows = [["block_index", "component_id", "status", "reason", "face_count", "face_index_min",
                       "face_index_max", "face_indices_sha256", "area_m2", "segment_index_first",
                       "segment_count", "gap_count", "point_count", "row_index_first", "row_count",
                       "scan_length_m", "connector_length_m", "connected_row_turns"]
                      + (["connected_detours", "mesh_fragment_count", "planned_fragment_count", "continuity_status",
                          "coverage_status", "uncovered_face_count", "selected_pixels", "geometry_unconfirmed_pixels",
                          "revisit_connector_samples", "link_failure_counts"] if selection else [])]
        for block in blocks["items"]:
            row = [block.get(key) for key in block_rows[0][:18]]
            if selection:
                selection_counts = block.get("selection") or {}
                support = selection_counts.get("support") or {}
                revisit = block.get("revisit") or {}
                row.extend([block.get("connected_detours"), block.get("mesh_fragment_count"),
                            block.get("planned_fragment_count"), (block.get("continuity") or {}).get("status"),
                            (block.get("coverage") or {}).get("status"), (block.get("coverage") or {}).get("uncovered_face_count"),
                            selection_counts.get("selected_pixels"), support.get("geometry_unconfirmed_pixels"),
                            revisit.get("connector_sample_count"),
                            json.dumps(block.get("link_failure_counts") or {}, sort_keys=True)])
            block_rows.append(row)
        _write_csv(folder / "blocks.csv", block_rows)


def _next_block(trajectory: SurfaceTrajectory, gap_index: int) -> Any:
    segments = [item for item in trajectory.ordered_items if isinstance(item, ScanSegment)]
    return segments[gap_index + 1].block_index if gap_index + 1 < len(segments) else None


def write_files_manifest(folder: str | Path, trajectory: SurfaceTrajectory, *,
                         coverage_metadata: dict[str, Any] | None = None,
                         check: Callable[[], None] | None = None) -> Path:
    folder = Path(folder)
    paths = tuple(path for path in sorted(folder.iterdir(), key=lambda value: value.name)
                  if path.is_file() and path.name != "trajectory_files.json")
    batch = sha256_files(paths, check, hasher=_sha256)
    files = {path.name: batch.hashes[path] for path in paths}
    payload = {"schema": FILES_SCHEMA, "result_sha256": trajectory.to_dict()["result_sha256"],
               "files": files}
    if coverage_metadata is not None:payload['coverage_evidence'] = coverage_metadata
    path = folder / "trajectory_files.json"
    path.write_bytes(canonical_json_bytes(payload))
    return path


@diagnosed('native.save', operation=True)
@metric_operation()
def export_trajectory(trajectory: SurfaceTrajectory, parent: str | Path, *,
                      check: Callable[[], None] = lambda: None,
                      commit: Callable[[Callable[[], Any]], Any] | None = None,
                      augment: Callable[[Path, SurfaceTrajectory, Callable[[], None]], None] | None = None,
                      include_visuals: bool = True, coverage_certificate=None, verified_sink=None,
                      progress: Callable[[str], None] = lambda stage: None) -> Path:
    """Publish a complete result directory; cancellation leaves no partial folder."""
    if trajectory.planner['algorithm_version'] in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
        from femto.planning.variable_footprint import require_current
        require_current(trajectory.planner['parameters']['variable_footprint'])
    if include_visuals:
        from femto.planning.trajectory_diagram import prepare_export_trajectory, write_visual_outputs
        trajectory = prepare_export_trajectory(trajectory)
        if augment is None:
            augment = write_visual_outputs
    validate_trajectory(trajectory)
    if trajectory.planner['algorithm_version'] in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-refined-surface-v1') and coverage_certificate is None:
        raise ValueError('补充路线必须随完整覆盖证据一起导出')
    target = None
    if trajectory.planning_context is not None:
        from femto.planning import target_surface
        context = trajectory.planning_context
        params = trajectory.planner['parameters']
        progress('重建并核验原目标')
        target = target_surface.read(Path(trajectory.source['region_manifest']).parent,
            context['target'], context['revision'], evidence=params['quality_evidence'],
            domain=params['effective_domain'], check=check)
    if coverage_certificate is not None and target is None:raise ValueError('覆盖导出缺少原目标台账')
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / ("trajectory_" + uuid.uuid4().hex[:12])
    with staging_directory(parent, ".trajectory_staging_") as staging:
        check()
        progress('写入三维路线与报告')
        _write_core(staging, trajectory, check)
        if augment is not None:
            augment(staging, trajectory, check)
        if target is not None:
            progress('写入并回读原目标资产')
            target_surface.write(staging, target, check)
            target_surface.read(staging, context['target'], context['revision'], evidence=params['quality_evidence'],
                                domain=params['effective_domain'], check=check, reproduce=False)
        if coverage_certificate is not None:
            from femto.planning import coverage_workflow
            progress('复算并核验模型覆盖证据')
            coverage_workflow.write(staging, coverage_certificate, trajectory, target, check)
            (staging/'model_coverage_report.txt').write_text(
                coverage_workflow.summary_text(coverage_certificate.metadata)+'\n'
                +'原目标面积保留，未定位面积未知；名义模型核验不代表物理作用覆盖。\n', encoding='utf-8')
        check()
        write_files_manifest(staging, trajectory,
            coverage_metadata=None if coverage_certificate is None else coverage_certificate.metadata, check=check)
        check()

        def finish() -> Path:
            if trajectory.planner['algorithm_version'] in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
                from femto.planning.variable_footprint import require_current
                require_current(trajectory.planner['parameters']['variable_footprint'])
            if final.exists():
                raise FileExistsError(final)
            os.replace(staging, final)
            return final

        progress('发布三维路线包')
        result = publish(finish, check, commit)
        if verified_sink is not None and coverage_certificate is not None and trajectory.planner['algorithm_version'] in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
            from femto.planning.refinement_archive import issue_verified_assets
            verified_sink(issue_verified_assets(trajectory, target, coverage_certificate))
        return result


def export_blocked(parent: str | Path, source: str | Path, error: Exception,
                   *, details: Any = None) -> Path:
    """Publish one explicit validation refusal, not a partial trajectory."""
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / ("trajectory_" + uuid.uuid4().hex[:12])
    with staging_directory(parent, ".trajectory_blocked_staging_") as staging:
        payload = {"schema": "femto-surface-trajectory-blocked-v1", "source": str(Path(source).resolve()),
                   "reason": type(error).__name__, "message": str(error), "details": details,
                   "execution_ready": False, "physical_accuracy_verified": False}
        (staging / "trajectory_blocked.json").write_bytes(canonical_json_bytes(payload))
        os.replace(staging, final)
    return final


def _verify_files(folder: Path, result_sha256: str, check=None):
    path = folder / "trajectory_files.json"
    if not path.is_file():
        raise ValueError("Trajectory file integrity manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != FILES_SCHEMA or payload.get("result_sha256") != result_sha256:
        raise ValueError("Trajectory file integrity manifest mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or "trajectory.json" not in files:
        raise ValueError("Trajectory file integrity list is invalid")
    metadata = payload.get('coverage_evidence')
    for name in ('coverage_certificate.json', 'coverage_certificate.npz'):
        if (metadata is not None) != (name in files) or (metadata is not None) != (folder/name).is_file():
            raise ValueError('三维路线覆盖资产与证据清单不完整')
    for name, expected in files.items():
        target = (folder / name).resolve()
        if not target.is_relative_to(folder.resolve()) or not target.is_file() or _sha256(target, check) != expected:
            raise ValueError("Trajectory output file hash mismatch: " + str(name))
    return metadata


def current_algorithm_version(parameters: dict[str, Any]) -> str:
    """Current product version for these saved parameters (the API also retains v4)."""
    from femto.planning.mesh_slice_planner import ALGORITHM_VERSION
    from femto.planning.surface_trajectory import BLOCK_ALGORITHM, CONNECTED_ALGORITHM, ORDERED_SELECTION_ALGORITHM
    if parameters.get('source_adjustment') is not None:return 'source-anchored-surface-v1'
    if parameters.get('variable_footprint') is not None:
        return 'nominal-variable-width-refined-surface-v1' if parameters.get('coverage_refinement',{}).get('schema')=='femto-variable-width-refinement-v1' else 'nominal-variable-width-surface-v1'
    if parameters.get('surface_support') is not None:
        return 'constructed-coverage-refined-surface-v1' if parameters.get('coverage_refinement') is not None else 'constructed-surface-route-v1'
    if parameters.get('coverage_refinement') is not None:return 'coverage-refined-surface-v1'
    if parameters.get('continuous_routing') is not None:
        return 'quality-constrained-surface-v3'
    if parameters.get('effective_domain') is not None:
        return 'quality-constrained-surface-v2'
    if parameters.get('quality_evidence') is not None:
        return 'quality-constrained-surface-v1'
    if parameters.get("selection_blocks") is True:
        return ORDERED_SELECTION_ALGORITHM
    if parameters.get("block_planning") is True:
        return BLOCK_ALGORITHM
    return CONNECTED_ALGORITHM if parameters.get("connect_rows") is True else ALGORITHM_VERSION


def product_algorithm_notice(parameters: dict[str, Any], saved_version: Any) -> str | None:
    """User-facing notice when a saved file is not the current selection-block product rule."""
    from femto.planning.surface_trajectory import SELECTION_ALGORITHMS, ORDERED_SELECTION_ALGORITHM
    current = current_algorithm_version(parameters)
    if saved_version == 'source-anchored-surface-v1':
        return '来源点修正版本：两份原路线与修正点来源可重建，已核对整边几何；新版本覆盖需要独立核验。'
    if saved_version == 'nominal-variable-width-refined-surface-v1':
        return '变宽补充版本：全部边重新绑定段内宽度，原确认集合保留且有新增扫描见证；完整补充历史须独立回读。'
    if saved_version == 'nominal-variable-width-surface-v1':
        return '名义变宽模型版本：扫描和连接按段内上界核验，覆盖按下界计算；设备偏移上界未知，原严格路线保留。'
    if saved_version in ('constructed-surface-route-v1', 'constructed-coverage-refined-surface-v1'):
        return '严格表面路径版本：逐点构造与逐边支持可重放；宽度、质量及模型覆盖分别核验，原常规路线保留。'
    if saved_version == 'coverage-refined-surface-v1':
        return '补充规划版本：原路线保留，新增路线与逐轮覆盖证据须共同回读；未确认范围仍保留。'
    if saved_version in ('quality-constrained-surface-v1', 'quality-constrained-surface-v2', 'quality-constrained-surface-v3'):
        return '已按参考质量与覆盖宽度约束；仍为待复核研发结果，配准不确定性和完整覆盖未验证。'
    if (saved_version in SELECTION_ALGORITHMS and saved_version != ORDERED_SELECTION_ALGORITHM
            and parameters.get("selection_blocks") is True):
        return (f'旧算法版本 {saved_version}（当前 {ORDERED_SELECTION_ALGORITHM}）：'
                '可回放已保存结果，也可按保存的旧版算法重算；要采用当前算法请重新生成，不会自动升级')
    if saved_version != current:
        return f'旧算法版本 {saved_version}（当前 {current}）：仅回放已保存结果，未重算；要采用当前算法请重新生成'
    if saved_version != ORDERED_SELECTION_ALGORITHM:
        return (f'此结果由算法 {saved_version} 生成；当前产品规则为 {ORDERED_SELECTION_ALGORITHM}'
                '（工作块按医师选择组件划分、块内单条连续链）。仅回放，未重算；要采用当前规则请重新选择并导出区域后重新生成')
    return None


@diagnosed('native.read', operation=True)
def load_trajectory(path: str | Path, *, check: Callable[[], None] = lambda: None,
                    reproduce: bool = True, allow_older_versions: bool = False, coverage_sink=None) -> SurfaceTrajectory:
    """Read, hash-check and validate a saved trajectory; by default also re-plan and compare.

    Selection-block v4/v4.1/v4.2/v4.3/v4.4 files reproduce using their saved version, never silently
    upgrading. Other accepted older versions without a retained planner are validated
    and returned for viewing only with allow_older_versions=True; otherwise rejected."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "trajectory.json"
    if not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("Trajectory JSON is missing or exceeds 256 MiB")
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = SurfaceTrajectory.from_dict(payload)
    coverage_metadata = _verify_files(path.parent, payload["result_sha256"], check)
    from femto.planning.refinement_archive import check_inventory
    check_inventory({p.name for p in path.parent.iterdir() if p.is_file()}, trajectory, coverage_metadata)
    if trajectory.planner['algorithm_version'] in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-refined-surface-v1') and coverage_metadata is None:
        raise ValueError('补充路线缺少覆盖核验证据')
    from femto.planning.strict_path import is_strict
    support_path = path.parent/'surface_support_report.json'
    if is_strict(trajectory):
        if (not support_path.is_file() or support_path.stat().st_size > 64*1024*1024
                or json.loads(support_path.read_text(encoding='utf-8')) != trajectory.gap_report['surface_support']):
            raise ValueError('严格构造报告缺失或与路线不一致')
    elif support_path.exists():raise ValueError('常规路线混入了未绑定的严格构造报告')
    for name, expected in (('variable_footprint.json', trajectory.planner['parameters'].get('variable_footprint')),
                           ('variable_width_report.json', trajectory.gap_report.get('variable_width'))):
        asset = path.parent/name
        if trajectory.planner['algorithm_version'] in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
            if not asset.is_file() or asset.stat().st_size > 64*1024*1024 or json.loads(asset.read_text(encoding='utf-8')) != expected:
                raise ValueError('变宽模型资产缺失或与路线不一致：'+name)
        elif asset.exists():raise ValueError('旧路线混入了未绑定的变宽模型资产')
    source = trajectory.source
    manifest_path = Path(source.get("region_manifest", "")).resolve()
    if not manifest_path.is_file() or _sha256(manifest_path, check) != source.get("region_manifest_sha256"):
        raise ValueError("Trajectory region manifest source mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    mesh_path = (manifest_path.parent / manifest["files"]["mesh_file"]).resolve()
    if _sha256(mesh_path, check) != source.get("mesh_file_sha256"):
        raise ValueError("Trajectory region mesh source mismatch")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    # Selection-block files are also checked against the selection intent of their bound export.
    validate_trajectory(trajectory, mesh=mesh, expected_result_sha256=payload["result_sha256"],
                        region=manifest.get("region"), check=check)
    # Optional treatment-head record: hash-listed above; its inputs and coverage must reproduce.
    from femto.planning.treatment_head import load_record
    load_record(path.parent, trajectory, mesh=mesh, check=check)
    domain_path=path.parent/'effective_domain.json'
    if domain_path.is_file():
        if json.loads(domain_path.read_text(encoding='utf-8'))!=trajectory.planner['parameters'].get('effective_domain'):
            raise ValueError('有效区域文件与保存路线不一致')
    target = None
    if trajectory.planning_context is not None:
        from femto.planning import target_surface
        context = trajectory.planning_context
        params = trajectory.planner['parameters']
        target = target_surface.read(path.parent, context['target'], context['revision'], evidence=params['quality_evidence'],
                                     domain=params['effective_domain'], check=check)
    if coverage_metadata is not None:
        if target is None:raise ValueError('三维路线覆盖证据缺少原目标台账')
        from femto.planning import coverage_workflow
        certificate = coverage_workflow.read(path.parent, coverage_metadata, trajectory, target, check)
    if reproduce and trajectory.planner.get('kind') == 'source_anchored_surface':
        from femto.planning.source_anchored_adjustment import reproduce_sources
        reproduce_sources(trajectory,manifest_path,mesh,manifest['region'],check)
    if reproduce and trajectory.planner.get('kind') == 'coverage_refined_surface':
        from femto.planning.refinement_validation import reproduce_base
        reproduce_base(trajectory, manifest_path, mesh, manifest['region'], check)
    if reproduce and trajectory.planner.get('kind') == 'constructed_surface':
        from femto.planning.strict_validation import reproduce_base
        reproduce_base(trajectory, manifest_path, mesh, manifest['region'], check)
    if reproduce and trajectory.planner.get('kind') == 'variable_width_surface':
        from femto.planning.strict_validation import reproduce_base
        base = SurfaceTrajectory.from_dict(trajectory.gap_report['variable_width']['base_trajectory'])
        reproduce_base(base, manifest_path, mesh, manifest['region'], check)
    if reproduce and trajectory.planner.get('kind') == 'variable_width_refined_surface':
        from femto.planning.strict_validation import reproduce_base
        original=SurfaceTrajectory.from_dict(trajectory.gap_report['refinement']['base_trajectory'])
        base=SurfaceTrajectory.from_dict(original.gap_report['variable_width']['base_trajectory'])
        reproduce_base(base,manifest_path,mesh,manifest['region'],check)
    if reproduce and trajectory.planner.get('kind') == 'quality_constrained_surface':
        from femto.planning.quality_constrained import constrain
        from femto.planning.mesh_slice_planner import plan_mesh_slice
        params = dict(trajectory.planner['parameters'])
        quality = params.pop('quality_evidence')
        width = params.pop('effective_width_mm')
        domain = params.pop('effective_domain',None)
        if trajectory.planner['algorithm_version']=='quality-constrained-surface-v3':
            from femto.planning.continuous_quality import candidate as continuous_candidate, replan
            routing=params.pop('continuous_routing');params.pop('preserve_face_crossings')
            saved_layout=params.get('fixed_width_layout')
            if saved_layout is not None and saved_layout['request']['version'] in ('fixed-width-local-direction-repair-v3','fixed-width-local-direction-budgeted-v4','fixed-width-local-direction-foldback-v5','research-fixed-width-local-direction-adaptive-half-v6','research-fixed-width-local-direction-minimal-v7','fixed-width-local-direction-spacing-floor-v9',ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION):
                from femto.planning.repair_workflow import generate as repair_generate
                params.pop('fixed_width_layout')
                regenerated=repair_generate(manifest_path,mesh,manifest['region'],quality,width,domain,
                                             saved_layout['request'],check=check,routing=routing,**params)
            else:
                candidate = continuous_candidate(manifest_path,check=check,layout_evidence=quality,**params)
                regenerated = replan(candidate,mesh,manifest['region'],quality,width,check=check,domain=domain,routing=routing)
        else:
            candidate = plan_mesh_slice(manifest_path, check=check,
                selection_algorithm=trajectory.planner['candidate_algorithm'], **params)
            regenerated = constrain(candidate, mesh, manifest['region'], quality, width, check,domain=domain)
        saved, fresh = trajectory.to_dict(), regenerated.to_dict()
        saved.pop('route_diagram_status', None);fresh.pop('route_diagram_status', None)
        # The retained v3 planner reproduces the route. The independent P1
        # inventory was reproduced from its own complete assets immediately above.
        saved.pop('planning_context', None);fresh.pop('planning_context', None)
        if trajectory.planner['algorithm_version']=='quality-constrained-surface-v3' and saved_layout is not None and saved_layout['request']['version'] in ('fixed-width-local-direction-repair-v3','fixed-width-local-direction-budgeted-v4','fixed-width-local-direction-foldback-v5','research-fixed-width-local-direction-adaptive-half-v6','research-fixed-width-local-direction-minimal-v7','fixed-width-local-direction-spacing-floor-v9',ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION):
            from copy import deepcopy
            saved,fresh=deepcopy(saved),deepcopy(fresh)
            # The embedded initial route has already been validated against
            # the final route's immutable geometry and selection binding.
            # Its original snapshot path/checksum may differ after export.
            for payload in (saved,fresh):
                payload.pop('result_sha256',None)
                initial=payload['gap_report']['row_repair'].get('initial_trajectory')
                if initial is not None:
                    initial.pop('source',None);initial.pop('result_sha256',None)
        if saved != fresh:raise ValueError('质量约束路线不能由已绑定输入重现')
    if reproduce and trajectory.planner.get("kind") == "mesh_slice":
        saved_version = trajectory.planner.get("algorithm_version")
        current = current_algorithm_version(trajectory.planner.get("parameters", {}))
        from femto.planning.surface_trajectory import SELECTION_ALGORITHMS
        retained_selection = saved_version in SELECTION_ALGORITHMS
        if saved_version != current and not retained_selection:
            if allow_older_versions:
                return trajectory
            raise ValueError("Trajectory was generated by an older algorithm version "
                             f"({saved_version} vs {current}); regenerate it")
        check()
        from femto.planning.mesh_slice_planner import plan_mesh_slice
        dispatch = {"selection_algorithm": saved_version} if retained_selection else {}
        regenerated = plan_mesh_slice(manifest_path, check=check, **trajectory.planner["parameters"], **dispatch)
        saved = trajectory.to_dict()
        fresh = regenerated.to_dict()
        for optional in ("route_diagram_status",):
            saved.pop(optional, None)
            fresh.pop(optional, None)
        if saved != fresh:
            raise ValueError("Trajectory does not reproduce from its bound mesh and parameters")
    check()
    if coverage_metadata is not None and coverage_sink is not None:coverage_sink(certificate)
    return trajectory
