"""Operations bound to saved region exports, independent of the active selection."""
from pathlib import Path
import json
import re
import shutil


def region_manifest(row):
    if row.get('kind') != '区域':
        raise ValueError('请先选择一条区域导出记录')
    source = Path(row.get('target', row['source'])).resolve()
    data = json.loads(source.read_text(encoding='utf-8-sig'))
    region = data['regions'][row['region_index']]
    name = region.get('region_manifest_file')
    if not name:
        raise ValueError('此旧区域缺少独立清单，请重新导出区域')
    manifest = (source.parent / name).resolve()
    if not manifest.is_relative_to(source.parent) or not manifest.is_file():
        raise ValueError('区域清单丢失或路径越界')
    record = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if record.get('region', {}).get('annotation_id') != region.get('annotation_id'):
        raise ValueError('区域清单身份不一致')
    return manifest


def history_rows(reconstruction, check=lambda: None):
    from femto.planning.export_history import scan_exports
    rows = [r for r in scan_exports(reconstruction, check=check)
            if r['kind'] in ('区域', '三维表面轨迹')]
    for row in rows:
        check()
        try:
            if row['kind'] == '区域':
                row['export_manifest'] = region_manifest(row)
            else:
                payload = json.loads(Path(row.get('target', row['source'])).read_text(encoding='utf-8-sig'))
                row['parent_manifest'] = Path(payload['source']['region_manifest']).resolve()
                row['block_status'] = saved_block_status(payload)
                row['result_complete'] = saved_result_complete(payload)
                if row['block_status']:
                    row['status'] = row.get('status', '') + '；' + row['block_status']
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            pass
    # The product history presents one current result per saved region: the newest complete
    # result. A newer result with unfinished blocks is listed next to it instead of replacing it.
    latest = {}
    latest_partial = {}
    for row in rows:
        parent = row.get('parent_manifest')
        if parent is None:
            continue
        rank = (row.get('stamp', 0), str(row['source']))
        bucket = latest if row.get('result_complete', True) else latest_partial
        previous = bucket.get(parent)
        if previous is None or rank > (previous.get('stamp', 0), str(previous['source'])):
            bucket[parent] = row
    keep = set()
    for parent, row in latest.items():
        keep.add(id(row))
    for parent, row in latest_partial.items():
        complete = latest.get(parent)
        if complete is None or (row.get('stamp', 0), str(row['source'])) > (complete.get('stamp', 0), str(complete['source'])):
            keep.add(id(row))
    return [row for row in rows if row.get('parent_manifest') is None or id(row) in keep]


def generate(row, control):
    from femto.planning.mesh_slice_planner import plan_mesh_slice
    from femto.planning.surface_trajectory import BIDIRECTIONAL_SELECTION_ALGORITHM
    from femto.planning.trajectory_export import export_trajectory, load_trajectory
    manifest = region_manifest(row)
    old = previous_trajectories(manifest)
    # v4.3: work blocks remain the physician's selection components; each planned block is one chain.
    # Exports without selection intent are refused with an explicit message and keep old results.
    trajectory = plan_mesh_slice(manifest, connect_rows=True, block_planning=True, selection_blocks=True,
                                 selection_algorithm=BIDIRECTIONAL_SELECTION_ALGORITHM, check=control.check)
    folder = export_trajectory(trajectory, manifest.parent, check=control.check, commit=control.publish)
    load_trajectory(folder, reproduce=False)
    summary = block_summary(trajectory)
    if not result_complete(trajectory):
        # Unfinished work (broken chain, unplannable block, uncovered fragment or geometry-unconfirmed
        # pixels): the new partial result is published for preview, but the old valid trajectory is
        # neither replaced in the history nor deleted (user decision pending).
        return dict(folder=folder, removed=[], warnings=[], blocks=summary, kept_previous=[str(p) for p in old],
                    cleanup='skipped_incomplete_result')
    removed, warnings = remove_previous(manifest, old, folder)
    return dict(folder=folder, removed=removed, warnings=warnings, blocks=summary, kept_previous=[],
                cleanup='done')


def result_complete(payload_or_trajectory):
    """One rule for "the new result may replace the old one": every block is a single chain AND
    coverage is complete (no unplannable block, no uncovered mesh fragment, no geometry-unconfirmed
    selected pixel). Continuity alone is never enough. v1–v3.1 files have no such record and count
    as complete for the legacy cleanup flow."""
    if isinstance(payload_or_trajectory, dict):
        stats = payload_or_trajectory.get('statistics'); blocks = payload_or_trajectory.get('blocks')
    else:
        stats = payload_or_trajectory.statistics; blocks = payload_or_trajectory.blocks
    if not isinstance(stats, dict) or not isinstance(blocks, dict) or blocks.get('schema') != 'femto-surface-trajectory-blocks-v2':
        return True
    return (all(b.get('status') == 'planned' for b in blocks.get('items', []))
            and stats.get('continuity_status') == 'complete'
            and stats.get('coverage_status') == 'complete'
            and stats.get('geometry_unconfirmed_pixel_count') == 0
            and stats.get('uncovered_fragment_count') == 0)


def block_summary(trajectory):
    """Short user-facing block status; None for single-surface (v1/v2) results."""
    stats = trajectory.statistics if isinstance(trajectory.statistics, dict) else {}
    if trajectory.blocks is None or 'block_count' not in stats:
        return None
    selection = trajectory.blocks.get('schema') == 'femto-surface-trajectory-blocks-v2'
    missing = [b for b in trajectory.blocks['items'] if b.get('status') != 'planned']
    if selection:
        incomplete = [b for b in missing if b.get('status') == 'incomplete']
        unplannable = [b for b in missing if b.get('status') == 'unplannable']
        complete = result_complete(trajectory)
        unconfirmed = stats.get('geometry_unconfirmed_pixel_count')
        text = (f"共 {stats['block_count']} 个工作块（按医师选择组件），单链完成 {stats['planned_block_count']} 块，"
                f"未完成 {len(incomplete)} 块，无法规划 {len(unplannable)} 块；"
                f"覆盖状态 {'完整' if stats['coverage_status'] == 'complete' else '部分'}（未覆盖网格碎片 {stats['uncovered_fragment_count']} 个 / {stats['uncovered_face_count']} 面；"
                f"几何待确认像素 {'未知' if unconfirmed is None else unconfirmed}）；"
                + ('全部工作块已成为连续单链且覆盖完整。' if complete else
                   ('未完成块：' + '、'.join(
                       f"块 {b['block_index']+1}（{b['face_count']} 面，"
                       + (f"{b['continuity']['chain_count']} 段路线 / {b['continuity']['internal_gap_count']} 处断开：{b['link_failure_counts']}"
                          if b['status'] == 'incomplete' else str(b.get('reason')))
                       + '）' for b in missing[:8]) + ('…' if len(missing) > 8 else '') + '；' if missing else '路线连续但覆盖未完整；')
                   + '旧轨迹已保留，未替换或删除。'))
        return dict(text=text, complete=complete, coverage_status=stats['coverage_status'],
                    continuity_status=stats['continuity_status'], block_count=stats['block_count'],
                    plannable_block_count=stats['plannable_block_count'],
                    geometry_unconfirmed_pixel_count=unconfirmed,
                    unplannable=[dict(block_index=b['block_index'], face_count=b['face_count'], reason=b.get('reason'),
                                      status=b['status']) for b in missing])
    text = (f"共 {stats['block_count']} 块，可规划 {stats['plannable_block_count']} 块，"
            f"未规划 {stats['unplannable_block_count']} 块（{stats['unplannable_face_count']} 面）；"
            + ('全部块已规划。' if stats['coverage_status'] == 'complete' else
               '部分覆盖：未规划块 ' + '、'.join(f"块 {b['block_index']+1}（{b['face_count']} 面，{b['reason']}）"
                                                   for b in missing[:8]) + ('…' if len(missing) > 8 else '') + '。'))
    return dict(text=text, complete=True, coverage_status=stats['coverage_status'], block_count=stats['block_count'],
                plannable_block_count=stats['plannable_block_count'],
                unplannable=[dict(block_index=b['block_index'], face_count=b['face_count'], reason=b.get('reason'))
                             for b in missing])


def saved_result_complete(payload):
    """True unless a v4 file reports unfinished work: broken chains, unplannable blocks, uncovered
    fragments or geometry-unconfirmed selected pixels (same rule as `result_complete`)."""
    return result_complete(payload) if isinstance(payload, dict) else True


def saved_block_status(payload):
    """Cheap block coverage text from a saved trajectory.json; None for v1/v2 files."""
    stats = payload.get('statistics') if isinstance(payload, dict) else None
    if not isinstance(stats, dict) or 'block_count' not in stats or not isinstance(payload.get('blocks'), dict):
        return None
    if payload['blocks'].get('schema') == 'femto-surface-trajectory-blocks-v2':
        return (f"{stats['block_count']} 个工作块 / 单链完成 {stats.get('planned_block_count')} / 未完成 {stats.get('incomplete_block_count')} / 无法规划 {stats['unplannable_block_count']}"
                f" / 覆盖 {'完整' if stats.get('coverage_status') == 'complete' else '部分'}"
                + ('；全部完成' if result_complete(payload) else '；未完成结果，旧轨迹保留'))
    return (f"{stats['block_count']} 块 / 可规划 {stats['plannable_block_count']} / "
            + ('全部块已规划' if stats.get('coverage_status') == 'complete'
               else f"部分覆盖（{stats['unplannable_block_count']} 块未规划）"))


def previous_trajectories(manifest):
    """Only direct, concrete result directories bound to this exact region export."""
    manifest = Path(manifest).resolve()
    found = []
    for directory in manifest.parent.iterdir():
        if not re.fullmatch(r'trajectory_[0-9a-f]{12}', directory.name):continue
        if directory.is_symlink() or directory.resolve() != directory.absolute():continue
        path = directory / 'trajectory.json'
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            if (payload.get('schema') == 'femto-surface-trajectory-v1' and
                    Path(payload['source']['region_manifest']).resolve() == manifest):
                found.append(directory)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return found


def remove_previous(manifest, candidates, current):
    """Delete only validated old output folders after new output verification."""
    manifest = Path(manifest).resolve(); current = Path(current).resolve()
    permitted = set(previous_trajectories(manifest))
    allowed = {'trajectory.json', 'trajectory.csv', 'gaps.csv', 'blocks.csv', 'trajectory_files.json',
               'region_with_trajectory.glb', 'route_diagram.png', 'depth_diagram.png',
               'trajectory_report.txt'}
    removed = [];warnings = []
    for candidate in candidates:
        directory = Path(candidate)
        if directory not in permitted or directory.resolve() == current:continue
        try:
            resolved = directory.resolve()
            if resolved.parent != manifest.parent or resolved != directory.absolute():
                raise ValueError('目录越界或重定向')
            entries = list(directory.iterdir())
            if any(p.name not in allowed or not p.is_file() or p.is_symlink() or
                   p.resolve().parent != resolved for p in entries):
                raise ValueError('目录包含非轨迹文件，保留待人工检查')
            shutil.rmtree(resolved)
            removed.append(str(resolved))
        except (OSError, ValueError) as error:
            warnings.append(str(directory)+': '+str(error))
    return removed, warnings


def prepare_report(row, control):
    """Verify an existing trajectory and calculate a current report without regenerating it."""
    from femto.planning.trajectory_export import load_trajectory, product_algorithm_notice
    from femto.planning.trajectory_diagram import report_text
    if row.get('kind') != '三维表面轨迹':
        raise ValueError('请先选择一条三维表面轨迹记录')
    control.check()
    trajectory = load_trajectory(row.get('target') or row['source'], check=control.check,
                                 reproduce=False, allow_older_versions=True)
    control.check()
    text = report_text(trajectory)
    notice = product_algorithm_notice(trajectory.planner.get('parameters', {}), trajectory.planner.get('algorithm_version'))
    control.check()
    return (notice + '\n\n' if notice else '') + text


def prepare_playback(row, reconstruction, control):
    from femto.planning.trajectory_export import load_trajectory, product_algorithm_notice
    from femto.annotation.fixed_rgb_geometry import FixedRGBModel
    from femto.planning.surface_trajectory import to_player_plan
    # Playback is viewing only: an accepted older-version file is hash-checked and mesh-validated
    # but not re-planned; regeneration stays an explicit user action.
    trajectory = load_trajectory(row.get('target', row['source']), check=control.check,
                                 reproduce=False, allow_older_versions=True)
    manifest = json.loads(Path(trajectory.source['region_manifest']).read_text(encoding='utf-8-sig'))
    model = FixedRGBModel(reconstruction, cancel=control)
    if (model.store.digest != trajectory.source['geometry_sha256'] or
            model.store.origin_reference != trajectory.origin_reference):
        raise ValueError('当前模型与历史轨迹来源不一致')
    view = manifest['region']['view_reference']
    for key in ('rgb_sha256', 'depth_sha256', 'undistorted_intrinsic', 'image_size'):
        if model.view_reference.get(key) != view.get(key):
            raise ValueError('历史轨迹参考图像或标定不一致：'+key)
    control.check()
    plan = to_player_plan(trajectory)
    notice = product_algorithm_notice(trajectory.planner.get('parameters', {}), trajectory.planner.get('algorithm_version'))
    if notice:
        plan['algorithm_notice'] = notice
    return model, manifest['region'], plan
