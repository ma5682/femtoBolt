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
                if row['block_status']:
                    row['status'] = row.get('status', '') + '；' + row['block_status']
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            pass
    # The product history presents one current result per saved region. Older
    # complete directories remain untouched and accessible on disk.
    latest = {}
    for row in rows:
        parent = row.get('parent_manifest')
        if parent is not None:
            previous = latest.get(parent)
            rank = (row.get('stamp', 0), str(row['source']))
            if previous is None or rank > (previous.get('stamp', 0), str(previous['source'])):
                latest[parent] = row
    return [row for row in rows if row.get('parent_manifest') is None or
            latest[row['parent_manifest']] is row]


def generate(row, control):
    from femto.planning.mesh_slice_planner import plan_mesh_slice
    from femto.planning.trajectory_export import export_trajectory, load_trajectory
    manifest = region_manifest(row)
    old = previous_trajectories(manifest)
    # Block planning: each connected mesh surface is scanned completely before the next one.
    trajectory = plan_mesh_slice(manifest, connect_rows=True, block_planning=True, check=control.check)
    folder = export_trajectory(trajectory, manifest.parent, check=control.check, commit=control.publish)
    load_trajectory(folder, reproduce=False)
    removed, warnings = remove_previous(manifest, old, folder)
    return dict(folder=folder, removed=removed, warnings=warnings, blocks=block_summary(trajectory))


def block_summary(trajectory):
    """Short user-facing block status; None for single-surface (v1/v2) results."""
    stats = trajectory.statistics if isinstance(trajectory.statistics, dict) else {}
    if trajectory.blocks is None or 'block_count' not in stats:
        return None
    missing = [b for b in trajectory.blocks['items'] if b.get('status') != 'planned']
    text = (f"共 {stats['block_count']} 块，可规划 {stats['plannable_block_count']} 块，"
            f"未规划 {stats['unplannable_block_count']} 块（{stats['unplannable_face_count']} 面）；"
            + ('全部块已规划。' if stats['coverage_status'] == 'complete' else
               '部分覆盖：未规划块 ' + '、'.join(f"块 {b['block_index']+1}（{b['face_count']} 面，{b['reason']}）"
                                                   for b in missing[:8]) + ('…' if len(missing) > 8 else '') + '。'))
    return dict(text=text, coverage_status=stats['coverage_status'], block_count=stats['block_count'],
                plannable_block_count=stats['plannable_block_count'],
                unplannable=[dict(block_index=b['block_index'], face_count=b['face_count'], reason=b.get('reason'))
                             for b in missing])


def saved_block_status(payload):
    """Cheap block coverage text from a saved trajectory.json; None for v1/v2 files."""
    stats = payload.get('statistics') if isinstance(payload, dict) else None
    if not isinstance(stats, dict) or 'block_count' not in stats or not isinstance(payload.get('blocks'), dict):
        return None
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


def prepare_playback(row, reconstruction, control):
    from femto.planning.trajectory_export import load_trajectory
    from femto.annotation.fixed_rgb_geometry import FixedRGBModel
    from femto.planning.surface_trajectory import to_player_plan
    trajectory = load_trajectory(row.get('target', row['source']), check=control.check)
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
    return model, manifest['region'], to_player_plan(trajectory)
