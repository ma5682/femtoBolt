"""Independent, validated reconstruction sources for the two GS workflows."""
from __future__ import annotations
from typing import Any

import json
from pathlib import Path

MARKERS = {'keyframe': 'latest_keyframe_reconstruction.txt',
           'moving': 'latest_moving_reconstruction.txt'}


def mode_of(root: str | Path) -> str | None:
    root = Path(root).resolve()
    try:
        data = json.loads((root / 'demo_summary.json').read_text(encoding='utf-8-sig'))
        if not isinstance(data,dict):
            return None
        if data.get('publication_required') or (root/'publication.json').exists():
            publication = json.loads((root / 'publication.json').read_text(encoding='utf-8-sig'))
            if (not isinstance(publication,dict)
                    or publication.get('schema') != 'femto-local-publication-v1'
                    or publication.get('status') != 'published'
                    or publication.get('session_id') != data.get('publication_id')
                    or Path(publication.get('result', '')).resolve() != root):
                return None
        if data.get('data_kind') != 'real_femto' or not (root / 'reconstruction' / 'reconstruction_report.json').is_file():
            return None
        if data.get('fixed_camera') is True and data.get('frame_count') == 1:
            from femto.dataset.single_frame_source import resolve_source
            resolve_source(root)
            return 'keyframe'
        if data.get('fixed_camera') is False and int(data.get('frame_count', 0)) > 1:
            from femto.reconstruction_profile import local_route_result, LOCAL_ROUTE_SKIP_REASON
            reconstruction = json.loads((root / 'reconstruction/reconstruction_report.json').read_text(encoding='utf-8-sig'))
            if ((root / 'nerfstudio' / 'transforms.json').is_file()
                    or (data.get('local_measured') is True
                        and data.get('reconstruction_source_mode') == 'moving'
                        and (data.get('nerfstudio_skipped_reason') == 'local_capture_has_fewer_than_12_views'
                             or local_route_result(data, reconstruction)
                                and data.get('nerfstudio_skipped_reason') == LOCAL_ROUTE_SKIP_REASON))):
                return 'moving'
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def remember(outputs: str | Path, root: str | Path) -> Any:
    root = Path(root).resolve()
    mode = mode_of(root)
    if mode:
        marker = Path(outputs) / MARKERS[mode]
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(str(root), encoding='utf-8')
        temporary.replace(marker)
    return mode


def restore(outputs: str | Path) -> Any:
    """Recover old installations and CLI-created results without trusting folder names."""
    outputs = Path(outputs)
    candidates = set(p.parent for p in outputs.glob('*/demo_summary.json'))
    for name in MARKERS.values():
        try:
            candidates.add(Path((outputs / name).read_text(encoding='utf-8-sig').strip()).resolve())
        except (OSError, ValueError):
            pass
    result = dict.fromkeys(MARKERS)
    def timestamp(root):
        try:
            data = json.loads((root / 'demo_summary.json').read_text(encoding='utf-8-sig'))
            if not isinstance(data,dict):
                return '',0
            return str(data.get('created_local', '')), (root / 'demo_summary.json').stat().st_mtime_ns
        except (OSError, ValueError):
            return '', 0
    for root in sorted(candidates, key=timestamp, reverse=True):
        mode = mode_of(root)
        if mode and result[mode] is None:
            result[mode] = root
    return result


def describe(root: str | Path) -> Any:
    if not root:
        return '-'
    try:
        data = json.loads((Path(root) / 'demo_summary.json').read_text(encoding='utf-8-sig'))
        return f"{data.get('created_local', Path(root).name)} | {data.get('frame_count', '?')} RGB-D\n{Path(root).name}"
    except (OSError, ValueError):
        return str(root)
