"""Resolve a fixed-keyframe model or RGB-D photo into its single source frame (schema checks only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from femto.io_utils import load_json as read_json


def resolve_source(source):
    source = Path(source).resolve()
    model = None
    if (source/'demo_summary.json').is_file():
        summary = read_json(source/'demo_summary.json')
        if summary.get('frame_count') != 1 or summary.get('fixed_camera') is not True:
            raise ValueError('Single-frame GS requires a completed fixed keyframe model')
        model = source
        source = Path(summary['source_dataset'])
        if not source.is_absolute(): source = model/source
        source = source.resolve()
    manifest = read_json(source/'manifest.json')
    if manifest.get('schema') == 'femto-rgbd-photo-v1':
        if model is not None: raise ValueError('Keyframe model points to an unrelated photo')
        return source, manifest, dict(source_kind='photo', source_model=None, selected_source_frame=None)
    if (manifest.get('schema') != 'femto-full-scene-rgbd-v1'
            or manifest.get('status') != 'derived_fixed_camera_keyframe'
            or manifest.get('frame_count') != 1
            or manifest.get('parameters',{}).get('fixed_camera') is not True):
        raise ValueError('Select a derived fixed keyframe, not an unselected capture sequence')
    frames = [json.loads(line) for line in (source/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(frames) != 1: raise ValueError('Keyframe dataset must contain exactly one RGB-D frame')
    parameters = manifest['parameters']
    adapted = dict(manifest, alignment='depth_to_color', files=dict(color=frames[0]['rgb'],depth=frames[0]['depth']),
                   image=dict(depth_range_m=[parameters['depth_min_mm']/1000,parameters['depth_max_mm']/1000]))
    return source, adapted, dict(source_kind='fixed_keyframe', source_model=str(model) if model else None,
        source_model_summary_sha256=hashlib.sha256((model/'demo_summary.json').read_bytes()).hexdigest() if model else None,
        selected_source_frame=frames[0].get('source_frame'), source_capture=manifest.get('source_dataset'))
