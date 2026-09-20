"""Replay saved aligned RGB-D evidence without opening the camera or training."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from depth_quality import DepthQualityConfig, process_depth_quality
from depth_confidence_cloud import build_confident_depth_cloud, write_confident_point_cloud


def replay_depth_quality(source: Path, *, frame_index: int = 0, mode: str = 'quality') -> Path:
    source = Path(source).resolve()
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] == 'femto-rgbd-photo-v1':
        rgb_path, depth_path = [source / manifest['files'][key] for key in ('color','depth')]
        low, high = manifest['image']['depth_range_m']
    elif manifest['schema'] == 'femto-full-scene-rgbd-v1':
        frames = [json.loads(line) for line in (source/'frames.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        frame = frames[frame_index]
        rgb_path, depth_path = source/frame['rgb'], source/frame['depth']
        low, high = manifest['camera']['depth_range_m']
    else:
        raise ValueError('Unsupported saved RGB-D schema')
    for path in (rgb_path, depth_path):
        path.resolve().relative_to(source)
    evidence = [source/'manifest.json', rgb_path, depth_path, *sorted((source/'calibration').glob('*.json')),
                *sorted(source.glob('*.ply'))]
    if (source/'frames.jsonl').exists():
        evidence.append(source/'frames.jsonl')
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence}
    with Image.open(depth_path) as im:
        depth = np.asarray(im)
    with Image.open(rgb_path) as im:
        rgb = np.asarray(im.convert('RGB'))
    output = source/'derived'/('depth_quality_replay_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    filled, confidence, report, paths = process_depth_quality(
        depth, low*1000, high*1000, output, rgb_bgr=rgb[:,:,::-1], config=DepthQualityConfig(mode=mode))
    intr = SimpleNamespace(**json.loads((source/'calibration/color_intrinsic.json').read_text(encoding='utf-8')))
    distortion = json.loads((source/'calibration/distortion.json').read_text(encoding='utf-8'))['rgb']
    cloud, point_confidence = build_confident_depth_cloud(depth,filled,confidence,rgb[:,:,::-1],intr,distortion,low*1000,high*1000)
    write_confident_point_cloud(output/'point_cloud_hole_filled_m.ply',cloud,point_confidence)
    unchanged = all(hashlib.sha256((source/p).read_bytes()).hexdigest() == sha for p,sha in hashes.items())
    observed_unchanged = bool(np.array_equal(filled[confidence == 1],depth[confidence == 1]))
    summary = dict(source=str(source), frame_index=frame_index, pixel_grid='original depth-to-color aligned; before K6 rectification',
                   source_sha256=hashes, source_unchanged=unchanged, original_valid_depth_unchanged=observed_unchanged,
                   coordinate_system='color camera RDF metres; K6; no world pose applied',
                   original_invalid_rate=report['original_invalid_rate'], filled_invalid_rate=report['filled_invalid_rate'],
                   repaired_pixels=report['repaired_pixels'], policy=report['policy'], quality_files=paths,
                   mode=mode, processing_ms=report['processing_ms'], depth_error_check=report['depth_error_check'],
                   material_acceptance='Unlabelled existing real capture; not a five-material physical acceptance test')
    (output/'replay_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    if not unchanged or not observed_unchanged:
        raise AssertionError('Original evidence changed')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--frame-index',type=int,default=0)
    parser.add_argument('--mode',choices=('fast','quality'),default='quality')
    args = parser.parse_args()
    print(replay_depth_quality(args.source,frame_index=args.frame_index,mode=args.mode))
