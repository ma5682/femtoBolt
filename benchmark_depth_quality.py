"""Repeat in-memory single-frame processing; excludes file I/O and XYZ export."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from depth_quality import DepthQualityConfig, assess_and_fill_depth


def benchmark(directory: Path, repeats=10):
    with Image.open(directory/'original_depth_mm.png') as im:
        depth = np.asarray(im).copy()
    metadata = json.loads((directory/'depth_quality.json').read_text(encoding='utf-8'))
    low,high = metadata['active_range_mm']
    results = {}
    for mode in ('fast','quality'):
        config = DepthQualityConfig(mode=mode)
        assess_and_fill_depth(depth,low,high,config=config)
        timings = []
        for _ in range(repeats):
            *_, report = assess_and_fill_depth(depth,low,high,config=config)
            timings.append(report['processing_ms'])
        results[mode] = dict(median_ms=float(np.median(timings)),p95_ms=float(np.percentile(timings,95)),
                             samples_ms=timings,repaired_pixels=report['repaired_pixels'])
    output = dict(input=str(directory),shape=list(depth.shape),repeats=repeats,
                  method='one warmup per mode, depth-only in-memory, no RGB guidance, I/O or XYZ export',
                  results=results, realtime_capture_changed=False,
                  runtime_policy='Offline/photo worker only. Do not synchronously insert into 15 FPS camera loop.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = benchmark(args.directory)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result['results'],indent=2))
