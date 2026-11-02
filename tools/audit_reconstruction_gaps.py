"""Replay recorded reconstruction and locate the stage responsible for reference gaps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    from femto.app.qt_planning_io import load_source
    from femto.capture.scan_roi import read, assess_model
    from femto.pipeline.reconstruction_gap_audit import audit
    source, output = args.source.resolve(), args.output.resolve()
    model = load_source(source, None, print, require_confirmed=False)
    dataset = Path(model.store.origin_reference['source_dataset']).resolve()
    roi = read(dataset.parents[1])
    if roi is None:raise ValueError('Missing original scan region')
    _, _, classes = assess_model(model, roi, evidence=True)
    report = audit(model, classes != 0, output, progress=print)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':main()
