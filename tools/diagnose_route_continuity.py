"""Replay one saved planning input and compare v4.4 to the quality constraint.

Reads source evidence without changing it. Writes a new diagnostic directory;
never replaces the clinician's plan or exports an executable trajectory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from femto.planning.mesh_slice_planner import _load_input, plan_mesh_slice
from femto.planning.quality_constrained import constrain, edge_checks
from femto.planning.effective_domain import validate as validate_domain
from femto.planning.surface_trajectory import ORDERED_SELECTION_ALGORITHM, ScanSegment


def crossing_preserving_samples(points, edge_faces, step_m):
    """Diagnostic alternative only: keep all original triangle crossings."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = lengths > 1e-12
    if not keep.any():return []
    clean, faces = [points[0]], []
    for index, valid in enumerate(keep):
        if valid:
            clean.append(points[index+1])
            faces.append(int(edge_faces[index]))
    points = np.asarray(clean)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    distances = np.unique(np.r_[np.arange(0., cumulative[-1], step_m), cumulative])
    samples = []
    for distance in distances:
        edge = min(len(lengths)-1, max(0, int(np.searchsorted(cumulative, distance, side='right')-1)))
        fraction = (distance-cumulative[edge])/lengths[edge]
        samples.append((points[edge]*(1-fraction)+points[edge+1]*fraction, faces[edge]))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--spacing-mm', type=float, required=True)
    parser.add_argument('--preserve-face-crossings', action='store_true',
                        help='Diagnostic experiment; does not change product or old planner source')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Diagnostic output already exists; choose a new directory')
    assessment = json.loads((args.manifest.parent/'planning_assessment.json').read_text(encoding='utf-8'))
    _, manifest, mesh = _load_input(args.manifest)
    record, domain = assessment['quality_evidence'], assessment['effective_domain']
    stored = assessment['route_report']['quality_constraint']
    width = stored['effective_width_mm']
    allowed = validate_domain(domain, record, width, mesh, manifest['region'])
    print('Replaying original connected serpentine planner...', flush=True)
    from femto.planning.mesh_slice_planner import _resample
    with patch('femto.planning.mesh_slice_planner._resample',
               crossing_preserving_samples if args.preserve_face_crossings else _resample):
        candidate = plan_mesh_slice(args.manifest, row_spacing_mm=args.spacing_mm,
            connect_rows=True, block_planning=True, selection_blocks=True,
            selection_algorithm=ORDERED_SELECTION_ALGORITHM)
    segments = [s for s in candidate.ordered_items if isinstance(s, ScanSegment)]
    print('Checking each original scan/connector edge...', flush=True)
    accepted, reasons, margins = edge_checks(segments, mesh, record, width, allowed_faces=allowed)
    groups = {'scan': Counter(), 'connector': Counter()}
    lengths = {'scan': Counter(), 'connector': Counter()}
    boundary_details = {'scan': Counter(), 'connector': Counter()}
    from femto.planning.quality_constrained import checked_evidence, _rectangle
    classes, _ = checked_evidence(record)
    k = np.asarray(record['view_reference']['undistorted_intrinsic'])
    turns, offset = [], 0
    for segment in segments:
        turn = None
        for left, right in zip(segment.points, segment.points[1:]):
            role = 'connector' if right.get('role') == 'connector' else 'scan'
            reason = reasons[offset]
            groups[role][reason] += 1
            lengths[role][reason] += float(np.linalg.norm(np.asarray(right['xyz_m'])-left['xyz_m']))*1000
            if reason == 'quality_or_width_boundary':
                a, b = np.asarray(left['xyz_m']), np.asarray(right['xyz_m'])
                rect = _rectangle((a+b)/2, np.linalg.norm(a-b)/2+width/2000+margins[offset]/1000, k)
                if rect is None:
                    boundary_details[role]['unprojectable_ball'] += 1
                else:
                    x0, y0, x1, y1 = rect
                    h, w = classes.shape
                    if not (0 <= x0 <= x1 < w and 0 <= y0 <= y1 < h):
                        boundary_details[role]['outside_image'] += 1
                    crop = classes[max(0,y0):min(h,y1+1), max(0,x0):min(w,x1+1)]
                    if (crop == 0).any():boundary_details[role]['includes_unselected_pixels'] += 1
                    if (crop > 1).any():boundary_details[role]['includes_pending_pixels'] += 1
            if role == 'connector':
                if turn is None:
                    turn = dict(edges=0, retained=0, rejection_counts=Counter())
                turn['edges'] += 1
                turn['retained'] += int(accepted[offset])
                if not accepted[offset]:turn['rejection_counts'][reason] += 1
            elif turn is not None:
                turns.append(turn)
                turn = None
            offset += 1
        if turn is not None:turns.append(turn)
    print('Replaying current post-filter and comparing the saved report...', flush=True)
    filtered = constrain(candidate, mesh, manifest['region'], record, width, domain=domain)
    fresh = filtered.gap_report['quality_constraint']
    matches = {key: fresh[key] == stored[key] for key in
               ('candidate_edges', 'retained_edges', 'rejected_edges', 'rejection_counts', 'edge_decision_runs')}
    summary = dict(source_manifest=str(args.manifest.resolve()), source_assessment=str(
        (args.manifest.parent/'planning_assessment.json').resolve()),
        row_spacing_mm=args.spacing_mm, effective_width_mm=width,
        diagnostic_face_crossings_preserved=args.preserve_face_crossings,
        saved_report_reproduced=matches, original=candidate.statistics, filtered=filtered.statistics,
        edge_counts_by_role=groups, edge_length_mm_by_role=lengths,
        quality_width_boundary_details_nonexclusive=boundary_details,
        connector_runs=dict(total=len(turns), fully_retained=sum(t['retained']==t['edges'] for t in turns),
                            fully_rejected=sum(t['retained']==0 for t in turns), details=turns),
        max_local_residual_mm=float(margins.max(initial=0)),
        pending_pixels=stored['pending_pixels'], selected_pixels=stored['selected_pixels'])
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in [('comparison.json', summary), ('candidate_v44.json', candidate.to_dict()),
                        ('filtered_v2.json', filtered.to_dict())]:
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    compact = {key: value for key, value in summary.items() if key not in ('original', 'filtered')}
    compact['original_chains'] = candidate.statistics['segment_count']
    compact['filtered_chains'] = filtered.statistics['segment_count']
    compact['connector_runs'].pop('details')
    print(json.dumps(compact, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
