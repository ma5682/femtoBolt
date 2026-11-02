"""Fixed-input scale comparison; emits evidence and never adopts a candidate model."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from femto.io_utils import load_json,write_json_strict
from femto.annotation.fixed_rgb_geometry import FixedRGBModel
from femto.capture.scan_roi import read,assess_model
from femto.pipeline.scene_reconstruction import reconstruct_scene
from femto.local.reconstruction_gap_audit import GapProbe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();audit=args.audit.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Use a new diagnostic directory')
    original=load_json(audit/'report.json');source=Path(original['source'])
    recorded=load_json(source/'reconstruction_report.json');dataset=Path(recorded['source_dataset'])
    def hashes():return {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in original['source_hashes']}
    before=hashes()
    if before!=original['source_hashes']:raise ValueError('Source differs from the fixed stage audit')
    baseline=FixedRGBModel(source);roi=read(dataset.parents[1])
    _,_,old_classes=assess_model(baseline,roi,evidence=True)
    with np.load(audit/'gap_pixels.npz') as data:pixels=data['pixels_xy'].copy();old_cause=data['cause'].copy()
    probe=GapProbe(pixels,baseline.projection.k,baseline.projection.depth)
    parameters=dict(recorded['parameters']);parameters['voxel_m']=.0025
    output.mkdir(parents=True,exist_ok=False)
    result=reconstruct_scene(dataset,output/'reconstruction',geometry_observer=probe,
        **{key:parameters[key] for key in ('image_scale','voxel_m','sdf_trunc_m','depth_min_m','depth_max_m','point_stride','single_frame_quality')})
    model=FixedRGBModel(output/'reconstruction')
    new_summary,_,new_classes=assess_model(model,roi,evidence=True)
    if not np.array_equal(old_classes!=0,new_classes!=0):raise ValueError('Comparison changed requested target')
    stage_report,cause=probe.report();selected=old_classes!=0
    gains=int(((old_classes!=1)&(new_classes==1)&selected).sum())
    losses=int(((old_classes==1)&(new_classes!=1)).sum())
    extraction=old_cause==0;new_extracted=np.isfinite(probe.depths['tsdf_extracted'])
    rows=dict(schema='femto-gap-reconstruction-scale-comparison-v1',source_audit=str(audit),
        baseline_voxel_m=recorded['parameters']['voxel_m'],candidate_voxel_m=.0025,
        sdf_trunc_m=parameters['sdf_trunc_m'],same_accepted_frames_and_poses=True,
        original_extraction_gaps=int(extraction.sum()),
        original_extraction_gaps_with_candidate_extraction_hit=int((extraction&new_extracted).sum()),
        original_extraction_gaps_still_missing_at_candidate_extraction=int((extraction&~new_extracted).sum()),
        improved_reference_pixels=gains,degraded_reference_pixels=losses,
        candidate_scan_summary=new_summary,stage_audit=stage_report,
        candidate_metrics=result['metrics'],source_unchanged=before==hashes(),source_hashes=before,
        adoption='diagnostic_only_default_unchanged',
        interpretation='scale_sensitivity_and_recorded_view_consistency_do_not_establish_physical_accuracy_or_a_unique_root_cause',
        physical_accuracy_verified=False,execution_ready=False)
    np.savez_compressed(output/'comparison_pixels.npz',baseline_classes=old_classes,candidate_classes=new_classes,
        original_gap_pixels=pixels,candidate_stage_cause=cause)
    write_json_strict(output/'report.json',rows)
    print({key:rows[key] for key in ('original_extraction_gaps_with_candidate_extraction_hit',
        'improved_reference_pixels','degraded_reference_pixels','source_unchanged')},flush=True)
    if not rows['source_unchanged']:raise RuntimeError('Source changed during scale comparison')


if __name__=='__main__':main()
