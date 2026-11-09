"""Offline same-input performance/equivalence record; never modifies a source."""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback
from contextlib import nullcontext


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--code-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--detail',action='store_true')
    parser.add_argument('--generate-only',action='store_true')
    parser.add_argument('--defer-target',action='store_true',help='Use the UI route workflow with area geometry calculated on demand')
    parser.add_argument('--reconstruct',action='store_true',help='Replay posed RGB-D and the saved confirmed selection first')
    args=parser.parse_args();sys.path.insert(0,str(args.code_root.resolve()))
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    from benchmark_observations import Observation,source_identity
    environment=source_identity(args.code_root,root)
    import numpy as np
    from femto.annotation.fixed_rgb_geometry import FixedRGBModel
    from femto.geometry.local_region import model_selection
    from femto.annotation.scene_annotations import mesh_digest
    from femto.planning.patch_reference import create
    from femto.app.qt_planning_io import extract_region
    from femto.planning.surface_workspace import generate,export
    from femto.planning.treatment_head import head_parameters
    from femto.planning.region_project import save_project,open_project
    from femto.planning.save_validation import SaveValidationSession
    from femto.planning.planning_revision import array_record,digest
    from femto.geometry.region_reuse import operation
    from femto.operation_diagnostics import record_operation
    from femto.io_utils import sha256_file
    if args.detail:
        from femto.performance_detail import collect
    result=dict(source=str(args.source.resolve()),code_root=str(args.code_root.resolve()),
                detailed_instrumentation=args.detail,scope='existing_local_model_to_saved_and_reopened_plan',
                includes_capture=False,includes_reconstruction=args.reconstruct,timings={},resources={},environment=environment)
    def stage(name,fn):
        started=time.perf_counter()
        with Observation() as observation,(collect() if args.detail else nullcontext()):
            with record_operation(name,root=root/'diagnostics'):
                value=fn()
        result['timings'][name]=time.perf_counter()-started
        result['resources'][name]=observation.result
        print(name,round(result['timings'][name],4),flush=True)
        return value
    started=time.perf_counter()
    try:
        source_file=args.source/'scene_tsdf_mesh_m.ply';before=sha256_file(source_file)
        model_source=args.source.resolve()
        if args.reconstruct:
            from femto.pipeline.scene_reconstruction import reconstruct_scene
            previous=json.loads((args.source/'reconstruction_report.json').read_text(encoding='utf-8'))
            parameters=previous['parameters'];dataset=Path(previous['source_dataset'])
            model_source=root/'reconstruction'
            rebuilt=stage('reconstruct',lambda:reconstruct_scene(dataset,model_source,
                image_scale=parameters['image_scale'],voxel_m=.005,sdf_trunc_m=.02,
                depth_min_m=parameters['depth_min_m'],depth_max_m=parameters['depth_max_m'],
                point_stride=parameters['point_stride'],single_frame_quality=parameters['single_frame_quality'],
                region_request=args.source/'region_request.json'))
            result['scope']='posed_rgbd_and_confirmed_selection_to_saved_and_reopened_plan'
            result['reconstruction_equivalence']=dict(metrics=rebuilt['metrics'],metrics_equal=rebuilt['metrics']==previous['metrics'],
                mesh_bytes_equal=sha256_file(model_source/'scene_tsdf_mesh_m.ply')==before,
                cloud_bytes_equal=sha256_file(model_source/'scene_fused_points_m.ply')==sha256_file(args.source/'scene_fused_points_m.ply'))
            assert all(result['reconstruction_equivalence'][key] for key in ('metrics_equal','mesh_bytes_equal','cloud_bytes_equal'))
        source=stage('load',lambda:FixedRGBModel(model_source))
        mask=model_selection(source)
        model=stage('patch',lambda:create(source,root/'model'))
        result['partition']=model.view_reference['derivation']['partition']
        result['source_surface_check']=model.source_surface_check
        result['planning_mesh_sha256']=mesh_digest(model.mesh)
        model,region=stage('selection',lambda:extract_region(model,mask,np.zeros_like(mask),'原始选区','',None))
        # Independent test directories must bind the same test selection ID;
        # random annotation IDs would legitimately change domain/revision hashes.
        region['annotation_id']='33dce309f9c44f5b811b3b46838461a0'
        head=head_parameters('fixed_width_auto',effective_treatment_width_mm=10.,overlap_percent=20.)
        result['target_area_on_demand']=args.defer_target
        plan=stage('generate',lambda:generate(model,region,head,defer_target=args.defer_target))
        result['equivalence']={key:digest(plan['surface_trajectory'][key])
                               for key in ('ordered_items','statistics','gap_report')}
        target_root=Path(plan['surface_trajectory']['source']['region_manifest']).parent
        with np.load(target_root/'target_surface.npz',allow_pickle=False) as arrays:
            result['equivalence']['target_arrays']={key:array_record(arrays[key]) for key in arrays.files}
        result['equivalence']['target_summary']=plan['planning_context']['target']['summary']
        result['equivalence']['target_metrics']=plan['planning_context']['target']['metrics']
        if args.generate_only:
            assert before==sha256_file(source_file)
            result.update(status='passed',source_file_sha256=before,scope='existing_local_model_to_generated_plan')
            return 0
        with operation(),SaveValidationSession() as session:
            native=stage('native_save',lambda:export(model,region,plan,target_session=session))
            stage('portable_save',lambda:save_project(model,region,native/'planning.region.zip',plan=plan,target_session=session))
        result['save_complete_wall_seconds']=time.perf_counter()-started
        reopened=stage('portable_open',lambda:open_project(native/'planning.region.zip',workspace=root/'reopened'))
        assert reopened.saved_plan['surface_trajectory']['ordered_items']==plan['surface_trajectory']['ordered_items']
        assert before==sha256_file(source_file)
        result.update(status='passed',native=str(native),source_file_sha256=before)
    except Exception as error:
        result.update(status='failed',error=str(error));traceback.print_exc()
    finally:
        result['wall_seconds']=time.perf_counter()-started
        (root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print('result',result['status'],root,flush=True)
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
