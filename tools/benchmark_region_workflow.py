"""Offline synthetic region-first workflow; no camera, robot, or production indexes.

Preparation/reconstruction are called in-process for comparable baselines.
GUI load, route generation, full save and portable readback use production code.
First run is distinguished from repeat runs; no OS cold-cache claim is made.
"""
import argparse
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'tests'))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')


class BaselineImports(importlib.abc.MetaPathFinder):
    def __init__(self,root):self.root=root
    def find_spec(self,fullname,path=None,target=None):
        if not fullname.startswith('femto.'):return None
        candidate=self.root/Path(*fullname.split('.')).with_suffix('.py')
        if candidate.is_file():return importlib.util.spec_from_file_location(fullname,candidate)
        return None


def set_synthetic_surface(capture,dataset,surface):
    """Create a known curved fixture before reference binding and source hashes.

    Each camera ray intersects the same world sphere using its saved pose.
    Identical depth pixels are never copied across moving views.
    """
    if surface=='plane':return dict(kind='plane',z_m=.5)
    import numpy as np
    from femto.io_utils import load_jsonl,imwrite_unicode
    intrinsics=capture.camera.intrinsics
    y,x=np.indices((intrinsics.height,intrinsics.width))
    rays=np.stack([(x-intrinsics.cx)/intrinsics.fx,(y-intrinsics.cy)/intrinsics.fy,np.ones_like(x)],axis=-1)
    center=np.array([0.,0.,1.1]);radius=.6
    records={row['frame_id']:row for row in capture.accepted}
    for row in load_jsonl(dataset/'frames.jsonl'):
        pose=np.asarray(records[row['frame']]['T_WC'])
        direction=rays@pose[:3,:3].T
        offset=pose[:3,3]-center
        a=np.sum(direction*direction,axis=-1)
        b=2*(direction@offset)
        c=offset@offset-radius**2
        discriminant=b*b-4*a*c
        depth=(-b-np.sqrt(np.maximum(discriminant,0)))/(2*a)
        valid=(discriminant>=0)&(depth>0)&(depth<65.535)
        points=pose[:3,3]+direction[valid]*depth[valid,None]
        np.testing.assert_allclose(np.linalg.norm(points-center,axis=1),radius,rtol=0,atol=1e-12)
        millimeters=np.zeros(depth.shape,np.uint16)
        millimeters[valid]=np.rint(depth[valid]*1000).astype(np.uint16)
        imwrite_unicode(dataset/row['depth'],millimeters)
    return dict(kind='sphere',center_m=center.tolist(),radius_m=radius,
                depth_quantization_mm=1,poses='fixture_world_camera_transforms')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--frames',type=int,nargs='+',default=[1,3])
    parser.add_argument('--width',type=int,default=320)
    parser.add_argument('--shapes',nargs='+',choices=['small','large','split'],default=['small','large','split'])
    parser.add_argument('--surface',choices=['plane','sphere'],default='plane')
    parser.add_argument('--continue-on-error',action='store_true',help='Record failures and continue the same requested matrix')
    args=parser.parse_args()
    if args.baseline:sys.meta_path.insert(0,BaselineImports(args.baseline.resolve()))
    import numpy as np
    from unittest.mock import patch
    from test_region_first_reconstruction import dataset_fixture
    from test_qt_planning import QtPlanningTests
    from femto.geometry.local_region import load_reference,save_request
    from femto.io_utils import load_json_object,load_jsonl,sha256_file,write_json_strict
    from femto.local.local_processing import LocalProcessing
    from femto.pipeline.complete_demo import run_demo
    from femto.pipeline.local_offline import publish_result
    from femto.capture.final_assessment import assess_final
    from femto.dataset.acquisition_completion import confirm_for_planning
    from femto.dataset.task_context import new_task,save_task
    from femto.planning.region_project import open_project
    from femto.app.qt_planning_io import masks_from_region
    from femto.operation_diagnostics import record_operation
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    suite=dict(schema='region-workflow-benchmark-v2',data_kind='synthetic_calibrated_'+args.surface,
        platform=platform.platform(),python=platform.python_version(),resolution=[args.width,args.width*3//4],
        baseline=str(args.baseline) if args.baseline else None,
        scope='in-process reconstruction + Qt load/route + native and portable save + independent reopen',
        excludes=['camera acquisition','synthetic fixture creation','human review time','offline worker process startup','actual GPU screen presentation'],
        automatic_timing='wall clock from selection request commit through complete native and portable save; excludes independent reopen',
        presentation_timing='Qt offscreen widget grab after route installation; CPU paint and copy, not display/GPU latency',
        cache='fresh result each run; immutable inputs and OS file cache may be warm; no cross-request acceptance cache',
        pre_selection_preparation=[],cases=[])
    import importlib
    suite['dependencies']={name:importlib.import_module(name).__version__ for name in ('numpy','cv2','open3d','PySide6','trimesh')}
    suite['source_sha256']={str(path.relative_to(ROOT)):sha256_file(path) for folder in ('femto','tests')
                           for path in (ROOT/folder).rglob('*.py')}
    def measure(times,key,fn):
        start=time.perf_counter()
        try:return fn()
        finally:times[key]=time.perf_counter()-start
    for count in args.frames:
        source=out/f'source_{count}';source.mkdir()
        capture,dataset=dataset_fixture(source,count,width=args.width,height=args.width*3//4)
        suite['synthetic_surface']=set_synthetic_surface(capture,dataset,args.surface)
        started=time.perf_counter();ref=load_reference(dataset)
        suite['pre_selection_preparation'].append(dict(frames=count,reference_load_s=time.perf_counter()-started))
        h,w=ref.depth.shape
        raw_paths=[p for p in capture.session.rglob('*') if p.is_file()]
        original={str(p):sha256_file(p) for p in raw_paths}
        # Acquisition guidance is full-frame; assess_final uses the saved paint
        # mask for a region-first model, retaining missing target pixels.
        capture.scan_roi=dict(schema='femto-scan-roi-v1',rect_norm=[0.,0.,1.,1.],image_size=[w,h],
            status='bound',session_id=capture.session_id,reference_frame_id=0,
            calibration_sha256=capture.calibration['sha256'],raw_rgbd_cropped=False)
        for shape in args.shapes:
            mask=np.zeros((h,w),np.uint8)
            if shape=='small':mask[int(.44*h):int(.56*h),int(.44*w):int(.56*w)]=1
            elif shape=='large':mask[int(.32*h):int(.68*h),int(.32*w):int(.68*w)]=1
            else:
                mask[int(.4*h):int(.6*h),int(.25*w):int(.37*w)]=1
                mask[int(.4*h):int(.6*h),int(.63*w):int(.75*w)]=1
            for repeat in range(args.repeats):
                case=out/f'{count}_{shape}_{repeat}';case.mkdir();times={}
                row=dict(frames=count,shape=shape,repetition=repeat,first_run=repeat==0,selected_pixels=int(mask.sum()),times=times)
                suite['cases'].append(row)
                fixture=QtPlanningTests();fixture.setUpClass();fixture.setUp();page=fixture.page
                try:
                    with record_operation('benchmark.local_workflow',root=case/'diagnostics') as trace:
                        automatic_started=time.perf_counter()
                        request=measure(times,'selection_commit',lambda:save_request(ref,mask,case/'request',25))
                        job=LocalProcessing(dataset,case/'processing')
                        selected,_=measure(times,'pose_and_temporal',job.run)
                        def reconstruct():
                            return run_demo(argparse.Namespace(mode='latest',dataset=selected,datasets_root=source/'data',
                                outputs_root=case,no_viewer=True,publish_latest=False,no_source_report=True,
                                publication_id=capture.session_id,
                                local_processing_report=str(job.path),region_request=request))
                        result,_=measure(times,'reconstruction_and_exports',reconstruct)
                        completion=measure(times,'model_quality_check',lambda:assess_final(capture,result))
                        measure(times,'publish_isolated_result',lambda:publish_result(result,case,capture.session_id))
                        measure(times,'model_review_record',lambda:confirm_for_planning(result,completion['binding']))
                        save_task(result,new_task())
                        def load():
                            page.load_source(result/'reconstruction',restore_draft=False);fixture.wait_task()
                        measure(times,'load_and_prepare_region',load)
                        if page.selection_dirty:
                            measure(times,'extra_region_confirmation',lambda:(page.commit_region(),fixture.wait_task()))
                        measure(times,'route_generation_and_checks',lambda:(page.generate(),fixture.wait_task()))
                        assert page.usable(),page._error
                        row['presentation_size']=[page.width(),page.height()]
                        measure(times,'result_present_cpu',page.grab)
                        page.set_stage(2);page.reviewed.setChecked(True)
                        # Keep all benchmark output together, with fresh paths.
                        with patch('femto.app.qt_planning.OUTPUTS_ROOT',case/'saved'):
                            measure(times,'save_native_and_portable',lambda:(page.save_plan(),fixture.wait_task()))
                        times['automatic_total_to_save']=time.perf_counter()-automatic_started
                        times['measured_stages_sum_to_save']=sum(v for key,v in times.items() if key!='automatic_total_to_save')
                        assert times['automatic_total_to_save']>=times['measured_stages_sum_to_save']
                        reopened=measure(times,'portable_reopen',lambda:open_project(page.archive/'planning.region.zip',workspace=case/'reopened'))
                        restored,_=masks_from_region(reopened.store.regions[0],mask.shape)
                        np.testing.assert_array_equal(restored,mask)
                        report=load_json_object(result/'reconstruction/reconstruction_report.json')
                        proof=report['reconstruction_scope'].get('input_provenance')
                        if not args.baseline:
                            assert proof is not None,'Current local reconstruction must retain input fingerprints'
                            assert page.model.capture_source_verification['input_files_verified']
                            assert reopened.store.local_context['reconstruction_scope']['input_provenance']==proof
                            row['input_fingerprints']=dict(frame_count=len(proof['frame_ids']),
                                file_count=len(proof['files']),sha256=proof['sha256'])
                        row.update(status='passed',local_scope=report['reconstruction_scope'],
                            mesh_vertices=report['metrics'].get('mesh_vertices'),mesh_triangles=report['metrics']['mesh_triangles'],
                            path_points=sum(len(item.get('points',[])) for item in page.plan.get('surface_trajectory',{}).get('ordered_items',[])),
                            algorithms=page.plan.get('planning_context',{}).get('revision',{}).get('inputs',{}).get('algorithms'),
                            result=str(result),archive=str(page.archive),diagnostics=str(trace.directory),
                            source_unchanged=all(sha256_file(p)==value for p,value in original.items()))
                        assert row['source_unchanged']
                except Exception as error:
                    row.update(status='failed',error=str(error))
                    if not args.continue_on_error:raise
                finally:
                    page.dirty=False;fixture.doCleanups()
                    write_json_strict(out/'summary.json',suite)
                print(json.dumps(dict(frames=count,shape=shape,repeat=repeat,times=times),ensure_ascii=False),flush=True)
    return int(any(row['status']!='passed' for row in suite['cases']))


if __name__=='__main__':raise SystemExit(main())
