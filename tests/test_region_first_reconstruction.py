"""Region-first geometry, unchanged evidence and fixed-reference UI regression."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
import queue
import threading
import time

import cv2
import numpy as np

from femto.capture.local_capture import LocalCapture
from femto.capture.scan_session import save_scan_frame, finalize_scan
from femto.geometry.pose_geometry import Intrinsics
from femto.geometry.local_region import load_reference,save_request,read_request,RegionDepthFilter,model_selection
from femto.io_utils import sha256_file
from test_segmented_capture import make_camera,make_frame


def dataset_fixture(root, count=3, *, width=160, height=120):
    w,h = width,height
    camera = make_camera('wide_detail',w*h)
    camera.intrinsics = Intrinsics(w,h,w*1.25,w*1.25,(w-1)/2,(h-1)/2)
    with mock.patch.object(LocalCapture,'_run'):
        capture = LocalCapture(camera,root/'data',root/'outputs',queue.Queue(),region_first=True)
        capture.thread.join()
    for i in range(count):
        frame=make_frame([500]*(w*h),i+1)
        frame.depth_mm=frame.depth_mm.reshape(h,w)
        y,x=np.indices((h,w));frame.color_bgr=np.stack([(x*13)%255,(y*17)%255,((x+y)*7)%255],axis=-1).astype(np.uint8)
        save_scan_frame(capture.scan,frame,lossless=True)
        pose=np.eye(4);pose[0,3]=(i%3)*.012;pose[1,3]=(i//3)*.012
        capture.accepted.append(dict(frame_id=i,T_WC=pose.tolist(),reference_frame_id=0,capture_timestamp_us=i+1))
    finalize_scan(capture.scan,capture.camera)
    dataset=capture._export_poses()
    from femto.io_utils import write_json_strict
    from femto.io_utils import write_jsonl
    write_jsonl(capture.session/'derived/live_tracking.jsonl',[
        dict(row,state='tracking',step_m=.012 if row['frame_id'] else 0.,step_deg=0.,
             color_timestamp_us=row['capture_timestamp_us'],depth_timestamp_us=row['capture_timestamp_us'])
        for row in capture.accepted])
    write_json_strict(capture.session/'derived/local_capture_report.json',dict(
        schema='femto-local-tracking-v1',session_id=capture.session_id,reference_frame_id=0,
        coordinate_frame='world_rdf',unit='metre',transform_direction='T_W0_Ct',
        pose_source='online_rgbd_research_grade',accepted_frames=list(range(count)),
        reference_provenance=capture.reference_provenance))
    return capture,dataset


class RegionFirstTests(TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.capture,self.dataset=dataset_fixture(self.root)
        self.reference=load_reference(self.dataset)
        self.mask=np.zeros(self.reference.depth.shape,np.uint8);self.mask[40:80,60:100]=1
        self.request=save_request(self.reference,self.mask,self.root/'request',25)

    def test_moving_view_projects_region_and_rejects_occluding_depth(self):
        output=self.root/'geometry';output.mkdir()
        region=RegionDepthFilter(self.request,self.dataset,.25,2.21,output)
        first=region.filter(self.reference.depth,np.eye(4),self.reference.k,0)
        pose=np.eye(4);pose[0,3]=.024
        other=region.filter(self.reference.depth,pose,self.reference.k,1)
        # A right-moving camera sees the same target further left, not at
        # the original pixel coordinates. No fabricated identity extrinsics.
        self.assertLess(np.nonzero(other)[1].mean(),np.nonzero(first)[1].mean()-8)
        wrong=self.reference.depth.copy();wrong[50:70,70:90]=800
        filtered=region.filter(wrong,np.eye(4),self.reference.k,2)
        self.assertFalse(filtered[50:70,70:90].any())
        self.assertFalse(filtered[:20].any())
        region.save()
        np.testing.assert_array_equal(read_request(output/'region_request.json')[1],self.mask)

    def test_missing_depth_does_not_shrink_intent_or_admit_background(self):
        region=RegionDepthFilter(self.request,self.dataset,.25,2.21,self.root)
        region.reference.depth[45:55,65:75]=0
        other=np.full_like(region.reference.depth,500)
        result=region.filter(other,np.eye(4),region.reference.k,0)
        self.assertFalse(result[45:55,65:75].any())
        self.assertEqual(int(region.mask.sum()),1600)
        self.assertTrue(region.mask[45:55,65:75].all())

    def test_tampered_mask_and_cross_session_rejected(self):
        from femto.io_utils import imwrite_unicode
        bad=self.mask.copy();bad[0,0]=1
        imwrite_unicode(self.request.parent/'requested_mask.png',bad)
        with self.assertRaisesRegex(ValueError,'已改变'):read_request(self.request)
        other=self.root/'other';other.mkdir()
        _,dataset=dataset_fixture(other)
        with self.assertRaisesRegex(ValueError,'不一致'):read_request(self.request,dataset)

    def test_actual_tsdf_is_local_and_keeps_raw_reference(self):
        from femto.pipeline.scene_reconstruction import reconstruct_scene
        from femto.annotation.fixed_rgb_geometry import FixedRGBModel
        sources={p:sha256_file(p) for p in self.capture.session.rglob('*') if p.is_file()}
        output=self.root/'reconstruction'
        timings=[]
        report=reconstruct_scene(self.dataset,output,depth_min_m=.25,depth_max_m=2.21,region_request=self.request,
            timing_sink=lambda name,elapsed:timings.append((name,elapsed)))
        self.assertEqual(timings,[])  # Cloud display normals were removed, not deferred.
        self.assertEqual(report['auxiliary_computations']['status'],'not_computed')
        self.assertNotIn('second_sheet_fraction',report['metrics'])
        self.assertTrue(all(row['integrated_pixels']<6000 for row in report['reconstruction_scope']['frames']))
        self.assertGreater(report['metrics']['mesh_triangles'],30)
        model=FixedRGBModel(output)
        np.testing.assert_array_equal(model.rgb,self.reference.rgb)
        np.testing.assert_array_equal(model_selection(model),self.mask)
        from femto.planning.planning_resolution import create_reduced_model
        from femto.planning.route_sampling import CURVATURE_POLICY
        derived=create_reduced_model(model,self.root/'curvature_reference',stride=4,
            route_sample_step_mm=5.,sampling_policy=CURVATURE_POLICY)
        self.assertIsNone(model_selection(derived))
        self.assertEqual(derived.store.local_context['reconstruction_scope'],model.store.local_context['reconstruction_scope'])
        np.testing.assert_array_equal(model_selection(model),self.mask)
        from copy import deepcopy
        bad=deepcopy(derived.source_lineage);derived.source_lineage['original_view_reference']={}
        with self.assertRaisesRegex(ValueError,'来源'):model_selection(derived)
        derived.source_lineage=bad
        self.assertTrue(all(sha256_file(path)==digest for path,digest in sources.items()))
        self.assertLess(np.ptp(np.asarray(model.mesh.vertices)[:,0]),.2)
        from femto.io_utils import load_json_object,write_json_strict
        changed=load_json_object(output/'region_request.json')
        changed['support_margin_mm']=100
        write_json_strict(output/'region_request.json',changed)
        with self.assertRaisesRegex(ValueError,'重建范围不一致'):
            model_selection(model)

    def test_only_one_roi_contributor_uses_pixel_surface_and_reports_exclusions(self):
        from femto.io_utils import load_jsonl,imwrite_unicode,load_json_object
        from femto.pipeline.scene_reconstruction import reconstruct_scene
        from femto.annotation.fixed_rgb_geometry import FixedRGBModel
        rows=load_jsonl(self.dataset/'frames.jsonl')
        # These are valid posed RGB-D views, but see a different depth layer.
        # They must not influence this reference target or trigger TSDF.
        for row in rows[1:]:
            imwrite_unicode(self.dataset/row['depth'],np.full_like(self.reference.depth,800))
        hashes={p:sha256_file(p) for p in self.capture.session.rglob('*') if p.is_file()}
        output=self.root/'one_contributor'
        with mock.patch('open3d.pipelines.integration.ScalableTSDFVolume',side_effect=AssertionError('Unexpected TSDF')):
            report=reconstruct_scene(self.dataset,output,depth_min_m=.25,depth_max_m=2.21,region_request=self.request)
        self.assertEqual(report['frame_count'],1)
        self.assertEqual(report['parameters']['mesh_method'],'single_frame_pixel_adjacency')
        self.assertEqual(report['depth_quality']['integrations_per_frame'],[0])
        scope=report['reconstruction_scope']
        self.assertEqual(scope['input_frame_count'],3)
        self.assertEqual(scope['contributing_frame_count'],1)
        self.assertEqual([r['used_for_surface'] for r in scope['frames']],[True,False,False])
        self.assertTrue(all(r['exclusion_reason'] and r['surface_input_pixels']==0 for r in scope['frames'][1:]))
        self.assertEqual(scope,load_json_object(output/'region_request.json'))
        model=FixedRGBModel(output)
        np.testing.assert_array_equal(model_selection(model),self.mask)
        np.testing.assert_array_equal(model.rgb,self.reference.rgb)
        self.assertTrue(all(sha256_file(path)==value for path,value in hashes.items()))

    def test_no_local_surface_preserves_missing_intent_and_fails(self):
        from femto.io_utils import load_jsonl,imwrite_unicode,load_json_object
        from femto.pipeline.scene_reconstruction import reconstruct_scene
        for row in load_jsonl(self.dataset/'frames.jsonl'):
            imwrite_unicode(self.dataset/row['depth'],np.zeros_like(self.reference.depth))
        reference=load_reference(self.dataset)
        request=save_request(reference,self.mask,self.root/'missing_request',25)
        output=self.root/'no_surface'
        with mock.patch('open3d.pipelines.integration.ScalableTSDFVolume',side_effect=AssertionError('Unexpected TSDF')):
            with self.assertRaisesRegex(ValueError,'没有可用的实测表面'):
                reconstruct_scene(self.dataset,output,depth_min_m=.25,depth_max_m=2.21,region_request=request)
        scope=load_json_object(output/'region_request.json')
        self.assertEqual(scope['contributing_frame_count'],0)
        self.assertEqual(scope['reference_missing_selected_pixels'],int(self.mask.sum()))
        self.assertFalse((output/'reconstruction_report.json').exists())
        np.testing.assert_array_equal(read_request(output/'region_request.json')[1],self.mask)

    def test_waits_for_selection_retries_and_cancels_without_full_scene_job(self):
        capture=self.capture
        calls=[]
        def reconstruct(posed,*,region_request=None):
            calls.append(read_request(region_request)[1].copy())
            if len(calls)==1:raise ValueError('synthetic reconstruction failure')
        errors=[]
        def worker():
            try:capture._region_workflow(self.dataset)
            except Exception as error:errors.append(error)
        with mock.patch.object(capture,'_reconstruct_posed',side_effect=reconstruct):
            thread=threading.Thread(target=worker);thread.start()
            try:
                self.assertTrue(capture.awaiting_region.wait(5));self.assertEqual(calls,[])
                self.assertTrue(capture.select_region(self.mask,25))
                self.assertTrue(capture.awaiting_region.wait(5));self.assertEqual(len(calls),1)
                capture.finish(False)
            finally:
                capture.processing_cancel.set();thread.join(5)
        self.assertFalse(thread.is_alive());self.assertEqual(len(calls),1)
        self.assertEqual(getattr(errors[0],'status',None),'cancelled')

    def test_offline_child_receives_region_and_keeps_publication_pending(self):
        from femto.pipeline.local_offline import LocalOfflineJob
        from femto.io_utils import load_json_object
        job=LocalOfflineJob(self.dataset,self.root/'job',self.root/'outputs',
            self.capture.session_id,lambda:False,region_request=self.request)
        try:
            result,summary=job.run()
        except Exception:
            log=self.root/'job/worker.log'
            if log.exists():print(log.read_text(encoding='utf-8',errors='replace')[-6000:])
            raise
        scope=load_json_object(result/'reconstruction/region_request.json')
        self.assertEqual(scope['mask_sha256'],read_request(self.request)[2]['mask_sha256'])
        self.assertEqual(summary['frame_count'],3)
        self.assertEqual(load_json_object(result/'publication.json')['status'],'pending')
        self.assertFalse((self.root/'outputs/latest_complete_demo.txt').exists())
        processing=load_json_object(result/'processing_report.json')
        self.assertEqual(processing['frame_counts']['used_for_reconstruction'],3)
        self.assertEqual(processing['frame_counts']['valid_poses'],3)
        timing=load_json_object(Path(processing['performance_record']))
        self.assertEqual(timing['status'],'complete')
        self.assertTrue({'selection.map_depth','reconstruction.tsdf_integrate','reconstruction.tsdf_extract_mesh'} <= timing['stages'].keys())


class RegionEditorTests(TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.app=QApplication.instance() or QApplication([])

    def test_capture_enters_fixed_image_stage_and_preserves_strokes_on_retry(self):
        from femto.app.qt_capture import CapturePage
        with TemporaryDirectory() as temp:
            capture,dataset=dataset_fixture(Path(temp));reference=load_reference(dataset)
            page=CapturePage(connect_camera=False);page.timer.stop()
            try:
                page.session.begin();page.session.session_id=capture.session_id
                page.host.local_capture=capture
                payload=dict(session_id=capture.session_id,reference=reference,source=str(capture.session))
                page.handle_event('local_region_ready',payload)
                self.assertEqual(page.workflow.current_step,6)
                self.assertFalse(page.primary.isEnabled())
                page.region_editor.canvas.work[40:80,60:100]=1
                page.region_editor.canvas.rebuild_layers();page.refresh()
                self.assertTrue(page.primary.isEnabled())
                snapshot=page.region_editor.canvas.work.copy()
                page.handle_event('local_region_ready',dict(payload,error='重试'))
                np.testing.assert_array_equal(page.region_editor.canvas.work,snapshot)
                self.assertIs(page.region_editor.canvas.model,reference)
                with mock.patch.object(capture,'select_region',return_value=False):
                    page.primary_action()
                self.assertIn('选区尚未提交', page.region_editor.note.text())
                self.assertTrue(page.region_progress.isHidden())
                np.testing.assert_array_equal(page.region_editor.canvas.work,snapshot)
                with mock.patch.object(capture,'select_region',return_value=True) as submit:
                    page.primary_action()
                    np.testing.assert_array_equal(submit.call_args.args[0],snapshot)
                self.assertFalse(page.region_editor.isEnabled())
                self.assertFalse(page.primary.isEnabled())
                self.assertIn('正在重建', page.primary.text())
                self.assertIn('正在重建', page.region_editor.note.text())
                self.assertNotIn('再点击', page.region_editor.note.text())
                self.assertFalse(page.region_progress.isHidden())
                page.handle_event('local_region_ready',dict(payload,error='重建失败'))
                self.assertTrue(page.region_progress.isHidden())
                self.assertTrue(page.primary.isEnabled())
                np.testing.assert_array_equal(page.region_editor.canvas.work,snapshot)
            finally:
                page.session.active=False;page.shutdown();page.close()

    def test_local_model_to_real_plan_save_and_portable_reopen(self):
        self.exercise_plan_save(3)

    def test_single_frame_model_plan_save_and_portable_reopen(self):
        self.exercise_plan_save(1)

    def test_two_frame_model_plan_save_and_portable_reopen(self):
        self.exercise_plan_save(2)

    def test_single_frame_direct_route_preview_then_generate_and_save(self):
        self.exercise_plan_save(1,preview=True)

    def test_multiframe_direct_route_preview_then_generate_and_save(self):
        self.exercise_plan_save(3,preview=True)

    def test_local_tsdf_default_survives_previous_standard_model_preference(self):
        self.exercise_plan_save(1,preview=True,prefer_standard=True)

    def test_three_frames_one_contributor_plan_publish_and_reopen(self):
        self.exercise_plan_save(3,contributors=1)

    def test_margin_only_rebuild_keeps_target_and_previous_model(self):
        self.exercise_plan_save(1,margin_only=True)

    def test_portable_replay_reports_missing_changed_and_restored_capture(self):
        self.exercise_plan_save(1,check_sources=True)

    def test_multiframe_portable_replay_checks_nonreference_inputs(self):
        self.exercise_plan_save(3,check_sources=True)

    def test_selected_only_single_frame_plan_save_and_reopen(self):
        self.exercise_plan_save(1,margin_mm=0)

    def test_selected_only_two_frames_plan_save_and_reopen(self):
        self.exercise_plan_save(2,margin_mm=0)

    def test_selected_only_multiframe_plan_save_and_reopen(self):
        self.exercise_plan_save(3,margin_mm=0)

    def test_selected_only_one_contributor_plan_save_and_reopen(self):
        self.exercise_plan_save(3,contributors=1,margin_mm=0)

    def exercise_plan_save(self, count, contributors=None,margin_only=False,check_sources=False,margin_mm=40,preview=False,prefer_standard=False):
        import test_qt_planning as planning_fixture
        from femto.capture.scan_roi import prepare
        from femto.dataset.acquisition_completion import confirm_for_planning
        from femto.dataset.task_context import new_task,save_task
        from femto.planning.region_project import open_project
        from femto.app.qt_planning_io import masks_from_region
        fixture=planning_fixture.QtPlanningTests();fixture.setUpClass();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root=Path(fixture.temp.name)/'region_flow';root.mkdir()
        capture,dataset=dataset_fixture(root,count);reference=load_reference(dataset)
        expected_contributors=count if contributors is None else contributors
        if expected_contributors<count:
            from femto.io_utils import load_jsonl,imwrite_unicode
            for row in load_jsonl(dataset/'frames.jsonl')[expected_contributors:]:
                imwrite_unicode(dataset/row['depth'],np.full_like(reference.depth,800))
        mask=np.zeros(reference.depth.shape,np.uint8);mask[40:80,60:100]=1
        request=save_request(reference,mask,root/'request',margin_mm)
        first=make_frame([500]*19200,1);first.depth_mm=first.depth_mm.reshape(120,160)
        first.color_bgr=reference.rgb[:,:,::-1].copy()
        capture.scan_roi=dict(prepare([.1,.1,.9,.9],first,250,2210),status='bound',
            session_id=capture.session_id,reference_frame_id=0,calibration_sha256=capture.calibration['sha256'])
        from femto.io_utils import write_json_strict
        write_json_strict(capture.session/'derived/scan_roi.json',capture.scan_roi)
        capture._reconstruct_posed(dataset,region_request=request)
        events=list(capture.events.queue)
        finished=next(payload for kind,payload in events if kind=='local_finished')
        result=Path(finished['result'])
        from femto.io_utils import load_json_object
        reconstruction=load_json_object(result/'reconstruction/reconstruction_report.json')
        self.assertEqual(reconstruction['frame_count'],expected_contributors)
        self.assertEqual(reconstruction['parameters']['mesh_method'],
            'single_frame_pixel_adjacency' if expected_contributors==1 else 'tsdf')
        if expected_contributors == 1:
            self.assertEqual(reconstruction['depth_quality']['integrations_per_frame'],[0])
        if count == 1:
            self.assertEqual(finished['quality']['refinement']['reason'],'single_accepted_view')
        completion=finished['acquisition_completion']
        self.assertEqual(completion['scan_summary']['target_scope'],'painted_region')
        self.assertEqual(completion['scan_summary']['requested_pixels'],int(mask.sum()))
        if not preview:confirm_for_planning(result,completion['binding'])
        completion_bytes=(result/'acquisition_completion.json').read_bytes()
        task=new_task();save_task(result,task)
        from femto.pipeline.local_history import scan_history
        history=scan_history(root/'data',root/'outputs')
        matching=next(row for row in history if row.get('result')==str(result))
        self.assertEqual(matching['count'],count)
        self.assertEqual(matching['contributing_frame_count'],expected_contributors)
        self.assertEqual(matching['quality']['frame_counts']['used_for_reconstruction'],expected_contributors)
        self.assertTrue(matching['fixed'],matching.get('fixed_reason'))
        if expected_contributors==1:
            from femto.app.qt_capture import read_review
            with mock.patch('femto.capture.scan_roi.read',return_value=None):
                review=read_review(result,threading.Event(),lambda message:None)
            self.assertEqual(review['frame_count'],count)
            self.assertEqual(review['scan_roi_quality']['target_scope'],'painted_region')
            self.assertEqual(review['scan_roi_quality']['requested_pixels'],int(mask.sum()))
        page=fixture.page
        if prefer_standard:page._new_source_coarse=False
        if preview:
            from femto.app.qt_main import WorkstationWindow
            from femto.app.qt_capture import CaptureState
            from PySide6.QtCore import QEventLoop,QTimer
            self.assertTrue(page.shutdown());page.close()
            window=WorkstationWindow(connect_camera=False)
            self.addCleanup(window.close)
            page=fixture.page=window.planning
            if prefer_standard:page._new_source_coarse=False
            window.capture.task_context=task
            window.capture.session=CaptureState(session_id=capture.session_id,active=True,state='processing',count=count)
            window.capture.handle_event('local_finished',finished)
            loop=QEventLoop();deadline=QTimer();deadline.setSingleShot(True)
            deadline.timeout.connect(loop.quit);window.capture.review_task.finished.connect(loop.quit)
            try:
                deadline.start(60000)
                if window.capture.review_task.busy:loop.exec()
            finally:
                deadline.stop();window.capture.review_task.finished.disconnect(loop.quit)
            self.assertFalse(window.capture.review_task.busy)
            self.assertIs(window.pages.currentWidget(),page)
            self.assertEqual(page.loading_source,(result/'reconstruction').resolve())
        else:
            self.assertTrue(page.load_source(result/'reconstruction',restore_draft=False))
        fixture.wait_task()
        if preview:self.assertIs(window.pages.currentWidget(),page)
        self.assertEqual(page.model.view_reference.get('derivation',{}).get('schema'),
                         None if preview else 'surface-patch-planning-reference-v1')
        if preview:
            self.assertEqual((result/'acquisition_completion.json').read_bytes(),completion_bytes)
            self.assertFalse(page.model.acquisition_completion['confirmed_for_planning'])
            from femto.planning.saved_source import load_source
            with self.assertRaisesRegex(ValueError,'先检查并确认'):
                load_source(result/'reconstruction')
        np.testing.assert_array_equal(page.canvas.work,mask)
        self.assertFalse(page.selection_dirty)
        self.assertIsNotNone(page.region)
        self.assertEqual(page._stage,1)
        # Single and multiple contributors share one route-only destination.
        self.assertTrue(page.scene_panel.is_3d)
        self.assertFalse(page.canvas.editable)
        self.assertTrue(page.tools.isHidden())
        self.assertTrue(page.clear_button.isHidden())
        self.assertFalse(page.inspector_scroll.isHidden())
        page.set_stage(0)
        self.assertEqual(page._stage,1)
        self.assertGreater(page.canvas.zoom,1.)
        page.generate();fixture.wait_task()
        self.assertTrue(page.usable())
        from femto.local.planning_tsdf import PROFILE as TSDF_PROFILE
        self.assertEqual(page.model.view_reference['derivation']['profile'],TSDF_PROFILE)
        if preview:
            confirmed=load_json_object(result/'acquisition_completion.json')
            self.assertTrue(confirmed['confirmed_for_planning'])
            self.assertEqual(confirmed['feedback'],completion['feedback'])
            self.assertEqual(page.model.view_reference['derivation']['schema'],'surface-patch-planning-reference-v1')
            np.testing.assert_array_equal(page.canvas.work,mask)
        page.set_stage(2);page.reviewed.setChecked(True)
        page.save_plan();fixture.wait_task()
        reopened=open_project(page.archive/'planning.region.zip',workspace=root/'reopened')
        self.assertEqual(reopened.capture_source_verification['status'],'not_checked')
        selected,_=masks_from_region(reopened.store.regions[0],mask.shape)
        np.testing.assert_array_equal(selected,mask)
        self.assertEqual(reopened.task_context['task_id'],task['task_id'])
        if count==1 and contributors is None and not margin_only and not check_sources:
            import shutil
            from femto.planning.saved_source import load_source as load_saved_source
            native_only=root/'native_only'
            shutil.copytree(page.archive,native_only,ignore=shutil.ignore_patterns('planning.region.zip'))
            native_model=load_saved_source(native_only/'trajectory.json')
            self.assertEqual(native_model.task_context['task_id'],task['task_id'])
            self.assertEqual(native_model.saved_plan['surface_trajectory']['ordered_items'],
                             reopened.saved_plan['surface_trajectory']['ordered_items'])
        self.assertEqual(reopened.store.local_context['reconstruction_scope']['mask_sha256'],
                         read_request(request)[2]['mask_sha256'])
        self.assertEqual(reopened.store.local_context['geometry_source'],
            'raw_depth_supported_pixel_adjacency' if expected_contributors==1 else 'raw_depth_supported_tsdf')
        if check_sources:
            self.check_portable_sources(page,root,reopened)
        old_mesh=result/'reconstruction/scene_tsdf_mesh_m.ply'
        original_digest=sha256_file(old_mesh)
        capture.acquisition_complete=True
        self.assertTrue(capture.restart_region())
        try:
            self.assertTrue(capture.awaiting_region.wait(5))
            changed=mask.copy()
            if not margin_only:changed[40:80,55:60]=1
            new_margin=80 if margin_only else margin_mm
            self.assertTrue(capture.select_region(changed,new_margin))
            capture.thread.join(15)
            self.assertFalse(capture.thread.is_alive())
            completed=[p for k,p in capture.events.queue if k=='local_finished']
            self.assertEqual(len(completed),2)
            self.assertNotEqual(completed[-1]['result'],str(result))
            self.assertEqual(sha256_file(old_mesh),original_digest)
            rebuilt=Path(completed[-1]['result'])/'reconstruction'
            rebuilt_request=read_request(rebuilt/'region_request.json')
            np.testing.assert_array_equal(rebuilt_request[1],changed)
            self.assertEqual(rebuilt_request[2]['support_margin_mm'],new_margin)
            if margin_only:
                np.testing.assert_array_equal(rebuilt_request[1],mask)
                self.assertGreater(rebuilt_request[2]['frames'][0]['surface_input_pixels'],
                                   reconstruction['reconstruction_scope']['frames'][0]['surface_input_pixels'])
        finally:
            capture.finish(False);capture.thread.join(5)

    def check_portable_sources(self,page,root,reopened):
        from femto.planning.saved_source import load_source
        from femto.app.qt_review import ReviewPage
        from femto.app.qt_planning_io import masks_from_region
        from femto.io_utils import load_jsonl,imread_unicode,imwrite_unicode
        bundle=page.archive/'planning.region.zip'
        bundle_hash=sha256_file(bundle)
        current=load_source(bundle,workspace=root/'verified')
        self.assertEqual(current.capture_source_verification['status'],'verified')
        self.assertTrue(current.capture_source_verification['input_files_verified'])
        dataset=Path(current.store.origin_reference['source_dataset']).resolve()
        unavailable=dataset.with_name(dataset.name+'_temporarily_unavailable')
        # Mutate only this test's private fixture, never a user capture.
        self.assertTrue(dataset.is_relative_to(root.resolve()))
        self.assertTrue(unavailable.is_relative_to(root.resolve()))
        self.assertFalse(unavailable.exists())

        def check_unresolved(status,source_path=None):
            loaded=load_source(bundle,workspace=root/status)
            self.assertIsNone(loaded.capture_source)
            self.assertEqual(loaded.capture_source_verification['status'],status)
            self.assertEqual(loaded.saved_plan['surface_trajectory']['ordered_items'],
                             reopened.saved_plan['surface_trajectory']['ordered_items'])
            self.assertEqual(loaded.saved_plan['planning_context'],reopened.saved_plan['planning_context'])
            np.testing.assert_array_equal(np.asarray(loaded.mesh.vertices),np.asarray(reopened.mesh.vertices))
            selected,_=masks_from_region(loaded.store.regions[0],loaded.rgb.shape[:2])
            expected,_=masks_from_region(reopened.store.regions[0],loaded.rgb.shape[:2])
            np.testing.assert_array_equal(selected,expected)
            page.install_model(loaded,bundle)
            self.assertIn(loaded.capture_source_verification['message'],page.task_context_label.text())
            self.assertTrue(page.task_context_label.isVisible())
            expected_path=source_path or (dataset if status=='missing' else current.capture_source)
            self.assertIn(str(expected_path),page.source_label.text())
            review=ReviewPage();review.show()
            try:
                review.install_model(loaded,bundle)
                self.assertIn(loaded.capture_source_verification['message'],review.source_label.text())
                self.assertFalse(review.capture.isEnabled())
                self.assertIsNotNone(review.plan)
            finally:review.shutdown();review.close()

        dataset.rename(unavailable)
        try:check_unresolved('missing')
        finally:unavailable.rename(dataset)
        row=load_jsonl(dataset/'frames.jsonl')[0]
        rgb=dataset/row['rgb'];original=rgb.read_bytes()
        image=imread_unicode(rgb,cv2.IMREAD_COLOR);image[0,0]^=255
        imwrite_unicode(rgb,image)
        try:check_unresolved('mismatch')
        finally:rgb.write_bytes(original)
        rows=load_jsonl(dataset/'frames.jsonl')
        if len(rows)>1:
            # Non-reference RGB-D and poses must be bound too, even when the
            # first frame and already saved mesh still match perfectly.
            for key in ('depth','pose'):
                path=(dataset/rows[-1][key]).resolve();original=path.read_bytes()
                self.assertTrue(path.is_relative_to(root.resolve()))
                path.write_bytes(original+b'\n')
                try:check_unresolved('mismatch',path)
                finally:path.write_bytes(original)
            hidden=path.with_suffix('.temporarily_unavailable')
            self.assertTrue(hidden.is_relative_to(root.resolve()))
            path.rename(hidden)
            try:check_unresolved('missing',path)
            finally:hidden.rename(path)
        restored=load_source(bundle,workspace=root/'restored')
        self.assertEqual(restored.capture_source_verification['status'],'verified')
        self.assertEqual(restored.capture_source_verification['issues'],[])
        self.assertEqual(sha256_file(bundle),bundle_hash)
