"""Bounded, SDK-independent local capture worker; raw evidence stays immutable."""
from __future__ import annotations
from typing import Any

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
from types import SimpleNamespace
import uuid
from collections import Counter

import numpy as np
from femto.io_utils import append_jsonl, write_json, write_jsonl
from femto.tracking.moving_quality import POLICY, rotation_deg
from femto.dataset.calibration_files import write_calibration
from femto.tracking.scene_live_tracker import LiveTracker, PairEstimator, PreparedRGBD
from femto.dataset.local_reference import calibration_identity, reference_provenance
from femto.tracking.local_stability_feedback import CandidateStability
from femto.tracking.local_frame_selection import ViewSelector, SELECTION_POLICY
from femto.tracking.local_frame_selection import LOCAL_RADIUS_M, LOCAL_ROTATION_DEG, local_workspace

TRACKER_OPTIONS = dict(feature_seed=False, keyframe_anchor=False, relocalization=False,
                       relocalization_window_s=15., relocalization_confirmations=2)

class LocalCapture:
    def __init__(self, camera, datasets, outputs, events, *, session_id=None, shutdown_event=None, cancel_event=None, tracker_options=None):
        self.tracker_options=dict(TRACKER_OPTIONS)
        if tracker_options:
            if set(tracker_options)-set(TRACKER_OPTIONS):raise ValueError('Unknown tracker option')
            self.tracker_options.update(tracker_options)
        from femto.capture.scan_session import ActiveScan
        self.session_id = uuid.UUID(hex=session_id).hex if session_id is not None else uuid.uuid4().hex
        self.session = Path(datasets) / (time.strftime('scene_%Y%m%d_%H%M%S_local_')+self.session_id[:8])
        for child in ('rgb', 'depth', 'pose', 'derived'):
            (self.session / child).mkdir(parents=True, exist_ok=False)
        write_calibration(self.session, camera, camera.intrinsics)
        self.calibration = calibration_identity(self.session)
        self.reference_provenance = None
        # Snapshot only; background code never calls the camera SDK.
        self.camera = SimpleNamespace(metadata=copy.deepcopy(camera.metadata), intrinsics=camera.intrinsics,
            min_depth_mm=camera.min_depth_mm, max_depth_mm=camera.max_depth_mm)
        now = time.monotonic()
        self.scan = ActiveScan(self.session, self.session/'frames.jsonl', now, now,
            300, 0, 0, False, 'moving', camera.min_depth_mm, camera.max_depth_mm)
        self.outputs, self.events = Path(outputs), events
        self.queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.closing = threading.Event()
        self.processing_cancel = cancel_event if cancel_event is not None else threading.Event()
        self.shutdown_event = shutdown_event
        self.successful_model_published = False
        self.reconstruct = False
        self.paused = False
        self.requested = None
        self.stability = CandidateStability(self.session_id)
        self.view_selector = ViewSelector()
        self._last_stability_event = None
        self.dropped = 0
        self.last_submitted_received = None
        self.finished_at = None
        self.abort_reason = None
        self.abort_detail = None
        self.records, self.accepted = [], []
        self.preview_worker = None
        self.processing_job = None
        self.thread = threading.Thread(target=self._run, name='FemtoLocalCapture', daemon=False)
        self.thread.start()

    def emit(self, kind: str, **payload: Any) -> None:
        if kind=='pose':payload['record'].setdefault('status_updated_at',datetime.now(timezone.utc).isoformat())
        self.events.put(('local_'+kind, dict(session_id=self.session_id, source=str(self.session), **payload)))

    def _cancelled(self):
        shutdown=getattr(self,'shutdown_event',None)
        return self.processing_cancel.is_set() or (shutdown is not None and shutdown.is_set())

    def submit(self, frame: Any) -> None:
        from femto.capture.scan_session import copy_frame
        if self.closing.is_set():
            return
        if self.queue.full():
            # Prefer fresh input over accumulating two old frames at camera FPS.
            # Only unprocessed input is discarded; processed raw evidence remains.
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
        received = time.monotonic()
        self.last_submitted_received = received
        try:
            self.queue.put_nowait((copy_frame(frame, include_native_depth=False), received))
        except queue.Full:
            self.dropped += 1

    def request(self) -> None:
        with self.lock:
            if not self.paused and not self.closing.is_set() and self.requested is None:
                self.requested = time.monotonic()
                if hasattr(self,'stability'):self.stability.request(self.requested)
            self._emit_stability()

    def _emit_stability(self):
        if not hasattr(self,'stability'):return
        current=self.stability.snapshot()
        if current != getattr(self,'_last_stability_event',None):
            self._last_stability_event=current
            self.emit('stability', stability=current)

    def pause(self, value: Any) -> None:
        with self.lock:
            if self.closing.is_set():
                return
            self.paused = value
            self.requested = None
            if hasattr(self,'stability'):self.stability.reset('paused' if value else 'idle')
            self._emit_stability()
        self.emit('pause', paused=value)

    def finish(self, reconstruct: bool=True) -> None:
        # Accepted prefix only: queued frames and requests are not accepted on stop.
        with self.lock:
            if not reconstruct and not self.successful_model_published:
                self.processing_cancel.set()
            if self.closing.is_set():
                return
            self.reconstruct = reconstruct
            self.requested = None
            if hasattr(self,'stability'):self.stability.reset('ended' if reconstruct else 'cancelled')
            self._emit_stability()
            self.finished_at = time.monotonic()
            self.closing.set()

    def abort(self, reason: str, detail: Any=None) -> None:
        """A device failure is an error, not a user cancellation or new origin."""
        with self.lock:
            if self.closing.is_set():
                return
            self.abort_reason, self.abort_detail = reason, detail
            self.reconstruct = False
            self.requested = None
            if hasattr(self,'stability'):self.stability.reset('lost', reason)
            self._emit_stability()
            self.finished_at = time.monotonic()
            self.closing.set()

    def _select(self, record, received, prepared=None):
        record['selection'] = dict(state='quality_insufficient', reason=record.get('rejection_reason', 'untrusted_pose'))
        if record['state'] != 'tracking':
            return False
        from femto.dataset.scene_dataset_validator import validate_pose
        invalid = (record.get('T_WC') is None or validate_pose(record['T_WC']) or
            record.get('quality',{}).get('usable') is False or
            record.get('continuity',{}).get('current_frame_verified') is False or
            (self.accepted and record.get('reference_frame_id') != self.accepted[0].get('reference_frame_id')))
        if invalid:
            record['selection']=dict(state='quality_insufficient',reason='pose_quality_or_reference_mismatch')
            return False
        if record.get('motion_discontinuity'):
            with self.lock:
                self.requested=None
                if hasattr(self,'stability'):self.stability.reset('reset','pose_correction_recheck')
            record['selection']=dict(state='waiting_still',reason='pose_correction_recheck')
            return False
        record['local_workspace'] = local_workspace(record['T_WC'])
        within_limit=record['local_workspace']['within_limit']
        if not self.accepted:
            if within_limit and isinstance(prepared, PreparedRGBD):
                view, record['selection'] = self.view_selector.evaluate(prepared, record)
                self.view_selector.accept(view)
            else:
                record['selection'] = dict(state='accepted' if within_limit else 'outside_range',
                                          reason='initial_reference', geometry_evaluated=False)
            return within_limit
        with self.lock:
            requested = self.requested
            record['selection'] = dict(state='outside_range' if not within_limit else 'moving' if record.get('step_m', 1) > .003 or record.get('step_deg', 180) > 1 else 'waiting_still',
                                       reason='no_pending_request', geometry_evaluated=False)
            if self.paused or self.closing.is_set() or requested is None or received < requested:
                return False
            if not within_limit:
                record['selection'] = dict(state='outside_range', reason='outside_local_workspace')
                self.requested = None
                if hasattr(self,'stability'):self.stability.reset('reset','outside_local_workspace')
                self.emit('notice', key='outside_local_workspace')
                return False
            if time.monotonic()-requested > 3:
                record['selection'] = dict(state='request_timeout', reason='request_timeout')
                self.requested = None
                if hasattr(self,'stability'):self.stability.reset('timeout','request_timeout')
                self.emit('notice', key='request_timeout')
                return False
            # Stop-and-capture: wait for low inter-frame motion after the request.
            # Initial engineering gate, not a measured speed or accuracy claim.
            if record.get('step_m', 1) > .003 or record.get('step_deg', 180) > 1.0:
                record['selection'] = dict(state='moving', reason='waiting_for_stillness')
                return False
            if isinstance(prepared, PreparedRGBD) and hasattr(self, 'view_selector'):
                view, selection = self.view_selector.evaluate(prepared, record)
                record['selection'] = selection
                self.requested = None
                if selection['state'] == 'accepted':
                    self.view_selector.accept(view)
                    return True
                self.emit('notice', key='duplicate' if selection['state'] == 'duplicate' else 'selection_quality')
                if hasattr(self, 'stability'):
                    self.stability.reset('duplicate' if selection['state'] == 'duplicate' else 'reset', selection['reason'])
                return False
            relative = np.linalg.inv(np.array(self.accepted[-1]['T_WC'])) @ np.array(record['T_WC'])
            moved = (np.linalg.norm(relative[:3, 3]) >= POLICY['keyframe_translation_m']
                     or rotation_deg(relative) >= POLICY['keyframe_rotation_deg'])
            self.requested = None
            if not moved:
                if hasattr(self,'stability'):self.stability.reset('duplicate')
                self.emit('notice', key='duplicate')
            record['selection'] = dict(state='accepted' if moved else 'duplicate', reason='pose_only_compatibility', geometry_evaluated=False)
            return bool(moved)

    def _run(self):
        from femto.capture.scan_session import save_scan_frame, finalize_scan
        tracker = None
        raw_writer = None
        finalized = False
        report_written = False
        try:
            options=dict(getattr(self,'tracker_options',TRACKER_OPTIONS))
            feature_seed=options.pop('feature_seed')
            tracker = LiveTracker(PairEstimator(self.session, self.scan.depth_min_mm, self.scan.depth_max_mm,feature_seed=feature_seed),
                                  self.scan.depth_min_mm, self.scan.depth_max_mm,**options)
            self.tracker=tracker
            from femto.capture.local_raw_writer import RawFrameWriter
            raw_writer = RawFrameWriter(save_scan_frame)
            from femto.capture.local_live_preview import LivePreviewWorker
            self.preview_worker = LivePreviewWorker(self.session,self.session_id,
                self.scan.depth_min_mm,self.scan.depth_max_mm)
            self.emit('started', preview_mailbox=self.preview_worker.mailbox)
            while not self.closing.is_set():
                if self._cancelled():
                    self.finish(False)
                    break
                try:
                    frame, received = self.queue.get(timeout=.1)
                except queue.Empty:
                    with self.lock:
                        if self.requested is not None and time.monotonic()-self.requested > 3:
                            self.requested = None
                            self.stability.reset('timeout','request_timeout')
                            self.emit('notice', key='request_timeout')
                        self.stability.expire(time.monotonic())
                        self._emit_stability()
                    if tracker.relocalizing or (tracker.last_valid_received is not None and time.monotonic()-tracker.last_valid_received > tracker.max_gap_s):
                        now=time.monotonic()
                        reason=('no_frames' if self.last_submitted_received is None or now-self.last_submitted_received>tracker.max_gap_s else 'tracking_gap')
                        record=tracker.watchdog(reason)
                        self.emit('pose',record=record,accepted=len(self.accepted))
                        with self.lock:
                            self.requested=None;self.stability.reset('reset' if tracker.relocalizing else 'lost',reason)
                        if tracker.lost:self.closing.set()
                    elif tracker.reference_id is None and time.monotonic()-self.scan.started > 15:
                        raise ValueError('origin_timeout')
                    continue
                index = self.scan.frame_count
                write_started = time.monotonic()
                raw_writer.submit(self.scan, frame, index)
                pose_started = time.monotonic()
                previous_valid_id, previous_valid_received = tracker.last_valid_frame_id, tracker.last_valid_received
                record = tracker.process(frame, index, received)
                pose_finished = time.monotonic()
                receipt = raw_writer.commit(record)
                durable_finished = time.monotonic()
                # A trustworthy computation is not an admissible frame until
                # the matching raw evidence has successfully been persisted.
                if durable_finished-received > tracker.max_gap_s and record['state'] in ('tracking','relocalizing'):
                    tracker.last_valid_frame_id, tracker.last_valid_received = previous_valid_id, previous_valid_received
                    tracker.relocalization_candidates=[]
                    if record.get('continuity',{}).get('relocalized') and tracker.relocalization_events:
                        tracker.relocalization_events[-1].update(event='recovery_rejected_at_raw_receipt',reason='raw_write_stale')
                    record.update(state='lost', T_WC=None, rejection_reason='raw_write_stale',
                                  last_valid_frame_id=previous_valid_id,
                                  loss_context=tracker.mark_lost('raw_write_stale', received=received, frame_id=index))
                    record['continuity'].update(phase='lost', current_frame_verified=False, accepted_for_fusion=False)
                    record.pop('xyz_mm', None); record.pop('pitch_yaw_roll_deg', None)
                record.update(session_id=self.session_id, dropped_input_frames=self.dropped,
                    raw_write_receipt=receipt, durable_latency_ms=(durable_finished-received)*1000,
                    timing_ms=dict(queue_wait=max(0., write_started-received)*1000,
                                   raw_write=receipt['write_ms'],
                                   raw_write_wait=(durable_finished-pose_finished)*1000,
                                   pose_processing=(pose_finished-pose_started)*1000))
                with self.lock:
                    stopped = self.closing.is_set()
                if record['state']=='relocalizing':
                    with self.lock:
                        self.requested=None;self.stability.reset('reset','relocalizing')
                selected = not stopped and not self._cancelled() and self._select(record, received, tracker.previous)
                record['selected_for_fusion'] = selected
                with self.lock:
                    if not stopped:
                        record['candidate_stability'] = self.stability.observe(
                            record, time.monotonic(), accepted=selected, paused=self.paused)
                if selected:
                    if not self.accepted:
                        self.reference_provenance = reference_provenance(self.session_id, record, self.calibration)
                        write_json(self.session/'derived'/'local_reference.json', self.reference_provenance)
                    self.accepted.append(record)
                    tracker.add_keyframe(tracker.previous,record['T_WC'],index)
                record['calibration_sha256'] = self.calibration['sha256']
                record['reference_capture_timestamp_us'] = (
                    self.reference_provenance['capture_timestamp_us'] if self.reference_provenance else None)
                record['status_updated_at']=datetime.now(timezone.utc).isoformat()
                self.records.append(record)
                append_jsonl(self.session/'derived'/'live_tracking.jsonl', record)
                if selected:
                    self.preview_worker.submit(record)
                self.emit('pose', record=record, accepted=len(self.accepted))
                if tracker.reference_id is None and time.monotonic()-self.scan.started > 15:
                    raise ValueError('origin_timeout')
                if tracker.lost:
                    self.closing.set()
                if time.monotonic()-self.scan.started >= 300:
                    self.emit('notice', key='session_limit')
                    self.finish(True)
            self.finished_at = self.finished_at or time.monotonic()
            self.preview_worker.close()
            finalize_scan(self.scan, self.camera)
            finalized = True
            if tracker.relocalizing and self.reconstruct:tracker.mark_lost('relocalization_interrupted')
            self._write_tracking_report(tracker)
            report_written = True
            if self.abort_reason:
                raise ValueError(self.abort_reason)
            if tracker.lost:
                raise ValueError('tracking_lost')
            if not self.reconstruct or self._cancelled():
                self.emit('cancelled'); return
            if not self.records or self.records[-1]['state'] != 'tracking':
                raise ValueError('final_pose_invalid')
            if self.finished_at is not None and self.finished_at-self.records[-1]['received_monotonic'] > tracker.max_gap_s:
                raise ValueError('final_pose_invalid')
            if len(self.accepted) < 3:
                raise ValueError('need_three_views')
            posed = self._export_poses()
            self.emit('processing')
            from femto.pipeline.local_offline import LocalOfflineJob, publish_result
            self.processing_job = LocalOfflineJob(posed,
                self.session/'derived'/('processing_'+uuid.uuid4().hex[:8]),
                self.outputs, self.session_id, self._cancelled,
                progress=lambda quality: self.emit('processing', quality=quality))
            result, summary = self.processing_job.run()
            # Short commit is serialized with finish(False). Once committed, a
            # late cancellation cannot turn a completed publication into failure.
            with self.lock:
                if self._cancelled():
                    from femto.pipeline.local_comparison import StageStopped
                    raise StageStopped('cancelled', 'Cancelled before publication', self.processing_job.stats)
                publish_result(result, self.outputs, self.session_id, cancelled=self._cancelled)
                self.successful_model_published = True
            self.emit('finished', result=str(result), summary=summary, quality=self.processing_job.report)
        except Exception as error:
            self._handle_run_failure(error,tracker,raw_writer,finalized,report_written,finalize_scan)
        finally:
            self.closing.set()
            if raw_writer is not None:
                raw_writer.close()
            if self.preview_worker is not None:
                self.preview_worker.close()
                self.preview_worker.thread.join(timeout=5.)

    def _handle_run_failure(self,error,tracker,raw_writer,finalized,report_written,finalize_scan):
        # A frame write may be in flight if pose preprocessing itself threw.
        # Finish that owned write before finalizing its raw frame index.
        if raw_writer is not None:
            raw_writer.close()
        self.finished_at = self.finished_at or time.monotonic()
        status = 'cancelled' if getattr(error, 'status', None) == 'cancelled' else 'failed'
        reporting_errors = []
        if self.processing_job is not None:
            try:
                reporting_errors = self.processing_job.terminal(status, str(error))
            except Exception as report_error:
                reporting_errors = [str(report_error)]
        publication = getattr(error, 'details', {})
        failure = dict(schema='femto-local-failure-v1', error=str(error), session_id=self.session_id,
            source=str(self.session), reference_frame_id=tracker.reference_id if tracker else None,
            accepted_frames=[r['frame_id'] for r in self.accepted],
            termination=tracker.loss_context if tracker else None,
            last_rejection=tracker.last_rejection if tracker else None,
            device_error=self.abort_detail, status=status,
            successful_model_published=bool(self.successful_model_published or publication.get('successful_model_published')),
            publication=publication, processing_reporting_errors=reporting_errors,
            processing_report=str(self.processing_job.path) if self.processing_job is not None else None)
        try:
            if not finalized:
                finalize_scan(self.scan, self.camera)
            if tracker is not None and not report_written:
                self._write_tracking_report(tracker, error=str(error))
        except Exception as report_error:
            failure['report_error'] = str(report_error)
        try:
            write_json(self.session/'derived'/'local_failure.json', failure)
        except Exception as report_error:
            failure['failure_report_error'] = str(report_error)
        self.emit('cancelled' if status == 'cancelled' else 'error', error=str(error), failure=failure,
                  quality=self.processing_job.report if self.processing_job is not None else None)

    def _write_tracking_report(self, tracker, *, error=None):
        elapsed = max(0., (self.finished_at or time.monotonic()) - self.scan.started)
        # Very fast mocked/loopback sessions can start and finish in the same
        # monotonic clock tick.  Keep recorded FPS finite and self-consistent.
        if self.records and elapsed == 0.:
            elapsed = float(np.finfo(float).eps)
        valid_count = sum(r['state'] == 'tracking' for r in self.records)
        latencies = [r['latency_ms'] for r in self.records]
        stages = {}
        for key in ('queue_wait', 'raw_write', 'raw_write_wait', 'pose_processing'):
            values = [r['timing_ms'][key] for r in self.records if key in r.get('timing_ms', {})]
            stages[key] = dict(mean=float(np.mean(values)) if values else None,
                               p50=float(np.percentile(values, 50)) if values else None,
                               p95=float(np.percentile(values, 95)) if values else None)
        report = dict(schema='femto-local-tracking-v1', session_id=self.session_id,
            reference_frame_id=tracker.reference_id,
            state='lost' if tracker.lost else 'error' if self.abort_reason or error else 'stopped',
            coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct',
            frame_count=len(self.records), accepted_frames=[r['frame_id'] for r in self.accepted],
            dropped_input_frames=self.dropped, max_gap_s=tracker.max_gap_s,
            local_workspace_radius_m=LOCAL_RADIUS_M,
            local_workspace_rotation_deg=LOCAL_ROTATION_DEG,
            selection_policy=SELECTION_POLICY,
            latency_ms=latencies, reference_provenance=self.reference_provenance,
            calibration=self.calibration, duration_s=elapsed,
            processed_fps=len(self.records)/elapsed if elapsed > 0 else None,
            valid_pose_fps=valid_count/elapsed if elapsed > 0 else None,
            valid_pose_count=valid_count, rejected_frame_count=len(self.records)-valid_count,
            rejection_counts=dict(Counter(r['rejection_reason'] for r in self.records if r.get('rejection_reason'))),
            tracking_lost_events=int(tracker.lost), termination=tracker.loss_context,
            abort_reason=self.abort_reason, error=error,
            retried_pair_count=sum(len(r.get('pair', {}).get('attempts', r.get('pair_attempts', []))) > 1
                                  for r in self.records),
            recovered_pair_count=sum(r['state'] == 'tracking' and
                                      r.get('pair', {}).get('initialization') in ('previous_valid_step','feature_pnp')
                                      for r in self.records),
            latency_mean_ms=float(np.mean(latencies)) if latencies else None,
            latency_p50_ms=float(np.percentile(latencies, 50)) if latencies else None,
            latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
            timing_ms=stages,
            timing_definition='Session-average FPS; latency is host receipt to pose completion; durable_latency includes matching raw receipt. Raw write overlaps pose; stages are not additive. Not sensor-to-display latency.',
            pose_source='online_rgbd_research_grade', relocalization=getattr(tracker,'relocalization',False),
            tracker_options=tracker.options() if hasattr(tracker,'options') else dict(TRACKER_OPTIONS),tracking_algorithm='local-evidence-tracker-v2',
            relocalization_events=getattr(tracker,'relocalization_events',[]),
            feature_seed_pair_count=sum(r.get('pair',{}).get('initialization')=='feature_pnp' and r['state']=='tracking' for r in self.records),
            anchor_used_count=sum(bool(r.get('anchor',{}).get('used')) for r in self.records),
            candidate_stability=dict(descriptor_limit=5, history_descriptor_limit=9, image_depth_audit='pending_offline',
                max_timestamp_gap_s=1., max_window_span_s=4., max_translation_m=.001, max_rotation_deg=.2,
                selection_unchanged=False, final_state=self.stability.snapshot()))
        from femto.tracking.local_tracking_diagnostics import summarize
        report['diagnostics'] = summarize(self.records, elapsed, self.dropped)
        report['raw_save_policy'] = dict(max_outstanding_writes=1, receipt_required_before_admission=True,
                                        input_queue_limit=1, full_aligned_rgbd_preserved=True)
        write_json(self.session/'derived'/'local_capture_report.json', report)

    def _export_poses(self):
        # Check all associations before creating any posed dataset.
        if calibration_identity(self.session) != self.calibration:
            raise ValueError('calibration_changed')
        raw = [json.loads(s) for s in self.scan.frames_log.read_text(encoding='utf-8').splitlines()]
        if not self.accepted:
            raise ValueError('No accepted reference frame')
        seen = set()
        from femto.dataset.scene_dataset_validator import validate_pose
        for record in self.accepted:
            index = record.get('frame_id')
            if (type(index) is not int or not 0 <= index < len(raw) or index in seen
                    or raw[index]['frame'] != index
                    or record.get('state', 'tracking') != 'tracking'
                    or record.get('reference_frame_id') != self.accepted[0]['frame_id']
                    or record.get('capture_timestamp_us') != raw[index]['camera_timestamp_us']
                    or validate_pose(np.asarray(record['T_WC'], dtype=float))):
                raise ValueError('Accepted RGB-D / pose frame mismatch')
            if not local_workspace(record['T_WC'])['within_limit']:
                raise ValueError('outside_local_workspace')
            seen.add(index)
        first = self.accepted[0]
        row = raw[first['frame_id']]
        proof = reference_provenance(self.session_id, dict(first,
            color_timestamp_us=row['color_timestamp_us'], depth_timestamp_us=row['depth_timestamp_us']), self.calibration)
        if self.reference_provenance is not None and proof != self.reference_provenance:
            raise ValueError('Initial reference provenance changed')
        self.reference_provenance = proof
        output = self.session/'derived'/'local_posed'
        output.mkdir(exist_ok=False)
        (output/'pose').mkdir()
        shutil.copytree(self.session/'calibration', output/'calibration')
        derived = []
        for n, record in enumerate(self.accepted):
            frame = dict(raw[record['frame_id']])
            for key in ('rgb', 'depth'):
                frame[key] = os.path.relpath(self.session/frame[key], output).replace('\\', '/')
            frame['pose'], frame['pose_valid'] = f'pose/{n:06d}.json', True
            write_json(output/frame['pose'], dict(schema='femto-scene-frame-pose-v1', frame=frame['frame'],
                pose_valid=True, T_WC=record['T_WC'], pose_source='online_rgbd_research_grade',
                tracking_algorithm='local-evidence-tracker-v2',
                tracking_options=dict(getattr(self, 'tracker_options', TRACKER_OPTIONS)),
                continuity=record.get('continuity'),
                coordinate_system='+X right, +Y down, +Z forward; metres', session_id=self.session_id,
                reference_frame_id=record['reference_frame_id'], capture_timestamp_us=record['capture_timestamp_us'],
                reference_provenance=proof))
            derived.append(frame)
        write_jsonl(output/'frames.jsonl', derived, ensure_ascii=True, compact=False)
        manifest = json.loads((self.session/'manifest.json').read_text(encoding='utf-8'))
        manifest.update(status='derived_pose_estimated', source_dataset=str(self.session), frame_count=len(derived),
            pose=dict(valid=True, source='online_rgbd_research_grade'),
            local_capture=dict(session_id=self.session_id, reference_frame_id=self.accepted[0]['frame_id'],
                reference_provenance=proof,
                measured_only=True, local_workspace_radius_m=LOCAL_RADIUS_M,
                local_workspace_rotation_deg=LOCAL_ROTATION_DEG,
                selection_policy=SELECTION_POLICY,
                tracking_algorithm='local-evidence-tracker-v2',
                tracking_options=dict(getattr(self, 'tracker_options', TRACKER_OPTIONS)),
                tracking_report=str(self.session/'derived'/'local_capture_report.json')))
        write_json(output/'manifest.json', manifest)
        # Keep the standard quality-report reader compatible with the online path.
        from femto.tracking.moving_quality import pose_diversity
        span_m, span_deg = pose_diversity([np.asarray(r['T_WC']) for r in self.accepted])
        write_json(output/'pose_quality_report.json', dict(status='estimated_research_grade',
            source_dataset=str(self.session), source_frame_count=len(self.records), frame_count=len(derived),
            accepted_frame_ordinals=[r['frame_id'] for r in self.accepted],
            frame_quality=[dict(frame_ordinal=r['frame_id'], **r['quality']) for r in self.records if 'quality' in r],
            rejected_frames=[r for r in self.records if r['state'] != 'tracking'],
            pairs=[dict(inlier_rmse_m=r['pair']['rmse_m'],cycle_consistency=r['pair']['cycle'])
                   for r in self.records if 'pair' in r and 'cycle' in r['pair']],
            translation_span_m=span_m, rotation_span_deg=span_deg,
            keyframe_selection=dict(tracked_frame_ordinals=[r['frame_id'] for r in self.records if r['state']=='tracking'],
                redundant_frame_ordinals=[r['frame_id'] for r in self.records if r['state']=='tracking' and not r['selected_for_fusion']]),
            online_tracking=dict(session_id=self.session_id, reference_frame_id=self.accepted[0]['reference_frame_id'],
                reference_provenance=proof,
                max_gap_s=2., dropped_input_frames=self.dropped,
                latency_ms=[r['latency_ms'] for r in self.records]),
            tracker_options=getattr(self,'tracker_options',TRACKER_OPTIONS),
            limitations=['Research-grade; optional same-session keyframe relocalization only; no loop closure or pose graph; physical accuracy pending']))
        return output
