"""Stage snapshots and bounded reconstruction while the capture owner keeps tracking.

Only ``poll`` runs on the capture thread, after the preceding raw write commits.
The background worker receives frozen metadata and immutable raw frame paths.
Stage results are not published as final models or business latest indexes.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import threading
import time
import uuid

import numpy as np

from femto.dataset.local_reference import calibration_identity
from femto.io_utils import write_json, write_jsonl


MAX_STAGE_ROUNDS = 8  # Resource limit; not a completeness threshold.


def freeze_stage(capture, tracker, stage_id):
    """Freeze a committed prefix without closing or modifying the live source."""
    if capture.closing.is_set() or capture._cancelled():
        raise ValueError('checkpoint_capture_closed')
    if (tracker.lost or tracker.relocalizing or tracker.last_valid_received is None
            or time.monotonic() - tracker.last_valid_received > tracker.max_gap_s
            or not capture.records or capture.records[-1]['state'] != 'tracking'):
        raise ValueError('checkpoint_pose_untrusted')
    if len(capture.accepted) < 3:
        raise ValueError('need_more_views')
    if capture.scan_roi is None or capture.scan_roi.get('status') != 'bound':
        raise ValueError('checkpoint_target_unbound')
    if (capture.scan_roi.get('session_id') != capture.session_id
            or capture.scan_roi.get('reference_frame_id') != capture.accepted[0]['frame_id']
            or capture.scan_roi.get('calibration_sha256') != capture.calibration['sha256']):
        raise ValueError('checkpoint_target_reference_mismatch')
    if calibration_identity(capture.session) != capture.calibration:
        raise ValueError('calibration_changed')
    # Validate the stage name before using it as a filesystem component.
    if uuid.UUID(hex=stage_id).hex != stage_id:
        raise ValueError('Invalid stage identity')
    parent = capture.session / 'derived' / 'checkpoints' / stage_id
    source = parent / 'source'
    source.mkdir(parents=True, exist_ok=False)
    shutil.copytree(capture.session / 'calibration', source / 'calibration')
    rows = [json.loads(line) for line in capture.scan.frames_log.read_text(
        encoding='utf-8-sig').splitlines() if line.strip()]
    if len(rows) != capture.scan.frame_count or len(rows) != len(capture.records):
        raise ValueError('Checkpoint prefix is not fully committed')
    for row in rows:
        for key in ('rgb', 'depth', 'pose'):
            original = (capture.session / row[key]).resolve()
            if not original.is_file() or capture.session.resolve() not in original.parents:
                raise ValueError('Checkpoint source frame missing or outside session')
            row[key] = os.path.relpath(original, source).replace('\\', '/')
    write_jsonl(source / 'frames.jsonl', rows, ensure_ascii=True, compact=False)
    frozen = copy.copy(capture)
    frozen.session = source
    frozen.records = copy.deepcopy(capture.records)
    frozen.accepted = copy.deepcopy(capture.accepted)
    frozen.reference_provenance = copy.deepcopy(capture.reference_provenance)
    frozen.scan = copy.copy(capture.scan)
    frozen.scan.session = source
    frozen.scan.frames_log = source / 'frames.jsonl'
    frozen.scan.last_preview = None
    frozen.finished_at = time.monotonic()
    # finalize_scan writes only inside this private checkpoint source and its
    # parent. Its private latest_scene.txt cannot change the live business index.
    from femto.capture.scan_session import finalize_scan
    finalize_scan(frozen.scan, frozen.camera)
    write_jsonl(source / 'derived' / 'live_tracking.jsonl', frozen.records)
    write_json(source / 'derived' / 'scan_roi.json', copy.deepcopy(capture.scan_roi))
    posed = frozen._export_poses()
    frozen._write_tracking_report(tracker)
    manifest_path = posed / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    manifest['local_capture'].update(capture_scope='stage_checkpoint', stage_id=stage_id,
                                     original_source_dataset=str(capture.session.resolve()))
    write_json(manifest_path, manifest)
    request = dict(schema='femto-capture-checkpoint-v1', stage_id=stage_id,
                   session_id=capture.session_id, source_dataset=str(capture.session.resolve()),
                   frozen_source=str(source.resolve()), posed_dataset=str(posed.resolve()),
                   accepted_frame_ids=[row['frame_id'] for row in frozen.accepted],
                   reference_provenance=frozen.reference_provenance,
                   scan_roi=copy.deepcopy(capture.scan_roi),
                   status='frozen', execution_ready=False, physical_accuracy_verified=False)
    write_json(parent / 'checkpoint.json', request)
    return parent, posed, request


class CaptureCheckpoints:
    """One cancellable background stage; requests consumed at raw commit boundaries."""
    def __init__(self, capture):
        self.capture = capture
        self.lock = threading.Lock()
        self.requested = False
        self.thread = None
        self.cancel = threading.Event()
        self.rounds = 0
        self.last_result = None
        self.last_accepted_ids = None
        self.last_feedback = None
        self.running = False

    def _stop_reason_unlocked(self):
        if self.rounds>=MAX_STAGE_ROUNDS:return 'checkpoint_round_limit'
        if self.last_feedback and self.last_feedback[0].get('stop_reason'):
            return 'checkpoint_no_gain_limit'
        return None

    @property
    def stop_reason(self):
        with self.lock:return self._stop_reason_unlocked()

    def request(self):
        with self.lock:
            if self.capture.closing.is_set() or self.capture._cancelled():
                return False
            stop=self._stop_reason_unlocked()
            if stop:
                self.capture.emit('notice',key=stop)
                return False
            if self.requested or self.running or (self.thread is not None and self.thread.is_alive()):
                return False
            self.requested = True
        self.capture.emit('checkpoint_queued')
        return True

    def poll(self, tracker):
        with self.lock:
            if not self.requested:
                return
            self.requested = False
            self.running = True
        stage_id = uuid.uuid4().hex
        try:
            if self.rounds >= MAX_STAGE_ROUNDS:
                raise ValueError('checkpoint_round_limit')
            ids = tuple(row['frame_id'] for row in self.capture.accepted)
            if ids == self.last_accepted_ids:
                raise ValueError('checkpoint_no_new_views')
            parent, posed, request = freeze_stage(self.capture, tracker, stage_id)
            self.rounds += 1
            self.cancel = threading.Event()
            self.capture.emit('checkpoint_started', stage_id=stage_id, round=self.rounds,
                              accepted_frame_ids=request['accepted_frame_ids'])
            self.thread = threading.Thread(target=self._run,
                args=(parent, posed, request, self.cancel), name='FemtoCaptureCheckpoint', daemon=False)
            self.thread.start()
        except Exception as error:
            with self.lock:
                self.running = False
            self.capture.emit('checkpoint_error', stage_id=stage_id, error=str(error))

    def _run(self, parent, posed, request, cancel):
        from femto.pipeline.local_offline import LocalOfflineJob
        stage_id = request['stage_id']
        stopped = lambda: cancel.is_set() or self.capture.closing.is_set() or self.capture._cancelled()
        try:
            job = LocalOfflineJob(posed, parent / 'processing', self.capture.outputs,
                                  self.capture.session_id, stopped,
                                  progress=lambda quality: self.capture.emit(
                                      'checkpoint_progress', stage_id=stage_id, quality=quality))
            result, _ = job.run()
            if stopped():
                raise RuntimeError('checkpoint_cancelled')
            from femto.annotation.fixed_rgb_geometry import FixedRGBModel
            from femto.capture.scan_roi import assess_model
            from femto.capture.scan_feedback import build_feedback, target_key
            from femto.pipeline.local_comparison import sha256
            model = FixedRGBModel(Path(result)/'reconstruction', cancel=cancel)
            summary, preview, classes = assess_model(model, request['scan_roi'], cancel,
                                                     evidence=True)
            binding=dict(target_key=target_key(request['scan_roi'],request['reference_provenance'],
                                               model.view_reference),stage_id=stage_id,
                session_id=request['session_id'],reference_frame_id=request['scan_roi']['reference_frame_id'],
                model_sha256=sha256(model.store.source))
            feedback=build_feedback(classes,binding,self.last_feedback,
                                    raw_mapping=summary.get('raw_target_mapping'))
            if stopped():
                raise RuntimeError('checkpoint_cancelled')
            write_json(parent/'scan_feedback.json',feedback)
            from femto.capture.scan_exchange import task_batch
            write_json(parent/'scan_tasks.json',task_batch(feedback,
                request['reference_provenance']['calibration']['sha256']),allow_nan=False)
            if classes is not None:
                np.save(parent/'diagnosis.npy',classes,allow_pickle=False)
            from femto.io_utils import imwrite_unicode
            imwrite_unicode(parent/'scan_check.png',preview[:,:,::-1])
            with self.lock:
                if stopped():
                    raise RuntimeError('checkpoint_cancelled')
                completed = dict(request, status='checked', result=str(result),
                                 scan_summary=summary, feedback=feedback,
                                 feedback_path=str(parent/'scan_feedback.json'),
                                 preview_path=str(parent/'scan_check.png'))
                write_json(parent / 'checkpoint.json', completed)
                self.last_result = completed
                self.last_accepted_ids = tuple(request['accepted_frame_ids'])
                self.last_feedback=(feedback,classes.copy() if classes is not None else None)
            self.capture.emit('checkpoint_ready', stage_id=stage_id, checkpoint=completed,
                              operation_stop_reason=self.stop_reason)
        except Exception as error:
            try:
                write_json(parent/'checkpoint.json',dict(request,status='cancelled' if stopped() else 'failed',
                                                        error=str(error)))
            except OSError:
                pass  # Journal write failure must not suppress the terminal UI event.
            self.capture.emit('checkpoint_error', stage_id=stage_id, error=str(error),
                              operation_stop_reason=self.stop_reason,
                              previous_stage_id=(self.last_result or {}).get('stage_id'))
        finally:
            with self.lock:
                self.running = False

    def cancel_stage(self):
        with self.lock:
            queued = self.requested and not self.running
            self.requested = False
            self.cancel.set()
        if queued:
            self.capture.emit('checkpoint_cancelled')

    def close(self):
        self.cancel_stage()
        if self.thread is not None:
            # LocalOfflineJob's supervisor terminates the child on cancellation.
            self.thread.join(timeout=10.)
            if self.thread.is_alive():
                self.capture.emit('checkpoint_error', error='checkpoint_supervisor_not_stopped')
