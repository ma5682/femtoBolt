"""Qt acquisition workspace using the existing single-owner SDK and local capture pipeline.

Only widget presentation changes here. Raw-frame admission, reference tracking,
reconstruction and evidence publication remain owned by ``LocalCapture``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import threading
import time
import traceback
from types import SimpleNamespace
import uuid

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QFrame,
    QComboBox, QCheckBox, QScrollArea, QProgressBar, QMessageBox, QLineEdit,
)

from femto.app.camera_worker import CameraWorker
from femto.app.capture_viewport import (
    camera_oblique_points, current_camera_pose, read_review_surface,
    render_scene, sample_aligned_frame_cloud,
)
from femto.app.qt_common import TaskRunner, show_text, open_path, rgb_image
from femto.app.qt_visuals import draw_empty_viewport
from femto.capture.scan_roi import MIN_INITIAL_VALID_FRACTION
from femto.capture.local_capture_policy import MAX_ACCEPTED_VIEWS
from femto.paths import DATASETS_ROOT, OUTPUTS_ROOT


REASONS = {
    'waiting': '请保持相机和目标稳定，等待建立参考位置。',
    'tracking': '参考位置已建立；缓慢移动，停稳后补采。',
    'uncertain': '当前位姿待确认，请停稳并保持与上一视角重叠。',
    'relocalizing': '正在尝试重定位；当前不接纳补采。',
    'lost': '跟踪中断，已有数据保留；请重新开始采集。',
    'tracking_lost': '跟踪中断，已有数据保留；请重新开始采集。',
    'camera_error': '相机读取失败，请检查连接后重新连接。',
    'camera_not_ready': '相机连接已中断，请重新连接后重试。',
    'processing': '正在检查采集质量并生成模型。',
    'loading_review': '模型已生成，正在准备三维检查视图。',
    'ready': '模型已生成，请检查质量后进入工作区域规划。',
    'review_load_error': '模型已生成，但三维预览加载失败；请查看报告或重新打开历史记录。',
    'cancelled': '本次采集已取消，原始数据保留。',
    'duplicate': '视角变化不足，本次未增加接纳帧。',
    'selection_quality': '三维重叠或深度一致性不足，请保持稳定背景并停稳补采。',
    'moving_wait': '正在移动，请停稳后补采。',
    'request_timeout': '补采请求超时，请停稳后重试。',
    'session_limit': '已到 5 分钟会话上限，正在结束。',
    'accepted_view_limit': '已达到该采集入口的视角上限，正在自动结束采集并检查质量。',
    'need_three_views': '至少需要 3 个有效且不同的视角，请重新采集。',
    'need_more_views': '至少接纳 1 个有效视角后才能完成采集，请等待首帧通过检查。',
    'need_valid_view': '没有已接纳的有效 RGB-D 视角，请重新采集。',
    'origin_timeout': '15 秒内未能建立参考位置，请调整目标和距离。',
    'final_pose_invalid': '结束时位姿无效，本次未发布模型。',
    'calibration_changed': '标定文件发生变化，本次未发布模型。',
    'outside_local_workspace': '当前位置超出参考位置周围 30 mm 或总旋转 10° 的范围，请缓慢返回。',
    'relocalized': '已恢复定位，请重新停稳并请求补采。',
    'pose_correction_recheck': '位姿修正后需要重新停稳。',
    'paused_finish': '补采已暂停，请先继续补采并停稳后再完成采集。',
    'scan_roi_reference_rejected': '所选扫描范围对应的首帧未通过定位或质量检查；请重新取景并采集。',
    'checkpoint_pose_untrusted': '当前位姿不可信，请保持重叠并停稳后再检查。',
    'checkpoint_target_unbound': '本次扫描范围尚未绑定参考位置，请检查采集状态。',
    'checkpoint_target_reference_mismatch': '扫描范围与本次参考来源不一致，停止检查并核对数据。',
    'checkpoint_round_limit': '已达到阶段检查次数上限，请核对已有结果并决定结束或重新采集。',
    'checkpoint_no_new_views': '尚未接纳新的视角，上次阶段结果保留；先补采再检查。',
    'checkpoint_no_gain_limit': '连续两轮未观察到改善，已停止补采接纳；请核对原因并结束，或新建采集。',
    'checkpoint_cancelled': '阶段检查已取消，上次成功结果保留。',
    'checkpoint_supervisor_not_stopped': '后台阶段检查尚未结束，请等待并检查报告。',
}


@dataclass
class CaptureState:
    """Small UI state, independent of Qt widgets and SDK objects."""
    session_id: str | None = None
    active: bool = False
    finishing: bool = False
    cancelling: bool = False
    paused: bool = False
    state: str = 'waiting'
    count: int = 0
    source: Path | None = None
    result: Path | None = None
    record: dict = field(default_factory=dict)
    stability: dict = field(default_factory=dict)
    quality: dict | None = None
    failure: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    notice: str = ''
    notice_until: float = 0.
    # ``LocalCapture`` emits all events from one worker, but the Qt mailbox can
    # be drained after a session has already advanced.  Keep the last pose
    # update so a delayed event from the same session cannot move the UI back
    # to an older frame/count.
    last_pose_update: float | None = None
    last_pose_frame_id: int | None = None
    checkpoint_busy: bool = False
    checkpoint_id: str | None = None
    checkpoint_result: dict | None = None
    checkpoint_error: str = ''
    acquisition_completion: dict | None = None
    checkpoint_stop_reason: str | None = None
    finish_reason: str | None = None
    region_reference: object | None = None

    def begin(self):
        fresh = CaptureState(session_id=uuid.uuid4().hex, active=True)
        self.__dict__.update(fresh.__dict__)

    def fresh_pose(self, now=None):
        now = time.monotonic() if now is None else now
        received = self.record.get('received_monotonic')
        if received is None:
            return False
        try:
            age = now - float(received)
        except (TypeError, ValueError):
            return False
        return (self.active and not self.finishing and self.state == 'tracking'
                and self.record.get('state') == 'tracking' and 0 <= age <= 2.
                and self.record.get('local_workspace', {}).get('within_limit', True))

    def accept(self, kind, payload):
        """Ignore stale sessions before touching any current-session data."""
        if (not self.active or not isinstance(payload, dict) or not self.session_id
                or payload.get('session_id') != self.session_id):
            return False
        source = payload.get('source')
        source_path = None
        if source:
            try:
                source_path = Path(source)
            except (TypeError, ValueError):
                return False
        if kind.startswith('local_checkpoint_'):
            if kind == 'local_checkpoint_queued':
                self.checkpoint_busy = True
                self.checkpoint_error = ''
            elif kind == 'local_checkpoint_started':
                stage_id = payload.get('stage_id')
                if not isinstance(stage_id,str) or not stage_id:
                    return False
                self.checkpoint_id = stage_id
                self.checkpoint_busy = True
            elif kind == 'local_checkpoint_ready':
                stage = payload.get('checkpoint')
                if (payload.get('stage_id') != self.checkpoint_id or not isinstance(stage,dict)
                        or stage.get('stage_id') != self.checkpoint_id
                        or stage.get('session_id') != self.session_id or self.finishing):
                    return False
                self.checkpoint_result = stage
                self.checkpoint_busy = False
                self.checkpoint_stop_reason = payload.get('operation_stop_reason')
            elif kind == 'local_checkpoint_error':
                # Preflight rejection has no started event and clears a queued request.
                if (self.checkpoint_id and payload.get('stage_id') != self.checkpoint_id
                        and not self.checkpoint_busy):
                    return False
                self.checkpoint_busy = False
                self.checkpoint_error = str(payload.get('error','阶段检查失败'))
                self.checkpoint_stop_reason = payload.get('operation_stop_reason')
            elif kind == 'local_checkpoint_progress':
                if payload.get('stage_id') != self.checkpoint_id:
                    return False
            elif kind == 'local_checkpoint_cancelled':
                self.checkpoint_busy = False
                self.checkpoint_error = '阶段检查已取消，上次成功结果保留。'
            else:
                return False
            return True
        if kind == 'local_pose':
            if self.region_reference is not None:
                return False  # A frozen reference never resumes from a late live pose.
            record = payload.get('record')
            if not isinstance(record, dict) or not isinstance(record.get('state'), str):
                return False
            try:
                accepted = int(payload.get('accepted'))
            except (TypeError, ValueError):
                return False
            if accepted < 0 or accepted < self.count:
                return False
            # ``status_updated_monotonic`` is assigned by the tracker after
            # processing.  Tests and older records may only carry the receipt
            # clock, so use that as a fallback.  Missing clocks remain valid;
            # event order is still protected by the accepted-frame count.
            update_clock = record.get('status_updated_monotonic', record.get('received_monotonic'))
            try:
                update_clock = float(update_clock) if update_clock is not None else None
            except (TypeError, ValueError):
                update_clock = None
            if (update_clock is not None and self.last_pose_update is not None
                    and update_clock < self.last_pose_update):
                return False
            frame_id = record.get('frame_id')
            if (isinstance(frame_id, int) and self.last_pose_frame_id is not None
                    and frame_id < self.last_pose_frame_id):
                return False
            self.record = record
            self.count = accepted
            if update_clock is not None:
                self.last_pose_update = update_clock
            if isinstance(frame_id, int):
                self.last_pose_frame_id = frame_id
            self.state = self.record['state']
            self.stability = self.record.get('candidate_stability') or {}
            if self.record.get('selected_for_fusion'):
                self.notice = ''
        elif kind == 'local_stability':
            stability = payload.get('stability')
            if isinstance(stability, dict):
                self.stability = stability
            else:
                return False
        elif kind == 'local_pause':
            if 'paused' not in payload:
                return False
            self.paused = bool(payload['paused'])
        elif kind == 'local_notice':
            key = payload.get('key')
            if not isinstance(key, str):
                return False
            self.notice = key
            if key in ('checkpoint_round_limit','checkpoint_no_gain_limit'):
                self.checkpoint_stop_reason=key
            self.notice_until = time.monotonic() + 4.
        elif kind == 'local_finishing':
            self.finishing = True
            self.finish_reason = payload.get('reason')
        elif kind == 'local_region_ready':
            reference = payload.get('reference')
            if reference is None or reference.binding.get('session_id') != self.session_id:
                return False
            self.region_reference = reference
            self.state = 'awaiting_region'
            self.finishing = False
            self.notice = ''
        elif kind == 'local_processing':
            self.finishing = True
            self.state = 'processing'
        elif kind in ('local_finished', 'local_cancelled', 'local_error'):
            self.active = self.finishing = self.cancelling = False
            self.checkpoint_busy = False
            self.notice = ''
            self.state = ('ready' if kind == 'local_finished' else 'cancelled'
                          if kind == 'local_cancelled' else str(payload.get('error', 'lost')))
            if kind == 'local_error':
                failure = payload.get('failure')
                self.failure = (failure if isinstance(failure, dict)
                                else {'error': payload.get('error', 'lost')})
            else:
                self.failure = None
            if kind == 'local_finished':
                result = payload.get('result')
                if not result:
                    # A malformed completion event must not leave the page in
                    # a falsely successful state or crash the poll timer.
                    self.state = 'lost'
                    self.failure = {'error': 'missing_result'}
                else:
                    self.result = Path(result)
                    self.acquisition_completion = payload.get('acquisition_completion')
        if payload.get('quality') is not None:
            self.quality = payload['quality']
        # Publish the source only after the event payload itself has passed
        # validation, so a malformed or stale same-session message cannot
        # rewrite the current evidence path.
        if source_path is not None:
            self.source = source_path
        return True


class _CameraWorker(CameraWorker):
    def _cmd_local_start(self, state, payload):
        super()._cmd_local_start(state, payload)
        self.app.local_capture = state.local

    def _process_frame(self, state, frame, motion_executor):
        # Independent receipt clock: a frozen last frame cannot claim live readiness.
        with self.app.frame_lock:
            self.app.latest_received = time.monotonic()
        super()._process_frame(state, frame, motion_executor)


class CameraHost:
    """Widget-free adapter. Only CameraWorker.run owns and calls the camera SDK."""
    def __init__(self):
        self.event_queue = queue.Queue()
        self.command_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_received = 0.
        self.local_capture = None
        self.camera_thread = None
        self.photo_thread = None

    def start(self):
        if self.camera_thread and self.camera_thread.is_alive():
            return False
        if self.local_capture is not None and self.local_capture.thread.is_alive():
            return False
        # A prior worker's stopped/error events must never affect the new camera.
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break
        self.stop_event.clear()
        with self.frame_lock:
            self.latest_frame = None
            self.latest_received = 0.
        self.camera_thread = threading.Thread(target=self._run, name='FemtoQtCamera', daemon=True)
        self.camera_thread.start()
        return True

    def is_running(self):
        """Return whether commands can still reach the single SDK owner.

        ``camera_ready`` is deliberately a data-plane flag.  It can remain
        true for one Qt polling interval after a worker exits, while the last
        frame is still inside the freshness window.  Starting or finishing a
        session during that interval would enqueue a command that nobody can
        consume, leaving the page stuck in an active state.
        """
        thread = self.camera_thread
        return bool(thread is not None and thread.is_alive() and not self.stop_event.is_set())

    def _run(self):
        try:
            from femto.hardware.orbbec_rgbd import OrbbecAlignedCamera
            from femto.capture.scan_session import wait_for_aligned_frame, copy_frame, save_rgbd_photo
            dependencies = SimpleNamespace(
                OrbbecAlignedCamera=OrbbecAlignedCamera, APP_DEPTH_MODE='wide_detail',
                APP_COLOR_SIZE=(1920, 1080), DATASETS_ROOT=DATASETS_ROOT,
                OUTPUTS_ROOT=OUTPUTS_ROOT, ThreadPoolExecutor=ThreadPoolExecutor,
                wait_for_aligned_frame=wait_for_aligned_frame, traceback=traceback,
                threading=threading, time=time, copy_frame=copy_frame,
                save_rgbd_photo=save_rgbd_photo,
            )
            self.worker = _CameraWorker(self, dependencies)
            self.worker.run()
        except Exception as error:
            self.event_queue.put(('camera_error', f'{error}\n\n{traceback.format_exc(limit=5)}'))
            self.event_queue.put(('camera_stopped', None))

    def _photo_worker(self, frame, camera):
        self.worker.photo_worker(frame, camera)

    def stop(self, timeout_ms=1800):
        self.stop_event.set()
        self.command_queue.put(('shutdown', None))
        if self.local_capture is not None:
            self.local_capture.finish(False)
        deadline = time.monotonic() + timeout_ms / 1000
        threads = [self.camera_thread, self.photo_thread,
                   None if self.local_capture is None else self.local_capture.thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(max(0., deadline - time.monotonic()))
        return not any(thread and thread.is_alive() for thread in threads)


class CaptureViewport(QWidget):
    changed = Signal()
    roi_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('captureViewport')
        self.setMinimumSize(280, 260)
        self.image = QImage()
        self.caption = '等待相机画面'
        self.detail = '请确认 Femto Bolt 已连接'
        self.is_cloud = False
        self.zoom = 1.
        self.pan = np.zeros(2)
        self._drag = None
        self.roi_enabled = False
        self.roi_rect = None
        self._roi_draft = None
        self.setToolTip('XYZ 三维正视 / 正交投影 · 仅滚轮缩放')

    def set_image(self, value, *, cloud=False):
        self.image = value
        self.is_cloud = cloud
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor('#06101A'))
        if not self.image.isNull():
            size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            rect = QRectF((self.width()-size.width())/2, (self.height()-size.height())/2,
                          size.width(), size.height())
            painter.drawImage(rect, self.image)
            if not self.is_cloud:
                selected = self._roi_draft or self.roi_rect
                if selected is not None:
                    left, top, right, bottom = selected
                    outline = QRectF(rect.x()+left*rect.width(), rect.y()+top*rect.height(),
                                     (right-left)*rect.width(), (bottom-top)*rect.height())
                    painter.setPen(QPen(QColor('#26D6A2'), 3))
                    painter.setBrush(QColor(38, 214, 162, 38))
                    painter.drawRect(outline)
        else:
            draw_empty_viewport(painter, self.rect(), self.caption, self.detail)

    def wheelEvent(self, event):
        if self.is_cloud:
            self.zoom = float(np.clip(self.zoom * (1.12 if event.angleDelta().y() > 0 else 1/1.12), .25, 6.))
            self.changed.emit()
            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if self.roi_enabled and not self.is_cloud and not self.image.isNull() and event.button() == Qt.LeftButton:
            point = self._image_point(event.position())
            if point is not None:
                self._drag = point
                self._roi_draft = None
                event.accept()
                return
        # The model inspection plane is reference-locked.  Keep right-button
        # presses inert so the old pan gesture cannot shift the XYZ view.
        self._drag = None
        event.accept()

    def mouseMoveEvent(self, event):
        if self.roi_enabled and self._drag is not None:
            point = self._image_point(event.position(), clamp=True)
            if point is not None:
                self._roi_draft = (min(self._drag[0], point[0]), min(self._drag[1], point[1]),
                                   max(self._drag[0], point[0]), max(self._drag[1], point[1]))
                self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.roi_enabled and self._drag is not None and event.button() == Qt.LeftButton:
            point = self._image_point(event.position(), clamp=True)
            if point is not None:
                rect = (min(self._drag[0], point[0]), min(self._drag[1], point[1]),
                        max(self._drag[0], point[0]), max(self._drag[1], point[1]))
                self.roi_changed.emit(rect)
        self._drag = None
        self._roi_draft = None
        self.update()
        event.accept()

    def _image_point(self, point, *, clamp=False):
        if self.image.isNull():
            return None
        size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        left, top = (self.width()-size.width())/2, (self.height()-size.height())/2
        x, y = (point.x()-left)/size.width(), (point.y()-top)/size.height()
        if not clamp and not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        return float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.is_cloud:
            self.changed.emit()


def read_review(result, cancel, progress, *, prepared=None):
    """Bound display copies; authoritative meshes remain untouched on disk."""
    from femto.geometry.fusion import read_geometry_safe
    from femto.local.local_processing import result_quality
    result = Path(result)
    clouds = {}
    for key, name in [('initial', 'initial_observed_points_m.ply'), ('final', 'scene_fused_points_m.ply')]:
        if cancel.is_set():
            return None
        path = result / 'reconstruction' / name
        if not path.is_file() and key == 'initial':
            continue
        progress('正在读取' + ('初始点云' if key == 'initial' else '融合点云'))
        cloud = read_geometry_safe(path, mesh=False)
        stride = max(1, (len(cloud.points)+34999)//35000)
        points = np.asarray(cloud.points)[::stride].copy()
        colors = np.asarray(cloud.colors)[::stride].copy()
        if not len(points) or not np.isfinite(points).all():
            raise ValueError('模型点云为空或包含无效坐标')
        clouds[key] = (points, colors)
    # scene_fused_points_m.ply is the saved, voxel-downsampled accumulation
    # of accepted frames. It is the historical counterpart of the live-only
    # observed preview; the final view also overlays the TSDF surface.
    clouds['accumulated'] = clouds['final']
    if cancel.is_set():
        return None
    surface = read_review_surface(result)
    # The historical capture page has no live CaptureState from which to
    # recover the accepted-frame count.  Read the small reconstruction report
    # alongside the display assets so the replay card can retain that context.
    frame_count = None
    source_dataset = None
    report = result / 'reconstruction' / 'reconstruction_report.json'
    try:
        if report.is_file() and report.stat().st_size <= 10 * 1024 * 1024:
            metadata = json.loads(report.read_text(encoding='utf-8-sig'))
            frame_count = metadata.get('frame_count')
            scope=metadata.get('reconstruction_scope')
            if isinstance(scope,dict):frame_count=scope.get('input_frame_count',frame_count)
            source_dataset = metadata.get('source_dataset')
    except (OSError, ValueError, TypeError):
        pass
    from femto.dataset.acquisition_completion import read_completion
    from femto.annotation.fixed_rgb_geometry import GeometryBudget
    completion = read_completion(result,check=GeometryBudget(cancel).check)
    roi = None
    roi_quality = None
    roi_preview = None
    if source_dataset and not cancel.is_set():
        from femto.capture.scan_roi import read as read_scan_roi, assess_model
        try:
            roi = read_scan_roi(source_dataset)
            if roi is None:
                # The reconstruction input is a posed/refined derivative. Its
                # immutable tracking report identifies the original ROI owner.
                manifest_path = Path(source_dataset)/'manifest.json'
                manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
                tracking_path = (manifest.get('local_capture') or {}).get('tracking_report')
                if tracking_path:
                    roi = read_scan_roi(Path(tracking_path).parent.parent)
            if roi is None and completion is not None:
                # A verified saved completion carries its acquisition context.
                # An optional UI settings file need not exist to re-assess the
                # immutable painted target against this model.
                roi = completion.get('scan_roi')
            if roi is not None:
                from femto.annotation.fixed_rgb_geometry import FixedRGBModel
                from femto.dataset.prepared_review import PreparedCapture
                progress('正在检查扫描范围内的实测几何与参考深度')
                reused = (prepared.review(result, roi, GeometryBudget(cancel).check)
                          if isinstance(prepared, PreparedCapture) else None)
                if reused is None:
                    model = FixedRGBModel(result/'reconstruction', cancel=cancel)
                    roi_quality, roi_preview = assess_model(model, roi, cancel, preview=True)
                else:
                    roi_quality, roi_preview = reused
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            roi_quality = dict(status='unavailable', message=str(error))
    return dict(clouds=clouds, surface=surface, quality=result_quality(result),
                frame_count=frame_count, source_dataset=source_dataset,
                scan_roi=roi, scan_roi_quality=roi_quality, scan_roi_preview=roi_preview,
                acquisition_completion=completion)


class CapturePage(QWidget):
    planning_requested = Signal(str)
    planning_preview_requested = Signal(str)
    history_requested = Signal()
    status_changed = Signal(str)

    def __init__(self, parent=None, *, connect_camera=True):
        super().__init__(parent)
        self.setObjectName('capturePage')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.session = CaptureState()
        from femto.dataset.task_context import new_task
        self.task_context=new_task()
        self.host = CameraHost()
        self.camera_ready = False
        self.camera_error = ''
        self.camera_intrinsics = None
        self.camera_rgb_distortion = None
        self.device_info = {}
        self.preview_mode = not connect_camera
        self.view = 'preview'
        self.clouds = {}
        self.surface = None
        self.preview_snapshot = None
        self.preview_mailbox = None
        self.center = None
        self.span = 1.
        self.last_frame = None
        self._last_status = None
        self._render_serial = 0
        self._render_dirty = True
        self._closed = False
        self._history_replay = False
        self._auto_plan_result = None
        self._workflow_view = None
        self.scan_roi_rect = None
        self.scan_roi_size = None
        self.scan_roi_quality = None
        self.scan_roi_preview = None
        self.current_depth_check = None
        self.roi_message = ''
        # Result root currently being read for a read-only historical review.
        # Keeping this separately from ``session.result`` lets a user return
        # from the planning page while the bounded review read is still in
        # flight without starting a second task or showing an empty viewport.
        self._review_request_result = None
        self._prepared_review = None
        self.review_task = TaskRunner(self)
        self.render_task = TaskRunner(self)
        self._build_ui()
        self.review_task.progress.connect(self.preview_note.setText)
        self.render_task.progress.connect(self.preview_note.setText)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.timer.start()
        self.refresh()
        if connect_camera:
            QTimer.singleShot(0, self.connect_camera)

    @staticmethod
    def _button(text, callback, parent=None):
        button = QPushButton(text, parent)
        button.setMinimumHeight(42)
        button.clicked.connect(callback)
        return button

    def _build_ui(self) -> None:
        from femto.app.qt_workflow import WorkflowProgress
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.workflow = WorkflowProgress()
        outer.addWidget(self.workflow)
        body = QHBoxLayout()
        body.setSpacing(10)
        outer.addLayout(body, 1)
        center = QVBoxLayout()
        center.setContentsMargins(16, 8, 16, 8)
        center.setSpacing(8)
        self._build_capture_views(center)
        self.alert = QLabel()
        self.alert.setObjectName('captureAlert')
        self.alert.setWordWrap(True)
        self.alert.hide()
        center.addWidget(self.alert)
        self.region_progress = QProgressBar()
        self.region_progress.setRange(0, 0)
        self.region_progress.setTextVisible(False)
        self.region_progress.setMaximumHeight(6)
        self.region_progress.setObjectName('captureProgress')
        self.region_progress.setAccessibleName('正在重建选中区域')
        self.region_progress.hide()
        center.addWidget(self.region_progress)
        self.viewport = CaptureViewport()
        self.viewport.changed.connect(self.invalidate_render)
        self.viewport.roi_changed.connect(self._roi_changed)
        center.addWidget(self.viewport, 1)
        from femto.app.qt_region_capture import CaptureRegionEditor
        self.region_editor = CaptureRegionEditor()
        self.region_editor.changed.connect(self.refresh)
        center.addWidget(self.region_editor,1)
        self.region_editor.hide()
        self.preview_note = QLabel('真实 RGB-D 采集 · 研发规划预览')
        self.preview_note.setObjectName('statusTelemetry')
        self.preview_note.setWordWrap(True)
        center.addWidget(self.preview_note)
        body.addLayout(center, 1)
        self._build_capture_inspector()
        # Live progress and historical metrics share the viewport toolbar;
        # capture does not reserve a sidebar beside the reference image.
        self.history_toolbar.insertWidget(0, self.history_metrics)
        self._build_capture_actions(outer)

    def _build_capture_views(self, center: QVBoxLayout) -> None:
        from femto.app.qt_icons import line_icon
        view_bar = QFrame()
        self.capture_view_bar = view_bar
        view_bar.setObjectName('viewBar')
        toolbar = QHBoxLayout(view_bar)
        self.capture_toolbar = toolbar
        toolbar.setContentsMargins(6, 4, 6, 4)
        toolbar.setSpacing(6)
        self.views = {}
        for mode, title in [('preview', 'RGB'), ('depth', '深度'),
                            ('frame_cloud', '实时点云'), ('3d', '三维模型'), ('scan_roi', '缺口检查')]:
            button = self._button(title, lambda checked=False, m=mode: self.set_view(m))
            button.setObjectName('viewButton')
            button.setCheckable(True)
            button.setMinimumWidth(64)
            if mode == 'frame_cloud':
                button.setToolTip('当前对齐 RGB-D 帧生成的单帧点云；相机坐标系，未配准、未累计')
            self.views[mode] = button
            if mode in ('3d', 'scan_roi'):
                button.setIcon(line_icon('cube' if mode == '3d' else 'warning'))
            toolbar.addWidget(button)
        self.cloud_choice = QComboBox()
        self.cloud_choice.setObjectName('cloudChoice')
        self.cloud_choice.addItem('累计点云', 'live')
        self.cloud_choice.addItem('初始点云', 'initial')
        self.cloud_choice.addItem('最终模型', 'final')
        self.cloud_choice.setMinimumHeight(42)
        self.cloud_choice.currentIndexChanged.connect(self._cloud_selected)
        toolbar.addWidget(self.cloud_choice)
        self.history_rescan = self._button('重新采集', self.rescan_current_region)
        self.history_rescan.setIcon(line_icon('refresh'))
        toolbar.addWidget(self.history_rescan)
        self.history_rescan.hide()
        toolbar.addStretch()
        self.history_toolbar = QHBoxLayout()
        toolbar.addLayout(self.history_toolbar)
        self.connection = self._button('连接相机', self.connect_camera)
        toolbar.addWidget(self.connection)
        self.fit_button = self._button('复位视图', self.fit_view)
        self.fit_button.setIcon(line_icon('target'))
        toolbar.addWidget(self.fit_button)
        center.addWidget(view_bar)

    def _build_capture_inspector(self) -> None:
        inspector = QFrame()
        inspector.setObjectName('rightPanel')
        inspector.setMinimumWidth(280)
        info = QVBoxLayout(inspector)
        info.setContentsMargins(14, 12, 14, 12)
        info.setSpacing(10)
        self.inspector_heading = QLabel('局部采集')
        self.inspector_heading.setObjectName('inspectorHeading')
        info.addWidget(self.inspector_heading)
        self.workflow_view_notice = QLabel('已结束采集，仅回看本次记录。修改扫描范围请新建采集。')
        self.workflow_view_notice.setWordWrap(True)
        self.workflow_view_notice.setProperty('role', 'secondary')
        self.workflow_view_notice.hide()
        info.addWidget(self.workflow_view_notice)
        self.task_name=QLineEdit(self.task_context['name']);self.task_name.setMaxLength(120)
        self.task_name.setPlaceholderText('任务名称');self.task_name.setAccessibleName('任务名称')
        self.scan_roi_button = self._button('在 RGB 画面框选扫描范围', self.choose_scan_roi)
        info.addWidget(self.scan_roi_button)
        self.scan_roi_status = QLabel('扫描范围待确认')
        self.scan_roi_status.setWordWrap(True)
        self.scan_roi_status.setProperty('role', 'secondary')
        info.addWidget(self.scan_roi_status)
        self.history_metrics = QFrame()
        self.history_metrics.setObjectName('historyMetricPanel')
        metrics = QHBoxLayout(self.history_metrics)
        metrics.setContentsMargins(14, 8, 14, 8)
        metrics.setSpacing(16)
        self.history_metric_labels = [QLabel() for _ in range(3)]
        for index, (label, title) in enumerate(zip(self.history_metric_labels,
                                ('支持像素 / 总数', '待检查像素', '采集帧数'))):
            if index:
                separator = QFrame()
                separator.setObjectName('historyMetricSeparator')
                separator.setFixedWidth(1)
                metrics.addWidget(separator)
            column = QVBoxLayout()
            column.setSpacing(3)
            label.setObjectName('historyMetricNumber')
            column.addWidget(label)
            caption = QLabel(title)
            caption.setObjectName('historyMetricCaption')
            column.addWidget(caption)
            metrics.addLayout(column)
        self.history_metrics.hide()
        info.addWidget(self.history_metrics)
        self.scan_roi_button.setToolTip('扫描范围包含目标及所需周边；完成采集后先涂抹，再局部重建。')
        self.rescan_button = self._button('重新采集', self.rescan_current_region)
        info.addWidget(self.rescan_button)
        self.task_name.textEdited.connect(self.status_changed)
        state_card = QFrame()
        state_card.setObjectName('captureState')
        status = QVBoxLayout(state_card)
        status.setContentsMargins(0, 0, 0, 0)
        status.setSpacing(6)
        self.state_title = QLabel('等待连接')
        self.state_title.setObjectName('captureStateTitle')
        self.state_title.setWordWrap(True)
        status.addWidget(self.state_title)
        self.device_label = QLabel()
        self.device_label.setWordWrap(True)
        self.device_label.setProperty('role', 'secondary')
        info.addWidget(state_card)
        self.capture_heading = QLabel('本次采集')
        self.capture_heading.setObjectName('sectionHeading')
        info.addWidget(self.capture_heading)
        self.count_label = QLabel('0 帧')
        self.count_label.setObjectName('captureFrameCount')
        self.count_label.setProperty('tone', 'good')
        info.addWidget(self.count_label)
        self.count_label.setToolTip('最多接纳 3 个有效视角，含首帧；接纳一帧后也可提前完成。帧数不代表覆盖合格。')
        self.stability_bar = QProgressBar()
        self.stability_bar.setRange(0, 5)
        self.stability_bar.setFormat('候选帧 %v / %m')
        self.stability_bar.setMinimumHeight(24)
        self.stability_bar.setObjectName('captureProgress')
        info.addWidget(self.stability_bar)
        self.pose_label = QLabel('参考位置未建立')
        self.pose_label.setWordWrap(True)
        info.addWidget(self.pose_label)
        self.highlight = QCheckBox('高亮新增采样点')
        self.highlight.setChecked(True)
        self.highlight.setMinimumHeight(40)
        self.highlight.toggled.connect(self.invalidate_render)
        info.addWidget(self.highlight)
        self.pause_button = self._button('暂停补采', self.pause)
        self.check_button = self._button('检查当前采集', self.check_current)
        self.stage_review_button = self._button('查看上次阶段检查', self.show_checkpoint)
        self.checkpoint_hint = QLabel('阶段检查保留本次参考位置，可继续补采。')
        self.checkpoint_hint.setWordWrap(True)
        self.checkpoint_hint.setProperty('role','secondary')
        info.addWidget(self.check_button)
        info.addWidget(self.stage_review_button)
        info.addWidget(self.checkpoint_hint)
        self.finish_button = self._button('完成采集并选区', lambda: self.finish(True))
        self.cancel_button = self._button('取消本次采集', lambda: self.finish(False))
        self.cancel_button.setObjectName('quietButton')
        info.addWidget(self.task_name)
        info.addWidget(self.device_label)
        info.addStretch()
        self.quality_button = self._button('检查采集质量', self.show_quality)
        self.report_button = self._button('查看本次报告', self.open_report)
        self.quality_button.setObjectName('quietButton')
        self.report_button.setObjectName('quietButton')
        info.addWidget(self.quality_button)
        info.addWidget(self.report_button)
        # Keep legacy state widgets owned but outside the visible layout.
        # New captures use an automatic task name and the full live frame.
        scroll = QScrollArea(self)
        scroll.setObjectName('captureInspector')
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(inspector)
        scroll.setMinimumWidth(292)
        self.inspector_scroll = scroll
        scroll.hide()
        self.frame_progress = QFrame()
        self.frame_progress.setObjectName('captureFrameProgress')
        progress = QHBoxLayout(self.frame_progress)
        progress.setContentsMargins(10, 0, 10, 0);progress.setSpacing(8)
        progress.addWidget(self.capture_heading);progress.addWidget(self.count_label)
        self.frame_steps = []
        for _ in range(MAX_ACCEPTED_VIEWS):
            segment = QFrame();segment.setObjectName('captureFrameSegment')
            segment.setFixedSize(36, 8);progress.addWidget(segment)
            self.frame_steps.append(segment)
        self.frame_progress.setToolTip('有效视角进度：第三帧保存成功后自动进入选区；也可在一帧或两帧时提前完成。')
        self.capture_toolbar.insertWidget(self.capture_toolbar.indexOf(self.connection), self.frame_progress)

    def _build_capture_actions(self, outer: QVBoxLayout) -> None:
        footer = QFrame()
        self.action_bar = footer
        footer.setObjectName('actionBar')
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(18, 8, 18, 8)
        self.guidance = QLabel()
        self.guidance.setObjectName('actionNotice')
        self.guidance.setWordWrap(True)
        footer_layout.addWidget(self.guidance)
        actions = QHBoxLayout()
        footer_layout.addLayout(actions)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.pause_button)
        self.new_button = self._button('新建采集', self.new_capture)
        actions.addWidget(self.new_button)
        self.retry_button = self._button('重试加载模型', self.retry_review)
        actions.addWidget(self.retry_button)
        self.selection_footer_note = QLabel('固定首帧 RGB · 尚未生成机械臂轨迹')
        self.selection_footer_note.setObjectName('regionFooterNote')
        self.selection_footer_note.setAlignment(Qt.AlignCenter)
        self.selection_footer_note.setWordWrap(True)
        self.selection_footer_note.setToolTip('原始采集帧及位姿保留')
        self.selection_footer_note.hide()
        actions.addWidget(self.selection_footer_note,1)
        actions.addStretch()
        self.finish_button.setMinimumHeight(52)
        actions.addWidget(self.finish_button)
        self.prepare_action = self._button('连接相机', self.prepare_next)
        self.prepare_action.setProperty('role', 'primary')
        self.prepare_action.setMinimumWidth(250)
        self.prepare_action.setAccessibleName('采集准备下一步')
        actions.addWidget(self.prepare_action)
        self.primary = self._button('建立参考位置并开始采集', self.primary_action)
        self.primary.setProperty('role', 'primary')
        self.primary.setMinimumHeight(52)
        self.primary.setMinimumWidth(285)
        self.primary.setAccessibleName('采集阶段主操作')
        self.primary.setShortcut('Ctrl+Return')
        self.primary.setToolTip('快捷键：Ctrl+Enter')
        actions.addWidget(self.primary)
        outer.addWidget(footer)

    def prepare_next(self):
        """Prepare camera and scan ROI directly on the acquisition page."""
        if self.session.active or self.session.result or self.review_task.busy:
            return
        if not self.live_frame_valid():
            self.connect_camera()
        else:
            self.start()

    def _clear_scan_roi(self):
        self.scan_roi_rect = self.scan_roi_size = self.scan_roi_quality = self.scan_roi_preview = None
        self.current_depth_check = None
        self.roi_message = ''
        self.viewport.roi_enabled = False
        self.viewport.roi_rect = None
        self.viewport.update()

    def _depth_limits(self):
        value = self.device_info.get('depth_range_m')
        if (not isinstance(value, (tuple, list)) or len(value) != 2
                or not np.isfinite(value).all() or not 0 < value[0] < value[1]):
            raise ValueError('相机深度范围未就绪')
        return float(value[0])*1000, float(value[1])*1000

    def choose_scan_roi(self):
        if self.session.active or self.session.result or self.review_task.busy:
            return
        if not self.live_frame_valid():
            self.roi_message = '请先连接相机，等待实时 RGB-D 画面'
            self.refresh()
            return
        self.set_view('preview')
        self.update_live_image()
        self.viewport.roi_enabled = True
        self.roi_message = '请在当前 RGB 画面中拖动鼠标，框出需要扫描的范围。'
        self.refresh()

    def _roi_changed(self, rect):
        if not self.viewport.roi_enabled or not self.live_frame_valid():
            return
        from femto.capture.scan_roi import pixel_bounds
        with self.host.frame_lock:
            frame = self.host.latest_frame
        depth = np.asarray(frame.depth_mm)
        size = [int(depth.shape[1]), int(depth.shape[0])]
        try:
            pixel_bounds(rect, size)
        except ValueError as error:
            self.roi_message = str(error)
            self.refresh()
            return
        self.scan_roi_rect = list(rect)
        self.scan_roi_size = size
        self.scan_roi_quality = None
        self.viewport.roi_rect = self.scan_roi_rect
        self.viewport.roi_enabled = False
        self.roi_message = ''
        self._refresh_current_depth(frame)
        self.refresh()

    def _refresh_current_depth(self, frame):
        if self.scan_roi_rect is None:
            return
        from femto.capture.scan_roi import depth_check
        try:
            depth = np.asarray(frame.depth_mm)
            if [depth.shape[1], depth.shape[0]] != self.scan_roi_size:
                raise ValueError('画面尺寸已改变，请重新确认扫描范围')
            self.current_depth_check = depth_check(depth, self.scan_roi_rect, *self._depth_limits())
            self.roi_message = ''
        except (ValueError, IndexError) as error:
            self.current_depth_check = None
            self.roi_message = str(error)

    def rescan_current_region(self):
        if self.session.active or self.review_task.busy or not self.session.result:
            return
        from femto.dataset.task_context import checked_task
        task = checked_task(self.task_context)
        if task:
            task.pop('gap_followup', None)
        self.new_capture(task=task)
        self.set_view('preview')

    def new_capture(self, *, task=None):
        if self.session.active or self.review_task.busy:
            return
        self._discard_prepared_review()
        from femto.dataset.task_context import new_task, checked_task
        context = checked_task(task) if task else new_task()
        self.session = CaptureState()
        self._workflow_view = None
        self.task_context=context
        self._clear_scan_roi()
        self.task_name.setText(self.task_context['name'])
        self._history_replay = False
        self._review_request_result = None
        self.clouds = {}
        self.surface = self.center = None
        self.preview_mailbox = self.preview_snapshot = None
        self._configure_accumulated_choice()
        self.set_view('preview')
        self.invalidate_render()
        self.refresh()

    def retry_review(self):
        if self.session.result and not self.session.active and not self.review_task.busy:
            self.load_review()
            self.refresh()

    def connect_camera(self):
        if self.session.active:
            return
        if self.host.start():
            self.preview_mode = False
            self.camera_ready = False
            self.camera_error = ''
            self.viewport.caption = '正在连接相机'
            self.viewport.detail = '等待同步 RGB-D 数据'
            self.refresh()

    def send(self, command, value=None, **extra):
        if command.startswith('local_') and not self.host.is_running():
            return False
        payload = dict(session_id=self.session.session_id, value=value)
        payload.update(extra)
        if command == 'local_start':
            payload['cancel_event'] = self.session.cancel_event
        self.host.command_queue.put((command, payload))
        return True

    def start(self):
        if self.session.active or self.session.result or self.review_task.busy or not self.live_frame_valid():
            return
        from femto.capture.scan_roi import prepare
        from femto.capture.scan_session import copy_frame
        with self.host.frame_lock:
            latest = self.host.latest_frame
            initial = (copy_frame(latest, include_native_depth=False)
                       if latest is not None else None)
        try:
            if initial is None:
                raise ValueError('相机画面已过期，请重新取景')
            # Capture full raw RGB-D by default. The actual target is painted
            # only on the locked reference after acquisition has finished.
            rect = self.scan_roi_rect if self.scan_roi_rect is not None else [0.,0.,1.,1.]
            scan_roi = prepare(rect, initial, *self._depth_limits())
            if self.scan_roi_rect is not None and scan_roi['image_size'] != self.scan_roi_size:
                raise ValueError('画面尺寸已改变，请重新确认扫描范围')
        except ValueError as error:
            self.roi_message = str(error)
            self.refresh()
            return
        self.task_context.update(name=self.task_name.text().strip() or self.task_context['name'])
        self.task_name.setText(self.task_context['name'])
        self._history_replay = False
        self._configure_accumulated_choice()
        self.session.begin()
        self._review_request_result = None
        self.clouds = {}
        self.surface = None
        self.center = None
        self.preview_mailbox = self.preview_snapshot = None
        self.viewport.zoom = 1.
        self.viewport.pan[:] = 0
        self.invalidate_render()
        self.set_view('preview')
        if not self.send('local_start', initial_frame=initial, scan_roi=scan_roi, region_first=True):
            # A camera can terminate between ``live_frame_valid`` and queue
            # insertion.  Do not leave a retry stuck in an active session
            # whose start command can never be consumed.
            self.session.active = False
            self.session.state = 'camera_error'
            self.session.failure = {'error': 'camera_not_ready'}
            self.session.notice = 'camera_not_ready'
            self.session.notice_until = time.monotonic() + 4
        self.refresh()

    def request_view(self):
        if (self.session.active and not self.session.finishing and self.session.count < MAX_ACCEPTED_VIEWS
                and self.session.fresh_pose() and not self.session.paused and self.live_frame_valid()
                and not self.session.checkpoint_stop_reason):
            if not self.send('local_request'):
                self.session.notice = 'camera_not_ready'
                self.session.notice_until = time.monotonic() + 4
                self.refresh()

    def pause(self):
        if not self.session.active or self.session.finishing or self.session.state in ('lost', 'tracking_lost'):
            return
        self.session.paused = not self.session.paused
        if not self.send('local_pause', self.session.paused):
            self.session.paused = not self.session.paused
            self.session.notice = 'camera_not_ready'
            self.session.notice_until = time.monotonic() + 4
        self.refresh()

    def finish(self, reconstruct=True):
        state = self.session
        if not state.active or state.cancelling or (reconstruct and state.finishing):
            return
        if reconstruct and state.paused:
            state.notice = 'paused_finish'
            state.notice_until = time.monotonic() + 4
            self.refresh()
            return
        if reconstruct and (state.count < 1 or not state.fresh_pose() or not self.live_frame_valid()):
            if state.count < 1:
                state.notice = 'need_more_views'
            elif not state.fresh_pose():
                state.notice = 'uncertain'
            else:
                state.notice = 'camera_not_ready'
            state.notice_until = time.monotonic() + 4
            self.refresh()
            return
        state.finishing = True
        state.cancelling = not reconstruct
        if not reconstruct:
            state.cancel_event.set()
        if not self.send('local_finish', reconstruct):
            if reconstruct:
                state.finishing = False
                state.cancelling = False
            else:
                # The camera owner can disappear between the liveness check
                # and queue insertion.  Ask the local worker directly when it
                # still exists; otherwise close this orphaned UI session.
                local = self.host.local_capture
                local_thread = getattr(local, 'thread', None)
                if local is not None and local_thread is not None and local_thread.is_alive():
                    local.finish(False)
                else:
                    state.active = state.finishing = state.cancelling = False
                    state.state = 'camera_error'
                    state.failure = {'error': 'camera_not_ready'}
            state.notice = 'camera_not_ready'
            state.notice_until = time.monotonic() + 4
        self.refresh()

    def primary_action(self):
        if self.session.state == 'awaiting_region':
            local = self.host.local_capture
            mask = self.region_editor.canvas.work
            if local is not None and mask is not None and mask.any():
                if local.select_region(mask,self.region_editor.margin.value()):
                    self.session.finishing = True
                    self.session.state = 'processing'
                    self.region_editor.setEnabled(False)
                else:
                    self.region_editor.note.setText('选区尚未提交：采集会话未就绪，请稍后重试。当前涂抹已保留。')
            elif local is None:
                self.region_editor.note.setText('选区未提交：采集会话已不可用，请重新采集。当前涂抹已保留。')
            self.refresh()
            return
        if self.session.result and self._workflow_view == 1 and not self.has_reconstruction_selection():
            if self.review_ready():
                self.workflow.step_requested.emit(self._workflow_view + 1)
            return
        if self.session.result and not self.session.active:
            if self.review_ready():
                if self.has_reconstruction_selection():
                    self.planning_preview_requested.emit(str(self.session.result / 'reconstruction'))
                elif self.session.acquisition_completion is not None:
                    self.confirm_final_for_planning()
                else:
                    self.planning_requested.emit(str(self.session.result / 'reconstruction'))
        elif self.session.active:
            self.request_view()
        else:
            self.start()

    def has_reconstruction_selection(self):
        return bool((self.session.acquisition_completion or {}).get('reconstruction_scope'))

    def can_rebuild_region(self):
        local = self.host.local_capture
        return bool(local is not None and getattr(local,'region_first',False)
            and getattr(local,'acquisition_complete',False) and not local.thread.is_alive()
            and self.session.result and self.session.session_id == local.session_id
            and not self.session.active and not self.review_task.busy and not self._history_replay)

    def rebuild_region(self):
        if not self.can_rebuild_region():
            return False
        local = self.host.local_capture
        if not local.restart_region():
            return False
        self.session = CaptureState(session_id=local.session_id,active=True,count=len(local.accepted),
            source=local.session,state='processing',finishing=True,cancel_event=local.processing_cancel)
        self._workflow_view = None
        self.clouds = {};self.surface = None
        self.refresh()
        return True

    def check_current(self):
        state = self.session
        if state.checkpoint_busy:
            self.send('local_check_cancel')
        elif state.active and not state.finishing and state.fresh_pose() and state.count>=3:
            # Only the backend's queued event claims acceptance. A rejected
            # request during thread teardown must not leave the UI busy forever.
            self.send('local_check')
        self.refresh()

    def show_checkpoint(self):
        if self.session.checkpoint_result:
            from femto.app.qt_scan_checkpoint import show_checkpoint
            show_checkpoint(self,self.session.checkpoint_result)

    def confirm_final_for_planning(self):
        from femto.dataset.acquisition_completion import confirm_for_planning
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        session_id,result=self.session.session_id,self.session.result
        expected=dict(self.session.acquisition_completion['binding'])
        def work(cancel,progress):
            progress('正在核对最终模型版本并保存确认记录')
            return confirm_for_planning(result,expected,check=GeometryBudget(cancel).check)
        def ready(record):
            if session_id==self.session.session_id and result==self.session.result:
                self.session.acquisition_completion=record
                self.planning_requested.emit(str(result/'reconstruction'))
                self.refresh()
        def failed(error):
            self.preview_note.setText('最终模型未确认：'+str(error))
            self.refresh()
        self.review_task.start(work,ready,failed)
        self.refresh()

    def review_ready(self):
        """Only a successfully loaded, nonempty review can enter planning."""
        final = self.clouds.get('final')
        return bool(self.session.result and not self.session.active
                    and self.session.state == 'ready' and not self.review_task.busy
                    and final is not None and len(final[0]) > 0)

    def live_frame_valid(self):
        with self.host.frame_lock:
            frame, received = self.host.latest_frame, self.host.latest_received
        # The last frame may still be fresh after the SDK owner has exited.
        # Treat that state as disconnected so a retry cannot enqueue
        # ``local_start`` into a queue with no consumer.
        return bool(self.camera_ready and self.host.is_running() and frame is not None
                    and 0 <= time.monotonic()-received <= 2)

    def handle_event(self, kind, payload):
        if kind.startswith('local_'):
            if not self.session.accept(kind, payload):
                return
            if kind == 'local_started':
                self.preview_mailbox = payload.get('preview_mailbox')
            elif kind == 'local_region_ready':
                self.region_editor.install(payload['reference'],payload.get('error',''))
            elif kind == 'local_finished':
                self._history_replay = False
                self._auto_plan_result = self.session.result if self.has_reconstruction_selection() else None
                self.load_review()
            self.invalidate_render()
        elif kind == 'camera_ready':
            self.camera_ready = True
            self.camera_error = ''
            self.device_info = dict(payload)
            self.camera_intrinsics = self.device_info.pop('_intrinsics', None)
            self.camera_rgb_distortion = self.device_info.pop('_rgb_distortion', None)
            with self.host.frame_lock:
                if self.host.latest_frame is not None:
                    self.host.latest_received = time.monotonic()
        elif kind == 'camera_error':
            self.camera_ready = False
            self.camera_error = str(payload)
            self._abort_active_capture_for_camera_failure()
        elif kind == 'camera_stopped':
            self.camera_ready = False
            self._abort_active_capture_for_camera_failure()
        self.refresh()

    def _abort_active_capture_for_camera_failure(self):
        """Close the UI side when the SDK owner dies before local events arrive."""
        state = self.session
        if not state.active or state.finishing or state.region_reference is not None:
            return
        state.cancel_event.set()
        local = self.host.local_capture
        local_thread = getattr(local, 'thread', None)
        if local is not None and local_thread is not None and local_thread.is_alive():
            # Let LocalCapture finish its raw prefix and publish the normal
            # cancellation/error event.  The UI remains active until that
            # event so the session cannot be restarted over a live writer.
            state.finishing = True
            state.cancelling = True
            local.finish(False)
            return
        # The start command may never have been consumed.  There is no worker
        # left that can emit a terminal event, so fail closed and make retry
        # available once the camera is connected again.
        state.active = state.finishing = state.cancelling = False
        state.state = 'camera_error'
        state.failure = {'error': 'camera_not_ready'}
        state.notice = 'camera_not_ready'
        state.notice_until = time.monotonic() + 4

    def poll(self):
        if self._closed:
            return
        for _ in range(160):
            try:
                kind, payload = self.host.event_queue.get_nowait()
            except queue.Empty:
                break
            self.handle_event(kind, payload)
        if self.preview_mailbox is not None:
            try:
                data = self.preview_mailbox.get_nowait()
            except queue.Empty:
                data = None
            if data and data.get('session_id') == self.session.session_id:
                if data.get('status') == 'error':
                    self.preview_note.setText('累计预览不可用；原始采集与跟踪继续。')
                else:
                    self.preview_snapshot = data
                    self.clouds['live'] = (data['points'], data['colors'].astype(float)/255)
                    if self.session.result is None:
                        self.clouds['initial'] = (data['initial_points'], data['initial_colors'].astype(float)/255)
                    if self.center is None and self.view == 'frame_cloud' and len(data['points']):
                        points = np.asarray(data['points'])
                        self.center = (points.min(axis=0) + points.max(axis=0)) * .5
                        self.span = max(float(np.ptp(points[:, :2], axis=0).max()), .01)
                    elif self.center is None:
                        self.fit_view()
                    self.invalidate_render()
        if self.view in ('preview', 'depth'):
            self.update_live_image()
        elif self.view == 'scan_roi':
            pass
        elif self.view == 'frame_cloud':
            if not self.live_frame_valid():
                if self.last_frame is not None or not self.viewport.image.isNull():
                    self.last_frame = None
                    self.viewport.set_image(QImage(), cloud=True)
            else:
                with self.host.frame_lock:
                    frame = self.host.latest_frame
                if frame is not self.last_frame and not self.render_task.busy:
                    self.last_frame = frame
                    self.invalidate_render()
                if self._render_dirty and not self.render_task.busy:
                    self.render_cloud()
        elif self._render_dirty and not self.render_task.busy:
            self.render_cloud()
        self.refresh()

    def update_live_image(self):
        with self.host.frame_lock:
            frame = self.host.latest_frame
        if frame is None or frame is self.last_frame:
            return
        self.last_frame = frame
        color = np.asarray(frame.color_bgr)
        depth = np.asarray(frame.depth_mm)
        if color.ndim != 3 or color.shape[2] != 3 or depth.ndim != 2:
            self.camera_error = '相机帧格式无效，请重新连接。'
            self.camera_ready = False
            return
        if self.scan_roi_rect is not None and self.session.result is None:
            self._refresh_current_depth(frame)
        if self.view == 'depth':
            import cv2
            valid = np.isfinite(depth) & (depth >= 250) & (depth <= 2210)
            normalized = np.clip((np.nan_to_num(depth)-250)*255/1960, 0, 255).astype(np.uint8)
            color = cv2.applyColorMap(255-normalized, cv2.COLORMAP_TURBO)
            color[~valid] = (26, 16, 6)
        self.viewport.set_image(rgb_image(np.ascontiguousarray(color[..., ::-1])))

    def set_view(self, mode):
        if mode == 'frame_cloud' and self._history_replay and self.session.result:
            return
        if mode == 'scan_roi' and self.scan_roi_preview is None:
            return
        self.view = self.cloud_choice.currentData() if mode == '3d' else mode
        self.viewport.is_cloud = self.view not in ('preview', 'depth', 'scan_roi')
        self.viewport.roi_rect = (self.scan_roi_rect if self.view in ('preview', 'depth') else None)
        if self.view != 'preview':
            self.viewport.roi_enabled = False
        self.last_frame = None
        self.viewport.image = QImage()
        if self.view == 'scan_roi':
            self.viewport.caption = '扫描范围检查'
            self.viewport.detail = '首帧参考图：绿为可靠几何，紫/橙为待检查或缺失'
            self.viewport.set_image(self.scan_roi_preview, cloud=False)
            self.preview_note.setText('首帧参考图与当前模型的几何支持检查；像素比例不代表治疗覆盖。')
        elif self.view == 'frame_cloud':
            self.viewport.caption = '等待实时点云'
            self.viewport.detail = '连接相机后显示当前 RGB-D 帧；无需建立参考位置'
            self.preview_note.setText('实时单帧点云 · 相机坐标 C · 未配准、未累计；仅供预览')
        elif self.viewport.is_cloud:
            self.viewport.caption = '等待三维采集数据'
            self.viewport.detail = '建立参考位置后显示真实采样点云'
            if self._history_replay:
                self.preview_note.setText({
                    'accumulated': '历史累计点云 · 已保存的融合点云，只读',
                    'initial': '历史初始点云 · 第一视角采样，只读',
                    'final': '历史最终模型 · 融合点云与实测表面，只读',
                }.get(self.view, '历史三维数据 · 只读'))
            else:
                self.preview_note.setText({
                    'live': '本次采集累计点云 · 已接纳采样；参考坐标 W0',
                    'initial': '本次采集初始点云 · 参考坐标 W0',
                    'final': '本次采集最终模型 · 参考坐标 W0',
                }.get(self.view, '本次采集三维数据'))
        elif self.view in ('preview', 'depth'):
            self.viewport.caption = '等待相机画面'
            self.viewport.detail = '请确认 Femto Bolt 已连接'
            self.preview_note.setText(('当前相机 RGB' if self.view == 'preview' else '当前相机对齐深度')
                + (' · 与历史记录分开' if self._history_replay else ' · 研发采集预览'))
        self.invalidate_render()
        self.viewport.update()
        self.refresh()

    def _cloud_selected(self):
        if self.view not in ('preview', 'depth', 'frame_cloud', 'scan_roi'):
            self.set_view('3d')

    def _configure_accumulated_choice(self):
        """Keep the live preview and archived accumulation distinct in UI state."""
        if self._history_replay:
            self.cloud_choice.setItemData(0, 'accumulated')
            self.cloud_choice.setItemData(0, '历史重建结果中的累计融合点云；不是实时采集预览', Qt.ItemDataRole.ToolTipRole)
        else:
            self.cloud_choice.setItemData(0, 'live')
            self.cloud_choice.setItemData(0, '当前采集会话中的累计采样点云', Qt.ItemDataRole.ToolTipRole)

    def invalidate_render(self, *args):
        self._render_dirty = True
        self._render_serial += 1

    def fit_view(self):
        from femto.app.capture_viewport import model_framing
        if self.view == 'frame_cloud':
            self.viewport.zoom = 1.
            self.invalidate_render()
            return
        key = self.view if self.view in self.clouds else 'final' if 'final' in self.clouds else 'live'
        if key in self.clouds and len(self.clouds[key][0]):
            points = self.clouds[key][0]
            self.center, self.span = model_framing(points)
            self.fit_button.setToolTip('按主体范围居中取景；远处散点可能在视野外。滚轮缩小可查看外围，原始点云保持完整。')
        self.viewport.zoom = 1.
        self.viewport.pan[:] = 0
        self.invalidate_render()

    def render_cloud(self):
        self._render_dirty = False
        serial = self._render_serial
        view = self.view
        session_id = self.session.session_id
        if view == 'frame_cloud':
            frame = self.last_frame
            intrinsics = self.camera_intrinsics
            distortion = self.camera_rgb_distortion
            depth_range = self.device_info.get('depth_range_m', ())
            width, height = self.viewport.width(), self.viewport.height()
            zoom = self.viewport.zoom
            def render_live(cancel, progress):
                if cancel.is_set() or frame is None:
                    return None
                if intrinsics is None or distortion is None or len(depth_range) != 2:
                    raise ValueError('相机 RGB-D 标定或深度范围缺失')
                points, colors = sample_aligned_frame_cloud(
                    frame.depth_mm, frame.color_bgr, intrinsics, distortion,
                    float(depth_range[0]) * 1000, float(depth_range[1]) * 1000)
                if cancel.is_set():
                    return None
                display = camera_oblique_points(points) if len(points) else points
                if len(display):
                    low, high = np.percentile(display[:, :2], [0.5, 99.5], axis=0)
                    center = np.array([(low[0]+high[0])/2, (low[1]+high[1])/2, 0.])
                    span = max(float(np.max(high-low)), .01)
                else:
                    center, span = None, 1.
                return np.asarray(render_scene(width, height, display, colors,
                    center=center, span=span, zoom=zoom, live_frame=True)).copy()
            def ready_live(image):
                if (image is not None and serial == self._render_serial
                        and view == self.view and not self._closed and self.live_frame_valid()):
                    self.viewport.set_image(rgb_image(image), cloud=True)
            def failed_live(error):
                if view == self.view and not self._closed:
                    self.viewport.set_image(QImage(), cloud=True)
                    self.preview_note.setText('实时点云预览不可用：'+str(error))
            self.render_task.start(render_live, ready_live, failed_live)
            return
        points, colors = self.clouds.get(view, (None, None))
        highlight = None
        snapshot = self.preview_snapshot
        if view == 'live' and snapshot is not None and self.highlight.isChecked() and snapshot['accepted_preview_frames'] > 1:
            highlight = snapshot['sources'] == snapshot['frame_id']
        camera = current_camera_pose(self.session.record, active=self.session.active,
            finishing=self.session.finishing, state=self.session.state, now=time.monotonic())
        kwargs = dict(center=None if self.center is None else self.center.copy(), span=self.span,
            zoom=self.viewport.zoom, pan=self.viewport.pan.copy(),
            surface=self.surface if view == 'final' else None, camera=camera, highlight=highlight,
            reference=bool(self.session.count or self.session.result))
        width, height = self.viewport.width(), self.viewport.height()
        def render(cancel, progress):
            if cancel.is_set():
                return None
            return np.asarray(render_scene(width, height, points, colors, **kwargs)).copy()
        def ready(image):
            if (image is not None and serial == self._render_serial and session_id == self.session.session_id
                    and view == self.view and not self._closed):
                self.viewport.set_image(rgb_image(image), cloud=True)
        self.render_task.start(render, ready, lambda error: self.preview_note.setText('三维显示失败：'+str(error)))

    def _discard_prepared_review(self):
        from femto.dataset.prepared_review import close_prepared
        prepared, self._prepared_review = self._prepared_review, None
        close_prepared(prepared)

    def take_prepared_review(self, source):
        """Transfer this result's private model once; readers recheck its bytes."""
        prepared = self._prepared_review
        if prepared is None or Path(source).resolve() not in (prepared.root, prepared.root/'reconstruction'):
            return None
        self._prepared_review = None
        return prepared

    def load_review(self):
        session_id, result = self.session.session_id, self.session.result
        historical=self._history_replay
        self._discard_prepared_review()
        local = self.host.local_capture
        prepared = getattr(local, 'prepared_review', None) if not historical else None
        if prepared is not None and prepared.root == Path(result).resolve():
            local.prepared_review = None
            self._prepared_review = prepared
        else:
            prepared = None
        task=dict(self.task_context or {})
        self._review_request_result = Path(result).resolve() if result else None
        self.session.state = 'loading_review'
        def ready(data):
            if session_id != self.session.session_id or result != self.session.result or data is None:
                self._discard_prepared_review()
                return
            self.clouds.update(data['clouds'])
            self.surface = data['surface']
            self.session.quality = data['quality']
            self.session.acquisition_completion = data.get('acquisition_completion')
            self.scan_roi_quality = data.get('scan_roi_quality')
            self.scan_roi_preview = (rgb_image(data['scan_roi_preview'])
                                     if data.get('scan_roi_preview') is not None else None)
            if data.get('scan_roi') is not None and not self._history_replay:
                self.scan_roi_rect = data['scan_roi']['rect_norm']
                self.scan_roi_size = data['scan_roi']['image_size']
            if data.get('frame_count') is not None:
                try:
                    self.session.count = max(0, int(data['frame_count']))
                except (TypeError, ValueError):
                    pass
            if data.get('source_dataset'):
                try:
                    self.session.source = Path(data['source_dataset']).resolve()
                except (OSError, TypeError, ValueError):
                    pass
            self.session.state = 'ready'
            self.task_context=data.get('task_context')
            self.task_name.setText((self.task_context or {}).get('name','历史记录（未填写任务信息）'))
            self._configure_accumulated_choice()
            self.preview_note.setText('历史记录回放 · 三维数据只读' if self._history_replay
                                      else '真实 RGB-D 采集 · 研发规划预览')
            self.cloud_choice.setCurrentIndex(2)
            # Reference diagnostics can be the first displayed view. Prepare
            # geometry framing here too, before that path bypasses fit_view.
            self.fit_view()
            if self._workflow_view == 1:
                self.show_workflow_step(self._workflow_view)
            else:
                self.show_model_check()
            self.refresh()
            if data.get('task_warning'):self.preview_note.setText('任务信息未保存或不可用：'+data['task_warning'])
            if (not historical and not self._closed and self._auto_plan_result == result
                    and self.has_reconstruction_selection() and self.review_ready()):
                # Navigation is not a human confirmation. The route action
                # records the decision to use this displayed model later.
                self._auto_plan_result = None
                self.planning_preview_requested.emit(str(result / 'reconstruction'))
        def failed(error):
            self._discard_prepared_review()
            if session_id == self.session.session_id and result == self.session.result:
                self.session.state = 'review_load_error'
                self.preview_note.setText('三维预览加载失败：'+str(error))
                self.refresh()
        def job(cancel,progress):
            data=(read_review(result,cancel,progress,prepared=prepared) if prepared is not None
                  else read_review(result,cancel,progress))
            if data is None:return None
            from femto.dataset.task_context import read_task,save_task
            try:data['task_context']=read_task(result) if historical else save_task(result,task)
            except (OSError,ValueError,KeyError,TypeError) as error:data['task_warning']=str(error)
            return data
        self.review_task.start(job, ready, failed)

    def show_model_check(self):
        """Prefer existing, verified reference diagnostics; never invent coverage."""
        if (self.scan_roi_quality or {}).get('status') == 'checked' and self.scan_roi_preview is not None:
            self.set_view('scan_roi')
        else:
            self.set_view('final')
            self.fit_view()

    def show_workflow_step(self, index: int) -> bool:
        """Inspect a completed acquisition without changing its capture state."""
        if index not in (1, 2) or self.session.active or self.session.finishing or not self.session.result:
            return False
        if not self.review_task.busy and not self.review_ready():
            return False
        self._workflow_view = index
        if self.review_task.busy:
            self.refresh()
            return True
        if index == 1:
            self.cloud_choice.setCurrentIndex(1 if self.clouds.get('initial') is not None else 2)
            self.set_view('3d')
        else:
            self.show_model_check()
        self.inspector_scroll.verticalScrollBar().setValue(0)
        self.refresh()
        return True

    @staticmethod
    def review_result_root(source):
        """Resolve a planning source to its owning capture result directory.

        History opens planning at ``<result>/reconstruction`` while the
        capture review reader takes ``<result>``.  Region ZIPs and native
        trajectory files do not carry enough information to reconstruct a
        capture preview, so they intentionally return ``None``.
        """
        try:
            path = Path(source).resolve()
        except (OSError, TypeError, ValueError):
            return None
        if path.is_dir() and path.name.casefold() == 'reconstruction':
            return path.parent
        if path.is_dir() and (path / 'reconstruction').is_dir():
            return path
        return None

    def restore_review_for_source(self, source):
        """Restore a historical result as a read-only capture-page review.

        The planning page deliberately owns only planning state.  When it is
        opened from history, returning to capture must therefore repopulate
        the review clouds explicitly; otherwise the live capture page has no
        session result and renders the empty-grid placeholder.  Loading is
        asynchronous and uses the same bounded ``read_review`` path as a
        freshly completed capture.

        Returns ``True`` when ``source`` identifies a capture result (whether
        the data is already present or is still being read), and ``False`` for
        a standalone region/trajectory project.
        """
        result = self.review_result_root(source)
        if result is None:
            return False
        current = None
        if self.session.result is not None:
            try:
                current = Path(self.session.result).resolve()
            except (OSError, TypeError, ValueError):
                current = None
        requested = self._review_request_result
        if self.review_task.busy:
            # Do not cancel/restart a bounded disk read merely because the
            # user switched pages while it was completing.
            return requested == result
        if current == result and self.clouds:
            self._workflow_view = None
            self.cloud_choice.setCurrentIndex(2)
            self.set_view('3d')
            self.refresh()
            return True
        if self.session.active:
            return False

        # A historical review is display-only.  Use a fresh state token so
        # delayed events from an old live session cannot mutate its clouds.
        self._history_replay = True
        self._workflow_view = None
        self._clear_scan_roi()
        self._configure_accumulated_choice()
        self.session = CaptureState(session_id=uuid.uuid4().hex, state='ready',
                                    result=result)
        self.clouds = {}
        self.surface = None
        self.center = None
        self.span = 1.
        self.preview_snapshot = None
        self.preview_mailbox = None
        self.last_frame = None
        self.cloud_choice.setCurrentIndex(2)
        self.viewport.caption = '正在读取历史三维数据'
        self.viewport.detail = '从历史采集结果读取点云与表面'
        self.set_view('3d')
        self.load_review()
        self.refresh()
        return True

    def refresh(self) -> None:
        state = self.session
        for mode in ('preview','depth','3d'):
            self.views[mode].setEnabled(True)
        preparing = not state.active and not state.result
        inspecting = self._refresh_task_context(state, preparing)
        live = self.live_frame_valid()
        self._refresh_capture_range(state, preparing, live)
        selecting = state.region_reference is not None and state.active
        if self.property('selectionMode') != selecting:
            self.setProperty('selectionMode',selecting)
            for widget in (self, self.action_bar, self.cancel_button, self.primary):
                widget.style().unpolish(widget); widget.style().polish(widget)
        self.inspector_scroll.hide()
        stale = self.camera_ready and not live
        connected = self.host.camera_thread is not None and self.host.camera_thread.is_alive()
        fresh = self._refresh_capture_telemetry(state, live, stale, connected)
        title, message = self._capture_message(state, live, fresh, stale)
        if state.state == 'awaiting_region':
            title, message = '标记工作区域', self.region_editor.note.text()
        self._refresh_capture_notice(state, title, message, fresh, stale)
        self._refresh_capture_actions(state, live, fresh, preparing, connected, message, inspecting)
        self._refresh_history_summary()
        self.region_editor.setVisible(selecting)
        self.region_progress.setVisible(selecting and state.state == 'processing' and not state.cancelling)
        self.capture_view_bar.setVisible(not selecting)
        self.viewport.setVisible(not selecting)
        self.selection_footer_note.setVisible(selecting)
        if selecting:
            # Selection has its own single instruction/error area; stale
            # camera telemetry must not duplicate it as an amber warning.
            self.alert.hide(); self.guidance.hide(); self.preview_note.hide()
            if state.state == 'processing':
                self.region_editor.note.setText('正在重建选中区域，请稍候；完成后自动进入模型与路线。')
            for button in (self.pause_button,self.finish_button,self.check_button,self.stage_review_button):
                button.hide()
            self.inspector_scroll.hide()
            self.region_editor.setEnabled(state.state == 'awaiting_region')
            for button in self.views.values():button.setEnabled(False)
            self.preview_note.setText('固定首帧 RGB · 原始采集帧及位姿保留 · 尚未生成机械臂轨迹')
        # Stage checkpoints are superseded by automatic three-view finalization.
        self.check_button.hide()
        step = self._workflow_view if state.result and self._workflow_view is not None else (
            6 if selecting else 2 if state.result or state.finishing else 1)
        self.workflow.set_step(step, ('已结束采集，只读回看本次记录。' if inspecting else '') + message)
        if message != self._last_status:
            self._last_status = message
            self.status_changed.emit(message)

    def _refresh_history_summary(self):
        historical = bool(self._history_replay and self.session.result)
        compact = self.width() < 1350
        if self.history_metrics.property('compact') != compact:
            self.history_metrics.setProperty('compact', compact)
            layout = self.history_metrics.layout()
            layout.setContentsMargins(8 if compact else 14, 7, 8 if compact else 14, 7)
            layout.setSpacing(8 if compact else 16)
            for widget in [self.history_metrics, *self.history_metrics.findChildren(QLabel)]:
                widget.style().unpolish(widget);widget.style().polish(widget)
        self.history_metrics.setVisible(historical)
        self.frame_progress.setVisible(not historical)
        self.history_rescan.setVisible(historical)
        self.history_rescan.setEnabled(self.rescan_button.isEnabled())
        self.inspector_heading.setVisible(not historical)
        self.state_title.setVisible(not historical)
        self.state_title.parentWidget().setVisible(not historical)
        if not historical:
            return
        self.inspector_scroll.hide()
        # Pending counts stay visible in the toolbar, with the full guidance
        # available on hover. Loading failures retain their actionable alert.
        if not self.review_task.busy and self.session.state == 'ready':
            self.alert.hide()
        self.history_metrics.setToolTip(self.scan_roi_status.text())
        self.views['scan_roi'].setToolTip(self.scan_roi_status.text())
        self.history_rescan.setToolTip('新建独立采集，保留当前记录')
        self.new_button.hide()
        for widget in (self.workflow_view_notice, self.scan_roi_status,
                       self.capture_heading, self.count_label, self.pose_label):
            widget.hide()
        quality = self.scan_roi_quality or {}
        checked = quality.get('status') == 'checked'
        values = (str(quality['supported_pixels'])+' / '+str(quality['requested_pixels']) if checked else '—',
                  str(quality['pending_pixels']) if checked else '—', str(self.session.count))
        hints = (
            '扫描范围内有实测网格与深度支持的参考像素 / 总参考像素。不是治疗覆盖率。' if checked else self.scan_roi_status.text(),
            '缺失或待检查的参考像素；请在缺口检查中核对。' if checked else '尚未完成参考像素检查，不能视为零缺失。',
            '历史采集帧数；只读回放，不代表当前相机位姿。')
        for index, (label, value, color, hint) in enumerate(zip(self.history_metric_labels, values,
                ('#06d6a0' if checked else '#8295a8', '#f4b942' if checked else '#8295a8', '#75b8ea'), hints)):
            display_value = value
            if index == 0 and checked:
                display_value = (str(quality['supported_pixels'])+
                    '<span style="color:#83abc5;font-size:90%"> / '+str(quality['requested_pixels'])+'</span>')
            label.setText(f'<span style="color:{color};font-size:70%">●</span>&nbsp; {display_value}')
            label.setToolTip(hint)
            label.setAccessibleName(hint)
            label.setAccessibleDescription(value)
        self.rescan_button.setToolTip('新建独立采集，保留当前记录')

    def _refresh_task_context(self, state: CaptureState, preparing: bool) -> bool:
        inspecting = bool(state.result and self._workflow_view == 1)
        self.inspector_heading.setText('局部采集 · 回看' if inspecting else '检查模型' if state.result else '局部采集')
        self.workflow_view_notice.setVisible(inspecting)
        self.task_name.setEnabled(not state.active and not state.result)
        return inspecting

    def _refresh_capture_range(self, state: CaptureState, preparing: bool, live: bool) -> None:
        self.scan_roi_button.setVisible(preparing)
        self.scan_roi_button.setEnabled(not state.active and not state.result and not self.review_task.busy
                                        and live)
        self.scan_roi_button.setText('调整采集范围（可选）')
        self.views['scan_roi'].setVisible(bool(state.result))
        self.views['scan_roi'].setEnabled(self.scan_roi_preview is not None)
        for mode in ('preview', 'depth', 'frame_cloud'):
            self.views[mode].setVisible(not state.result)
        self.connection.setVisible(not state.result)
        self.views['3d'].setVisible(bool(state.active or state.result or self.clouds))
        self.rescan_button.setVisible(bool(state.result))
        self.rescan_button.setEnabled(bool(state.result) and not self.review_task.busy)
        if state.result:
            quality = self.scan_roi_quality
            if quality and quality.get('status') == 'checked':
                supported = quality['supported_pixels']
                pending = quality['pending_pixels']
                total = quality['requested_pixels']
                target_name='涂抹选区' if quality.get('target_scope')=='painted_region' else '扫描范围'
                self.scan_roi_status.setText(
                    f'{target_name}内参考图像像素：{supported}/{total} 有实测网格与深度支持；'
                    + (f'{pending} 个缺失或待检查。请检查模型，必要时重新采集。' if pending
                       else '当前检查未发现缺失，仍需核对模型。'))
            elif quality:
                self.scan_roi_status.setText('扫描范围覆盖待核验：'+quality.get('message', '读取失败'))
            else:
                self.scan_roi_status.setText('此记录没有可核验的扫描范围；请在真实模型上检查目标与边界。')
        elif self.scan_roi_rect is None:
            self.scan_roi_status.setText(self.roi_message or '默认采集完整画面，完成后再涂抹目标区域。')
        else:
            current = self.current_depth_check
            detail = (f"当前画面范围内有效深度 {current['valid_fraction']:.0%}；"
                      if current else '当前画面的深度状态待检查；')
            self.scan_roi_status.setText((self.roi_message+'；' if self.roi_message else '')
                                         +detail+'最终几何覆盖在模型生成后核验。')
        self.scan_roi_status.setVisible(bool(state.result or self.scan_roi_rect is not None or self.roi_message))
        return None

    def _refresh_capture_telemetry(self, state: CaptureState, live: bool, stale: bool, connected: bool) -> bool:
        self.connection.setText('相机已连接' if live else '正在连接…' if connected and not self.camera_error else '重新连接' if self.camera_error else '连接相机')
        self.connection.setEnabled(not connected and not state.active)
        is_model = self.view not in ('preview', 'depth', 'frame_cloud', 'scan_roi')
        is_cloud = is_model or self.view == 'frame_cloud'
        self.cloud_choice.setVisible(is_model)
        self.fit_button.setEnabled(is_cloud and (live if self.view == 'frame_cloud' else bool(self.clouds)))
        self.fit_button.setVisible(is_cloud)
        self.highlight.setVisible(is_model and self.view == 'live')
        for key, button in self.views.items():
            button.setChecked((key == '3d' and is_model) or key == self.view)
        self.views['frame_cloud'].setEnabled(not (self._history_replay and state.result))
        if self._history_replay and state.result:
            self.device_label.setText('历史三维记录  ·  只读\n来源  ·  '+state.result.name
                +'\n实时相机  ·  '+('已连接' if live else '未连接'))
            self.capture_heading.setText('历史采集记录')
        else:
            self.device_label.setText('相机  ·  ' + ('已连接' if live else '画面已过期' if stale else '未连接')
                + '\nRGB / 深度  ·  ' + ('已获取' if live else '等待数据')
                + '\n参考位置  ·  ' + ('已建立' if state.count else '未建立'))
            self.capture_heading.setText('本次采集')
        limited = not state.result and not self._history_replay and state.count <= MAX_ACCEPTED_VIEWS
        self.count_label.setText(f'{state.count} / {MAX_ACCEPTED_VIEWS} 帧' if limited else f'{state.count} 帧')
        self.count_label.setAccessibleName('本次已接纳 '+self.count_label.text())
        self.count_label.setToolTip('通过质量与原始写入检查的有效视角，包含初始参考帧；重复或不合格帧不计入。')
        self.capture_heading.setVisible(self.width() >= 1350)
        self.count_label.show()
        for index, segment in enumerate(self.frame_steps):
            segment.setVisible(limited)
            segment.setFixedWidth(36 if self.width() >= 1350 else 12)
            accepted = index < state.count
            if segment.property('accepted') != accepted:
                segment.setProperty('accepted', accepted)
                segment.style().unpolish(segment);segment.style().polish(segment)
        self.pose_label.setVisible(bool(state.active or state.count or state.result))
        count_tone = 'good' if state.count else 'idle'
        if self.count_label.property('tone') != count_tone:
            self.count_label.setProperty('tone', count_tone)
            self.count_label.style().unpolish(self.count_label)
            self.count_label.style().polish(self.count_label)
        self.stability_bar.setValue(min(5, int(state.stability.get('count', 0))))
        self.stability_bar.setVisible(state.active and not state.finishing)
        fresh = state.fresh_pose()
        self.pose_label.setText('历史模型回放 · 不代表当前相机位姿' if self._history_replay and state.result else
            '参考位置未建立' if not state.count else
            '位姿跟踪中 · 范围 30 mm / 10°' if fresh else '参考位置保留 · 当前位姿待确认' if state.active else '本次采集已结束')
        return fresh

    def _capture_message(self, state: CaptureState, live: bool, fresh: bool, stale: bool) -> tuple[str, str]:
        if state.cancelling:
            title, message = '正在取消', '正在取消任务，保留已采集的原始数据。'
        elif state.finishing and state.state != 'processing':
            title = '正在结束采集'
            message = (REASONS['accepted_view_limit'] if state.finish_reason == 'accepted_view_limit'
                       else '正在结束采集并检查本次数据。')
        elif state.active:
            title = {'waiting':'建立参考位置', 'tracking':'补采已暂停' if state.paused else '移动补采',
                     'processing':'正在生成模型', 'uncertain':'等待确认', 'relocalizing':'正在重定位'}.get(state.state, '采集受阻')
            message = REASONS.get(state.state, state.state)
            if state.notice and time.monotonic() < state.notice_until:
                message = REASONS.get(state.notice, state.notice)
            elif state.record.get('local_workspace', {}).get('within_limit') is False:
                message = REASONS['outside_local_workspace']
            elif state.state == 'tracking' and not fresh:
                message = '位姿更新已过期，已暂停补采接纳；请停稳并等待恢复。'
            elif state.paused:
                message = '补采已暂停，参考位置与跟踪保留。'
            elif state.count >= MAX_ACCEPTED_VIEWS and not state.finishing:
                message = '已接纳 3 个有效视角，正在自动结束采集并准备固定参考图。'
            elif state.state == 'tracking' and fresh:
                message = f'已接纳 {state.count} / {MAX_ACCEPTED_VIEWS} 帧，可提前完成；也可移动并停稳后补采，满 3 帧自动进入选区。'
        elif state.result:
            if state.state == 'review_load_error':
                title, message = '模型预览不可用', '三维数据读取失败，请点击“重试加载模型”。'
            elif state.state == 'loading_review':
                title, message = '正在读取模型', '请等待点云和表面加载完成，再核对并进入规划。'
            elif not self.review_ready():
                title, message = '模型数据待确认', '最终点云为空或不可用，请检查历史记录与模型文件。'
            else:
                title, message = '模型可检查', REASONS.get(state.state, REASONS['ready'])
                pending = (self.scan_roi_quality or {}).get('pending_pixels')
                if pending:
                    message = (f'扫描范围内 {pending} 个参考图像像素缺失或待检查；'
                               '先查看“缺口检查”，再切换“三维模型”核对目标与边界，必要时重新采集。')
                elif (self.scan_roi_quality or {}).get('status') == 'checked':
                    message = '当前参考图检查未发现缺失；请切换“三维模型”核对目标与边界，再进入工作区域规划。'
                else:
                    message = '此记录的扫描范围覆盖尚未核验；请检查三维模型中的目标与边界，不应将未知当作完整。'
        elif state.state == 'cancelled':
            title = '本次采集已取消'
            message = ('已记录的原始数据保留在：' + str(state.source) if state.source
                       else '本次采集已取消；尚无已记录的数据目录。')
        elif state.failure:
            from femto.app.qt_failure_details import capture_failure_text
            title, message = '本次采集未完成', capture_failure_text(state.failure, state.quality, state.source)
        elif self.camera_error:
            title, message = '相机未就绪', self.camera_error.split('\n', 1)[0]
        else:
            title = '等待建立参考位置' if live else '等待相机连接'
            message = ('请保持相机与目标稳定，建立本次采集的参考位置。' if live
                       else '界面预览模式；可打开历史数据，连接相机后开始采集。' if self.preview_mode
                       else '请连接 Femto Bolt，并等待同步 RGB-D 画面。')
        if not state.active and not state.result and not state.failure and self.scan_roi_rect is None and live:
            message = self.roi_message or '取景清楚后点击“开始采集”，完成后在固定参考图上涂抹区域。'
        if stale and not state.result:
            message = (message+'\n' if state.failure else '')+'相机画面已过期，请检查设备；新采集与补采暂不可用。'
        return title, message

    def _refresh_capture_notice(self, state: CaptureState, title: str, message: str, fresh: bool, stale: bool) -> None:
        # Derive the state severity before styling the headline. ``refresh``
        # runs during page construction, including preview mode without a camera.
        blocked = bool(self.camera_error or state.failure or state.state in ('lost', 'tracking_lost'))
        warning = (blocked or (state.active and not state.finishing and not fresh and state.count > 0)
                   or stale or bool(state.result and (self.scan_roi_quality or {}).get('pending_pixels')))
        self.state_title.setText(title)
        self.state_title.setVisible(bool(state.active or state.result or state.failure or state.state == 'cancelled'))
        tone = 'error' if blocked else 'warning' if warning else 'good' if title in ('模型可检查', '相机已连接') else 'current'
        self.state_title.setProperty('tone', tone)
        self.state_title.style().unpolish(self.state_title)
        self.state_title.style().polish(self.state_title)
        self.guidance.setText(message)
        self.guidance.setVisible(not warning)
        self.alert.setProperty('tone', 'error' if blocked else 'warning')
        self.alert.style().unpolish(self.alert)
        self.alert.style().polish(self.alert)
        historical_ready = bool(self._history_replay and state.result
                                and state.state == 'ready' and not self.review_task.busy)
        self.alert.setVisible(warning and not historical_ready)
        if warning:
            self.alert.setText(message)

    def _refresh_capture_actions(self, state: CaptureState, live: bool, fresh: bool, preparing: bool, connected: bool, message: str, inspecting: bool) -> None:
        self.pause_button.setText('继续补采' if state.paused else '暂停补采')
        self.check_button.setVisible(state.active and not state.finishing)
        self.check_button.setText('取消阶段检查' if state.checkpoint_busy else '检查当前采集')
        self.check_button.setEnabled(state.active and not state.finishing and
            (state.checkpoint_busy or (state.count>=3 and fresh and live and not state.checkpoint_stop_reason)))
        self.stage_review_button.setVisible(bool(state.checkpoint_result))
        self.checkpoint_hint.setVisible(bool(state.checkpoint_busy or state.checkpoint_error
                                            or state.checkpoint_result or state.checkpoint_stop_reason))
        self.check_button.setToolTip('至少采集 3 个有效视角后可检查；检查后保留参考位置，可继续补采。')
        stage=state.checkpoint_result or {}
        feedback=stage.get('feedback',{})
        hint=('正在生成阶段模型与缺口检查；本次跟踪继续。' if state.checkpoint_busy else
              '阶段检查未完成：'+REASONS.get(state.checkpoint_error,state.checkpoint_error) if state.checkpoint_error else
              f"上次检查：{feedback.get('pending_pixels','未知')} 个参考像素待检查；可继续补采或查看原因。"
              if stage else '至少采集 3 个有效视角后检查当前采集；本次参考位置保持。')
        if feedback.get('stop_reason'):
            hint+=' 多轮未观察到改善，已停止继续补采接纳，请检查原因。'
        if state.checkpoint_stop_reason:
            hint=REASONS.get(state.checkpoint_stop_reason,state.checkpoint_stop_reason)
        self.checkpoint_hint.setText(hint)
        for button in (self.pause_button, self.finish_button, self.cancel_button):
            button.setVisible(state.active)
        self.pause_button.setEnabled(state.active and not state.finishing and state.state not in ('lost', 'tracking_lost'))
        self.finish_button.setEnabled(state.count >= 1 and fresh and live and not state.paused)
        self.finish_button.setToolTip('接纳一帧后可提前完成；最多接纳 3 个有效视角，第三帧保存后自动进入选区。' if not state.paused else REASONS['paused_finish'])
        self.cancel_button.setEnabled(state.active and not state.cancelling)
        self.new_button.setVisible(bool(state.result))
        self.new_button.setEnabled(not state.active and not self.review_task.busy)
        self.retry_button.setVisible(bool(state.result) and state.state == 'review_load_error')
        self.retry_button.setEnabled(not self.review_task.busy)
        self.quality_button.setEnabled(bool(state.quality or state.record))
        self.report_button.setEnabled(bool(state.source or state.result))
        self.quality_button.setVisible(self.quality_button.isEnabled())
        self.report_button.setVisible(self.report_button.isEnabled())
        if state.region_reference is not None and state.active:
            # Choose the selection state before the live-camera branch. A
            # temporary disable/re-enable on every 100 ms poll clears Qt's
            # pressed state and drops clicks released after that poll.
            self.primary.setText('确认选区并重建' if state.state == 'awaiting_region' else '正在重建局部表面…')
            mask = self.region_editor.canvas.work
            self.primary.setEnabled(state.state == 'awaiting_region' and not state.cancelling
                                    and mask is not None and bool(mask.any()))
        elif state.result and not state.active:
            self.primary.setText('进入路线规划' if self.has_reconstruction_selection() else '下一步：检查模型'
                                 if inspecting else '已查看模型，继续规划')
            self.primary.setEnabled(self.review_ready())
        elif state.active:
            self.primary.setText('正在生成模型…' if state.finishing else
                                 '正在结束采集…' if state.count >= MAX_ACCEPTED_VIEWS else '停稳补采一帧')
            self.primary.setEnabled(fresh and live and not state.paused and not state.checkpoint_stop_reason
                                    and state.count < MAX_ACCEPTED_VIEWS)
        else:
            self.primary.setText('重新开始采集' if state.failure or state.count else '开始采集')
            depth_ok = (self.current_depth_check is not None
                        and self.current_depth_check['valid_fraction'] >= MIN_INITIAL_VALID_FRACTION)
            self.primary.setEnabled(live and not self.review_task.busy
                                    and (self.scan_roi_rect is None or depth_ok) and not self.viewport.roi_enabled)
        self.primary.setToolTip(message if not self.primary.isEnabled() else '快捷键：Ctrl+Enter')
        needs_preparation = preparing and not live
        self.prepare_action.setVisible(needs_preparation)
        self.primary.setVisible(not needs_preparation)
        if not live:
            self.prepare_action.setText('等待相机画面…' if connected and not self.camera_error else '连接相机')
            self.prepare_action.setEnabled(not connected and not self.review_task.busy)
        else:
            self.prepare_action.setText('请在画面拖动框选' if self.viewport.roi_enabled else '框选扫描范围')
            self.prepare_action.setEnabled(not self.review_task.busy and not self.viewport.roi_enabled)
        self.prepare_action.setToolTip(message)
        self.preview_note.setVisible(self.preview_note.text() not in (
            '真实 RGB-D 采集 · 研发规划预览',
            '当前相机 RGB · 研发采集预览', '当前相机对齐深度 · 研发采集预览'))

    def show_quality(self):
        from femto.local.local_processing import format_quality
        from femto.tracking.local_stability_feedback import format_stability
        parts = []
        if self.session.quality:
            parts.append(format_quality(self.session.quality, 'zh'))
        if self.session.stability:
            parts.append(format_stability(self.session.stability, 'zh'))
        quality = self.session.record.get('quality')
        if quality:
            parts.append('当前帧质量\n'+json.dumps(quality, ensure_ascii=False, indent=2))
        show_text(self, '采集质量', '\n\n'.join(parts) or '尚无可用质量记录。')

    def show_details(self):
        state = self.session
        details = dict(session_id=state.session_id, state=state.state, accepted_frames=state.count,
            source=str(state.source or ''), result=str(state.result or ''), current_pose=state.record,
            stability=state.stability, failure=state.failure, camera=self.device_info)
        text = (self.guidance.text()+'\n\n'+self.device_label.text()+'\n\n'
                +json.dumps(details, ensure_ascii=False, indent=2, default=str))
        if self.camera_error:
            text += '\n\n'+self.camera_error
        if state.failure:
            from femto.app.qt_failure_details import capture_failure_text
            text += '\n\n'+capture_failure_text(state.failure, state.quality, state.source)
        show_text(self, '本次采集详情', text)

    def open_report(self):
        choices = []
        if self.session.result:
            choices.extend(self.session.result / name for name in ('processing_report.html', 'result_quality.html'))
        if self.session.quality and self.session.quality.get('report_path'):
            choices.append(Path(self.session.quality['report_path']).with_suffix('.html'))
        if self.session.source:
            choices.extend(self.session.source / 'derived' / name for name in ('local_failure.json', 'local_capture_report.json'))
        for path in choices:
            if path.is_file():
                open_path(path)
                return
        self.show_quality()

    def can_leave(self):
        return not self.session.active and not self.review_task.busy

    def shutdown(self):
        self.session.cancel_event.set()
        self.review_task.cancel()
        self.render_task.cancel()
        stopped = self.host.stop()
        reviewed = self.review_task.shutdown(timeout_ms=500)
        rendered = self.render_task.shutdown(timeout_ms=500)
        if stopped and reviewed and rendered:
            self._discard_prepared_review()
            self.timer.stop()
            self._closed = True
            return True
        return False
