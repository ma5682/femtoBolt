"""Asynchronous, observed-only accumulation preview; never reconstruction input.

Consumes already saved, accepted RGB-D frames. The tracker and SDK never wait
for point conversion or UI rendering. Preview voxel cells retain actual samples
(not averaged/inferred surface points), including their source frame IDs.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

from scene_dataset_validator import validate_pose
from scene_reconstruction import _distortion_maps, _read_image
from scene_capture import json_dump


class ObservedAccumulator:
    def __init__(self, *, voxel_m=.005, max_points=120000):
        if voxel_m <= 0 or max_points < 1:
            raise ValueError('Invalid preview budget')
        self.voxel_m, self.max_points = voxel_m, max_points
        self.points = np.empty((0, 3), dtype=np.float64)
        self.colors = np.empty((0, 3), dtype=np.uint8)
        self.sources = np.empty(0, dtype=np.int64)
        self.cells = set()
        self.frame_ids = set()
        self.initial = None
        self.limited = False

    def add(self, points, colors, frame_id, transform):
        transform = np.asarray(transform, dtype=np.float64)
        error = validate_pose(transform)
        if error:
            raise ValueError(error)
        if frame_id in self.frame_ids:
            raise ValueError('Duplicate preview frame')
        points = np.asarray(points, dtype=np.float64)
        colors = np.asarray(colors, dtype=np.uint8)
        if points.ndim != 2 or points.shape[1:] != (3,) or colors.shape != points.shape:
            raise ValueError('Invalid preview samples')
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        world = points[valid] @ transform[:3, :3].T + transform[:3, 3]
        colors = colors[valid]
        cells = np.floor(world / self.voxel_m).astype(np.int64)
        _, unique = np.unique(cells, axis=0, return_index=True)
        unique.sort()
        novel = [int(i) for i in unique if tuple(cells[i]) not in self.cells]
        remaining = self.max_points-len(self.points)
        if len(novel) > remaining:
            self.limited = True
            # Spread the remaining samples across the candidate surface.
            novel = [novel[i] for i in np.linspace(0,len(novel)-1,remaining,dtype=int)] if remaining else []
        self.cells.update(tuple(cells[i]) for i in novel)
        start = len(self.points)
        self.points = np.concatenate((self.points, world[novel]))
        self.colors = np.concatenate((self.colors, colors[novel]))
        self.sources = np.concatenate((self.sources, np.full(len(novel),frame_id,dtype=np.int64)))
        self.frame_ids.add(frame_id)
        if self.initial is None:
            self.initial = (self.points.copy(), self.colors.copy())
        return dict(frame_id=frame_id, added_samples=len(novel), point_count=len(self.points),
                    highlighted_from=start, budget_limited=self.limited)

    def snapshot(self, record, *, display_budget=30000):
        # Display subset only; source-frame association remains explicit.
        ids = np.arange(len(self.points))
        if len(ids) > display_budget:
            ids = np.linspace(0,len(ids)-1,display_budget,dtype=int)
        return dict(points=self.points[ids].copy(), colors=self.colors[ids].copy(),
                    sources=self.sources[ids].copy(), **record)


class LivePreviewWorker:
    def __init__(self, dataset, session_id, minimum_mm, maximum_mm):
        self.dataset, self.session_id = Path(dataset), session_id
        self.minimum_mm, self.maximum_mm = minimum_mm, maximum_mm
        self.mailbox = queue.Queue(maxsize=1)
        self.publish_lock = threading.Lock()
        self.pending = queue.Queue(maxsize=128)  # Small path/pose descriptors only.
        self.closing = threading.Event()
        self.done = threading.Event()
        self.failed = False
        self.accumulator = ObservedAccumulator()
        self.records = []
        self.thread = threading.Thread(target=self._run, name='FemtoObservedPreview', daemon=False)
        self.thread.start()

    def submit(self, record):
        if self.closing.is_set() or self.failed:
            return False
        if record.get('state') != 'tracking' or not record.get('selected_for_fusion'):
            return False
        try:
            self.pending.put_nowait((copy.deepcopy(record), time.monotonic()))
            return True
        except queue.Full:
            # Fail this optional preview, never silently present a partial backlog as current.
            self.failed = True
            self.closing.set()
            self.publish(dict(status='error',error='preview_backlog'))
            return False

    def close(self):
        self.closing.set()

    def publish(self, data):
        data = dict(session_id=self.session_id, **data)
        with self.publish_lock:
            try:
                self.mailbox.get_nowait()
            except queue.Empty:
                pass
            self.mailbox.put_nowait(data)

    def read_samples(self, frame_id, maps):
        mx, my, k, width, height = maps
        stem = f'{frame_id:06d}'
        rgb = _read_image(self.dataset/'rgb'/f'{stem}.png', cv2.IMREAD_COLOR)
        depth = _read_image(self.dataset/'depth'/f'{stem}.png', cv2.IMREAD_UNCHANGED)
        if rgb.shape[:2] != (height,width) or depth.shape != (height,width) or depth.dtype != np.uint16:
            raise ValueError('Preview RGB-D/calibration mismatch')
        # Select pixels on the undistorted grid; nearest-neighbour depth only.
        stride = 4
        map_x, map_y = mx[::stride,::stride], my[::stride,::stride]
        sampled_depth = cv2.remap(depth,map_x,map_y,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        sampled_rgb = cv2.remap(rgb,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        y,x = np.mgrid[0:height:stride,0:width:stride]
        valid = (sampled_depth >= self.minimum_mm) & (sampled_depth <= self.maximum_mm) & (sampled_depth > 0)
        z = sampled_depth[valid].astype(np.float64)/1000.
        points = np.column_stack(((x[valid]-k[0,2])*z/k[0,0], (y[valid]-k[1,2])*z/k[1,1], z))
        return points, sampled_rgb[valid,::-1]

    def _run(self):
        error = None
        try:
            maps = _distortion_maps(self.dataset)
            while not self.closing.is_set() or not self.pending.empty():
                if self.failed:
                    raise ValueError('preview_backlog')
                try:
                    record, requested = self.pending.get(timeout=.1)
                except queue.Empty:
                    continue
                points, colors = self.read_samples(record['frame_id'],maps)
                update = self.accumulator.add(points,colors,record['frame_id'],record['T_WC'])
                update.update(capture_timestamp_us=record['capture_timestamp_us'],
                              updated_monotonic=time.monotonic(), latency_ms=(time.monotonic()-requested)*1000,
                              accepted_preview_frames=len(self.records)+1)
                self.records.append(update)
                if not self.failed:
                    snapshot = self.accumulator.snapshot(update)
                    initial_points, initial_colors = self.accumulator.initial
                    ids = np.linspace(0,len(initial_points)-1,min(30000,len(initial_points)),dtype=int)
                    snapshot.update(initial_points=initial_points[ids].copy(),initial_colors=initial_colors[ids].copy())
                    self.publish(dict(status='ready', **snapshot))
        except Exception as exc:
            self.failed = True
            error = str(exc)
            self.publish(dict(status='error',error=error))
        finally:
            try:
                json_dump(self.dataset/'derived'/'live_preview_report.json',dict(
                    schema='femto-observed-preview-v1', session_id=self.session_id,
                    status='error' if self.failed else 'closed', error=error,
                    coordinate_frame='world_rdf', transform_direction='T_W0_Ct',unit='metre',
                    voxel_m=self.accumulator.voxel_m,max_points=self.accumulator.max_points,
                    pixel_stride=4,display_budget=30000,records=self.records,
                    purpose='observed_sample_preview_only_not_reconstruction_or_coverage_measurement'))
            except Exception as exc:
                self.failed = True
                self.publish(dict(status='error',error='preview_report: '+str(exc)))
            finally:
                self.done.set()
