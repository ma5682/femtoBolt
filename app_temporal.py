"""Temporal burst on the GUI's existing camera owner; no SDK creation here."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
import json

import numpy as np

from io_utils import imwrite_unicode, write_json as json_dump
from scene_capture import write_calibration


class TemporalBurst:
    def __init__(self, camera, count, mode):
        if count not in (1,5,10,15) or mode not in ('fast','quality'):
            raise ValueError('Invalid temporal count/mode')
        self.count,self.mode=count,mode
        self.frames=[]; self.intrinsics=[]
        self.deadline=monotonic()+20
        # A detached snapshot: worker never reads the live SDK/camera object.
        self.camera=SimpleNamespace(metadata=deepcopy(camera.metadata),calibration=deepcopy(camera.calibration),
                                    intrinsics=deepcopy(camera.intrinsics),
                                    min_depth_mm=camera.min_depth_mm,max_depth_mm=camera.max_depth_mm)

    def append(self, frame, intrinsics):
        if len(self.frames)>=self.count or (self.frames and frame.timestamp_us<=self.frames[-1].timestamp_us):
            return False
        self.frames.append(replace(frame,depth_mm=frame.depth_mm.copy(),color_bgr=frame.color_bgr.copy(),
                                   frame_metadata=deepcopy(frame.frame_metadata),native_depth_mm=None))
        self.intrinsics.append(deepcopy(intrinsics.to_dict()))
        return True

    @property
    def complete(self): return len(self.frames)==self.count

    def save(self, root: Path):
        output=Path(root)/('temporal_capture_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir(parents=True,exist_ok=False)
        write_calibration(output,self.camera,self.camera.intrinsics)
        records=[]
        for i,frame in enumerate(self.frames):
            rgb=f'raw/frame_{i+1:03d}_rgb.png'; depth=f'raw/frame_{i+1:03d}_depth_mm.png'
            imwrite_unicode(output/rgb,frame.color_bgr)
            imwrite_unicode(output/depth,np.rint(np.clip(frame.depth_mm,0,65535)).astype(np.uint16))
            records.append(dict(frame=i,rgb=rgb,depth=depth,camera_timestamp_us=frame.timestamp_us,
                                color_camera_timestamp_us=frame.color_timestamp_us,depth_camera_timestamp_us=frame.depth_timestamp_us,
                                rgb_depth_timestamp_delta_us=frame.color_timestamp_us-frame.depth_timestamp_us,
                                camera_frame_index=frame.frame_index,frame_metadata=frame.frame_metadata,
                                aligned_intrinsics=self.intrinsics[i],pose_valid=False))
        manifest=dict(schema='femto-full-scene-rgbd-v1',frame_count=len(self.frames),requested_frame_count=self.count,
                      status='captured' if self.complete else 'incomplete_capture',camera=self.camera.metadata,
                      parameters=dict(fixed_camera=True,processing_mode='temporal',quality_mode=self.mode,
                                      depth_min_mm=self.camera.min_depth_mm,depth_max_mm=self.camera.max_depth_mm),
                      pose=dict(valid=False,source='not_estimated; fixed temporal grid only'),
                      capture_policy='Existing GUI camera; bounded memory burst; encoding and fusion on worker')
        json_dump(output/'manifest.json',manifest)
        json_dump(output/'metadata.json',dict(**manifest,timestamps_us=[f.timestamp_us for f in self.frames]))
        (output/'frames.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
        return output


def process_burst(burst, root, events, *, fuse=True):
    source=None
    try:
        events.put(('temporal_processing',None))
        source=burst.save(root)
        if not fuse or not burst.complete:
            events.put(('temporal_finished',dict(status='cancelled_or_incomplete',source=str(source),output=str(source))))
            return
        from scene_temporal_fusion import build_temporal_dataset
        output=build_temporal_dataset(source,frame_count=burst.count,mode=burst.mode)
        report=json.loads((output/'temporal_report.json').read_text(encoding='utf-8'))
        events.put(('temporal_finished',dict(status=report['status'],source=str(source),output=str(output),report=report)))
    except Exception as error:
        events.put(('temporal_error',dict(error=str(error),source=str(source) if source else None)))


def process_keyframe_burst(burst, root, events, *, fuse=True):
    """Same button: geometry and GS inputs only; never start a trainer."""
    source=None
    try:
        events.put(('keyframe_stage','saving'))
        source=burst.save(root)
        if not fuse or not burst.complete:
            events.put(('keyframe_stopped',str(source))); return
        from temporal_keyframe import build_temporal_keyframe
        dataset,report=build_temporal_keyframe(source,lambda stage:events.put(('keyframe_stage',stage)))
        from complete_demo import run_demo
        from argparse import Namespace
        events.put(('keyframe_stage','geometry_inputs'))
        result,summary=run_demo(Namespace(mode='latest',dataset=dataset,datasets_root=root,
                                         outputs_root=dataset.parent/'models',synthetic_frames=14,no_viewer=True))
        events.put(('reconstruction_done',dict(result=str(result),reconstruction=summary['reconstruction'],
                    source_dataset=str(source),
                    metrics=summary['reconstruction_metrics'],processing_mode='keyframe',
                    processing_report=dict(selected_valid_depth_ratio=1-report['after_invalid_rate']),
                    temporal_report=report)))
    except Exception as error:
        import traceback
        events.put(('reconstruction_error',f'{error}\n{source}\n{traceback.format_exc(limit=5)}'))
