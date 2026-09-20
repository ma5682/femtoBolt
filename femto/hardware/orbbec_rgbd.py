"""Synchronized Femto Bolt RGB-D capture with depth aligned to color.

Color-format decoding is adapted from Orbbec's Apache-2.0 pyorbbecsdk
``examples/utils.py`` and has been rewritten for this project's data model.
Copyright (c) 2024 Orbbec 3D Technology, Inc. See
``licenses/pyorbbecsdk2/LICENSE`` and ``SOURCE_PROVENANCE.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import time

from femto.geometry.pose_geometry import Intrinsics

from pyorbbecsdk import (
    AlignFilter,
    Config,
    Context,
    OBFormat,
    OBFrameMetadataType,
    OBSensorType,
    OBStreamType,
    Pipeline,
)


DEPTH_MODE_SPECS = {
    "wide_detail": {
        "name": "WFOV_UNBINNED", "width": 1024, "height": 1024,
        "fps": 15, "min_depth_mm": 250.0, "max_depth_mm": 2210.0,
    },
    "wide": {
        "name": "WFOV_2X2_BINNED",
        "width": 512,
        "height": 512,
        "fps": 30,
        "min_depth_mm": 250.0,
        "max_depth_mm": 2880.0,
    },
    "near": {
        "name": "NFOV_UNBINNED",
        "width": 640,
        "height": 576,
        "fps": 30,
        "min_depth_mm": 500.0,
        "max_depth_mm": 3860.0,
    },
    "far": {
        "name": "NFOV_2X2_BINNED",
        "width": 320,
        "height": 288,
        "fps": 30,
        "min_depth_mm": 500.0,
        "max_depth_mm": 5460.0,
    },
}


@dataclass
class AlignedFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    timestamp_us: int
    system_timestamp_us: int
    frame_index: int
    color_timestamp_us: int
    color_system_timestamp_us: int
    depth_timestamp_us: int
    depth_system_timestamp_us: int
    frame_metadata: dict[str, dict[str, int]] = field(default_factory=dict)
    native_depth_mm: np.ndarray | None = None


FRAME_METADATA_FIELDS = {
    "frame_number": OBFrameMetadataType.FRAME_NUMBER,
    "sensor_timestamp": OBFrameMetadataType.SENSOR_TIMESTAMP,
    "timestamp": OBFrameMetadataType.TIMESTAMP,
    "actual_frame_rate": OBFrameMetadataType.ACTUAL_FRAME_RATE,
    "exposure": OBFrameMetadataType.EXPOSURE,
    "exposure_priority": OBFrameMetadataType.EXPOSURE_PRIORITY,
    "gain": OBFrameMetadataType.GAIN,
    "white_balance": OBFrameMetadataType.WHITE_BALANCE,
    "auto_exposure": OBFrameMetadataType.AUTO_EXPOSURE,
    "auto_white_balance": OBFrameMetadataType.AUTO_WHITE_BALANCE,
    "laser_power": OBFrameMetadataType.LASER_POWER,
    "laser_power_level": OBFrameMetadataType.LASER_POWER_LEVEL,
    "laser_status": OBFrameMetadataType.LASER_STATUS,
}


def read_frame_metadata(frame: Any) -> dict[str, int]:
    """Read metadata exposed by the active SDK/firmware without making it mandatory."""
    metadata: dict[str, int] = {}
    for name, metadata_type in FRAME_METADATA_FIELDS.items():
        try:
            if frame.has_metadata(metadata_type):
                metadata[name] = int(frame.get_metadata_value(metadata_type))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Metadata availability differs by stream, firmware and Windows driver.
            continue
    return metadata


def _optional_device_info(info: Any, method_name: str) -> Any | None:
    try:
        return getattr(info, method_name)()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def frame_to_bgr_image(frame: Any) -> np.ndarray | None:
    """Decode the color formats used by Orbbec's official Python examples."""
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return data.reshape(height, width, 3).copy()
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if color_format == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
    if color_format == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.I420:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_I420)
    if color_format == OBFormat.NV12:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
    if color_format == OBFormat.NV21:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV21)
    raise RuntimeError(f"Unsupported Orbbec color format: {color_format}")


class OrbbecAlignedCamera:
    """Blocking aligned RGB-D reader for one Femto Bolt.

    The returned depth image is in millimetres and shares the color pixel grid.
    ``intrinsics`` therefore describes both 2-D landmarks and aligned depth.
    """

    def __init__(
        self, depth_mode: str = "near", *, color_size: tuple[int, int] | None = None,
        fps: int | None = None, retain_native_depth: bool = False,
    ) -> None:
        if depth_mode not in DEPTH_MODE_SPECS:
            raise ValueError(f"Unknown depth mode: {depth_mode}")
        self.depth_mode = depth_mode
        self.color_size = color_size
        self.depth_spec = dict(DEPTH_MODE_SPECS[depth_mode])
        if fps is not None:
            allowed_fps = (5, 15, 25) if depth_mode == "wide_detail" else (5, 15, 25, 30)
            if fps not in allowed_fps:
                raise ValueError(f"Unsupported {depth_mode} frame rate: {fps}")
            self.depth_spec["fps"] = fps
        self.retain_native_depth = retain_native_depth
        self.min_depth_mm = float(self.depth_spec["min_depth_mm"])
        self.max_depth_mm = float(self.depth_spec["max_depth_mm"])
        self.context = None
        self.devices = None
        self.device = None
        self.pipeline = None
        self.depth_profile = None
        self.color_profile = None
        self.align = None
        self.intrinsics: Intrinsics | None = None
        self.metadata: dict = {}
        self.calibration: dict = {}
        self._read_attempts = 0
        self._read_successes = 0
        self._last_read_status = "not_started"
        self._read_failure_counts: dict[str, int] = {}

    def _read_failure(self, reason: str) -> None:
        self._last_read_status = reason
        self._read_failure_counts[reason] = self._read_failure_counts.get(reason, 0) + 1
        return None

    def read_diagnostics_snapshot(self) -> dict[str, object]:
        """Return non-sensitive startup/read counters for failure evidence."""
        return {
            "attempts": int(self._read_attempts),
            "successful_aligned_frames": int(self._read_successes),
            "last_status": self._last_read_status,
            "failure_counts": dict(sorted(self._read_failure_counts.items())),
        }

    def start(self) -> "OrbbecAlignedCamera":
        self.context = Context()
        self.devices = self.context.query_devices()
        if self.devices.get_count() == 0:
            raise RuntimeError("No Orbbec camera found")
        self.device = self.devices.get_device_by_index(0)
        self.pipeline = Pipeline(self.device)
        config = Config()
        self.depth_profile = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        ).get_video_stream_profile(
            int(self.depth_spec["width"]),
            int(self.depth_spec["height"]),
            OBFormat.Y16,
            int(self.depth_spec["fps"]),
        )
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self.color_profile = (
            color_profiles.get_default_video_stream_profile()
            if self.color_size is None else
            color_profiles.get_video_stream_profile(*self.color_size, OBFormat.MJPG, int(self.depth_spec["fps"]))
        )
        config.enable_stream(self.depth_profile)
        config.enable_stream(self.color_profile)
        self.pipeline.enable_frame_sync()
        self.pipeline.start(config)
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        # Camera parameters become valid after the active profiles have started.
        first = None
        startup_deadline = time.monotonic() + 15.0
        while time.monotonic() < startup_deadline:
            first = self.pipeline.wait_for_frames(1000)
            if first is not None and first.get_depth_frame() is not None and first.get_color_frame() is not None:
                break
            first = None
            time.sleep(0.02)
        if first is None:
            self.stop()
            raise RuntimeError("Camera started but produced no RGB-D frame")
        camera_param = self.pipeline.get_camera_param()
        rgb = camera_param.rgb_intrinsic
        self.intrinsics = Intrinsics(
            width=int(self.color_profile.get_width()),
            height=int(self.color_profile.get_height()),
            fx=float(rgb.fx),
            fy=float(rgb.fy),
            cx=float(rgb.cx),
            cy=float(rgb.cy),
        )
        info = self.device.get_device_info()
        self.metadata = {
            "device": info.get_name(),
            "firmware": info.get_firmware_version(),
            "hardware": _optional_device_info(info, "get_hardware_version"),
            "serial_number": _optional_device_info(info, "get_serial_number"),
            "uid": _optional_device_info(info, "get_uid"),
            "vid": _optional_device_info(info, "get_vid"),
            "pid": _optional_device_info(info, "get_pid"),
            "connection_type": _optional_device_info(info, "get_connection_type"),
            "depth_profile": str(self.depth_profile),
            "depth_mode": self.depth_spec["name"],
            "depth_range_m": [self.min_depth_mm / 1000.0, self.max_depth_mm / 1000.0],
            "color_profile": str(self.color_profile),
            "alignment": "depth_to_color",
            "depth_unit_after_conversion": "millimetres",
            "coordinate_system": "+X right, +Y down, +Z forward",
        }
        self.calibration = {
            "schema": "orbbec-femto-calibration-v1",
            "rgb_intrinsic": {
                "width": int(camera_param.rgb_intrinsic.width),
                "height": int(camera_param.rgb_intrinsic.height),
                "fx": float(camera_param.rgb_intrinsic.fx),
                "fy": float(camera_param.rgb_intrinsic.fy),
                "cx": float(camera_param.rgb_intrinsic.cx),
                "cy": float(camera_param.rgb_intrinsic.cy),
            },
            "depth_intrinsic": {
                "width": int(camera_param.depth_intrinsic.width),
                "height": int(camera_param.depth_intrinsic.height),
                "fx": float(camera_param.depth_intrinsic.fx),
                "fy": float(camera_param.depth_intrinsic.fy),
                "cx": float(camera_param.depth_intrinsic.cx),
                "cy": float(camera_param.depth_intrinsic.cy),
            },
            "rgb_distortion": {
                "model": str(camera_param.rgb_distortion.model),
                **{
                    key: float(getattr(camera_param.rgb_distortion, key))
                    for key in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
                },
            },
            "depth_distortion": {
                "model": str(camera_param.depth_distortion.model),
                **{
                    key: float(getattr(camera_param.depth_distortion, key))
                    for key in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
                },
            },
            "depth_to_color": {
                "rotation_row_major": np.asarray(
                    camera_param.transform.rot, dtype=np.float64
                ).reshape(-1).tolist(),
                "translation_mm": np.asarray(
                    camera_param.transform.transform, dtype=np.float64
                ).reshape(-1).tolist(),
            },
            "is_mirrored": bool(camera_param.is_mirrored),
        }
        return self

    def read(self, timeout_ms: int = 1000) -> AlignedFrame | None:
        if self.pipeline is None or self.align is None or self.intrinsics is None:
            raise RuntimeError("Camera is not started")
        self._read_attempts += 1
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return self._read_failure("timeout")
        native_depth_mm = None
        if self.retain_native_depth:
            native = frames.get_depth_frame()
            if native is None:
                return self._read_failure("missing_native_depth")
            native_depth_mm = np.frombuffer(native.get_data(), dtype=np.uint16).reshape(
                native.get_height(), native.get_width(),
            ).astype(np.float32) * float(native.get_depth_scale())
        try:
            aligned = self.align.process(frames)
        except Exception:
            self._read_failure("align_error")
            raise
        if aligned is None:
            return self._read_failure("align_none")
        frame_set = aligned.as_frame_set()
        if frame_set is None:
            return self._read_failure("frame_set_none")
        color_frame = frame_set.get_color_frame()
        depth_frame = frame_set.get_depth_frame()
        if color_frame is None:
            return self._read_failure("missing_color")
        if depth_frame is None:
            return self._read_failure("missing_depth")
        color = frame_to_bgr_image(color_frame)
        if color is None:
            return self._read_failure("color_decode")
        width, height = depth_frame.get_width(), depth_frame.get_height()
        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        if raw.size != width * height:
            return self._read_failure("depth_size")
        depth_mm = raw.reshape(height, width).astype(np.float32)
        depth_mm *= float(depth_frame.get_depth_scale())

        if color.shape[:2] != depth_mm.shape:
            # Some firmware/profile combinations keep a different aligned target
            # size. Resize color and scale the active color intrinsics together.
            source_height, source_width = color.shape[:2]
            color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
            scale_x, scale_y = width / source_width, height / source_height
            current = self.intrinsics
            self.intrinsics = Intrinsics(
                width, height,
                current.fx * scale_x, current.fy * scale_y,
                current.cx * scale_x, current.cy * scale_y,
            )
        elif (self.intrinsics.width, self.intrinsics.height) != (width, height):
            current = self.intrinsics
            scale_x, scale_y = width / current.width, height / current.height
            self.intrinsics = Intrinsics(
                width, height,
                current.fx * scale_x, current.fy * scale_y,
                current.cx * scale_x, current.cy * scale_y,
            )
        result = AlignedFrame(
            native_depth_mm=native_depth_mm,
            color_bgr=color,
            depth_mm=depth_mm,
            timestamp_us=int(depth_frame.get_timestamp_us()),
            system_timestamp_us=int(depth_frame.get_system_timestamp_us()),
            frame_index=int(depth_frame.get_index()),
            color_timestamp_us=int(color_frame.get_timestamp_us()),
            color_system_timestamp_us=int(color_frame.get_system_timestamp_us()),
            depth_timestamp_us=int(depth_frame.get_timestamp_us()),
            depth_system_timestamp_us=int(depth_frame.get_system_timestamp_us()),
            frame_metadata={
                "color": read_frame_metadata(color_frame),
                "depth": read_frame_metadata(depth_frame),
            },
        )
        self._read_successes += 1
        self._last_read_status = "aligned_frame"
        return result

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            finally:
                self.pipeline = None

    def __enter__(self) -> "OrbbecAlignedCamera":
        return self.start()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop()
