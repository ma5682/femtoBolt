"""Write the calibration JSON set that every dataset carries (see local_reference.CALIBRATION_FILES)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from femto.io_utils import write_json


def write_calibration(session: Path, camera: Any, intrinsics: Any) -> None:
    """Save raw factory calibration and the aligned RGB-D pixel-grid intrinsic."""
    calibration = camera.calibration
    write_json(session / "calibration" / "color_intrinsic.json", intrinsics.to_dict())
    write_json(
        session / "calibration" / "depth_intrinsic.json",
        calibration["depth_intrinsic"],
    )
    write_json(
        session / "calibration" / "depth_to_color.json",
        calibration["depth_to_color"],
    )
    write_json(
        session / "calibration" / "distortion.json",
        {
            "rgb": calibration["rgb_distortion"],
            "depth": calibration["depth_distortion"],
        },
    )
    write_json(session / "calibration" / "factory_calibration.json", calibration)
    write_json(session / "calibration" / "camera_metadata.json", camera.metadata)
