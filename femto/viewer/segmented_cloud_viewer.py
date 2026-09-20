"""Interactive viewer for a merged static-scene full-range point cloud."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from femto.geometry.fusion import read_geometry_safe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud", type=Path)
    args = parser.parse_args()
    cloud_path = args.cloud.resolve()
    cloud = read_geometry_safe(cloud_path, mesh=False)
    if not cloud.has_points():
        raise ValueError(f"Point cloud is empty: {cloud_path}")
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.35)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
        window_name=f"Femto Bolt full-range merged cloud | {cloud_path.parent.name}",
        width=1280,
        height=820,
    )
    viewer.add_geometry(cloud, reset_bounding_box=True)
    viewer.add_geometry(axes, reset_bounding_box=False)
    options = viewer.get_render_option()
    options.background_color = np.array([0.04, 0.06, 0.08])
    options.point_size = 2.0
    options.light_on = False
    viewer.run()
    viewer.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
