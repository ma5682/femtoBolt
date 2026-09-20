"""Open the latest fused patient surface with 3-D joints and region markers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from femto.geometry.fusion import read_geometry_safe
from femto.geometry.pose_geometry import POSE_CONNECTIONS


from femto.paths import REPO_ROOT as ROOT


def latest_session() -> Path:
    marker = ROOT / "outputs" / "latest_session.txt"
    if marker.exists():
        path = Path(marker.read_text(encoding="utf-8").strip())
        if path.exists():
            return path
    candidates = sorted((ROOT / "outputs").glob("patient_top_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No completed patient session was found")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = args.session.resolve() if args.session else latest_session()
    geometries = []
    filled_mesh_path = session / "patient_surface_tsdf_filled_m.ply"
    mesh_path = session / "patient_surface_tsdf_m.ply"
    cloud_path = session / "patient_surface_points_m.ply"
    if filled_mesh_path.exists() or mesh_path.exists():
        selected_mesh = filled_mesh_path if filled_mesh_path.exists() else mesh_path
        surface = read_geometry_safe(selected_mesh, mesh=True)
        if len(surface.triangles):
            labels, counts, _ = surface.cluster_connected_triangles()
            labels_array = np.asarray(labels)
            counts_array = np.asarray(counts)
            if len(counts_array) > 1:
                # Display the patient-sized principal surface so isolated TSDF
                # specks do not make the fitted subject tiny. Disk data stays
                # untouched for quality review.
                surface.remove_triangles_by_mask(labels_array != int(np.argmax(counts_array)))
                surface.remove_unreferenced_vertices()
        surface.compute_vertex_normals()
        geometries.append(surface)
    elif cloud_path.exists():
        surface = read_geometry_safe(cloud_path, mesh=False)
        geometries.append(surface)
    else:
        raise FileNotFoundError(f"No surface PLY in {session}")

    pose = json.loads((session / "joints_3d_m.json").read_text(encoding="utf-8"))
    points = np.full((33, 3), np.nan, dtype=np.float64)
    for item in pose["joints"]:
        if item["xyz_m"] is not None:
            points[item["index"]] = item["xyz_m"]
    # Display-only Y flip gives a conventional upright view. Disk coordinates
    # remain the Orbbec camera convention (+Y down).
    display_points = points.copy()
    display_points[:, 1] *= -1.0
    valid = np.isfinite(display_points).all(axis=1)
    joints = o3d.geometry.PointCloud()
    joints.points = o3d.utility.Vector3dVector(display_points[valid])
    joints.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.15, 0.08], (valid.sum(), 1)))
    geometries.append(joints)

    compact_indexes = {original: compact for compact, original in enumerate(np.flatnonzero(valid))}
    lines = [
        (compact_indexes[a], compact_indexes[b])
        for a, b in POSE_CONNECTIONS
        if a in compact_indexes and b in compact_indexes
    ]
    if lines:
        skeleton = o3d.geometry.LineSet()
        skeleton.points = o3d.utility.Vector3dVector(display_points[valid])
        skeleton.lines = o3d.utility.Vector2iVector(lines)
        skeleton.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.8, 0.05], (len(lines), 1))
        )
        geometries.append(skeleton)

    # Mirror surface Y only in the display copy.
    transform = np.eye(4)
    transform[1, 1] = -1.0
    geometries[0].transform(transform)
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name=f"Patient top surface + 3D joints | {session.name}",
        width=1280,
        height=820,
    )
    for index, geometry in enumerate(geometries):
        # Let the surface initialise the Open3D view bounds. Extra joints and
        # axes must not expand the fitted patient view.
        visualizer.add_geometry(geometry, reset_bounding_box=index == 0)
    options = visualizer.get_render_option()
    options.background_color = np.array([0.055, 0.065, 0.08])
    options.point_size = 4.0
    options.light_on = False
    options.mesh_color_option = o3d.visualization.MeshColorOption.Color
    options.mesh_show_back_face = True
    control = visualizer.get_view_control()
    visualizer.reset_view_point(True)
    # Open3D's front vector points from the target toward the virtual camera.
    # The physical Femto camera is on the -Z side and observes along +Z.
    control.set_front([0.0, 0.0, -1.0])
    control.set_up([0.0, 1.0, 0.0])
    control.set_zoom(0.90)
    surface_vertices = np.asarray(geometries[0].vertices)
    if len(surface_vertices):
        control.set_lookat(np.median(surface_vertices, axis=0))
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    main()
