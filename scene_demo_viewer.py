"""Bilingual interactive viewer for complete scene reconstruction outputs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Callable

import numpy as np
from PIL import Image
import open3d as o3d
from open3d.visualization import gui, rendering

from fusion import read_geometry_safe
from scene_surface_picker import SurfacePicker
from scene_annotation_ui import AnnotationUI, surface_readout_text
from ui_i18n import LANGUAGE_NAMES, load_language, save_language, tr


ROOT = Path(__file__).resolve().parent

# Every scene object owns one stable geometry name so the render scene, the
# checkbox that toggles it and the fit-to-view bounds all address one object.
GEOMETRY_NAMES = {
    "mesh": "scene_mesh",
    "mesh_wireframe": "scene_mesh_wireframe",
    "cloud": "scene_pointcloud",
    "raw_cloud": "scene_raw_pointcloud",
    "gaussian": "gaussian_points",
    "camera_path": "camera_path",
    "camera_frustums": "camera_frustums",
    "grid": "scene_grid",
}
SCENE_OBJECT_KEYS = ("mesh", "cloud", "raw_cloud", "gaussian", "camera_path", "camera_frustums")
MESH_SHADING_KEYS = ("texture", "shaded", "wireframe", "normals", "vertex_color")
CLOUD_SHADING_KEYS = ("rgb", "uniform", "normals", "depth")
VIEW_KEYS = ("front", "back", "left", "right", "top", "bottom")
VIEW_BUTTON_ROWS = (("front", "back"), ("left", "right"), ("top", "bottom"))

# Filament material names shipped in the Open3D wheel; O3DVisualizer switches
# between exactly these for its own shading modes.
SHADER_LIT = "defaultLit"
SHADER_UNLIT = "defaultUnlit"
SHADER_NORMALS = "normals"
SHADER_DEPTH = "depth"
SHADER_LINE = "unlitLine"
MESH_SHADERS = {
    "texture": SHADER_UNLIT,
    "shaded": SHADER_LIT,
    "wireframe": SHADER_LIT,  # the filled surface is hidden; lines carry the mode
    "normals": SHADER_NORMALS,
    "vertex_color": SHADER_UNLIT,
}
CLOUD_SHADERS = {
    "rgb": SHADER_UNLIT,
    "uniform": SHADER_UNLIT,  # applied to a uniformly painted copy of the cloud
    "normals": SHADER_NORMALS,
    "depth": SHADER_DEPTH,
}

UNIFORM_CLOUD_COLOR = (0.62, 0.78, 0.94)
WIREFRAME_COLOR = (0.55, 0.78, 0.95)
CAMERA_PATH_COLOR = (1.0, 0.20, 0.06)
CAMERA_FRUSTUM_COLOR = (0.35, 0.85, 1.0)
GRID_COLOR = (0.24, 0.31, 0.40)

FIELD_OF_VIEW_DEG = 60.0
MINIMUM_VIEW_EXTENT_M = 0.05
CAMERA_FRUSTUM_DEPTH_FRACTION = 0.12
CAMERA_FRUSTUM_DEPTH_RANGE_M = (0.05, 0.60)
GRID_STEP_CANDIDATES_M = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0)
GRID_MAX_DIVISIONS = 24

# Eye offset direction and screen-up vector for each standard view, expressed in
# the Orbbec/OpenCV world frame used on disk (+X right, +Y down, +Z forward).
VIEW_DIRECTIONS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "front": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    "back": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    "left": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "right": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    "top": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "bottom": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
}


def _open3d_ascii_resources() -> Path:
    """Mirror Filament assets because its Windows loader rejects Chinese paths."""
    source = Path(o3d.__file__).resolve().parent / "resources"
    if not source.is_dir():
        raise FileNotFoundError(f"Open3D resources are missing: {source}")
    cache = (
        Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
        / "FemtoSceneDemo"
        / f"open3d_resources_{o3d.__version__.replace('.', '_')}"
    )
    if not (cache / "defaultLit.filamat").exists():
        if cache.exists():
            shutil.rmtree(cache)
        shutil.copytree(source, cache)
    return cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View Mesh, point cloud, Gaussian init and camera path")
    parser.add_argument("reconstruction", nargs="?", type=Path)
    parser.add_argument("--lang", choices=("zh", "en"), help="Initial UI language")
    return parser.parse_args()


def latest_reconstruction() -> Path:
    marker = ROOT / "outputs" / "latest_complete_demo.txt"
    if not marker.exists():
        raise FileNotFoundError("No outputs/latest_complete_demo.txt marker was found")
    path = Path(marker.read_text(encoding="utf-8-sig").strip()).resolve()
    if not (path / "reconstruction_report.json").is_file():
        raise FileNotFoundError(f"No reconstruction_report.json in {path}")
    return path


def _line_set(points: np.ndarray, lines: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    geometry.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    geometry.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return geometry


def camera_path_from_report(report: dict[str, Any]) -> o3d.geometry.LineSet | None:
    """Connect the reconstructed camera centres; a single view has no path."""
    centers = [
        item["camera_center_m"]
        for item in report.get("frames", [])
        if item.get("camera_center_m") is not None
    ]
    if len(centers) < 2:
        return None
    points = np.asarray(centers, dtype=np.float64)
    lines = np.column_stack((np.arange(len(points) - 1), np.arange(1, len(points))))
    return _line_set(points, lines, CAMERA_PATH_COLOR)


def load_capture_poses(report: dict[str, Any]) -> tuple[list[np.ndarray], dict[str, float] | None]:
    """Read validated per-frame ``T_WC`` and colour intrinsics from the source dataset.

    Camera poses are never invented here: when the immutable capture dataset is
    not readable the caller must disable the frustum control instead.
    """
    dataset_value = report.get("source_dataset")
    if not dataset_value:
        return [], None
    dataset = Path(str(dataset_value))
    frames_path = dataset / "frames.jsonl"
    intrinsic_path = dataset / "calibration" / "color_intrinsic.json"
    if not frames_path.is_file() or not intrinsic_path.is_file():
        return [], None
    try:
        intrinsic = json.loads(intrinsic_path.read_text(encoding="utf-8-sig"))
        poses: list[np.ndarray] = []
        for line in frames_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            frame = json.loads(line)
            record = json.loads((dataset / frame["pose"]).read_text(encoding="utf-8-sig"))
            if not record.get("pose_valid", False):
                return [], None
            matrix = np.asarray(record["T_WC"], dtype=np.float64)
            rotation = matrix[:3, :3] if matrix.shape == (4, 4) else np.empty((0, 0))
            if (
                matrix.shape != (4, 4)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
            ):
                return [], None
            poses.append(matrix)
    except (OSError, ValueError, TypeError, KeyError):
        return [], None
    if not poses or any(key not in intrinsic for key in ("fx", "fy", "cx", "cy", "width", "height")):
        return [], None
    return poses, intrinsic


def camera_frustums(
    poses: list[np.ndarray], intrinsic: dict[str, float], depth_m: float
) -> o3d.geometry.LineSet | None:
    """Draw the measured colour-camera view volume for every validated pose."""
    if not poses or depth_m <= 0.0:
        return None
    fx, fy = float(intrinsic["fx"]), float(intrinsic["fy"])
    cx, cy = float(intrinsic["cx"]), float(intrinsic["cy"])
    width, height = float(intrinsic["width"]), float(intrinsic["height"])
    if fx <= 0.0 or fy <= 0.0 or width <= 0.0 or height <= 0.0:
        return None
    corners_px = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)
    rays = np.column_stack(
        (
            (corners_px[:, 0] - cx) / fx,
            (corners_px[:, 1] - cy) / fy,
            np.ones(len(corners_px)),
        )
    ) * depth_m
    local = np.vstack((np.zeros((1, 3), dtype=np.float64), rays))
    edges = np.array(
        [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int32
    )
    points = np.vstack([local @ pose[:3, :3].T + pose[:3, 3] for pose in poses])
    lines = np.vstack([edges + index * len(local) for index in range(len(poses))])
    return _line_set(points, lines, CAMERA_FRUSTUM_COLOR)


def grid_step_m(span_m: float) -> float:
    """Pick a round grid spacing that keeps the helper grid legible."""
    for candidate in GRID_STEP_CANDIDATES_M:
        if span_m / candidate <= GRID_MAX_DIVISIONS:
            return candidate
    return GRID_STEP_CANDIDATES_M[-1]


def reference_grid(bounds: o3d.geometry.AxisAlignedBoundingBox) -> o3d.geometry.LineSet:
    """Build an XZ reference grid at the lowest measured surface (+Y is down)."""
    lower = np.asarray(bounds.get_min_bound(), dtype=np.float64)
    upper = np.asarray(bounds.get_max_bound(), dtype=np.float64)
    center = (lower + upper) / 2.0
    span = max(float(upper[0] - lower[0]), float(upper[2] - lower[2]), 0.2) * 1.4
    step = grid_step_m(span)
    half = math.ceil(span / (2.0 * step)) * step
    ticks = np.arange(-half, half + step * 0.5, step)
    plane_y = float(upper[1])
    points: list[list[float]] = []
    lines: list[list[int]] = []
    for tick in ticks:
        base = len(points)
        points.extend(
            [
                [center[0] + tick, plane_y, center[2] - half],
                [center[0] + tick, plane_y, center[2] + half],
                [center[0] - half, plane_y, center[2] + tick],
                [center[0] + half, plane_y, center[2] + tick],
            ]
        )
        lines.extend([[base, base + 1], [base + 2, base + 3]])
    return _line_set(np.asarray(points), np.asarray(lines), GRID_COLOR)


def inflated_bounds(
    bounds: o3d.geometry.AxisAlignedBoundingBox, minimum_extent_m: float
) -> o3d.geometry.AxisAlignedBoundingBox:
    """Keep degenerate bounds usable for near/far plane setup."""
    lower = np.asarray(bounds.get_min_bound(), dtype=np.float64)
    upper = np.asarray(bounds.get_max_bound(), dtype=np.float64)
    center = (lower + upper) / 2.0
    half = np.maximum((upper - lower) / 2.0, minimum_extent_m / 2.0)
    return o3d.geometry.AxisAlignedBoundingBox(center - half, center + half)


@dataclass
class SceneObject:
    """One named geometry inside the render scene plus its display state."""

    key: str
    name: str
    geometry: o3d.geometry.Geometry3D
    material: rendering.MaterialRecord
    visible: bool = False
    fittable: bool = True


class SceneObjectManager:
    """The only owner of geometry names inside one ``rendering.Open3DScene``.

    Mesh, point cloud and line set objects live side by side; visibility is a
    per-object flag instead of a single mutually exclusive display mode.
    """

    def __init__(self, scene: rendering.Open3DScene) -> None:
        self._scene = scene
        self._objects: dict[str, SceneObject] = {}

    def has(self, key: str) -> bool:
        return key in self._objects

    def geometry_of(self, key: str) -> o3d.geometry.Geometry3D | None:
        record = self._objects.get(key)
        return record.geometry if record else None

    def add_geometry(
        self,
        key: str,
        geometry: o3d.geometry.Geometry3D,
        material: rendering.MaterialRecord,
        *,
        visible: bool = False,
        fittable: bool = True,
    ) -> SceneObject:
        if key in self._objects:
            self.remove_geometry(key)
        name = GEOMETRY_NAMES[key]
        self._scene.add_geometry(name, geometry, material)
        record = SceneObject(
            key=key, name=name, geometry=geometry, material=material, visible=visible, fittable=fittable
        )
        self._objects[key] = record
        self._scene.show_geometry(name, visible)
        return record

    def remove_geometry(self, key: str) -> None:
        record = self._objects.pop(key, None)
        if record is not None and self._scene.has_geometry(record.name):
            self._scene.remove_geometry(record.name)

    def replace_geometry(self, key: str, geometry: o3d.geometry.Geometry3D) -> None:
        """Re-upload one object, keeping its name, material and visibility."""
        record = self._objects.get(key)
        if record is None or record.geometry is geometry:
            return
        if self._scene.has_geometry(record.name):
            self._scene.remove_geometry(record.name)
        record.geometry = geometry
        self._scene.add_geometry(record.name, geometry, record.material)
        self._scene.show_geometry(record.name, record.visible)

    def set_geometry_visible(self, key: str, visible: bool) -> None:
        record = self._objects.get(key)
        if record is None:
            return
        record.visible = bool(visible)
        self._scene.show_geometry(record.name, record.visible)

    def is_visible(self, key: str) -> bool:
        record = self._objects.get(key)
        return bool(record and record.visible)

    def update_material(self, key: str, material: rendering.MaterialRecord | None = None) -> None:
        """Change shading or point size without re-uploading any geometry."""
        record = self._objects.get(key)
        if record is None:
            return
        if material is not None:
            record.material = material
        self._scene.modify_geometry_material(record.name, record.material)

    def visible_bounds(self) -> o3d.geometry.AxisAlignedBoundingBox | None:
        return self._bounds(lambda record: record.visible and record.fittable)

    def fittable_bounds(self) -> o3d.geometry.AxisAlignedBoundingBox | None:
        return self._bounds(lambda record: record.fittable)

    def _bounds(self, accept: Callable[[SceneObject], bool]) -> o3d.geometry.AxisAlignedBoundingBox | None:
        lower: np.ndarray | None = None
        upper: np.ndarray | None = None
        for record in self._objects.values():
            if not accept(record) or record.geometry.is_empty():
                continue
            minimum = np.asarray(record.geometry.get_min_bound(), dtype=np.float64)
            maximum = np.asarray(record.geometry.get_max_bound(), dtype=np.float64)
            if not (np.isfinite(minimum).all() and np.isfinite(maximum).all()):
                continue
            lower = minimum if lower is None else np.minimum(lower, minimum)
            upper = maximum if upper is None else np.maximum(upper, maximum)
        if lower is None or upper is None:
            return None
        return o3d.geometry.AxisAlignedBoundingBox(lower, upper)


class FemtoSceneViewer(AnnotationUI):
    """Open3D SceneWidget viewer with project-owned, runtime-localized controls."""

    def __init__(self, reconstruction: Path, language: str) -> None:
        self.reconstruction = reconstruction.resolve()
        self.language = language
        self._updating_ui = False
        self.mesh_shading = "shaded"
        self.cloud_shading = "rgb"
        self.point_size = 3.0
        self.selected_surface = None
        self._surface_picker = None
        self._pick_busy = False
        self._pick_generation = 0
        self._closing = False
        self._pick_status = 'ready'

        report_path = self.reconstruction / "reconstruction_report.json"
        self.report = json.loads(report_path.read_text(encoding="utf-8"))
        mesh_outputs = self.report.get("outputs", {}).get("mesh", {})
        mesh_path = Path(mesh_outputs.get("ply", self.reconstruction / "scene_tsdf_mesh_m.ply"))
        self.mesh = read_geometry_safe(mesh_path, mesh=True)
        self.mesh.compute_vertex_normals()
        self._init_annotations(mesh_path)
        self.texture_image = None
        self.textured_mesh = None
        if mesh_outputs.get("texture") and (mesh_outputs.get("vertex_uv") or mesh_outputs.get("triangle_uv")):
            if mesh_outputs.get("triangle_uv"):
                corners = np.load(mesh_outputs["triangle_uv"], allow_pickle=False)
                if corners.shape != (len(self.mesh.triangles), 3, 2) or not np.isfinite(corners).all():
                    raise ValueError("Mesh corner UV does not match its triangles")
            else:
                uv = np.load(mesh_outputs["vertex_uv"], allow_pickle=False)
                if uv.shape != (len(self.mesh.vertices), 2) or not np.isfinite(uv).all():
                    raise ValueError("Mesh texture UV does not match its vertices")
                corners = uv[np.asarray(self.mesh.triangles)]
            with Image.open(mesh_outputs["texture"]) as source:
                self.texture_image = o3d.geometry.Image(np.ascontiguousarray(source.convert("RGB")))
            self.textured_mesh = o3d.geometry.TriangleMesh(self.mesh)
            self.textured_mesh.vertex_colors = o3d.utility.Vector3dVector()
            self.textured_mesh.triangle_uvs = o3d.utility.Vector2dVector(corners.reshape(-1, 2))
            self.mesh_shading = "texture"
        self.cloud = read_geometry_safe(self.reconstruction / "scene_fused_points_m.ply", mesh=False)
        raw_path = self.reconstruction / "scene_raw_points_m.ply"
        self.raw_cloud = read_geometry_safe(raw_path, mesh=False) if raw_path.is_file() else None
        self.gaussian = read_geometry_safe(
            self.reconstruction / "gaussian" / "gaussian_init_preview.ply", mesh=False
        )
        if len(self.mesh.vertices) == 0 or len(self.mesh.triangles) == 0:
            raise ValueError("Reconstruction Mesh is empty")
        if len(self.cloud.points) == 0 or len(self.gaussian.points) == 0:
            raise ValueError("Reconstruction point cloud or Gaussian preview is empty")
        # The normal shader needs per-point normals; estimating them here only
        # affects the in-memory display copy, never the exported PLY.
        for cloud in (self.cloud, self.gaussian):
            if not cloud.has_normals():
                cloud.estimate_normals()

        self.camera_path = camera_path_from_report(self.report)
        self.capture_poses, self.capture_intrinsic = load_capture_poses(self.report)
        self.frustums: o3d.geometry.LineSet | None = None
        self._wireframe: o3d.geometry.LineSet | None = None
        self._uniform_clouds: dict[str, o3d.geometry.PointCloud] = {}
        # The professional viewer starts with the two measured geometric layers
        # together. Every layer remains independently switchable afterwards.
        self.scene_visible = {
            key: key in {"mesh", "cloud"} for key in SCENE_OBJECT_KEYS
        }
        if self.textured_mesh is not None:
            self.scene_visible["cloud"] = False
        self.helper_visible = {"axes": True, "ground": False, "grid": False}

        app = gui.Application.instance
        self.window = app.create_window("Femto", 1360, 860)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background(np.array([0.025, 0.035, 0.05, 1.0], dtype=np.float32))
        self.scene_widget.scene.show_skybox(False)
        self.scene_widget.scene.show_axes(self.helper_visible["axes"])
        self.scene_widget.scene.show_ground_plane(False, rendering.Scene.GroundPlane.XZ)
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE)
        self.window.add_child(self.scene_widget)

        self.materials = self._create_materials()
        self.objects = SceneObjectManager(self.scene_widget.scene)
        self._register_scene_objects()

        self._build_panel()
        self.window.set_on_layout(self._on_layout)
        self.scene_widget.set_on_mouse(self._on_pick_mouse)
        self.window.set_on_close(self._on_picker_close)
        self._apply_language()
        self.update_mesh_material()
        self.update_pointcloud_material()
        if len(self.report.get("frames", [])) <= 1:
            self.sensor_camera_view()
        else:
            self.fit_visible_geometry()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    @staticmethod
    def _create_materials() -> dict[str, rendering.MaterialRecord]:
        mesh = rendering.MaterialRecord()
        mesh.shader = SHADER_LIT
        mesh.base_roughness = 0.82
        mesh.base_metallic = 0.0
        wireframe = rendering.MaterialRecord()
        wireframe.shader = SHADER_LINE
        wireframe.line_width = 1.0
        cloud = rendering.MaterialRecord()
        cloud.shader = SHADER_UNLIT
        cloud.point_size = 3.0
        gaussian = rendering.MaterialRecord()
        gaussian.shader = SHADER_UNLIT
        gaussian.point_size = 3.0
        camera_path = rendering.MaterialRecord()
        camera_path.shader = SHADER_LINE
        camera_path.line_width = 4.0
        frustums = rendering.MaterialRecord()
        frustums.shader = SHADER_LINE
        frustums.line_width = 2.0
        grid = rendering.MaterialRecord()
        grid.shader = SHADER_LINE
        grid.line_width = 1.0
        return {
            "mesh": mesh,
            "mesh_wireframe": wireframe,
            "cloud": cloud,
            "gaussian": gaussian,
            "camera_path": camera_path,
            "camera_frustums": frustums,
            "grid": grid,
        }

    def _register_scene_objects(self) -> None:
        self.objects.add_geometry(
            "mesh", self.mesh, self.materials["mesh"], visible=self.scene_visible["mesh"]
        )
        self.objects.add_geometry(
            "cloud", self.cloud, self.materials["cloud"], visible=self.scene_visible["cloud"]
        )
        if self.raw_cloud is not None:
            raw_material = rendering.MaterialRecord()
            raw_material.shader = SHADER_UNLIT
            raw_material.point_size = 2.0
            self.materials["raw_cloud"] = raw_material
            self.objects.add_geometry("raw_cloud", self.raw_cloud, raw_material, visible=False)
        self.objects.add_geometry(
            "gaussian",
            self.gaussian,
            self.materials["gaussian"],
            visible=self.scene_visible["gaussian"],
        )
        if self.camera_path is not None:
            self.objects.add_geometry(
                "camera_path",
                self.camera_path,
                self.materials["camera_path"],
                visible=self.scene_visible["camera_path"],
            )
        data_bounds = self.objects.fittable_bounds()
        if self.capture_poses and self.capture_intrinsic is not None and data_bounds is not None:
            diagonal = float(np.linalg.norm(np.asarray(data_bounds.get_extent(), dtype=np.float64)))
            depth_m = float(
                np.clip(
                    diagonal * CAMERA_FRUSTUM_DEPTH_FRACTION,
                    CAMERA_FRUSTUM_DEPTH_RANGE_M[0],
                    CAMERA_FRUSTUM_DEPTH_RANGE_M[1],
                )
            )
            self.frustums = camera_frustums(self.capture_poses, self.capture_intrinsic, depth_m)
            if self.frustums is not None:
                self.objects.add_geometry(
                    "camera_frustums",
                    self.frustums,
                    self.materials["camera_frustums"],
                    visible=self.scene_visible["camera_frustums"],
                )
        if data_bounds is not None:
            # The grid is a display aid, so it must never expand fit-to-view.
            self.objects.add_geometry(
                "grid",
                reference_grid(data_bounds),
                self.materials["grid"],
                visible=self.helper_visible["grid"],
                fittable=False,
            )

    def is_object_available(self, key: str) -> bool:
        return self.objects.has(key)

    # ------------------------------------------------------------------
    # Panel
    # ------------------------------------------------------------------
    def _build_panel(self) -> None:
        em = self.window.theme.font_size
        self.panel_width = int(23.0 * em)
        # The panel now holds four sections, so it scrolls on short windows.
        container = getattr(gui, "ScrollableVert", gui.Vert)
        self.panel = container(0.4 * em, gui.Margins(0.8 * em, 0.8 * em, 0.8 * em, 0.8 * em))
        if hasattr(self.panel, "background_color"):
            self.panel.background_color = gui.Color(0.07, 0.105, 0.145, 1.0)
        self.section_labels: dict[str, gui.Label] = {}
        self.object_checkboxes: dict[str, gui.Checkbox] = {}
        self.object_labels: dict[str, gui.Label] = {}

        self.heading_label = gui.Label("")
        self.panel.add_child(self.heading_label)
        self.language_label = gui.Label("")
        self.panel.add_child(self.language_label)
        self.language_combo = gui.Combobox()
        self.language_combo.add_item(LANGUAGE_NAMES["zh"])
        self.language_combo.add_item(LANGUAGE_NAMES["en"])
        self.language_combo.set_on_selection_changed(self._on_language_changed)
        self.panel.add_child(self.language_combo)

        self._add_section("pick")
        self.pick_checkbox, self.pick_mode_label = self._add_checkbox_row(lambda checked: None)
        self.pick_checkbox.checked = False
        self.pick_result_label = gui.Label("")
        self.panel.add_child(self.pick_result_label)
        self.clear_pick_button = gui.Button("")
        self.clear_pick_button.set_on_clicked(self.clear_surface_pick)
        self.panel.add_child(self.clear_pick_button)
        self._build_annotations()

        self._add_section("scene")
        for key in SCENE_OBJECT_KEYS:
            checkbox, label = self._add_checkbox_row(
                lambda checked, key=key: self._on_scene_toggle(key, checked)
            )
            checkbox.checked = self.scene_visible[key]
            self.object_checkboxes[key] = checkbox
            self.object_labels[key] = label

        self._add_section("render")
        self.mesh_display_label = gui.Label("")
        self.panel.add_child(self.mesh_display_label)
        self.mesh_combo = gui.Combobox()
        self.mesh_combo.set_on_selection_changed(self._on_mesh_shading_changed)
        self.panel.add_child(self.mesh_combo)
        self.cloud_shading_label = gui.Label("")
        self.panel.add_child(self.cloud_shading_label)
        self.cloud_combo = gui.Combobox()
        self.cloud_combo.set_on_selection_changed(self._on_cloud_shading_changed)
        self.panel.add_child(self.cloud_combo)
        point_row = gui.Horiz(0.5 * em)
        self.point_size_label = gui.Label("")
        point_row.add_child(self.point_size_label)
        self.point_size_slider = gui.Slider(gui.Slider.INT)
        self.point_size_slider.set_limits(1, 12)
        self.point_size_slider.int_value = int(self.point_size)
        self.point_size_slider.set_on_value_changed(self.update_point_size)
        point_row.add_child(self.point_size_slider)
        self.panel.add_child(point_row)

        self._add_section("helper")
        self.axes_checkbox, self.axes_label = self._add_checkbox_row(self._on_show_axes)
        self.ground_checkbox, self.ground_label = self._add_checkbox_row(self._on_show_ground)
        self.grid_checkbox, self.grid_label = self._add_checkbox_row(self._on_show_grid)
        self.axes_checkbox.checked = self.helper_visible["axes"]
        self.ground_checkbox.checked = self.helper_visible["ground"]
        self.grid_checkbox.checked = self.helper_visible["grid"]

        self._add_section("view")
        self.sensor_button = gui.Button("")
        self.sensor_button.set_on_clicked(self.sensor_camera_view)
        self.panel.add_child(self.sensor_button)
        self.view_buttons: dict[str, gui.Button] = {}
        for row_keys in VIEW_BUTTON_ROWS:
            row = gui.Horiz(0.35 * em)
            for key in row_keys:
                button = gui.Button("")
                button.horizontal_padding_em = 0.3
                button.vertical_padding_em = 0.1
                button.set_on_clicked(lambda key=key: self.update_camera_view(key))
                row.add_child(button)
                self.view_buttons[key] = button
            self.panel.add_child(row)
        self.fit_button = gui.Button("")
        self.fit_button.set_on_clicked(self.fit_visible_geometry)
        self.panel.add_child(self.fit_button)

        self.panel.add_fixed(0.6 * em)
        self.help_label = gui.Label("")
        self.panel.add_child(self.help_label)
        self.warning_label = gui.Label("")
        self.panel.add_child(self.warning_label)
        self.window.add_child(self.panel)

    def _add_section(self, key: str) -> None:
        em = self.window.theme.font_size
        self.panel.add_fixed(0.45 * em)
        label = gui.Label("")
        label.text_color = gui.Color(0.53, 0.72, 0.94, 1.0)
        self.panel.add_child(label)
        self.section_labels[key] = label

    def _add_checkbox_row(self, on_checked: Callable[[bool], None]) -> tuple[gui.Checkbox, gui.Label]:
        em = self.window.theme.font_size
        row = gui.Horiz(0.35 * em)
        checkbox = gui.Checkbox("")
        checkbox.set_on_checked(on_checked)
        label = gui.Label("")
        row.add_child(checkbox)
        row.add_child(label)
        self.panel.add_child(row)
        return checkbox, label

    def _on_layout(self, _layout_context: gui.LayoutContext) -> None:
        content = self.window.content_rect
        panel_width = min(self.panel_width, max(300, int(content.width * 0.36)))
        scene_width = max(320, content.width - panel_width)
        self.scene_widget.frame = gui.Rect(content.x, content.y, scene_width, content.height)
        self.panel.frame = gui.Rect(content.x + scene_width, content.y, panel_width, content.height)
        self._layout_annotation_readouts(
            _layout_context, max(100, panel_width - int(1.6 * self.window.theme.font_size)))

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------
    def _t(self, key: str, **values: object) -> str:
        return tr(self.language, key, **values)

    def _apply_language(self) -> None:
        self._updating_ui = True
        try:
            self.window.title = self._t(
                "viewer.window_title", name=self.reconstruction.parent.name
            )
            self.heading_label.text = self._t("viewer.heading")
            self.pick_mode_label.text = self._t("viewer.pick.mode")
            self.clear_pick_button.text = self._t("viewer.pick.clear")
            self._apply_annotation_language()
            self._update_pick_label()
            self.language_label.text = self._t("viewer.language")
            self.language_combo.selected_index = 0 if self.language == "zh" else 1
            for key, label in self.section_labels.items():
                # Avoid box-drawing glyphs: Open3D's Windows font fallback can
                # render those as question marks even when Chinese glyphs work.
                label.text = self._t(f"viewer.section.{key}")
            for key, label in self.object_labels.items():
                label.text = self._t(f"viewer.object.{key}")
            self.mesh_display_label.text = self._t("viewer.mesh_display")
            self.mesh_combo.clear_items()
            for key in MESH_SHADING_KEYS:
                self.mesh_combo.add_item(self._t(f"viewer.mesh_mode.{key}"))
            self.mesh_combo.selected_index = MESH_SHADING_KEYS.index(self.mesh_shading)
            self.cloud_shading_label.text = self._t("viewer.cloud_shading")
            self.cloud_combo.clear_items()
            for key in CLOUD_SHADING_KEYS:
                self.cloud_combo.add_item(self._t(f"viewer.cloud_mode.{key}"))
            self.cloud_combo.selected_index = CLOUD_SHADING_KEYS.index(self.cloud_shading)
            self.axes_label.text = self._t("viewer.show_axes")
            self.ground_label.text = self._t("viewer.show_ground")
            self.grid_label.text = self._t("viewer.show_grid")
            self.sensor_button.text = self._t("viewer.sensor_view")
            for key, button in self.view_buttons.items():
                button.text = self._t(f"viewer.view.{key}")
            self.fit_button.text = self._t("viewer.fit_all")
            self.help_label.text = self._t("viewer.mouse_help")
            self._update_point_size_label()
            self._update_control_states()
            self._update_warning()
            self.window.set_needs_layout()
            self.window.post_redraw()
        finally:
            self._updating_ui = False

    def _on_language_changed(self, _text: str, index: int) -> None:
        if self._updating_ui:
            return
        self.language = "zh" if index == 0 else "en"
        save_language(self.language)
        self._apply_language()

    def _update_pick_label(self):
        self.pick_result_label.text = surface_readout_text(
            self.annotations, self.selected_surface, self._pick_status, self.language,
            getattr(self, '_readout_columns', 38))
        self.pick_result_label.tooltip = self.annotations.source
        self._refresh_annotation_choices()
        self._update_annotation_label()
        self.window.set_needs_layout()

    def clear_surface_pick(self):
        self._pick_generation += 1
        self.selected_surface = None
        self._selected_annotation_index = None
        self._selected_region_index = None
        self._pick_status = 'ready'
        if self.scene_widget.scene.has_geometry('selected_surface_marker'):
            self.scene_widget.scene.remove_geometry('selected_surface_marker')
        self._update_pick_label()
        self.scene_widget.force_redraw()

    def _on_picker_close(self):
        self._closing = True
        self._pick_generation += 1
        return True

    def _on_pick_mouse(self, event):
        enabled = self.pick_checkbox.checked or event.is_modifier_down(gui.KeyModifier.CTRL)
        if not enabled:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            frame = self.scene_widget.frame
            if self.annotation_mode.selected_index == 2:
                self._polygon_click(event.x - frame.x, event.y - frame.y)
            else:
                self.pick_surface_pixel(event.x - frame.x, event.y - frame.y)
            return gui.Widget.EventCallbackResult.HANDLED
        if event.type == gui.MouseEvent.Type.DRAG and event.is_button_down(gui.MouseButton.LEFT):
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def pick_surface_pixel(self, x, y):
        frame = self.scene_widget.frame
        # Every attempted click invalidates the previous result, including clicks
        # outside the viewport or while an older asynchronous request is pending.
        self.clear_surface_pick()
        if not (np.isfinite([x, y]).all() and 0 <= x < frame.width and 0 <= y < frame.height):
            self._pick_status = 'miss'
            self._update_pick_label()
            return
        if self._pick_busy:
            self._pick_status = 'retry'
            self._update_pick_label()
            return
        if not self.scene_visible['mesh']:
            self._pick_status = 'show_mesh'
            self._update_pick_label()
            return
        camera = self.scene_widget.scene.camera
        view_matrix = np.asarray(camera.get_view_matrix()).copy()
        projection_matrix = np.asarray(camera.get_projection_matrix()).copy()
        viewport_size = (frame.width, frame.height)
        try:
            near = camera.unproject(x, y, 0., frame.width, frame.height)
            far = camera.unproject(x, y, .999999, frame.width, frame.height)
        except Exception as exc:
            self._pick_error = str(exc)
            self._pick_status = 'error'
            self._update_pick_label()
            return
        generation = self._pick_generation
        mesh, annotations = self.mesh, self.annotations
        name, note = self.annotation_name.text_value.strip(), self.annotation_note.text_value.strip()
        picker = self._surface_picker
        self._pick_busy = True
        self._pick_status = 'busy'
        self._update_pick_label()
        def worker():
            hit, error = None, None
            active_picker = picker
            try:
                if active_picker is None:
                    active_picker = SurfacePicker(mesh)
                hit = active_picker.pick(near, far)
            except Exception as exc:
                error = str(exc)
            def finish():
                self._pick_busy = False
                if (self._closing or generation != self._pick_generation
                        or self.mesh is not mesh or self.annotations is not annotations):
                    return
                self._surface_picker = active_picker
                current_camera = self.scene_widget.scene.camera
                current_frame = self.scene_widget.frame
                if (not self.scene_visible['mesh']
                        or viewport_size != (current_frame.width, current_frame.height)
                        or not np.allclose(view_matrix, current_camera.get_view_matrix())
                        or not np.allclose(projection_matrix, current_camera.get_projection_matrix())):
                    self._pick_status = 'ready'
                    self._update_pick_label()
                    return
                self._pick_error = error
                self._pick_status = 'error' if error else ('hit' if hit else 'miss')
                if hit and not error:
                    try:
                        self._accept_annotation_point(hit, name=name, note=note)
                    except (ValueError, KeyError, TypeError) as exc:
                        self.clear_surface_pick()
                        self._pick_error = str(exc)
                        self._pick_status = 'error'
                self._update_pick_label()
                self.window.set_needs_layout()
                self.scene_widget.force_redraw()
            if not self._closing:
                gui.Application.instance.post_to_main_thread(self.window, finish)
        threading.Thread(target=worker, daemon=True, name='surface-pick').start()

    # ------------------------------------------------------------------
    # Scene visibility
    # ------------------------------------------------------------------
    def _on_scene_toggle(self, key: str, checked: bool) -> None:
        if self._updating_ui:
            return
        self.set_object_visible(key, checked)

    def set_object_visible(self, key: str, visible: bool) -> None:
        """Toggle one scene object; Mesh, point cloud and line sets coexist."""
        self.scene_visible[key] = bool(visible) and self.is_object_available(key)
        if key == 'mesh' and not self.scene_visible[key]:
            self.clear_surface_pick()
        if key == "mesh":
            self._apply_mesh_visibility()
        else:
            self.objects.set_geometry_visible(key, self.scene_visible[key])
        self._update_control_states()
        self._update_warning()
        self.scene_widget.force_redraw()

    def _apply_mesh_visibility(self) -> None:
        """Wireframe replaces the filled surface instead of faking line shading."""
        show = self.scene_visible["mesh"]
        wireframe = show and self.mesh_shading == "wireframe"
        self.objects.set_geometry_visible("mesh", show and not wireframe)
        if self.objects.has("mesh_wireframe"):
            self.objects.set_geometry_visible("mesh_wireframe", wireframe)

    def _update_control_states(self) -> None:
        enabled_color = gui.Color(0.90, 0.94, 0.98, 1.0)
        disabled_color = gui.Color(0.44, 0.51, 0.60, 1.0)
        previous = self._updating_ui
        self._updating_ui = True
        try:
            for key, checkbox in self.object_checkboxes.items():
                available = self.is_object_available(key)
                checkbox.enabled = available
                checkbox.checked = self.scene_visible[key]
                self.object_labels[key].text_color = enabled_color if available else disabled_color
            clouds_visible = self.scene_visible["cloud"] or self.scene_visible["gaussian"]
            self.mesh_combo.enabled = self.scene_visible["mesh"]
            self.cloud_combo.enabled = clouds_visible
            self.point_size_slider.enabled = clouds_visible or self.scene_visible.get("raw_cloud", False)
        finally:
            self._updating_ui = previous

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _on_mesh_shading_changed(self, _text: str, index: int) -> None:
        if self._updating_ui or not 0 <= index < len(MESH_SHADING_KEYS):
            return
        self.mesh_shading = MESH_SHADING_KEYS[index]
        previous = self._updating_ui
        self._updating_ui = True
        self.mesh_combo.selected_index = index
        self._updating_ui = previous
        self.update_mesh_material()
        self._post_selection_redraw()

    def update_mesh_material(self) -> None:
        """Apply the Mesh display mode. Wireframe uses a real TriangleMesh→LineSet."""
        if self.mesh_shading == "wireframe":
            self._ensure_wireframe()
        material = self.materials["mesh"]
        material.shader = MESH_SHADERS[self.mesh_shading]
        textured = self.mesh_shading == "texture" and self.textured_mesh is not None
        material.albedo_img = self.texture_image if textured else None
        material.base_color = [1., 1., 1., 1.]
        self.objects.replace_geometry("mesh", self.textured_mesh if textured else self.mesh)
        self.objects.update_material("mesh", material)
        self._apply_mesh_visibility()
        self.scene_widget.force_redraw()

    def _ensure_wireframe(self) -> None:
        if self.objects.has("mesh_wireframe"):
            return
        self._wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(self.mesh)
        line_count = len(self._wireframe.lines)
        self._wireframe.colors = o3d.utility.Vector3dVector(
            np.tile(WIREFRAME_COLOR, (line_count, 1))
        )
        self.objects.add_geometry(
            "mesh_wireframe", self._wireframe, self.materials["mesh_wireframe"]
        )

    def _on_cloud_shading_changed(self, _text: str, index: int) -> None:
        if self._updating_ui or not 0 <= index < len(CLOUD_SHADING_KEYS):
            return
        self.cloud_shading = CLOUD_SHADING_KEYS[index]
        previous = self._updating_ui
        self._updating_ui = True
        self.cloud_combo.selected_index = index
        self._updating_ui = previous
        self.update_pointcloud_material()
        self._post_selection_redraw()

    def _post_selection_redraw(self) -> None:
        """Refresh ComboBox text after Open3D finishes closing its popup."""
        gui.Application.instance.post_to_main_thread(self.window, self.window.post_redraw)

    def update_pointcloud_material(self) -> None:
        """Apply point-cloud colour without changing the Gaussian diagnostic layer."""
        material = self.materials["cloud"]
        material.shader = CLOUD_SHADERS[self.cloud_shading]
        material.point_size = self.point_size
        self.objects.update_material("cloud", material)
        # Uniform colour needs a painted in-memory copy; every other mode is a
        # material switch on the original measured colours. No file is re-read.
        self.objects.replace_geometry("cloud", self._cloud_geometry("cloud"))
        self.scene_widget.force_redraw()

    def _cloud_geometry(self, key: str) -> o3d.geometry.PointCloud:
        source = self.cloud if key == "cloud" else self.gaussian
        if self.cloud_shading != "uniform":
            return source
        painted = self._uniform_clouds.get(key)
        if painted is None:
            painted = o3d.geometry.PointCloud(source)
            painted.paint_uniform_color(UNIFORM_CLOUD_COLOR)
            self._uniform_clouds[key] = painted
        return painted

    def update_point_size(self, value: float) -> None:
        """Point size is a material parameter, so no geometry is re-uploaded."""
        if self._updating_ui:
            return
        self.point_size = float(max(1, int(round(value))))
        for key in ("cloud", "gaussian"):
            self.materials[key].point_size = self.point_size
            self.objects.update_material(key)
        if "raw_cloud" in self.materials:
            self.materials["raw_cloud"].point_size = self.point_size
            self.objects.update_material("raw_cloud")
        self._update_point_size_label()
        self._post_selection_redraw()
        self.scene_widget.force_redraw()

    def _update_point_size_label(self) -> None:
        self.point_size_label.text = f"{self._t('viewer.point_size')}: {int(self.point_size)}"

    def _update_warning(self) -> None:
        key = "viewer.multi_warning" if self.objects.has("camera_path") else "viewer.single_warning"
        text = self._t(key)
        if self.scene_visible["gaussian"]:
            text += "\n\n" + self._t("viewer.gaussian_warning")
        if not self.is_object_available("camera_frustums"):
            text += "\n\n" + self._t("viewer.camera_pose_missing")
        self.warning_label.text = text

    # ------------------------------------------------------------------
    # Helper display
    # ------------------------------------------------------------------
    def _on_show_axes(self, checked: bool) -> None:
        self.helper_visible["axes"] = bool(checked)
        self.scene_widget.scene.show_axes(checked)
        self.scene_widget.force_redraw()

    def _on_show_ground(self, checked: bool) -> None:
        self.helper_visible["ground"] = bool(checked)
        self.scene_widget.scene.show_ground_plane(checked, rendering.Scene.GroundPlane.XZ)
        self.scene_widget.force_redraw()

    def _on_show_grid(self, checked: bool) -> None:
        self.helper_visible["grid"] = bool(checked) and self.objects.has("grid")
        self.objects.set_geometry_visible("grid", checked)
        self.scene_widget.force_redraw()

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def _camera_bounds(
        self, *, allow_fallback: bool = False
    ) -> o3d.geometry.AxisAlignedBoundingBox | None:
        """Return visible-object bounds; optional fallback is sensor-view only."""
        bounds = self.objects.visible_bounds()
        if bounds is None and allow_fallback:
            bounds = self.objects.fittable_bounds()
        return inflated_bounds(bounds, MINIMUM_VIEW_EXTENT_M) if bounds is not None else None

    def fit_visible_geometry(self) -> None:
        bounds = self._camera_bounds()
        if bounds is None:
            self.warning_label.text = self._t("viewer.no_visible_geometry")
            return
        self.scene_widget.setup_camera(FIELD_OF_VIEW_DEG, bounds, bounds.get_center())
        self.scene_widget.force_redraw()

    def update_camera_view(self, key: str) -> None:
        """Camera-only operation: no geometry is added, removed or reloaded."""
        bounds = self._camera_bounds()
        if bounds is None:
            self.warning_label.text = self._t("viewer.no_visible_geometry")
            return
        center = np.asarray(bounds.get_center(), dtype=np.float64)
        radius = float(np.linalg.norm(np.asarray(bounds.get_extent(), dtype=np.float64))) * 0.5
        distance = max(radius / math.tan(math.radians(FIELD_OF_VIEW_DEG * 0.5)) * 1.15, 0.25)
        direction, up = VIEW_DIRECTIONS[key]
        eye = center + np.asarray(direction, dtype=np.float64) * distance
        # setup_camera establishes the projection near/far planes before the
        # explicit eye placement is applied.
        self.scene_widget.setup_camera(FIELD_OF_VIEW_DEG, bounds, bounds.get_center())
        self.scene_widget.look_at(
            center.astype(np.float32),
            eye.astype(np.float32),
            np.asarray(up, dtype=np.float32),
        )
        self.scene_widget.force_redraw()

    def sensor_camera_view(self) -> None:
        points = np.asarray(self.cloud.points)
        center = (
            np.median(points, axis=0).astype(np.float32)
            if len(points)
            else np.array([0.0, 0.0, 2.0], dtype=np.float32)
        )
        eye = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        # setup_camera establishes projection near/far planes before applying
        # the physical sensor-origin view.
        bounds = self._camera_bounds(allow_fallback=True)
        if bounds is None:
            self.warning_label.text = self._t("viewer.no_visible_geometry")
            return
        self.scene_widget.setup_camera(FIELD_OF_VIEW_DEG, bounds, bounds.get_center())
        self.scene_widget.look_at(center, eye, up)
        self.scene_widget.force_redraw()


def run_viewer(reconstruction: Path, language: str | None = None) -> None:
    app = gui.Application.instance
    app.initialize(str(_open3d_ascii_resources()))
    if os.name == "nt":
        chinese_font = Path("C:/Windows/Fonts/msyh.ttc")
        if chinese_font.is_file():
            font = gui.FontDescription()
            font.add_typeface_for_language(str(chinese_font), "zh_all")
            app.set_font(gui.Application.DEFAULT_FONT_ID, font)
    # Keep the Python callback owner alive for the whole native GUI lifetime.
    viewer = FemtoSceneViewer(reconstruction, language or load_language())
    app.run()
    del viewer


def main() -> int:
    args = parse_args()
    reconstruction = args.reconstruction.resolve() if args.reconstruction else latest_reconstruction()
    run_viewer(reconstruction, args.lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
