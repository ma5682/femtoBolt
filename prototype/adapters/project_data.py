"""Read-only adapter for the repository's published reconstruction artifacts.

The adapter validates the evidence links before exposing anything to the UI. It
does not call the production planner and never writes to the source folders.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import struct


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() and path.exists() else root / value


@dataclass
class ProjectData:
    label: str
    dataset: Path
    reconstruction: Path
    rgb_path: Path
    depth_path: Path
    mesh_path: Path
    region_mesh_path: Path
    points_path: Path
    region_path: Path
    trajectory_dir: Path
    report: dict
    local_geometry: dict
    region: dict
    trajectory: dict
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    colors: list[tuple[int, int, int]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    region_vertices: list[tuple[float, float, float]] = field(default_factory=list)
    region_colors: list[tuple[int, int, int]] = field(default_factory=list)
    region_faces: list[tuple[int, int, int]] = field(default_factory=list)
    cloud_points: list[tuple[float, float, float]] = field(default_factory=list)
    cloud_colors: list[tuple[int, int, int]] = field(default_factory=list)
    route_groups: list[tuple[str, list[tuple[float, float, float]]]] = field(default_factory=list)
    verified_files: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _read_binary_ply(path: Path, load_faces: bool = False):
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PLY header incomplete: {path.name}")
            decoded = line.decode("ascii", errors="strict").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            raise ValueError(f"仅支持 binary_little_endian PLY: {path.name}")
        vertex_count = next(int(row.split()[2]) for row in header if row.startswith("element vertex "))
        face_count = next((int(row.split()[2]) for row in header if row.startswith("element face ")), 0)
        properties = []
        in_vertex = False
        for row in header:
            if row == "element vertex " + str(vertex_count):
                in_vertex = True
                continue
            if row.startswith("element ") and in_vertex:
                break
            if in_vertex and row.startswith("property "):
                properties.append(row.split()[1:])
        formats = {"double": "d", "float": "f", "uchar": "B", "uint8": "B"}
        try:
            fmt = "<" + "".join(formats[item[0]] for item in properties)
        except KeyError as exc:
            raise ValueError(f"PLY property unsupported: {exc}") from exc
        stride = struct.calcsize(fmt)
        xyz_indices = [index for index, item in enumerate(properties) if item[-1] in ("x", "y", "z")]
        rgb_indices = [index for index, item in enumerate(properties) if item[-1] in ("red", "green", "blue")]
        vertices, colors = [], []
        for _ in range(vertex_count):
            record = stream.read(stride)
            if len(record) != stride:
                raise ValueError(f"PLY vertex data incomplete: {path.name}")
            values = struct.unpack(fmt, record)
            vertices.append(tuple(float(values[index]) for index in xyz_indices))
            colors.append(tuple(int(values[index]) for index in rgb_indices) if len(rgb_indices) == 3 else (136, 154, 165))
        faces = []
        if load_faces:
            for _ in range(face_count):
                count_bytes = stream.read(1)
                if not count_bytes:
                    raise ValueError(f"PLY face data incomplete: {path.name}")
                count = struct.unpack("<B", count_bytes)[0]
                indices = struct.unpack("<" + "I" * count, stream.read(4 * count))
                if count >= 3:
                    faces.append((int(indices[0]), int(indices[1]), int(indices[2])))
        return vertices, colors, faces


def _choose_annotation(reconstruction: Path) -> tuple[Path, Path]:
    candidates = []
    for region_path in reconstruction.glob("annotations/*/region_1.json"):
        folder = region_path.parent
        trajectories = list(folder.glob("trajectory_*/trajectory.json"))
        if trajectories:
            candidate = max(trajectories, key=lambda item: item.stat().st_mtime)
            candidates.append((candidate.stat().st_mtime, region_path, candidate.parent))
    if not candidates:
        raise FileNotFoundError("没有找到带轨迹的区域导出记录")
    _, region_path, trajectory_dir = max(candidates, key=lambda item: item[0])
    return region_path, trajectory_dir


def _verify(data: ProjectData) -> None:
    verified = data.verified_files
    expected_region_file = data.region.get("file_sha256", {}).get(data.region_mesh_path.name)
    if expected_region_file and _sha256(data.region_mesh_path) != expected_region_file:
        data.warnings.append("区域导出网格 SHA-256 与区域记录不一致，未标记为可信回放。")
    elif data.region_mesh_path.exists():
        verified[str(data.region_mesh_path)] = _sha256(data.region_mesh_path)
    else:
        data.warnings.append("区域导出网格不存在，无法核验轨迹来源。")
    if data.mesh_path.exists():
        verified[str(data.mesh_path)] = _sha256(data.mesh_path)
    region_file_hashes = data.region.get("file_sha256", {})
    for filename, expected in region_file_hashes.items():
        path = data.region_path.parent / filename
        if not path.exists() or _sha256(path) != expected:
            data.warnings.append(f"区域文件校验失败：{filename}")
        else:
            verified[str(path)] = expected
    files_manifest = data.trajectory_dir / "trajectory_files.json"
    if files_manifest.exists():
        manifest = _json(files_manifest)
        for filename, expected in manifest.get("files", {}).items():
            path = data.trajectory_dir / filename
            if not path.exists() or _sha256(path) != expected:
                data.warnings.append(f"轨迹文件校验失败：{filename}")
            else:
                verified[str(path)] = expected
    source = data.trajectory.get("source", {})
    if source.get("region_manifest_sha256") and _sha256(data.region_path) != source["region_manifest_sha256"]:
        data.warnings.append("轨迹来源区域记录校验失败。")
    if source.get("mesh_file_sha256") and _sha256(data.region_mesh_path) != source["mesh_file_sha256"]:
        data.warnings.append("轨迹来源区域网格校验失败。")
    region = data.region.get("region", {})
    reference = region.get("view_reference", {})
    for path, expected, label in (
        (data.rgb_path, reference.get("rgb_sha256"), "参考 RGB"),
        (data.depth_path, reference.get("depth_sha256"), "参考深度"),
    ):
        if expected and _sha256(path) != expected:
            data.warnings.append(f"{label}校验失败。")
        elif expected:
            verified[str(path)] = expected
    origin = data.local_geometry.get("reference_provenance", {})
    pose = data.dataset / "derived" / "local_posed" / "pose" / "000000.json"
    if origin.get("reference_frame_id") != 0 or origin.get("unit") != "metre" or origin.get("coordinate_frame") != "world_rdf":
        data.warnings.append("W0/RDF/米坐标绑定不完整。")
    if origin.get("reference_pose_sha256") and pose.exists() and _sha256(pose) != origin["reference_pose_sha256"]:
        data.warnings.append("参考位姿 SHA-256 校验失败。")
    if data.trajectory.get("execution_ready") is not False:
        data.warnings.append("轨迹执行就绪标记不符合当前项目约束。")
    if data.region.get("physical_accuracy_verified") is not False:
        data.warnings.append("区域物理精度验证状态不是 false。")


def _read_route(trajectory: dict):
    groups = []
    for item in trajectory.get("ordered_items", []):
        if item.get("kind") != "scan_segment":
            continue
        points = [tuple(float(v) for v in point["xyz_m"]) for point in item.get("points", []) if point.get("xyz_m")]
        if len(points) >= 2:
            groups.append(("scan", points))
    return groups


def load_project_data(project_root: Path, reconstruction: Path | None = None) -> ProjectData:
    project_root = Path(project_root)
    dataset = next(project_root.glob("datasets/scene_*_local_*"), None)
    reconstruction = Path(reconstruction) if reconstruction else project_root / "outputs" / "complete_demo_20260917_102931" / "reconstruction"
    if dataset is None or not reconstruction.exists():
        raise FileNotFoundError("没有找到可回放的 datasets/scene_* 或 reconstruction")
    region_path, trajectory_dir = _choose_annotation(reconstruction)
    region = _json(region_path)
    trajectory = _json(trajectory_dir / "trajectory.json")
    mesh_path = reconstruction / "scene_tsdf_mesh_m.ply"
    region_mesh_path = region_path.parent / "region_1_mesh_m.ply"
    points_path = reconstruction / "scene_fused_points_m.ply"
    rgb_path = dataset / "rgb" / "000000.png"
    depth_path = dataset / "depth" / "000000.png"
    report = _json(reconstruction / "reconstruction_report.json")
    local_geometry = _json(reconstruction / "local_geometry.json")
    vertices, colors, faces = _read_binary_ply(mesh_path, load_faces=True)
    region_vertices, region_colors, region_faces = _read_binary_ply(region_mesh_path, load_faces=True)
    cloud_points, cloud_colors, _ = _read_binary_ply(points_path, load_faces=False)
    data = ProjectData(
        label=f"项目回放 · {dataset.name}", dataset=dataset, reconstruction=reconstruction,
        rgb_path=rgb_path, depth_path=depth_path, mesh_path=mesh_path, points_path=points_path,
        region_mesh_path=region_mesh_path,
        region_path=region_path, trajectory_dir=trajectory_dir, report=report,
        local_geometry=local_geometry, region=region, trajectory=trajectory,
        vertices=vertices, colors=colors, faces=faces,
        region_vertices=region_vertices, region_colors=region_colors, region_faces=region_faces,
        cloud_points=cloud_points,
        cloud_colors=cloud_colors, route_groups=_read_route(trajectory),
    )
    _verify(data)
    return data
