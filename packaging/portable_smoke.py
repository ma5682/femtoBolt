"""Run a no-camera startup and native-library check in the actual frozen exe."""
import json
import sys
from pathlib import Path


def main():
    import numpy as np
    import cv2
    import open3d as o3d
    import vtk
    import mediapipe
    import pyorbbecsdk
    from mediapipe.tasks.python.core.mediapipe_c_bindings import load_raw_library
    assert load_raw_library() is not None
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from femto.app.qt_main import WorkstationWindow, apply_theme
    from femto.app.qt_data_maintenance import DataMaintenanceDialog
    from femto.paths import REPO_ROOT

    app = QApplication([])
    apply_theme(app)
    window = WorkstationWindow(connect_camera=False)
    window.show()
    dialog = DataMaintenanceDialog(REPO_ROOT)
    points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.zeros((3, 3))))
    assert len(points.points) == 3
    assert cv2.cvtColor(np.zeros((2, 2, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY).shape == (2, 2)
    assert vtk.vtkSphereSource() is not None
    import tempfile
    import subprocess
    with tempfile.TemporaryDirectory() as folder:
        mesh_path = str(Path(folder) / 'synthetic_mesh.ply')
        mesh = o3d.geometry.TriangleMesh.create_box()
        assert o3d.io.write_triangle_mesh(mesh_path, mesh)
        assert len(o3d.io.read_triangle_mesh(mesh_path).triangles) == len(mesh.triangles)
        from types import SimpleNamespace
        from femto.annotation.fixed_rgb_geometry import FixedRGBModel
        from femto.annotation.scene_annotations import AnnotationStore
        from femto.planning.planning_resolution import create_reduced_model, route_sample_step_mm
        reference = FixedRGBModel.__new__(FixedRGBModel)
        reference.mesh = mesh
        reference.store = AnnotationStore(mesh, mesh_path)
        reference.reconstruction = Path(folder)
        reference.rgb = np.full((100, 100, 3), 140, np.uint8)
        intrinsic = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
        reference.projection = SimpleNamespace(k=intrinsic, depth=np.full((100, 100), 500, np.uint16),
            minimum=.1, maximum=2., image_valid=np.ones((100, 100), bool))
        reference.view_reference = dict(image_size=[100, 100], undistorted_intrinsic=intrinsic.tolist())
        reference.max_radius_mm = 0
        reduced = create_reduced_model(reference, Path(folder)/'reduced')
        assert reduced.rgb.shape[:2] == (12, 12)
        assert route_sample_step_mm(reduced.view_reference) == 3.
        assert reduced.store.digest == reference.store.digest
        assert Path(reduced.store.source).read_bytes() == Path(mesh_path).read_bytes()
    maintenance = subprocess.run([sys.executable, '-m', 'femto.app.qt_data_maintenance', '--help'],
        timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
    assert maintenance.returncode == 0
    for module in ('femto.pipeline.local_offline', 'femto.pipeline.local_comparison'):
        worker = subprocess.run([sys.executable, '-X', 'utf8', '-m', module, '--help'],
            timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        assert worker.returncode == 0, module
    assert REPO_ROOT == Path(sys.executable).resolve().parent
    result = {'status': 'passed', 'frozen': bool(getattr(sys, 'frozen', False)),
              'workspace': str(REPO_ROOT), 'title': window.windowTitle(),
              'checks': ['main_window', 'theme', 'maintenance_dialog', 'numpy', 'opencv',
                         'open3d_native', 'vtk_native', 'mediapipe_native', 'orbbec_sdk_import',
                         'mesh_save_reload', 'reduced_planning_reference', 'maintenance_subprocess', 'offline_worker_entry', 'comparison_worker_entry'],
              'camera_capture_tested': False}
    def finish():
        window.grab().save(str(REPO_ROOT / 'Log' / 'package-preview.png'))
        dialog.close()
        window.close()
        (REPO_ROOT / 'Log' / 'package-smoke.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        app.quit()
    QTimer.singleShot(2000, finish)
    app.exec()
