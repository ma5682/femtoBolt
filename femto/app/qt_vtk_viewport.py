"""Qt-owned VTK renderer; Open3D geometry is copied through a numeric adapter."""
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from femto.viewer.vtk_scene import scene_snapshot


class VTKViewport(QWidget):
    """Read-only mesh/region/route view with original mesh cell picking.

    ``picked`` emits ``triangle_index`` and ``point_world_m`` in world RDF.
    ``set_scene`` accepts FixedRGBModel plus saved region/plan dictionaries.
    """
    picked = Signal(dict)

    def __init__(self, parent=None):
        if QApplication.platformName() in ('offscreen', 'minimal'):
            raise RuntimeError('三维显示需要原生 Qt 窗口与 OpenGL 3.2；当前无屏幕平台不支持')
        super().__init__(parent)
        # Select the binding before importing the QVTK bridge.
        import vtkmodules.qt
        vtkmodules.qt.PyQtImpl = 'PySide6'
        import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
        import vtkmodules.vtkInteractionStyle  # noqa: F401
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        from vtkmodules.vtkRenderingCore import vtkRenderer, vtkCellPicker
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
        self.widget = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.widget)
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(.025, .055, .085)
        self.widget.GetRenderWindow().AddRenderer(self.renderer)
        self.style = vtkInteractorStyleTrackballCamera()
        self.widget.SetInteractorStyle(self.style)
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(.0005)
        self.picker.PickFromListOn()
        self._actors = []
        self._region_actor = self._route_actor = self._coverage_actor = None
        self._coverage_display = None
        self._press = None
        self._closed = False
        self._render_failed = False
        self.widget.GetRenderWindow().AddObserver('ErrorEvent', self._render_error)
        # Observe the style: trackball grabs focus during drags, so an interactor
        # release observer alone can miss the release and never deliver picks.
        self.style.AddObserver('LeftButtonPressEvent', self._pressed)
        self.style.AddObserver('LeftButtonReleaseEvent', self._released)
        self.style.AddObserver('RightButtonPressEvent', lambda *_: self.style.OnMiddleButtonDown())
        self.style.AddObserver('RightButtonReleaseEvent', lambda *_: self.style.OnMiddleButtonUp())
        # Suppress trackball keyboard commands (including camera flight).
        self.style.AddObserver('CharEvent', lambda *_: None)
        try:
            self.widget.Initialize()
            if not self.widget.GetRenderWindow().SupportsOpenGL() or self._render_failed:
                raise RuntimeError('当前显卡驱动无法创建三维显示所需的 OpenGL 3.2 环境')
        except Exception:
            self.widget.Finalize()
            self.widget.deleteLater()
            self._closed = True
            self.deleteLater()
            raise

    def _render_error(self, caller, event):
        self._render_failed = True

    def _render(self):
        self._render_failed = False
        self.widget.GetRenderWindow().Render()
        if self._render_failed:
            raise RuntimeError('三维渲染失败，请检查显卡驱动或返回参考图继续操作')

    def _check_thread(self):
        if QThread.currentThread() != self.thread():
            raise RuntimeError('VTK viewport must be updated on its Qt GUI thread')
        if self._closed:
            raise RuntimeError('VTK viewport is closed')

    def _actor(self, vertices, cells, *, lines=False, points=False, colors=None, ids=None, cell_colors=None):
        import numpy as np
        from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
        vtk_points = vtkPoints()
        vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices), deep=True))
        data = vtkPolyData()
        data.SetPoints(vtk_points)
        sizes = np.asarray([len(cell) for cell in cells], dtype=np.int64) if lines else np.full(len(cells), 1 if points else 3, np.int64)
        offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
        connectivity = np.concatenate(cells).astype(np.int64) if lines and len(cells) else np.asarray(cells, np.int64).reshape(-1)
        array = vtkCellArray()
        array.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(connectivity, deep=True))
        (data.SetLines if lines else data.SetVerts if points else data.SetPolys)(array)
        if colors is not None:
            data.GetPointData().SetScalars(numpy_to_vtk(colors, deep=True))
        if cell_colors is not None:
            data.GetCellData().SetScalars(numpy_to_vtk(cell_colors, deep=True))
        if ids is not None:
            original = numpy_to_vtkIdTypeArray(np.ascontiguousarray(ids, dtype=np.int64), deep=True)
            original.SetName('original_triangle_id')
            data.GetCellData().AddArray(original)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(data)
        if cell_colors is not None:
            mapper.SetScalarModeToUseCellData();mapper.SetColorModeToDirectScalars()
        actor = vtkActor()
        actor.SetMapper(mapper)
        self.renderer.AddActor(actor)
        self._actors.append(actor)
        return actor

    def set_scene(self, model, region=None, plan=None, *, reset_camera=True):
        import numpy as np
        self._check_thread()
        scene = scene_snapshot(model, region, plan) if model is not None else None
        self.renderer.RemoveAllViewProps()
        self.picker.InitializePickList()
        self._actors.clear()
        self._region_actor = self._route_actor = self._coverage_actor = None
        if scene is not None:
            if len(scene.triangles):
                mesh = self._actor(scene.vertices, scene.triangles, colors=scene.colors,
                                   ids=np.arange(len(scene.triangles)))
                self.picker.AddPickList(mesh)
            else:
                mesh = self._actor(scene.vertices, np.arange(len(scene.vertices)), points=True, colors=scene.colors)
                mesh.GetProperty().SetPointSize(3)
            mesh.GetProperty().SetColor(.7, .76, .8)
            if len(scene.region_ids):
                overlay = self._actor(scene.vertices, scene.triangles[scene.region_ids])
                self._region_actor = overlay
                overlay.GetProperty().SetColor(.02, .85, .6)
                overlay.GetProperty().SetOpacity(.42)
                overlay.PickableOff()
                overlay.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
                overlay.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-1, -1)
            if scene.route_runs:
                coords = np.concatenate(scene.route_runs)
                offsets = np.cumsum([0] + [len(run) for run in scene.route_runs])
                cells = [np.arange(a, b) for a, b in zip(offsets[:-1], offsets[1:])]
                route = self._actor(coords, cells, lines=True)
                self._route_actor = route
                route.GetProperty().SetColor(1., .8, .25)
                route.GetProperty().SetLineWidth(3)
                route.GetProperty().LightingOff()
                route.PickableOff()
        if reset_camera:
            self.reset_camera()
        else:
            self._render()

    def set_coverage(self, display=None, reason=0):
        import numpy as np
        from femto.planning.coverage_display import COLORS
        self._check_thread()
        self._coverage_display = display
        if self._coverage_actor is not None:
            self.renderer.RemoveActor(self._coverage_actor)
            self._actors.remove(self._coverage_actor);self._coverage_actor = None
        for actor in (self._region_actor, self._route_actor):
            if actor is not None:actor.SetVisibility(display is None)
        if display is not None:
            keep = np.ones(len(display.status), bool) if reason == 0 else display.status == reason
            triangles = display.triangles[keep]
            if len(triangles):
                actor = self._actor(triangles.reshape(-1,3), np.arange(3*len(triangles)).reshape(-1,3),
                    cell_colors=COLORS[display.status[keep], :3])
                actor.GetProperty().SetOpacity(.82);actor.GetProperty().LightingOff();actor.PickableOff()
                actor.GetMapper().SetResolveCoincidentTopologyToPolygonOffset()
                actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-1,-1)
                self._coverage_actor = actor
        self._render()

    def focus_coverage(self, reason=0):
        import numpy as np
        self._check_thread()
        display = self._coverage_display
        if display is None:return False
        keep = np.ones(len(display.status), bool) if reason == 0 else display.status == reason
        if not np.any(keep):return False
        points = display.triangles[keep].reshape(-1,3)
        low,high = points.min(axis=0),points.max(axis=0)
        self.renderer.ResetCamera(low[0],high[0],low[1],high[1],low[2],high[2])
        self.renderer.ResetCameraClippingRange();self._render()
        return True

    def reset_camera(self):
        self._check_thread()
        camera = self.renderer.GetActiveCamera()
        # RDF has image-down +Y; keep the initial camera upright without changing data axes.
        camera.SetPosition(0, 0, -1)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, -1, 0)
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self._render()

    def _pressed(self, caller, event):
        self._press = self.widget.GetEventPosition()
        self.style.OnMiddleButtonDown()

    def _released(self, caller, event):
        self.style.OnMiddleButtonUp()
        position = self.widget.GetEventPosition()
        if self._press is None or sum((a-b)**2 for a, b in zip(position, self._press)) > 16:
            self._press = None
            return
        self._press = None
        if self.picker.Pick(*position, 0, self.renderer):
            cell = self.picker.GetCellId()
            dataset = self.picker.GetDataSet()
            ids = dataset.GetCellData().GetArray('original_triangle_id') if dataset else None
            if ids is not None and 0 <= cell < ids.GetNumberOfTuples():
                self.picked.emit(dict(triangle_index=int(ids.GetValue(cell)),
                                      point_world_m=list(self.picker.GetPickPosition())))

    def finalize(self):
        if not self._closed:
            self._check_thread()
            self.widget.Finalize()
            self._closed = True

    def closeEvent(self, event):
        self.finalize()
        super().closeEvent(event)

    def shutdown(self):
        self.finalize()
