"""Software display of saved project vertices. No reconstruction or replanning."""
from math import sin, cos, radians, ceil
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class ReplayViewport(QWidget):
    regions_edited = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 380)
        self.data = None
        self.stage = 0
        self.mode = "RGB"
        self.zoom = 1.0
        self.pan = QPointF()
        self.yaw = self.pitch = 0.0
        self.drag = None
        self.focus = ""
        self.images = {}
        self.cache = None
        self.cache_key = None
        self.route_groups = []
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(55)
        self._pulse_timer.timeout.connect(self._pulse_tick)
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(25)
        self._redraw_timer.timeout.connect(self.update)
        self.setToolTip("左键旋转三维视图 · 右键平移 · 滚轮缩放")

    def set_data(self, data, images, route_groups):
        self.data, self.images, self.route_groups = data, images, route_groups
        self._pulse_timer.stop()
        self._pulse_phase = 0.0
        raw_faces = {int(index) for index in data.region["region"].get("triangle_indices", [])}
        self.selected_faces = raw_faces
        # Selection indices are usually indices in the full reconstruction mesh;
        # some exports use the compact region mesh instead.  Detect the format
        # from the index range so both saved variants replay correctly.
        self.region_mesh_selection = bool(self.selected_faces) and max(self.selected_faces) < len(data.region_faces)
        source_faces = data.region_faces if self.region_mesh_selection else data.faces
        if any(index < 0 or index >= len(source_faces) for index in self.selected_faces):
            raise ValueError("区域三角形索引超出回放网格范围")
        self.selected_vertices = {i for face in self.selected_faces for i in source_faces[face]}
        self.reset_view()

    def set_view(self, stage, mode):
        if (self.stage, self.mode) != (stage, mode):
            self.stage, self.mode = stage, mode
            self.focus = ""
            self._pulse_timer.stop()
            self.reset_view()
        else:
            self.update()

    def reset_view(self):
        self.zoom, self.pan = 1.0, QPointF()
        self.yaw = self.pitch = 0.0
        self.cache_key = None
        self.update()

    def focus_issue(self, kind):
        self.focus = "gap" if kind == "route" else kind
        if self.focus in ("gap", "geometry", "capture", "edge"):
            self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
        self.update()

    def _pulse_tick(self):
        self._pulse_phase = (self._pulse_phase + 0.22) % (2 * 3.141592653589793)
        self.update()

    def _pulse_strength(self) -> float:
        return 0.5 + 0.5 * sin(self._pulse_phase)

    def is_3d(self):
        return self.mode in ("模型", "三维模型", "点云")

    def project(self, point):
        x, y, z = (point[i]-self.center[i] for i in range(3))
        a, b = radians(self.yaw), radians(self.pitch)
        x, z = x*cos(a)+z*sin(a), -x*sin(a)+z*cos(a)
        y, z = y*cos(b)-z*sin(b), y*sin(b)+z*cos(b)
        return (self.width()/2+self.pan.x()+x*self.scale, self.height()/2+self.pan.y()+y*self.scale, z)

    def prepare_projection(self):
        use_region = self.stage >= 1 and self.region_mesh_selection
        vertices = self.data.region_vertices if use_region else self.data.vertices
        fitting = [vertices[i] for i in self.selected_vertices] if self.stage >= 1 else vertices
        if not fitting:
            fitting = vertices
        bounds = [(min(p[i] for p in fitting), max(p[i] for p in fitting)) for i in range(3)]
        self.center = [(a+b)/2 for a, b in bounds]
        self.scale = min((self.width()-90)/max(.01, bounds[0][1]-bounds[0][0]), (self.height()-100)/max(.01, bounds[1][1]-bounds[1][0]))*.86*self.zoom

    def render_geometry(self):
        key = (self.width(), self.height(), self.stage, self.mode, self.yaw, self.pitch, self.zoom, self.pan.x(), self.pan.y())
        if key == self.cache_key:
            return self.cache
        self.prepare_projection()
        image = QImage(self.size(), QImage.Format_RGB32)
        image.fill(QColor("#0b0e11"))
        p = QPainter(image)
        p.setClipRect(image.rect())
        if self.mode == "点云":
            cloud = list(zip(self.data.cloud_points, self.data.cloud_colors))
            if len(cloud) > 12000:
                step = ceil(len(cloud) / 12000)
                cloud = cloud[::step]
            items = [(self.project(point), color) for point, color in cloud]
            for (x, y, _), color in sorted(items, key=lambda item: -item[0][2]):
                p.setPen(QPen(QColor(*color), 2))
                p.drawPoint(QPointF(x, y))
        else:
            use_region = self.stage >= 1 and self.region_mesh_selection
            vertices = self.data.region_vertices if use_region else self.data.vertices
            colors = self.data.region_colors if use_region else self.data.colors
            faces = self.data.region_faces if use_region else self.data.faces
            projected = [self.project(point) for point in vertices]
            indices = list(self.selected_faces if self.stage >= 1 else range(len(faces)))
            # A software prototype should stay responsive on ordinary clinical
            # workstations.  The saved mesh remains untouched; this only limits
            # the number of painter polygons used for the interactive preview.
            if len(indices) > 3200:
                step = ceil(len(indices) / 3200)
                indices = indices[::step]
            indices = sorted(indices, key=lambda index: -sum(projected[i][2] for i in faces[index]))
            p.setPen(Qt.NoPen)
            for index in indices:
                face = faces[index]
                coordinates = [projected[i] for i in face]
                if all(x < -50 or x > self.width()+50 or y < -50 or y > self.height()+50 for x, y, _ in coordinates):
                    continue
                rgb = tuple(sum(colors[i][channel] for i in face)//3 for channel in range(3))
                if self.stage >= 1:
                    rgb = (int(rgb[0]*.65), int(rgb[1]*.65+55), int(rgb[2]*.65+22))
                p.setBrush(QColor(*rgb))
                p.drawPolygon(QPolygonF([QPointF(x, y) for x, y, _ in coordinates]))
            if self.stage >= 2:
                p.setRenderHint(QPainter.Antialiasing)
                p.setBrush(Qt.NoBrush)
                for role, points in self.route_groups:
                    if len(points) < 2:
                        continue
                    p.setPen(QPen(QColor("#fcca46" if role == "scan" else "#48cae4"), 1.45, Qt.SolidLine if role == "scan" else Qt.DashLine))
                    poly = QPainterPath()
                    for index, point in enumerate(points):
                        x, y, _ = self.project(point)
                        if index == 0:
                            poly.moveTo(x, y)
                        else:
                            poly.lineTo(x, y)
                    p.drawPath(poly)
        p.end()
        self.cache, self.cache_key = image, key
        return image

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#0b0e11"))
        if not self.data:
            p.setPen(QColor("#b8cdd8"))
            p.drawText(self.rect(), Qt.AlignCenter, "正在读取已保存的项目数据…")
            p.end()
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        if self.is_3d():
            p.drawImage(0, 0, self.render_geometry())
            self.draw_gaps(p)
            hint = "左键旋转 · 右键平移 · 滚轮缩放"
        else:
            key = "route" if self.mode == "RGB 路线图" else "selected" if self.mode in ("选区 RGB", "固定 RGB") else "depth" if self.mode == "深度" else "rgb"
            img = self.images[key]
            ratio = min(self.width()/img.width(), (self.height()-62)/img.height())*self.zoom
            rect = QRectF((self.width()-img.width()*ratio)/2+self.pan.x(), (self.height()-img.height()*ratio)/2+self.pan.y(), img.width()*ratio, img.height()*ratio)
            p.drawImage(rect, img)
            hint = "右键平移 · 滚轮缩放"
            if key == "depth":
                hint += "  |  深度色标 250–2210 mm；黑色为无效值"
        self.draw_focus_marker(p)
        if self.focus == "geometry":
            pulse = self._pulse_strength()
            left = self.width() * 0.24
            top = self.height() * 0.18
            width = self.width() * 0.52
            height = self.height() * 0.58
            p.setBrush(QColor(157, 78, 221, 18 + int(20 * pulse)))
            p.setPen(QPen(QColor(157, 78, 221, 125 + int(80 * pulse)), 2.0 + 1.2 * pulse))
            p.drawRoundedRect(QRectF(left, top, width, height), 10, 10)
            p.setBrush(QColor(22, 29, 39, 230))
            p.setPen(QPen(QColor(157, 78, 221, 200), 1))
            p.drawRoundedRect(QRectF(left + 14, top + 14, 188, 30), 5, 5)
            p.setPen(QColor("#eadcff"))
            p.drawText(QRectF(left + 24, top + 14, 168, 30), Qt.AlignVCenter, "三维表面支持未确认")
        p.setPen(QColor("#e7edf4"))
        p.fillRect(QRectF(12, 12, 188, 28), QColor(11, 17, 24, 220))
        p.drawText(QRectF(22, 12, 178, 28), Qt.AlignVCenter, f"{self.mode} · 项目回放")
        p.end()

    def draw_gaps(self, p):
        if self.stage < 2:
            return
        p.setRenderHint(QPainter.Antialiasing)
        pulse = self._pulse_strength() if self.focus == "gap" else 0.0
        p.setBrush(QColor("#161d27"))
        gaps = [item for item in self.data.trajectory["ordered_items"] if item["kind"] == "gap"]
        for item in gaps:
            for key in ("previous_point", "next_point"):
                x, y, _ = self.project(item[key]["xyz_m"])
                radius = 3 + 2 * pulse
                p.setPen(QPen(QColor(239, 71, 111, 185 + int(60 * pulse)), 1.6 + 1.0 * pulse))
                p.drawEllipse(QPointF(x, y), radius, radius)
                if self.focus == "gap":
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor(239, 71, 111, 50 + int(85 * pulse)), 1.1))
                    p.drawEllipse(QPointF(x, y), radius + 5 + 3 * pulse, radius + 5 + 3 * pulse)
                    p.setBrush(QColor("#161d27"))
        if self.focus == "gap":
            p.setPen(QColor("#ffd166"))
            p.drawText(22, 68, "粉色圆点：断开段端点（缺口之间不连线）")

    def draw_focus_marker(self, p):
        """Pulse a focused geometry/edge area on both mesh and image views."""
        if self.focus not in ("geometry", "edge"):
            return
        pulse = self._pulse_strength()
        color = QColor("#9d4edd" if self.focus == "geometry" else "#ffd166")
        if self.is_3d() and getattr(self, "selected_vertices", None):
            vertices = self.data.region_vertices if self.region_mesh_selection else self.data.vertices
            points = [self.project(vertices[index]) for index in self.selected_vertices if index < len(vertices)]
            if points:
                cx = sum(point[0] for point in points) / len(points)
                cy = sum(point[1] for point in points) / len(points)
            else:
                cx, cy = self.width() / 2, self.height() / 2
        else:
            cx, cy = self.width() / 2, self.height() / 2
        radius = 34 + 8 * pulse
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(color.red(), color.green(), color.blue(), 52 + int(36 * pulse)))
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 145 + int(80 * pulse)), 2.2))
        p.drawEllipse(QPointF(cx, cy), radius, radius * .72)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 48 + int(70 * pulse)), 1.2))
        p.drawEllipse(QPointF(cx, cy), radius + 10 + 4 * pulse, radius * .72 + 7 + 3 * pulse)

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self.drag = (event.button(), event.position())

    def mouseMoveEvent(self, event):
        if self.drag:
            button, previous = self.drag
            delta = event.position()-previous
            self.drag = (button, event.position())
            if button == Qt.LeftButton and self.is_3d():
                self.yaw += delta.x()*.35
                self.pitch = min(85, max(-85, self.pitch+delta.y()*.35))
            elif button == Qt.RightButton:
                self.pan += delta
            self._redraw_timer.start()

    def mouseReleaseEvent(self, event):
        self.drag = None
        self._redraw_timer.stop()
        self.update()

    def wheelEvent(self, event):
        self.zoom = max(.35, min(6, self.zoom*(1.15 if event.angleDelta().y() > 0 else 1/1.15)))
        self.update()
        event.accept()
