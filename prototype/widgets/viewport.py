"""QPainter demonstration viewport. Workflow decisions belong to the controller."""
from copy import deepcopy
from math import sin
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QLabel
from models.task_state import Stroke
from widgets.medical_viewport import Medical3DViewport, ViewportScene


class MockViewport(Medical3DViewport):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = ViewportScene()
        self.setMinimumSize(520, 380)
        self.setMouseTracking(True)
        self._drawing = None
        self._zoom = 1.0
        self._pan = QPointF()
        self._pan_last = None
        self._scale, self._offset = 1.0, QPointF()
        # Enable a quiet breathing highlight only while an actionable issue
        # is present.  It draws attention without distracting from planning.
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(55)
        self._pulse_timer.timeout.connect(self._pulse_tick)
        self.tag = QLabel(self)
        self.tag.setObjectName("viewportTag")
        self.hint = QLabel(self)
        self.hint.setObjectName("viewportHint")
        self.hint.setWordWrap(True)
        self.hint.setFixedWidth(224)
        self.legend = QLabel(self)
        self.legend.setObjectName("viewportLegend")
        self.setToolTip("左键绘制规划区 · 右键平移 · 滚轮缩放")

    def set_scene(self, scene):
        self.state = deepcopy(scene)
        self._drawing = None
        self.tag.setText(f"{scene.view} · 实时画面")
        missing = scene.scenario == "geometry" and scene.region_complete and not scene.geometry_repaired
        problem = "该区域缺少可靠三维数据。已画选区保留，补采后可继续。" if missing else "左侧三维数据缺失，请移动相机继续补采。" if scene.capture_phase == "quality_bad" else "扫描线在高亮位置断开，请返回规划检查处理。" if scene.trajectory_failed else ""
        self.hint.setText(problem)
        self.hint.setVisible(bool(problem))
        if scene.stage == 0:
            legend = "绿色 · 可靠采集   灰色 · 未采集   黄色 · 待检查   红色 · 不可用"
        elif scene.stage == 1:
            legend = "绿色 · 规划区   红色 · 禁入区   紫色 · 缺失三维"
        else:
            legend = "黄色 · 扫描段   蓝色虚线 · 不可治疗连接   红色 · 禁入区"
        self.legend.setText(legend)
        self.legend.setVisible(False)
        has_issue = bool(problem) or self.state.issue_focus in ("geometry", "route", "edge")
        if has_issue:
            if not self._pulse_timer.isActive():
                self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
        self._position_overlays()
        self.setCursor(Qt.CrossCursor if scene.editable else Qt.ArrowCursor)
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPointF()
        self.update()

    def _pulse_tick(self):
        self._pulse_phase = (self._pulse_phase + 0.22) % (2 * 3.141592653589793)
        self.update()

    def _pulse_strength(self) -> float:
        return 0.5 + 0.5 * sin(self._pulse_phase)

    def _position_overlays(self):
        self.tag.adjustSize()
        self.tag.move(18, 16)
        self.hint.adjustSize()
        self.hint.move(self.width()-self.hint.width()-18, 16)
        self.legend.adjustSize()
        self.legend.move(18, self.height()-self.legend.height()-14)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_overlays()

    def _view_transform(self):
        scale = min(self.width()/850, self.height()/700) * self._zoom
        offset = QPointF((self.width()-850*scale)/2, (self.height()-700*scale)/2) + self._pan
        self._scale, self._offset = scale, offset
        return scale, offset

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#080d12"))
        if self.state.view not in ("RGB", "固定 RGB", "RGB 路线图"):
            self._draw_space_grid(p)
        scale, offset = self._view_transform()
        p.translate(offset)
        p.scale(scale, scale)
        self._draw_surface(p)
        if self.state.stage == 0:
            self._draw_capture(p)
        else:
            p.save()
            p.setClipPath(self._surface_path())
            self._draw_strokes(p)
            if self.state.stage >= 2:
                self._draw_route(p)
            p.restore()
            if self.state.stage == 1:
                self._draw_geometry_issue(p)
            elif self.state.trajectory_failed or self.state.issue_focus in ("route", "edge"):
                x, y = self._work_center()
                if self.state.issue_focus == "edge" and not self.state.trajectory_failed:
                    self._issue_patch(p, x-60, y+70, QColor("#ffd166"), "边缘未覆盖")
                else:
                    self._issue_patch(p, x, y, QColor("#ef476f"), "断开")
        p.end()

    def _draw_capture(self, p):
        if self.state.capture_phase == "idle":
            return
        p.save()
        p.setClipPath(self._surface_path())
        p.setPen(Qt.NoPen)
        height = 60 + self.state.capture_progress*74
        p.fillRect(QRectF(180, 112, 480, min(500, height)), QColor(34, 201, 151, 68))
        if self.state.capture_phase == "capturing":
            p.setPen(QPen(QColor("#06d6a0"), 1.8))
            p.drawLine(QPointF(180, min(600, 112+height)), QPointF(650, min(600, 112+height)))
            p.setBrush(QColor(255, 209, 102, 70))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(480, 380, 90, 130))
        p.restore()
        if self.state.capture_phase == "quality_bad":
            self._issue_patch(p, 315, 490, QColor("#ef476f"), "待补采")
        if self.state.issue_focus == "geometry":
            self._draw_strokes(p)
            self._draw_geometry_issue(p)

    def _draw_strokes(self, p):
        for stroke in self.state.strokes:
            if not stroke.points:
                continue
            color = QColor(6, 214, 160, 112) if stroke.kind == "work" else QColor(239, 71, 111, 165)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(color, 66, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            if len(stroke.points) == 1:
                p.drawPoint(QPointF(*stroke.points[0]))
            else:
                curve = QPainterPath(QPointF(*stroke.points[0]))
                for point in stroke.points[1:]:
                    curve.lineTo(QPointF(*point))
                p.drawPath(curve)

    def _draw_route(self, p):
        p.setBrush(Qt.NoBrush)
        for segment in self.state.segments:
            p.setPen(QPen(QColor("#fcca46" if segment.kind == "scan" else "#48cae4"), 2.6 if segment.kind == "scan" else 1.8, Qt.SolidLine if segment.kind == "scan" else Qt.DashLine, Qt.RoundCap))
            path = QPainterPath(QPointF(*segment.points[0]))
            for point in segment.points[1:]:
                path.lineTo(QPointF(*point))
            p.drawPath(path)

    def _scene_point(self, position):
        scale, offset = self._view_transform()
        return ((position.x()-offset.x())/scale, (position.y()-offset.y())/scale)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if not self.state.editable or event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        x, y = self._scene_point(event.position())
        if not self._surface_path().contains(QPointF(x, y)):
            return
        if self.state.tool == "erase":
            from mock.mock_planning import distance_to_stroke
            candidates = [(distance_to_stroke(x, y, stroke), i) for i, stroke in enumerate(self.state.strokes)]
            if candidates and min(candidates)[0] <= 40:
                self.state.strokes.pop(min(candidates)[1])
                self.regions_edited.emit(deepcopy(self.state.strokes))
            return
        self._drawing = Stroke(self.state.tool, [(x, y)])
        self.state.strokes.append(self._drawing)
        self.update()

    def mouseMoveEvent(self, event):
        if self._pan_last is not None:
            self._pan += event.position() - self._pan_last
            self._pan_last = event.position()
            self.update()
            return
        if self._drawing is not None:
            x, y = self._scene_point(event.position())
            if self._surface_path().contains(QPointF(x, y)):
                self._drawing.points.append((x, y))
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._pan_last = None
            self.setCursor(Qt.CrossCursor if self.state.editable else Qt.ArrowCursor)
            return
        if self._drawing is not None:
            self._drawing = None
            self.regions_edited.emit(deepcopy(self.state.strokes))
            self.update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self._zoom = min(2.2, max(0.65, self._zoom * (1.1 if event.angleDelta().y() > 0 else 1/1.1)))
        self.update()
        event.accept()

    def _draw_space_grid(self, p: QPainter) -> None:
        w, h = self.width(), self.height()
        horizon = h * 0.55
        center = w * 0.5
        p.setPen(QPen(QColor(0, 180, 216, 14), 1))
        for x in range(0, w + 1, 42):
            p.drawLine(QPointF(x, 0), QPointF(x, horizon))
        for y in range(0, int(horizon), 42):
            p.drawLine(QPointF(0, y), QPointF(w, y))
        p.setPen(QPen(QColor(72, 202, 228, 22), 1))
        for x in range(-w, 2 * w + 1, max(48, w // 13)):
            p.drawLine(QPointF(center, horizon), QPointF(x, h))
        for fraction in (0.08, 0.16, 0.27, 0.40, 0.57, 0.77, 1.0):
            y = horizon + (h - horizon) * fraction
            p.drawLine(QPointF(0, y), QPointF(w, y))
        if not self.state.show_axes:
            return
        origin = QPointF(83, h - 78)
        axes = ((74, 0, "X", "#ef476f"), (0, -64, "Y", "#06d6a0"), (-42, 27, "Z", "#48cae4"))
        p.setFont(QFont("Segoe UI", 9, QFont.DemiBold))
        for dx, dy, name, color in axes:
            pen = QPen(QColor(color), 2)
            p.setPen(pen)
            target = origin + QPointF(dx, dy)
            p.drawLine(origin, target)
            p.drawEllipse(target, 2.5, 2.5)
            p.drawText(target + QPointF(5, -4), name)
        p.setPen(QColor("#9aabb9"))
        p.drawText(origin + QPointF(-13, 22), "参考")

    def _surface_path(self) -> QPainterPath:
        path = QPainterPath(QPointF(367, 58))
        path.cubicTo(391, 66, 440, 66, 463, 58)
        path.cubicTo(466, 82, 471, 106, 490, 115)
        path.cubicTo(548, 121, 606, 144, 638, 182)
        path.cubicTo(656, 204, 655, 234, 640, 252)
        path.cubicTo(626, 271, 608, 275, 590, 278)
        path.cubicTo(578, 354, 584, 456, 570, 534)
        path.cubicTo(567, 560, 562, 581, 548, 596)
        path.cubicTo(494, 620, 338, 620, 282, 596)
        path.cubicTo(268, 581, 262, 560, 259, 534)
        path.cubicTo(246, 453, 253, 351, 239, 278)
        path.cubicTo(208, 275, 186, 259, 176, 237)
        path.cubicTo(166, 209, 184, 177, 213, 157)
        path.cubicTo(246, 135, 292, 120, 340, 115)
        path.cubicTo(359, 106, 363, 82, 367, 58)
        path.closeSubpath()
        return path

    def _draw_surface(self, p: QPainter) -> None:
        body = self._surface_path()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(6, 214, 160, 12))
        
        p.setBrush(QColor(0, 0, 0, 95))
        p.drawEllipse(QRectF(220, 617, 390, 26))
        if self.state.view in ("RGB", "固定 RGB", "RGB 路线图"):
            surface_color = "#26323b"
        elif self.state.view == "深度":
            surface_color = "#173445"
        else:
            surface_color = "#172530"
        for width, alpha in ((3, 18),):
            p.setPen(QPen(QColor(0, 180, 216, alpha), width))
            p.setBrush(Qt.NoBrush)
            p.drawPath(body)
        p.setBrush(QColor(surface_color))
        p.setPen(QPen(QColor("#8ba6ae"), 1.4))
        p.drawPath(body)
        p.save()
        p.setClipPath(body)
        sheen = QRadialGradient(QPointF(418, 310), 290)
        sheen.setColorAt(0, QColor(57, 166, 183, 34))
        sheen.setColorAt(0.8, QColor(0, 0, 0, 0))
        p.fillRect(QRectF(165, 55, 500, 565), sheen)
        p.setBrush(Qt.NoBrush)
        rgb = self.state.view in ("RGB", "固定 RGB", "RGB 路线图")
        for y in range(150, 610, 29):
            narrowing = abs(y - 355) * 0.07
            curve = QPainterPath(QPointF(207 + narrowing, y))
            curve.cubicTo(335, y + 18, 498, y + 18, 624 - narrowing, y)
            p.setPen(QPen(QColor(105, 198, 209, 8 if rgb else 48), 1))
            p.drawPath(curve)
        for x in range(250, 620, 34):
            curve = QPainterPath(QPointF(x, 112))
            curve.cubicTo(x - 31, 270, x + 29, 420, x, 605)
            p.setPen(QPen(QColor(91, 170, 185, 5 if rgb else 30), 1))
            p.drawPath(curve)
        midline = QPainterPath(QPointF(415, 150))
        midline.cubicTo(406, 260, 420, 420, 415, 570)
        p.setPen(QPen(QColor(123, 221, 225, 41), 1))
        p.drawPath(midline)
        p.setPen(QPen(QColor(131, 218, 223, 60), 1))
        p.setBrush(QColor("#112a35"))
        p.drawEllipse(QPointF(415, 407), 3.4, 3.4)
        p.restore()
        if self.state.view == "点云":
            p.save()
            p.setClipPath(body)
            p.setPen(QPen(QColor("#b5f0ed"), 2))
            for y in range(95, 608, 12):
                for x in range(180 + (y % 3) * 4, 653, 12):
                    p.drawPoint(x, y)
            p.restore()
        p.setPen(QColor("#85a4b1"))
        p.setFont(QFont("Microsoft YaHei", 9))
        p.drawText(QPointF(270, 675), "模拟患者表面 · 仅用于交互演示")

    def _draw_geometry_issue(self, p: QPainter) -> None:
        if self.state.scenario == "geometry" and self.state.region_complete and not self.state.geometry_repaired:
            x, y = self._work_center()
            self._issue_patch(p, x, y, QColor("#9d4edd"), "缺失几何")

    def _work_center(self) -> tuple[float, float]:
        pts = [point for stroke in self.state.strokes if stroke.kind == "work" for point in stroke.points]
        if not pts:
            return 390, 430
        return (sum(point[0] for point in pts) / len(pts), sum(point[1] for point in pts) / len(pts))

    def _issue_patch(self, p: QPainter, x: float, y: float, color: QColor, label: str) -> None:
        pulse = self._pulse_strength()
        fill_alpha = 78 + int(60 * pulse)
        outer_alpha = 120 + int(90 * pulse)
        radius_x, radius_y = 48 + 7 * pulse, 37 + 5 * pulse
        p.setBrush(QColor(color.red(), color.green(), color.blue(), fill_alpha))
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), outer_alpha), 3.5))
        p.drawEllipse(QPointF(x, y), radius_x, radius_y)
        # A second, faint ring reads as a breathing alert at a glance.
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 42 + int(55 * pulse)), 1.4))
        p.drawEllipse(QPointF(x, y), radius_x + 9 + 4 * pulse, radius_y + 7 + 3 * pulse)
        p.setBrush(QColor("#161d27"))
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 220), 2))
        p.drawRoundedRect(QRectF(x + 35, y - 50, 112, 30), 4, 4)
        p.setPen(QColor("#ffffff"))
        p.drawText(QRectF(x + 35, y - 50, 112, 30), Qt.AlignCenter, label)
