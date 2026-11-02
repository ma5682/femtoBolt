"""Illustrative QPainter viewport; coordinates stay stable for marked regions."""
from copy import deepcopy
from math import cos, exp, pi, sin
from random import Random

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QLabel

from models.task_state import Stroke
from styles.tokens import COLORS
from widgets.medical_viewport import Medical3DViewport, ViewportScene


def color(name, alpha=None):
    result = QColor(COLORS.get(name, name))
    if alpha is not None:
        result.setAlpha(alpha)
    return result


def font(size, weight=QFont.Normal):
    result = QFont("Microsoft YaHei")
    result.setPixelSize(size)
    result.setWeight(weight)
    return result


class MockViewport(Medical3DViewport):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = ViewportScene()
        self.setMinimumSize(520, 380)
        self.setMouseTracking(True)
        self._drawing = None
        self._navigation_tool = "select"
        self._zoom = 1.0
        self._pan = QPointF()
        self._pan_last = None
        self._scale, self._offset = 1.0, QPointF()
        self._pulse_phase = 0.0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(55)
        self._pulse_timer.timeout.connect(self._pulse_tick)
        self._surface_points = self._make_surface_points()
        self.tag = QLabel(self)
        self.tag.setObjectName("viewportTag")
        self.tag.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hint = QLabel(self)
        self.hint.setObjectName("viewportHint")
        self.hint.setWordWrap(True)
        self.hint.setFixedWidth(228)
        self.hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.legend = QLabel(self)
        self.legend.setObjectName("viewportLegend")
        self.legend.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.legend.hide()
        self.setToolTip("模拟数据 · 左键标记 · 右键平移 · 滚轮缩放")
        self.set_scene(self.state)

    def set_scene(self, scene):
        self.state = deepcopy(scene)
        self._drawing = None
        self.tag.setText(f"{scene.view} · {'等待画面' if self._is_empty() else '模拟数据'}")
        missing = scene.scenario == "geometry" and scene.region_complete and not scene.geometry_repaired
        problem = (
            "三维数据缺失\n已画区域保留，返回补采后继续。" if missing
            else "采集质量不足\n移动相机补充高亮区域。" if scene.capture_phase == "quality_bad"
            else "轨迹断开\n请返回规划检查高亮位置。" if scene.trajectory_failed
            else ""
        )
        self.hint.setText(problem)
        self.hint.setVisible(bool(problem))
        if problem or scene.issue_focus in ("geometry", "route", "edge"):
            if not self._pulse_timer.isActive():
                self._pulse_timer.start()
        else:
            self._pulse_timer.stop()
        self._position_overlays()
        self._update_cursor()
        self.update()

    def _is_empty(self):
        return self.state.stage == 0 and self.state.capture_phase == "idle"

    def set_navigation_tool(self, tool):
        if tool not in ("select", "pan", "zoom"):
            raise ValueError(f"未知视图工具: {tool}")
        self._navigation_tool = tool
        self._pan_last = None
        self._update_cursor()

    def _update_cursor(self):
        if self._navigation_tool == "pan":
            cursor = Qt.OpenHandCursor
        elif self._navigation_tool == "zoom" or self.state.editable:
            cursor = Qt.CrossCursor
        else:
            cursor = Qt.ArrowCursor
        self.setCursor(cursor)

    def zoom_in(self):
        self._zoom = min(2.2, self._zoom * 1.15)
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPointF()
        self.update()

    def _pulse_tick(self):
        self._pulse_phase = (self._pulse_phase + 0.22) % (2 * pi)
        self.update()

    def _pulse_strength(self):
        return 0.5 + 0.5 * sin(self._pulse_phase)

    def _position_overlays(self):
        self.tag.adjustSize()
        self.tag.move(136 if self.state.stage == 1 else 16, 14)
        self.hint.adjustSize()
        self.hint.move(max(16, self.width() - self.hint.width() - 16), 14)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_overlays()

    def _view_transform(self):
        scale = min(self.width() / 850, self.height() / 700) * self._zoom
        offset = QPointF((self.width() - 850 * scale) / 2, (self.height() - 700 * scale) / 2) + self._pan
        self._scale, self._offset = scale, offset
        return scale, offset

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), color("viewport"))
        self._draw_space_grid(p)
        if self._is_empty():
            self._draw_empty(p)
            p.end()
            return
        scale, offset = self._view_transform()
        p.save()
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
                    self._issue_patch(p, x - 60, y + 70, color("warning"), "边缘未覆盖")
                else:
                    self._issue_patch(p, x, y, color("error"), "轨迹断开")
        if self.state.reference_ready:
            self._draw_reference(p)
        p.restore()
        if self.state.show_axes or self.state.view in ("三维", "点云", "模型", "三维模型"):
            self._draw_axes(p)
        self._draw_legend(p)
        p.setFont(font(12))
        p.setPen(color("subtle"))
        p.drawText(QRectF(16, self.height() - 28, max(100, self.width() - 245), 18), Qt.AlignLeft | Qt.AlignVCenter, "模拟表面 · 仅用于交互演示")
        p.end()

    def _draw_space_grid(self, p):
        w, h = self.width(), self.height()
        p.save()
        p.setPen(QPen(color("grid", 76), 1))
        horizon = h * 0.57
        for x in range(0, w + 1, 64):
            p.drawLine(QPointF(x, 0), QPointF(x, horizon))
        for y in range(0, int(horizon), 64):
            p.drawLine(QPointF(0, y), QPointF(w, y))
        for x in range(-w, 2 * w + 1, max(70, w // 10)):
            p.drawLine(QPointF(w * .5, horizon - h * .12), QPointF(x, h))
        for fraction in (.06, .16, .30, .49, .72, 1.0):
            y = horizon + (h - horizon) * fraction
            p.drawLine(QPointF(0, y), QPointF(w, y))
        p.restore()

    def _draw_empty(self, p):
        p.save()
        cx, cy = self.width() / 2, self.height() / 2
        frame_w = min(self.width() * .70, 520)
        frame_h = min(self.height() * .54, 310)
        rect = QRectF(cx - frame_w / 2, cy - frame_h / 2 - 14, frame_w, frame_h)
        p.setPen(QPen(color("border"), 1.5))
        corner = 25
        for x, y, sx, sy in ((rect.left(), rect.top(), 1, 1), (rect.right(), rect.top(), -1, 1), (rect.left(), rect.bottom(), 1, -1), (rect.right(), rect.bottom(), -1, -1)):
            p.drawLine(QPointF(x, y), QPointF(x + sx * corner, y))
            p.drawLine(QPointF(x, y), QPointF(x, y + sy * corner))
        icon_center = QPointF(cx, cy - 50)
        p.setPen(QPen(color("subtle"), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(cx - 28, cy - 70, 56, 39), 4, 4)
        p.drawEllipse(icon_center, 10, 10)
        p.drawLine(QPointF(cx - 12, cy - 70), QPointF(cx - 6, cy - 77))
        p.drawLine(QPointF(cx - 6, cy - 77), QPointF(cx + 10, cy - 77))
        p.drawLine(QPointF(cx + 10, cy - 77), QPointF(cx + 16, cy - 70))
        p.setFont(font(22, QFont.DemiBold))
        p.setPen(color("text"))
        p.drawText(QRectF(cx - 210, cy - 5, 420, 36), Qt.AlignCenter, "等待相机画面")
        p.setFont(font(14))
        p.setPen(color("muted"))
        p.drawText(QRectF(cx - 220, cy + 37, 440, 25), Qt.AlignCenter, "请确认 Femto Bolt 已连接")
        p.setPen(QPen(color("subtle", 130), 1))
        cross_y = rect.bottom() - 21
        p.drawLine(QPointF(cx - 5, cross_y), QPointF(cx + 5, cross_y))
        p.drawLine(QPointF(cx, cross_y - 5), QPointF(cx, cross_y + 5))
        p.setFont(font(12))
        p.setPen(color("subtle"))
        p.drawText(QRectF(18, self.height() - 33, self.width() - 36, 20), Qt.AlignLeft | Qt.AlignVCenter, "参考位置未建立  /  W0 待建立")
        p.restore()

    @staticmethod
    def _surface_path():
        path = QPainterPath(QPointF(369, 59))
        path.cubicTo(393, 66, 438, 66, 461, 59)
        path.cubicTo(462, 88, 471, 107, 490, 117)
        path.cubicTo(522, 127, 578, 134, 608, 156)
        path.cubicTo(638, 177, 651, 212, 647, 240)
        path.cubicTo(635, 263, 613, 277, 588, 282)
        path.cubicTo(579, 326, 566, 357, 566, 408)
        path.cubicTo(565, 457, 557, 480, 563, 518)
        path.cubicTo(567, 551, 574, 579, 551, 600)
        path.cubicTo(500, 621, 335, 621, 281, 600)
        path.cubicTo(258, 579, 265, 551, 269, 518)
        path.cubicTo(275, 480, 267, 457, 266, 408)
        path.cubicTo(266, 357, 253, 326, 244, 282)
        path.cubicTo(219, 277, 197, 263, 185, 240)
        path.cubicTo(181, 212, 194, 177, 224, 156)
        path.cubicTo(254, 134, 310, 127, 342, 117)
        path.cubicTo(361, 107, 368, 88, 369, 59)
        path.closeSubpath()
        return path

    def _make_surface_points(self):
        rng = Random(732)
        path = self._surface_path()
        points = []
        for y in range(58, 618, 5):
            for x in range(180, 652, 5):
                px, py = x + rng.uniform(-2, 2), y + rng.uniform(-2, 2)
                if not path.contains(QPointF(px, py)):
                    continue
                center_light = max(0, cos((px - 416) / 240 * pi / 2))
                chest = exp(-((py - 233) / 90) ** 2) * (.65 + .35 * cos((px - 416) / 78))
                side_light = .10 * sin(px * .043 + py * .012)
                strength = max(.18, min(1, .20 + .49 * center_light + .19 * chest + side_light + rng.uniform(-.14, .14)))
                points.append((px, py, strength, rng.uniform(.7, 1.2)))
        return points

    @staticmethod
    def _acquired_path():
        path = QPainterPath(QPointF(338, 218))
        path.cubicTo(371, 195, 414, 210, 438, 199)
        path.cubicTo(466, 192, 483, 214, 478, 237)
        path.cubicTo(497, 248, 484, 277, 491, 302)
        path.cubicTo(482, 322, 490, 353, 474, 369)
        path.cubicTo(480, 393, 458, 410, 465, 435)
        path.cubicTo(452, 453, 464, 480, 440, 494)
        path.cubicTo(414, 504, 397, 484, 375, 491)
        path.cubicTo(353, 479, 325, 474, 320, 450)
        path.cubicTo(301, 428, 319, 406, 302, 384)
        path.cubicTo(287, 361, 305, 338, 297, 313)
        path.cubicTo(293, 286, 317, 271, 314, 249)
        path.cubicTo(312, 234, 326, 230, 338, 218)
        path.closeSubpath()
        return path

    def _draw_surface(self, p):
        body = self._surface_path()
        rgb = self.state.view in ("RGB", "固定 RGB", "RGB 路线图", "选区 RGB")
        cloud = self.state.view == "点云"
        depth_mode = self.state.view == "深度"
        mesh = not (rgb or cloud or depth_mode)
        p.save()
        p.setPen(Qt.NoPen)
        p.setBrush(color("panel", 120))
        p.drawEllipse(QRectF(235, 624, 365, 18))
        gradient = QLinearGradient(220, 0, 610, 0)
        gradient.setColorAt(0, color("viewport"))
        gradient.setColorAt(.30, color("subtle" if rgb else "secondary"))
        gradient.setColorAt(.62, color("muted" if rgb else "hover" if mesh else "control"))
        gradient.setColorAt(1, color("viewport"))
        p.setBrush(gradient)
        p.setPen(QPen(color("muted", 90), 1.1))
        p.drawPath(body)
        p.setClipPath(body)
        sheen = QRadialGradient(QPointF(387, 254), 270)
        sheen.setColorAt(0, color("text" if rgb else "muted", 40 if rgb else 35))
        sheen.setColorAt(1, color("viewport", 0))
        p.fillRect(QRectF(175, 55, 485, 570), sheen)
        if rgb or mesh:
            # Broad matte highlights describe volume without implying a photo.
            for cx, cy, radius, strength in ((334, 238, 116, 22), (490, 238, 116, 17), (405, 429, 155, 22)):
                contour = QRadialGradient(QPointF(cx, cy), radius)
                contour.setColorAt(0, color("text", strength))
                contour.setColorAt(1, color("text", 0))
                p.fillRect(QRectF(cx - radius, cy - radius, radius * 2, radius * 2), contour)
        if depth_mode:
            depth = QLinearGradient(0, 70, 0, 620)
            depth.setColorAt(0, color("primary_pressed"))
            depth.setColorAt(.25, color("primary_hover"))
            depth.setColorAt(.58, color("good"))
            depth.setColorAt(.84, color("warning"))
            depth.setColorAt(1, color("pending"))
            p.fillPath(body, depth)
        p.setBrush(Qt.NoBrush)
        if mesh or depth_mode:
            for y in range(136, 613, 23 if mesh else 35):
                curve = QPainterPath(QPointF(178, y))
                curve.cubicTo(294, y + 31, 505, y + 31, 655, y - 7)
                p.setPen(QPen(color("muted", 94) if mesh else color("viewport", 45), .9))
                p.drawPath(curve)
        if mesh:
            for x in range(230, 616, 31):
                curve = QPainterPath(QPointF(x, 118))
                curve.cubicTo(x + (416 - x) * .10, 283, x + (416 - x) * .19, 393, x, 612)
                p.setPen(QPen(color("muted", 62), .8))
                p.drawPath(curve)
        if cloud:
            for px, py, strength, radius in self._surface_points:
                alpha = int(44 + strength * 150)
                p.setPen(QPen(color("muted" if strength < .55 else "text", alpha), radius * 1.1))
                p.drawPoint(QPointF(px, py))
        p.restore()

    def _draw_capture(self, p):
        progress = min(1.0, max(.12, self.state.capture_progress / 6))
        acquired = self._acquired_path()
        p.save()
        p.setClipPath(self._surface_path())
        p.setClipRect(QRectF(175, 90, 490, 130 + 420 * progress), Qt.IntersectClip)
        p.setBrush(color("good", 25 if self.state.view == "点云" else 42 if self.state.view != "深度" else 0))
        p.setPen(QPen(color("good", 165), 1.35))
        p.drawPath(acquired)
        if self.state.view == "点云":
            for px, py, strength, radius in self._surface_points:
                if acquired.contains(QPointF(px, py)):
                    p.setPen(QPen(color("good", int(95 + 150 * strength)), radius * 1.4))
                    p.drawPoint(QPointF(px, py))
                elif 470 < px < 518 + 7 * sin(py / 22) and 208 < py < 502:
                    p.setPen(QPen(color("pending", 185), radius * 1.25))
                    p.drawPoint(QPointF(px, py))
        else:
            pending = QPainterPath(QPointF(489, 212))
            pending.cubicTo(525, 262, 513, 313, 515, 363)
            pending.cubicTo(506, 406, 521, 460, 484, 497)
            p.setPen(QPen(color("pending", 160), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawPath(pending)
        p.restore()
        if self.state.capture_phase == "quality_bad":
            self._issue_patch(p, 315, 490, color("warning"), "待补采")
        if self.state.issue_focus == "geometry":
            p.save()
            p.setClipPath(self._surface_path())
            self._draw_strokes(p)
            p.restore()
            self._draw_geometry_issue(p)

    def _draw_strokes(self, p):
        for stroke in self.state.strokes:
            if not stroke.points:
                continue
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(color("good", 102) if stroke.kind == "work" else color("error", 125), 66, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
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
            if not segment.points:
                continue
            p.setPen(QPen(color("warning" if segment.kind == "scan" else "primary"), 2.6 if segment.kind == "scan" else 1.8, Qt.SolidLine if segment.kind == "scan" else Qt.DashLine, Qt.RoundCap))
            path = QPainterPath(QPointF(*segment.points[0]))
            for point in segment.points[1:]:
                path.lineTo(QPointF(*point))
            p.drawPath(path)

    def _draw_reference(self, p):
        p.save()
        origin = QPointF(415, 392)
        p.setPen(QPen(color("primary_hover"), 1.6))
        p.setBrush(color("viewport", 180))
        p.drawEllipse(origin, 10, 10)
        p.drawLine(origin + QPointF(-18, 0), origin + QPointF(18, 0))
        p.drawLine(origin + QPointF(0, -18), origin + QPointF(0, 18))
        p.setFont(font(14, QFont.DemiBold))
        p.setPen(color("text"))
        p.drawText(origin + QPointF(17, -14), "W0")
        p.restore()

    def _draw_axes(self, p):
        p.save()
        origin = QPointF(44, self.height() - 66)
        p.setFont(font(12, QFont.DemiBold))
        for dx, dy, name, tone in ((38, 14, "X", "error"), (29, -20, "Y", "good"), (0, -43, "Z", "primary")):
            target = origin + QPointF(dx, dy)
            p.setPen(QPen(color(tone), 1.7))
            p.drawLine(origin, target)
            p.drawEllipse(target, 1.8, 1.8)
            p.drawText(target + QPointF(5, -2), name)
        p.restore()

    def _draw_legend(self, p):
        p.save()
        if self.state.stage == 0:
            rows = [("good", "已采区域"), ("pending", "待补采"), ("primary", "参考位置（W0）" if self.state.reference_ready else "参考位置待建立")]
        elif self.state.stage == 1:
            rows = [("good", "规划区"), ("error", "禁入区")]
            if self.state.scenario == "geometry" and not self.state.geometry_repaired:
                rows.append(("warning", "缺失三维数据"))
        else:
            rows = [("warning", "扫描段"), ("primary", "连接段（不计覆盖）"), ("error", "禁入区")]
        width, height = 190, 18 + 25 * len(rows)
        x, y = self.width() - width - 16, self.height() - height - 16
        p.setBrush(color("panel", 238))
        p.setPen(QPen(color("border"), 1))
        p.drawRoundedRect(QRectF(x, y, width, height), 5, 5)
        p.setFont(font(12))
        for index, (tone, label) in enumerate(rows):
            cy = y + 19 + index * 25
            p.setPen(Qt.NoPen)
            p.setBrush(color(tone))
            p.drawEllipse(QPointF(x + 17, cy), 4, 4)
            p.setPen(color("muted"))
            p.drawText(QRectF(x + 30, cy - 10, width - 36, 20), Qt.AlignVCenter, label)
        p.restore()

    def _draw_geometry_issue(self, p):
        if self.state.scenario == "geometry" and self.state.region_complete and not self.state.geometry_repaired:
            x, y = self._work_center()
            self._issue_patch(p, x, y, color("warning"), "缺失三维")

    def _work_center(self):
        points = [point for stroke in self.state.strokes if stroke.kind == "work" for point in stroke.points]
        if not points:
            return 390, 430
        return sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)

    def _issue_patch(self, p, x, y, tone, label):
        p.save()
        pulse = self._pulse_strength()
        fill, outline = QColor(tone), QColor(tone)
        fill.setAlpha(35 + int(25 * pulse))
        outline.setAlpha(145 + int(85 * pulse))
        p.setBrush(fill)
        p.setPen(QPen(outline, 1.6))
        p.drawEllipse(QPointF(x, y), 48 + 5 * pulse, 37 + 4 * pulse)
        p.setBrush(color("panel", 240))
        p.drawRoundedRect(QRectF(x + 35, y - 50, 112, 30), 4, 4)
        p.setFont(font(13, QFont.DemiBold))
        p.setPen(color("text"))
        p.drawText(QRectF(x + 35, y - 50, 112, 30), Qt.AlignCenter, label)
        p.restore()

    def _scene_point(self, position):
        scale, offset = self._view_transform()
        return (position.x() - offset.x()) / scale, (position.y() - offset.y()) / scale

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton or (event.button() == Qt.LeftButton and self._navigation_tool == "pan"):
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._navigation_tool == "zoom":
            self.zoom_in()
            event.accept()
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
                self.update()
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
        if self._pan_last is not None:
            self._pan_last = None
            self._update_cursor()
            return
        if self._drawing is not None:
            self._drawing = None
            self.regions_edited.emit(deepcopy(self.state.strokes))
            self.update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self._zoom = min(2.2, max(.65, self._zoom * (1.1 if event.angleDelta().y() > 0 else 1 / 1.1)))
        self.update()
        event.accept()
