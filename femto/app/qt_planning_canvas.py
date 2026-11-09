"""Fixed-reference Qt painting view; masks remain in original image pixels."""
from __future__ import annotations

import numpy as np
import math
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from femto.app.qt_common import rgb_image
from femto.app.qt_visuals import draw_empty_viewport


class PlanningCanvas(QWidget):
    changed = Signal()
    editing_started = Signal()
    unchanged_stroke = Signal()
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('planningViewport')
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setToolTip('左键涂抹或擦除 · 滚轮缩放 · 右键平移')
        self.setFocusPolicy(Qt.StrongFocus)
        self.model = None
        self.work = self.forbidden = None
        self._rgb = QImage()
        self._display_rgb = QImage()
        self._display_rect = None
        self._layers = []
        self.role = 'work'
        self.erase = False
        self.brush_size = 40
        self.brush_shape = 'square'
        self.work_color = (73,135,225,76)
        self.selection_outline = False
        self._outline_path = None
        self.show_zoom = True
        self.background_color = '#06101A'
        self.editable = True
        self.support = False
        self._quality_image = None
        self._coverage_image = None
        self.zoom = 1.
        self.pan = QPointF()
        self._last = self._before = self._pan_last = self._cursor = None
        self._undo = []
        self._redo = []
        self.plan = None
        self.show_plan = False
        self.show_regions = True
        self._runs = []
        self._nominal_bands = []
        self.show_nominal_bands = False
        self._samples = []
        self._play_index = 0
        self.timer = QTimer(self)
        self.timer.setInterval(35)
        self.timer.timeout.connect(self._advance)

    def set_model(self, model, work=None, forbidden=None):
        self.model = model
        self.set_quality_overlay(None)
        self.set_coverage_overlay(None)
        self._rgb = rgb_image(model.rgb)
        display = getattr(model, 'display_rgb', None)
        self._display_rgb = QImage() if display is None else rgb_image(display)
        self._display_rect = getattr(model, 'display_rect', None)
        shape = model.rgb.shape[:2]
        self.work = np.zeros(shape, np.uint8) if work is None else np.asarray(work, np.uint8).copy()
        self.forbidden = np.zeros(shape, np.uint8) if forbidden is None else np.asarray(forbidden, np.uint8).copy()
        self._undo.clear()
        self._redo.clear()
        self._last = self._before = None
        self.set_plan(None)
        self.rebuild_layers()
        self.reset_view()

    def reset_view(self):
        self.zoom = 1.
        self.pan = QPointF()
        self.zoom_changed.emit(self.zoom)
        self.update()

    def image_rect(self):
        if self._rgb.isNull():
            return QRectF()
        scale = min(self.width() / self._rgb.width(), self.height() / self._rgb.height()) * self.zoom
        w, h = self._rgb.width() * scale, self._rgb.height() * scale
        return QRectF((self.width()-w)/2+self.pan.x(), (self.height()-h)/2+self.pan.y(), w, h)

    def fit_selection(self):
        if self.work is None:return
        selected=(self.work.astype(bool) if self.work.any() else
                  np.ones(self.work.shape,bool) if self.forbidden.any() else np.zeros(self.work.shape,bool))
        self.fit_mask(selected & ~self.forbidden.astype(bool))

    def fit_mask(self, mask):
        ys,xs=np.nonzero(mask)
        if not len(xs):self.reset_view();return
        base=min(self.width()/self._rgb.width(),self.height()/self._rgb.height())
        self.zoom=float(np.clip(min(self.width()/(xs.max()-xs.min()+1),
                                    self.height()/(ys.max()-ys.min()+1))*.85/base,.4,6.))
        scale=base*self.zoom
        self.pan=QPointF((self._rgb.width()/2-(xs.min()+xs.max()+1)/2)*scale,
                         (self._rgb.height()/2-(ys.min()+ys.max()+1)/2)*scale)
        self.zoom_changed.emit(self.zoom)
        self.update()

    def image_point(self, position):
        rect = self.image_rect()
        if rect.isEmpty() or not rect.contains(position):
            return None
        return (min(self._rgb.width()-1, max(0., (position.x()-rect.x())*self._rgb.width()/rect.width())),
                min(self._rgb.height()-1, max(0., (position.y()-rect.y())*self._rgb.height()/rect.height())))

    def rebuild_layers(self):
        self._layers = []
        self._outline_path = None
        if self.model is None:
            return
        for mask, rgba in ((self.work, self.work_color), (self.forbidden, (239,71,111,115))):
            layer = np.zeros((*mask.shape, 4), np.uint8)
            layer[mask.astype(bool)] = rgba
            self._layers.append(QImage(layer.data, layer.shape[1], layer.shape[0], layer.strides[0], QImage.Format_RGBA8888).copy())
        self.update()

    def _update_mask_layer(self, index, left, top, right, bottom):
        mask = self.work if index == 0 else self.forbidden
        image = self._layers[index]
        pixels = np.ndarray((image.height(), image.width(), 4), np.uint8,
                            buffer=image.bits(), strides=(image.bytesPerLine(), 4, 1))
        patch = pixels[top:bottom, left:right]
        selected = mask[top:bottom, left:right].astype(bool)
        patch[:] = 0
        patch[selected] = self.work_color if index == 0 else (239,71,111,115)
        if index == 0:self._outline_path = None

    def _draw_selection_outline(self, painter, rect):
        if self._outline_path is None:
            import cv2
            contours, _ = cv2.findContours(self.work, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            path = QPainterPath()
            for contour in contours:
                points = contour[:,0,:]
                if len(points) < 2:continue
                path.moveTo(float(points[0,0]), float(points[0,1]))
                for x,y in points[1:]:path.lineTo(float(x),float(y))
                path.closeSubpath()
            self._outline_path = path
        painter.save(); painter.setClipRect(rect)
        painter.translate(rect.x(), rect.y())
        painter.scale(rect.width()/self._rgb.width(), rect.height()/self._rgb.height())
        pen = QPen(QColor('#00E2D3'), 1.4); pen.setCosmetic(True)
        painter.setPen(pen); painter.setBrush(Qt.NoBrush)
        painter.drawPath(self._outline_path); painter.restore()

    def set_quality_overlay(self, rgba):
        self._quality_image = None if rgba is None else rgb_image(rgba)
        self.update()

    def set_plan(self, plan):
        self.timer.stop()
        self.plan = plan
        self._runs = []
        self._nominal_bands = []
        self._samples = []
        self._play_index = 0
        if plan:
            from femto.planning.surface_workspace import display_runs
            self._runs = list(display_runs(plan))
            # Segment separators are explicit, so playback pauses rather than interpolating a gap.
            for chain in plan.get('segments', []):
                self._samples.extend(point.get('pixel') for point in chain)
                self._samples.extend([None] * 12)
        if self.show_nominal_bands:self.set_nominal_bands(True)
        self.update()

    def set_nominal_bands(self, shown):
        self.show_nominal_bands=bool(shown)
        if shown and self.plan and self.model is not None and not self._nominal_bands:
            from femto.planning.surface_workspace import canonical_region_mesh
            from femto.planning.nominal_band_display import ribbons
            region=self.model.store.regions[0]
            mesh=canonical_region_mesh(self.model.store.mesh,region)
            self._nominal_bands=ribbons(self.plan,mesh,region['view_reference']['undistorted_intrinsic'])
        self.update()

    def play(self):
        if self.timer.isActive():
            self.timer.stop()
        elif self._samples:
            if self._play_index >= len(self._samples):
                self._play_index = 0
            self.timer.start()

    def _advance(self):
        self._play_index += 1
        if self._play_index >= len(self._samples):
            self.timer.stop()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self.background_color))
        if self._rgb.isNull():
            draw_empty_viewport(p,self.rect(),'加载模型，开始区域规划',
                                '打开已采集的三维模型或保存的区域方案。',kind='planning')
            return
        rect = self.image_rect()
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(rect, self._rgb)
        if not self._display_rgb.isNull() and self._display_rect is not None:
            x,y,w,h = self._display_rect
            sx,sy = rect.width()/self._rgb.width(),rect.height()/self._rgb.height()
            p.save();p.setClipRect(rect)
            p.drawImage(QRectF(rect.x()+x*sx,rect.y()+y*sy,w*sx,h*sy),self._display_rgb)
            p.restore()
        # Show actual coarse selection cells; smooth RGB must not suggest a
        # finer editable mask than the planner really receives.
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        if self._coverage_image is not None:
            p.drawImage(rect, self._coverage_image)
        elif self.show_regions:
            for image in self._layers:p.drawImage(rect, image)
        if self._coverage_image is None and self.support and self._quality_image is not None:
            p.drawImage(rect, self._quality_image)
        p.setRenderHint(QPainter.Antialiasing)
        if self.selection_outline and self.show_regions and self._coverage_image is None:
            self._draw_selection_outline(p, rect)
        def point(value):
            return QPointF(rect.x()+value[0]*rect.width()/self._rgb.width(), rect.y()+value[1]*rect.height()/self._rgb.height())
        if self.show_plan and self.show_nominal_bands and self._coverage_image is None:
            # Fill the union once so dense samples do not darken the display.
            path=QPainterPath();path.setFillRule(Qt.WindingFill)
            for quad in self._nominal_bands:
                path.moveTo(point(quad[0]))
                for value in quad[1:]:path.lineTo(point(value))
                path.closeSubpath()
            p.fillPath(path,QColor(30,190,215,85))
        # Automatic layout review needs both the nominal proof status and its
        # centerline visible. Drawing the centerline does not alter any proof.
        overlay_route = bool(self.plan and self.plan.get('treatment_head',{}).get('mode')=='fixed_width_auto')
        if self.show_plan and (self._coverage_image is None or overlay_route):
            for role, points in self._runs:
                if len(points)<2: continue
                path=QPainterPath(point(points[0]))
                for value in points[1:]: path.lineTo(point(value))
                pen=QPen(QColor('#fcca46'), 1.8)
                p.setPen(pen);p.drawPath(path)
        if (self._coverage_image is None and self.show_plan and self._samples and self._play_index<len(self._samples)
                and self._samples[self._play_index] is not None):
            p.setBrush(QColor('#EAF2F8'));p.setPen(QPen(QColor('#087DDE'),2))
            p.drawEllipse(point(self._samples[self._play_index]),5,5)
        if self.editable and self._cursor is not None and rect.contains(self._cursor):
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor('#ef476f' if self.erase or self.role=='forbidden' else '#06d6a0'),1.5))
            half=self.brush_size/2
            cursor = QRectF(self._cursor.x()-half,self._cursor.y()-half,self.brush_size,self.brush_size)
            if self.brush_shape == 'round':p.drawEllipse(cursor)
            else:p.drawRect(cursor)
        if self.show_zoom:
            p.setPen(QColor('#EAF2F8'))
            p.drawText(12,24,f'{self.zoom*100:.0f}%')

    def set_coverage_overlay(self, rgba):
        self._coverage_image = None
        if rgba is not None:
            value = np.ascontiguousarray(rgba, dtype=np.uint8)
            self._coverage_image = QImage(value.data, value.shape[1], value.shape[0], value.strides[0], QImage.Format_RGBA8888).copy()
        self.update()

    def _paint(self, pixel):
        from femto.annotation.square_brush import stroke
        if self.brush_shape == 'round':
            from femto.annotation.round_brush import stroke
        mask = self.work if self.role=='work' else self.forbidden
        scale=self._rgb.width()/self.image_rect().width()
        start = self._last or pixel
        size = self.brush_size*scale
        stroke(mask, start, pixel, size, self.erase)
        self._last=pixel
        # The original brush still writes the original mask. Refresh only its
        # swept bounds; two pixels enclose 1/16px rounding and raster endpoints.
        half = max(1., float(size))/2
        left = max(0, math.floor(min(start[0], pixel[0])-half)-2)
        top = max(0, math.floor(min(start[1], pixel[1])-half)-2)
        right = min(mask.shape[1], math.ceil(max(start[0], pixel[0])+half)+2)
        bottom = min(mask.shape[0], math.ceil(max(start[1], pixel[1])+half)+2)
        index = 0 if self.role=='work' else 1
        self._update_mask_layer(index, left, top, right, bottom)
        self.update()

    def finish_stroke(self):
        if self._before is None:
            return
        before=self._before
        self._before=self._last=None
        if not (np.array_equal(before[0],self.work) and np.array_equal(before[1],self.forbidden)):
            self._undo.append(before)
            self._redo.clear()
            # Keep image-scale undo bounded to 64 MiB.
            max_items=max(1,min(50,64*1024**2//(self.work.nbytes+self.forbidden.nbytes)))
            self._undo=self._undo[-max_items:]
            self.changed.emit()
        else:
            self.unchanged_stroke.emit()

    def mousePressEvent(self,event):
        self.setFocus()
        self._cursor=event.position()
        if event.button()==Qt.RightButton:
            self._pan_last=event.position();return
        pixel=self.image_point(event.position())
        if event.button()==Qt.LeftButton and self.editable and pixel is not None:
            self.set_quality_overlay(None)
            self.editing_started.emit()
            self._before=(self.work.copy(),self.forbidden.copy())
            self._last=None;self._paint(pixel)

    def mouseMoveEvent(self,event):
        self._cursor=event.position()
        if self._pan_last is not None:
            self.pan+=event.position()-self._pan_last;self._pan_last=event.position()
        elif self._before is not None and self.editable:
            pixel=self.image_point(event.position())
            if pixel is not None:self._paint(pixel)
            else:self._last=None
        self.update()

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.LeftButton:
            if self._before is not None and self.editable:
                pixel=self.image_point(event.position())
                if pixel is not None:self._paint(pixel)
            self.finish_stroke()
        elif event.button()==Qt.RightButton:self._pan_last=None

    def leaveEvent(self,event):
        self._cursor=None;self.update()

    def wheelEvent(self,event):
        old=self.image_rect()
        position=event.position()
        self.zoom=max(.4,min(6.,self.zoom*(1.12 if event.angleDelta().y()>0 else 1/1.12)))
        new=self.image_rect()
        if not old.isEmpty():
            ratio=new.width()/old.width()
            anchor=QPointF(new.x()+(position.x()-old.x())*ratio,
                           new.y()+(position.y()-old.y())*ratio)
            self.pan+=position-anchor
        self.zoom_changed.emit(self.zoom)
        self.update()

    def undo(self):
        if not self.editable or not self._undo:return
        self.finish_stroke()
        self._redo.append((self.work.copy(),self.forbidden.copy()))
        self.work,self.forbidden=self._undo.pop()
        self.rebuild_layers();self.changed.emit()

    def redo(self):
        if not self.editable or not self._redo:return
        self._undo.append((self.work.copy(),self.forbidden.copy()))
        self.work,self.forbidden=self._redo.pop()
        self.rebuild_layers();self.changed.emit()

    def clear_work(self):
        """Clear editable work without dropping imported exclusion constraints."""
        if not self.editable or self.work is None:return
        self.finish_stroke()
        self._before=(self.work.copy(),self.forbidden.copy())
        self.work[:]=0
        self.rebuild_layers();self.finish_stroke()

    def clear(self):
        if not self.editable or self.work is None:return
        self._before=(self.work.copy(),self.forbidden.copy())
        self.work[:]=0;self.forbidden[:]=0
        self.rebuild_layers();self.finish_stroke()
