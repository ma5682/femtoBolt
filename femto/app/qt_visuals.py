"""Shared display-only painting for the Qt workstation's empty viewports."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QRadialGradient


def draw_empty_viewport(painter, rect, title, detail, *, kind='capture'):
    """Draw a quiet reference grid and an empty state, never synthetic camera data."""
    painter.save()
    background = QRadialGradient(QPointF(rect.width()*.5, rect.height()*.72), rect.width()*.7)
    background.setColorAt(0, QColor('#103b54'))
    background.setColorAt(.55, QColor('#092333'))
    background.setColorAt(1, QColor('#051520'))
    painter.fillRect(rect, background)
    painter.setPen(QPen(QColor('#1b455c'), 1))
    for x in range(24, rect.width(), 42):
        for y in range(24, rect.height(), 42):
            painter.drawPoint(x, y)
    # Decorative perspective grid belongs only to the empty state. Acquired
    # geometry keeps its calibrated orthographic grid and metric scale.
    horizon = rect.height()*.4
    vanishing = QPointF(rect.width()/2, horizon)
    painter.setPen(QPen(QColor(47, 108, 141, 65), .8))
    for bottom in range(-rect.width(), rect.width()*2, max(80, rect.width()//10)):
        painter.drawLine(vanishing, QPointF(bottom, rect.height()))
    for fraction in (.06, .12, .20, .31, .46, .65, .88, 1.):
        y = horizon+(rect.height()-horizon)*fraction
        painter.drawLine(QPointF(0,y), QPointF(rect.width(),y))
    painter.setRenderHint(QPainter.Antialiasing)
    cx, cy = rect.center().x(), rect.center().y() - 56
    painter.setBrush(QColor(12, 70, 99, 80))
    painter.setPen(QPen(QColor('#245674'), 1))
    painter.drawEllipse(QRectF(cx-110,cy+8,220,55))
    painter.drawRect(QRectF(cx-78,cy-48,156,98))
    painter.drawLine(QPointF(cx-78,cy-48),QPointF(cx-45,cy-68))
    painter.drawLine(QPointF(cx+78,cy-48),QPointF(cx+45,cy-68))
    painter.setBrush(Qt.NoBrush)
    painter.setPen(QPen(QColor('#a2d6ef'), 2))
    if kind == 'planning':
        points = [QPointF(cx, cy-20), QPointF(cx+19, cy-10), QPointF(cx+19, cy+12),
                  QPointF(cx, cy+22), QPointF(cx-19, cy+12), QPointF(cx-19, cy-10)]
        for index, point in enumerate(points):
            painter.drawLine(point, points[(index+1) % len(points)])
        painter.drawLine(points[1], QPointF(cx, cy))
        painter.drawLine(points[5], QPointF(cx, cy))
        painter.drawLine(QPointF(cx, cy), points[3])
    else:
        painter.drawRoundedRect(QRectF(cx-23, cy-16, 46, 33), 5, 5)
        painter.drawEllipse(QPointF(cx, cy), 8, 8)
        painter.drawLine(QPointF(cx-10, cy-21), QPointF(cx+10, cy-21))
    font = painter.font()
    font.setPixelSize(21)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor('#DDE9F0'))
    painter.drawText(QRectF(20, cy+53, rect.width()-40, 34), Qt.AlignCenter, title)
    font.setPixelSize(13)
    font.setBold(False)
    painter.setFont(font)
    painter.setPen(QColor('#91A7B6'))
    painter.drawText(QRectF(24, cy+92, rect.width()-48, 55),
                     Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, detail)
    painter.setPen(QPen(QColor('#233846'), 1))
    painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(QRectF(rect).adjusted(.5, .5, -.5, -.5), 6, 6)
    painter.restore()
