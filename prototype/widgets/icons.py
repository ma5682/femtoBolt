"""Small workstation icons rendered with Qt paths, independent of UI fonts."""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def workstation_icon(name: str, color: str = "#91A9BB") -> QIcon:
    """Return a 24 px vector-style icon backed by a crisp 2× raster.

    Drawing in logical pixels keeps the stroke width consistent across the
    toolbar. Unknown names raise early so a typo cannot create an empty button.
    """
    supported = {"fit", "zoom", "pan", "fullscreen", "cube"}
    if name not in supported:
        raise ValueError(f"Unknown workstation icon: {name}")
    pixmap = QPixmap(48, 48)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor(color), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)

    def polyline(*points):
        path = QPainterPath(QPointF(*points[0]))
        for point in points[1:]:
            path.lineTo(QPointF(*point))
        painter.drawPath(path)

    if name in ("fit", "fullscreen"):
        inset, length = (4, 4) if name == "fit" else (3, 5)
        outer = 24 - inset
        polyline((inset, inset + length), (inset, inset), (inset + length, inset))
        polyline((outer - length, inset), (outer, inset), (outer, inset + length))
        polyline((outer, outer - length), (outer, outer), (outer - length, outer))
        polyline((inset + length, outer), (inset, outer), (inset, outer - length))
        if name == "fit":
            painter.drawRoundedRect(QRectF(8, 8, 8, 8), 0.6, 0.6)
    elif name == "zoom":
        painter.drawEllipse(QRectF(3.5, 3.5, 12.5, 12.5))
        polyline((14.5, 14.5), (20.5, 20.5))
        polyline((6.8, 9.75), (12.7, 9.75))
        polyline((9.75, 6.8), (9.75, 12.7))
    elif name == "pan":
        polyline((12, 3), (12, 21))
        polyline((3, 12), (21, 12))
        polyline((9, 6), (12, 3), (15, 6))
        polyline((9, 18), (12, 21), (15, 18))
        polyline((6, 9), (3, 12), (6, 15))
        polyline((18, 9), (21, 12), (18, 15))
    else:  # cube
        polyline((12, 2.8), (20, 7.3), (20, 16.7), (12, 21.2), (4, 16.7), (4, 7.3), (12, 2.8))
        polyline((4, 7.3), (12, 12), (20, 7.3))
        polyline((12, 12), (12, 21.2))

    painter.end()
    return QIcon(pixmap)
