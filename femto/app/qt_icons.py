"""Small vector line icons for workstation controls."""
from functools import lru_cache

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


@lru_cache(maxsize=80)
def line_icon(name, active=False):
    paths = {
        'workspace': '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>',
        'folder': '<path d="M3 7V4h7l3 3h8v14H3Z"/><path d="M3 10h18"/>',
        'history': '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>',
        'device': '<path d="M9 3h6l1 4 4 2v6l-4 2-1 4H9l-1-4-4-2V9l4-2Z"/><circle cx="12" cy="12" r="3"/>',
        'cube': '<path d="m12 2 9 5v10l-9 5-9-5V7Zm-9 5 9 5 9-5M12 12v10"/>',
        'warning': '<path d="m12 3 10 18H2Z"/><path d="M12 9v5m0 3v1"/>',
        'refresh': '<path d="M21 10a9 9 0 0 0-16-4L2 9m0-6v6h6M3 14a9 9 0 0 0 16 4l3-3m0 6v-6h-6"/>',
        'target': '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="3"/><path d="M12 1v5m0 12v5M1 12h5m12 0h5"/>',
        'arrow': '<path d="M3 12h18m-6-6 6 6-6 6"/>',
        'search': '<circle cx="10" cy="10" r="7"/><path d="m15 15 6 6"/>',
        'document': '<path d="M5 2h9l5 5v15H5ZM14 2v6h5M8 12h8m-8 4h8"/>',
        'database': '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 4 18 4 18 0V5M3 12c0 4 18 4 18 0"/>',
        'eye': '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
        'info': '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',
        'trash': '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7"/>',
        'undo': '<path d="m8 3-6 6 6 6M2 9h12a6 6 0 0 1 0 12h-3"/>',
        'shield': '<path d="m12 2 9 4v6c0 5-9 10-9 10S3 17 3 12V6Zm-4 10 3 3 5-6"/>',
        'link': '<path d="m9 15 6-6m-8 4-2 2a4 4 0 0 0 6 6l3-3m-4-12 3-3a4 4 0 0 1 6 6l-2 2"/>',
        'play': '<path d="m7 3 14 9-14 9Z"/>',
        'pause': '<path d="M8 3v18M16 3v18"/>',
        'stop': '<rect x="4" y="4" width="16" height="16" rx="1"/>',
        'close': '<path d="m5 5 14 14M5 19 19 5"/>',
        'export': '<path d="M12 16V2m-5 5 5-5 5 5M3 14v7h18v-7"/>',
        'bell': '<path d="M4 17h16l-2-4V9a6 6 0 0 0-12 0v4Zm6 4h4"/>',
        'brush': '<path d="m14 4 6 6-9 9-6-6ZM14 4l2-2 6 6-2 2M5 13l-3 9 9-3M4 18l2 2"/>',
        'eraser': '<path d="m3 14 10-11 9 9-8 9H9ZM7 10l9 9M14 21h8"/>',
        'redo': '<path d="m16 3 6 6-6 6M22 9H10a6 6 0 0 0 0 12h3"/>',
        'sliders': '<path d="M3 6h18M3 12h18M3 18h18"/><circle cx="8" cy="6" r="2"/><circle cx="16" cy="12" r="2"/><circle cx="9" cy="18" r="2"/>',
    }
    icon = QIcon()
    for mode, state, color in ((QIcon.Normal, QIcon.Off, '#20cfff' if active else '#8eafc4'),
                               (QIcon.Normal, QIcon.On, '#20cfff'),
                               (QIcon.Active, QIcon.Off, '#20cfff'),
                               (QIcon.Active, QIcon.On, '#20cfff'),
                               (QIcon.Disabled, QIcon.Off, '#496579'),
                               (QIcon.Disabled, QIcon.On, '#496579')):
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
               f'fill="none" stroke="{color}" stroke-width="1.75" stroke-linecap="round" '
               'stroke-linejoin="round">'+paths[name]+'</svg>')
        pixmap = QPixmap(48, 48)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        QSvgRenderer(QByteArray(svg.encode())).render(painter)
        painter.end()
        icon.addPixmap(pixmap, mode, state)
    return icon
