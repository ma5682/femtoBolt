"""Concrete source checks for geometry gaps; calculations remain in workers."""
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QDialogButtonBox

from femto.app.region_quality import rescan_context


class GapInspectionDialog(QDialog):
    def __init__(self, model, mask, guidance, parent=None, *, allow_acquisition=True, allow_continuation=False,
                 allow_exact_partition=False):
        super().__init__(parent)
        self.action = None
        self.setWindowTitle('检查缺口来源');self.resize(720, 540)
        layout = QVBoxLayout(self)
        text = QLabel(guidance);text.setWordWrap(True);layout.addWidget(text)
        mask = np.asarray(mask, bool);depth = np.asarray(model.projection.depth)
        selected = depth[mask]
        valid = (selected >= model.projection.minimum*1000) & (selected <= model.projection.maximum*1000)
        summary = QLabel(f'定位范围：{int(mask.sum())} 像素；其中参考深度有效 {int(valid.sum())} 像素。\n'
                         '下图为同一参考坐标下的实测深度，粉色标出当前定位范围；深度有效不等于网格有效。')
        summary.setWordWrap(True);layout.addWidget(summary)
        if mask.any():
            yy, xx = np.nonzero(mask)
            y0,y1=max(0,int(yy.min())-12),min(depth.shape[0],int(yy.max())+13)
            x0,x1=max(0,int(xx.min())-12),min(depth.shape[1],int(xx.max())+13)
            raw = depth[y0:y1,x0:x1].astype(float)
            good = (raw >= model.projection.minimum*1000) & (raw <= model.projection.maximum*1000)
            lo,hi = (float(raw[good].min()),float(raw[good].max())) if good.any() else (0.,1.)
            gray = np.where(good, np.clip((raw-lo)/max(1.,hi-lo)*220+20,0,255),0).astype(np.uint8)
            rgb = np.repeat(gray[:,:,None],3,axis=2)
            region = mask[y0:y1,x0:x1]
            rgb[region] = ((rgb[region].astype(float)+[230,50,180])/2).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)
            image = QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format_RGB888).copy()
            preview = QLabel();preview.setPixmap(QPixmap.fromImage(image).scaledToWidth(660))
            preview.setMaximumHeight(300);preview.setScaledContents(True);layout.addWidget(preview)
        report = Path(model.reconstruction)/'reconstruction_report.json'
        self.audit_button = QPushButton('按原参数排查重建缺面')
        self.audit_button.setEnabled(report.is_file() and bool(np.any(mask & (model.projection.reasons == 1)
            & (depth >= model.projection.minimum*1000) & (depth <= model.projection.maximum*1000))))
        self.audit_button.clicked.connect(lambda:self.choose('audit'));layout.addWidget(self.audit_button)
        self.rescan_button = QPushButton('为该任务重新采集')
        self.rescan_button.setEnabled(allow_acquisition and rescan_context(model) is not None)
        self.rescan_button.clicked.connect(lambda:self.choose('rescan'));layout.addWidget(self.rescan_button)
        self.continue_button=QPushButton('继续建立原目标的下一块')
        self.continue_button.setVisible(allow_continuation)
        self.continue_button.clicked.connect(lambda:self.choose('continue_target'));layout.addWidget(self.continue_button)
        self.exact_button=QPushButton('精确核验数值与部分定位范围')
        self.exact_button.setVisible(allow_exact_partition)
        self.exact_button.clicked.connect(lambda:self.choose('exact_target'));layout.addWidget(self.exact_button)
        note = QLabel('重采请求只关联观测类缺口；预算与数值范围另行处理。重建排查生成独立结果，当前模型和路线保留。新采集返回后需要确认新模型、重新选区、生成路线并核验覆盖。')
        note.setWordWrap(True);layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def choose(self, action):
        self.action = action;self.accept()
