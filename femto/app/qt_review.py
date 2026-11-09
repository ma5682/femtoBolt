"""Read-only saved-result review; editing always enters a separate workspace."""
from pathlib import Path

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QScrollArea, QCheckBox

from femto.app.qt_common import TaskRunner, show_text, breakable_text
from femto.app.qt_planning_canvas import PlanningCanvas
from femto.app.qt_scene_panel import ScenePanel
from femto.app.qt_region_quality import RegionQualityPanel
from femto.app.qt_coverage import CoveragePanel
from femto.app.region_quality import rescan_context
from femto.app.qt_planning_io import load_source, masks_from_region
from femto.app.qt_planning import review_findings
from femto.app.qt_workflow import WorkflowProgress, STEPS
from femto.planning.surface_workspace import report_text


class ReviewPage(QWidget):
    rescan_requested=Signal()
    edit_requested=Signal(str)
    capture_requested=Signal(str)
    history_requested=Signal()
    new_capture_requested=Signal()
    simulation_requested=Signal(str)
    loaded=Signal(str)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.source=self.model=self.region=self.plan=None
        self._workflow_view=None
        self.runner=TaskRunner(self)
        self.runner.progress.connect(self._status)
        self.runner.finished.connect(self.refresh)
        self.runner.cancel_requested.connect(self._cancellation_requested)
        layout=QVBoxLayout(self)
        self.workflow=WorkflowProgress();layout.addWidget(self.workflow)
        self.heading=QLabel('历史结果 · 只读回看');self.heading.setObjectName('heading')
        layout.addWidget(self.heading)
        self.source_label=QLabel('选择已保存结果后查看来源、区域与路线。')
        self.source_label.setTextFormat(Qt.PlainText)
        self.source_label.setWordWrap(True)
        source_row=QHBoxLayout();source_row.addWidget(self.source_label,1);layout.addLayout(source_row)
        self.source_toggle=QPushButton('来源详情');self.source_toggle.setCheckable(True)
        self.source_toggle.setObjectName('disclosureButton');source_row.addWidget(self.source_toggle)
        self.source_details=QLabel();self.source_details.setTextFormat(Qt.PlainText)
        self.source_details.setWordWrap(True);self.source_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.source_details);self.source_details.hide()
        self.source_toggle.toggled.connect(self.source_details.setVisible)
        self.source_toggle.toggled.connect(lambda shown:self.source_toggle.setText('收起来源' if shown else '来源详情'))
        self.layer_toolbar=QWidget()
        row=QHBoxLayout(self.layer_toolbar);row.setContentsMargins(0,0,0,0)
        self.layers=QComboBox();self.layers.addItems(['参考画面','工作区域与禁区','已保存路线'])
        self.layers.currentIndexChanged.connect(self._layer);row.addWidget(self.layers)
        self.fit=QPushButton('适应选区 / 模型');self.fit.clicked.connect(self._fit);row.addWidget(self.fit)
        self.play=QPushButton('播放 / 暂停');self.play.clicked.connect(self._play);row.addWidget(self.play)
        row.addStretch();layout.addWidget(self.layer_toolbar)
        body=QHBoxLayout()
        self.canvas=PlanningCanvas();self.canvas.editable=False
        self.canvas.setToolTip('只读回看 · 滚轮缩放 · 右键平移')
        self.scene_panel=ScenePanel(self.canvas,readonly=True);body.addWidget(self.scene_panel,1)
        self.scene_panel.mode.setItemText(0,'参考画面')
        self.scene_panel.mode_changed.connect(self._view_mode_changed)
        panel=QWidget();side=QVBoxLayout(panel)
        self.summary=QLabel();self.summary.setWordWrap(True);side.addWidget(self.summary)
        self.summary.setTextFormat(Qt.PlainText)
        self.coverage=CoveragePanel(readonly=True);side.insertWidget(0,self.coverage)
        self.coverage.layer_changed.connect(self._coverage_layer)
        self.coverage.focus_requested.connect(self._focus_coverage)
        self.quality_toggle=QCheckBox('显示当前选区几何检查色层');side.addWidget(self.quality_toggle)
        self.quality_toggle.toggled.connect(self._quality_toggle)
        self.quality=RegionQualityPanel(self.canvas);side.addWidget(self.quality)
        self.quality.focus_requested.connect(self._focus_quality)
        self.quality.rescan_requested.connect(self.rescan_requested)
        self.report=QPushButton('查看完整规划检查');self.report.clicked.connect(self.show_report);side.addWidget(self.report)
        self.capture=QPushButton('查看关联采集模型');self.capture.clicked.connect(self._capture);side.addWidget(self.capture)
        side.addStretch()
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(panel);scroll.setMinimumWidth(300);scroll.setMaximumWidth(360)
        self.inspector_scroll=scroll
        body.addWidget(scroll);layout.addLayout(body,1)
        self.status=QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        self.status.setTextFormat(Qt.PlainText)
        footer=QHBoxLayout()
        history=QPushButton('返回历史');history.clicked.connect(self.history_requested);footer.addWidget(history)
        fresh=QPushButton('新建采集');fresh.clicked.connect(self.new_capture_requested);footer.addWidget(fresh)
        history.hide();fresh.hide()
        self.cancel=QPushButton('取消读取');self.cancel.clicked.connect(self.runner.cancel);footer.addWidget(self.cancel)
        footer.addStretch()
        self.simulate=QPushButton('用此方案模拟联调')
        self.simulate.setMinimumHeight(48)
        self.simulate.clicked.connect(self._simulate);footer.addWidget(self.simulate)
        self.edit=QPushButton('基于此版本修改');self.edit.setProperty('role','primary');self.edit.setMinimumHeight(48)
        self.edit.clicked.connect(self._edit);footer.addWidget(self.edit);layout.addLayout(footer)
        self.refresh()

    @property
    def busy(self):return self.runner.busy

    def _status(self,text):
        self.status.setText(breakable_text(str(text)))
        self.status.setToolTip(str(text))
        self.status.setVisible(str(text) != '已加载保存版本')

    def _cancellation_requested(self):
        if self.runner.cancelling:
            self._status('正在取消读取，请等待当前处理结束；当前记录保留。')
            self.refresh()

    def load_source(self,path, *, expected_record=None):
        if self.busy:return False
        path=Path(path).resolve()
        self.quality.invalidate('正在切换来源，当前检查已清除。')
        self.scene_panel.hide()
        self._status('正在读取并核验保存记录…')
        def done(result):
            if self.runner._cancel.is_set():
                failed(RuntimeError('读取已取消')); return
            model, opened = result
            self.install_model(model,opened)
            self.loaded.emit(str(path))
        def failed(error):
            self.scene_panel.setVisible(self.model is not None)
            self.quality.request()
            self._status('记录未切换，读取未完成：'+str(error))
        def job(cancel, progress):
            from femto.annotation.fixed_rgb_geometry import GeometryBudget
            from femto.planning.catalog import scan_capture_exports
            check = GeometryBudget(cancel).check
            capture = path/'reconstruction' if (path/'reconstruction').is_dir() else path
            opened = path
            if (capture/'reconstruction_report.json').is_file():
                progress('查找此采集关联的保存方案')
                for row in scan_capture_exports(capture, check=check):
                    if row.get('project') and row.get('file_status') == '工程资产指纹一致':
                        opened = Path(row['project']); break
            model = load_source(opened, cancel, progress, require_confirmed=False)
            if opened != path and Path(getattr(model, 'capture_source', '') or '.') != capture:
                raise ValueError('保存方案与所选采集的来源核验不一致，请在区域与轨迹中单独查看。')
            if not model.store.regions:
                from femto.geometry.local_region import model_selection
                from femto.app.qt_planning_io import extract_region
                import numpy as np
                mask = model_selection(model)
                if mask is not None:
                    progress('恢复已保存的涂抹区域')
                    model, _ = extract_region(model, mask, np.zeros_like(mask), '工作区域 01', '', cancel)
            if expected_record is not None:
                from femto.app.task_records import verify_record_context
                verify_record_context(model, expected_record)
            if cancel.is_set(): raise RuntimeError('读取已取消')
            return model, opened
        result=self.runner.start(job,done,failed)
        self.refresh();return result

    def install_model(self,model,source):
        # Reinstalling the same model also rebuilds the reference canvas. Reset
        # the layer control so its checked state cannot outlive that overlay.
        self.coverage.show_layer.setChecked(False)
        self.model=model;self.source=Path(source).resolve()
        self.region=model.store.regions[0] if model.store.regions else None
        self.plan=getattr(model,'saved_plan',None)
        self._workflow_view=5 if self.plan else 3 if self.region else 2
        self.canvas.set_model(model,*(masks_from_region(self.region,model.rgb.shape[:2]) if self.region else (None,None)))
        self.quality.request()
        self.canvas.editable=False;self.canvas.set_plan(self.plan)
        self.scene_panel.show()
        self.layers.setCurrentIndex(2 if self.plan else 1 if self.region else 0)
        self._layer(self.layers.currentIndex())
        if self.region:self.canvas.fit_selection()
        task=getattr(model,'task_context',None) or {}
        revision=getattr(model,'revision_id',None)
        parent=getattr(model,'parent_revision_id',None)
        self.source_label.setText(breakable_text(f"{task.get('name','历史记录（未填写任务信息）')}"))
        self.source_details.setText(breakable_text(f"来源：{self.source}"))
        self.source_label.setToolTip(str(self.source))
        verification=getattr(model,'capture_source_verification',None)
        if verification:
            if verification['status'] not in ('verified','not_recorded'):
                self.source_label.setText(self.source_label.text()+'\n'+verification['message'])
            for issue in verification['issues']:
                self.source_details.setText(self.source_details.text()+breakable_text(f"\n{issue['reason']}：{issue['source']}"))
        lines=[f"任务时间：{task.get('created_at','未记录')}",f"保存版本：{revision or '旧记录，未提供版本标识'}"]
        if parent:lines.append('基于版本：'+str(parent))
        if self.region:
            lines.extend(['区域：'+self.region.get('name','未命名'),f"工作块：{len(self.region.get('selection_intent',{}).get('components',[]))}"])
        if self.plan:
            lines.append(f"保存行距：{self.plan['spacing_mm']:g} mm")
            findings=review_findings(self.plan,self.region)
            lines.append('保存路线的待处理项：\n'+'\n'.join('• '+item for item in findings) if findings else '保存路线报告无待处理项；当前参考影像质量见下方检查。')
        else:lines.append(getattr(model,'plan_warning',None) or '此记录没有可回显的已保存路线。')
        if verification:lines.append(verification['message'])
        elif not getattr(model,'capture_source',None):lines.append('关联采集未能核验；当前仅显示此记录自带的参考画面与几何。')
        if getattr(model,'task_warning',None):lines.append('任务信息提示：'+model.task_warning)
        self.edit.setToolTip('修改后保存为新版本，原版本保留。')
        self.summary.setText('\n'.join(lines))
        self.source_details.setText(self.source_details.text()+'\n\n'+self.summary.text())
        self._status('已加载保存版本')
        self.refresh()

    def _quality_toggle(self, checked):
        if checked:
            self.coverage.show_layer.setChecked(False)
            # This evidence is bound to reference pixels, not VTK triangles.
            self.scene_panel.show_reference()
        self.canvas.support=checked;self.canvas.update()

    def _coverage_layer(self,active,reason):
        display=self.coverage.display if active else None
        if active:
            self.canvas.timer.stop();self.quality_toggle.setChecked(False)
        self.canvas.set_coverage_overlay(None if display is None else display.rgba(reason, pixel_reason=self.coverage.reference_reason))
        self.scene_panel.set_coverage(display,reason)
        self.play.setEnabled(bool(self.plan and not self.busy and self.layers.currentIndex()==2 and not active))

    def _focus_coverage(self,reason):
        if self.busy or self.coverage.display is None:return
        self.coverage.show_layer.setChecked(True)
        if not self.scene_panel.focus_coverage(reason):
            self.scene_panel.show_reference();self.canvas.fit_mask(self.coverage.display.mask(reason, pixel_reason=self.coverage.reference_reason))

    def _focus_quality(self):
        self.scene_panel.show_reference();self.quality_toggle.setChecked(True)

    def _layer(self,index):
        if index == 1 and not self.region or index == 2 and not self.plan:
            self.layers.setCurrentIndex(1 if self.region else 0)
            return
        self._workflow_view=5 if index==2 and self._workflow_view==5 else (2,3,4)[index]
        self.canvas.show_regions=index>0
        self.canvas.show_plan=index==2
        self.canvas.timer.stop();self.canvas.update();self.refresh()

    def _fit(self):
        if self.scene_panel.is_3d:self.scene_panel.reset_camera()
        else:self.canvas.fit_selection()

    def _view_mode_changed(self,is_3d):
        if not is_3d:self.scene_panel.notice.hide()
        labels=(['三维模型','已确认工作区域','已保存路线'] if is_3d else
                ['参考画面','工作区域与禁区','已保存路线'])
        for index,label in enumerate(labels):self.layers.setItemText(index,label)

    def _play(self):
        self.scene_panel.show_reference();self.canvas.play()

    def show_workflow_step(self,index: int) -> bool:
        if self.busy or self.model is None or index not in (2,3,4,5):return False
        if index==3 and self.region is None or index>3 and self.plan is None:return False
        self._workflow_view=index
        self.layers.setCurrentIndex(0 if index==2 else 1 if index==3 else 2)
        self.canvas.timer.stop()
        self.refresh()
        return True

    def refresh(self):
        self.scene_panel.set_scene(self.model,
              self.region,
            self.plan if self.layers.currentIndex()==2 else None)
        if not self.scene_panel.is_3d:self.scene_panel.notice.hide()
        ready=self.model is not None and not self.busy
        self.coverage.setVisible(bool(self.plan and self.layers.currentIndex()==2))
        self.coverage.set_context(self.model,self.plan,ready=ready and self.layers.currentIndex()==2,busy=self.busy)
        self.quality.rescan.setEnabled(bool(ready and rescan_context(self.model)))
        self.quality.rescan.setToolTip('在当前任务下重新采集，历史方案保留。' if rescan_context(self.model) else '缺少可核验的任务信息；请新建采集。')
        self.quality_toggle.setEnabled(ready and self.region is not None)
        self.quality_toggle.setVisible(self.region is not None)
        self.quality.setVisible(self.region is not None)
        self.quality.rescan.setVisible(bool(rescan_context(self.model)))
        self.quality_toggle.setToolTip('此记录没有已保存的工作区域' if self.region is None else '')
        for index, available, reason in (
            (1, bool(self.region), '此记录没有已保存的工作区域。'),
            (2, bool(self.plan), '此记录没有可回显的已保存路线；请查看右侧记录说明。'),
        ):
            self.layers.model().item(index).setEnabled(available)
            self.layers.setItemData(index, '' if available else reason, Qt.ToolTipRole)
        self.layers.setEnabled(ready);self.fit.setEnabled(ready)
        self.play.setEnabled(bool(ready and self.plan and self.layers.currentIndex()==2 and self.canvas._coverage_image is None))
        self.play.setVisible(bool(self.plan))
        self.report.setEnabled(bool(ready and self.plan))
        self.report.setVisible(bool(self.plan))
        self.simulate.setEnabled(bool(ready and self.source and self.plan and self.plan.get('surface_trajectory')))
        self.simulate.setToolTip('带入当前已保存版本，再次核验并显示任务、来源与版本；不自动运行。'
                                 if self.plan and self.plan.get('surface_trajectory') else
                                 '此记录没有可核验的已保存三维表面路线；请先生成并保存方案。')
        self.capture.setEnabled(bool(ready and getattr(self.model,'capture_source',None)))
        self.capture.setVisible(bool(getattr(self.model,'capture_source',None)))
        self.capture.setToolTip('仅打开已核验为同一来源的采集模型。')
        self.edit.setEnabled(ready and (not self._needs_confirmation() or bool(getattr(self.model,'capture_source',None))))
        self.edit.setToolTip('关联采集不可用，无法确认最终模型。' if ready and self._needs_confirmation()
                            and not getattr(self.model,'capture_source',None) else '修改后保存为新版本，原版本保留。')
        self.source_toggle.setEnabled(ready)
        self.edit.setText('检查并确认最终模型' if self._needs_confirmation() else
                          '基于此版本修改' if self.region else '开始工作区域规划')
        self.cancel.setVisible(self.busy)
        self.cancel.setEnabled(self.busy and not self.runner.cancelling)
        self.cancel.setText('正在取消…' if self.runner.cancelling else '取消读取')
        self.heading.setText(STEPS[self._workflow_view-1]+' · 历史只读回看' if ready else '历史结果 · 只读回看')
        model_check = self.model is not None and self._workflow_view == 2
        # Model, selection and route share the same workflow stage. Keep
        # their local selector available even when the model layer is shown.
        self.layer_toolbar.setVisible(self.model is not None)
        self.layers.setVisible(True)
        self.fit.setVisible(True)
        self.inspector_scroll.setVisible(not model_check)
        self.workflow.set_step(self._workflow_view if ready else None,
            '历史结果只读回看；流程位置不代表此前步骤已通过质量检查。')

    def _capture(self):
        if not self.busy and self.model and getattr(self.model,'capture_source',None):
            self.capture_requested.emit(str(self.model.capture_source))

    def _edit(self):
        if self._needs_confirmation():
            self._capture()
            return
        if self.source and not self.busy:self.edit_requested.emit(str(self.source))

    def _simulate(self):
        if self.source and not self.busy and self.plan and self.plan.get('surface_trajectory'):
            self.simulation_requested.emit(str(self.source))

    def _needs_confirmation(self):
        completion=getattr(self.model,'acquisition_completion',None)
        return bool(completion is not None and completion.get('confirmed_for_planning') is False)

    def show_report(self):
        if not self.plan or self.busy:return
        self.runner.start(lambda cancel,progress:report_text(self.plan,self.model.mesh,self.region),
                          lambda text:show_text(self,'保存版本的规划检查',text),self._status)
        self.refresh()

    def shutdown(self):
        self.canvas.timer.stop()
        if not self.quality.shutdown():return False
        if not self.runner.shutdown(200):return False
        self.scene_panel.shutdown()
        return True
