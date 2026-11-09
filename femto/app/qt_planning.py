"""Production Qt region → surface-route → archive workflow.

All selection, geometry, route validation and file formats use the same backend
as the original workbench. This module only owns presentation and task dispatch.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QSize
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFormLayout,
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QStackedWidget, QTabWidget, QVBoxLayout, QWidget, QFileDialog)

from femto.app.qt_common import TaskRunner, show_text, open_path, breakable_text
from femto.app.qt_icons import line_icon
from femto.app.qt_planning_canvas import PlanningCanvas
from femto.app.qt_scene_panel import ScenePanel
from femto.app.qt_region_quality import RegionQualityPanel
from femto.app.qt_coverage import CoveragePanel
from femto.app.region_quality import assess_selection, findings as quality_findings, rescan_context, require_geometry
from femto.app.qt_planning_io import clone_model, extract_region, load_source, masks_from_region
from femto.planning.treatment_head import head_parameters, PLACEHOLDER_DEFAULTS, _inputs
from femto.planning import planning_state, surface_workspace
from femto.paths import OUTPUTS_ROOT


def review_findings(plan, region):
    """Surface the saved trajectory's unresolved review items in the UI."""
    stats = plan.get('surface_trajectory', {}).get('statistics', {})
    support = region.get('selection_support', {}) if region else {}
    findings = []
    quality = plan.get('surface_trajectory', {}).get('gap_report', {}).get('quality_constraint')
    strict = plan.get('surface_trajectory', {}).get('gap_report', {}).get('surface_support')
    if quality:
        findings.append(('常规路线基线：' if strict else '')+f"质量与覆盖宽度约束排除 {quality['rejected_edges']} 段边"
                        + ('，原目标尚未完成核验' if plan.get('coverage_evidence') is None else '；覆盖见独立核验结果'))
        findings.append('参考残差不是实物误差上界；配准不确定性未知')
    elif plan:
        findings.append('此历史路线未经过当前质量与覆盖宽度联合约束；重新计算后才能应用新规则')
    unconfirmed = stats.get('geometry_unconfirmed_pixel_count',
                            support.get('geometry_unconfirmed_pixel_count'))
    if unconfirmed is None:
        findings.append('选区几何支持状态未知')
    elif unconfirmed:
        findings.append(f'{unconfirmed} 个选中像素缺少可信三维几何')
    if not quality and stats.get('coverage_status') is None:
        findings.append('几何覆盖状态未知')
    elif not quality and stats['coverage_status'] != 'complete':
        findings.append('实测网格覆盖未完整')
    if stats.get('unplannable_block_count'):
        findings.append(f"{stats['unplannable_block_count']} 个工作块无法规划")
    internal = sum(not gap.get('inter_block', False) for gap in plan.get('chain_breaks', []))
    if internal:
        findings.append(f'{internal} 处工作块内路线断开')
    if stats.get('block_count', 0) > 1:
        findings.append(f"{stats['block_count'] - 1} 处工作块间移动未规划")
    if plan.get('spacing_report', {}).get('status') == 'not_fully_verified':
        findings.append('曲面行距未完全证实')
    if plan:
        from femto.planning.target_surface import summary_text
        from femto.planning.coverage_workflow import summaries, summary_text as coverage_text
        findings.append(summary_text((plan.get('planning_context') or {}).get('target'))
                        if plan.get('coverage_evidence') is None else summaries(plan)[1])
        findings.append(coverage_text(plan.get('coverage_evidence')))
        if plan.get('surface_trajectory', {}).get('planner', {}).get('algorithm_version') in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
            from femto.planning.variable_width_route import summary_text as variable_text
            findings.append(variable_text(plan['surface_trajectory']))
        if strict:
            from femto.planning.strict_path import summary_text as strict_text
            findings.append(strict_text(plan['surface_trajectory']))
        if plan.get('surface_trajectory', {}).get('planner', {}).get('algorithm_version') in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1'):
            from femto.planning.coverage_refinement import summary_text as refinement_text
            findings.append(refinement_text(plan['surface_trajectory']))
    return findings


class HeadForm(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setting=False
        self.values_status='placeholder_not_device_parameters'
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(10)
        self.notice=QLabel();self.notice.setWordWrap(True);self.notice.setProperty('role','secondary');layout.addWidget(self.notice)
        self.notice.hide()
        self.mode=QComboBox()
        self.mode.addItem('直接指定行距','direct_row_spacing')
        self.mode.addItem('有效作用宽度 ×（1 − 重叠率）','from_effective_width')
        self.mode.addItem('固定作用宽度自动排布','fixed_width_auto')
        self.mode.setCurrentIndex(2)
        self.mode.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.mode.setMinimumContentsLength(12)
        self.layout_choice=QComboBox()
        self.layout_choice.addItem('常规表面排布',None)
        from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION
        from femto.planning.continuous_quality import IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING
        self.layout_choice.addItem('局部方向适配（试验）',OPTIMIZED_PATCH_VERSION)
        self.layout_choice.addItem('局部方向与连接优化（试验）',IMPROVED_ROUTING)
        self.layout_choice.addItem('局部方向与连接优化（试验新版）',SUPPORT_ROUTING)
        self.layout_choice.addItem('局部方向与绕行优化（试验）',BALANCED_ROUTING)
        self.layout_choice.addItem('局部方向与接缝顺接（试验）',SEAM_ROUTING)
        self.layout_choice.addItem('局部方向与接缝联合优化（试验）',SEAM_BALANCED_ROUTING)
        self.layout_choice.addItem('局部方向与减少绕行（试验）',PRESERVED_DETOUR_ROUTING)
        self.layout_choice.addItem('局部方向与短距换序（试验）',SHORT_DETOUR_ROUTING)
        self.layout_choice.addItem('局部方向与联合换序（试验）',JOINT_REORDER_ROUTING)
        self.layout_choice.addItem('局部方向与两步换序（试验）',BRIDGE_REORDER_ROUTING)
        self.layout_choice.addItem('局部方向与扫描段移动（试验）',RELOCATION_ROUTING)
        self.layout_choice.addItem('局部方向与深度换序（试验）',DEEP_RELOCATION_ROUTING)
        self.layout_choice.addItem('局部疏密与绕行优化（试验）',FLOOR_DETOUR_ROUTING)
        self.layout_choice.addItem('局部疏密与快速换序（试验）',UNIFIED_FLOOR_ROUTING)
        self.layout_choice.addItem('自适应分区与快速换序（试验）',ADAPTIVE_FLOOR_PATCH_VERSION)
        self.layout_choice.addItem('自适应分区与接缝协调（试验）',COORDINATED_FLOOR_PATCH_VERSION)
        self.layout_choice.addItem('自适应分区与统一换序预算（试验）',GLOBAL_FLOOR_ROUTING)
        self.layout_choice.addItem('贴面连接优化（试验）',REFINED_CONNECTOR_ROUTING)
        self.layout_choice.addItem('局部方向与折返识别（试验）',FOLDBACK_PATCH_VERSION)
        self.layout_choice.setAccessibleName('表面路线排布方式')
        self.layout_choice.setToolTip('试验选项可按局部表面和选区边界估计调整扫描行，连接优化受计算预算限制。结果仍需单独核验覆盖。')
        form=QFormLayout();form.setHorizontalSpacing(10);form.setVerticalSpacing(10)
        self.form=form
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.fields={}
        names=(('row_spacing_mm','指定行距 / mm'),('effective_treatment_width_mm','有效作用宽度 / mm'),
               ('overlap_percent','重叠率 / %'))
        for key,label in names:
            edit=QLineEdit(str(PLACEHOLDER_DEFAULTS[key]));edit.setMinimumWidth(92);edit.setMaximumWidth(132)
            edit.setAlignment(Qt.AlignRight);edit.setAccessibleName(label)
            self.fields[key]=edit;form.addRow(label,edit)
            edit.textChanged.connect(self._edit)
        layout.addLayout(form)
        self.fields['effective_treatment_width_mm'].setToolTip('始终参与覆盖带边界检查；与外壳直径不同。')
        self.fields['overlap_percent'].setToolTip('仅在按有效作用宽度计算行距时使用。')
        self.spacing=QLabel();self.spacing.setObjectName('routeMetric');layout.addWidget(self.spacing)
        self.extra_button=QPushButton('高级设置');self.extra_button.setCheckable(True)
        self.extra_button.setToolTip('行距模式与试验算法；治疗深度、外壳直径与避让余量仅记录。')
        self.extra_button.setObjectName('disclosureButton')
        layout.addWidget(self.extra_button)
        self.extra=QWidget();extra_form=QFormLayout(self.extra);extra_form.setContentsMargins(0,0,0,0)
        extra_form.setHorizontalSpacing(10);extra_form.setVerticalSpacing(10)
        extra_form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        extra_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        extra_form.addRow('行距模式',self.mode)
        extra_form.addRow('试验排布',self.layout_choice)
        self.extra_form = extra_form
        for key,label in (('treatment_depth_mm','治疗深度 / mm'),('head_diameter_mm','外壳直径 / mm'),('safety_margin_mm','外壳避让余量 / mm')):
            edit=QLineEdit(str(PLACEHOLDER_DEFAULTS[key]));edit.setMinimumWidth(92);edit.setMaximumWidth(132)
            edit.setAlignment(Qt.AlignRight);edit.setAccessibleName(label)
            self.fields[key]=edit;extra_form.addRow(label,edit);edit.textChanged.connect(self._edit)
        self.extra.hide();layout.addWidget(self.extra)
        self.extra_button.toggled.connect(self.extra.setVisible)
        self.extra_button.toggled.connect(lambda shown:self.extra_button.setText('收起高级设置' if shown else '高级设置'))
        self.mode.currentIndexChanged.connect(self._edit)
        self.layout_choice.currentIndexChanged.connect(self._layout_edit)
        self.error=QLabel();self.error.setWordWrap(True);self.error.setObjectName('inlineError');self.error.setProperty('tone','error');layout.addWidget(self.error)
        self._refresh()

    def record(self):
        try:values={key:float(edit.text()) for key,edit in self.fields.items()}
        except ValueError:raise ValueError('参数必须填写有效数字') from None
        if self.mode.currentData()!='direct_row_spacing':values['row_spacing_mm']=None
        return head_parameters(self.mode.currentData(),values_status=self.values_status,**values)

    def error_text(self,error):
        message=str(error)
        for key,field in self.fields.items():
            if message==f'Invalid {key}':
                return f'{field.accessibleName()}：请填写有效数字。'
            prefix=f'{key} must be within '
            if message.startswith(prefix):
                return f'{field.accessibleName()}：允许范围 {message[len(prefix):]}。'
        prefix='Computed row spacing '
        separator=' mm is outside the planner range '
        if message.startswith(prefix) and separator in message:
            value,limits=message[len(prefix):].split(separator,1)
            return f'计算行距 {value} mm 超出允许范围 {limits}；请调整有效作用宽度或重叠率。'
        return message

    def set_record(self,record):
        self._setting=True
        self.mode.setCurrentIndex(self.mode.findData(record['mode']))
        values=_inputs(record)
        values['row_spacing_mm']=record['row_spacing_mm']
        for key,edit in self.fields.items():edit.setText(f'{values[key]:g}')
        self.values_status=record['values_status']
        self._setting=False;self._refresh()

    def layout_algorithm(self):
        return self.layout_choice.currentData() if self.mode.currentData()=='fixed_width_auto' else None

    def set_layout_algorithm(self,value):
        index=self.layout_choice.findData(value)
        if index<0:raise ValueError('草稿或方案的局部分区版本无效')
        self._setting=True;self.layout_choice.setCurrentIndex(index);self._setting=False;self._refresh()

    def _layout_edit(self,*_):
        if self._setting:return
        self._refresh();self.changed.emit()

    def _edit(self,*_):
        if self._setting:return
        self.values_status='user_entered_unverified'
        self._refresh();self.changed.emit()

    def _refresh(self):
        automatic=self.mode.currentData()!='direct_row_spacing'
        self.extra_form.setRowVisible(self.layout_choice,self.mode.currentData()=='fixed_width_auto')
        self.form.setRowVisible(self.fields['row_spacing_mm'],not automatic)
        self.form.setRowVisible(self.fields['overlap_percent'],automatic)
        self.fields['row_spacing_mm'].setEnabled(not automatic)
        self.fields['overlap_percent'].setEnabled(automatic)
        self.notice.setText('研发占位值，非设备参数。' if self.values_status=='placeholder_not_device_parameters'
                            else '手动录入参数，尚未验证。')
        self.mode.setToolTip(self.notice.text())
        self.extra.setToolTip(self.extra_button.toolTip())
        self.spacing.setVisible(automatic)
        try:
            record=self.record()
            label='最大行距' if record['mode']=='fixed_width_auto' else '最终规划行距'
            self.spacing.setText(f"{label}  {record['row_spacing_mm']:g} mm")
            if hasattr(self,'error'):self.error.clear()
        except ValueError as error:
            self.spacing.setText('最终规划行距  —')
            if hasattr(self,'error'):self.error.setText('参数无效：'+self.error_text(error))
        self.error.setVisible(bool(self.error.text()))


class _StageTabs(QTabWidget):
    """Let the inspector size its active stage, not the tallest hidden form."""
    def sizeHint(self):
        page=self.currentWidget()
        if page is None:return super().sizeHint()
        return page.sizeHint()+QSize(4,0 if self.tabBar().isHidden() else self.tabBar().sizeHint().height()+4)

    def minimumSizeHint(self):
        page=self.currentWidget()
        if page is None:return super().minimumSizeHint()
        return page.minimumSizeHint()+QSize(4,0 if self.tabBar().isHidden() else self.tabBar().minimumSizeHint().height()+4)

    def heightForWidth(self,width):
        page=self.currentWidget()
        if page is None:return super().heightForWidth(width)
        height=page.heightForWidth(max(0,width-4))
        return (height if height>=0 else page.sizeHint().height())+(0 if self.tabBar().isHidden() else self.tabBar().sizeHint().height()+4)


class PlanningPage(QWidget):
    rescan_requested = Signal()
    back_requested = Signal()
    status_changed = Signal(str)
    saved = Signal(str)
    loaded = Signal(str)
    stage_changed = Signal(int)
    review_requested = Signal(str)
    new_capture_requested = Signal()
    history_requested = Signal()
    load_return_requested = Signal()
    load_started = Signal()
    _preparation_preview_ready = Signal(object, object, object)

    def __init__(self,parent=None):
        super().__init__(parent)
        self.model=None;self.region=None;self.plan=None
        self._input_generation=0
        self.source=None;self.archive=None;self.journal=None
        self.dirty=False;self.selection_dirty=False;self._installing=False
        self._new_source_coarse=True
        self._stage=0;self._error='';self._report=''
        self._save_warning=''
        self._loading_request=None
        self._loading_prepared=None
        self._loading_preview_token=None
        self._return_after_load=False
        self.runner=TaskRunner(self)
        self.runner.progress.connect(self._status)
        self.runner.finished.connect(self.refresh)
        self.runner.cancel_requested.connect(self._cancellation_requested)
        self._build()
        self._preparation_preview_ready.connect(self._show_preparation_preview, Qt.QueuedConnection)
        self.loaded.connect(self._complete_preparation)
        self.runner.finished.connect(self._preparation_finished)
        self.parameters_button.toggled.connect(self.refresh)
        self.route_details_toggle.toggled.connect(self.refresh)
        self.canvas.timer.timeout.connect(self._refresh_playback)
        self.draft_timer=QTimer(self);self.draft_timer.setSingleShot(True)
        self.draft_timer.setInterval(500);self.draft_timer.timeout.connect(self.checkpoint)
        self.draft_status_timer=QTimer(self)
        self.draft_status_timer.setInterval(250)
        self.draft_status_timer.timeout.connect(self._refresh_draft_status)
        self.draft_status_timer.start()
        self._refresh_draft_status()
        self.refresh()

    @property
    def busy(self):return self.runner.busy

    @property
    def current_source(self):return self.source

    @property
    def loading_source(self):
        return self._loading_request['path'] if self._loading_request is not None else None

    @property
    def fixed_selection(self):
        """Region-first captures edit their reconstruction mask in step 02."""
        return bool(self.model is not None and
                    (self.model.store.local_context or {}).get('reconstruction_scope'))

    def _build(self) -> None:
        from femto.app.qt_workflow import WorkflowProgress
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        self.workflow=WorkflowProgress();layout.addWidget(self.workflow)
        self._build_planning_header(layout)
        self.body_stack=QStackedWidget()
        self.planning_body=QWidget()
        body=QHBoxLayout(self.planning_body);body.setContentsMargins(16,0,0,8);body.setSpacing(12)
        self.canvas=PlanningCanvas();self.canvas.changed.connect(self._selection_changed)
        self.scene_panel=ScenePanel(self.canvas);body.addWidget(self.scene_panel,1)
        self.scene_panel.mode.setToolTip('参考图可编辑区域；三维检查铺满工作区，拖动平移、滚轮缩放。切回参考图恢复操作面板。')
        self.scene_panel.mode_changed.connect(self._view_changed)
        self.scene_panel.retry.clicked.connect(self.refresh)
        self.scene_panel.reset.clicked.connect(self.refresh)
        scroll=QScrollArea();scroll.setObjectName('planningInspector');scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame);scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320);scroll.setMaximumWidth(360)
        panel=QWidget();panel.setObjectName('inspectorPanel')
        panel_layout=QVBoxLayout(panel);panel_layout.setContentsMargins(12,10,12,10);panel_layout.setSpacing(8)
        self.stage_panel_heading=QLabel('选区工具');self.stage_panel_heading.setObjectName('regionQualityHeading')
        panel_layout.addWidget(self.stage_panel_heading)
        self.tabs=_StageTabs();self.tabs.setObjectName('workflowTabs')
        # Wrapped review findings need their height for the actual sidebar width.
        # Maximum caps the tab at its unwrapped size hint and clips the summary.
        self.tabs.setSizePolicy(QSizePolicy.Preferred,QSizePolicy.Preferred)
        self.tabs.setDocumentMode(True);self.tabs.tabBar().setExpanding(True);self.tabs.tabBar().setUsesScrollButtons(False)
        self.tabs.tabBar().hide()
        panel_layout.addWidget(self.tabs)
        self.pages=[]
        for name in ('工作区域','路线参数','核对保存'):
            page=QWidget();self.tabs.addTab(page,name);self.pages.append(page)
        for index,name in enumerate(('工作区域','蛇形路线','核对保存')):
            self.tabs.setTabToolTip(index,f'全程第 {index+4} 步：{name}')
        region_layout = self._build_region_controls()
        self._build_route_controls()
        self._build_review_controls()
        self._build_planning_evidence(panel_layout, region_layout)
        self.inspector_scroll=scroll
        scroll.setWidget(panel);body.addWidget(scroll)
        from femto.app.qt_planning_loading import PlanningPreparation
        self.preparation=PlanningPreparation()
        self.preparation.cancel_requested.connect(self._return_from_preparation)
        self.preparation.retry_requested.connect(self._retry_preparation)
        self.body_stack.addWidget(self.planning_body)
        self.body_stack.addWidget(self.preparation)
        layout.addWidget(self.body_stack,1)
        self._build_planning_actions(layout)
        for button,icon in ((self.undo_button,'undo'),(self.redo_button,'redo'),
                (self.clear_button,'trash'),(self.fit_button,'refresh'),
                (self.source_toggle,'info'),(self.previous,'undo')):
            button.setIcon(line_icon(icon));button.setIconSize(QSize(18,18))
        for button in (self.undo_button,self.redo_button,self.clear_button,self.fit_button):
            button.setProperty('regionControl',True)

    def _build_planning_header(self, layout: QVBoxLayout) -> None:
        title_bar=QFrame();title_bar.setObjectName('titleBar')
        header=QVBoxLayout(title_bar);header.setContentsMargins(20,8,20,8);header.setSpacing(6)
        top=QHBoxLayout();top.setSpacing(12);header.addLayout(top)
        title_text=QVBoxLayout();title_text.setSpacing(3)
        self.title=QLabel('工作区域规划');self.title.setObjectName('planningTitle')
        title_text.addWidget(self.title)
        self.subtitle=QLabel('扫描范围决定采集哪里；现在涂抹选择需要规划的工作区域，擦除排除。')
        self.subtitle.setObjectName('pageSubtitle');self.subtitle.setWordWrap(True)
        top.addLayout(title_text,1)
        self.source_label=QLabel('尚未加载模型 · 未绑定病例')
        self.source_label.setProperty('role','secondary');self.source_label.setWordWrap(True)
        self.source_label.setTextFormat(Qt.PlainText)
        self.task_context_label=QLabel('尚未加载采集数据')
        self.task_context_label.setWordWrap(True)
        self.task_context_label.setProperty('role','secondary')
        self.task_context_label.setTextFormat(Qt.PlainText)
        title_text.addWidget(self.task_context_label)
        self.source_toggle=QPushButton('来源与操作');self.source_toggle.setCheckable(True)
        self.source_toggle.setObjectName('disclosureButton');top.addWidget(self.source_toggle)
        self.source_details=QWidget();source_layout=QVBoxLayout(self.source_details)
        source_layout.setContentsMargins(0,0,0,0);source_layout.setSpacing(6)
        source_layout.addWidget(self.source_label);source_layout.addWidget(self.subtitle)
        self.step_heading=QLabel('03 / 04  ·  模型与路线');self.step_heading.setProperty('role','secondary')
        source_layout.addWidget(self.step_heading)
        source_actions=QHBoxLayout();source_layout.addLayout(source_actions)
        self.open_button=QPushButton('打开区域方案');self.open_button.clicked.connect(self.choose_source);source_actions.addWidget(self.open_button)
        self.open_button.setObjectName('quietButton')
        self.return_button=QPushButton('返回采集 / 补采');self.return_button.clicked.connect(self.return_capture);source_actions.addWidget(self.return_button)
        self.return_button.setObjectName('quietButton')
        source_actions.addStretch();header.addWidget(self.source_details)
        self.source_details.hide();self.source_toggle.toggled.connect(self.source_details.setVisible)
        self.source_toggle.toggled.connect(lambda shown:self.source_toggle.setText('收起来源与操作' if shown else '来源与操作'))
        layout.addWidget(title_bar)

    def _build_region_controls(self) -> QVBoxLayout:
        region_layout=QVBoxLayout(self.pages[0]);region_layout.setContentsMargins(0,0,0,0);region_layout.setSpacing(4)
        self.resolution_label=QLabel('标准规划参考');self.resolution_label.setProperty('role','secondary');self.resolution_label.setWordWrap(True)
        self.resolution_label.setMinimumWidth(0);self.resolution_label.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        self.source_details.layout().addWidget(self.resolution_label)
        resolution_row=QHBoxLayout();resolution_row.setSpacing(8)
        self.reduce_model_button=QPushButton('创建简化规划参考')
        self.reduce_model_button.setProperty('regionControl',True)
        self.reduce_model_button.setToolTip('在选区前创建简化规划参考，保留原三维网格和原始数据；已有标记或路线时不可切换。')
        self.reduce_model_button.clicked.connect(self.create_reduced_planning_model)
        self.planning_mode=QComboBox();self.planning_mode.addItems(('粗分区规划（快速）','标准规划参考'))
        self.planning_mode.setStyleSheet('QComboBox { min-height: 34px; padding: 0 8px; font-size: 13px; }')
        self.planning_mode.setAccessibleName('规划参考模式')
        self.planning_mode.setToolTip('粗分区参考采用8像素格，沿线按选区大小自动取点；完整模型保留。标准模式按原图逐像素处理。已有方案和草稿保持原模式。')
        self.planning_mode.currentIndexChanged.connect(self._planning_mode_changed)
        resolution_row.addWidget(self.planning_mode,1)
        resolution_row.addWidget(self.reduce_model_button)
        self.new_coarse_button=QPushButton('新建粗分区版本')
        self.new_coarse_button.setProperty('regionControl',True)
        self.new_coarse_button.setToolTip('保留当前方案与草稿，创建独立粗分区参考；需要重新选区和生成路线。')
        self.new_coarse_button.clicked.connect(self.create_coarse_planning_version)
        resolution_row.addWidget(self.new_coarse_button)
        self.resolution_toggle=QPushButton('高级规划设置');self.resolution_toggle.setCheckable(True)
        region_layout.addWidget(self.resolution_toggle)
        self.resolution_options=QWidget();resolution_options=QVBoxLayout(self.resolution_options)
        resolution_options.setContentsMargins(0,0,0,0);resolution_options.addLayout(resolution_row)
        self.resolution_options.hide();self.resolution_toggle.toggled.connect(self.resolution_options.setVisible)
        region_layout.addWidget(self.resolution_options)
        self.curvature_model_button=QPushButton('新建自适应取点版本（试验）')
        self.curvature_model_button.setToolTip('保留当前方案与草稿，新建独立参考后重新选区。平缓段取点较疏，弯折与行端加密；行距仍由治疗头参数决定。')
        self.curvature_model_button.clicked.connect(self.create_curvature_planning_version)
        resolution_options.addWidget(self.curvature_model_button)
        self.tools=QFrame();self.tools.setObjectName('regionToolCard')
        tool_layout=QVBoxLayout(self.tools);tool_layout.setContentsMargins(8,4,8,4);tool_layout.setSpacing(4)
        row=QHBoxLayout();self.paint_group=QButtonGroup(self)
        self.paint_buttons=[]
        for index,text in enumerate(('涂抹工作区','擦除')):
            button=QPushButton(text);button.setCheckable(True);button.setObjectName('toolButton');self.paint_group.addButton(button,index)
            button.setIcon(line_icon('brush' if index==0 else 'eraser'));button.setIconSize(QSize(20,20))
            row.addWidget(button);self.paint_buttons.append(button)
        self.paint_buttons[0].setChecked(True);self.paint_group.idClicked.connect(self._paint_mode)
        tool_layout.addLayout(row)
        brush_row=QHBoxLayout();brush_row.addWidget(QLabel('笔刷大小'));brush_row.addStretch()
        self.brush_label=QLabel('40 px');brush_row.addWidget(self.brush_label);tool_layout.addLayout(brush_row)
        self.brush_label.setObjectName('brushValue')
        self.brush=QSlider(Qt.Horizontal);self.brush.setRange(8,160);self.brush.setValue(40)
        self.brush.setObjectName('regionBrushSlider')
        self.brush.setAccessibleName('笔刷大小，屏幕像素');self.brush.setToolTip('屏幕像素；缩放只改变观察，不改变已保存的选区。')
        self.brush.valueChanged.connect(self._brush);tool_layout.addWidget(self.brush)
        region_layout.addWidget(self.tools)
        self.support=QCheckBox('显示几何检查');self.support.toggled.connect(self._support)
        self.support.setToolTip('检查颜色不改变标记；绿为网格与深度一致，紫为待检查，橙为缺失或冲突。')
        self.support.setChecked(True)
        row=QHBoxLayout();self.undo_button=QPushButton('撤销');self.undo_button.clicked.connect(self.canvas.undo);row.addWidget(self.undo_button)
        self.redo_button=QPushButton('重做');self.redo_button.clicked.connect(self.canvas.redo);row.addWidget(self.redo_button)
        self.undo_button.setToolTip('撤销上一步选区操作（Ctrl+Z）')
        self.redo_button.setToolTip('恢复刚撤销的一步操作（Ctrl+Shift+Z），不会清空选区。')
        region_layout.addLayout(row)
        region_layout.addWidget(self.support)
        self.support_hint=QLabel('');self.support_hint.setWordWrap(True)
        self.support_hint.setProperty('role','secondary');self.scene_panel.layout().addWidget(self.support_hint)
        self.more_region=QWidget();more_layout=QVBoxLayout(self.more_region)
        more_layout.setContentsMargins(0,0,0,0);more_layout.setSpacing(8)
        row=QHBoxLayout()
        self.clear_button=QPushButton('清空工作区域');self.clear_button.clicked.connect(self.canvas.clear_work);row.addWidget(self.clear_button)
        self.fit_button=QPushButton('复位画面');self.fit_button.clicked.connect(self.canvas.reset_view);row.addWidget(self.fit_button);more_layout.addLayout(row)
        self.undo_shortcut=QShortcut(QKeySequence('Ctrl+Z'),self.canvas)
        self.undo_shortcut.setContext(Qt.WidgetWithChildrenShortcut);self.undo_shortcut.activated.connect(self.canvas.undo)
        self.redo_shortcut=QShortcut(QKeySequence('Ctrl+Shift+Z'),self.canvas)
        self.redo_shortcut.setContext(Qt.WidgetWithChildrenShortcut);self.redo_shortcut.activated.connect(self.canvas.redo)
        # Metadata remains part of saved projects and drafts, but stage 04
        # concentrates on selection rather than asking for naming fields.
        self.metadata=QWidget(self.pages[0]);metadata=QVBoxLayout(self.metadata);metadata.setContentsMargins(0,0,0,0)
        self.name=QLineEdit('工作区域 01');self.name.setPlaceholderText('区域名称');self.name.setAccessibleName('区域名称');metadata.addWidget(self.name)
        self.note=QLineEdit();self.note.setPlaceholderText('备注（可选）');self.note.setAccessibleName('区域备注');metadata.addWidget(self.note)
        self.metadata.hide()
        self.name.textEdited.connect(self._metadata_changed);self.note.textEdited.connect(self._metadata_changed)
        self.region_summary=QLabel('按住左键标记；滚轮缩放，右键平移。',self.pages[0]);self.region_summary.setWordWrap(True);self.region_summary.setProperty('role','secondary')
        self.region_summary.hide()
        self.canvas.setToolTip('左键涂抹或擦除 · 滚轮缩放 · 右键平移；可选多个分离区域或保留内部留空。')
        self.legacy_exclusion_notice=QLabel('红色为历史方案保留的禁区，仍从工作区域中扣除；涂抹、擦除和清空工作区域不会移除它。')
        self.legacy_exclusion_notice.setWordWrap(True);self.legacy_exclusion_notice.setProperty('role','secondary')
        region_layout.addWidget(self.legacy_exclusion_notice);region_layout.addStretch()
        return region_layout

    def _build_route_controls(self) -> None:
        route_layout=QVBoxLayout(self.pages[1]);route_layout.setContentsMargins(0,8,0,0);route_layout.setSpacing(8)
        self.route_summary=QLabel();self.route_summary.setWordWrap(True);self.route_summary.setObjectName('routeMetric');route_layout.addWidget(self.route_summary)
        self.parameter_summary=QLabel();self.parameter_summary.setWordWrap(True);self.parameter_summary.setObjectName('routeParameterSummary');route_layout.addWidget(self.parameter_summary)
        self.parameters_button=QPushButton('收起路线参数');self.parameters_button.setCheckable(True);self.parameters_button.setChecked(True)
        route_layout.addWidget(self.parameters_button)
        self.parameters_button.setIcon(line_icon('sliders'));self.parameters_button.setProperty('regionControl',True)
        self.head=HeadForm();self.head.changed.connect(self._parameters_changed);route_layout.addWidget(self.head)
        self.head.extra_button.setIcon(line_icon('sliders'))
        self.parameters_button.toggled.connect(self.head.setVisible)
        self.parameters_button.toggled.connect(lambda shown:self.parameters_button.setText('收起路线参数' if shown else '编辑路线参数'))
        self.play_button=QPushButton('播放路线');self.play_button.clicked.connect(self._play_reference)
        self.play_button.setToolTip('在固定参考图上播放同一条已生成路线；播放不代表设备执行。')
        self.scene_panel.view_controls.insertWidget(1,self.play_button)
        self.band_toggle=QCheckBox('名义宽度示意')
        self.band_toggle.setToolTip('青色为三维扫描段的切平面宽度示意，不是已确认覆盖；曲面可能有接缝。覆盖核验图开启时隐藏示意，保留黄色中心线。')
        self.band_toggle.toggled.connect(self.canvas.set_nominal_bands)
        self.scene_panel.view_controls.insertWidget(2,self.band_toggle)
        self.route_legend=QLabel('黄线：扫描与沿面连接');self.route_legend.setWordWrap(True)
        self.route_legend.setProperty('role','secondary');self.route_legend.setMinimumWidth(0)
        self.route_legend.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        self.route_legend.setToolTip('扫描与换行/绕行连接统一显示黄色实线；线条表示模型表面的几何路线，不表示抬升或开启治疗输出。'
            '分离工作块之间不补画连接，留空和未验证表面不自动桥接。实际持续贴面还需治疗头姿态、接触力及机器人运动验证。')
        self.scene_panel.view_controls.insertWidget(self.scene_panel.view_controls.count()-1,self.route_legend,1)
        self.recalculate=QPushButton('重新计算路线');self.recalculate.clicked.connect(self.generate)
        self.report_button=QPushButton('完整规划报告');self.report_button.clicked.connect(self.show_report)
        route_layout.addStretch()

    def _build_review_controls(self) -> None:
        save_layout=QVBoxLayout(self.pages[2]);save_layout.setContentsMargins(0,16,0,0);save_layout.setSpacing(12)
        section=QLabel('方案核对');section.setObjectName('sectionHeading');save_layout.addWidget(section)
        self.save_summary=QLabel();self.save_summary.setWordWrap(True);self.save_summary.setProperty('role','secondary');save_layout.addWidget(self.save_summary)
        self.save_summary.setMinimumWidth(0);self.save_summary.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        self.save_summary.setTextFormat(Qt.PlainText)
        self.reviewed=QCheckBox('已核对上述区域、断点与覆盖问题');self.reviewed.toggled.connect(self.refresh);save_layout.addWidget(self.reviewed)
        self.save_report_button=QPushButton('查看逐块规划检查');self.save_report_button.clicked.connect(self.show_report);save_layout.addWidget(self.save_report_button)
        notice=QLabel('当前方案为研发规划预览，不能直接执行治疗或机械臂运动。');notice.setWordWrap(True);notice.setProperty('role','muted');save_layout.addWidget(notice)
        self.folder_button=QPushButton('打开已保存方案');self.folder_button.clicked.connect(self.open_archive);save_layout.addWidget(self.folder_button)
        self.save_as_button=QPushButton('另存可编辑方案…');self.save_as_button.clicked.connect(self.save_as);save_layout.addWidget(self.save_as_button)
        self.saved_actions=QWidget();saved_actions=QVBoxLayout(self.saved_actions)
        self.saved_summary=QLabel();self.saved_summary.setWordWrap(True)
        self.saved_summary.setMinimumWidth(0)
        self.saved_summary.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        saved_actions.addWidget(self.saved_summary)
        self.view_saved_button=QPushButton('查看已保存结果');self.view_saved_button.clicked.connect(self.view_saved);saved_actions.addWidget(self.view_saved_button)
        fresh=QPushButton('新建采集');fresh.clicked.connect(self.new_capture_requested);saved_actions.addWidget(fresh)
        history=QPushButton('查看历史版本');history.clicked.connect(self.history_requested);saved_actions.addWidget(history)
        save_layout.addWidget(self.saved_actions)
        save_layout.addStretch()

    def _build_planning_evidence(self, panel_layout: QVBoxLayout, region_layout: QVBoxLayout) -> None:
        self.tabs.currentChanged.connect(self._stage_selected)
        self.coverage=CoveragePanel(compact=True)
        self.coverage.verify_requested.connect(self.verify_coverage)
        self.coverage.refine_requested.connect(self.refine_coverage)
        self.coverage.restore_requested.connect(self.restore_coverage_route)
        self.coverage.strict_requested.connect(lambda:self._strict_operation(False))
        self.coverage.ordinary_requested.connect(lambda:self._strict_operation(True))
        self.coverage.variable_requested.connect(self.edit_variable_model)
        self.coverage.variable_restore_requested.connect(lambda:self._variable_operation(None))
        self.coverage.layer_changed.connect(self._coverage_layer)
        self.coverage.focus_requested.connect(self._focus_coverage)
        self.coverage.task_requested.connect(self._coverage_task)
        self.coverage.tasks_export_requested.connect(self._export_coverage_tasks)
        self.route_attention=QLabel();self.route_attention.setWordWrap(True);self.route_attention.setObjectName('routeAttention')
        panel_layout.addWidget(self.route_attention)
        panel_layout.addWidget(self.coverage)
        self.route_details_toggle=QPushButton('展开检查与统计');self.route_details_toggle.setCheckable(True)
        self.route_details_toggle.setObjectName('disclosureButton');panel_layout.addWidget(self.route_details_toggle)
        self.warning=QLabel();self.warning.setWordWrap(True);self.warning.setObjectName('panelWarning');self.warning.setProperty('tone','warning')
        self.warning.setTextFormat(Qt.PlainText)
        self.warning_scroll=QScrollArea();self.warning_scroll.setObjectName('routeFindings')
        self.warning_scroll.setWidgetResizable(True);self.warning_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.warning_scroll.setWidget(self.warning);panel_layout.addWidget(self.warning_scroll)
        self.warning.setMinimumWidth(0);self.warning.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        self.route_stats=QLabel();self.route_stats.setWordWrap(True);self.route_stats.setProperty('role','secondary');panel_layout.addWidget(self.route_stats)
        self.quality=RegionQualityPanel(self.canvas,compact=True);panel_layout.addWidget(self.quality)
        self.route_extra=QWidget();extra=QHBoxLayout(self.route_extra);extra.setContentsMargins(0,0,0,0)
        self.report_button.setText('完整报告');self.report_button.setIcon(line_icon('document'))
        self.recalculate.setText('重新计算');self.recalculate.setIcon(line_icon('refresh'))
        for button in (self.report_button,self.recalculate):button.setProperty('regionControl',True)
        extra.addWidget(self.report_button);extra.addWidget(self.recalculate);panel_layout.addWidget(self.route_extra)
        region_layout.insertWidget(region_layout.count()-1,self.more_region)
        panel_layout.addStretch()
        self.quality.changed.connect(self._quality_changed)
        self.quality.focus_requested.connect(self._focus_quality)
        self.quality.rescan_requested.connect(self.rescan_requested)
        self.canvas.editing_started.connect(self.quality.invalidate)
        self.canvas.unchanged_stroke.connect(self.quality.request)
        self.draft_label=QLabel();self.draft_label.setWordWrap(True);panel_layout.addWidget(self.draft_label)
        self.retry_draft_button=QPushButton('重试保存草稿');self.retry_draft_button.clicked.connect(self.checkpoint);panel_layout.addWidget(self.retry_draft_button)
        self.draft_save_as=QPushButton('另存可编辑方案…');self.draft_save_as.clicked.connect(self.save_as);panel_layout.addWidget(self.draft_save_as)

    def _build_planning_actions(self, layout: QVBoxLayout) -> None:
        footer=QFrame();self.action_dock=footer;footer.setObjectName('actionDock');foot=QHBoxLayout(footer);foot.setContentsMargins(20,8,20,8);foot.setSpacing(10)
        status_area=QWidget();status_area.setObjectName('planningStatusArea')
        status_layout=QVBoxLayout(status_area);status_layout.setContentsMargins(0,0,0,0)
        self.status_label=QLabel('请加载三维模型');self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label);foot.addWidget(status_area,1)
        self.status_label.setObjectName('actionNotice');self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        self.cancel_button=QPushButton('取消任务');self.cancel_button.clicked.connect(self.runner.cancel);foot.addWidget(self.cancel_button)
        self.previous=QPushButton('上一步');self.previous.clicked.connect(self.go_back);foot.addWidget(self.previous)
        self.primary=QPushButton('下一步：蛇形路线规划');self.primary.setProperty('role','primary');self.primary.setMinimumHeight(52);self.primary.setMinimumWidth(260)
        self.primary.setAccessibleName('规划阶段主操作')
        self.primary.setShortcut('Ctrl+Return')
        self.primary.setToolTip('快捷键：Ctrl+Enter')
        self.primary.clicked.connect(self.primary_action);foot.addWidget(self.primary);layout.addWidget(footer)

    def _status(self,text):
        if self.loading_source is not None:
            from femto.app.qt_planning_loading import STAGE_MESSAGES, STAGES
            self.preparation.progress(str(text))
            if text in STAGE_MESSAGES:text=STAGES[STAGE_MESSAGES.index(text)]+'…'
        self._failure_details=''
        self.status_label.setToolTip('')
        self.status_label.setText(str(text));self.status_changed.emit(str(text))

    def _failure(self,error, *, diagnostics=None):
        from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
        from femto.app.qt_common import TaskResultDiscarded
        if self.loading_source is not None:
            self._error=str(error)
            from femto.dataset.prepared_review import close_prepared
            close_prepared(self._loading_prepared)
            self._loading_prepared=None
            if self.runner._cancel.is_set():
                self._return_after_load=True
                self.preparation.cancelling()
                self._status('正在返回，采集数据与选区保留。')
            else:
                self.preparation.failed(error)
                self._status('模型准备未完成，可重试或返回。')
                from femto.app.qt_failure_details import failure_text
                self._failure_details=failure_text(error,diagnostics,
                    retained_current='本次采集、已保存的选区及此前已载入的模型和参数。',
                    next_steps=('点击“重试”重新读取当前来源。','点击“返回”检查采集或原项目。'),
                    current_task=diagnostics is not None)
                self.status_label.setToolTip(self._failure_details)
            self.refresh()
            return
        self._error='' if isinstance(error,GeometryTaskCancelled) else str(error)
        self._status(str(error) if isinstance(error,TaskResultDiscarded) else
                     '任务已取消，当前区域与参数保留。' if not self._error else '操作失败：'+self._error)
        if self.quality.result is None:self.quality.request()
        self.refresh()
        from femto.app.qt_failure_details import failure_text,brief_failure_text
        details=failure_text(error,diagnostics,
            retained_current=('当前已载入的区域、参数与方案；失败任务没有替换当前方案。' if self.model is not None
                              else '本次失败没有安装新的模型或方案；尚未载入可用数据。'),
            next_steps=('检查上述原因后，再点击当前步骤的主操作重试。',
                        '需要返回修改时，使用“上一步”；有草稿时可使用既有草稿恢复入口。'),
            current_task=diagnostics is not None)
        self._status(breakable_text(brief_failure_text(error,diagnostics,
            has_model=self.model is not None,current_task=diagnostics is not None)))
        self._failure_details=details
        self.status_label.setToolTip(details)

    def _cancellation_requested(self):
        if self.runner.cancelling:
            if self.loading_source is not None:
                self._return_after_load=True
                self.preparation.cancelling()
                self._status('正在取消模型准备，结束后返回原页面。')
                self.refresh()
                return
            self._status('正在取消任务，请等待当前处理结束；保存结果以结束后的提示为准。')
            self.refresh()

    def _return_from_preparation(self):
        if self.busy:
            self.runner.cancel()
        else:
            self._dismiss_preparation()
            self._status('已返回，原有采集、选区与参数保留。')
            self.load_return_requested.emit()

    def _retry_preparation(self):
        if self._loading_request is not None and not self.busy:
            self.load_source(**self._loading_request)

    def _preparation_finished(self):
        if self.loading_source is not None and self._return_after_load:
            self._dismiss_preparation()
            self._status('准备已取消，原有采集、选区与参数保留。')
            self.load_return_requested.emit()

    def _complete_preparation(self, source):
        if self.loading_source is not None and Path(source).resolve()==self.loading_source:
            self._dismiss_preparation()

    def _show_preparation_preview(self, token, snapshot, assessment):
        if (token is not self._loading_preview_token or self.loading_source is None
                or not self.busy or self.runner._cancel.is_set()
                or self.preparation.state != 'running'):
            return
        if self._task_revision() != self._loading_preview_revision:return
        self.preparation.show_model(snapshot, assessment)

    def _dismiss_preparation(self):
        from femto.dataset.prepared_review import close_prepared
        close_prepared(self._loading_prepared)
        self._loading_prepared=None
        self._loading_preview_token=None
        self._loading_request=None
        self._return_after_load=False
        self._error=''
        self.preparation.stop()
        self.body_stack.setCurrentWidget(self.planning_body)
        self.planning_body.setEnabled(True)
        self.source_toggle.setEnabled(True)
        self.title.setText(('工作区域规划','三维蛇形路线规划','核对并保存规划')[self._stage])
        if self.model is not None:
            self._refresh_source_label()
            if self.quality.result is None:self.quality.request()
        else:self.task_context_label.setText('尚未加载采集数据')
        self.refresh()

    def _run(self,job,done):
        if self.busy:return False
        generation=self._input_generation
        self.canvas.finish_stroke();self.canvas.timer.stop();self._error=''
        # The caller already captured the job inputs. A completed stroke must
        # not bind an old region snapshot to the newly changed input revision.
        if generation!=self._input_generation:
            self._status('标记刚刚更新，请先确认工作区域后再继续。')
            self.refresh();return False
        revision=self._task_revision()
        def failed(error):
            self._failure(error,diagnostics=self.runner.last_diagnostics)
        started=self.runner.start(job,done,failed,revision=revision,current_revision=self._task_revision)
        self.refresh();return started

    def _task_revision(self):
        from femto.planning.planning_revision import capture, array_record
        if self.model is None:return (None,self._input_generation)
        try:head=self.head.record()
        except ValueError:head={key:value.text() for key,value in self.head.fields.items()}
        return (id(self.model),self._input_generation,capture(self.model,self.region,head)['sha256'],
                array_record(self.canvas.work)['sha256'],array_record(self.canvas.forbidden)['sha256'])

    def choose_source(self):
        path,_=QFileDialog.getOpenFileName(self,'打开区域工程或历史路线','','区域工程 / 路线 (*.zip *.json)')
        if path:self.load_source(path)

    def load_source(self,path, *, force=False,restore_draft=True,new_version=False,expected_revision=None,review_model=False,
                    preview=None,loading_context='',prepared=None):
        from femto.dataset.prepared_review import close_prepared
        if self.busy:
            close_prepared(prepared);return False
        path=Path(path).resolve()
        if not force and self.model is not None and path==self.source:
            close_prepared(prepared)
            self._status('已保留当前区域、参数与规划。');return True
        if self.dirty and self.loading_source is None:
            self.checkpoint()
            answer=QMessageBox.question(self,'切换规划项目','将确认当前草稿写入成功后打开另一个项目。是否继续？',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
            if answer!=QMessageBox.Yes:
                close_prepared(prepared);return False
        from femto.planning.planning_drafts import read
        from femto.app.qt_planning_resolution import journal_for,prepare_new_source
        from femto.app.qt_planning_io import needs_model_confirmation
        previous_journal=self.journal if self.dirty else None
        coarse_requested=self._new_source_coarse
        preview_token=object()
        from femto.app.qt_planning_loading import STAGE_MESSAGES
        def job(cancel,progress):
            progress(STAGE_MESSAGES[0])
            if previous_journal:
                from femto.annotation.fixed_rgb_geometry import GeometryBudget
                progress('正在确认当前草稿已保存…')
                previous_journal.wait_saved(GeometryBudget(cancel).check)
            options={'require_confirmed':False} if review_model else {}
            if prepared is not None:options['prepared']=prepared
            try:model=load_source(path,cancel,progress,**options)
            finally:close_prepared(prepared)
            from femto.geometry.local_region import model_selection
            selected = model_selection(model) if not model.store.regions else None
            if review_model and (selected is None or getattr(model,'capture_source',None) is None
                                 or getattr(model,'acquisition_completion',None) is None):
                raise ValueError('当前结果缺少可核验的局部重建选区或最终模型记录。')
            if selected is not None:
                # Keep the original painted intent, including missing pixels.
                # An existing selection must not be silently resampled/coarsened.
                model.pre_reconstruction_selection = selected
            if expected_revision is not None and getattr(model,'revision_id',None)!=expected_revision:
                raise ValueError('保存记录已变更，请重新读取并核对版本后再修改。')
            from femto.annotation.fixed_rgb_geometry import GeometryBudget
            pending=needs_model_confirmation(model)
            # Fresh, source-bound local captures share the TSDF planning
            # profile even if a previous imported model used standard mode.
            local_scope=(model.store.local_context or {}).get('reconstruction_scope') or {}
            prepare_surface=coarse_requested or bool(selected is not None and local_scope.get('input_provenance'))
            model.route_prepare_required=bool(pending and prepare_surface)
            progress(STAGE_MESSAGES[1])
            model=prepare_new_source(model,path,enabled=prepare_surface and not pending,restore_draft=restore_draft,
                check=GeometryBudget(cancel).check,progress=progress)
            selected=getattr(model,'pre_reconstruction_selection',selected)
            journal=journal_for(path,model);draft=None;warning=''
            if restore_draft and journal.path.is_file():
                try:draft=read(journal.path,model)
                except Exception as error:warning='草稿未恢复：'+str(error)
            region=model.store.regions[0] if model.store.regions else None
            work,forbidden=masks_from_region(region,model.rgb.shape[:2]) if region else (
                getattr(model,'pre_reconstruction_selection',np.zeros(model.rgb.shape[:2],np.uint8)),
                getattr(model,'pre_reconstruction_forbidden',np.zeros(model.rgb.shape[:2],np.uint8)))
            progress(STAGE_MESSAGES[2])
            assessment=assess_selection(model,work,forbidden,cancel)
            if selected is not None and not draft and assessment['status']=='checked' and assessment['supported']:
                # Owned display arrays cross the Qt signal; the mutable model
                # and its annotations remain private until the final commit.
                from femto.viewer.vtk_scene import scene_snapshot
                from femto.operation_diagnostics import record_operation
                GeometryBudget(cancel).check()
                try:
                    with record_operation('load.model_preview'):
                        snapshot=scene_snapshot(model)
                except (ValueError, RuntimeError, MemoryError):
                    snapshot=None
                GeometryBudget(cancel).check()
                if snapshot is not None:
                    self._preparation_preview_ready.emit(preview_token,snapshot,
                        {key:assessment[key] for key in ('ratio','selected','supported','pending')})
                # Reuse the exact pre-reconstruction intent; extraction runs in
                # the load worker instead of requiring a second selection click.
                model,region=extract_region(model,work,forbidden,'工作区域 01','',cancel)
                model.pre_reconstruction_selection=selected
            return model,journal,draft,warning,assessment
        def done(result):
            if self.runner._cancel.is_set():
                self._status('读取已取消，当前记录保留。'); self.quality.request();return
            model,journal,draft,warning,assessment=result
            self.install_model(model,path,journal,assessment=assessment,notify=False)
            if (getattr(model,'pre_reconstruction_selection',None) is not None
                    and self.region is not None and not draft):
                self.dirty=True;self.set_stage(1);self.canvas.fit_selection()
                self._status('请在此核对模型与缺口，设置作用宽度后点击“生成路线”，确认使用当前模型。'
                             if needs_model_confirmation(model) else '原选区已保留并完成几何检查。核对作用宽度后点击“生成路线”。')
            if new_version:
                self.dirty=True;self.archive=None
                self._status('正在编辑新版本；原保存文件保留，修改后请保存到新版本。')
                self.refresh()
            if draft and QMessageBox.question(self,'恢复标记草稿','发现与当前来源匹配的草稿，是否恢复区域和参数？\n恢复后需要重新计算路线。',QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)==QMessageBox.Yes:
                self.restore_draft(draft)
            if warning:self._status(warning)
            # Publish only after the destination stage and restored draft are
            # ready, so the shell never exposes an intermediate selection page.
            self.loaded.emit(str(self.source))
        self.quality.invalidate('正在切换来源，原检查结论已清除。')
        first_attempt=self.loading_source is None
        self._loading_prepared=prepared
        self._loading_preview_token=preview_token
        self._loading_preview_revision=self._task_revision()
        self._loading_request=dict(path=path,force=force,restore_draft=restore_draft,new_version=new_version,
            expected_revision=expected_revision,review_model=review_model,preview=preview,loading_context=loading_context)
        self._return_after_load=False
        self.preparation.start(preview)
        self.source_toggle.setChecked(False)
        self.body_stack.setCurrentWidget(self.preparation)
        if first_attempt:self.load_started.emit()
        self._status('正在准备模型，请稍候。')
        started=self._run(job,done)
        if not started:self._dismiss_preparation()
        return started

    def create_reduced_planning_model(self):
        from femto.app.qt_planning_resolution import start_creation
        try:return start_creation(self)
        except (ValueError, OSError) as error:
            self._failure(error);return False

    def create_coarse_planning_version(self):
        from femto.app.qt_planning_resolution import start_creation
        try:return start_creation(self,new_version=True)
        except (ValueError,OSError) as error:
            self._failure(error);return False

    def create_curvature_planning_version(self):
        from femto.app.qt_planning_resolution import start_creation,can_create
        from femto.planning.route_sampling import CURVATURE_POLICY
        try:return start_creation(self,new_version=not can_create(self),sampling_policy=CURVATURE_POLICY)
        except (ValueError,OSError) as error:
            self._failure(error);return False

    def _planning_mode_changed(self,index):
        if self._installing:return
        from femto.app.qt_planning_resolution import can_choose_mode,derivation,can_create
        if not can_choose_mode(self):
            self.planning_mode.blockSignals(True)
            self.planning_mode.setCurrentIndex(0 if derivation(self.model) else 1)
            self.planning_mode.blockSignals(False)
            return
        self._new_source_coarse=index==0
        if self.model is None:return
        if index==0 and can_create(self):
            self.create_reduced_planning_model()
        elif index==1 and derivation(self.model) and self.source is not None:
            # Enabled only on a blank reference. The actual source is re-read;
            # neither coarse measurements nor old selections are upsampled.
            self.load_source(self.source,force=True,restore_draft=True)

    def _refresh_source_label(self):
        source_name = next((parent.name for parent in self.source.parents
                            if parent.name.startswith('complete_demo_')),
                           self.source.parent.name if self.source.name.casefold() == 'reconstruction'
                           else f'{self.source.parent.name}/{self.source.name}' if self.source.is_file()
                           else self.source.name)
        self.source_label.setText(f'当前来源：{source_name}')
        self.source_label.setToolTip(str(self.source))
        task=getattr(self.model,'task_context',None) or {}
        if task:self.source_label.setText(f"任务：{task['name']} · 来源：{source_name}")
        self.source_label.setText(self.source_label.text()+'\n参考影像与模型来自当前记录 · 未绑定病例 · W0 / RDF / m')
        from femto.app.qt_planning_resolution import derivation, REDUCED_NOTICE
        if derivation(self.model):self.source_label.setText(self.source_label.text()+'\n'+REDUCED_NOTICE)
        self.task_context_label.setText(task.get('name') or '未关联任务')
        verification=getattr(self.model,'capture_source_verification',None)
        if verification:
            self.source_label.setText(self.source_label.text()+'\n'+verification['message'])
            for issue in verification['issues']:
                self.source_label.setText(self.source_label.text()+f"\n{issue['reason']}：{issue['source']}")
            if verification['status'] not in ('verified','not_recorded'):
                self.task_context_label.setText(self.task_context_label.text()+' · '+verification['message'])

    def install_model(self,model,source,journal=None,*,assessment=None,notify=True):
        self._input_generation+=1
        # Keep the operator's valid configuration across new captures within
        # this workstation session; saved plans below always restore their own.
        try:head=self.head.record()
        except ValueError:head=head_parameters('fixed_width_auto')
        saved_plan=getattr(model,'saved_plan',None)
        request=getattr(model,'planning_parameters',None)
        from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION
        from femto.planning.continuous_quality import IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING
        requested_layout=None
        if saved_plan and saved_plan.get('treatment_head'):
            head=surface_workspace.checked_head(saved_plan['treatment_head'])
        elif request and request.get('kind')==surface_workspace.KIND:
            head=surface_workspace.checked_head(request['values']['treatment_head'])
            requested_layout=request['values'].get('layout_algorithm')
            if requested_layout not in (None,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION,IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING) or (requested_layout is not None and head['mode']!='fixed_width_auto'):
                raise ValueError('保存的局部分区排布版本无效')
        self._installing=True
        self.model=model;self.source=Path(source).resolve();self.journal=journal
        self._gap_followup_request=None
        self._refresh_source_label()
        self.region=model.store.regions[0] if model.store.regions else None
        self.plan=getattr(model,'saved_plan',None)
        from femto.app.qt_planning_resolution import derivation
        self.planning_mode.blockSignals(True)
        self.planning_mode.setCurrentIndex(0 if derivation(model) else 1)
        self.planning_mode.blockSignals(False)
        self.archive=self.source.parent if self.source.is_file() else None
        self._error='';self._report=''
        self._save_warning=''
        self.dirty=False;self.selection_dirty=False
        self.canvas.set_model(model,*(masks_from_region(self.region,model.rgb.shape[:2]) if self.region else (None,None)))
        if self.region is None and getattr(model,'pre_reconstruction_selection',None) is not None:
            self.canvas.set_model(model,model.pre_reconstruction_selection)
            self.selection_dirty = True
            self.dirty = True
        if assessment is None:self.quality.request()
        else:
            self.quality.invalidate();self.quality.install(assessment)
        self.canvas.role='work';self.canvas.erase=False;self.paint_buttons[0].setChecked(True)
        if self.region:
            self.name.setText(self.region.get('name','工作区域 01'));self.note.setText(self.region.get('note',''))
        else:self.name.setText('工作区域 01');self.note.clear()
        self.head.set_record(head)
        layout=(((self.plan or {}).get('surface_trajectory') or {}).get('planner') or {}).get('parameters',{}).get('fixed_width_layout') or {}
        routing=(((self.plan or {}).get('surface_trajectory') or {}).get('planner') or {}).get('parameters',{}).get('continuous_routing')
        selected_layout=(REFINED_CONNECTOR_ROUTING if routing==REFINED_CONNECTOR_ROUTING else
                         GLOBAL_FLOOR_ROUTING if layout.get('request',{}).get('version')==COORDINATED_FLOOR_PATCH_VERSION and routing==GLOBAL_FLOOR_ROUTING else
                         COORDINATED_FLOOR_PATCH_VERSION if layout.get('request',{}).get('version')==COORDINATED_FLOOR_PATCH_VERSION and routing==UNIFIED_FLOOR_ROUTING else
                         ADAPTIVE_FLOOR_PATCH_VERSION if layout.get('request',{}).get('version')==ADAPTIVE_FLOOR_PATCH_VERSION and routing==UNIFIED_FLOOR_ROUTING else
                         UNIFIED_FLOOR_ROUTING if layout.get('request',{}).get('version')==FLOOR_PATCH_VERSION and routing==UNIFIED_FLOOR_ROUTING else
                         FLOOR_DETOUR_ROUTING if layout.get('request',{}).get('version')==FLOOR_PATCH_VERSION and routing==FLOOR_DETOUR_ROUTING else
                         FOLDBACK_PATCH_VERSION if layout.get('request',{}).get('version')==FOLDBACK_PATCH_VERSION and routing==SEAM_BALANCED_ROUTING else
                         routing if layout.get('request',{}).get('version')==OPTIMIZED_PATCH_VERSION and routing in (IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING)
                         else OPTIMIZED_PATCH_VERSION if layout.get('request',{}).get('version')==OPTIMIZED_PATCH_VERSION
                         else requested_layout if self.plan is None else None)
        self.head.set_layout_algorithm(selected_layout)
        self.parameters_button.setChecked(self.plan is None)
        self.canvas.set_plan(self.plan)
        self.reviewed.setChecked(False)
        self._installing=False
        # Imported projects retain region review. Region-first captures already
        # selected their reconstruction mask and enter route planning directly.
        self.set_stage(0)
        if self.fixed_selection and self.region and not self.selection_dirty:
            self.scene_panel.mode.setCurrentIndex(1)
        QTimer.singleShot(0,lambda:self.inspector_scroll.verticalScrollBar().setValue(0))
        self._status(getattr(model,'plan_warning',None) or ('已载入选区模型，核对路线参数后生成路线；修改选区请返回第 02 步并重新建模。' if self.fixed_selection else '请先核对历史区域及其三维支持，再进入路线。' if self.region else '请涂抹工作区域；不需要的部分可擦除或留空。'))
        if (getattr(model,'task_context',None) or {}).get('gap_followup'):
            self._status('此结果关联之前的缺口处理请求。请核对新模型并重新选区、生成路线、核验覆盖；关联记录不代表旧缺口已解决。')
        self.refresh()
        if notify:self.loaded.emit(str(self.source))

    def _paint_mode(self,index):
        self.scene_panel.show_reference()
        self.canvas.role='work'
        self.canvas.erase=bool(index);self.canvas.update()

    def _view_changed(self, is_3d):
        self.refresh()
        if is_3d and self.selection_dirty and not self.busy and self.canvas.work.any():
            self.commit_region(advance=False)
        elif self._stage == 0:
            self._status('固定朝向三维检查：拖动平移，滚轮缩放；修改区域请返回参考图。' if is_3d else
                         '请在参考图上涂抹或擦除；右键平移，滚轮缩放。')

    def _brush(self,value):self.canvas.brush_size=value;self.brush_label.setText(f'{value} px');self.canvas.update()
    def _support(self,value):
        self.canvas.support=bool(value and self._stage==0);self.canvas.update()
        if hasattr(self,'support_hint'):self.refresh()

    def _selection_changed(self):
        if self._installing:return
        self._input_generation+=1
        self.selection_dirty=True;self.dirty=True;self._invalidate()
        self.quality.request()
        self._status('标记已更新，下一步将检查三维几何。');self.draft_timer.start();self.refresh()

    def _invalidate(self):
        self.plan=None;self.archive=None;self._report='';self._error=''
        self.canvas.set_plan(None);self.reviewed.setChecked(False)
        self.canvas.set_coverage_overlay(None);self.scene_panel.set_coverage(None)

    def _parameters_changed(self):
        if self._installing:return
        self._input_generation+=1
        self.dirty=True;self._report='';self._error=''
        self.reviewed.setChecked(False);self.canvas.timer.stop()
        if self.plan:
            saved_head=self.plan.get('treatment_head')
            try:current=self.head.record()
            except ValueError:current=None
            if saved_head and current:
                old_inputs,new_inputs=_inputs(saved_head),_inputs(current)
                old_inputs.pop('values_status',None);new_inputs.pop('values_status',None)
                if old_inputs==new_inputs and saved_head['mode']==current['mode']:
                    self.head.set_record(saved_head)
        self._status('参数无效，请检查输入。' if self.head.error.text() else
                     '参数已修改，请重新计算路线。' if self.plan and not self.usable() else
                     '参数与已有路线一致，请重新核对。' if self.plan else
                     '参数已更新，可计算路线。')
        self.draft_timer.start();self.refresh()

    def _metadata_changed(self):
        if self._installing or self.model is None:return
        if self.region:
            index=next(i for i,r in enumerate(self.model.store.regions) if r['annotation_id']==self.region['annotation_id'])
            self.region=self.model.store.update_annotation('region',index,name=self.name.text(),note=self.note.text())
        self.dirty=True;self.archive=None;self.draft_timer.start();self.refresh()

    def request(self):
        return dict(kind=surface_workspace.KIND,values={'treatment_head':self.head.record()})

    def _layout_choice_current(self):
        if not self.plan:return False
        from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION
        from femto.planning.continuous_quality import IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING
        params=self.plan.get('surface_trajectory',{}).get('planner',{}).get('parameters',{})
        layout=params.get('fixed_width_layout') or {}
        saved=(REFINED_CONNECTOR_ROUTING if params.get('continuous_routing')==REFINED_CONNECTOR_ROUTING else
               GLOBAL_FLOOR_ROUTING if layout.get('request',{}).get('version')==COORDINATED_FLOOR_PATCH_VERSION and params.get('continuous_routing')==GLOBAL_FLOOR_ROUTING else
               COORDINATED_FLOOR_PATCH_VERSION if layout.get('request',{}).get('version')==COORDINATED_FLOOR_PATCH_VERSION and params.get('continuous_routing')==UNIFIED_FLOOR_ROUTING else
               ADAPTIVE_FLOOR_PATCH_VERSION if layout.get('request',{}).get('version')==ADAPTIVE_FLOOR_PATCH_VERSION and params.get('continuous_routing')==UNIFIED_FLOOR_ROUTING else
               UNIFIED_FLOOR_ROUTING if layout.get('request',{}).get('version')==FLOOR_PATCH_VERSION and params.get('continuous_routing')==UNIFIED_FLOOR_ROUTING else
               FLOOR_DETOUR_ROUTING if layout.get('request',{}).get('version')==FLOOR_PATCH_VERSION and params.get('continuous_routing')==FLOOR_DETOUR_ROUTING else
               FOLDBACK_PATCH_VERSION if layout.get('request',{}).get('version')==FOLDBACK_PATCH_VERSION and params.get('continuous_routing')==SEAM_BALANCED_ROUTING else
               params.get('continuous_routing') if layout.get('request',{}).get('version')==OPTIMIZED_PATCH_VERSION and params.get('continuous_routing') in (IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING)
               else OPTIMIZED_PATCH_VERSION if layout.get('request',{}).get('version')==OPTIMIZED_PATCH_VERSION else None)
        return saved==self.head.layout_algorithm()

    def usable(self):
        try:request=self.request()
        except ValueError:return False
        revision=None
        if self.plan and not self._layout_choice_current():return False
        if self.model and self.plan and self.plan.get('planning_context') is not None:
            from femto.planning.planning_revision import capture_for
            try:revision=capture_for(self.model,self.region,request['values']['treatment_head'],self.plan['planning_context']['revision'])
            except (ValueError,TypeError,AttributeError):return False
        return planning_state.state(self.region,self.plan,busy=self.busy,draft=self.selection_dirty,
            request=request,geometry_digest=self.model.store.digest if self.model else None,
            computation_revision=revision)['can_use']

    def _quality_changed(self):
        if not self._installing:
            self.reviewed.setChecked(False)
            self.refresh()

    def _focus_quality(self):
        self.set_stage(0);self.scene_panel.show_reference();self.support.setChecked(True)
        self.canvas.support=True;self.canvas.update()

    def current_findings(self):
        return review_findings(self.plan,self.region)+quality_findings(self.quality.result)

    def refresh(self,*_):
        from femto.app.qt_planning_resolution import can_create, profile_text, derivation, REDUCED_NOTICE
        self.resolution_label.setText(profile_text(self.model) if self.model is not None else '尚未加载规划参考')
        self.resolution_label.setToolTip(REDUCED_NOTICE if derivation(self.model) else '')
        self.reduce_model_button.setEnabled(can_create(self))
        from femto.app.qt_planning_resolution import can_choose_mode
        blank=can_choose_mode(self)
        self.planning_mode.setEnabled(blank)
        self.reduce_model_button.setVisible(self.model is not None and not derivation(self.model) and blank)
        self.new_coarse_button.setEnabled(self.model is not None and not self.busy and not derivation(self.model))
        self.new_coarse_button.setVisible(self.model is not None and not derivation(self.model) and not blank)
        self.curvature_model_button.setVisible(self.model is not None and not derivation(self.model))
        self.curvature_model_button.setEnabled(self.model is not None and not self.busy and not derivation(self.model))
        self.scene_panel.set_scene(self.model,
            self.region if not self.selection_dirty else None,
            self.plan if self._stage>0 and self.usable() else None)
        loaded=self.model is not None;busy=self.busy
        self.quality.rescan.setEnabled(bool(loaded and not busy and rescan_context(self.model)))
        self.quality.rescan.setVisible(bool(loaded and rescan_context(self.model)))
        self.quality.rescan.setToolTip('建立独立采集，旧模型、选区和路线保留。' if rescan_context(self.model) else '缺少可核验的任务信息；请新建采集。')
        self.legacy_exclusion_notice.setVisible(loaded and bool(self.canvas.forbidden.any()))
        reference=not self.scene_panel.is_3d
        self.canvas.editable=loaded and not busy and self._stage==0 and reference and not self.fixed_selection
        inspection=not reference and self.scene_panel.stack.currentWidget() is self.scene_panel.viewport
        self.inspector_scroll.setVisible(not inspection or self._stage>0 or self.fixed_selection)
        self.action_dock.setVisible(True)
        self.previous.setVisible(True)
        self.primary.setVisible(True)
        self.scene_panel.notice.setVisible(not reference and not inspection)
        self.more_region.setVisible(self._stage==0 and reference and not self.fixed_selection)
        for control in (self.tools,self.undo_button,self.redo_button,self.support,self.clear_button,self.resolution_toggle):
            control.setVisible(reference and not self.fixed_selection)
        if self.fixed_selection:self.resolution_options.hide()
        self.scene_panel.mode.setItemText(0,'参考图核对' if self.fixed_selection else '参考图与标记')
        self.scene_panel.mode.setToolTip('查看已选区域模型或原参考图；修改选区请返回第 02 步并重新建模。'
                                       if self.fixed_selection else '参考图可编辑区域；三维视图检查模型与路线。')
        if self.fixed_selection:
            self.scene_panel.stack.setToolTip('拖动平移 · 滚轮缩放；修改选区请返回第 02 步并重新建模。')
        has_selection = self.canvas.work is not None and bool(self.canvas.work.any())
        self.support_hint.setVisible(reference and self._stage==0 and not self.fixed_selection and self.support.isChecked()
                                     and has_selection and self.canvas._quality_image is None)
        self.support_hint.setText('请先涂抹工作区域；色层仅显示选区内的检查结果，不对整张照片着色。'
            if self.canvas.work is None or not self.canvas.work.any() else
            '色层已显示在选区内；绿色表示网格与参考深度一致，不代表临床可用。'
            if self.canvas._quality_image is not None else
            '当前选区尚无可显示的检查色层，请查看下方“选区检查”的进度或原因。')
        self.tools.setEnabled(loaded and not busy and reference);self.metadata.setEnabled(loaded and not busy)
        self.undo_button.setEnabled(loaded and not busy and reference and bool(self.canvas._undo))
        self.redo_button.setEnabled(loaded and not busy and reference and bool(self.canvas._redo))
        self.undo_shortcut.setEnabled(self.canvas.editable)
        self.redo_shortcut.setEnabled(self.canvas.editable)
        self.clear_button.setEnabled(loaded and not busy and reference and has_selection)
        self.clear_button.setToolTip('清空后可撤销；历史禁区保留' if has_selection else '尚无可清空的工作区域')
        self.head.setEnabled(loaded and not busy)
        self.open_button.setEnabled(not busy);self.return_button.setEnabled(not busy)
        self.cancel_button.setVisible(busy)
        self.cancel_button.setEnabled(busy and not self.runner.cancelling)
        self.cancel_button.setText('正在取消…' if self.runner.cancelling else '取消任务')
        self.previous.setEnabled(not busy)
        for index in range(3):self.tabs.setTabEnabled(index,not busy and (
            index==0 and not self.fixed_selection or index==1 and self.fixed_selection
            or bool(self.region) and not self.selection_dirty and (index==1 or index==2 and self.usable())))
        usable=self.usable()
        self.coverage.setVisible(self._stage>0 and loaded and bool(self.plan))
        self.coverage.set_context(self.model,self.plan,ready=usable and self._stage>0,busy=busy)
        if not usable and (self.plan or {}).get('surface_trajectory', {}).get('planner', {}).get('algorithm_version') in ('nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1'):
            current = self._variable_source_current()
            if self.plan['surface_trajectory']['planner']['algorithm_version']=='nominal-variable-width-refined-surface-v1':
                self.coverage.restore.setEnabled(current and not busy)
            else:
                self.coverage.variable.setEnabled(current and not busy)
                self.coverage.variable_restore.setEnabled(current and not busy)
        self.play_button.setEnabled(usable);self.report_button.setEnabled(usable);self.save_report_button.setEnabled(usable);self.recalculate.setEnabled(bool(self.region) and not busy and not self.selection_dirty)
        if self.canvas._coverage_image is not None:self.play_button.setEnabled(False)
        self.folder_button.setEnabled(bool(self.archive) and not busy)
        self.save_as_button.setEnabled(bool(self.region) and not busy and not self.selection_dirty)
        completed=bool(self.archive and not self.dirty)
        self.saved_actions.setVisible(completed)
        self.saved_actions.setEnabled(not busy)
        self.save_summary.setVisible(not completed)
        self.reviewed.setVisible(not completed)
        self.reviewed.setEnabled(not busy and self.quality.result is not None and self.quality.result['status']=='checked')
        self.save_report_button.setVisible(not completed)
        self.save_as_button.setVisible(not completed)
        self.saved_summary.setText('保存完成\n'+Path(self.archive).name+'\n'+self._save_warning if completed else '')
        self.saved_summary.setToolTip(str(self.archive) if completed else '')
        if self.selection_dirty:
            self.region_summary.setText('标记已修改 · 三维支持待检查\n蓝底为涂抹意图；擦除或留空的部分不纳入规划。')
        elif self.region:
            intent=self.region.get('selection_intent',{});support=self.region.get('selection_support',{})
            self.region_summary.setText(f"工作块  {len(intent.get('components',[]))} 个\n实测网格  {len(self.region['triangle_indices'])} 面\n几何待确认像素  {support.get('geometry_unconfirmed_pixel_count','未知')}")
        else:self.region_summary.setText('尚未标记工作区域\n在当前记录的参考图上涂抹选入，擦除排除；可选多个分离区域或在区域内部留空。\n按住左键标记；滚轮缩放，右键平移。' if loaded else '未加载数据：请打开区域方案，或返回采集页采集并检查模型。')
        if usable:
            stats=self.plan.get('surface_trajectory',{}).get('statistics',{})
            self.route_summary.setText(f"{stats.get('block_count','未知')} 个工作块 · {stats.get('segment_count','未知')} 条连续路线")
            findings=self.current_findings()
            self.warning.setText('待核对：\n'+'\n'.join(f'{index+1}. {item}' for index,item in enumerate(findings)) if findings else '')
            review_state='待复核研发草稿' if findings else '几何检查无待处理项的研发预览'
            stats=self.plan.get('surface_trajectory',{}).get('statistics',{})
            items='\n'.join('• '+item for item in findings)
            self.save_summary.setText(breakable_text(f"{self.name.text()}\n状态：{review_state}\n规划行距  {self.plan['spacing_mm']:g} mm\n"
                +f"工作块  {stats.get('block_count','未知')} 个\n"
                +(f'待处理 {len(findings)} 项：\n{items}' if findings else '当前几何检查未发现待处理项')
                +'\n保存包含路线、参数、检查报告及可编辑工程；不代表临床治疗可用。'))
        else:
            self.route_summary.setText('正在处理，请稍候…' if busy else '旧路线待更新' if self.plan else '尚未生成路线')
            self.route_stats.clear()
            self.warning.setText('参数已修改，显示的旧路线仅供参考；请重新计算后再保存。' if self.plan and not busy else '')
            self.save_summary.setText('请等待当前处理完成后再核对保存。' if busy else '当前参数尚无可保存的路线，请返回路线页重新计算。')
        route_stage=self._stage==1
        self.route_attention.setVisible(route_stage and usable and loaded and not busy and not (self.plan and not usable)
                                        and not (self.plan or {}).get('coverage_evidence'))
        attention=self._route_attention_text() if usable else self.quality.summary.text()
        # Keep only the actionable state on the route page; detailed findings
        # remain available to the existing review/save gates and hover help.
        self.route_attention.setText('<br>'.join(attention.split('<br>')[:2])
            + ('<br>当前选区几何支持尚未核验' if usable and self.quality.result is None else ''))
        self.route_attention.setToolTip('\n'.join(self.current_findings()) if usable else '')
        self.route_details_toggle.hide()
        self.warning.setVisible(bool(self.warning.text()) and (not route_stage or not usable or bool(self._error)))
        self.warning_scroll.setVisible(not self.warning.isHidden())
        self.warning_scroll.setFixedHeight(180 if route_stage else 240)
        self.route_stats.hide()
        self.route_extra.hide()
        self.quality.setVisible(not route_stage or not self.plan)
        try:
            head=self.head.record()
            mode=('自动排布最大行距' if head['mode']=='fixed_width_auto' else
                  '宽度计算' if head['mode']=='from_effective_width' else '指定行距')
            origin='研发占位值，非设备参数' if head['values_status']=='placeholder_not_device_parameters' else '手动参数，尚未验证'
            self.parameter_summary.setText(f"{mode}  {head['row_spacing_mm']:g} mm  ·  作用宽度 {head['effective_treatment_width_mm']:g} mm"
                +(f"\n期望重叠 {head['overlap_percent']:g}%" if head['mode']!='direct_row_spacing' else ''))
            if self.plan and head['mode']=='fixed_width_auto':
                layout=self.plan.get('surface_trajectory',{}).get('planner',{}).get('parameters',{}).get('fixed_width_layout')
                if layout is not None:
                    from femto.planning.fixed_width_layout import summary,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION
                    from femto.planning.continuous_quality import IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING
                    routing=self.plan['surface_trajectory']['planner']['parameters'].get('continuous_routing')
                    title=('贴面连接优化（试验） · ' if routing==REFINED_CONNECTOR_ROUTING else
                           '自适应分区与统一换序预算（试验） · ' if routing==GLOBAL_FLOOR_ROUTING else
                           '自适应分区与接缝协调（试验） · ' if layout.get('request',{}).get('version')==COORDINATED_FLOOR_PATCH_VERSION else
                           '自适应分区与快速换序（试验） · ' if layout.get('request',{}).get('version')==ADAPTIVE_FLOOR_PATCH_VERSION else
                           '局部疏密与快速换序（试验） · ' if routing==UNIFIED_FLOOR_ROUTING else
                           '局部疏密与绕行优化（试验） · ' if routing==FLOOR_DETOUR_ROUTING else
                           '局部方向与折返识别（试验） · ' if layout.get('request',{}).get('version')==FOLDBACK_PATCH_VERSION else
                           '局部方向与深度换序（试验） · ' if routing==DEEP_RELOCATION_ROUTING else
                           '局部方向与扫描段移动（试验） · ' if routing==RELOCATION_ROUTING else
                           '局部方向与两步换序（试验） · ' if routing==BRIDGE_REORDER_ROUTING else
                           '局部方向与联合换序（试验） · ' if routing==JOINT_REORDER_ROUTING else
                           '局部方向与短距换序（试验） · ' if routing==SHORT_DETOUR_ROUTING else
                           '局部方向与减少绕行（试验） · ' if routing==PRESERVED_DETOUR_ROUTING else
                           '局部方向与接缝联合优化（试验） · ' if routing==SEAM_BALANCED_ROUTING else
                           '局部方向与接缝顺接（试验） · ' if routing==SEAM_ROUTING else
                           '局部方向与绕行优化（试验） · ' if routing==BALANCED_ROUTING else
                           '局部方向与连接优化（试验新版） · ' if routing==SUPPORT_ROUTING else
                           '局部方向与连接优化（试验） · ' if routing==IMPROVED_ROUTING else
                           '局部方向适配（试验） · ' if layout.get('request',{}).get('version')==OPTIMIZED_PATCH_VERSION else '')
                    improvement=(self.plan['surface_trajectory'].get('gap_report',{}).get('reconnection',{}).get('ordering',{}).get('improvement') or {})
                    seam_stage=improvement.get('seam') or {}
                    balanced_stage=improvement.get('balanced') or {}
                    detour_stage=self.plan['surface_trajectory'].get('gap_report',{}).get('reconnection',{}).get('ordering',{}).get('preserved_detour') or {}
                    joint_stage=self.plan['surface_trajectory'].get('gap_report',{}).get('reconnection',{}).get('ordering',{}).get('joint_reorder') or {}
                    relocation_stage=self.plan['surface_trajectory'].get('gap_report',{}).get('reconnection',{}).get('ordering',{}).get('relocation') or {}
                    deep_stage=self.plan['surface_trajectory'].get('gap_report',{}).get('reconnection',{}).get('ordering',{}).get('relocation_deep') or {}
                    floor_stage=self.plan['surface_trajectory'].get('gap_report',{}).get('reconnection',{}).get('ordering',{}).get('relocation_floor') or {}
                    unified_stage=self.plan['surface_trajectory'].get('gap_report',{}).get('reconnection',{}).get('ordering',{}).get('relocation_unified') or {}
                    if routing==REFINED_CONNECTOR_ROUTING:
                        refinement=self.plan['surface_trajectory'].get('gap_report',{}).get('reconnection',{}).get('connector_refinement') or {}
                        shortened=sum(a.get('status')=='adopted' for a in refinement.get('attempts',[]))
                        title+=(f'已缩短 {shortened} 处连接 · ' if shortened else '保留原连接 · ')
                    elif routing in (UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING) and unified_stage.get('adopted') is False:
                        title+='统一预算搜索未采用新连接，保留原换序 · '
                    elif routing==FLOOR_DETOUR_ROUTING and floor_stage.get('adopted') is False:
                        title+='有界搜索未采用新连接，保留基础换序 · '
                    elif (improvement.get('skipped')=='geometry_budget' or improvement.get('reason') in ('geometry_budget','pair_budget','segment_budget')
                            or seam_stage.get('reason') in ('geometry_budget','pair_budget','segment_budget')
                            or balanced_stage.get('skipped')=='geometry_budget' or detour_stage.get('skipped')=='geometry_budget'
                            or joint_stage.get('skipped')=='geometry_budget'):
                        title+='连接优化因计算预算回退 · '
                    elif routing==DEEP_RELOCATION_ROUTING and deep_stage.get('skipped'):
                        title+='深度换序条件不足，保留上一条路线 · '
                    elif routing==DEEP_RELOCATION_ROUTING and deep_stage.get('adopted') is False:
                        title+='未找到更优深度换序，保留上一条路线 · '
                    elif routing in (RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING) and relocation_stage.get('skipped'):
                        title+='移动换序条件不足，保留原路线 · '
                    elif routing in (RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING) and relocation_stage.get('adopted') is False:
                        title+='未找到更优移动换序，保留原路线 · '
                    elif joint_stage.get('skipped')=='baseline_face_handoff' and routing not in (RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING):
                        title+='来源面交接差异，保留原路线 · '
                    elif routing in (JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING) and joint_stage.get('adopted') is False:
                        title+='未找到更优换序，保留原路线 · '
                    self.parameter_summary.setText(title+summary(layout,self.plan['surface_trajectory']['statistics']['row_count'],self.plan['surface_trajectory']))
            self.parameter_summary.setToolTip(origin+'；名义作用宽度用于规划与模型覆盖核验。')
        except ValueError as error:
            self.parameter_summary.setText('参数无效：'+self.head.error_text(error))
        self.head.setVisible(not self.plan or self.parameters_button.isChecked())
        self.parameter_summary.setVisible(bool(self.plan) and not self.parameters_button.isChecked())
        self.parameters_button.setVisible(route_stage and bool(self.plan))
        self.route_summary.setVisible(busy or bool(self.plan) and not usable)
        self.stage_panel_heading.setText(('选区工具','路线参数','保存核对')[self._stage])
        self.play_button.setVisible(route_stage and usable)
        self.route_legend.setVisible(route_stage and usable and reference)
        self.band_toggle.setVisible(route_stage and usable and reference and
                                    self.plan.get('treatment_head',{}).get('mode')=='fixed_width_auto')
        self._refresh_playback()
        try:self.head.record();head_valid=True
        except ValueError:head_valid=False
        if self._stage==0:
            text='检查选区并继续';ready=loaded and bool(self.canvas.work.any()) if loaded else False
            if loaded and self.canvas.forbidden.any() and not self.canvas.work.any():
                self.region_summary.setText('尚无工作区域 · 请先涂抹选入\n历史禁区仍保留，不会自动选择整幅画面。')
        elif self._stage==1:
            if self.fixed_selection and (self.selection_dirty or not self.region):
                text='检查模型并继续';ready=has_selection
            else:
                text='下一步：核对并保存' if usable else '生成路线';ready=bool(self.region) and not self.selection_dirty and head_valid
        else:
            findings=self.current_findings() if usable else []
            text='规划方案已保存' if self.archive and not self.dirty else '保存待复核研发草稿' if findings else '保存研发规划预览'
            ready=usable and self.reviewed.isChecked() and self.quality.result is not None and (self.dirty or not self.archive)
        self.primary.setText(text);self.primary.setEnabled(bool(ready and not busy))
        self.title.setToolTip(self.subtitle.text())
        self.primary.setToolTip('请等待当前任务结束' if busy else '请完成标记、有效参数和当前步骤核对' if not ready else '')
        self.workflow.set_step(self._stage+3 if loaded else None,
            '方案已保存；再次修改会使当前结果需要重新核对。' if completed else
            '当前步骤处理中，请等待完成或取消。' if busy else self.subtitle.text())
        self.status_label.setVisible(self.loading_source is None)
        self.source_toggle.setVisible(self.loading_source is None)
        if self.loading_source is not None:
            # The old model stays intact for cancellation, but none of its
            # controls, evidence or rendered surface belongs to this load.
            self.planning_body.setEnabled(False)
            self.canvas.editable=False
            self.undo_shortcut.setEnabled(False);self.redo_shortcut.setEnabled(False)
            self.source_toggle.setEnabled(False)
            self.title.setText('模型准备')
            self.task_context_label.setText(self._loading_request['loading_context'] or self.loading_source.name)
            self.primary.setText('生成路线');self.primary.setEnabled(False)
            self.primary.setToolTip('模型准备完成后可生成路线')
            self.previous.hide();self.cancel_button.hide()
            self.workflow.set_step(4,'正在准备模型；完成后自动显示。')

    def _stage_selected(self,index):
        if index==0 and self.fixed_selection:
            index=1
        if index==self._stage:
            previous=self.tabs.blockSignals(True)
            try:self.tabs.setCurrentIndex(index)
            finally:self.tabs.blockSignals(previous)
            return
        if (self.busy or index>0 and (not self.region or self.selection_dirty) and not (index==1 and self.fixed_selection)
                or index==2 and not self.usable()):
            # A fixed selection redirects tab 0 to tab 1. During a background
            # conversion, rolling back to stage 0 must not emit that redirect
            # again (it otherwise recursively bounces between the two tabs).
            previous=self.tabs.blockSignals(True)
            try:self.tabs.setCurrentIndex(self._stage)
            finally:self.tabs.blockSignals(previous)
            return
        previous=self.tabs.blockSignals(True)
        try:self.tabs.setCurrentIndex(index)
        finally:self.tabs.blockSignals(previous)
        self._stage=index;self.title.setText(('工作区域规划','三维蛇形路线规划','核对并保存规划')[index])
        if index==1:self.head.extra_button.setChecked(False)
        self.tabs.updateGeometry()
        self.inspector_scroll.verticalScrollBar().setValue(0)
        self.subtitle.setText(('扫描范围决定采集哪里；现在涂抹选择需要规划的工作区域，擦除排除。','核对下方路线参数，再点击底部“计算三维蛇形路线”；生成后可播放检查。','逐项核对区域、断点与覆盖提示，确认后保存本次规划方案。')[index])
        self.step_heading.setText(('03 / 04  ·  检查选区','03 / 04  ·  模型与路线','04 / 04  ·  核对并保存')[index])
        self.canvas.show_plan=index>0
        self.canvas.support=self.support.isChecked() and index==0
        self.canvas.update()
        self.canvas.timer.stop();self.stage_changed.emit(index);self.refresh()
        QTimer.singleShot(0,lambda:self.inspector_scroll.verticalScrollBar().setValue(0))

    def set_stage(self,index):
        if index==0 and self.fixed_selection:index=1
        self.refresh()
        self.tabs.setCurrentIndex(index)
        self._stage_selected(index)

    def _play_reference(self):
        if not self.usable():return
        self.scene_panel.show_reference()
        self.canvas.play()
        self.refresh()

    def _refresh_playback(self):
        self.play_button.setEnabled(self.usable() and bool(self.canvas._samples))
        self.play_button.setText('暂停播放' if self.canvas.timer.isActive() else
            '重播路线' if self.canvas._samples and self.canvas._play_index>=len(self.canvas._samples) else '播放路线')

    def _route_attention_text(self):
        """Summarize existing evidence without treating rendered lines as a pass."""
        findings=self.current_findings()
        stats=self.plan.get('surface_trajectory',{}).get('statistics',{})
        constrained=bool(self.plan.get('surface_trajectory',{}).get('gap_report',{}).get('quality_constraint'))
        coverage=('目标覆盖尚未完成核验' if constrained else '覆盖状态未知' if stats.get('coverage_status') is None
                  else '覆盖未完整' if stats['coverage_status']!='complete' else '实测网格覆盖完整')
        if self.plan.get('coverage_evidence') is not None:
            coverage=('名义模型覆盖已确认' if self.plan['coverage_evidence']['summary']['original_target_coverage_complete']
                      else '模型覆盖仍有未确认范围')
        lines=['<b>'+ (f'待核对 {len(findings)} 项' if findings else '当前未发现待核对项')+'</b>',
               '<span style="color:#b5cbd9">'+coverage+'</span>']
        quality=self.quality.result
        lines.append(f"<span style='color:#bca3eb'>●</span> 当前选区 {quality['pending']} 像素待检查" if quality and quality['status']=='checked'
                     else '当前选区几何支持尚未核验')
        target=(self.plan.get('planning_context') or {}).get('target')
        from femto.planning.target_surface import is_deferred
        lines.append('目标面积按需计算' if is_deferred(target) else
                     f"<span style='color:#f2b84b'>●</span> 未定位 {target['summary']['unlocated_pixel_count']} 像素 · 面积未知"
                     if target else '原目标台账：旧记录未核验')
        internal=(sum(not gap.get('inter_block',False) for gap in self.plan['chain_breaks'])
                  if 'chain_breaks' in self.plan else '未知')
        blocks=stats.get('block_count')
        inter=max(0,blocks-1) if blocks is not None else '未知'
        lines.append(f'块内断点 {internal} · 块间移动未规划 {inter}')
        if stats.get('unplannable_block_count'):lines.append(f"{stats['unplannable_block_count']} 个工作块无法规划")
        return '<br>'.join(lines)

    def primary_action(self):
        if self.busy:return
        if self._stage==0:self.commit_region()
        elif self._stage==1:
            if self.fixed_selection and (self.selection_dirty or not self.region):self.commit_region()
            elif self.usable():self.set_stage(2)
            else:self.generate()
        elif self.reviewed.isChecked():self.save_plan()

    def go_back(self):
        if self.busy:return
        if self.fixed_selection and self._stage<=1:self.return_capture()
        elif self._stage:self.set_stage(self._stage-1)
        else:self.return_capture()

    def return_capture(self):
        if self.can_leave():self.back_requested.emit()

    def commit_region(self, *, advance=True):
        if self.model is None or self.busy:return
        self.canvas.finish_stroke()
        if not self.canvas.work.any():
            self._status('请先涂抹工作区域；不需要的部分可擦除或留空。')
            return
        if self.region and not self.selection_dirty:self.canvas.fit_selection();self.set_stage(1);return
        model=self.model;work=self.canvas.work.copy();forbidden=self.canvas.forbidden.copy()
        self.quality.invalidate('正在核验工作区域…')
        name,note=self.name.text(),self.note.text()
        def done(result):
            if self.runner._cancel.is_set():
                self._status('区域检查已取消，标记保留。');self.quality.request();return
            candidate,region,assessment=result
            self.quality.install(assessment)
            try:require_geometry(assessment)
            except ValueError as error:self._failure(error);return
            self.model,self.region=candidate,region;self.canvas.model=self.model
            self.selection_dirty=False;self.dirty=True;self._invalidate()
            if advance:
                self.canvas.fit_selection()
                self.set_stage(1)
            else:
                self.refresh()
            self.checkpoint()
            unconfirmed=assessment['pending']
            self._status(f'区域已就绪；{unconfirmed} 个像素几何待确认，规划结果将保留覆盖提示。' if unconfirmed else
                         '已同步核验后的三维工作区域；修改请返回参考图。' if not advance else '区域已就绪，设置行距后计算路线。')
        self._status('正在检查选区的实测三维几何…')
        def job(cancel,progress):
            assessment=assess_selection(model,work,forbidden,cancel)
            if assessment['status']!='checked' or not assessment['supported']:
                return None,None,assessment
            candidate,region=extract_region(model,work,forbidden,name,note,cancel)
            return candidate,region,assessment
        self._run(job,done)

    def generate(self):
        if self.model is None or self.region is None or self.busy or self.selection_dirty:return
        try:head=self.head.record();layout_algorithm=self.head.layout_algorithm()
        except ValueError as error:self._failure(error);return
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        model=clone_model(self.model,freeze_inputs=True);region=deepcopy(self.region)
        from femto.app.qt_planning_io import needs_model_confirmation,prepare_route_model
        prepare_model=needs_model_confirmation(model) or getattr(model,'route_prepare_required',False)
        source=self.source
        # Keep a previously accepted route visible while a recomputation runs. A
        # failed request must not erase the last reviewable result; parameter and
        # region edits already invalidate it before this method is reached.
        self.dirty=True
        self.reviewed.setChecked(False)
        def done(result):
            if self.runner._cancel.is_set():
                self._status('路线计算已取消，上一条结果保留。');return
            plan,assessment,candidate,candidate_region=result
            if prepare_model:
                from femto.app.qt_planning_resolution import journal_for
                self.install_model(candidate,source,journal_for(source,candidate),assessment=assessment,notify=False)
                self.head.set_layout_algorithm(layout_algorithm)
                self.dirty=True
            self.quality.invalidate();self.quality.install(assessment)
            self.plan=plan;self.canvas.set_plan(plan);self.set_stage(1);self.checkpoint()
            self.parameters_button.setChecked(False)
            self._status('路线已生成；请播放检查，再核对保存。')
        work,forbidden=self.canvas.work.copy(),self.canvas.forbidden.copy()
        def job(cancel,progress):
            candidate,candidate_region=(prepare_route_model(model,region,work,forbidden,source,cancel,progress)
                                        if prepare_model else (model,region))
            return calculate(candidate,candidate_region,cancel,progress)
        def calculate(model,region,cancel,progress):
            assessment=assess_selection(model,work,forbidden,cancel)
            if assessment['status']!='checked':require_geometry(assessment)
            from femto.geometry.region_reuse import operation
            with operation():
                if layout_algorithm is None:
                    plan=surface_workspace.generate_checked(model,region,head,GeometryBudget(cancel).check,progress,
                                                    validation_check_factory=lambda:GeometryBudget(cancel).check,
                                                    defer_target=True)
                else:
                    from femto.planning.continuous_quality import IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING
                    from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION
                    plan=surface_workspace.generate_checked(model,region,head,GeometryBudget(cancel).check,progress,
                                                    validation_check_factory=lambda:GeometryBudget(cancel).check,
                                                    defer_target=True,
                                                    layout_algorithm=COORDINATED_FLOOR_PATCH_VERSION if layout_algorithm in (GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING) else layout_algorithm if layout_algorithm in (ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION) else FLOOR_PATCH_VERSION if layout_algorithm in (FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING) else FOLDBACK_PATCH_VERSION if layout_algorithm==FOLDBACK_PATCH_VERSION else OPTIMIZED_PATCH_VERSION,
                                                    **({'routing_algorithm':UNIFIED_FLOOR_ROUTING} if layout_algorithm in (ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION) else {'routing_algorithm':SEAM_BALANCED_ROUTING} if layout_algorithm==FOLDBACK_PATCH_VERSION else
                                                       {'routing_algorithm':layout_algorithm} if layout_algorithm in (IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING) else {}))
            return plan,assessment,model,region
        self._run(job,done)

    def show_report(self):
        if not self.usable():return
        model=self.model;region=deepcopy(self.region);plan=deepcopy(self.plan)
        def done(text):self._report=text;show_text(self,'三维路线与覆盖检查',text)
        self._run(lambda cancel,progress:surface_workspace.report_text(plan,model.mesh,region),done)

    def verify_coverage(self):
        if not self.usable() or self.plan.get('planning_context') is None:return
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        from femto.planning import coverage_workflow
        from femto.planning.coverage_certificate import CoverageBudget
        from femto.planning.planning_revision import digest
        model=clone_model(self.model,freeze_inputs=True);region=deepcopy(self.region);plan=deepcopy(self.plan)
        expected=digest(self.plan)
        def job(cancel,progress):
            check=GeometryBudget(cancel).check
            from femto.planning.target_preparation import prepare
            prepared=prepare(model,region,plan,check,progress)
            # Both native and portable saves replay this proof within the
            # existing 120 s operation limit. Reserve time for both replays
            # and input validation; unprocessed targets remain explicit.
            certificate,ledger=coverage_workflow.calculate(model,region,prepared,check,progress,
                                                          budget=CoverageBudget(seconds=30))
            result=coverage_workflow.attach(model,region,prepared,certificate,ledger,check=check)
            check()
            return result,certificate,ledger,model.coverage_display
        def done(result):
            if digest(self.plan)!=expected:
                self._status('路线已改变，旧覆盖结果未安装；请重新核验。');return
            if result[0]['surface_trajectory']['planner']['algorithm_version']=='nominal-variable-width-surface-v1':
                from femto.planning.variable_footprint import require_current
                try:require_current(result[0]['surface_trajectory']['planner']['parameters']['variable_footprint'])
                except ValueError as error:self._status(str(error));return
            self.plan,certificate,ledger,display=result
            self.model.coverage_certificate=certificate;self.model.target_ledger=ledger;self.model.coverage_display=display
            self.canvas.set_plan(self.plan)
            self.dirty=True;self.archive=None;self._report='';self.reviewed.setChecked(False)
            self.refresh();self.coverage.show_layer.setChecked(True);self.checkpoint()
            self._status(coverage_workflow.summary_text(certificate.metadata)+' 正式保存后可完整回读。')
        self._run(job,done)

    def _coverage_layer(self,active,reason):
        display=self.coverage.display if active else None
        if active:self.canvas.timer.stop()
        self.canvas.set_coverage_overlay(None if display is None else display.rgba(reason, pixel_reason=self.coverage.reference_reason))
        self.scene_panel.set_coverage(display,reason)
        if hasattr(self,'play_button'):self.play_button.setEnabled(self.usable() and not active)

    def _strict_operation(self, restore):
        if not self.usable() or self.plan.get('planning_context') is None:return
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        from femto.planning import strict_workflow
        from femto.planning.planning_revision import digest
        model=clone_model(self.model,freeze_inputs=True);region=deepcopy(self.region);plan=deepcopy(self.plan)
        expected=digest(self.plan)
        def job(cancel,progress):
            check=GeometryBudget(cancel).check
            result=(strict_workflow.restore(model,region,plan,check) if restore else
                    strict_workflow.calculate(model,region,plan,check,progress))
            check()
            return result,model.target_ledger
        def done(result):
            if digest(self.plan)!=expected:
                self._status('路线已改变，旧重建结果未安装。');return
            self.plan,ledger=result
            self.model.target_ledger=ledger;self.model.coverage_certificate=None;self.model.coverage_display=None
            self.canvas.set_plan(self.plan)
            self.dirty=True;self.archive=None;self._report='';self.reviewed.setChecked(False)
            self.refresh();self.checkpoint()
            self._status(('已恢复常规路线。' if restore else '严格路径已重建并重新检查。')
                         +'请重新核验模型覆盖、核对路线后保存为新版本。')
        self._run(job,done)

    def refine_coverage(self):
        self._refinement_operation(False)

    def edit_variable_model(self):
        if self.busy or not self.plan:return
        from femto.app.qt_variable_footprint import VariableFootprintDialog
        params=self.plan['surface_trajectory']['planner']['parameters']
        dialog=VariableFootprintDialog(params['effective_width_mm'],params.get('variable_footprint'),self)
        if dialog.exec() and dialog.profile is not None:self._variable_operation(dialog.profile)

    def _variable_operation(self, profile):
        if self.busy or not self.plan or self.plan.get('planning_context') is None:return
        if not self.usable() and not self._variable_source_current():
            self._status('区域或生成参数已变化，请先重新生成路线。');return
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        from femto.planning import variable_width_workflow, variable_footprint
        from femto.planning.planning_revision import digest
        model=clone_model(self.model,freeze_inputs=True);region=deepcopy(self.region);plan=deepcopy(self.plan)
        profile=deepcopy(profile);expected=digest(self.plan)
        def job(cancel,progress):
            check=GeometryBudget(cancel).check
            result=(variable_width_workflow.restore(model,region,plan,check) if profile is None else
                    variable_width_workflow.calculate(model,region,plan,profile,check,progress))
            check();return result,model.target_ledger
        def done(result):
            if digest(self.plan)!=expected:
                self._status('路线已改变，旧变宽模型结果未安装。');return
            if profile is not None:
                try:variable_footprint.require_current(profile)
                except ValueError as error:self._status(str(error));return
            self.plan,ledger=result
            self.model.target_ledger=ledger;self.model.coverage_certificate=None;self.model.coverage_display=None
            self.canvas.set_plan(self.plan)
            self.dirty=True;self.archive=None;self._report='';self.reviewed.setChecked(False)
            self.refresh();self.checkpoint()
            self._status(('已恢复应用模型前的严格路线。' if profile is None else '已应用名义变宽模型并核验所有扫描与连接边。')
                         +'请重新核验覆盖、核对路线并保存新版本。')
        self._run(job,done)

    def _variable_source_current(self):
        # Expiry must allow restoration, while changed geometry/selection/head
        # inputs must not be installed under the old model's parameters.
        if self.model is None or self.region is None or self.selection_dirty or not self.plan or not self._layout_choice_current():return False
        from femto.planning import planning_revision
        try:
            if self.head.record()!=self.plan['treatment_head']:return False
            revision=self.plan['planning_context']['revision']
            return planning_revision.capture_for(self.model,self.region,self.head.record(),revision)==revision
        except (ValueError,KeyError,TypeError):return False

    def restore_coverage_route(self):
        self._refinement_operation(True)

    def _refinement_operation(self, restore):
        if self.busy or not self.plan or self.plan.get('coverage_evidence') is None:return
        variable_refined=self.plan['surface_trajectory']['planner']['algorithm_version']=='nominal-variable-width-refined-surface-v1'
        if not self.usable() and not (restore and variable_refined and self._variable_source_current()):return
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        from femto.planning import refinement_workflow, coverage_refinement
        from femto.planning.planning_revision import digest
        model=clone_model(self.model,freeze_inputs=True);region=deepcopy(self.region);plan=deepcopy(self.plan)
        expected=digest(self.plan)
        def job(cancel,progress):
            check=GeometryBudget(cancel).check
            result=(refinement_workflow.restore(model,region,plan,check) if restore else
                    refinement_workflow.calculate(model,region,plan,check,progress))
            check()
            return result,model.coverage_certificate,model.target_ledger,model.coverage_display
        def done(result):
            if digest(self.plan)!=expected:
                self._status('路线已改变，旧补充结果未安装。');return
            if not restore and result[0]['surface_trajectory']['planner']['algorithm_version']=='nominal-variable-width-refined-surface-v1':
                from femto.planning.variable_footprint import require_current
                try:require_current(result[0]['surface_trajectory']['planner']['parameters']['variable_footprint'])
                except ValueError as error:self._status(str(error));return
            self.plan,certificate,ledger,display=result
            self.model.coverage_certificate=certificate;self.model.target_ledger=ledger;self.model.coverage_display=display
            self.canvas.set_plan(self.plan)
            self.dirty=True;self.archive=None;self._report='';self.reviewed.setChecked(False)
            self.refresh();self.coverage.show_layer.setChecked(True);self.checkpoint()
            if not restore and self.plan['surface_trajectory']['planner']['algorithm_version']=='nominal-variable-width-refined-surface-v1':
                from femto.planning.variable_width_refinement import summary_text
                text=summary_text(self.plan['surface_trajectory'])
            else:text=coverage_refinement.summary_text(self.plan['surface_trajectory']) if not restore else '已恢复补充前路线和覆盖证据，请保存为新版本。'
            self._status(text)
        self._run(job,done)

    def _focus_coverage(self,reason):
        if not self.usable() or self.coverage.display is None:return
        self.coverage.show_layer.setChecked(True)
        if not self.scene_panel.focus_coverage(reason):
            self.scene_panel.show_reference();self.canvas.fit_mask(self.coverage.display.mask(reason, pixel_reason=self.coverage.reference_reason))
        from femto.planning.coverage_display import NAMES
        self._status('已定位：'+NAMES[reason]+'。未定位部分只显示参考像素，不补造三维位置。' if reason==8 else
                     '已定位：'+NAMES[reason]+'；可切换参考图与三维模型检查。')

    def _coverage_task(self, reason):
        if not self.usable() or self.coverage.display is None:return
        from femto.planning.coverage_tasks import operation
        tasks = self.coverage.display.tasks
        try:tasks.require_plan(self.plan)
        except ValueError as error:self._status(str(error));return
        task = tasks.task(reason)
        action, _ = operation(task, self.plan['surface_trajectory']['planner']['algorithm_version'])
        if action == 'refine':self.refine_coverage()
        elif action == 'recalculate':self.verify_coverage()
        elif action == 'restore':self.restore_coverage_route()
        elif action is not None:
            self._focus_coverage(reason)
            self._status(task.guidance+' 当前路线与证据保留；完成数据检查后再重新生成和核验。')
            from femto.app.qt_gap_inspection import GapInspectionDialog
            mask=self.coverage.display.mask(reason, pixel_reason=self.coverage.reference_reason)
            details=dict(task.unknown_by_reason)
            observation_reasons={'missing_reference_depth','depth_outside_range','no_source_pixel',
                'no_mesh_surface','depth_conflict','occlusion_or_multiple_surface_ambiguity'}
            from femto.planning.target_surface import PIXEL_STATES
            selected_reason=PIXEL_STATES.get(self.coverage.reference_reason)
            allow_acquisition=(reason==2 or (reason==8 and (
                selected_reason in observation_reasons if self.coverage.reference_reason
                else any(details.get(key,0) for key in observation_reasons))))
            from femto.planning.target_continuation import available as target_continuation_available
            from femto.planning.target_reconstruction import available as target_exact_available
            dialog=GapInspectionDialog(self.model,mask,task.guidance,self,allow_acquisition=allow_acquisition,
                allow_continuation=(reason==8 and target_continuation_available(self.plan) and
                    (self.coverage.reference_reason in (None,9))),
                allow_exact_partition=(reason==8 and target_exact_available(self.plan) and
                    (self.coverage.reference_reason in (None,8,10))))
            if dialog.exec():
                if dialog.action=='rescan':
                    acquisition_mask=(mask & np.isin(self.coverage.display.reference_status,[2,3,4,5,6,7])
                                      if reason==8 else mask)
                    self._prepare_gap_rescan(acquisition_mask,reason,self.coverage.reference_reason)
                elif dialog.action=='audit':self._audit_reconstruction_gap(mask)
                elif dialog.action=='continue_target':self._continue_original_target()
                elif dialog.action=='exact_target':self._continue_original_target(exact=True)

    def _prepare_gap_rescan(self,mask,reason,pixel_reason):
        from uuid import uuid4
        from femto.planning.gap_followup import prepare
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        model=clone_model(self.model,freeze_inputs=True);plan=deepcopy(self.plan)
        mask=np.array(mask,dtype=bool,copy=True);output=OUTPUTS_ROOT/('gap_followup_'+uuid4().hex)
        def job(cancel,progress):
            progress('记录缺口原因、原目标与来源方案，再进入独立采集')
            return prepare(model,plan,mask,reason,pixel_reason,output,GeometryBudget(cancel).check)
        def done(request):
            self._gap_followup_request=request
            QTimer.singleShot(0,self.rescan_requested.emit)
        self._run(job,done)

    def _continue_original_target(self, *, exact=False):
        from uuid import uuid4
        if exact:
            from femto.planning.target_reconstruction import calculate
        else:
            from femto.planning.target_continuation import calculate
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        from femto.planning.planning_revision import digest
        if not self.usable():return
        model=clone_model(self.model,freeze_inputs=True);region=deepcopy(self.region);plan=deepcopy(self.plan)
        expected=digest(self.plan)
        output=OUTPUTS_ROOT/(('target_exact_partition_' if exact else 'target_continuation_')+uuid4().hex)
        def job(cancel,progress):
            return calculate(model,region,plan,output,GeometryBudget(cancel).check,progress)
        def done(result):
            if digest(self.plan)!=expected:
                self._status('方案已改变，续算结果未安装；当前方案保留。');return
            self.plan,ledger=result
            self.model.target_ledger=ledger
            self.model.coverage_certificate=None;self.model.coverage_display=None
            self.canvas.set_plan(self.plan);self.dirty=True;self.archive=None;self._report=''
            self.reviewed.setChecked(False);self.refresh();self.checkpoint()
            remaining=ledger.summary['unlocated_pixel_count'] if exact else ledger.summary['unknown_by_reason']['budget_unprocessed']
            message=(f'精确分区核验完成；含未定位范围剩余 {remaining} 像素。' if exact else
                     f'原目标已续算；预算未处理剩余 {remaining} 像素。')
            self._status(message+'请重新核验模型覆盖并保存新版本。旧方案证据：'+str(output/'previous'))
        self._run(job,done)

    def _audit_reconstruction_gap(self, mask):
        from datetime import datetime
        from uuid import uuid4
        from femto.pipeline.reconstruction_gap_audit import audit
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        model=clone_model(self.model,freeze_inputs=True);mask=np.array(mask,dtype=bool,copy=True)
        output=OUTPUTS_ROOT/('gap_inspection_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
        def job(cancel,progress):
            return audit(model,mask,output,check=GeometryBudget(cancel).check,progress=progress)
        def done(report):
            names={'no_hit_at_tsdf_extraction':'TSDF提取时没有交点','topology_cleanup':'拓扑清理',
                'initial_component_filter':'初次小分量过滤','measured_support_filter':'实测支持校验',
                'final_component_filter':'末次小分量过滤','surface_present_in_replay':'重放出现表面'}
            text='\n'.join(f'{names[key]}：{count} 像素' for key,count in report['counts'].items())
            text+='\n\n'+'\n'.join(item['guidance'] for item in report.get('recommended_actions',()))
            show_text(self,'缺面分阶段排查',text+'\n\n诊断结果：'+str(output/'report.json')+
                '\n这是处理阶段定位；仍需结合原始观测检查具体原因。当前模型、路线与确认状态保留。')
            self._status('重建缺面排查已完成：'+str(output/'report.json'))
        self._run(job,done)

    def _export_coverage_tasks(self):
        if not self.usable() or self.coverage.display is None:return
        tasks = self.coverage.display.tasks
        try:tasks.require_plan(self.plan)
        except ValueError as error:self._status(str(error));return
        path, _ = QFileDialog.getSaveFileName(self, '导出后续任务', '覆盖后续任务.json', 'JSON (*.json)')
        if not path:return
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        from femto.planning.coverage_tasks import write
        from femto.planning.planning_revision import digest
        plan = deepcopy(self.plan);expected = digest(plan)
        def job(cancel, progress):
            progress('导出当前原目标与覆盖证据对应的后续任务')
            write(path, tasks, plan, GeometryBudget(cancel).check)
            return path
        def done(result):
            self._status('后续任务已导出：'+str(result)+('；路线已变化，请重新导出当前版本。' if digest(self.plan)!=expected else ''))
        self._run(job, done)

    def save_plan(self):
        if not self.usable() or not self.reviewed.isChecked():return
        if self.quality.result is None or self.quality.result['status']!='checked':
            self._status('请等待当前选区几何检查完成后重新核对。');return
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        from femto.planning.region_project import save_project
        model=clone_model(self.model,freeze_inputs=True);region=deepcopy(self.region);plan=deepcopy(self.plan)
        work,forbidden=self.canvas.work.copy(),self.canvas.forbidden.copy()
        def job(cancel,progress):
            from contextlib import nullcontext
            from femto.geometry.region_reuse import operation as geometry_operation
            from femto.planning.save_validation import SaveValidationSession
            from femto.operation_diagnostics import record_operation
            check=GeometryBudget(cancel).check
            progress('1/4 检查当前选区')
            with record_operation('save.selection_assessment'):
                require_geometry(assess_selection(model,work,forbidden,cancel))
            verified=[]
            extra={'verified_sink':verified.append} if plan.get('coverage_evidence') is not None and plan['surface_trajectory']['planner']['algorithm_version'] in ('coverage-refined-surface-v1', 'constructed-coverage-refined-surface-v1', 'nominal-variable-width-surface-v1', 'nominal-variable-width-refined-surface-v1') else {}
            # Own geometry preparation in this worker for both save formats.
            # The validation session and every file/input recheck stay intact.
            with geometry_operation(), (SaveValidationSession() if plan.get('planning_context') else nullcontext()) as session:
                session_args={'target_session':session} if session is not None else {}
                with record_operation('save.route_and_native'):
                    folder=surface_workspace.export(model,region,plan,check,lambda text:progress('2/4 '+text),**extra,**session_args)
                try:
                    with record_operation('save.portable_project'):
                        save_project(model,region,folder/'planning.region.zip',plan,check,progress=lambda text:progress('3/4 '+text),catalog_root=OUTPUTS_ROOT/'saved_projects',allow_replace=False,
                                     **({'verified_assets':verified[0]} if verified else {}),**session_args)
                except Exception as error:
                    raise RuntimeError(f'三维路线已保存到 {folder}；可编辑工程保存未完成：{error}。当前方案仍保留，可重新保存。') from error
            progress('4/4 文件已保存，更新工作台记录')
            return folder
        self._run(job,lambda path:self._saved(path,saved_model=model))

    def _saved(self,path, *, source_path=None,saved_model=None):
        """Commit a saved result and, for Save As, move the draft binding.

        A portable project created by Save As becomes the current editable
        source.  Keeping the old source/journal here would write later edits
        under the old project key, so reopening the new ZIP could miss them.
        Normal route exports still keep the current source project.
        """
        previous_journal=self.journal
        self.archive=Path(path).resolve();self.dirty=False
        if saved_model is not None:
            self.model.project_id=getattr(saved_model,'project_id',None)
            self.model.revision_id=getattr(saved_model,'revision_id',None)
        old_draft_error=None
        if previous_journal:
            try:previous_journal.discard()
            except OSError as error:old_draft_error=error
        if source_path is not None:
            from femto.app.qt_planning_resolution import journal_for
            self.source=Path(source_path).resolve()
            self._refresh_source_label()
            root=previous_journal.path.parent if previous_journal else None
            self.journal=journal_for(self.source,self.model,root)
            self.loaded.emit(str(self.source))
        self._save_warning=getattr(saved_model,'last_save_warning',None) or ''
        if old_draft_error is not None:
            self._save_warning+='\n旧草稿清理失败：'+str(old_draft_error)
        self._status('规划方案已保存。'+('\n'+self._save_warning if self._save_warning else ''))
        self.status_label.setToolTip('保存位置：'+str(path));self.saved.emit(str(path));self.refresh()

    def save_as(self):
        if self.busy or self.region is None or self.selection_dirty:return
        path,_=QFileDialog.getSaveFileName(self,'另存可编辑方案','规划方案.region.zip','区域工程 (*.zip)')
        if not path:return
        if Path(path).exists():
            QMessageBox.information(self,'保留原版本','请选择一个新的文件名保存新版本，原文件不会被覆盖。')
            return
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        from femto.planning.region_project import save_project
        model=clone_model(self.model,freeze_inputs=True);region=deepcopy(self.region);plan=deepcopy(self.plan) if self.usable() else None
        try:request=self.request()
        except ValueError as error:self._failure(error);return
        if plan is None and self.head.layout_algorithm() is not None:
            request['values']['layout_algorithm']=self.head.layout_algorithm()
        work,forbidden=self.canvas.work.copy(),self.canvas.forbidden.copy()
        def job(cancel,progress):
            from femto.geometry.region_reuse import operation as geometry_operation
            with geometry_operation():
                if plan is not None:require_geometry(assess_selection(model,work,forbidden,cancel))
                return save_project(model,region,path,plan,GeometryBudget(cancel).check,planning_parameters=request,progress=progress,catalog_root=OUTPUTS_ROOT/'saved_projects',allow_replace=False)
        self._run(job,
                  lambda saved:self._saved(saved,source_path=saved,saved_model=model))

    def view_saved(self):
        if self.busy or self.dirty or not self.archive:return
        path=self.archive/'planning.region.zip' if self.archive.is_dir() else self.archive
        self.review_requested.emit(str(path))

    def open_archive(self):
        if self.archive:
            try:open_path(self.archive if self.archive.is_dir() else self.archive.parent)
            except Exception as error:self._failure(error)

    def checkpoint(self):
        if self.journal is None or self.model is None or not self.dirty:return
        from femto.planning.planning_drafts import SCHEMA,source_binding
        from femto.annotation.selection_intent import encode_runs
        try:request=self.request()
        except ValueError:request=None
        data=dict(schema=SCHEMA,created_at=time.time(),source=source_binding(self.model),
            regions=deepcopy(self.model.store.regions),points=deepcopy(self.model.store.points),
            parameters={self.region['annotation_id']:request} if self.region and request else {},
            polygon=[],exclusion_target=None,selection_polygons={'work':[],'forbidden':[]},paint_role=self.canvas.role,
            focus=self.region['annotation_id'] if self.region else None,mode='region',name=self.name.text(),note=self.note.text(),
            path_policy='not_restored_regenerate_required',qt_masks=dict(image_size=[self.canvas.work.shape[1],self.canvas.work.shape[0]],
                work=encode_runs(self.canvas.work),forbidden=encode_runs(self.canvas.forbidden),pending=self.selection_dirty),
            qt_parameters=request,qt_layout_algorithm=self.head.layout_algorithm())
        self.journal.submit(data)
        self._refresh_draft_status()

    def _refresh_draft_status(self):
        state=self.journal.status() if self.journal else 'empty'
        self.draft_label.setText(('草稿保存失败：'+str(self.journal.error) if state=='error' else
                                 '草稿保存中…' if state=='saving' else '草稿已保存' if state=='saved' and self.dirty else ''))
        self.draft_label.setVisible(bool(self.draft_label.text()))
        self.retry_draft_button.setVisible(state=='error')
        self.retry_draft_button.setEnabled(not self.busy)
        self.draft_save_as.setVisible(state=='error')
        self.draft_save_as.setEnabled(not self.busy and self.region is not None and not self.selection_dirty)
        self.draft_save_as.setToolTip('请先完成区域几何检查，再另存可编辑工程。' if self.selection_dirty or self.region is None else '')

    def restore_draft(self,data):
        from femto.annotation.selection_intent import decode_runs
        from femto.annotation.square_brush import polygon_mask
        shape=self.model.rgb.shape[:2];size=[shape[1],shape[0]]
        qt_masks=data.get('qt_masks')
        if qt_masks:
            if qt_masks['image_size']!=size:raise ValueError('草稿图像尺寸不匹配')
            work=decode_runs(qt_masks['work'],size,values=(0,1)).astype(np.uint8)
            forbidden=decode_runs(qt_masks['forbidden'],size,values=(0,1)).astype(np.uint8)
            pending=bool(qt_masks['pending'])
        else:
            polygons=data.get('selection_polygons',{})
            work=polygon_mask(polygons.get('work',[]),size);forbidden=polygon_mask(polygons.get('forbidden',[]),size)
            pending=bool(work.any() or forbidden.any())
            if not pending and data['regions']:work,forbidden=masks_from_region(data['regions'][0],shape)
        candidate=clone_model(self.model);candidate.store.regions=deepcopy(data['regions']);candidate.store.points=deepcopy(data['points'])
        region=next((r for r in candidate.store.regions if r['annotation_id']==data.get('focus')),None)
        if region is None and candidate.store.regions:region=candidate.store.regions[0]
        request=data.get('qt_parameters') or data['parameters'].get(region['annotation_id'] if region else '')
        head=None
        if request and request.get('kind')==surface_workspace.KIND:head=surface_workspace.checked_head(request['values']['treatment_head'])
        from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,FLOOR_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION
        from femto.planning.continuous_quality import IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING
        layout_algorithm=data.get('qt_layout_algorithm')
        if layout_algorithm not in (None,OPTIMIZED_PATCH_VERSION,FOLDBACK_PATCH_VERSION,ADAPTIVE_FLOOR_PATCH_VERSION,COORDINATED_FLOOR_PATCH_VERSION,IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING,JOINT_REORDER_ROUTING,BRIDGE_REORDER_ROUTING,RELOCATION_ROUTING,DEEP_RELOCATION_ROUTING,FLOOR_DETOUR_ROUTING,UNIFIED_FLOOR_ROUTING,GLOBAL_FLOOR_ROUTING,REFINED_CONNECTOR_ROUTING) or (layout_algorithm is not None and (head is None or head['mode']!='fixed_width_auto')):
            raise ValueError('草稿的局部分区排布版本无效')
        # Validate all recovered values before replacing live state.
        self.model=candidate;self.region=region;self.canvas.set_model(candidate,work,forbidden)
        self.quality.request()
        self.name.setText(data['name']);self.note.setText(data['note'])
        if head:self.head.set_record(head)
        self.head.set_layout_algorithm(layout_algorithm)
        self.selection_dirty=pending or not region;self.dirty=True;self._invalidate();self.set_stage(0)
        self._status('草稿已恢复，区域与参数已保留；请重新核对并生成路线。');self.refresh()

    def show_details(self):
        if self.loading_source is not None:
            text=(getattr(self,'_failure_details','') or self.preparation.last_progress)
            show_text(self,'模型准备详情',text+'\n\n当前来源：'+str(self.loading_source)
                      +'\n\n准备完成后自动显示模型。取消或失败不会替换此前已加载的模型与选区。')
            return
        interaction=('本页沿用重建前选区，仅查看模型和规划路线；修改选区请返回第 02 步并重新建模。'
                     if self.fixed_selection else '左键涂抹；右键平移；滚轮缩放。笔刷尺寸为屏幕像素。')
        text=(getattr(self,'_failure_details','') or self.status_label.text().replace('\u200b',''))+'\n\n'+interaction+'\n只规划涂抹选入的工作区域；擦除或留空部分不参与规划。历史禁区保留并扣除，不自动选择整幅画面。\n检查色层：绿为网格与参考深度一致，紫为证据待检查，橙为缺失或冲突。比例是像素比例，不代表表面积或治疗覆盖。\n深度、外壳直径和避让余量只记录，当前规划不使用。'
        if self.source:text+='\n\n来源：'+str(self.source)
        if self.archive:text+='\n\n已保存位置：'+str(self.archive)
        if self.journal and self.journal.error:text+='\n草稿保存失败：'+self.journal.error
        show_text(self,'规划工作台详情',text)

    def can_leave(self):
        if self.busy:
            self._status('任务正在运行，请等待完成或点击取消任务。');return False
        self.canvas.finish_stroke();self.canvas.timer.stop();self.checkpoint();return True

    def shutdown(self):
        self.preparation.stop()
        self.draft_timer.stop();self.canvas.timer.stop()
        if not self.quality.shutdown():return False
        if not self.runner.shutdown(200):return False
        self.canvas.finish_stroke();self.checkpoint()
        if self.journal:self.journal.wait()
        if self.journal and self.dirty and self.journal.status() in ('saving','error'):
            self._refresh_draft_status()
            return False
        self.draft_status_timer.stop()
        self.scene_panel.shutdown()
        return True
