"""Explicit, transactional creation of a derived planning model before painting."""
from __future__ import annotations

import hashlib
from pathlib import Path
import uuid

import numpy as np


DERIVATION_SCHEMA = 'reduced-planning-reference-v1'
REDUCED_NOTICE = '简化规划参考，保留原三维网格和原始数据。'


def derivation(model):
    value = (getattr(model, 'view_reference', None) or {}).get('derivation')
    return value if isinstance(value, dict) and value.get('schema') in (DERIVATION_SCHEMA, 'surface-patch-planning-reference-v1') else None


def profile_text(model):
    reduced = derivation(model)
    if reduced and reduced.get('schema') == 'surface-patch-planning-reference-v1':
        from femto.local.planning_tsdf import PROFILE as TSDF_PROFILE, LEGACY_PROFILE
        if reduced.get('profile') in (TSDF_PROFILE, LEGACY_PROFILE):
            return '统一表面规划／沿线自动取点（原始测量与选区保留）'
        return '表面粗分区／沿线自动取点（原选区与完整表面保留）'
    from femto.planning.route_sampling import CURVATURE_POLICY
    if reduced and reduced.get('sampling_policy')==CURVATURE_POLICY:
        return '粗分区规划／曲面弯折与边缘自适应取点（试验）'
    if reduced and reduced.get('sampling_policy'):
        return '粗分区规划／沿线自动取点（行距保持）'
    return '粗分区规划／简化参考（保留原三维网格）' if derivation(model) else '标准规划参考'


def blank_model(model):
    return (model is not None and not any(getattr(model.store,name,None) for name in ('points','regions','_history'))
            and getattr(model,'saved_plan',None) is None)


def can_choose_mode(page):
    if page.busy:return False
    if page.model is None:return True
    if not blank_model(page.model) or page.region is not None or page.plan is not None:return False
    canvas=page.canvas
    return not (any(value is not None and np.any(value) for value in (canvas.work,canvas.forbidden))
                or canvas._undo or canvas._redo or canvas._before is not None)


def prepare_new_source(model, source, *, enabled, restore_draft, check, progress):
    """Coarsen fresh empty sources; retain every saved selection/route/draft."""
    if not enabled or not blank_model(model) or derivation(model):return model
    if getattr(model,'pre_reconstruction_selection',None) is not None:
        from femto.planning.patch_reference import create
        from femto.app.qt_planning import OUTPUTS_ROOT
        if restore_draft and journal_for(source,model).path.is_file():
            from copy import copy
            from femto.planning.planning_drafts import read
            from femto.annotation.selection_intent import decode_runs
            old = read(journal_for(source,model).path, model)
            masks = old.get('qt_masks')
            if masks:
                model = copy(model)
                model.pre_reconstruction_selection = decode_runs(masks['work'], masks['image_size'], values=(0,1)).astype(np.uint8)
                model.pre_reconstruction_forbidden = decode_runs(masks['forbidden'], masks['image_size'], values=(0,1)).astype(np.uint8)
                model.planning_parameters = old.get('qt_parameters')
        # Existing standard drafts remain intact. The same original pixel grid
        # lets this new version preserve the intent without resampling it.
        return create(model, Path(OUTPUTS_ROOT)/'planning_models'/uuid.uuid4().hex,
                      check=check, progress=progress)
    # Small imported/synthetic images would have too few measured rays after
    # stride-8 sampling. Keep their standard reference rather than making a
    # new capture unexpectedly unplannable. Explicit conversion remains opt-in.
    if min(model.rgb.shape[:2]) < 256:return model
    if restore_draft and journal_for(source,model).path.is_file():return model
    from femto.app.qt_planning import OUTPUTS_ROOT
    return create_reduced_model(model,Path(OUTPUTS_ROOT)/'planning_models'/uuid.uuid4().hex,
                                check=check,progress=progress)


def journal_for(source, model, root=None):
    """Keep standard drafts compatible and isolate each reduced-model profile."""
    from femto.planning.planning_drafts import Journal, ROOT
    root = Path(ROOT if root is None else root)
    reduced = derivation(model)
    if reduced:
        profile = str(reduced.get('profile') or DERIVATION_SCHEMA)
        key = 'reduced-' + hashlib.sha256(profile.encode('utf-8')).hexdigest()[:16]
        if root.name != key:
            root = root / key
    return Journal(source, root)


def can_create(page):
    model = page.model
    if (model is None or page.busy or page._stage != 0 or page.source is None
            or derivation(model) or page.region is not None or page.plan is not None
            or getattr(model, 'saved_plan', None) is not None):
        return False
    if any(getattr(model.store, name, None) for name in ('points', 'regions', '_history')):
        return False
    canvas = page.canvas
    return not (any(value is not None and np.any(value) for value in (canvas.work, canvas.forbidden))
                or canvas._undo or canvas._redo or canvas._before is not None)


def create_reduced_model(source, destination, *, check, progress, sampling_policy=None):
    # Defer the geometry backend import until the explicit user action.
    from femto.planning.planning_resolution import create_reduced_model as create
    from femto.planning.route_sampling import POLICY
    return create(source, destination, check=check, progress=progress,
                  route_sample_step_mm=5., sampling_policy=sampling_policy or POLICY)


def start_creation(page, *, new_version=False, sampling_policy=None):
    allowed = (page.model is not None and not page.busy and page.source is not None
               and not derivation(page.model)) if new_version else can_create(page)
    if not allowed:
        return False
    from femto.annotation.fixed_rgb_geometry import GeometryBudget
    from femto.app.qt_planning_io import clone_model
    from femto.app.region_quality import assess_selection
    from femto.app.qt_planning import OUTPUTS_ROOT
    from femto.planning.planning_drafts import read
    from PySide6.QtWidgets import QMessageBox

    source = page.source
    previous_journal = page.journal
    snapshot = clone_model(page.model, freeze_inputs=True)
    if new_version:
        # Only the candidate is cleared. Old source, selection, plan and draft
        # remain available until a complete new reference is installed.
        snapshot.store.points=[];snapshot.store.regions=[];snapshot.store._history=[]
        snapshot.saved_plan=None;snapshot.target_ledger=None;snapshot.coverage_certificate=None
    head = page.head.record()
    surface_patch = bool(new_version and (snapshot.store.local_context or {}).get('reconstruction_scope'))
    original_work = np.array(page.canvas.work,copy=True) if surface_patch else None
    original_forbidden = np.array(page.canvas.forbidden,copy=True) if surface_patch else None
    destination = Path(OUTPUTS_ROOT) / 'planning_models' / uuid.uuid4().hex
    page.draft_timer.stop()
    page.canvas.timer.stop()
    page.checkpoint()
    wait_for_draft = previous_journal is not None and (
        page.dirty or previous_journal.status() == 'saving')
    page.quality.invalidate('正在创建简化规划参考，完成后重新检查选区。')

    def job(cancel, progress):
        check = GeometryBudget(cancel).check
        if wait_for_draft:
            previous_journal.wait_saved(check)
        progress('创建简化规划参考；保留原三维网格和原始数据')
        options={'sampling_policy':sampling_policy} if sampling_policy is not None else {}
        if surface_patch:
            from femto.planning.patch_reference import create
            from femto.app.qt_planning_io import extract_region
            model = create(snapshot, destination, check=check, progress=progress)
            model, _ = extract_region(model, original_work, original_forbidden, '工作区域 01', '', cancel)
        else:
            model = create_reduced_model(snapshot, destination, check=check, progress=progress,**options)
        check()
        if derivation(model) is None:
            raise ValueError('简化规划模型缺少参考派生记录，未替换当前模型')
        if not surface_patch and (model.store.points or model.store.regions or getattr(model, 'saved_plan', None)):
            raise ValueError('简化规划模型仍包含旧标记或路线，未替换当前模型')
        empty = np.zeros(model.rgb.shape[:2], np.uint8)
        assessment = assess_selection(model, original_work if surface_patch else empty,
                                      original_forbidden if surface_patch else empty, cancel)
        check()
        root = previous_journal.path.parent if previous_journal is not None else None
        journal = journal_for(source, model, root)
        draft = None
        warning = ''
        if journal.path.is_file():
            try:draft = read(journal.path, model)
            except Exception as error:warning = '此前简化参考草稿未恢复，原草稿保留：' + str(error)
        check()
        return model, journal, assessment, draft, warning

    def done(result):
        model, journal, assessment, draft, warning = result
        page.install_model(model, source, journal, assessment=assessment)
        page.archive = None
        page.head.set_record(head)
        page.scene_panel.show_reference()
        if surface_patch:
            page.dirty=True;page.set_stage(1)
            page._status('原选区已保留，已创建粗分区规划版本；点击生成路线。')
        else:page._status(REDUCED_NOTICE+' 请在此参考上重新选区。')
        if draft and QMessageBox.question(page, '恢复简化参考草稿',
                '发现与此简化规划参考匹配的草稿，是否恢复区域和参数？\n恢复后需要重新计算路线。',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes:
            page.restore_draft(draft)
        if warning:page._status(warning)
        page.refresh()

    started = page._run(job, done)
    if not started:
        page.quality.request()
    return started
