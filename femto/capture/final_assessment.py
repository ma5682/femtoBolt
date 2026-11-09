"""Assess final acquisition before publication; retain pending intent on manual end."""
from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np

from femto.annotation.fixed_rgb_geometry import FixedRGBModel, GeometryBudget
from femto.capture.scan_roi import assess_model
from femto.capture.scan_feedback import build_feedback, target_key
from femto.dataset.acquisition_completion import SCHEMA
from femto.io_utils import imwrite_unicode, load_json_object, sha256_file, write_json
from femto.operation_diagnostics import diagnosed, record_operation


@diagnosed('capture.final_assessment', operation=True)
def assess_final(capture, result):
    if capture.scan_roi is None:
        return None  # Existing non-ROI acquisition keeps its historical contract.
    root=Path(result)
    budget=GeometryBudget(capture.processing_cancel)
    budget.check()
    from femto.dataset.prepared_review import PreparedCapture, asset_fingerprints, close_prepared
    close_prepared(getattr(capture, 'prepared_review', None))
    capture.prepared_review = None
    try:
        before = asset_fingerprints(root, budget.check)
    except (OSError, ValueError, KeyError, TypeError):
        before = None  # Optional acceleration never changes acquisition admission.
    with record_operation('load.fixed_rgb_model'):
        model=FixedRGBModel(root/'reconstruction',cancel=capture.processing_cancel)
    with record_operation('load.final_reference_assessment'):
        summary,preview,classes=assess_model(model,capture.scan_roi,capture.processing_cancel,evidence=True)
    scope = load_json_object(root/'reconstruction/reconstruction_report.json').get('reconstruction_scope')
    target = dict(capture.scan_roi, reconstruction_scope=scope) if scope else capture.scan_roi
    binding=dict(target_key=target_key(target,capture.reference_provenance,model.view_reference),
        stage_id=uuid.uuid4().hex,session_id=capture.session_id,
        reference_frame_id=capture.scan_roi['reference_frame_id'],
        model_sha256=sha256_file(model.store.source,budget.check))
    feedback=build_feedback(classes,binding,capture.checkpoints.last_feedback,
                            raw_mapping=summary.get('raw_target_mapping'))
    if feedback['status']=='reference_support_met':
        end_status='reference_support_met'
    elif feedback['status']=='cannot_determine':
        end_status='manual_end_unknown'
    else:
        end_status='manual_end_pending'
    record=dict(schema=SCHEMA,binding=binding,assessment_status=feedback['status'],
        end_status=end_status,confirmed_for_planning=False,scan_summary=summary,
        operation_stop_reason=getattr(capture.checkpoints,'stop_reason',None),
        capture_end_reason=getattr(capture,'finish_reason',None),
        feedback=feedback,scan_roi=capture.scan_roi,reconstruction_scope=scope,
        accepted_frame_ids=[row['frame_id'] for row in capture.accepted],
        reconstruction_report_sha256=sha256_file(root/'reconstruction/reconstruction_report.json',budget.check),
        execution_ready=False,physical_accuracy_verified=False)
    budget.check()
    if classes is not None:np.save(root/'scan_diagnosis.npy',classes,allow_pickle=False)
    imwrite_unicode(root/'scan_check.png',preview[:,:,::-1])
    write_json(root/'acquisition_completion.json',record,allow_nan=False,check=budget.check,max_bytes=1024**2)
    from femto.capture.scan_exchange import task_batch
    write_json(root/'scan_tasks.json',task_batch(feedback,
        capture.reference_provenance['calibration']['sha256']),allow_nan=False,check=budget.check)
    # Persist the requirement only after the complete assessment exists. A
    # failed assessment cannot publish a model through the normal final path.
    manifest=load_json_object(root/'demo_summary.json')
    manifest['acquisition_completion_required']=True
    write_json(root/'demo_summary.json',manifest,allow_nan=False,check=budget.check)
    if before is not None:
        try:
            if asset_fingerprints(root, budget.check) == before:
                capture.prepared_review = PreparedCapture(root, model, before, record,
                                                          capture.scan_roi, summary, preview)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    budget.check()
    return record
