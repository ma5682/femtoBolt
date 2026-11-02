"""UI compatibility exports and actions for read-only reference quality."""
from femto.annotation.reference_quality import (
    assess_selection, require_geometry, LABELS, COLORS, LIMITS,
)

def findings(result):
    if not result or result['status'] == 'unavailable':
        return ['当前参考影像的选区几何支持尚未核验']
    if result['status'] == 'checked' and result['pending']:
        return [f"当前参考影像有 {result['pending']} 个选区像素缺失或待检查"]
    return []


def rescan_context(model):
    from femto.dataset.task_context import checked_task
    try:
        return checked_task(getattr(model, 'task_context', None))
    except (ValueError, TypeError, KeyError):
        return None
