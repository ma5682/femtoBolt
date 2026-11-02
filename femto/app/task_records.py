"""Revalidate historical task identity and file revision before installing a model."""
from pathlib import Path
from femto.dataset.task_context import checked_task
from femto.pipeline.local_history import existing_actions


def context_key(task):
    task=checked_task(task)
    return task['task_id'] if task else None


def verify_record_context(model, expected):
    """Recheck loaded content, not just the potentially stale index entry."""
    expected_key = context_key(expected['task'])
    if expected_key is None:
        raise ValueError('记录缺少可核验的任务标识，请从历史记录只读查看')
    if expected.get('capture_row') and not existing_actions(expected['capture_row'])['fixed']:
        raise ValueError('采集来源文件已变化或缺失；请刷新记录并检查来源')
    if context_key(getattr(model, 'task_context', None)) != expected_key:
        raise ValueError('记录的任务已变更；请刷新历史记录后重新选择')
    if expected.get('revision') and getattr(model, 'revision_id', None) != expected['revision']:
        raise ValueError('保存版本已变更；请刷新历史记录后重新选择')
    if expected.get('file_version'):
        stat = Path(expected['target']).stat()
        if (stat.st_size, stat.st_mtime_ns) != tuple(expected['file_version']):
            raise ValueError('方案文件在索引后发生变化；请刷新历史记录后重新选择')
