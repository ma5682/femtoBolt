"""Optional task labels for derived UI artifacts, never patient identity proof."""
from datetime import datetime
from pathlib import Path
import json
import uuid

from femto.asset_storage import publish_json

SCHEMA = 'femto-ui-task-v2'


def new_task():
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    return dict(schema=SCHEMA,task_id=uuid.uuid4().hex,name='采集任务 '+now[:16].replace('T',' '),
                created_at=now)


def checked_task(value):
    """Validate task-only metadata; retired anatomy schemas are unsupported."""
    if value is None:return None
    required={'schema','task_id','name','created_at'}
    if (not isinstance(value,dict) or value.get('schema')!=SCHEMA
            or not required <= set(value) or set(value)-required-{'gap_followup'}):
        raise ValueError('任务信息格式无效')
    for field,limit in (('task_id',64),('name',120),('created_at',64)):
        if not isinstance(value.get(field),str) or len(value[field])>limit:
            raise ValueError('任务信息字段无效：'+field)
    if not value['task_id'] or not value['name'].strip():raise ValueError('任务标识和名称不能为空')
    result={key:value[key] for key in ('schema','task_id','name','created_at')}
    if 'gap_followup' in value:
        from femto.dataset.gap_followup import checked
        result['gap_followup']=checked(value['gap_followup'],value['task_id'])
    return result


def read_task(result):
    root=Path(result).resolve();path=root/'task_context.json'
    if not path.is_file():return None
    if path.stat().st_size>16*1024:raise ValueError('任务信息超过读取预算')
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict) or value.get('schema')!='femto-capture-task-link-v1':
        raise ValueError('采集任务标签格式无效')
    if value.get('capture_result')!=str(root):raise ValueError('任务信息与采集来源不匹配')
    return checked_task(value['task'])


def save_task(result, task):
    root=Path(result).resolve();path=root/'task_context.json'
    if path.exists():return read_task(root)
    task=checked_task(task)
    try:
        publish_json(path,dict(schema='femto-capture-task-link-v1',capture_result=str(root),task=task),
                     create_only=True)
    except FileExistsError:
        # Another caller may have bound this result after our initial read.
        return read_task(root)
    return task
