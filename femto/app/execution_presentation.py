"""Chinese summaries of recorded simulation metadata, without Qt or devices."""
from pathlib import Path



STATE_TEXT = {
    'unprepared': '尚未检查', 'checking': '检查中', 'ready_for_simulation': '可开始模拟',
    'running': '模拟运行中', 'pause_requested': '正在请求暂停', 'pause_confirmed': '暂停已确认',
    'resume_requested': '正在请求继续', 'stop_requested': '正在请求停止', 'stopped': '已停止',
    'cancelled': '已取消', 'abandoned_unrun': '已放弃、未运行', 'succeeded': '模拟步骤完成',
    'failed': '失败', 'result_unknown': '停止结果未知', 'interrupted_needs_review': '中断，待核对',
}
PLAN_SOURCE_TEXT = {'real_femto': '实采方案', 'synthetic_test': '合成软件测试', 'unknown': '来源未知'}
SOURCE_TEXT = {'real': '真实设备', 'simulation': '模拟设备', 'replay': '只读回放'}
READINESS_TEXT = {'ready': '就绪', 'not_ready': '未就绪', 'unknown': '未知'}
SPATIAL_TEXT = {'locked': '已锁定', 'unlocked': '已解锁', 'moving': '正在移动',
                'fixed_with_evidence': '固定结构，有安装证据', 'unknown': '未知', 'not_applicable': '不适用'}
HEAD_TEXT = {'placeholder_not_device_parameters': '占位参数，未作为设备参数验证',
             'user_entered_unverified': '人工输入，尚未验证'}
INTENT_TEXT = {'prepare': '准备', 'simulation_step': '模拟步骤', 'request_pause': '请求暂停',
               'request_resume': '请求继续', 'request_stop': '请求停止', 'cancel_request': '取消请求',
               'query_status': '查询状态', 'query_capabilities': '查询能力'}
EVENT_TEXT = {
    'task_frozen': '任务已冻结', 'task_connections_bound': '已绑定设备连接',
    'task_abandoned_unrun': '撤销冻结：已放弃、未运行',
    'preflight_started': '开始前置检查', 'preflight_result': '前置检查结果',
    'preflight_failed': '前置检查未通过', 'prepare_confirmed': '准备已确认',
    'operator_start': '开始模拟', 'operator_pause': '请求暂停', 'pause_all_confirmed': '暂停已确认',
    'operator_resume': '请求继续', 'resume_all_confirmed': '继续已确认',
    'operator_stop': '请求停止', 'stop_all_confirmed': '全部停止及输出关闭已确认',
    'stop_result_unknown': '停止结果未知', 'step_confirmed': '模拟步骤已确认',
    'simulation_step_confirmed': '模拟步骤已确认',
    'operator_cancel_unprepared': '尚未准备的任务已取消',
    'completion_requirements_failed': '完成条件未满足', 'completion_not_persisted': '完成状态未保存',
    'request_intent': '记录设备请求', 'request_sent': '设备请求已发送',
    'request_send_failed': '设备请求发送失败', 'device_feedback': '收到设备反馈',
    'feedback_rejected': '拒绝无效反馈', 'status_rejected': '拒绝无效状态',
    'device_status': '设备状态更新', 'simulation_instances_closed': '旧模拟实例已关闭',
    'persistence_gap_recovered': '已记录持久化缺口', 'interrupted_recovery': '中断恢复，待核对',
    'critical_queue_gap': '关键事件队列存在缺口', 'late_device_alarm': '结束后设备报警',
    'late_physical_state_alarm': '结束后物理状态矛盾', 'late_feedback_alarm': '结束后反馈矛盾',
    'stop_outcome_invalidated': '原结束结果已撤销', 'device_feedback_alarm': '设备反馈报警',
    'resource_busy': '设备资源被占用', 'request_timeout': '设备请求超时',
    'resource_reacquire_failed': '模拟资源保留失败', 'simulation_instances_close_failed': '旧模拟实例关闭失败',
}
ROLE_TEXT = {'scan': '扫描段', 'connector': '连接段', 'gap': '断点'}
PHASE_TEXT = {'received': '已收到请求', 'completed': '已完成请求', 'failed': '请求失败',
              'cancelled': '请求已取消', 'accepted': '已接纳'}


def plan_fields(summary):
    summary = summary or {}
    task = summary.get('task_context') or {}
    path = summary.get('source_path')
    fallback = Path(path).name if path else '合成软件测试方案' if summary.get('plan_source') == 'synthetic_test' else '未知'
    return dict(name=task.get('name') or fallback, task=task.get('name') or '未知',
                revision=summary.get('revision_id') or '未知',
                source=PLAN_SOURCE_TEXT.get(summary.get('plan_source'), '来源未知'))


def plan_text(summary, *, prefix='方案'):
    fields = plan_fields(summary)
    return (f"{prefix}：{fields['name']}　·　版本：{fields['revision']}　·　{fields['source']}\n"
            f"任务：{fields['task']}")



def event_reason(snapshot):
    if snapshot.get('last_error'):
        return str(snapshot['last_error'])
    preflight = snapshot.get('preflight') or {}
    failed = [check['name']+'：'+check['reason'] for check in preflight.get('checks', [])
              if check.get('scope') == 'simulation' and check.get('status') in ('failed', 'unknown')]
    if failed:
        return '；'.join(failed[:3])
    for event in reversed(snapshot.get('events', [])):
        if event.get('payload', {}).get('reason'):
            return str(event['payload']['reason'])
    return '请查看检查结果和关键事件。'


def anomaly(snapshot, action_error=None):
    value = snapshot or {}
    if value.get('persistence_gap'):
        return ('事件记录存在缺口', str(value.get('persistence_error') or '事件未完整保存'),
                '后续步骤已阻止；停止与反馈处理继续。',
                '恢复记录可写后重试记录并标记缺口，原缺失事件不会补造。')
    state = value.get('state')
    if state in ('result_unknown', 'interrupted_needs_review'):
        return (STATE_TEXT[state], event_reason(value), '原结果需核对，相关资源不能用于新的控制动作。',
                '先查看工程详情；仅模拟实例可显式关闭后建立独立实例，旧未知结果保留。')
    if state == 'failed':
        return ('模拟任务失败', event_reason(value), '当前任务不能继续，已发生的步骤和原方案保留。',
                '查看失败项，修正来源或配置后重新建立任务；需要更换实例时显式关闭旧实例。')
    if action_error:
        return ('操作未完成', str(action_error), '本次操作未取得有效结果，原任务或记录保持。',
                '核对来源与错误详情，再重试当前操作。')
    return None


def history_text(record):
    saved = record.get('snapshot', {}).get('plan', {})
    plan = saved.get('plan', {})
    summary = dict(plan, plan_source=saved.get('plan_source'))
    integrity = record.get('source_integrity') or {}
    labels = {'verified': '来源摘要核对通过', 'changed_or_missing': '来源变化或缺失，未重新绑定',
              'not_checked': '来源尚未核对', 'not_applicable': '无外部来源需要核对'}
    count = len(saved.get('steps', []))
    failure = event_reason(record)
    state = record.get('display_state')
    text = (f"只读模拟记录　·　{STATE_TEXT.get(state, '状态未知')}\n"
            f"记录 ID：{record.get('run_id', '未知')}\n{plan_text(summary)}\n"
            f"已确认软件步骤：{record.get('completed_steps', 0)} / {count}\n"
            f"来源核对：{labels.get(integrity.get('status'), '未知')}\n"
            '模拟记录不代表治疗、物理精度或当前患者状态。')
    if state in ('failed', 'result_unknown', 'interrupted_needs_review'):
        text += '\n异常原因：'+failure
    if plan.get('gap_report'):
        text += '\n原方案含缺口或覆盖检查记录，模拟完成不改变原几何与覆盖结论。详情见工程详情。'
    if integrity.get('failures'):
        text += '\n来源问题：'+'；'.join(str(item.get('reason')) for item in integrity['failures'][:3])
    backup = record.get('backup_context')
    if backup:
        text += '\n备份核对：'+str(backup.get('reason', '未知'))
    return text
