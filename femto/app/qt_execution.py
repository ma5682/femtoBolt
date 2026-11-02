"""Device/simulation page. One non-daemon owner serializes commands and events."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
import queue
import threading
import time
import uuid

from PySide6.QtCore import QObject, QTimer, Qt, Signal, Slot, QSize
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget, QScrollArea,
)

from femto.app.qt_common import TaskRunner, show_text
from femto.app.qt_icons import line_icon
from femto.app.execution_presentation import (
    STATE_TEXT, SOURCE_TEXT, READINESS_TEXT, SPATIAL_TEXT, HEAD_TEXT,
    INTENT_TEXT, EVENT_TEXT, ROLE_TEXT, PHASE_TEXT, plan_fields, plan_text, history_text, anomaly,
)
from femto.dataset.installation_calibration import evidence_status, load_record as load_calibration, save_record as save_calibration
from femto.execution.application import create_coordinator, synthetic_gap_demo
from femto.execution.history import ExecutionStore
from femto.execution.models import gap_summary, load_saved_plan, plan_summary, synthetic_plan
from femto.execution.state_machine import RunState, TERMINAL
from femto.hardware.configuration import load_configuration, save_configuration
from femto.hardware.simulation.scenario import MonotonicClock, Scenario, load_scenario, save_scenario
from femto.paths import OUTPUTS_ROOT

KIND_TEXT = {'arm':'机械臂','base':'底座','treatment_head':'治疗头','ultrasound_generator':'超声发生器'}
CHECK_TEXT = {'passed':'通过','failed':'未通过','not_applicable':'不适用','unknown':'未知'}
RECORD_ACTIONS = frozenset({'history', 'replay', 'export'})
LIVE_ACTIONS = frozenset({'preview_plan', 'freeze', 'connect', 'check', 'start', 'pause', 'resume', 'stop',
                          'abandon', 'reset', 'recover_log'})
TERMINAL_STATES = frozenset(state.value for state in TERMINAL)


def _physical_text(value, *, applicable=True):
    return ('不适用' if not applicable else
            '已确认' if value is True else '未停止 / 活动' if value is False else '未知')


def _device_rows(device):
    configuration, status = device['configuration'], device['status']
    binding = status['binding']
    title = configuration.get('model') or binding['device_id']
    kind = binding['kind']
    rows = [('设备', KIND_TEXT.get(kind, kind)),
            ('来源 / 连接', SOURCE_TEXT.get(binding['source'], '来源未知')+' · '+('已连接' if status['connected'] else '未连接')),
            ('就绪状态', READINESS_TEXT.get(status['readiness'], '未知')), ('故障', status['fault'] or '无报告'),
            ('机械停止', _physical_text(status['motion_stopped'], applicable=device.get('motion_applicable', kind in ('arm', 'base')))),
            ('输出关闭', _physical_text(status['output_off'], applicable=device.get('output_applicable', kind in ('treatment_head', 'ultrasound_generator')))),
            ('底座空间状态', SPATIAL_TEXT.get(status['spatial_state'], '未知') if kind == 'base' else '不适用'),
            ('状态时间', f"{status['timestamp_s']:.3f} s · 模拟时钟"),
            ('能力', '、'.join(INTENT_TEXT.get(intent, intent) for intent in device['capabilities']) or '空')]
    return ' '.join(str(title).split()), [(label, ' '.join(str(value).split())) for label, value in rows]


def _device_html(sections):
    parts=[]
    for title, rows in sections:
        parts.append('<h3>'+escape(str(title))+'</h3>'+'<br>'.join(
            '<b>'+escape(label)+'：</b>'+escape(str(value)) for label,value in rows))
    return ''.join(parts) or '<p>尚未配置或连接设备；能力为空，物理状态未知。</p>'


class DeviceDetails(QTextBrowser):
    """Update device values without replacing the document being read."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._sections = None
        self._bindings = None

    def update_devices(self, devices):
        sections = [_device_rows(device) for device in devices]
        bindings = [device['status']['binding'] for device in devices]
        if (self._sections is None or bindings != self._bindings
                or [title for title, _ in sections] != [title for title, _ in self._sections]):
            self.setHtml(_device_html(sections))
        elif sections != self._sections:
            document = self.document()
            search = QTextCursor(document)
            changes = []
            for (_, old_rows), (_, new_rows) in zip(self._sections, sections):
                for (label, old), (_, new) in zip(old_rows, new_rows):
                    found = document.find(label+'：'+old, search)
                    if found.isNull():
                        # A replaced document or changed Qt text normalization
                        # must not prevent new device state from being shown.
                        self.setHtml(_device_html(sections))
                        self._sections, self._bindings = sections, bindings
                        return
                    search = QTextCursor(found)
                    if old != new:
                        end = found.selectionEnd()
                        changes.append((end-len(old.encode('utf-16-le'))//2, end, new))
            cursor = self.textCursor()
            vertical, horizontal = self.verticalScrollBar().value(), self.horizontalScrollBar().value()
            edit = QTextCursor(document)
            edit.beginEditBlock()
            # Reverse order keeps saved UTF-16 positions valid. Qt moves the
            # reader's selection when text before it changes length.
            for start, end, text in reversed(changes):
                edit.setPosition(start)
                edit.setPosition(end, QTextCursor.KeepAnchor)
                edit.insertText(text, edit.charFormat())
            edit.endEditBlock()
            self.setTextCursor(cursor)
            self.verticalScrollBar().setValue(vertical)
            self.horizontalScrollBar().setValue(horizontal)
        self._sections, self._bindings = sections, bindings


def _devices_connected(snapshot):
    devices = snapshot.get('devices', [])
    return bool(devices) and all(device['status']['connected'] for device in devices)


def _check_html(preflight, plan):
    parts=[]
    if preflight:
        for scope,title in (('simulation','模拟软件检查'),('real_execution','真实执行待接条件')):
            parts.append('<h3>'+title+'</h3>')
            for check in preflight['checks']:
                if check['scope']==scope:
                    parts.append('<p><b>'+escape(CHECK_TEXT.get(check['status'],check['status'])+' · '+check['name'])+
                                 '</b><br>'+escape(check['reason'])+'</p>')
    else:
        parts.append('<p>尚未检查。核验并冻结方案、连接设备后进行模拟检查。</p>')
    if plan:
        parts.append('<h3>原方案信息</h3><p>'+escape(plan_text(plan)).replace('\n','<br>')+'</p>')
        head=plan.get('treatment_head') or {}
        parts.append('<p>治疗头参数状态：'+escape(HEAD_TEXT.get(head.get('values_status'), '未知'))+
                     '<br>工具姿态：未提供；几何与覆盖结论保持原记录。</p>')
        if plan.get('gap_report'):
            parts.append('<p>原方案包含缺口或覆盖检查记录；模拟完成不改变原结论。完整记录可在工程详情查看。</p>')
    return ''.join(parts)


def _event_text(event):
    payload=event['payload'];details=[]
    if payload.get('reason'):details.append(str(payload['reason']))
    message=payload.get('feedback') or payload.get('request') or payload.get('status')
    if message:
        binding=message.get('binding',{})
        details.append(KIND_TEXT.get(binding.get('kind'), '设备')+' · '+SOURCE_TEXT.get(binding.get('source'), '来源未知'))
        if message.get('intent'):details.append(INTENT_TEXT.get(message['intent'], message['intent']))
        if message.get('phase'):details.append(PHASE_TEXT.get(message['phase'], message['phase']))
        if message.get('error_code'):details.append(message['error_code'])
    if payload.get('step'):
        step=payload['step'];details.append('步骤 '+str(step['index'])+' · '+ROLE_TEXT.get(step['role'], step['role'])+((' · '+step['reason']) if step.get('reason') else ''))
    if payload.get('from_state')!=payload.get('to_state'):
        details.append(STATE_TEXT.get(payload['from_state'],payload['from_state'])+' → '+STATE_TEXT.get(payload['to_state'],payload['to_state']))
    return f"{payload['local_sequence']}. {EVENT_TEXT.get(event['kind'],'设备事件')}"+(' · '+' · '.join(details) if details else '')+(' [未落盘]' if not event['persisted'] else '')


class SimulationOwner(QObject):
    changed = Signal(str)
    notice = Signal(str)
    operation_failed = Signal(str, str)
    replay_loaded = Signal(str)
    details_loaded = Signal(str)
    plan_loaded = Signal(str)
    cleared = Signal()
    finished = Signal()

    def __init__(self, *, outputs, parent=None):
        super().__init__(parent)
        self.outputs = Path(outputs)
        self._commands = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._operator_stop = threading.Event()
        self._wake = threading.Event()
        self._cancel_load = threading.Event()
        self._overflow = threading.Event()
        self._thread = None
        self._closing = False
        self.busy = False
        self.loading = False
        self._latest = None
        self._display_lock = threading.Lock()
        self._display_pending = False
        self._revision = 0
        self._command_lock = threading.Lock()
        self._pending_actions = set()
        self._replay_revision = 0
        self._ready_revision = -1
        self._cleanup_failed = False
        self._clear_pending = False
        self.coordinator = None  # Owned exclusively by the worker once started.

    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def stop_pending(self):
        return self._operator_stop.is_set()

    def take_snapshot(self):
        with self._display_lock:
            result = self._latest
            self._display_pending = False
        return result

    def submit(self, action, payload=None):
        if self._closing or self._cleanup_failed:
            return False
        if action in RECORD_ACTIONS:
            raise ValueError('记录读取与导出必须使用独立的可取消后台任务')
        with self._command_lock:
            if action in self._pending_actions:
                return False
            self._pending_actions.add(action)
        cancel = None
        if action in ('freeze', 'preview_plan'):
            cancel = threading.Event()
            self._cancel_load = cancel
        replay_revision = None
        if action in ('preview_plan','discard_candidate','freeze','connect','abandon','reset','configuration','scenario'):
            self._replay_revision += 1
            replay_revision = self._replay_revision
        if action in ('discard_candidate','reset','configuration','scenario'):
            self._cancel_load.set()
        if action in ('stop', 'cancel_load'):
            self._cancel_load.set()
        try:
            if action == 'stop':
                self._operator_stop.set()
            else:
                self._commands.put_nowait((action, payload, cancel, replay_revision))
        except queue.Full:
            with self._command_lock: self._pending_actions.discard(action)
            self._overflow.set(); self._wake.set()
            self.notice.emit('操作队列已满，停止接纳新步骤并请求停止；结果需核对。')
            return False
        self.busy = True
        if not self.alive:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name='SimulationOwner', daemon=False)
            self._thread.start()
        self._wake.set()
        return True

    def _clear(self):
        with self._display_lock:
            self._latest = None
            self._display_pending = False
        self._ready_revision = -1
        self._clear_pending = True

    def _publish(self, force=False):
        if self.coordinator is None:
            return
        revision = (self.coordinator.event_count, self.coordinator.state, self.coordinator.completed_steps,
                    self.coordinator.status_revision)
        if not force and revision == self._ready_revision:
            return
        self._ready_revision = revision
        snapshot = self.coordinator.snapshot()
        with self._display_lock:
            self._latest = snapshot
            emit = not self._display_pending
            self._display_pending = True
        if emit: self.changed.emit('snapshot_available')

    def _run(self):
        clock = MonotonicClock()
        scenario = Scenario()
        configurations = None
        candidate = None
        candidate_token = None
        shutdown_started = None
        last_display = 0.
        def read_check(revision=None):
            if self._stop.is_set() or revision is not None and revision != self._replay_revision:
                raise RuntimeError('读取已取消或记录选择已切换')
            if self._operator_stop.is_set():
                raise RuntimeError('已请求停止，当前来源核验已取消')
        try:
            while True:
                if self._stop.is_set():
                    if shutdown_started is None:
                        shutdown_started = time.monotonic()
                        self._cancel_load.set()
                        if self.coordinator: self.coordinator.request_shutdown()
                    if (self.coordinator is None or self.coordinator.state in TERMINAL
                            or self.coordinator.run_id is None):
                        break
                    if time.monotonic()-shutdown_started >= scenario.shutdown_timeout_s:
                        self.coordinator.shutdown_expired()
                        break
                if self._operator_stop.is_set():
                    try:
                        if self.coordinator and self.coordinator.run_id:
                            self.coordinator.stop()
                    finally:
                        self._operator_stop.clear()
                        with self._command_lock:
                            self._pending_actions.discard('stop')
                            self.busy = bool(self._pending_actions)
                        self._publish(force=True)
                if self._overflow.is_set():
                    self._overflow.clear()
                    if self.coordinator and self.coordinator.run_id:
                        self.coordinator.critical_queue_gap('关键操作队列溢出')
                    # Already queued controls cannot run after a known gap.
                    while True:
                        try:
                            discarded = self._commands.get_nowait()
                            with self._command_lock: self._pending_actions.discard(discarded[0])
                        except queue.Empty:
                            break
                    self.busy = False
                try:
                    action, payload, cancel, replay_revision = self._commands.get_nowait()
                except queue.Empty:
                    action = None
                if action is not None:
                    try:
                        if self._stop.is_set():
                            continue
                        if action in ('scenario', 'configuration'):
                            if self.coordinator and self.coordinator.run_id and self.coordinator.state not in TERMINAL:
                                raise RuntimeError('请先停止任务，再更换场景或设备配置')
                            if action == 'scenario':
                                replacement = load_scenario(payload) if payload else Scenario()
                            else:
                                paths = (payload,) if isinstance(payload, (str, Path)) else tuple(payload)
                                if not paths:
                                    raise ValueError('请选择至少一份设备配置')
                                replacement = tuple(load_configuration(path) for path in paths)
                                if len({c.device_id for c in replacement}) != len(replacement):
                                    raise ValueError('所选配置包含重复设备身份，请保留每台设备的一个修订')
                                if len({c.resource_id for c in replacement}) != len(replacement):
                                    raise ValueError('所选设备配置重复占用同一资源')
                            if self.coordinator: self.coordinator.close_instances(); self.coordinator = None
                            self._clear()
                            candidate = candidate_token = None
                            if action == 'scenario': scenario, configurations = replacement, None
                            else: configurations = replacement
                            self.details_loaded.emit(json.dumps(dict(scenario=scenario.to_dict(),
                                configurations=[c.to_dict() for c in configurations] if configurations else None,
                                connected=False, reason='真实/回放适配器尚未实现；未配置设备能力为空，状态未知'), ensure_ascii=False))
                            self.notice.emit('配置已读取；设备尚未连接。真实适配器未实现时将明确拒绝连接。')
                        elif action == 'discard_candidate':
                            candidate = candidate_token = None
                        elif action in ('preview_plan', 'freeze', 'connect'):
                            if action == 'preview_plan' and self.coordinator and self.coordinator.run_id and self.coordinator.state not in TERMINAL:
                                raise RuntimeError('当前任务尚未结束；请先撤销冻结或请求停止')
                            if action != 'preview_plan' and self.coordinator is None:
                                self.coordinator = create_coordinator(outputs=self.outputs, scenario=scenario,
                                                                      clock=clock, device_configurations=configurations)
                            if action in ('preview_plan', 'freeze'):
                                self.loading = True
                                def verify_check():
                                    read_check(replay_revision)
                                    if cancel.is_set():raise RuntimeError('方案核验已取消，未建立任务')
                                self.notice.emit('正在核验方案摘要；可取消读取。' if action == 'preview_plan' else '正在核验并冻结方案；可取消读取。')
                                if payload.get('candidate_token'):
                                    if candidate is None or candidate_token != payload['candidate_token']:
                                        raise RuntimeError('已核验的候选方案已变化，请重新选择核验')
                                    plan = candidate
                                    plan.verify_assets(verify_check)
                                else:
                                    plan = (synthetic_plan() if payload['synthetic'] else
                                            load_saved_plan(payload['path'], cancel=cancel, progress=self.notice.emit))
                                saved = plan.public_plan
                                for expected, key in (('expected_revision', 'revision_id'), ('expected_result_sha256', 'result_sha256')):
                                    if payload.get(expected) is not None and saved.get(key) != payload[expected]:
                                        raise ValueError('已保存方案版本或路线摘要已变化，请回看原方案后重新选择')
                                if (cancel.is_set() or self._stop.is_set()
                                        or replay_revision != self._replay_revision):
                                    raise RuntimeError('方案核验已取消或输入已切换，未建立任务')
                                def freeze_check():
                                    read_check(replay_revision)
                                    if cancel.is_set(): raise RuntimeError('方案核验已取消，未建立任务')
                                if action == 'preview_plan':
                                    freeze_check()
                                    candidate, candidate_token = plan, payload['token']
                                    self.plan_loaded.emit(json.dumps(dict(token=candidate_token,
                                        summary=plan_summary(plan)), ensure_ascii=False))
                                    self.notice.emit('方案已核验。请核对任务、来源与版本，再冻结模拟任务。')
                                else:
                                    self.coordinator.freeze(plan, check=freeze_check)
                                    candidate = candidate_token = None
                                    self.notice.emit('模拟任务已冻结。')
                            else:
                                self.coordinator.connect()
                        elif action in ('check', 'start', 'pause', 'resume', 'stop', 'recover_log', 'abandon'):
                            if self.coordinator is None: raise RuntimeError('请先冻结方案并连接模拟设备')
                            if action == 'check' and (not self.coordinator.adapters or
                                    not all(adapter.status().connected for adapter in self.coordinator.adapters)):
                                raise RuntimeError('请先连接全部参与的模拟设备，再进行模拟前置检查')
                            method = {'recover_log':'recover_persistence', 'abandon':'abandon_unrun'}.get(action, action)
                            options = dict(check=read_check) if action in ('check','start','resume') else {}
                            if not getattr(self.coordinator, method)(**options):
                                self.notice.emit('当前状态或能力不允许此操作，请查看检查与事件。')
                            elif action == 'abandon':
                                candidate = candidate_token = None
                                self.notice.emit('冻结已撤销：已放弃、未运行。可以改选方案，原记录保留。')
                        elif action == 'reset':
                            if (self.coordinator and self.coordinator.run_id
                                    and self.coordinator.state not in TERMINAL
                                    and self.coordinator.state is not RunState.UNPREPARED):
                                raise RuntimeError('请先请求停止并等待结果，再关闭旧模拟实例')
                            if self.coordinator: self.coordinator.close_instances(); self.coordinator = None
                            self._clear()
                            candidate = candidate_token = None
                            self.notice.emit('旧模拟实例已关闭；未知结果保留。可建立独立的新实例。')
                        elif action == 'save_config':
                            if self.coordinator is None: raise RuntimeError('尚无模拟配置')
                            for adapter in self.coordinator.adapters:
                                save_configuration(adapter.configuration, outputs=self.outputs)
                            save_scenario(scenario, outputs=self.outputs)
                            self.notice.emit('设备配置与场景已按修订保存；再次保存同修订会被拒绝。')
                        elif action == 'calibration':
                            record = load_calibration(payload)
                            saved = save_calibration(record, outputs=self.outputs)
                            self.details_loaded.emit(json.dumps(dict(installation_record=record.to_dict(),
                                evidence_status=evidence_status(record)), ensure_ascii=False))
                            self.notice.emit('安装记录已保存：'+str(saved)+'；仅结构校验，不取得实测资格。')
                        elif action == 'cancel_load':
                            pass
                        elif action == 'gap_demo':
                            self.replay_loaded.emit(json.dumps(dict(gap_demo=synthetic_gap_demo()), ensure_ascii=False))
                    except Exception as error:
                        if action in ('preview_plan','freeze'):candidate = candidate_token = None
                        self.operation_failed.emit(action, '操作失败：'+str(error))
                    finally:
                        self.loading = False
                        with self._command_lock:
                            self._pending_actions.discard(action)
                            self.busy = bool(self._pending_actions)
                        if self._clear_pending:
                            self._clear_pending = False
                            self.cleared.emit()
                        self._publish(force=True)
                        self.changed.emit('operation_finished')
                if self.coordinator:
                    try:
                        self.coordinator.poll()
                    except Exception as error:
                        self.notice.emit('设备事件处理失败：'+str(error))
                        if self.coordinator.run_id and self.coordinator.state not in TERMINAL:
                            self.coordinator.stop(reason=str(error), outcome=RunState.FAILED)
                    if time.monotonic()-last_display >= .1:
                        self._publish()
                        last_display = time.monotonic()
                self._wake.wait(scenario.tick_s)
                self._wake.clear()
        except Exception as error:
            self.notice.emit('模拟所有者异常结束：'+str(error)+'；运行结果需核对。')
            if self.coordinator and self.coordinator.run_id and self.coordinator.state not in TERMINAL:
                self.coordinator.shutdown_expired()
        finally:
            try:
                if self.coordinator:
                    self.coordinator.close_instances()
            except Exception as error:
                self._cleanup_failed = True
                self.notice.emit('模拟实例清理失败：'+str(error)+'；旧资源和结果需核对。')
            finally:
                try:
                    if self.coordinator: self._publish(force=True)
                except Exception as error:
                    self.notice.emit('退出状态显示失败：'+str(error))
                finally:
                    self.busy = self.loading = False
                    self._operator_stop.clear()
                    with self._command_lock: self._pending_actions.clear()
                    self.finished.emit()

    def cancel_loading(self):
        self._cancel_load.set()

    def shutdown(self):
        self._closing = True
        self._stop.set(); self._cancel_load.set(); self._wake.set()
        return not self.alive


class ExecutionPage(QWidget):
    def __init__(self, parent=None, *, outputs=None):
        super().__init__(parent)
        self.setObjectName('executionPage')
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.outputs = Path(outputs if outputs is not None else OUTPUTS_ROOT)
        self.source = None
        self.model = None
        self._snapshot = None
        self._replay = False
        self._selected_run = None
        self._event_follow = True
        self._guidance_key = None
        self._closing = False
        self._history_cursor = None
        self._history_has_more = False
        self._candidate_token = None
        self._candidate_summary = None
        self._candidate_pending = None
        self._preview_inputs = None
        self._action_error = None
        self._failure_action = None
        self._engineering_record = None
        self._installation_details = None
        self._primary_action = None
        self._plan_context_text = None
        self.owner = SimulationOwner(outputs=self.outputs, parent=self)
        self.record_runner = TaskRunner(self)
        self.record_runner.finished.connect(self._buttons)
        self.owner.changed.connect(self._changed, Qt.QueuedConnection)
        self.owner.notice.connect(self._notice, Qt.QueuedConnection)
        self.owner.operation_failed.connect(self._operation_failed, Qt.QueuedConnection)
        self.owner.replay_loaded.connect(self._show_replay, Qt.QueuedConnection)
        self.owner.details_loaded.connect(self._show_details, Qt.QueuedConnection)
        self.owner.plan_loaded.connect(self._plan_loaded, Qt.QueuedConnection)
        self.owner.cleared.connect(self._clear_snapshot, Qt.QueuedConnection)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 18)
        title = QLabel('设备管理与模拟联调'); title.setObjectName('pageTitle')
        heading = QHBoxLayout()
        emblem = QLabel();emblem.setPixmap(line_icon('device', True).pixmap(44,44))
        heading.addWidget(emblem);heading.addWidget(title);heading.addStretch()
        layout.addLayout(heading)
        self.banner = QLabel('模拟联调，无真实运动及超声输出。真实设备协议、安装标定及物理精度尚未验证。')
        self.banner.setWordWrap(True); self.banner.setObjectName('workflowHint'); layout.addWidget(self.banner)
        self.workflow = QFrame(); self.workflow.setObjectName('simulationWorkflow')
        steps = QHBoxLayout(self.workflow); steps.setContentsMargins(8, 4, 8, 4)
        self.step_labels = []
        for index, name in enumerate(('选择方案', '冻结', '连接', '检查', '运行')):
            label = QLabel(f'{index+1:02d}  {name}'); label.setObjectName('simulationStep')
            label.setProperty('state', 'idle'); steps.addWidget(label, 1); self.step_labels.append(label)
        layout.addWidget(self.workflow)
        self.control_bar = QFrame(); self.control_bar.setObjectName('simulationControls')
        controls = QVBoxLayout(self.control_bar); controls.setContentsMargins(12, 8, 12, 8)
        state_row = QHBoxLayout()
        self.state = QLabel('当前模拟：尚未建立任务'); self.state.setObjectName('simulationState')
        self.state.setTextFormat(Qt.PlainText); state_row.addWidget(self.state, 2)
        self.progress = QProgressBar(); self.progress.setRange(0, 1); self.progress.setValue(0)
        self.progress.setFormat('确认的软件步骤：%v / %m'); state_row.addWidget(self.progress, 1)
        controls.addLayout(state_row)
        buttons = QHBoxLayout()
        self.connect_button = QPushButton('连接模拟设备')
        self.check_button = QPushButton('模拟前置检查'); self.start_button = QPushButton('开始模拟')
        self.pause_button = QPushButton('暂停'); self.resume_button = QPushButton('继续'); self.stop_button = QPushButton('请求停止')
        self.stop_button.setObjectName('simulationStop')
        self.abandon_button = QPushButton('撤销冻结')
        for button, action in ((self.connect_button,'connect'), (self.check_button,'check'), (self.start_button,'start'),
                               (self.pause_button,'pause'), (self.resume_button,'resume'), (self.stop_button,'stop'),
                               (self.abandon_button,'abandon')):
            buttons.addWidget(button); button.clicked.connect(lambda checked=False,a=action:self.command(a))
        self.result_button = QPushButton('查看本次结果'); self.result_button.clicked.connect(self.show_result)
        buttons.addWidget(self.result_button)
        controls.addLayout(buttons); layout.addWidget(self.control_bar)
        self.exception_box = QFrame(); self.exception_box.setObjectName('simulationException')
        problem_layout = QVBoxLayout(self.exception_box); problem_layout.setContentsMargins(12, 8, 12, 8)
        self.exception_title = QLabel(); self.exception_title.setObjectName('simulationExceptionTitle')
        self.exception_message = QLabel(); self.exception_message.setWordWrap(True)
        self.exception_message.setTextFormat(Qt.PlainText)
        self.exception_message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        problem_layout.addWidget(self.exception_title); problem_layout.addWidget(self.exception_message)
        actions = QHBoxLayout()
        self.exception_details = QPushButton('查看异常详情'); self.exception_details.clicked.connect(self.show_engineering)
        self.exception_reset = QPushButton('关闭旧模拟实例'); self.exception_reset.clicked.connect(self.reset_instances)
        self.exception_recover = QPushButton('重试记录并标记缺口'); self.exception_recover.clicked.connect(lambda:self.command('recover_log'))
        self.exception_retry = QPushButton('重试方案核验'); self.exception_retry.clicked.connect(self.retry_plan)
        for button in (self.exception_details,self.exception_reset,self.exception_recover,self.exception_retry):actions.addWidget(button)
        actions.addStretch(); problem_layout.addLayout(actions)
        layout.addWidget(self.exception_box); self.exception_box.hide()
        self.tabs = QTabWidget(); self.tabs.setObjectName('simulationTabs'); layout.addWidget(self.tabs, 1)
        live = QWidget(); live_layout = QVBoxLayout(live)
        source_box = QGroupBox('模拟方案与场景')
        source_box.setObjectName('simulationSource')
        form = QFormLayout(source_box)
        self.mode = QComboBox(); self.mode.addItems(['已保存方案（只读核验）', '显式合成软件测试'])
        form.addRow('方案来源', self.mode)
        row = QHBoxLayout(); self.path = QLineEdit(); self.path.setPlaceholderText('选择已保存 .region.zip 或 trajectory.json')
        self.choose = QPushButton('选择方案'); self.choose.clicked.connect(self._choose_plan)
        row.addWidget(self.path, 1); row.addWidget(self.choose); form.addRow('保存方案', row)
        row = QHBoxLayout()
        self.preview_button = QPushButton('核验方案摘要'); self.preview_button.clicked.connect(self.preview_plan)
        self.freeze_button = QPushButton('核验并冻结模拟任务'); self.freeze_button.clicked.connect(self.freeze)
        self.config_button = QPushButton('配置与场景'); self.config_button.clicked.connect(self._configuration)
        self.cancel_button = QPushButton('取消读取 / 导出'); self.cancel_button.clicked.connect(self.cancel_reading)
        for button in (self.preview_button,self.freeze_button,self.config_button,self.cancel_button): row.addWidget(button)
        form.addRow(row)
        self.plan_context = QLabel('尚未核验方案。选择方案后核对任务、来源与版本；缺失信息显示未知。')
        self.plan_context.setObjectName('simulationPlanSummary'); self.plan_context.setWordWrap(True)
        self.plan_context.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.plan_context.setMinimumHeight(62)
        self.plan_context.setTextFormat(Qt.PlainText); self.plan_context.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow(self.plan_context); live_layout.addWidget(source_box)
        self.reset_button = QPushButton('关闭旧模拟实例'); self.recover_button = QPushButton('重试记录并标记缺口')
        self.recovery_controls = QWidget()
        management = QHBoxLayout(self.recovery_controls)
        management.setContentsMargins(0, 0, 0, 0)
        management.addWidget(QLabel('实例与记录恢复'))
        management.addWidget(self.reset_button)
        management.addWidget(self.recover_button)
        self.reset_button.clicked.connect(self.reset_instances)
        self.recover_button.clicked.connect(lambda:self.command('recover_log'))
        management.addStretch();live_layout.addWidget(self.recovery_controls)
        self.engineering_button = QPushButton('工程详情'); self.engineering_button.clicked.connect(self.show_engineering)
        management.addWidget(self.engineering_button)
        splitter = QSplitter()
        splitter.setMinimumHeight(220)
        self.devices = DeviceDetails(); self.checks = QTextBrowser(); self.events = QTextBrowser()
        for widget,title,icon in ((self.devices,'设备状态与能力','cube'),(self.checks,'前置检查与原方案','document'),(self.events,'关键事件（最近400条）','bell')):
            widget.setMinimumHeight(192)
            panel=QGroupBox(title);panel.setObjectName('simulationCard')
            panel_layout=QVBoxLayout(panel);panel_layout.setContentsMargins(12,18,12,10)
            widget.setObjectName('simulationContent')
            panel_layout.addWidget(widget,1);splitter.addWidget(panel)
        splitter.setSizes([320,430,430])
        self.devices.update_devices([])
        self.checks.setHtml(_check_html(None,None))
        self.events.setPlainText('尚未建立模拟任务；关键事件将在首次写任务后保存。')
        live_layout.addWidget(splitter, 1)
        engineering_row = QHBoxLayout(); engineering_row.addStretch()
        self.live_details_button = QPushButton('查看工程详情'); self.live_details_button.clicked.connect(self.show_engineering)
        engineering_row.addWidget(self.live_details_button); live_layout.addLayout(engineering_row)
        live_scroll = QScrollArea();live_scroll.setWidgetResizable(True)
        live_scroll.setFrameShape(QScrollArea.NoFrame);live_scroll.setWidget(live)
        self.tabs.addTab(live_scroll, '设备、检查与事件')
        history = QWidget(); history_layout = QVBoxLayout(history)
        row = QHBoxLayout(); self.history_refresh = QPushButton('刷新模拟记录'); self.history_refresh.clicked.connect(lambda:self.command('history'))
        self.history_more = QPushButton('加载更早记录')
        self.history_more.clicked.connect(lambda:self.command('history', dict(before=self._history_cursor, append=True)))
        self.history_records = QComboBox(); self.replay_button = QPushButton('只读回看'); self.replay_button.clicked.connect(self.replay)
        self.history_records.currentIndexChanged.connect(self._buttons)
        self.export_button = QPushButton('导出模拟事件'); self.export_button.clicked.connect(self.export)
        row.addWidget(self.history_refresh); row.addWidget(self.history_more); row.addWidget(self.history_records,1); row.addWidget(self.replay_button); row.addWidget(self.export_button)
        history_layout.addLayout(row); self.replay_text = QTextBrowser(); history_layout.addWidget(self.replay_text,1)
        self.replay_text.setPlainText('尚未读取模拟记录。点击“刷新模拟记录”，选择记录后只读回看。\n回看不会连接设备或重发请求。')
        self.history_details_button = QPushButton('工程详情'); self.history_details_button.clicked.connect(self.show_history_engineering)
        history_layout.addWidget(self.history_details_button); self.history_details_button.setEnabled(False)
        self.tabs.addTab(history,'模拟记录 · 只读回看')
        details = QWidget(); detail_layout=QVBoxLayout(details)
        self.calibration_status = QLabel('实时测距不可用：缺少实测手眼、工具安装及同步位姿。导入矩阵仅完成结构核验。')
        self.calibration_status.setWordWrap(True); detail_layout.addWidget(self.calibration_status)
        self.calibration_button=QPushButton('导入版本化安装记录'); self.calibration_button.clicked.connect(self._calibration)
        self.gap_button=QPushButton('独立合成间隙演示（0.2 m）'); self.gap_button.clicked.connect(lambda:self.command('gap_demo'))
        detail_layout.addWidget(self.calibration_button);detail_layout.addWidget(self.gap_button);detail_layout.addStretch()
        self.details_text = QTextBrowser();detail_layout.addWidget(self.details_text,1)
        self.detail_engineering_button=QPushButton('工程详情');self.detail_engineering_button.clicked.connect(self.show_installation_engineering)
        detail_layout.addWidget(self.detail_engineering_button)
        self.detail_engineering_button.setEnabled(False)
        self.details_text.setPlainText('尚未导入安装记录或读取场景。实时间隙仍不可用。\n可读取配置查看设备组合，或使用独立合成间隙演示。')
        self.tabs.addTab(details,'配置、标定与间隙')
        self.status = QLabel('先选择已保存方案或显式合成测试，再核验并冻结模拟任务。')
        self.status.setObjectName('simulationNotice')
        self.status.setWordWrap(True); self.status.setTextFormat(Qt.PlainText); layout.addWidget(self.status)
        self.timer=QTimer(self); self.timer.setInterval(100); self.timer.timeout.connect(self._buttons); self.timer.start()
        self.tabs.currentChanged.connect(self._tab_changed)
        self.mode.currentIndexChanged.connect(self._source_edited)
        self.path.textChanged.connect(self._source_edited)
        for button,name in ((self.choose,'folder'),(self.preview_button,'search'),(self.freeze_button,'shield'),(self.connect_button,'link'),
                (self.config_button,'device'),(self.cancel_button,'close'),(self.check_button,'search'),
                (self.start_button,'play'),(self.pause_button,'pause'),(self.resume_button,'play'),
                (self.stop_button,'stop'),(self.reset_button,'close'),(self.recover_button,'refresh'),
                (self.history_refresh,'refresh'),(self.history_more,'history'),(self.replay_button,'eye'),
                (self.export_button,'export'),(self.calibration_button,'document'),(self.gap_button,'target'),
                (self.abandon_button,'close'), (self.result_button,'eye'),
                (self.live_details_button,'document'),(self.history_details_button,'document'),
                (self.detail_engineering_button,'document')):
            button.setIcon(line_icon(name));button.setIconSize(QSize(20,20))
        for index,name in enumerate(('device','document','device')):self.tabs.setTabIcon(index,line_icon(name))
        self._buttons()

    @property
    def busy(self):
        return self.owner.busy or self.owner.loading or self.record_runner.busy

    def _notice(self, text):
        if self._closing:return
        self.status.setText(text)
        if str(text).startswith(('操作失败', '记录操作未完成', '设备事件处理失败', '模拟所有者异常', '模拟实例清理失败')):
            self._action_error = str(text)
            self._failure_action = None
            if self._candidate_pending:
                self._candidate_pending=None
                self._set_plan_context('方案核验未完成，尚未建立任务。请查看异常原因并重试。')
        self._refresh_exception()

    @Slot(str, str)
    def _operation_failed(self, action, text):
        if self._closing:return
        self._notice(text)
        self._failure_action = action
        self._buttons()

    def _source_edited(self, *_):
        if self._candidate_token or self._candidate_pending:
            self.owner.submit('discard_candidate')
        self._candidate_token = self._candidate_summary = self._candidate_pending = None
        self._preview_inputs = None
        self._action_error = None
        self._failure_action = None
        if hasattr(self, 'plan_context'):
            self._set_plan_context('尚未核验方案。核验后显示任务、来源与版本。')
        self._buttons()

    def preview_plan(self, *, expected_revision=None, expected_result_sha256=None):
        synthetic = self.mode.currentIndex() == 1
        if not synthetic and not self.path.text().strip():
            self._notice('请选择已保存方案，或明确选择合成软件测试。')
            return False
        inputs = dict(synthetic=synthetic, path=None if synthetic else str(Path(self.path.text()).resolve()),
                      expected_revision=expected_revision, expected_result_sha256=expected_result_sha256)
        previous = self._preview_inputs or {}
        if (expected_revision is None and expected_result_sha256 is None
                and previous.get('path') == inputs['path'] and previous.get('synthetic') == synthetic):
            # Rechecking the same handoff must still verify the version shown
            # in review. Editing the source clears these expectations.
            for key in ('expected_revision', 'expected_result_sha256'):
                inputs[key] = previous.get(key)
        token = uuid.uuid4().hex
        self._candidate_token = self._candidate_summary = None
        self._candidate_pending = token
        self._preview_inputs = inputs
        self._set_plan_context('正在核验方案，完成后显示原记录摘要；尚未建立模拟任务。')
        if not self.command('preview_plan', dict(inputs, token=token)):
            self._candidate_pending = None
            return False
        return True

    def use_saved_plan(self, path, *, expected_revision=None, expected_result_sha256=None):
        value = self._snapshot or {}
        if (self._closing or self.busy or value.get('resource_held')
                or value.get('run_id') and value.get('state') not in TERMINAL_STATES):
            self._notice('当前任务尚未结束，请先撤销冻结或请求停止再选择方案。')
            return False
        self.mode.setCurrentIndex(0)
        self.path.setText(str(Path(path).resolve()))
        self._activate_live()
        return self.preview_plan(expected_revision=expected_revision, expected_result_sha256=expected_result_sha256)

    def _plan_loaded(self, text):
        value = json.loads(text)
        if self._closing or value.get('token') != self._candidate_pending:
            return
        self._candidate_token = value['token']
        self._candidate_pending = None
        self._candidate_summary = value['summary']
        self._action_error = None
        self._failure_action = None
        self._set_plan_context(plan_text(self._candidate_summary, prefix='待冻结方案')+
                               '\n核对摘要后冻结；任务标签不代表患者身份。')
        self._refresh_guidance(force=True)
        self._buttons()

    def retry_plan(self):
        if self._failure_action not in ('preview_plan', 'freeze'):return False
        if self._preview_inputs:
            return self.preview_plan(expected_revision=self._preview_inputs.get('expected_revision'),
                                     expected_result_sha256=self._preview_inputs.get('expected_result_sha256'))
        return self.preview_plan()

    def show_engineering(self):
        value = dict(self._snapshot or {})
        if self._candidate_summary:value['candidate_plan_summary'] = self._candidate_summary
        if self._action_error:value['last_operation_error'] = self._action_error
        show_text(self, '当前模拟 · 工程详情', json.dumps(value, ensure_ascii=False, indent=2))

    def _set_plan_context(self, text):
        if self._plan_context_text!=text:
            self._plan_context_text=text
            self.plan_context.setText(text)
            self.plan_context.updateGeometry()

    def show_history_engineering(self):
        if self._engineering_record:
            show_text(self, '只读模拟记录 · 工程详情', json.dumps(self._engineering_record, ensure_ascii=False, indent=2))

    def show_installation_engineering(self):
        if self._installation_details:
            show_text(self, '配置、安装与间隙 · 工程详情', json.dumps(self._installation_details, ensure_ascii=False, indent=2))

    def _refresh_exception(self):
        issue = anomaly(self._snapshot, self._action_error)
        self.exception_box.setVisible(issue is not None)
        if issue:
            title, cause, impact, next_step = issue
            self.exception_title.setText(title)
            short = cause if len(cause)<=180 else cause[:180]+'…（完整原因见异常详情）'
            self.exception_message.setText('发生了什么：'+short+'\n影响：'+impact+'\n下一步：'+next_step)
        value = self._snapshot or {}
        idle = not self.busy and not self._closing
        self.exception_reset.setVisible(bool(value.get('run_id')) and self._can_reset(value))
        self.exception_reset.setEnabled(idle and self._can_reset(value))
        self.exception_recover.setVisible(bool(value.get('persistence_gap')))
        self.exception_recover.setEnabled(idle and bool(value.get('persistence_gap')))
        retry = bool(self._action_error and self._failure_action in ('preview_plan', 'freeze')
                     and not value.get('resource_held')
                     and (not value.get('run_id') or value.get('state') in TERMINAL_STATES))
        self.exception_retry.setVisible(retry)
        self.exception_retry.setEnabled(idle and retry)

    def _clear_snapshot(self):
        self._snapshot=None;self._replay=False
        self._selected_run=None;self.source=None
        self._event_follow=True
        self._guidance_key=None
        self._candidate_token=self._candidate_summary=self._candidate_pending=None
        self._preview_inputs=None
        self._action_error=None
        self._failure_action=None
        self._engineering_record=None
        self.history_details_button.setEnabled(False)
        self.replay_text.setPlainText('旧实例已关闭，历史记录保留。点击“刷新模拟记录”后选择记录只读回看。\n回看不会连接设备或重发请求。')
        self._set_plan_context('尚未核验方案。核验后显示任务、来源与版本。')
        self.state.setText('当前模拟：实例已关闭 · 尚未建立任务')
        self.state.setToolTip('')
        self.devices.update_devices([]);self.checks.setHtml(_check_html(None,None))
        self.events.setPlainText('旧实例已关闭。历史记录保留，可只读回看。')
        self.progress.setRange(0,1);self.progress.setValue(0);self._buttons()

    def command(self, action, payload=None):
        if self._closing: return False
        if action in RECORD_ACTIONS:
            return self._record_command(action, payload)
        if action == 'abandon' and not (self._snapshot or {}).get('can_abandon_unrun'):
            self._notice('当前任务不能证明未下发请求，请使用请求停止并核对结果。')
            return False
        value = self._snapshot or {}
        if action == 'check' and not _devices_connected(value):
            self._notice('请先连接全部参与的模拟设备，再进行模拟前置检查。')
            return False
        if action == 'reset' and not self._can_reset(value):
            self._notice('请先请求停止并等待结果，再关闭旧模拟实例。')
            return False
        accepted=self.owner.submit(action,payload)
        if accepted:
            self._action_error=None
            self._failure_action=None
            if action in ('freeze','abandon'):
                self._candidate_token=self._candidate_summary=self._candidate_pending=None
            if action in LIVE_ACTIONS:
                self._activate_live()
            elif action not in ('gap_demo',):
                self._replay=False
            self._buttons()
        return accepted

    def _can_reset(self, value):
        return not value.get('run_id') or value.get('state') == 'unprepared' or value.get('state') in TERMINAL_STATES

    def reset_instances(self):
        if self._closing or self.owner.busy or self.owner.loading:
            return False
        value = self._snapshot or {}
        if not self._can_reset(value):
            self._notice('请先请求停止并等待结果，再关闭旧模拟实例。')
            return False
        if value.get('run_id'):
            consequence = ('任务尚未下发设备请求，将核验记录后保留为“已放弃、未运行”。'
                           if value.get('can_abandon_unrun') else
                           '尚未结束且无法证明未下发请求的任务保留为“停止结果未知”。'
                           if value.get('state') == 'unprepared' else '旧任务的结果和事件保持原样，未知结果不会改为已停止。')
            answer = QMessageBox.question(self, '关闭旧模拟实例',
                '将关闭当前软件模拟设备，之后需要重新连接并建立新任务。\n'+consequence+'\n是否关闭？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return False
        return self.command('reset')

    def _activate_live(self):
        self._replay = False
        self.tabs.setCurrentIndex(0)
        live = self.tabs.widget(0)
        if isinstance(live, QScrollArea):
            live.verticalScrollBar().setValue(0)
        self._changed('current_devices')
        self._selected_run = (self._snapshot or {}).get('run_id')
        self._refresh_guidance(force=True)

    def _record_command(self, action, payload):
        if self.record_runner.busy:
            return False
        current = self._snapshot or {}
        if (action == 'replay' and current.get('run_id')
                and current.get('state') not in {state.value for state in TERMINAL}):
            self._notice('请先结束模拟任务或关闭旧实例，再切换只读记录。')
            return False
        if action == 'replay':
            self.owner._replay_revision += 1
        revision = self.owner._replay_revision if action == 'replay' else None
        selected = self.history_records.currentData()
        current_run = current.get('run_id')
        options = dict(payload or {}) if action == 'history' else None
        request = dict(payload) if action == 'export' else payload
        outputs = self.outputs

        def read(cancel, progress):
            def check():
                if cancel.is_set() or revision is not None and revision != self.owner._replay_revision:
                    raise RuntimeError('记录操作已取消或输入已切换')
            check()
            store = ExecutionStore(outputs)
            if action == 'history':
                store.recover_interrupted(exclude=(current_run,) if current_run else (), check=check)
                rows = store.runs(limit=201, before=options.get('before'), with_summary=True)
                page = rows[:200]
                cursor = (page[-1]['created_utc'], page[-1]['run_id']) if page else None
                if selected and not options.get('append') and all(row['run_id'] != selected for row in page):
                    page.extend(store.runs(limit=1, run_id=selected, with_summary=True))
                check()
                return json.dumps(dict(rows=page, cursor=cursor, has_more=len(rows)>200,
                                       append=bool(options.get('append'))), ensure_ascii=False)
            if action == 'replay':
                record = store.read(request, check=check)
                check()
                return json.dumps(dict(record, view_revision=revision), ensure_ascii=False)
            return str(store.export(request['run_id'], request['path'], check=check))

        success = (self._history if action == 'history' else self._show_replay if action == 'replay'
                   else lambda path: self._notice('模拟记录已导出：'+path))
        started = self.record_runner.start(read, success,
            lambda error:self._operation_failed(action, '记录操作未完成：'+str(error)),
            revision=revision, current_revision=(lambda:self.owner._replay_revision) if revision is not None else None)
        if started:
            self._notice({'history':'正在读取模拟记录。', 'replay':'正在核验所选记录来源，可取消读取。',
                          'export':'正在导出模拟记录；设备反馈和停止操作继续可用。'}[action])
            self._buttons()
        return started

    def cancel_reading(self):
        self.owner.cancel_loading()
        self.record_runner.cancel()

    def _tab_changed(self, index):
        if index == 0 and self._replay and not self._closing:
            self._activate_live()
            self._buttons()

    def _choose_plan(self):
        path,_=QFileDialog.getOpenFileName(self,'选择已保存方案',str(self.outputs),'已保存方案 (*.zip *.json)')
        if path:
            self.path.setText(path);self.mode.setCurrentIndex(0)
            self._activate_live()
            self.preview_plan()

    def freeze(self):
        synthetic=self.mode.currentIndex()==1
        if not synthetic and not self.path.text().strip():
            self.status.setText('请选择已保存方案，或明确选择合成软件测试。');return False
        if not synthetic and not self._candidate_token:
            return self.preview_plan()
        self.source=None if synthetic else Path(self.path.text()).resolve()
        payload=dict(synthetic=synthetic,path=str(self.source) if self.source else None)
        if self._candidate_token:payload['candidate_token']=self._candidate_token
        return self.command('freeze',payload)

    def _configuration(self):
        from PySide6.QtWidgets import QMessageBox
        dialog=QMessageBox(self);dialog.setWindowTitle('配置与场景')
        dialog.setText('可同时选择多台设备的配置，全部校验通过后替换当前组合。读取不自动连接；能力以适配器反馈为准。')
        scene=dialog.addButton('读取场景',QMessageBox.ActionRole);device=dialog.addButton('读取设备配置（可多选）',QMessageBox.ActionRole)
        save=dialog.addButton('保存当前修订',QMessageBox.ActionRole);dialog.addButton('取消',QMessageBox.RejectRole);dialog.exec()
        if dialog.clickedButton() is save:self.command('save_config');return
        if dialog.clickedButton() is device:
            paths,_=QFileDialog.getOpenFileNames(self,'选择本次组合的设备配置',str(self.outputs),'版本化配置 (*.json)')
            if paths:self.command('configuration',paths)
        elif dialog.clickedButton() is scene:
            path,_=QFileDialog.getOpenFileName(self,'读取配置',str(self.outputs),'版本化配置 (*.json)')
            if path:self.command('scenario',path)

    def _calibration(self):
        path,_=QFileDialog.getOpenFileName(self,'导入安装记录',str(self.outputs),'安装记录 (*.json)')
        if path:self.command('calibration',path)

    def _refresh_guidance(self, *, force=False):
        if self._replay and not force:return
        value = self._snapshot or {}
        state = value.get('state')
        connected = _devices_connected(value)
        key = (value.get('run_id'), state, connected, value.get('can_pause'), value.get('can_resume'), self._candidate_token)
        urgent = value.get('persistence_gap') or value.get('last_error') or state in ('failed', 'result_unknown')
        if not force and key == self._guidance_key and not urgent:
            return
        self._guidance_key = key
        # A failed first load may publish an empty coordinator. Keep the load
        # error visible rather than replacing it with the initial instructions.
        if not (force or value.get('run_id') or connected):
            return
        if value.get('persistence_gap'):
            text = ('事件持久化缺口：'+str(value.get('persistence_error'))+
                    '；停止与反馈处理仍继续。记录恢复可写后，可重试记录并标记缺口。')
        elif state in ('failed', 'result_unknown', 'interrupted_needs_review') or value.get('last_error'):
            reasons = [event['payload'].get('reason') for event in value.get('events', [])
                       if event['payload'].get('reason')]
            reason = value.get('last_error') or (reasons[-1] if reasons else '请查看检查与关键事件')
            text = STATE_TEXT.get(state, state or '任务异常')+'：'+str(reason)
            if state in TERMINAL_STATES:
                text += '；需要新任务时，请先在“实例与记录恢复”中关闭旧模拟实例，旧结果保留。'
        elif self._candidate_token:
            text = '方案已核验。核对任务、来源与版本后，下一步：冻结模拟任务。'
        elif not value.get('run_id'):
            text = ('模拟设备已连接。' if connected else '')+'先选择已保存方案或显式合成测试，再核验并冻结模拟任务。'
        else:
            hints = {
                'unprepared': ('模拟任务已冻结，设备已连接。下一步：模拟前置检查。' if connected else
                               '模拟任务已冻结。下一步：连接模拟设备，再进行模拟前置检查。'),
                'checking': '正在检查方案、记录与模拟设备，等待准备确认。',
                'ready_for_simulation': '模拟前置检查通过。下一步：开始模拟。',
                'running': ('模拟运行中；可暂停或请求停止。进度为确认的软件步骤。' if value.get('can_pause') else
                            '模拟运行中；本场景不可暂停，可请求停止。进度为确认的软件步骤。'),
                'pause_requested': '已请求暂停，等待参与设备确认；仍可请求停止。',
                'pause_confirmed': '暂停已确认。可继续模拟或请求停止。',
                'resume_requested': '已请求继续，等待参与设备确认；仍可请求停止。',
                'stop_requested': '已请求停止，等待参与设备分别确认静止与输出关闭。',
                'succeeded': '模拟步骤完成：全部软件步骤已确认，参与设备静止/输出关闭已确认。记录可只读回看。',
                'stopped': '已停止：参与设备静止/输出关闭已确认，记录可只读回看。',
                'cancelled': '已取消：参与设备静止/输出关闭已确认，记录可只读回看。',
                'abandoned_unrun': '冻结已撤销：已放弃、未运行。可以改选方案，原记录保留。',
            }
            text = hints.get(state, STATE_TEXT.get(state, '请查看当前任务状态。'))
        self.status.setText(text)

    @Slot(str)
    def _changed(self, _):
        snapshot=self.owner.take_snapshot()
        if self._closing:return
        if snapshot is None:
            self._buttons()
            return
        previous=self._snapshot or {}
        self._snapshot=json.loads(snapshot);value=self._snapshot
        identity = '任务 '+value['run_id'][:12] if value['run_id'] else '尚未建立任务'
        self.state.setText('当前模拟：'+(STATE_TEXT.get(value['state'],value['state'])+' · '+identity if value['run_id'] else identity))
        self.state.setToolTip('当前模拟任务：'+(value['run_id'] or '尚未建立'))
        self.progress.setRange(0,max(1,value['total_steps']));self.progress.setValue(value['completed_steps'])
        self.devices.update_devices(value['devices'])
        if value['preflight'] != previous.get('preflight') or value.get('plan_summary') != previous.get('plan_summary'):
            self.checks.setHtml(_check_html(value['preflight'],value.get('plan_summary')))
        if value['run_id'] != previous.get('run_id') or value['events'] != previous.get('events'):
            scroll=self.events.verticalScrollBar();position=scroll.value()
            self._event_follow=self._event_follow or position>=scroll.maximum()-1
            if self.events.hasFocus() and position<scroll.maximum()-1:
                self._event_follow=False
            follow=self._event_follow
            self.events.setPlainText('\n'.join(_event_text(e) for e in value['events']))
            if follow:
                self.events.moveCursor(QTextCursor.End)
                self.events.ensureCursorVisible()
            else:
                scroll.setValue(position)
        self._refresh_guidance()
        if not self._replay:self._selected_run=value['run_id']
        if value.get('plan_summary') and not self._candidate_summary and not self._candidate_pending:
            self._set_plan_context(plan_text(value['plan_summary'], prefix='冻结方案'))
        self._buttons()

    def _buttons(self):
        value=self._snapshot or {};blocked=self._closing or self.busy
        control_blocked=self._closing or self.owner.busy or self.owner.loading
        for button,key in ((self.start_button,'can_start'),(self.pause_button,'can_pause'),(self.resume_button,'can_resume')):
            button.setEnabled(not control_blocked and bool(value.get(key)))
        self.stop_button.setEnabled(not self._closing and not self.owner.stop_pending and bool(value.get('can_stop')))
        state=value.get('state')
        has_result=bool(value.get('run_id')) and state in TERMINAL_STATES
        self.result_button.setVisible(has_result)
        self.result_button.setEnabled(not blocked and has_result)
        active=state not in (None,'unprepared',*TERMINAL_STATES)
        frozen_unprepared=state=='unprepared' and bool(value.get('run_id'))
        self.freeze_button.setEnabled(not blocked and not active and not frozen_unprepared and not value.get('resource_held',False)
                                      and (self.mode.currentIndex()==1 or bool(self._candidate_token)))
        self.preview_button.setEnabled(not blocked and not active and not frozen_unprepared and not value.get('resource_held',False))
        self.abandon_button.setEnabled(not control_blocked and bool(value.get('can_abandon_unrun')))
        self.abandon_button.setVisible(frozen_unprepared)
        self.abandon_button.setToolTip('仅在记录和当前状态共同证明未下发请求时撤销；原记录保留。')
        self.connect_button.setEnabled(not blocked and not active and not value.get('resource_held',False))
        self.config_button.setEnabled(not blocked and not active and not frozen_unprepared and not value.get('resource_held',False))
        editable=not blocked and not active and not frozen_unprepared
        self.mode.setEnabled(editable)
        self.path.setEnabled(editable and self.mode.currentIndex()==0)
        self.choose.setEnabled(editable and self.mode.currentIndex()==0)
        connected = _devices_connected(value)
        self.check_button.setEnabled(not blocked and state=='unprepared' and bool(value.get('run_id')) and connected)
        self.check_button.setToolTip('请先核验并冻结模拟任务。' if not value.get('run_id') else
                                     '请先连接全部参与的模拟设备。' if not connected else
                                     '本次任务已结束；修正原因后重新核验并冻结新任务。' if state in TERMINAL_STATES else
                                     '核验方案、记录和参与设备的准备状态。')
        self.reset_button.setEnabled(not blocked and self._can_reset(value) and bool(value.get('devices')))
        self.reset_button.setToolTip('关闭软件模拟设备，保留旧任务结果；关闭前需确认。' if self._can_reset(value) else
                                     '任务尚未结束，请先请求停止并等待结果。')
        self.recovery_controls.setVisible(bool(value.get('devices')) and (self._can_reset(value) or bool(value.get('persistence_gap'))))
        self.cancel_button.setEnabled((self.owner.loading or self.record_runner.busy) and not self._closing)
        self.recover_button.setEnabled(not blocked and bool(value.get('persistence_gap')))
        self.replay_button.setEnabled(not blocked and not active and not frozen_unprepared and self.history_records.count()>0)
        self.history_refresh.setEnabled(not blocked)
        self.history_more.setEnabled(not blocked and self._history_has_more)
        self.history_records.setEnabled(not blocked)
        self.export_button.setText('导出所选记录' if self.history_records.currentData() else '导出当前任务')
        self.export_button.setEnabled(not blocked and bool(self.history_records.currentData() or self._selected_run))
        self.calibration_button.setEnabled(not blocked and not active)
        self.gap_button.setEnabled(not blocked)
        self._highlight_next()
        self._refresh_exception()

    def _highlight_next(self):
        value=self._snapshot or {}; state=value.get('state'); run=value.get('run_id')
        if value.get('persistence_gap') or self._action_error or state in ('failed','result_unknown','interrupted_needs_review'):
            step, button, action = None,None,None
        elif self._candidate_token:
            step, button, action = 1, self.freeze_button, 'freeze'
        elif run and state in TERMINAL_STATES:
            step, button, action = 4,self.result_button,'result'
        elif not run:
            step, button, action = 0, self.freeze_button if self.mode.currentIndex()==1 else self.preview_button, 'freeze' if self.mode.currentIndex()==1 else 'preview_plan'
        elif state=='unprepared':
            step, button, action = ((3,self.check_button,'check') if _devices_connected(value) else (2,self.connect_button,'connect'))
        elif state=='checking':step, button, action = 3, None, None
        elif state=='ready_for_simulation':step, button, action = 4,self.start_button,'start'
        elif state=='pause_confirmed':step, button, action = 4,self.resume_button,'resume'
        elif state=='running':step, button, action = 4,self.pause_button if value.get('can_pause') else None,'pause' if value.get('can_pause') else None
        else:step, button, action = 4,None,None
        self._primary_action=action
        self.workflow.setToolTip('当前流程位置，不表示真实执行资格或几何质量通过。')
        for index,label in enumerate(self.step_labels):
            tone='current' if index==step else 'idle'
            if label.property('state')!=tone:
                label.setProperty('state',tone);label.style().unpolish(label);label.style().polish(label)
        for candidate in (self.preview_button,self.freeze_button,self.connect_button,self.check_button,self.start_button,self.pause_button,self.resume_button,self.result_button):
            role='primary' if candidate is button else 'secondary'
            if candidate.property('role')!=role:
                candidate.setProperty('role',role);candidate.style().unpolish(candidate);candidate.style().polish(candidate)

    def _add_history_row(self, row):
        stamp='时间未知'
        if type(row.get('created_utc')) in (int,float):
            try:stamp=time.strftime('%m-%d %H:%M',time.localtime(row['created_utc']))
            except (OSError,ValueError,OverflowError):pass
        fields=plan_fields(row)
        label=fields['name']+' · '+STATE_TEXT.get(row['state'],row['state'])+' · '+stamp
        index=self.history_records.findData(row['run_id'])
        if index<0:
            self.history_records.addItem(label,row['run_id'])
            index=self.history_records.count()-1
        else:
            self.history_records.setItemText(index,label)
        completed=row.get('completed_steps')
        self.history_records.setItemData(index,
            plan_text(row)+f"\n记录 ID：{row['run_id']}\n确认步骤：{completed if completed is not None else '未知'} / {row.get('total_steps') or '未知'}", Qt.ToolTipRole)
        return index

    def _history(self, text):
        if self._closing:return
        value=json.loads(text);selected=self.history_records.currentData()
        if not value['append']:self.history_records.clear()
        for row in value['rows']:
            self._add_history_row(row)
        if selected and self.history_records.findData(selected)>=0:
            self.history_records.setCurrentIndex(self.history_records.findData(selected))
        self._history_cursor=value['cursor'];self._history_has_more=value['has_more']
        self.status.setText(f'已读取 {self.history_records.count()} 条模拟记录。'+('可加载更早记录。' if self._history_has_more else ''))
        if not self.history_records.count():
            self.replay_text.setPlainText('暂无模拟记录。完成核验并冻结后，任务与事件将在这里保留。\n只读回看不会连接设备或重发请求。')
        self._buttons()

    def replay(self):
        run=self.history_records.currentData()
        if run and self.command('replay',run):self._selected_run=run

    def show_result(self):
        value=self._snapshot or {}
        if value.get('run_id') and value.get('state') in TERMINAL_STATES:
            return self.command('replay',value['run_id'])
        return False

    def _show_replay(self, text):
        if self._closing:return
        value=json.loads(text)
        if 'gap_demo' in value:
            self._show_details(text);return
        if value.get('view_revision') != self.owner._replay_revision:
            return
        self._replay=True
        self._selected_run=value['run_id']
        # Full arrays/routes and every event remain available in the export.
        # Show a bounded engineering view without re-laying out megabytes of
        # point JSON on the GUI thread when switching historical records.
        snapshot=value.get('snapshot',{});saved=snapshot.get('plan',{});plan=saved.get('plan',{})
        summary={key:value[key] for key in ('run_id','display_state','recorded_state','completed_steps',
                                           'event_sequence','source_integrity','backup_context') if key in value}
        summary.update(read_only=True,source='simulation',execution_ready=False,physical_accuracy_verified=False,
                       plan=dict(plan_source=saved.get('plan_source'),source_path=plan.get('source_path'),
                                 revision_id=plan.get('revision_id'),result_sha256=plan.get('result_sha256'),
                                 task_context=plan.get('task_context'),project_id=plan.get('project_id'),
                                 source_model=plan.get('source_model'),region=plan.get('region'),
                                 treatment_head=plan.get('treatment_head'),gap_report=gap_summary(plan.get('gap_report')),
                                 software_steps=len(saved.get('steps',[]))),
                       devices=snapshot.get('devices',[]),used_calibrations=snapshot.get('used_calibrations',[]),
                       recent_events=value.get('events',[])[-400:],
                       display_note='显示最近400条事件及方案摘要；导出包含完整冻结快照、原路线和全部事件。')
        self._engineering_record=summary
        self.history_details_button.setEnabled(True)
        index=self._add_history_row(dict(run_id=value['run_id'],state=value['display_state'],
            created_utc=value.get('created_utc'),completed_steps=value.get('completed_steps'),
            total_steps=len(saved['steps']) if 'steps' in saved else None,
            plan_source=saved.get('plan_source'),source_path=plan.get('source_path'),
            revision_id=plan.get('revision_id'),task_context=plan.get('task_context')))
        self.history_records.setCurrentIndex(index)
        self.replay_text.setPlainText(history_text(value))
        self.tabs.setCurrentIndex(1)
        self.status.setText('只读回看：'+STATE_TEXT.get(value['display_state'],value['display_state'])+'；不会创建设备或重新发送请求。')
        self._buttons()

    def _show_details(self, text):
        if self._closing:return
        value=json.loads(text)
        self._installation_details=value
        self.detail_engineering_button.setEnabled(True)
        if value.get('gap_demo'):
            gap=value['gap_demo']
            distance=(f"{gap['distance_m']:.3f} m" if gap.get('available') else '不可用')
            content=(f"独立合成间隙演示　·　沿轴距离：{distance}\n"
                     f"参考坐标：{gap.get('frame','未知')}\n"
                     '输入为已知合成平面，不代表当前患者测距、碰撞距离或物理精度。\n原计算字段在工程详情查看。')
        elif value.get('installation_record'):
            record=value['installation_record'];evidence=value.get('evidence_status') or {}
            source={'measured_import_unverified':'导入安装记录（实测资格未验证）',
                    'synthetic':'合成安装记录'}.get(record.get('source'),'来源未知')
            status={'synthetic_only':'仅用于合成测试','changed_or_missing':'来源文件变化或缺失',
                    'source_verified_measurement_unverified':'来源摘要核对通过，实测资格未验证'}.get(evidence.get('status'),'未知')
            content=(f"{source}　·　标定 ID：{record.get('calibration_id','未知')}　·　版本：{record.get('revision','未知')}\n"
                     f"坐标关系：{record.get('source_frame','未知')} → {record.get('target','未知')}\n"
                     f"证据状态：{status}\n实测手眼、工具安装与同步位姿尚未验证，实时测距不可用。")
        else:
            scenario=value.get('scenario') or {}
            devices=value.get('configurations') or []
            participants='、'.join(c.get('model') or KIND_TEXT.get(c.get('kind'),'设备') for c in devices) if devices else '场景默认模拟设备组合'
            adapter_modes='、'.join(SOURCE_TEXT.get(c.get('adapter'), '尚未配置') for c in devices)
            stop_feedback='停止反馈：'+('支持确认' if scenario.get('confirm_stop') else '停止完成未确认')
            content=(f"场景：{scenario.get('scenario_id','未知')}　·　版本：{scenario.get('revision','未知')}\n"
                     f"参与设备：{participants}\n适配模式：{adapter_modes or '模拟设备'}　·　{stop_feedback}\n"
                     '设备尚未连接，能力由连接后的反馈提供。\n'
                     '真实适配器与实测标定未实现；原始配置和时间预算可在工程详情查看。')
        self.details_text.setPlainText(content)
        self.tabs.setCurrentIndex(2)

    def export(self):
        run=self.history_records.currentData() or self._selected_run
        if not run:return
        path,_=QFileDialog.getSaveFileName(self,'导出模拟事件',str(self.outputs/f'simulation_events_{run[:12]}.json'),'模拟事件 (*.json)')
        if path:self.command('export',dict(run_id=run,path=path))

    def can_leave(self):
        value=self._snapshot or {}
        return not self.busy and value.get('state') not in ('checking','running','pause_requested','pause_confirmed','resume_requested','stop_requested')

    def shutdown(self):
        self._closing=True
        self.timer.stop()
        controls_done=self.owner.shutdown()
        records_done=self.record_runner.shutdown(timeout_ms=0)
        return controls_done and records_done
