"""Simulation prerequisites are separate from unavailable real execution gates."""
from dataclasses import dataclass

from femto.hardware.interface_contract import DeviceKind, Intent, Readiness, SourceKind, SpatialState


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    reason: str
    scope: str = 'simulation'

    def to_dict(self):
        return dict(name=self.name, status=self.status, reason=self.reason, scope=self.scope)


def inspect(plan, adapters, *, clock, scenario, storage_writable, spatial_context=None,
            refresh_status=lambda: None, verify_check=lambda: None):
    checks = []
    try:
        plan.verify_assets(verify_check)
        checks.append(Check('方案来源及冻结摘要', 'passed', '已使用原方案核验链；合成场景需显式选择'))
    except (OSError, ValueError) as error:
        checks.append(Check('方案来源及冻结摘要', 'failed', str(error)))
    # File verification can exceed a simulated heartbeat age. The owner polls
    # here before this pure check reads the provider's most recent snapshot.
    refresh_status()
    checks.append(Check('事件记录可写', 'passed' if storage_writable else 'failed',
                        '关键事件需落盘后分发；记录失败阻止新步骤'))
    if not adapters:
        checks.append(Check('参与设备', 'failed', '尚未显式连接模拟设备'))
    elif not any(a.binding.kind is not DeviceKind.BASE or scenario.base_mode != 'fixed' for a in adapters):
        checks.append(Check('参与步骤确认设备', 'failed', '只有固定结构底座，没有可确认软件步骤的设备'))
    for adapter in adapters:
        status, capabilities = adapter.status(), adapter.capabilities()
        binding = status.binding
        good = binding.source is SourceKind.SIMULATION and status.connected and status.readiness is Readiness.READY and not status.fault
        checks.append(Check(binding.device_id+' 连接及来源', 'passed' if good else 'failed',
                            'simulation 绑定及适配器反馈；配置字段不能证明连接'))
        fixed = binding.kind is DeviceKind.BASE and scenario.base_mode == 'fixed'
        supported = fixed or {Intent.PREPARE, Intent.SIMULATION_STEP, Intent.REQUEST_STOP} <= capabilities.supported_intents
        checks.append(Check(binding.device_id+' 控制能力', 'passed' if supported else 'failed',
                            '固定底座不创建运动接口' if fixed else '按适配器实际能力校验'))
        checks.append(Check(binding.device_id+' 暂停能力', 'passed' if Intent.REQUEST_PAUSE in capabilities.supported_intents else 'not_applicable',
                            '按能力启用暂停按钮；不支持时不替换成取消或停止'))
        if binding.kind is DeviceKind.BASE:
            stable = status.spatial_state in (SpatialState.LOCKED, SpatialState.FIXED_WITH_EVIDENCE)
            checks.append(Check('底座空间状态', 'passed' if stable else 'failed', status.spatial_state.value))
        if not fixed:
            physical = status.motion_stopped if binding.kind in (DeviceKind.ARM, DeviceKind.BASE) else status.output_off
            checks.append(Check(binding.device_id+' 独立静止/输出关闭反馈', 'passed' if physical is True else 'unknown',
                                '不把机械臂停止当作输出关闭'))
        age_ok = status.clock_domain == clock.domain and 0 <= clock.now()-status.timestamp_s <= scenario.status_max_age_s
        checks.append(Check(binding.device_id+' 状态时效', 'passed' if age_ok else 'failed', '仅比较同一模拟时钟域'))
    if spatial_context is not None:
        valid = spatial_context.invalid_reason is None
        checks.append(Check('合成空间上下文', 'passed' if valid else 'failed',
                            '合成来源；无真实标定资格' if valid else spatial_context.invalid_reason))
    else:
        checks.append(Check('真实坐标及安装标定', 'not_applicable', '纯步骤模拟不依赖；实时测距不可用'))
    checks.append(Check('方案质量与治疗头参数', 'not_applicable',
                        '原几何缺口与未知范围保持；参数状态：'+str((plan.public_plan.get('treatment_head') or {}).get('values_status','unknown'))))
    for name, reason in (
        ('厂家适配器与协议', '真实机械臂/超声协议未提供'),
        ('实测安装及手眼标定', '当前导入只证明结构，物理精度未验证'),
        ('运动、接触、硬件联锁', '待厂家资料及实机集成验证'),
        ('真实执行资格', '研发几何与模拟均不可执行'),
    ):
        checks.append(Check(name, 'unknown', reason, 'real_execution'))
    ready = all(check.status in ('passed', 'not_applicable') for check in checks if check.scope == 'simulation')
    return dict(schema='femto-simulation-preflight-v1', source='simulation', simulation_ready=ready,
                checks=[check.to_dict() for check in checks], execution_ready=False, physical_accuracy_verified=False)
