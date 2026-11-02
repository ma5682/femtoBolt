from dataclasses import dataclass
from app.app_state import AppState

STAGES = ("三维采集", "区域标记", "轨迹规划", "规划确认")
VIEWS = (("RGB", "深度", "点云", "模型"), ("固定 RGB", "三维模型"), ("三维模型", "RGB 路线图"), ("三维模型", "RGB 路线图"))


@dataclass(frozen=True)
class Action:
    key: str
    text: str
    enabled: bool = True
    reason: str = ""
    destructive: bool = False


def navigation_reason(s: AppState, stage: int) -> str:
    if not 0 <= stage < 4:
        return "无效步骤。"
    if s.replay_mode:
        return ""
    if s.busy:
        return "正在检查，请稍候。"
    if s.capture.phase == "capturing" and stage != 0:
        return "请先完成采集和质量检查。"
    if stage >= 1 and s.capture.phase != "passed":
        return "请先完成采集质量检查。"
    if stage >= 2 and s.region.support != "valid":
        return "请先标记规划区并通过三维支持检查。"
    if stage == 3 and not s.plan_current:
        return "请先生成轨迹并处理阻断项。"
    return ""


def actions(s: AppState) -> tuple[Action, list[Action]]:
    if s.replay_mode:
        if s.stage == 0:
            return Action("to_region", "下一步：区域标记  →"), [Action("close_replay", "退出项目回放")]
        if s.stage == 1:
            return Action("to_planning", "查看轨迹规划"), [Action("to_capture", "返回采集数据"), Action("close_replay", "退出项目回放")]
        if s.stage == 2:
            return Action("to_confirm", "查看规划确认"), [Action("to_region", "返回规划区"), Action("close_replay", "退出项目回放")]
        return Action("close_replay", "退出项目回放"), [Action("to_planning", "返回轨迹规划")]
    if s.stage == 0:
        if s.capture.phase == "idle":
            return Action("start", "建立参考位置并开始采集"), []
        if s.capture.phase == "capturing":
            return Action("finish", "完成采集并检查质量", s.capture.frames >= 3, "至少等待 3 个模拟视角"), [Action("restart", "重新开始", destructive=True)]
        if s.capture.phase == "checking":
            return Action("", "正在检查采集质量…", False, "检查完成后继续"), []
        if s.capture.phase == "failed":
            return Action("repair_capture", "继续补采"), [Action("focus_capture", "查看缺失位置")]
        return Action("to_region", "返回已画区域" if s.region.has_work else "进入区域标记"), [Action("restart", "重新采集", destructive=True)]
    if s.stage == 1:
        if s.region.support == "checking":
            return Action("", "正在检查三维支持…", False, "检查完成后继续"), []
        if s.region.support == "missing":
            return Action("repair_geometry", "返回补采"), [Action("undo", "撤销上一笔")]
        return Action("plan_region", "确认标记并检查三维几何", s.region.has_work, "请先在画面中标出规划区"), [Action("to_capture", "返回采集")]
    if s.stage == 2:
        if s.plan.phase == "generating":
            return Action("", "正在计算路径…", False, "正在计算规划预览"), []
        if s.plan.phase == "failed":
            return Action("repair_route", "重新规划"), [Action("to_region", "修改区域")]
        if not s.plan_current:
            _, error = s.plan.parameters.spacing()
            return Action("generate", "计算轨迹路径", not error, error), [Action("to_region", "修改区域")]
        return Action("confirm", "进入最终确认"), [Action("to_region", "修改区域")]
    return Action("save", "规划方案已保存" if s.plan.saved_id else "保存规划方案（预览）", s.plan_current and not s.plan.saved_id, "当前版本已保存" if s.plan.saved_id else "请先处理阻断项"), [Action("to_region", "修改区域"), Action("to_planning", "修改规划参数")]
