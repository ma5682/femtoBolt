def current_notice(s):
    if s.replay_mode:
        if s.stage == 2 and s.plan.phase == "failed":
            return "项目回放 · 只读核对：原始扫描路线存在局部断开，请查看高亮端点。", "warning"
        if s.stage == 3 and s.plan.phase == "failed":
            return "项目回放 · 只读结果：当前规划存在断开段，不能作为执行路线。", "error"
        if s.stage == 1:
            if s.region.support == "partial":
                return "项目回放 · 只读警告：部分三维支持未确认，请查看选区 RGB 与支持记录。", "warning"
            return "项目回放 · 只读核对：已显示原始区域与三维支持记录。", "normal"
        return "项目回放 · 只读核对：已载入保存的采集数据。", "normal"
    if s.message:
        return s.message, s.message_tone
    if s.stage == 0:
        if s.capture.return_to_region and s.capture.phase == "capturing":
            return "已保留区域标记。请补采高亮位置，完成后返回区域标记。", "warning"
        return {
            "idle": ("请调整相机位置，使目标区域完整进入画面。", "normal"),
            "capturing": ("正在采集。保持目标稳定，缓慢移动相机补采不同视角。", "normal"),
            "checking": ("正在检查深度、相机位姿和表面连续性…", "normal"),
            "passed": ("采集检查通过。请确认目标表面完整，再进入区域标记。", "normal"),
            "failed": ("左侧存在三维数据缺失，请继续补采高亮位置。", "error"),
        }[s.capture.phase]
    if s.stage == 1:
        return {
            "unchecked": ("标出规划区与禁入区，完成后确认标记并检查三维几何。", "normal"),
            "checking": ("正在检查选区对应的三维几何，已画标记会保留。", "normal"),
            "missing": ("高亮区域缺少可靠三维数据，请返回补采。", "error"),
            "valid": ("三维支持已确认。可继续修改标记或生成规划预览。", "normal"),
            "partial": ("部分三维支持未确认。可查看规划预览，但不能视为治疗覆盖已确认。", "warning"),
        }[s.region.support]
    if s.stage == 2:
        _, error = s.plan.parameters.spacing()
        if error:
            return error, "error"
        if s.plan.phase == "generating":
            return "正在计算轨迹路径…", "normal"
        if s.plan.phase == "failed":
            return "扫描线局部断开，请检查高亮位置并重新规划。", "error"
        if s.plan_current:
            return "请核对规划区、禁入区和边缘未覆盖位置，再进入最终确认。", "warning"
        return "参数已更新。请重新生成轨迹，核对变化后的扫描行距。", "normal"
    if s.plan.saved_id:
        return "规划已保存，可在历史中查看。当前记录是研发规划预览。", "normal"
    return "请核对任务、区域、行距及检查结果，确认后保存规划方案（预览）。", "normal"
