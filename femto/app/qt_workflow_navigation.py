"""Route workflow clicks through the existing page and source validity gates."""
from pathlib import Path

from PySide6.QtCore import QObject


class WorkflowNavigation(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._sources = {}
        self._return = None
        for page in (window.capture, window.planning, window.review):
            page.workflow.step_requested.connect(lambda step, owner=page: self.navigate(owner, step))
            page.workflow.refreshed.connect(lambda owner=page: self.refresh(owner))
        window.pages.currentChanged.connect(self.refresh)
        self.refresh()

    def _blocked_reason(self):
        window = self.window
        state = window.capture.session
        if not window.execution.can_leave():
            return '模拟任务进行中，请结束模拟后再切换项目。'
        if state.active or state.finishing:
            return '采集或建模进行中，请完成或取消后再切换。'
        if window.capture.review_task.busy or window.planning.busy or window.review.busy:
            return '读取、计算或保存进行中，请完成或取消后再切换。'
        if window.planning.loading_source is not None:
            return '模型准备未完成，请重试或返回原页面。'
        return ''

    def _capture_root(self, page):
        model = page.model
        source = getattr(model, 'capture_source', None) or page.source
        key = (id(model), str(source))
        if self._sources.get(page, (None, None))[0] != key:
            root = self.window.capture.review_result_root(source) if model is not None and source else None
            self._sources[page] = (key, root)
        return self._sources[page][1]

    def remember_capture_return(self, page, source):
        if page not in (self.window.planning, self.window.review):
            return
        root = self._capture_root(page)
        target = self.window.capture.review_result_root(source)
        self._return = (page, page.model, root) if root is not None and root == target else None

    def _return_page(self):
        if self._return is None or self.window.capture.session.result is None:
            return None
        page, model, root = self._return
        current = Path(self.window.capture.session.result).resolve()
        return page if page.model is model and current == root and self._capture_root(page) == root else None

    def _local_destinations(self, page):
        if page.model is None:
            return set()
        if page is self.window.planning:
            steps = {4} if page.fixed_selection else {3}
            if page.region and not page.selection_dirty:
                steps.add(4)
                if page.fixed_selection:steps.discard(3)
            if page.usable():
                steps.add(5)
            return steps
        return {2} | ({3} if page.region else set()) | ({4, 5} if page.plan else set())

    def destinations(self, page):
        blocked = self._blocked_reason()
        if blocked:
            return set(), {index: blocked for index in range(1, 7)}
        reasons = {
            1: '局部采集须由开始采集按钮启动；已结束的采集只能回看记录。',
            2: '请先完成采集并成功读取模型。',
            3: '请先检查当前模型，再进入路线规划。',
            4: '请先完成工作区域的几何检查。',
            5: '请先生成与当前区域和参数一致的有效路线。',
            6: '至少一帧通过检查后可提前完成，第三帧保存后自动进入选区；修改重建范围需重新建模。',
        }
        if page is self.window.capture:
            available = {1}
            if page.session.result:
                ready = page.review_ready()
                available = {1, 2, 3} if ready else set()
                if not ready:
                    reasons.update({index: '采集预览尚未读取成功，请重试读取。' for index in (1, 2)})
                origin = self._return_page()
                if origin is not None:
                    # A failed or cancelled preview read must not strand the
                    # already loaded draft; its model was never replaced.
                    available = ({1, 2} if ready else set()) | self._local_destinations(origin)
            if page.can_rebuild_region():available.add(6)
            return available, reasons
        available = self._local_destinations(page)
        if self._capture_root(page) is not None:
            available |= {1, 2}
            capture=self.window.capture
            if (capture.can_rebuild_region()
                    and Path(capture.session.result).resolve()==self._capture_root(page)):
                available.add(6)
        else:
            reasons[2] = '此规划的关联采集未能核验；可在当前页面切换三维模型检查。'
        return available, reasons

    def refresh(self, page=None):
        if page not in (self.window.capture, self.window.planning, self.window.review):
            page = self.window.pages.currentWidget()
        if page is self.window.execution:
            return
        available, reasons = self.destinations(page)
        page.workflow.set_navigation(available, reasons)

    def navigate(self, page, step):
        window = self.window
        if page is not window.pages.currentWidget() or step == page.workflow.current_step:
            return False
        available, _ = self.destinations(page)
        if step not in available:
            self.refresh(page)
            return False
        if step == 6 and page is not window.capture:
            if page is window.planning and not page.can_leave():
                return False
            if not window.show_capture():
                return False
            return window.capture.rebuild_region()
        if page is window.capture:
            if step == 6:
                return page.rebuild_region()
            origin = self._return_page()
            # Step 02 belongs to the historical viewer when capture is only
            # a detour from that viewer. Keep its loaded model and view state.
            if step < 3 and not (step == 2 and origin is window.review):
                return page.show_workflow_step(step)
            if origin is None:
                page.show_workflow_step(2)
                page.primary_action()
                return window.pages.currentWidget() is window.planning
            window.pages.setCurrentWidget(origin)
            window._set_header_active(window.open_button if origin is window.planning else window.history_button)
            return self._show_local(origin, step)
        if step >= 3 or page is window.review and step == 2:
            return self._show_local(page, step)
        if not window.show_capture():
            return False
        return window.capture.show_workflow_step(step)

    def _show_local(self, page, step):
        if page is self.window.planning:
            if step==3 and page.fixed_selection:step=4
            page.set_stage(step - 3)
            return page._stage == step - 3
        return page.show_workflow_step(step)

    def resume_planning(self, source):
        """A capture-page return keeps the exact editable project, including ZIPs."""
        window = self.window
        origin = self._return_page()
        if origin is not window.planning or self._blocked_reason():
            return False
        if window.capture.review_result_root(source) != self._capture_root(origin):
            return False
        window.pages.setCurrentWidget(origin)
        window._set_header_active(window.open_button)
        return self._show_local(origin, 3)
