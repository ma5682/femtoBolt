"""Compatibility name for the clinician workflow navigator."""

from widgets.workflow_sidebar import WorkflowSidebar


class WorkflowNav(WorkflowSidebar):
    """The current sidebar implementation exposed under the new product name.

    Keeping this as a subclass preserves the existing signal contract
    (``step_requested``) while allowing newer pages to import ``WorkflowNav``.
    """

    pass


StepNavigator = WorkflowNav

__all__ = ["WorkflowNav", "StepNavigator", "WorkflowSidebar"]
