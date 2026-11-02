"""Apply identical diagnostic stage timers to current and archived backends."""
from contextlib import ExitStack
from functools import wraps
from unittest.mock import patch


def observe_save_stages(function):
    @wraps(function)
    def observed(*args, **kwargs):
        import numpy as np
        from femto.operation_diagnostics import diagnosed
        from femto.planning import target_surface, trajectory_export, planning_io, coverage_workflow
        with ExitStack() as stack:
            for module, name, label in (
                (np, 'savez_compressed', 'asset.compression'),
                (target_surface, 'write', 'target.serialization'),
                (target_surface, 'read', 'target.asset_read'),
                (trajectory_export, '_write_core', 'route.serialization'),
                (trajectory_export, 'publish', 'file.publication'),
                (planning_io, 'publish', 'file.publication'),
                (coverage_workflow, 'write', 'coverage.write_verify'),
            ):
                stack.enter_context(patch.object(module, name, diagnosed(label)(getattr(module, name))))
            return function(*args, **kwargs)
    return observed
