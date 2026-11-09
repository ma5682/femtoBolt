"""One-owner handoff of a freshly assessed capture, never a global model cache.

Content is checked again at both handoffs. A stale handoff only removes this
optimization: ordinary source loading and its confirmation gates still run.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import threading

from femto.io_utils import load_json_object, sha256_file
from femto.operation_diagnostics import diagnosed


@diagnosed('load.capture_asset_fingerprints')
def asset_fingerprints(result, check):
    from femto.dataset.reconstruction_inputs import snapshot, MAX_FILES
    root = Path(result).resolve()
    if root.name == 'reconstruction':
        root = root.parent
    reconstruction = root / 'reconstruction'
    report = load_json_object(reconstruction / 'reconstruction_report.json')
    # Limit this optimization to genuine local, input-bound captures.
    context = report.get('local_capture') or {}
    if not context.get('session_id'):
        raise ValueError('Prepared review requires a local capture')
    inputs = snapshot(report['source_dataset'], check)
    paths = [p for p in reconstruction.rglob('*') if p.is_file()]
    tracking = context.get('tracking_report')
    if tracking:
        paths.append(Path(tracking).resolve())
    if len(paths) + len(inputs['files']) > MAX_FILES:
        raise ValueError('Prepared review asset count exceeds budget')
    result = dict(inputs['files'])
    for path in paths:
        check()
        result[str(path.resolve())] = sha256_file(path, check)
    check()
    return result


class PreparedCapture:
    """Capture worker -> review worker -> planning worker, exactly once each.

    The private model is never exposed to the UI or simultaneously shared with
    another worker. Preview pixels and summary are copied on the first handoff;
    the final handoff consumes the model and drops every retained reference.
    """
    def __init__(self, root, model, fingerprints, completion, roi, summary, preview):
        self.root = Path(root).resolve()
        self._model = model
        self._fingerprints = fingerprints
        self._completion = deepcopy(completion)
        self._roi = deepcopy(roi)
        self._summary = deepcopy(summary)
        self._preview = preview.copy()
        self._state = 'fresh'
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            self._clear()

    def _clear(self):
        self._state = 'closed'
        self._model = self._fingerprints = self._completion = None
        self._roi = self._summary = self._preview = None

    def _matches(self, root, check):
        from femto.dataset.acquisition_completion import read_completion, result_root
        check()
        if result_root(root) != self.root:
            return False
        # The completion gate still runs again in saved_source.load_source.
        if read_completion(self.root, check=check) != self._completion:
            return False
        return asset_fingerprints(self.root, check) == self._fingerprints

    @diagnosed('load.prepared_review_verify')
    def review(self, root, roi, check):
        with self._lock:
            if self._state != 'fresh':
                return None
            try:
                if roi != self._roi or not self._matches(root, check):
                    self._clear()
                    return None
                self._state = 'reviewed'
                return deepcopy(self._summary), self._preview.copy()
            except (OSError, ValueError, KeyError, TypeError):
                self._clear()
                return None
            except BaseException:
                self._clear()
                raise

    @diagnosed('load.prepared_model_verify')
    def take_model(self, root, check):
        with self._lock:
            try:
                if self._state != 'reviewed' or not self._matches(root, check):
                    return None
                return self._model
            except (OSError, ValueError, KeyError, TypeError):
                return None
            finally:
                self._clear()


def close_prepared(value):
    if isinstance(value, PreparedCapture):
        value.close()
