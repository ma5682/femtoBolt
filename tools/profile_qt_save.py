"""Time the actual PlanningPage save handler and its normal completion callback.

The Qt reference view runs offscreen. Catalogues and diagnostics are isolated;
no guards, validation functions, algorithms or task budgets are replaced.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import gc
import time
import tracemalloc
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from femto.io_utils import write_json_strict
from tools.benchmark_workstation import Measurements, file_digest
from tools.save_stage_observation import observe_save_stages
from tools.profile_saved_plan import archive_values, folder_values, route_digest, GCObservation
from tools.save_environment import performance_environment


@observe_save_stages
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if tracemalloc.is_tracing():
        raise RuntimeError('Disable allocation tracing for Qt save performance measurement')
    project, output = args.project.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Use a new workspace outputs directory')
    output.mkdir(parents=True, exist_ok=False)
    os.environ['FEMTO_DIAGNOSTICS_DIR'] = str(output/'diagnostics')
    code = {path.relative_to(ROOT).as_posix(): file_digest(path)
            for path in sorted((ROOT/'femto').rglob('*.py'))}
    metadata = dict(schema='qt-save-profile-v1', status='running', project=str(project),
                    project_sha256=file_digest(project), code_sha256=code,
                    probe_sha256=file_digest(Path(__file__)),
                    scope='Actual PlanningPage.save_plan through runner completion; offscreen reference view, not full desktop rendering',
                    release_scope='route_release_seconds: save_plan() call to the UI-thread notice that the route package is published '
                                  'while its editable project is still being written; absent when the backend releases only at completion')
    write_json_strict(output/'manifest.json', metadata)
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    from femto.app import qt_planning
    from femto.app.qt_planning_io import masks_from_region
    from femto.annotation.reference_quality import assess_selection
    from femto.annotation.fixed_rgb_geometry import GeometryBudget
    from femto.planning import region_project, trajectory_export
    from femto.planning.surface_trajectory import SurfaceTrajectory
    app = QApplication.instance() or QApplication([])
    measurements = Measurements(output)
    instrumentation = []
    def measured(stage, function):
        observer = GCObservation()
        row = dict(stage=stage, environment_before=performance_environment())
        gc.callbacks.append(observer.callback)
        try:
            return measurements.run(stage, 0, function)
        finally:
            gc.callbacks.remove(observer.callback)
            row.update(gc=observer.report(), environment_after=performance_environment())
            instrumentation.append(row)
            write_json_strict(output/'instrumentation.json', instrumentation)
    page = None
    try:
        model = measured('source_open', lambda: region_project.open_project(
            project, workspace=output/'opened'))
        region, plan = model.store.regions[0], model.saved_plan
        expected_route = route_digest(SurfaceTrajectory.from_dict(plan['surface_trajectory']))
        _, expected_evidence = archive_values(project)
        work, forbidden = masks_from_region(region, model.rgb.shape[:2])
        assessment = assess_selection(model, work, forbidden)
        from femto.planning.planning_drafts import Journal
        page = qt_planning.PlanningPage()
        page.resize(1280, 850)
        page.install_model(model, project, Journal(project, output/'drafts'), assessment=assessment)
        page.show()
        app.processEvents()
        page.set_stage(2)
        page.reviewed.setChecked(True)
        app.processEvents()
        saved_callbacks = []
        page.saved.connect(saved_callbacks.append)
        if not page.usable() or page.quality.result['status'] != 'checked':
            raise RuntimeError('The real saved plan is not eligible for the normal UI save action')

        def save():
            loop, watchdog = QEventLoop(), QTimer()
            watchdog.setSingleShot(True)
            expired = []
            def timeout():
                expired.append(True)
                page.runner.cancel()
            def released(_text):
                # Backends that release only at completion never report this state.
                if 'route_release_seconds' not in metadata and getattr(page, 'finishing_save', None) is not None:
                    metadata['route_release_seconds'] = time.perf_counter()-started
            watchdog.timeout.connect(timeout)
            page.runner.finished.connect(loop.quit)
            page.status_changed.connect(released)
            try:
                # This watchdog requests cancellation only; it does not extend
                # the worker's original GeometryBudget or fabricate success.
                watchdog.start(180000)
                started = time.perf_counter()
                page.save_plan()
                if not page.busy:
                    raise RuntimeError('UI save did not start: '+page.status_label.text())
                loop.exec()
                metadata['complete_seconds'] = time.perf_counter()-started
            finally:
                watchdog.stop()
                page.runner.finished.disconnect(loop.quit)
                page.status_changed.disconnect(released)
            if expired or page._error:
                raise RuntimeError('UI save failed: '+page.status_label.text())
            if (len(saved_callbacks) != 1 or page.archive is None
                    or Path(saved_callbacks[0]).resolve() != page.archive
                    or not (page.archive/'planning.region.zip').is_file() or page.dirty):
                raise AssertionError('Completion callback lacks a complete saved result')
            return page.archive

        with patch.object(qt_planning, 'OUTPUTS_ROOT', output/'isolated'/'outputs'):
            native = measured('qt_complete_save', save)
        portable = native/'planning.region.zip'
        restored_native = measured('native_readback', lambda: trajectory_export.load_trajectory(
            native, check=GeometryBudget().check, reproduce=True))
        restored_portable = measured('portable_readback', lambda: region_project.open_project(
            portable, workspace=output/'reopened'))
        _, actual_evidence = archive_values(portable)
        checks = dict(native_route_equal=route_digest(restored_native) == expected_route,
            portable_route_equal=route_digest(SurfaceTrajectory.from_dict(
                restored_portable.saved_plan['surface_trajectory'])) == expected_route,
            native_evidence_equal=folder_values(native) == expected_evidence,
            portable_evidence_equal=actual_evidence == expected_evidence)
        metadata.update(status='passed', checks=checks, native=str(native), portable=str(portable))
        if not all(checks.values()):
            raise AssertionError('UI saved or independently reopened evidence changed')
    except BaseException as error:
        metadata.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if page is not None and not page.busy:
            page.shutdown()
            page.close()
            app.processEvents()
        metadata['source_unchanged'] = file_digest(project) == metadata['project_sha256']
        metadata['code_unchanged'] = code == {p.relative_to(ROOT).as_posix(): file_digest(p)
                                              for p in sorted((ROOT/'femto').rglob('*.py'))}
        metadata['all_checks_passed'] = bool(metadata['status'] == 'passed'
            and metadata['source_unchanged'] and metadata['code_unchanged']
            and [row['stage'] for row in measurements.rows] == [
                'source_open', 'qt_complete_save', 'native_readback', 'portable_readback']
            and all(row['status'] == 'passed' and not row['sampling_errors'] for row in measurements.rows))
        if not metadata['all_checks_passed']:
            metadata['status'] = 'failed'
        write_json_strict(output/'manifest.json', metadata)
    if not metadata['all_checks_passed']:
        raise AssertionError('Incomplete Qt save evidence')
    print(json.dumps(metadata['checks']), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
