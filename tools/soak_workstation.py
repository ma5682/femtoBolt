"""Exercise complete offline workstation cycles in one process with natural GC.

Outputs are retained. Existing output directories are never reused or resumed.
A running checkpoint after a crash means incomplete, not recovered or passed.
This measures software replay; it does not validate camera or treatment hardware.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time
import traceback
import tracemalloc
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil

from femto.io_utils import write_json_strict
from tools.audit_workstation_benchmark import plan_values
from tools.benchmark_workstation import Measurements, file_digest, inventory

MIB = 1024**2
GIB = 1024**3


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_values(arrays):
    result = {}
    for name, array in arrays.items():
        value = np.ascontiguousarray(array)
        result[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                           sha256=hashlib.sha256(value.tobytes()).hexdigest())
    return result


def model_values(model):
    ledger, plan = model.target_ledger, model.saved_plan
    if ledger is None or plan is None:
        raise ValueError('The fixed project must contain a verified saved plan and target ledger')
    return dict(ordered_route_sha256=canonical_digest(plan['surface_trajectory']['ordered_items']),
                target_arrays=array_values(ledger.arrays), target_summary=ledger.metadata['summary'],
                has_coverage=bool(plan.get('coverage_evidence')))


class NaturalGC:
    """Only observe Python collection callbacks; never force or disable GC."""
    def __init__(self):
        self.count = 0
        self.seconds = 0.0
        self.collected = 0
        self.uncollectable = 0
        self.generations = [0, 0, 0]
        self.started = {}

    def callback(self, phase, info):
        generation = info['generation']
        if phase == 'start':
            self.started[generation] = time.perf_counter()
        elif phase == 'stop':
            started = self.started.pop(generation, None)
            self.seconds += time.perf_counter() - started if started is not None else 0.0
            self.count += 1
            self.generations[generation] += 1
            self.collected += info.get('collected', 0)
            self.uncollectable += info.get('uncollectable', 0)

    def snapshot(self):
        return dict(collections=self.count, seconds=self.seconds, collected=self.collected,
                    uncollectable=self.uncollectable, generations=list(self.generations))

    def __enter__(self):
        if not gc.isenabled():
            raise RuntimeError('Natural GC is disabled; refusing a misleading sustained-memory run')
        gc.callbacks.append(self.callback)
        return self

    def __exit__(self, *_):
        gc.callbacks.remove(self.callback)


def gc_delta(before, after):
    return {key: ([b-a for a, b in zip(before[key], after[key])]
                  if key == 'generations' else after[key]-before[key]) for key in before}


class WeakObjects:
    """Track lifetimes without retaining models, arrays or partition instances."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.cycle = 0
        self.objects = {}
        self.observed = 0
        self.unsupported_types = {}

    def watch(self, label, value):
        if not self.enabled:
            return
        existing = self.objects.get(id(value))
        if existing is not None and existing['reference']() is value:
            return
        try:
            reference = weakref.ref(value)
        except TypeError:
            name = type(value).__module__+'.'+type(value).__qualname__
            if name in self.unsupported_types or len(self.unsupported_types) < 8:
                self.unsupported_types[name] = self.unsupported_types.get(name, 0)+1
            return
        self.objects[id(value)] = dict(reference=reference, label=label, born=self.cycle,
                                      array_bytes=int(value.nbytes) if isinstance(value, np.ndarray) else 0)
        self.observed += 1

    def model(self, label, model):
        if not self.enabled:
            return
        seen = set()
        pending = [(model, label)]
        while pending:
            value, path = pending.pop()
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(value, np.ndarray):
                self.watch(path, value)
            elif isinstance(value, dict):
                pending.extend((item, path+'.'+str(key))
                               for key, item in reversed(tuple(value.items())))
            elif isinstance(value, (list, tuple)):
                pending.extend((value[index], path+'.'+str(index))
                               for index in range(len(value)-1, -1, -1))
            elif type(value).__module__.startswith(('femto.', 'trimesh.')) and hasattr(value, '__dict__'):
                self.watch(path, value)
                pending.append((vars(value), path))
            elif type(value).__module__.startswith('open3d.'):
                self.watch(path, value)

    def snapshot(self):
        live = {key: value for key, value in self.objects.items() if value['reference']() is not None}
        self.objects = live
        groups = {}
        for value in live.values():
            label = value['label'].split('.')[0]
            group = groups.setdefault(label, dict(alive=0, older_cycles=0, array_bytes=0, oldest_age=0))
            age = self.cycle-value['born']
            group['alive'] += 1
            group['older_cycles'] += int(age > 0)
            group['array_bytes'] += value['array_bytes']
            group['oldest_age'] = max(group['oldest_age'], age)
        return dict(enabled=self.enabled, observed=self.observed, alive=len(live), groups=groups,
                    unsupported_types=dict(self.unsupported_types),
                    older_cycles=sum(value['born'] < self.cycle for value in live.values()),
                    oldest_age=max((self.cycle-value['born'] for value in live.values()), default=0),
                    array_bytes=sum(value['array_bytes'] for value in live.values()))

    def survivor_details(self):
        """Keep only bounded descriptions, never referrers or live objects."""
        result = []
        for record in self.objects.values():
            if record['born'] >= self.cycle:
                continue
            value = record['reference']()
            if value is None:
                continue
            allocation = tracemalloc.get_object_traceback(value) if tracemalloc.is_tracing() else None
            referrers = gc.get_referrers(value)
            result.append(dict(label=record['label'], age=self.cycle-record['born'],
                type=type(value).__module__+'.'+type(value).__qualname__,
                allocation=[] if allocation is None else list(allocation.format())[-6:],
                referrer_types=[type(item).__module__+'.'+type(item).__qualname__ for item in referrers[:8]],
                note='Referrers are a bounded diagnostic sample and may include observer frames.'))
            del value, referrers
            if len(result) == 8:
                break
        return result


@contextmanager
def observe_partitions(tracker, enabled):
    """Add weak references after construction; leave all cache/GC behavior intact."""
    if not enabled:
        yield
        return
    from femto.geometry.exact_pixel_partition import PixelPartition
    original = PixelPartition.__init__
    def observed(instance, *args, **kwargs):
        original(instance, *args, **kwargs)
        tracker.watch('pixel_partition', instance)
        tracker.watch('pixel_partition.triangles', instance.triangles)
    PixelPartition.__init__ = observed
    try:
        yield
    finally:
        PixelPartition.__init__ = original


def linear_trend(samples):
    """Descriptive OLS spread, not an IID confidence or leak-proof guarantee."""
    if len(samples) < 3:
        return dict(sample_count=len(samples), status='insufficient_samples')
    xs = [float(row['elapsed_seconds']) for row in samples]
    ys = [float(row['rss_bytes']) for row in samples]
    center_x, center_y = statistics.mean(xs), statistics.mean(ys)
    variance_x = sum((x-center_x)**2 for x in xs)
    if variance_x == 0:
        return dict(sample_count=len(samples), status='insufficient_time_span')
    slope = sum((x-center_x)*(y-center_y) for x, y in zip(xs, ys))/variance_x
    residual = sum((y-center_y-slope*(x-center_x))**2 for x, y in zip(xs, ys))
    slope_spread = math.sqrt(residual/(len(samples)-2)/variance_x)
    window = min(3, len(samples)//2)
    first, last = statistics.mean(ys[:window]), statistics.mean(ys[-window:])
    return dict(status='descriptive', sample_count=len(samples),
                elapsed_span_seconds=xs[-1]-xs[0], min_rss_bytes=min(ys), max_rss_bytes=max(ys),
                slope_bytes_per_hour=slope*3600,
                slope_plus_minus_two_residual_standard_errors_bytes_per_hour=[
                    (slope-2*slope_spread)*3600, (slope+2*slope_spread)*3600],
                first_window_mean_bytes=first, last_window_mean_bytes=last,
                window_cycles=window, window_growth_bytes=last-first,
                note='RSS observations are autocorrelated; this band is descriptive, not a 95% IID confidence interval.')


def memory_report(cycles, stages, warmup_cycles):
    completed = [row for row in cycles if row['status'] == 'passed' and row['cycle'] >= warmup_cycles]
    cleanup = [dict(elapsed_seconds=row['elapsed_seconds'], rss_bytes=row['cleanup_rss_bytes'])
               for row in completed]
    by_stage = {}
    completed_ids = {row['cycle'] for row in completed}
    for row in stages:
        if row['repetition'] in completed_ids and row['status'] == 'passed':
            by_stage.setdefault(row['stage'], []).append(
                dict(elapsed_seconds=row['session_elapsed_seconds'], rss_bytes=row['rss_end_bytes']))
    return dict(warmup_cycles_excluded=warmup_cycles, post_warmup_cycles=len(completed),
                enough_cycles=len(completed) >= 5, cleanup=linear_trend(cleanup),
                same_stage_rss={name: linear_trend(rows) for name, rows in by_stage.items()},
                final_weak_references=cycles[-1].get('weak_references') if cycles else None,
                interpretation='Inspect stage-matched trends together with aged weak references; bounded RSS or one completed run does not prove absence of leaks.')


def directory_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob('*') if item.is_file())


class WorkstationCycle:
    def __init__(self, project, output, reconstruction_report=None, *, reuse_save_target=False,
                 preserve_exact_plan=False):
        from femto.planning import region_project, surface_workspace
        self.region_project, self.surface_workspace = region_project, surface_workspace
        self.project, self.output = Path(project).resolve(), Path(output).resolve()
        self.reuse_save_target = reuse_save_target
        self.preserve_exact_plan = preserve_exact_plan
        self.expected_original = plan_values(self.project)
        if preserve_exact_plan:
            import zipfile
            with zipfile.ZipFile(self.project) as archive:
                document=json.loads(archive.read('project.json'))
            self.exact_algorithms=document['plan']['planning_context']['revision']['inputs']['algorithms']
            if (self.exact_algorithms.get('target')!='reference-pixel-exact-partition-v2'
                    or self.expected_original['target_summary']['leaf_count']<=200000):
                raise ValueError('Exact soak requires a complete paged exact-partition saved plan, not a v1/small sample')
            from tools.profile_saved_plan import archive_values,route_digest
            from femto.planning.surface_trajectory import SurfaceTrajectory
            self.exact_route_digest=route_digest(SurfaceTrajectory.from_dict(document['plan']['surface_trajectory']))
            _,self.exact_evidence=archive_values(self.project)
        self.reconstruction_report = Path(reconstruction_report).resolve() if reconstruction_report else None
        self.reconstruction = None
        self.expected_reconstruction = None
        if self.reconstruction_report:
            from femto.pipeline.scene_reconstruction import reconstruct_scene
            data = json.loads(self.reconstruction_report.read_text(encoding='utf-8-sig'))
            self.expected_reconstruction = dict(
                mesh_sha256=file_digest(self.reconstruction_report.parent/'scene_tsdf_mesh_m.ply'),
                frame_count=data['frame_count'])
            self.reconstruction = (reconstruct_scene, Path(data['source_dataset']).resolve(),
                {key: value for key, value in data['parameters'].items()
                 if key in inspect.signature(reconstruct_scene).parameters})

    @property
    def expected_stages(self):
        return ((['reconstruct'] if self.reconstruction else []) + [
            'history_open', 'verify_original', 'planning_preserve_exact' if self.preserve_exact_plan else 'planning_generate',
            ] + (['cancelled_portable_save'] if self.preserve_exact_plan else [])
            + ['native_save', 'portable_save', 'saved_history_open']
            + (['native_history_open','verify_native_readback'] if self.preserve_exact_plan else [])
            + ['verify_outputs', 'sqlite_history', 'cleanup_idle'])

    def __call__(self, folder, stage, tracker, cycle):
        reconstruction = None
        if self.reconstruction:
            rebuild, dataset, parameters = self.reconstruction
            rebuilt = stage('reconstruct', lambda: rebuild(dataset, folder/'reconstruction', **parameters))
            reconstruction = dict(mesh_sha256=file_digest(folder/'reconstruction'/'scene_tsdf_mesh_m.ply'),
                                  frame_count=rebuilt['frame_count'])
            if reconstruction != self.expected_reconstruction:
                raise AssertionError('Reconstruction differs from the fixed reference mesh or frame count')
        from femto.annotation.fixed_rgb_geometry import GeometryBudget
        model = stage('history_open', lambda: self.region_project.open_project(
            self.project, workspace=folder/'opened'))
        tracker.model('original_model', model)
        original = stage('verify_original', lambda: model_values(model))
        if original != self.expected_original:
            raise AssertionError('Opened original route or target arrays differ from the fixed project')
        region = model.store.regions[0]
        head = deepcopy(model.saved_plan['treatment_head'])
        if self.preserve_exact_plan:
            def preserve():
                from femto.planning import planning_revision
                plan=deepcopy(model.saved_plan)
                revision=plan['planning_context']['revision']
                if revision['inputs']['algorithms']!=self.exact_algorithms:
                    raise AssertionError('Exact plan algorithms changed during history open')
                planning_revision.require_current(model,region,head,revision)
                return plan
            plan=stage('planning_preserve_exact',preserve)
        else:
            plan = stage('planning_generate', lambda: self.surface_workspace.generate(
                model, region, head, GeometryBudget().check))
        cancellation=None
        if self.preserve_exact_plan:
            def cancel_attempt():
                from threading import Event
                from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
                event=Event();budget=GeometryBudget(event)
                phase='复制参考网格与图像资产' if cycle%2==0 else '核验保存目标与方案来源'
                observed=[]
                def progress(message):
                    observed.append(message)
                    if message==phase:event.set()
                target=folder/'cancelled.region.zip'
                catalogue=folder/'cancelled_catalogue'
                try:
                    self.region_project.save_project(model,region,target,plan,budget.check,
                        progress=progress,catalog_root=catalogue,allow_replace=False)
                except GeometryTaskCancelled:
                    if not event.is_set():raise AssertionError('Cancellation occurred without the requested phase')
                else:
                    raise AssertionError('Cancellation attempt unexpectedly completed')
                if target.exists() or catalogue.exists() and any(catalogue.rglob('*.json')):
                    raise AssertionError('Cancelled exact save published an archive or history record')
                if model_values(model)!=self.expected_original:
                    raise AssertionError('Cancellation changed the original exact plan or target')
                return dict(cancelled=True,phase=phase,observed_progress=observed,
                            portable_absent=True,source_plan_unchanged=True)
            cancellation=stage('cancelled_portable_save',cancel_attempt)
        from contextlib import nullcontext
        from femto.planning.save_validation import SaveValidationSession
        reuse = self.reuse_save_target and bool(plan.get('planning_context'))
        with SaveValidationSession() if reuse else nullcontext() as session:
            options = dict(target_session=session) if session is not None else {}
            # The combined request uses the same normal budget as Qt saving.
            shared_check = GeometryBudget().check
            native = stage('native_save', lambda: self.surface_workspace.export(
                model, region, plan, shared_check, **options))
            portable = stage('portable_save', lambda: self.region_project.save_project(
                model, region, folder/'result.region.zip', plan,
                shared_check if reuse else GeometryBudget().check,
                catalog_root=self.output/'isolated'/'outputs'/'saved_projects', allow_replace=False, **options))
        reopened = stage('saved_history_open', lambda: self.region_project.open_project(
            portable, workspace=folder/'reopened'))
        tracker.model('reopened_model', reopened)
        if self.preserve_exact_plan:
            from femto.planning.trajectory_export import load_trajectory
            from tools.profile_saved_plan import folder_values,route_digest,archive_values
            native_reopened=stage('native_history_open',lambda:load_trajectory(
                native,check=GeometryBudget().check,reproduce=True))
            tracker.watch('native_reopened_trajectory',native_reopened)
            def verify_native():
                if (route_digest(native_reopened)!=self.exact_route_digest
                        or folder_values(native)!=self.exact_evidence
                        or archive_values(portable)[1]!=self.exact_evidence):
                    raise AssertionError('Full independent native reopen differs from complete exact input')
                return dict(full_replay=True,route_semantics_sha256=self.exact_route_digest)
            stage('verify_native_readback',verify_native)

        def verify():
            saved = plan_values(portable)
            if model_values(reopened) != saved:
                raise AssertionError('Independent reopen differs from saved target arrays or route')
            if canonical_digest(plan['surface_trajectory']['ordered_items']) != saved['ordered_route_sha256']:
                raise AssertionError('Generated and saved ordered routes differ')
            trajectory = json.loads((native/'trajectory.json').read_text(encoding='utf-8-sig'))
            with np.load(native/'target_surface.npz', allow_pickle=False) as arrays:
                native_arrays = array_values({name: arrays[name] for name in arrays.files})
            if (canonical_digest(trajectory['ordered_items']) != saved['ordered_route_sha256']
                    or native_arrays != saved['target_arrays']
                    or trajectory['planning_context']['target']['summary'] != saved['target_summary']):
                raise AssertionError('Native and portable routes, target arrays or summaries differ')
            return saved
        saved = stage('verify_outputs', verify)
        if self.preserve_exact_plan and saved!=self.expected_original:
            raise AssertionError('Exact save changed input route, arrays, target summary or coverage presence')
        history = stage('sqlite_history', lambda: self.history(portable))
        # Return only small digests/paths. All large model/plan locals leave
        # scope before the runner performs its identical cleanup observation.
        return dict(equivalence=dict(original=original, generated=saved, reconstruction=reconstruction),
                    plan_mode='preserved_complete_exact' if self.preserve_exact_plan else 'generated_default_v1',
                    exact_algorithms=self.exact_algorithms if self.preserve_exact_plan else None,
                    cancellation=cancellation,
                    save_target_reuse=reuse,
                    assets=dict(native=str(native), portable=str(portable)), history=history,
                    verification=dict(native=('Independent full native open and verified target ledger readback' if self.preserve_exact_plan
                                              else 'JSON route and every NPZ target array; no independent native load'),
                                      portable='Independent full open and verified target ledger readback'))

    def history(self, portable):
        from femto.history_store import HistoryStore
        from femto.planning.catalog import scan_saved_versions
        outputs = self.output/'isolated'/'outputs'
        store = HistoryStore(outputs)
        rows = scan_saved_versions(outputs)
        search_text = lambda row: ' '.join(str(row.get(key) or '') for key in
            ('name', 'project_id', 'revision_id', 'parent_revision_id'))
        store.replace('plans', rows, search_text=search_text)
        targets = [row for row in rows if Path(row.get('project') or '') == portable]
        if len(targets) != 1 or not targets[0].get('revision_id'):
            raise AssertionError('Saved project is missing or ambiguous in the isolated history')
        target, token = targets[0], targets[0]['revision_id']
        before = store.query('plans', token)
        if before['total'] != 1:
            raise AssertionError('Version search did not identify exactly one saved project')
        store.set_hidden([target])
        fresh = HistoryStore(outputs)
        if fresh.query('plans', token)['total'] or fresh.query('plans', token, hidden_only=True)['total'] != 1:
            raise AssertionError('Hidden state did not persist across connections')
        fresh.set_hidden([target], False)
        fresh.replace('plans', rows, search_text=search_text)
        after = fresh.query('plans', token)
        if after != before or fresh.query('plans', token, hidden_only=True)['total']:
            raise AssertionError('Restored version does not match its original indexed identity')
        return dict(indexed_count=len(rows), searched_revision=token, hidden_restored=True,
                    version_count=len(fresh.versions(target['project_id'])))


def fixed_inputs(project, reconstruction_report=None):
    paths = [Path(project).resolve()]
    sources = {}
    if reconstruction_report:
        report = Path(reconstruction_report).resolve()
        paths.extend([report, report.parent/'scene_tsdf_mesh_m.ply'])
        data = json.loads(report.read_text(encoding='utf-8-sig'))
        dataset = Path(data['source_dataset']).resolve()
        roots = {dataset}
        manifest = json.loads((dataset/'manifest.json').read_text(encoding='utf-8-sig'))
        if manifest.get('source_dataset'):
            roots.add(Path(manifest['source_dataset']).resolve())
        sources = {str(root): inventory(root) for root in sorted(roots)}
    return dict(files={str(path): file_digest(path) for path in paths}, datasets=sources,
                production_code={path.relative_to(ROOT).as_posix(): file_digest(path)
                    for path in sorted((ROOT/'femto').rglob('*.py'))},
                tools={name: file_digest(ROOT/'tools'/name) for name in
                       ('soak_workstation.py', 'runtime_memory.py', 'benchmark_workstation.py', 'audit_workstation_benchmark.py')})


def run_session(output, worker, snapshot_inputs, *, expected_stages, duration_seconds=3600, max_cycles=None,
                warmup_cycles=3, minimum_cycles=8, idle_seconds=1, reserve_bytes=4*GIB,
                initial_cycle_bytes=256*MIB, track_partitions=False, observation_mode='weak'):
    from tools.runtime_memory import MemoryDiagnostics
    output = Path(output).resolve()
    if (not math.isfinite(duration_seconds) or duration_seconds < 0 or warmup_cycles < 0
            or minimum_cycles < 1 or not math.isfinite(idle_seconds) or idle_seconds < 0):
        raise ValueError('Invalid duration, warmup, minimum cycles or idle interval')
    if max_cycles is not None and max_cycles < 1:
        raise ValueError('max_cycles must be positive')
    if reserve_bytes < 0 or initial_cycle_bytes < 0:
        raise ValueError('Disk budgets must be nonnegative')
    expected_stages = list(expected_stages)
    if (not expected_stages or expected_stages[-1] != 'cleanup_idle'
            or len(set(expected_stages)) != len(expected_stages)
            or any(not isinstance(stage, str) or not stage for stage in expected_stages)):
        raise ValueError('Expected stages must be unique nonempty names ending in cleanup_idle')
    required_cycles = max(minimum_cycles, warmup_cycles+5)
    output.mkdir(parents=True, exist_ok=False)
    diagnostics = MemoryDiagnostics(observation_mode)
    measurements, tracker = Measurements(output), WeakObjects(enabled=observation_mode != 'minimal')
    cycles, baseline, before, after = [], None, None, None
    started = time.monotonic()
    manifest = dict(schema='workstation-soak-v1', status='running', pid=psutil.Process().pid,
        python=sys.version, platform=platform.platform(), started_unix=time.time(),
        duration_seconds=duration_seconds, max_cycles=max_cycles, warmup_cycles=warmup_cycles,
        minimum_cycles=minimum_cycles, idle_seconds=idle_seconds, reserve_bytes=reserve_bytes,
        required_cycles=required_cycles,
        forced_gc=False, natural_gc_enabled=gc.isenabled(), partition_weakref_hook=track_partitions,
        observation_mode=observation_mode,
        allocation_tracing_is_timing_evidence=False,
        gc_threshold=gc.get_threshold(), physical_memory_bytes=psutil.virtual_memory().total,
        logical_cpu_count=psutil.cpu_count(), rss_sampling_seconds=.05,
        expected_stages=expected_stages,
        retained_assets='all cycles; no cleanup or output-directory reuse',
        scope='one-process offline software replay; reconstruction uses recorded poses, planning uses the fixed project')
    write_json_strict(output/'manifest.json', manifest)
    stop_reason, error_text = None, None
    observer = None
    active_started = None
    active_seconds = 0.0
    try:
        before = snapshot_inputs()
        write_json_strict(output/'inputs.json', before)
        with diagnostics, NaturalGC() as observer, observe_partitions(tracker, track_partitions and tracker.enabled):
            active_started = time.monotonic()
            while True:
                elapsed = time.monotonic()-active_started
                if elapsed >= duration_seconds and len(cycles) >= required_cycles:
                    stop_reason = 'requested_duration_and_cycles_completed'
                    break
                if max_cycles is not None and len(cycles) >= max_cycles:
                    stop_reason = 'max_cycles'
                    break
                budget = max([initial_cycle_bytes] + [row['asset_bytes'] for row in cycles])
                available = shutil.disk_usage(output).free
                if available < reserve_bytes + math.ceil(1.25*budget):
                    stop_reason = 'disk_guard'
                    manifest['disk_guard'] = dict(free_bytes=available, reserve_bytes=reserve_bytes,
                                                  next_cycle_budget_bytes=math.ceil(1.25*budget))
                    break
                cycle = len(cycles)
                tracker.cycle = cycle
                folder = output/f'cycle_{cycle:04d}'
                folder.mkdir()
                state = dict(cycle=cycle, status='running', stage='starting')
                write_json_strict(folder/'state.json', state)
                write_json_strict(output/'checkpoint.json', state)
                cycle_gc = observer.snapshot()

                def stage(name, function):
                    state.update(stage=name, status='running')
                    write_json_strict(folder/'state.json', state)
                    write_json_strict(output/'checkpoint.json', state)
                    prior_gc = observer.snapshot()
                    previous_row_count = len(measurements.rows)
                    try:
                        value = measurements.run(name, cycle, function)
                    finally:
                        if len(measurements.rows) > previous_row_count:
                            measured = measurements.rows[-1]
                            measured['gc'] = gc_delta(prior_gc, observer.snapshot())
                            measured['session_elapsed_seconds'] = time.monotonic()-active_started
                            measured['measurement_status'] = 'failed' if measured['sampling_errors'] else 'passed'
                            measured['memory_observation'] = diagnostics.snapshot()
                            write_json_strict(output/'stages.json', measurements.rows)
                            state['stage_status'] = measured['status']
                            state['measurement_status'] = measured['measurement_status']
                        else:
                            state['measurement_status'] = 'missing'
                        write_json_strict(folder/'state.json', state)
                        write_json_strict(output/'checkpoint.json', state)
                    if len(measurements.rows) != previous_row_count+1:
                        raise AssertionError('Missing or duplicate stage measurement')
                    if measurements.rows[-1]['sampling_errors']:
                        raise RuntimeError('RSS sampling failed: ' + repr(measurements.rows[-1]['sampling_errors']))
                    return value
                try:
                    result = worker(folder, stage, tracker, cycle)
                    state['result'] = result
                    if baseline is None:
                        baseline = result['equivalence']
                    if result['equivalence'] != baseline:
                        raise AssertionError('Cycle outputs differ from the first complete cycle')
                    if not gc.isenabled():
                        raise RuntimeError('Natural GC was disabled during the cycle')
                    stage('cleanup_idle', lambda: time.sleep(idle_seconds))
                    measured_cycle = [row for row in measurements.rows if row['repetition'] == cycle]
                    if [row['stage'] for row in measured_cycle] != expected_stages:
                        raise AssertionError('Cycle stages are incomplete, duplicated or out of order')
                    if any(row['status'] != 'passed' or row['sampling_errors'] for row in measured_cycle):
                        raise AssertionError('Cycle includes failed stage or RSS sampling evidence')
                    state.update(status='passed', elapsed_seconds=time.monotonic()-active_started,
                        cleanup_rss_bytes=measurements.rows[-1]['rss_end_bytes'],
                        weak_references=tracker.snapshot(), gc=gc_delta(cycle_gc, observer.snapshot()),
                        asset_bytes=directory_bytes(folder), result=result)
                    state['memory_observation'] = diagnostics.snapshot(allocation_detail=True)
                    if observation_mode == 'allocations':
                        state['survivor_details'] = tracker.survivor_details()
                    cycles.append(dict(state))
                    write_json_strict(folder/'state.json', state)
                    write_json_strict(output/'checkpoint.json', state)
                    write_json_strict(output/'cycles.json', cycles)
                    write_json_strict(output/'memory.json', memory_report(cycles, measurements.rows, warmup_cycles))
                except BaseException as error:
                    state.update(status='failed', error=repr(error), traceback=traceback.format_exc())
                    write_json_strict(folder/'state.json', state)
                    write_json_strict(output/'checkpoint.json', state)
                    raise
    except BaseException as error:
        stop_reason = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        error_text = traceback.format_exc()
    finally:
        if active_started is not None:
            active_seconds = time.monotonic()-active_started
    try:
        after = snapshot_inputs()
        write_json_strict(output/'inputs_after.json', after)
        inputs_unchanged = before is not None and before == after
    except Exception:
        inputs_unchanged = False
        error_text = (error_text or '') + '\nFinal input verification failed:\n' + traceback.format_exc()
    elapsed = time.monotonic()-started
    memory = memory_report(cycles, measurements.rows, warmup_cycles)
    functional = bool(cycles) and error_text is None and inputs_unchanged
    qualified = (functional and stop_reason == 'requested_duration_and_cycles_completed'
                 and active_seconds >= 3600 and len(cycles) >= 8 and memory['enough_cycles'])
    status = ('failed' if error_text or not inputs_unchanged else
              'completed' if stop_reason == 'requested_duration_and_cycles_completed' else 'stopped')
    manifest.update(status=status, stop_reason=stop_reason,
                    error=error_text, completed_cycles=len(cycles), elapsed_seconds=elapsed,
                    active_cycle_seconds=active_seconds,
                    inputs_and_code_unchanged=inputs_unchanged, functional_checks_passed=functional,
                    source_unchanged=bool(before is not None and after is not None
                        and all(before.get(key) == after.get(key) for key in ('files', 'datasets'))),
                    code_unchanged=bool(before is not None and after is not None
                        and all(before.get(key) == after.get(key) for key in ('production_code', 'tools'))),
                    sustained_hour_qualified=qualified, gc=observer.snapshot() if observer is not None else None)
    manifest['qualification_scope'] = ('sustained_hour_qualified means complete one-hour run evidence, '
        'not a memory-stability or leak-absence verdict; review stage-matched RSS trends and aged weak references')
    write_json_strict(output/'memory.json', memory)
    write_json_strict(output/'manifest.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--reconstruction-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration-seconds', type=float, default=3600)
    parser.add_argument('--max-cycles', type=int)
    parser.add_argument('--warmup-cycles', type=int, default=3)
    parser.add_argument('--observation-mode', choices=('minimal', 'weak', 'allocations'), default='weak')
    parser.add_argument('--reuse-save-target', action='store_true',
                        help='Exercise the Qt-style one-request native/portable target replay session')
    parser.add_argument('--preserve-exact-plan',action='store_true',
                        help='Preserve the complete paged exact saved plan; independently reopen native and portable every cycle')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Soak output must be a new directory under this workspace outputs')
    if output.exists():
        raise FileExistsError(output)
    worker = WorkstationCycle(args.project, output, args.reconstruction_report,
                             reuse_save_target=args.reuse_save_target,preserve_exact_plan=args.preserve_exact_plan)
    report = run_session(output, worker, lambda: fixed_inputs(args.project, args.reconstruction_report),
        expected_stages=worker.expected_stages, duration_seconds=args.duration_seconds, max_cycles=args.max_cycles,
        warmup_cycles=args.warmup_cycles, track_partitions=True, observation_mode=args.observation_mode)
    print(json.dumps({key: report[key] for key in (
        'status', 'stop_reason', 'completed_cycles', 'elapsed_seconds',
        'functional_checks_passed', 'sustained_hour_qualified')}, ensure_ascii=False), flush=True)
    return 0 if report['functional_checks_passed'] and report['stop_reason'] != 'disk_guard' else 1


if __name__ == '__main__':
    raise SystemExit(main())
