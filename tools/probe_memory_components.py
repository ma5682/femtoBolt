"""Allocation diagnosis of bounded components, never full-flow timing evidence."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import tracemalloc
import inspect
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.runtime_memory import MemoryDiagnostics
from tools.soak_workstation import WeakObjects, NaturalGC, fixed_inputs
from tools.benchmark_workstation import file_digest
from femto.app.qt_planning_io import clone_model
from femto.geometry.fusion import read_geometry_safe
from femto.planning import region_project, target_surface
from femto.annotation.fixed_rgb_geometry import GeometryBudget


class ComponentDiagnostics(MemoryDiagnostics):
    def snapshot(self, *, allocation_detail=False):
        result = super().snapshot(allocation_detail=allocation_detail)
        snapshot = tracemalloc.take_snapshot()
        domains = {}
        for trace in snapshot.traces:
            domains[str(trace.domain)] = domains.get(str(trace.domain), 0)+trace.size
        python = snapshot.filter_traces((tracemalloc.DomainFilter(True, 0),))
        result['snapshot_domain_bytes'] = domains
        result['python_domain_sites'] = [dict(site=str(row.traceback), bytes=row.size, count=row.count)
                                         for row in python.statistics('lineno')[:12]]
        result['domain_note'] = ('Domain 0 is Python allocator tracing; NumPy has a separate domain. '
            'Other base diagnostic site lists include all traced domains. Snapshot metadata is observer overhead.')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component', choices=('clone', 'mesh', 'target', 'reconstruction'), required=True)
    parser.add_argument('--reconstruction-report', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'outputs'):
        raise ValueError('Use a new workspace outputs directory')
    output.mkdir(exist_ok=False)
    code = {str(p.relative_to(ROOT)): file_digest(p) for p in sorted((ROOT/'femto').rglob('*.py'))}
    source_digest = file_digest(args.project)
    # Retained fixed input is intentional and loaded with normal full checks.
    model = region_project.open_project(args.project, workspace=output/'opened')
    context = model.saved_plan['planning_context']
    params = model.saved_plan['surface_trajectory']['planner']['parameters']
    tracker = WeakObjects()
    inputs = fixed_inputs(args.project, args.reconstruction_report)
    (output/'inputs.json').write_text(json.dumps(inputs, indent=2), encoding='utf-8')
    if args.component == 'reconstruction':
        if args.reconstruction_report is None:
            raise ValueError('Reconstruction component requires its fixed report')
        from femto.pipeline.scene_reconstruction import reconstruct_scene
        import open3d as o3d
        report = json.loads(args.reconstruction_report.read_text(encoding='utf-8-sig'))
        dataset = Path(report['source_dataset']).resolve()
        parameters = {key: value for key, value in report['parameters'].items()
                      if key in inspect.signature(reconstruct_scene).parameters}
        reference_mesh = file_digest(args.reconstruction_report.parent/'scene_tsdf_mesh_m.ply')
        original_volume = o3d.pipelines.integration.ScalableTSDFVolume
        def observed_volume(*args, **kwargs):
            volume = original_volume(*args, **kwargs)
            tracker.watch('open3d_tsdf_volume', volume)
            return volume

    def exercise():
        if args.component == 'clone':
            clone = clone_model(model, freeze_inputs=True)
            # Shared immutable target/proof caches belong to the retained base
            # model. Observe only the new owned buffers and native mesh here.
            for name, value in [('model', clone), ('store', clone.store), ('mesh', clone.mesh), ('rgb', clone.rgb)]:
                tracker.watch(name, value)
            for name in ('depth', 'image_valid', 'ids', 'z', 'reasons', 'valid', 'new_support', 'k'):
                if hasattr(clone.projection, name):
                    tracker.watch('projection.'+name, getattr(clone.projection, name))
        elif args.component == 'mesh':
            mesh = read_geometry_safe(Path(model.store.source), mesh=True)
            tracker.watch('open3d_mesh', mesh)
        elif args.component == 'target':
            ledger = target_surface.read(Path(model.store.source).parent, context['target'], context['revision'],
                evidence=params['quality_evidence'], domain=params['effective_domain'],
                check=GeometryBudget().check, reproduce=False)
            tracker.model('loaded_target', ledger)
        else:
            destination = output/f'reconstruction_{tracker.cycle:03d}'
            with patch.object(o3d.pipelines.integration, 'ScalableTSDFVolume', observed_volume):
                rebuilt = reconstruct_scene(dataset, destination, **parameters)
            if (rebuilt['frame_count'] != report['frame_count']
                    or file_digest(destination/'scene_tsdf_mesh_m.ply') != reference_mesh):
                raise AssertionError('Component reconstruction differs from fixed reference')

    result = dict(status='running', component=args.component, cycles=[],
        scope='Component allocation diagnosis; base input retained intentionally; not save or full-flow performance',
        warmup_cycles_excluded=3, forced_gc=False, source_sha256=source_digest, code_sha256=code)
    result['probe_sha256'] = file_digest(Path(__file__))
    try:
        with ComponentDiagnostics('allocations') as diagnostic, NaturalGC() as observer:
            for cycle in range(12):
                tracker.cycle = cycle
                exercise()
                time.sleep(1)
                row = dict(cycle=cycle, memory=diagnostic.snapshot(allocation_detail=True),
                    weak_references=tracker.snapshot(), survivor_details=tracker.survivor_details(),
                    natural_gc=observer.snapshot(), gc_enabled=gc.isenabled())
                result['cycles'].append(row)
                (output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                print(json.dumps(dict(cycle=cycle, rss=row['memory']['rss_bytes'],
                    traced=row['memory']['traced_current_bytes'], numpy=row['memory']['numpy_domain_bytes'],
                    alive=row['weak_references']['alive'])), flush=True)
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['source_unchanged'] = file_digest(args.project) == source_digest
        after = fixed_inputs(args.project, args.reconstruction_report)
        (output/'inputs_after.json').write_text(json.dumps(after, indent=2), encoding='utf-8')
        result['fixed_inputs_unchanged'] = after == inputs
        result['probe_unchanged'] = file_digest(Path(__file__)) == result['probe_sha256']
        result['code_unchanged'] = code == {str(p.relative_to(ROOT)): file_digest(p) for p in sorted((ROOT/'femto').rglob('*.py'))}
        (output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
