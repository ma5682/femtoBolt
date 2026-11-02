"""Independently check retained artifacts of a completed offline soak run.

Run after the worker exits. This does not turn incomplete runs into successes
and does not interpret memory trends as a proof of absence of leaks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from femto.io_utils import write_json_strict
from tools.audit_workstation_benchmark import plan_values
from tools.benchmark_workstation import file_digest
from tools.profile_saved_plan import archive_values, folder_values
from tools.soak_workstation import canonical_digest


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def audit(root):
    root = Path(root).resolve()
    manifest = read(root/'manifest.json')
    errors = []
    for key, expected in (('status', 'completed'), ('sustained_hour_qualified', True),
                          ('functional_checks_passed', True), ('source_unchanged', True),
                          ('code_unchanged', True), ('forced_gc', False)):
        if manifest.get(key) != expected:
            errors.append(f'{key} is not {expected!r}')
    if manifest.get('active_cycle_seconds', 0) < 3600:
        errors.append('Active workload is shorter than one hour')
    cycles = read(root/'cycles.json')
    stages = read(root/'stages.json')
    before, after = read(root/'inputs.json'), read(root/'inputs_after.json')
    if before != after:
        errors.append('Recorded input/code inventories differ')
    if len(cycles) < max(8, manifest['warmup_cycles']+5):
        errors.append('Insufficient completed cycles')
    if manifest.get('completed_cycles') != len(cycles):
        errors.append('Completed cycle count differs')
    if [row['cycle'] for row in cycles] != list(range(len(cycles))):
        errors.append('Cycle sequence differs')
    if any(row.get('repetition') not in range(len(cycles)) for row in stages):
        errors.append('Stage belongs to an undeclared cycle')
    baseline = cycles[0]['result']['equivalence'] if cycles else None
    exact_mode=bool(cycles and cycles[0]['result'].get('plan_mode')=='preserved_complete_exact')
    exact_input=None;expected_route=None;exact_replays=0
    if exact_mode:
        import zipfile
        from tools.profile_saved_plan import route_digest
        from femto.planning.surface_trajectory import SurfaceTrajectory
        projects=[Path(name) for name in before['files'] if Path(name).suffix.lower()=='.zip']
        if len(projects)!=1:
            errors.append('Exact run must identify one fixed portable input')
        else:
            exact_input=plan_values(projects[0])
            with zipfile.ZipFile(projects[0]) as archive:
                document=json.loads(archive.read('project.json'))
            algorithms=document['plan']['planning_context']['revision']['inputs']['algorithms']
            if algorithms.get('target')!='reference-pixel-exact-partition-v2' or exact_input['target_summary']['leaf_count']<=200000:
                errors.append('Exact run input is a v1 or truncated sample')
            expected_route=route_digest(SurfaceTrajectory.from_dict(document['plan']['surface_trajectory']))
    for cycle in cycles:
        index = cycle['cycle']
        measured = [row for row in stages if row['repetition'] == index]
        if ([row['stage'] for row in measured] != manifest['expected_stages']
                or any(row.get('status') != 'passed' or row.get('sampling_errors') != [] for row in measured)):
            errors.append(f'Cycle {index}: missing, failed or invalid measured stage')
        if cycle.get('status') != 'passed' or cycle['result']['equivalence'] != baseline:
            errors.append(f'Cycle {index}: failed or changed result')
        state = read(root/f'cycle_{index:04d}'/'state.json')
        if state != cycle:
            errors.append(f'Cycle {index}: state/checkpoint mismatch')
        portable = Path(cycle['result']['assets']['portable'])
        if not portable.is_relative_to(root):
            errors.append(f'Cycle {index}: portable artifact outside run')
        elif plan_values(portable) != cycle['result']['equivalence']['generated']:
            errors.append(f'Cycle {index}: retained route/target differs')
        native = Path(cycle['result']['assets']['native'])
        if not native.is_relative_to(root):
            errors.append(f'Cycle {index}: native artifact outside run')
        else:
            _, portable_evidence = archive_values(portable)
            trajectory = read(native/'trajectory.json')
            if (folder_values(native) != portable_evidence
                    or canonical_digest(trajectory['ordered_items'])
                    != cycle['result']['equivalence']['generated']['ordered_route_sha256']):
                errors.append(f'Cycle {index}: native and portable evidence differ')
            if exact_mode:
                from femto.annotation.fixed_rgb_geometry import GeometryBudget
                from femto.planning.trajectory_export import load_trajectory
                from tools.profile_saved_plan import route_digest
                replay=load_trajectory(native,check=GeometryBudget().check,reproduce=True)
                exact_replays+=1
                if expected_route is None or route_digest(replay)!=expected_route:
                    errors.append(f'Cycle {index}: full independent exact native replay differs')
        if exact_mode:
            cancellation=cycle['result'].get('cancellation') or {}
            if (cycle['result'].get('plan_mode')!='preserved_complete_exact'
                    or cycle['result']['equivalence']['original']!=exact_input
                    or cycle['result']['equivalence']['generated']!=exact_input
                    or not cancellation.get('cancelled') or not cancellation.get('portable_absent')
                    or not cancellation.get('source_plan_unchanged')
                    or (root/f'cycle_{index:04d}'/'cancelled.region.zip').exists()):
                errors.append(f'Cycle {index}: exact preservation or cancellation evidence differs')
        reconstruction = cycle['result']['equivalence'].get('reconstruction')
        if reconstruction:
            mesh = root/f'cycle_{index:04d}'/'reconstruction'/'scene_tsdf_mesh_m.ply'
            if file_digest(mesh) != reconstruction['mesh_sha256']:
                errors.append(f'Cycle {index}: retained mesh differs')
    if cycles and read(root/'checkpoint.json') != cycles[-1]:
        errors.append('Final checkpoint is not the final complete cycle')
    # Check current source bytes, not only agreement of old inventory records.
    checked = 0
    for name, expected in before['files'].items():
        checked += 1
        if file_digest(Path(name)) != expected:
            errors.append(f'Fixed input changed: {name}')
    for folder, inventory in before['datasets'].items():
        current = {path.relative_to(folder).as_posix(): file_digest(path)
                   for path in sorted(Path(folder).rglob('*')) if path.is_file()}
        checked += len(inventory)
        if current != inventory:
            errors.append(f'Dataset inventory changed: {folder}')
    current_code = {path.relative_to(ROOT).as_posix(): file_digest(path)
                    for path in sorted((ROOT/'femto').rglob('*.py'))}
    if current_code != before['production_code']:
        errors.append('Current production code differs from measured code')
    for name, expected in before['tools'].items():
        if file_digest(ROOT/'tools'/name) != expected:
            errors.append(f'Measurement tool changed: {name}')
    return dict(schema='workstation-soak-audit-v1', all_checks_passed=not errors,
                errors=errors, cycles_checked=len(cycles), input_files_checked=checked,
                exact_plan_preserved=exact_mode,independent_exact_native_replays=exact_replays,
                memory_interpretation='Review memory.json and aged weak references separately; this audit verifies run evidence.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    write_json_strict(args.run/'audit.json', result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
