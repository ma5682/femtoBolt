"""Repeat nominal coverage/reproof on declared meshes and a verified project.

Every attempt and failure gets a new record. This is a numerical/history replay
benchmark; it does not collect frames, publish a project or validate hardware.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil

from femto.operation_diagnostics import record_operation
from femto.planning import coverage_archive, coverage_certificate as coverage, coverage_partition
from femto.planning import coverage_refinement, coverage_residual, coverage_interleaved, coverage_workflow, footprint_model
from femto.planning import refinement_archive, refinement_candidates_v2, region_project
from femto.planning.coverage_domain import CoverageDomain, scan_endpoints
from femto.planning.planning_revision import array_record
from femto.planning.surface_trajectory import SurfaceTrajectory


def sha(path: Path) -> str:
    with path.open('rb') as stream:return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_files() -> dict:
    return {p.relative_to(ROOT).as_posix():sha(p) for folder in ('femto','tools','tests')
            for p in sorted((ROOT/folder).rglob('*.py'))}


def write(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def coverage_values(certificate: coverage.Certificate, areas: np.ndarray) -> dict:
    arrays = certificate.arrays
    states = {}
    for code, name in coverage.STATES.items():
        selected = np.flatnonzero(arrays['status'] == code)
        states[name] = dict(leaf_count=len(selected),
            weight_units=sum(coverage_partition.UNITS//4**int(arrays['depth'][i]) for i in selected),
            nominal_area_mm2=math.fsum(float(areas[int(arrays['root'][i])])/4**int(arrays['depth'][i])
                                      for i in selected)*1e6)
    return dict(original_cell_count=len(areas), leaf_count=len(arrays['root']),
        unlocated_pixel_count=certificate.summary['unlocated_pixel_count'], states=states,
        area_semantics='unchanged_original_nominal_root_weights_not_physical_accuracy')


def comparison(before, after, domain, new_seeds=None) -> dict:
    mask = None
    if new_seeds is not None:
        a = after.arrays; selected = a['status'] == 1
        mask = np.zeros(len(selected), bool)
        mask[selected] = new_seeds[a['seed_input_index'][a['proof_seed'][a['witness'][selected]]]]
    return coverage_partition.compare(before, after, domain.areas, new_witnesses=mask)


class Measurements:
    def __init__(self, output: Path, seconds: float = 120.) -> None:
        self.output, self.seconds, self.rows = output, seconds, []

    def run(self, sample: str, repetition: int, action: str, function: Callable, summarize: Callable = lambda value: {}) -> object:
        label = f'{sample}_{repetition}_{action}'
        started, cpu = time.perf_counter(), time.process_time()
        process = psutil.Process(); peak = [process.memory_info().rss]
        stopped = threading.Event()
        def watch():
            while not stopped.wait(.02):peak[0] = max(peak[0], process.memory_info().rss)
        watcher = threading.Thread(target=watch, daemon=True); watcher.start()
        row = dict(sample=sample, repetition=repetition, action=action, status='failed')
        value = None
        def check():
            if time.perf_counter()-started > self.seconds:raise TimeoutError('Original 120-second operation deadline')
        print(json.dumps(dict(sample=sample, repetition=repetition, action=action, status='started')), flush=True)
        try:
            with record_operation(label) as operation:
                value = function(check)
                check()
                # Accounting/asset writes are reported separately from calculation.
                row.update(status='complete', seconds=time.perf_counter()-started,
                           cpu_seconds=time.process_time()-cpu)
                observed = summarize(value)
                row.update(observed)
                row['diagnostics'] = None if operation.directory is None else str(operation.directory.relative_to(self.output))
                if row.get('acceptance_errors'):raise AssertionError('; '.join(row['acceptance_errors']))
        except Exception as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error))
            row.setdefault('seconds', time.perf_counter()-started)
            row.setdefault('cpu_seconds', time.process_time()-cpu)
            value = None
        finally:
            stopped.set(); watcher.join()
        row.update(sampled_rss_peak_bytes=peak[0], rss_sample_seconds=.02,
                   total_observed_seconds=time.perf_counter()-started)
        self.rows.append(row); write(self.output/(label+'.json'), row)
        print(json.dumps({k:row[k] for k in ('sample','repetition','action','status','seconds')}, ensure_ascii=False), flush=True)
        return value


def grid(size: int) -> tuple:
    vertices = np.asarray([(x*.002, y*.002, .5) for y in range(size+1) for x in range(size+1)])
    faces = []
    for y in range(size):
        for x in range(size):
            a = y*(size+1)+x
            faces.extend(((a,a+1,a+size+2),(a,a+size+2,a+size+1)))
    faces = np.asarray(faces, '<i8')
    def seeds(offset):
        return [dict(source_face=i, xyz_m=vertices[f].mean(axis=0).tolist(), role='scan')
                for i,f in enumerate(faces) if i%16 == offset]
    return vertices, faces, seeds(0), seeds(8), footprint_model.nominal(5.)


def proof_summary(folder: Path, certificate, domain, prior=None, added=None) -> dict:
    folder.mkdir(exist_ok=False)
    coverage_archive.write_assets(folder, certificate)
    result = dict(values=coverage_values(certificate, domain.areas), metrics=certificate.metadata['metrics'],
                  certificate_sha256=certificate.metadata['sha256'])
    if prior is not None:
        result['comparison'] = comparison(prior, certificate, domain, added)
        before = coverage_values(prior,domain.areas)['states']['budget_unprocessed']
        after = result['values']['states']['budget_unprocessed']
        result['budget_change'] = dict(before=before,after=after,
            delta={key:after[key]-before[key] for key in before},
            meaning='signed_after_minus_before_on_the_same_original_nominal_target')
    return result


def numerical_cases(measurements: Measurements, repeats: int, replay: bool = False) -> None:
    for size, sample in ((8,'small_declared_mesh'),(32,'medium_declared_mesh')):
        vertices, faces, seeds, extra, model = grid(size)
        first = None
        for repetition in range(repeats):
            domain = CoverageDomain.mesh(vertices, faces)
            certificate = measurements.run(sample, repetition, 'coverage',
                lambda check: coverage.evaluate(domain, seeds, model, check=check,
                    budget=coverage.CoverageBudget(seconds=30), algorithm=coverage.CURRENT_ALGORITHM),
                lambda value: proof_summary(measurements.output/f'{sample}_{repetition}_coverage_assets',value,domain,first))
            if certificate is None:continue
            if first is None:first = certificate
            domain = CoverageDomain.mesh(vertices, faces)
            measurements.run(sample, repetition, 'recalculation',
                lambda check: coverage.evaluate(domain, seeds, model, check=check, guide=first,
                    budget=coverage.CoverageBudget(seconds=30), algorithm=coverage.RECALCULATION_ALGORITHM),
                lambda value: proof_summary(measurements.output/f'{sample}_{repetition}_recalculation_assets',value,domain,first))
            # A fresh domain keeps per-instance geometry caches independent.
            domain = CoverageDomain.mesh(vertices, faces)
            measurements.run(sample, repetition, 'reproof',
                lambda check: coverage.evaluate(domain, seeds+extra, model, check=check, guide=first,
                    budget=coverage.CoverageBudget(seconds=4), algorithm=coverage.INTERLEAVED_ALGORITHM),
                lambda value: proof_summary(measurements.output/f'{sample}_{repetition}_reproof_assets', value, domain,
                    first, np.arange(len(seeds)+len(extra)) >= len(seeds)))
            if replay:
                for action in ('coverage', 'recalculation', 'reproof'):
                    folder = measurements.output/f'{sample}_{repetition}_{action}_assets'
                    if not folder.exists():continue
                    saved = coverage_archive.load_assets(folder)
                    saved_guide = coverage_archive.load_assets(measurements.output/f'{sample}_0_coverage_assets')
                    domain = CoverageDomain.mesh(vertices, faces)
                    measurements.run(sample, repetition, action+'_replay',
                        lambda check: coverage.validate(saved,domain,seeds+extra if action == 'reproof' else seeds,
                            model,check=check,**({'guide':saved_guide} if action == 'reproof' else {})),
                        lambda _:dict(saved_frontier_exact_replay=True,read_saved_assets=True))


def replay_refinement_assets(folder: Path, target, check) -> None:
    route = SurfaceTrajectory.from_dict(json.loads((folder/'trajectory.json').read_text(encoding='utf-8')))
    ledger = coverage_refinement.rebind_target(target,route.planning_context['revision'])
    final = coverage_archive.load_assets(folder,check)
    history = tuple(coverage_archive.load_assets(folder,check,prefix=f'round_{i}_')
                    for i in range(len(refinement_archive.history_metadata(route))))
    proof = coverage_refinement.RefinedCertificate(final.metadata,final.arrays,history)
    refinement_archive.replay(proof,route,ledger,check)


def project_cases(measurements: Measurements, project: Path, repeats: int, refine: bool, replay: bool, require_gain: bool,
                  *, compare_legacy: bool = False, guide_assets: Path | None = None,
                  require_p2_acceptance: bool = False) -> dict:
    output = measurements.output
    project_sha = sha(project)
    source = measurements.run('real_project',0,'open',
        lambda check: region_project.open_project(project, workspace=output/'input_workspace', progress=lambda _:check()))
    if source is None:raise ValueError('Project validation failed; see the retained operation diagnostics')
    if source.saved_plan is None or source.target_ledger is None:raise ValueError('Project requires a verified route and original target')
    base_record = source.saved_plan['surface_trajectory']
    prior = source.coverage_certificate
    if base_record['planner']['algorithm_version'] in coverage_refinement.ALGORITHMS:
        base_record = base_record['gap_report']['refinement']['base_trajectory']
        prior = prior.history[0]
    base = SurfaceTrajectory.from_dict(base_record)
    if base.planner['algorithm_version'] not in ('quality-constrained-surface-v3','constructed-surface-route-v1'):
        raise ValueError('This benchmark accepts fixed-model v3/P4 routes; restore before P3/P5 as appropriate')
    target = coverage_refinement.rebind_target(source.target_ledger,base.planning_context['revision'])
    if target.metadata != base.planning_context['target']:raise ValueError('Original target binding changed')
    original = {k:array_record(v) for k,v in target.arrays.items()}
    mesh = refinement_archive.local_mesh(target)
    region = base.planning_context['revision']['inputs']['selection']
    first = None
    guide_source = None
    if guide_assets is not None:
        def load_guide(check):
            value = coverage_archive.load_assets(guide_assets,check)
            domain,seeds,model,binding = coverage_workflow.bundle(base,target)
            coverage.validate(value,domain,seeds,model,route_binding=binding,check=check)
            if prior is not None:coverage_workflow.require_recalculation_retention(prior,value,target,check)
            return value
        first = measurements.run('real_project',0,'fixed_guide_replay',load_guide,
            lambda value:dict(saved_frontier_exact_replay=True,certificate_sha256=value.metadata['sha256']))
        if first is None:raise ValueError('The explicit fixed guide did not reproduce or retain the original set')
        guide_source = 'explicit_verified_assets'
    def save_guide():
        folder = output/'fixed_refinement_guide_assets'
        folder.mkdir(exist_ok=False);coverage_archive.write_assets(folder,first)
    if first is not None:save_guide()
    for repetition in range(repeats):
        strategies = [('real_fixed_p4',False)]
        if compare_legacy:strategies.append(('real_fixed_p4_legacy',True))
        if repetition % 2:strategies.reverse()
        for sample,legacy in strategies:
            domain, seeds, model, binding = coverage_workflow.bundle(base,target)
            selected = []
            def summarize_coverage(value):
                row = proof_summary(output/f'{sample}_{repetition}_coverage_assets',value,domain,prior)
                accepted = True
                if prior is not None:
                    try:coverage_workflow.require_recalculation_retention(prior,value,target)
                    except coverage_workflow.RecalculationRejected:accepted = False
                retained = value if accepted else prior
                selected.append(retained)
                row['recalculation'] = dict(
                    status='accepted' if accepted else 'retained_previous', candidate_installed=accepted,
                    reason=None if accepted else 'original_confirmed_set_not_reproved',
                    previous_certificate_sha256=None if prior is None else prior.metadata['sha256'],
                    selected_certificate_sha256=retained.metadata['sha256'],
                    selected_values=coverage_values(retained,domain.areas),
                    selected_comparison=None if prior is None else comparison(prior,retained,domain))
                row['acceptance_errors'] = (['Current P2 did not reprove the original confirmed set']
                    if require_p2_acceptance and not legacy and not accepted else [])
                return row
            algorithm = coverage.CURRENT_ALGORITHM if legacy or prior is None else coverage.RECALCULATION_ALGORITHM
            fresh = measurements.run(sample,repetition,'coverage',
                lambda check: coverage.evaluate(domain,seeds,model,route_binding=binding,check=check,
                    budget=coverage.CoverageBudget(seconds=30),algorithm=algorithm,
                    **({'guide':prior} if algorithm == coverage.RECALCULATION_ALGORITHM else {})),
                summarize_coverage)
            if fresh is not None and not legacy and first is None:
                first = selected[0]
                guide_source = 'first_current_recalculation' if first is fresh else 'retained_original_coverage'
                save_guide()
            # Reopen every saved raw P2 candidate, including rejected candidates
            # and later repeats. P3 replay alone cannot certify these records.
            if replay and (output/f'{sample}_{repetition}_coverage_assets').exists():
                def replay_coverage(check):
                    saved = coverage_archive.load_assets(output/f'{sample}_{repetition}_coverage_assets',check)
                    current_domain,current_seeds,current_model,current_binding = coverage_workflow.bundle(base,target)
                    coverage.validate(saved,current_domain,current_seeds,current_model,route_binding=current_binding,check=check)
                measurements.run(sample,repetition,'coverage_replay',replay_coverage,
                    lambda _:dict(saved_frontier_exact_replay=True,read_saved_assets=True))
        if not refine or first is None:continue
        for sample,legacy in strategies:
            domain, seeds, model, binding = coverage_workflow.bundle(base,target)
            def summarize(result):
                route, ledger, proof = result
                report = route.gap_report['refinement']
                folder = output/f'{sample}_{repetition}_refinement_assets'
                final_domain, current_seeds, _, _ = coverage_workflow.bundle(route,ledger)
                old_seeds = {(r['source_face'],*r['xyz_m']) for r in seeds}
                added = np.array([(r['source_face'],*r['xyz_m']) not in old_seeds for r in current_seeds],bool)
                values = proof_summary(folder,proof,final_domain,first,added)
                for index,item in enumerate(proof.history):coverage_archive.write_assets(folder,item,prefix=f'round_{index}_')
                write(folder/'trajectory.json',route.to_dict())
                values.update(fixed_guide_sha256=first.metadata['sha256'], rounds=report['rounds'],
                    accepted_rounds=sum(row['accepted'] for row in report['rounds']),
                    coverage_work=report['coverage_work'],
                    original_target_arrays_unchanged=original == {k:array_record(v) for k,v in ledger.arrays.items()})
                errors = []
                if values['comparison']['lost_weight_units'] or not values['original_target_arrays_unchanged']:
                    errors.append('Original confirmed set or target arrays were not retained')
                if require_gain and not legacy and not values['accepted_rounds']:errors.append('No real candidate was accepted in this repeat')
                values['acceptance_errors'] = errors
                return values
            result = measurements.run(sample,repetition,'refinement',
                lambda check: coverage_refinement.refine(base,target,first,mesh,region,check=check,
                    candidate_policy=deepcopy(refinement_candidates_v2.POLICY),
                    coverage_policy=deepcopy(coverage_residual.POLICY if legacy else coverage_interleaved.POLICY)), summarize)
            if result is not None and replay:
                measurements.run(sample,repetition,'refinement_replay',
                    lambda check: replay_refinement_assets(output/f'{sample}_{repetition}_refinement_assets',target,check),
                    lambda _:dict(saved_frontier_exact_replay=True,read_saved_assets=True))
    if sha(project) != project_sha:raise AssertionError('Source project was changed')
    if original != {k:array_record(v) for k,v in target.arrays.items()}:raise AssertionError('Original target arrays changed')
    return dict(path=str(project), sha256=project_sha, unchanged=True, original_target_arrays_unchanged=True,
                fixed_recalculation_guide_sha256=None if prior is None else prior.metadata['sha256'],
                refinement_guide_sha256=None if first is None else first.metadata['sha256'],
                refinement_guide_source=guide_source, refinement_guide_assets='fixed_refinement_guide_assets')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='覆盖复算多规模／重复计时、集合保留与持久诊断')
    parser.add_argument('--output',type=Path,required=True,help='当前工作区 outputs 内的新目录')
    parser.add_argument('--project',type=Path,help='同一固定模型的已保存 .region.zip；不访问相机')
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--refine',action='store_true',help='重复当前 P3；固定第一次当前 P2 接纳后的覆盖作为引导')
    parser.add_argument('--replay',action='store_true',help='从保存资产逐次完整回放数值案例、每个 P2 候选与每份 P3 历史')
    parser.add_argument('--require-gain',action='store_true',help='真实补线重复中无接纳时明确记录该次失败')
    parser.add_argument('--compare-legacy',action='store_true',help='在相同固定起点交替执行旧 P2／P3 v4 对照')
    parser.add_argument('--refinement-guide-assets',type=Path,help='可选：先完整回放指定覆盖资产，再固定用作全部 P3 的起点')
    parser.add_argument('--require-p2-acceptance',action='store_true',help='当前 P2 复算若未接纳，明确记录该次失败')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not 1 <= args.repetitions <= 10:parser.error('重复次数必须为 1–10')
    if args.require_gain and not (args.refine and args.project):parser.error('--require-gain 需要 --refine 和 --project')
    if (args.compare_legacy or args.require_p2_acceptance) and not args.project:parser.error('真实对照与 P2 接纳验收需要 --project')
    if args.refinement_guide_assets and not (args.refine and args.project):parser.error('固定 P3 引导需要 --refine 和 --project')
    if output == ROOT/'outputs' or not output.is_relative_to((ROOT/'outputs').resolve()):parser.error('输出必须位于本工作区 outputs 的新子目录')
    if output.exists():parser.error('输出目录已存在，请指定新目录')
    output.mkdir(parents=True,exist_ok=False)
    os.environ['FEMTO_DIAGNOSTICS_DIR'] = str(output/'diagnostics')
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    frozen = source_files(); write(output/'source_at_start.json',frozen)
    measurements = Measurements(output)
    source, failure = None, None
    try:
        numerical_cases(measurements,args.repetitions,args.replay)
        if args.project:source = project_cases(measurements,args.project.resolve(),args.repetitions,args.refine,args.replay,args.require_gain,
            compare_legacy=args.compare_legacy,guide_assets=args.refinement_guide_assets,
            require_p2_acceptance=args.require_p2_acceptance)
    except Exception as error:
        failure = dict(type=type(error).__name__,message=str(error))
    unchanged = source_files() == frozen
    groups = {}
    for row in measurements.rows:
        if row['status'] == 'complete':groups.setdefault((row['sample'],row['action']),[]).append(row['seconds'])
    summaries = [dict(sample=sample,action=action,complete_runs=len(values),minimum_seconds=min(values),
        median_seconds=statistics.median(values),maximum_seconds=max(values)) for (sample,action),values in groups.items()]
    okay = unchanged and failure is None and all(row['status'] == 'complete' for row in measurements.rows)
    write(output/'result.json',dict(schema='femto-coverage-stability-benchmark-v2',repetitions=args.repetitions,
        matched_legacy_comparison=args.compare_legacy,require_current_p2_acceptance=args.require_p2_acceptance,
        budgets=dict(operation_seconds=120,coverage_seconds=30,reproof_seconds=4), source=source,
        source_python_unchanged=unchanged,all_attempts_retained=True, rows=measurements.rows, timings=summaries,
        measurement_scope='validated_input_numerical_operations_separate_from_Qt_installation',
        failure=failure,all_checks_passed=okay,execution_ready=False,physical_accuracy_verified=False))
    print(json.dumps(dict(output=str(output),all_checks_passed=okay),ensure_ascii=False),flush=True)
    return 0 if okay else 1


if __name__ == '__main__':raise SystemExit(main())
