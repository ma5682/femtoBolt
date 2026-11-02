"""Replay a real saved project through exact partition, coverage and archives."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from femto.io_utils import write_json_strict
from femto.annotation.fixed_rgb_geometry import GeometryBudget
from femto.planning import (target_reconstruction,coverage_workflow,coverage_certificate,
                            surface_workspace,region_project,planning_revision)
from femto.planning.trajectory_export import load_trajectory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();project=args.project.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True,exist_ok=False)
    source_hash=hashlib.sha256(project.read_bytes()).hexdigest()
    write_json_strict(output/'source_start.json',dict(path=str(project),sha256=source_hash))
    rows=[]
    def run(name,fn):
        print(name,flush=True);start=time.monotonic();check=GeometryBudget().check
        try:
            value=fn(check);check();rows.append(dict(name=name,status='passed',seconds=time.monotonic()-start))
        except Exception as error:
            rows.append(dict(name=name,status='failed',seconds=time.monotonic()-start,error=repr(error)))
            write_json_strict(output/'stages.json',rows);raise
        write_json_strict(output/'stages.json',rows);return value
    model=run('open_original',lambda check:region_project.open_project(project,workspace=output/'input',progress=lambda _:check()))
    region=model.store.regions[0];original=model.saved_plan
    plan,ledger=run('exact_partition',lambda check:target_reconstruction.calculate(model,region,original,
        output/'exact_partition',check,lambda value:print(value,flush=True)))
    model.target_ledger=ledger;model.coverage_certificate=None;model.coverage_display=None
    preliminary=dict(old_summary=original['planning_context']['target']['summary'],new_summary=ledger.summary,
        changed_dependencies=planning_revision.changed(original['planning_context']['revision'],plan['planning_context']['revision']),
        same_ordered_items=original['surface_trajectory']['ordered_items']==plan['surface_trajectory']['ordered_items'],
        source_unchanged=hashlib.sha256(project.read_bytes()).hexdigest()==source_hash)
    write_json_strict(output/'partition_result.json',preliminary)
    certificate,_=run('new_target_coverage',lambda check:coverage_workflow.calculate(model,region,plan,check,
        print,budget=coverage_certificate.CoverageBudget(seconds=30)))
    plan=run('attach',lambda check:coverage_workflow.attach(model,region,plan,certificate,ledger,check=check))
    write_json_strict(output/'coverage_summary.json',certificate.summary)
    native=run('native_save',lambda check:surface_workspace.export(model,region,plan,check,print))
    replay=run('native_independent_read',lambda check:load_trajectory(native,check=check,reproduce=True))
    portable=run('portable_save',lambda check:region_project.save_project(model,region,output/'exact.region.zip',plan,check,progress=print))
    opened=run('portable_independent_read',lambda check:region_project.open_project(portable,workspace=output/'opened',progress=lambda _:check()))
    same=(opened.target_ledger.metadata==ledger.metadata and opened.coverage_certificate.metadata==certificate.metadata
          and replay.planning_context['target']==ledger.metadata)
    unchanged=hashlib.sha256(project.read_bytes()).hexdigest()==source_hash
    report=dict(preliminary,all_checks_passed=same and unchanged,source_unchanged=unchanged,
        roundtrip_equal=same,stages=rows,coverage_summary=certificate.summary,native=str(native),portable=str(portable),
        execution_ready=False,physical_accuracy_verified=False)
    write_json_strict(output/'report.json',report)
    if not report['all_checks_passed']:raise RuntimeError('Exact target flow mismatch')
    print('all_checks_passed',flush=True)


if __name__=='__main__':main()
