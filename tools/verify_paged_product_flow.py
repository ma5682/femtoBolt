"""Actual saved project → target continuation → coverage → native/portable replay."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from femto.io_utils import write_json_strict
from femto.annotation.fixed_rgb_geometry import GeometryBudget
from femto.planning import target_continuation, coverage_workflow, coverage_certificate
from femto.planning import surface_workspace, region_project
from femto.planning.trajectory_export import load_trajectory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();output=args.output.resolve();project=args.project.resolve()
    if output.exists():raise FileExistsError('Use a new verification directory')
    output.mkdir(parents=True,exist_ok=False)
    source_digest=hashlib.sha256(project.read_bytes()).hexdigest()
    rows=[]
    def run(name,fn):
        print(name,flush=True);started=time.monotonic();check=GeometryBudget().check
        try:
            value=fn(check);check();rows.append(dict(name=name,status='passed',seconds=time.monotonic()-started))
        except Exception as error:
            rows.append(dict(name=name,status='failed',seconds=time.monotonic()-started,error=repr(error)))
            write_json_strict(output/'stages.json',rows);raise
        write_json_strict(output/'stages.json',rows);return value
    model=run('open_original',lambda check:region_project.open_project(project,workspace=output/'input',progress=lambda _:check()))
    region=model.store.regions[0];original=model.saved_plan
    plan,ledger=run('continue_target',lambda check:target_continuation.calculate(
        model,region,original,output/'continuation',check,print))
    model.target_ledger=ledger;model.coverage_certificate=None;model.coverage_display=None
    certificate,_=run('calculate_full_target_coverage',lambda check:coverage_workflow.calculate(
        model,region,plan,check,print,budget=coverage_certificate.CoverageBudget(seconds=30)))
    plan=run('attach_full_target_coverage',lambda check:coverage_workflow.attach(model,region,plan,certificate,ledger,check=check))
    native=run('native_save',lambda check:surface_workspace.export(model,region,plan,check,print))
    replay=run('native_independent_read',lambda check:load_trajectory(native,check=check,reproduce=True))
    portable=run('portable_save',lambda check:region_project.save_project(model,region,output/'continued.region.zip',plan,check,progress=print))
    opened=run('portable_independent_read',lambda check:region_project.open_project(portable,workspace=output/'opened',progress=lambda _:check()))
    unchanged=hashlib.sha256(project.read_bytes()).hexdigest()==source_digest
    same=(opened.target_ledger.metadata==ledger.metadata and opened.coverage_certificate.metadata==certificate.metadata
        and opened.saved_plan['surface_trajectory']['ordered_items']==original['surface_trajectory']['ordered_items']
        and replay.planning_context['target']==ledger.metadata)
    report=dict(scope='real_saved_project_software_flow',all_checks_passed=unchanged and same,
        source_sha256=source_digest,source_unchanged=unchanged,roundtrip_equal=same,stages=rows,
        original_target_summary=original['planning_context']['target']['summary'],
        continued_target_summary=ledger.summary,coverage_summary=certificate.summary,
        native=str(native),portable=str(portable),execution_ready=False,physical_accuracy_verified=False)
    write_json_strict(output/'report.json',report)
    if not report['all_checks_passed']:raise RuntimeError('Paged product flow mismatch')
    print('all_checks_passed',flush=True)


if __name__=='__main__':main()
