"""Run four PCL-inspired RGB-D experiments without publishing planning inputs."""
import argparse
import cProfile
from pathlib import Path
import statistics
import sys
import time
from contextlib import ExitStack
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from femto.io_utils import write_json_strict,sha256_file,load_jsonl
from femto.local.organized_comparison import compare,analytic_normal_comparison
from femto.local.organized_fixture import generate


def profile_current(dataset,output):
    """Profile current production functions; no latest/history publication."""
    from femto.local.local_pose_refinement import refine
    from femto.pipeline.scene_reconstruction import reconstruct_scene
    from femto.pipeline import scene_reconstruction
    from femto.local import local_pose_refinement,local_geometry
    evidence={p:sha256_file(p) for p in Path(dataset).rglob('*') if p.is_file()}
    # Include referenced raw files if a posed dataset points to a parent session.
    for frame in load_jsonl(Path(dataset)/'frames.jsonl'):
        for key in ('rgb','depth','pose'):
            path=(Path(dataset)/frame[key]).resolve();evidence[path]=sha256_file(path)
    output.mkdir(parents=True,exist_ok=False)
    rows=[];wall_stages=[]
    def record(name,elapsed):wall_stages.append(dict(stage=name,elapsed_s=elapsed))
    def measured(function,label):
        def call(*args,**kwargs):
            started=time.perf_counter()
            try:return function(*args,**kwargs)
            finally:record(label,time.perf_counter()-started)
        return call
    def run(name,function):
        profiler=cProfile.Profile();started=time.perf_counter()
        result=profiler.runcall(function);elapsed=time.perf_counter()-started
        profiler.dump_stats(str(output/f'{name}.pstats'))
        stats=[]
        for entry in profiler.getstats():
            code=entry.code
            label=code if isinstance(code,str) else f'{code.co_filename}:{code.co_firstlineno}:{code.co_name}'
            stats.append(dict(function=label,calls=entry.callcount,self_s=entry.inlinetime,cumulative_s=entry.totaltime))
        stats.sort(key=lambda row:row['self_s'],reverse=True)
        rows.append(dict(stage=name,elapsed_s=elapsed,top_self_time=stats[:25],
            selected_functions=[row for row in stats if any(word in row['function'] for word in
                ('estimate_normals','registration_icp','metrics','retain_observed_triangles','_integrate_frame','write_triangle_mesh','export_mesh_formats'))]))
        return result
    with ExitStack() as stack:
        for module,name,label in ((local_pose_refinement,'solve','icp_solve'),
                (local_pose_refinement,'metrics','held_out_metrics'),
                (local_geometry,'retain_observed_triangles','measured_surface_support'),
                (scene_reconstruction,'_integrate_frame','frame_integration_and_cloud'),
                (scene_reconstruction,'export_mesh_formats','mesh_export'),
                (scene_reconstruction,'export_multiview_texture','multiview_texture_export')):
            stack.enter_context(patch.object(module,name,measured(getattr(module,name),label)))
        actual,decision=run('current_refinement',lambda:refine(dataset,output/'refinement'))
        run('current_reconstruction',lambda:reconstruct_scene(Path(actual),output/'reconstruction',
            depth_min_m=.25,depth_max_m=2.21,timing_sink=record))
    unchanged=all(sha256_file(path)==digest for path,digest in evidence.items())
    if not unchanged:raise ValueError('Source changed during profiling')
    report=dict(schema='current-rgbd-profile-v1',source_unchanged=unchanged,refinement_accepted=decision.get('accepted'),
        scope='Production refinement and full-scene reconstruction; profiling overhead and I/O included; no route/save or ROI request',
        wall_stage_totals={name:sum(row['elapsed_s'] for row in wall_stages if row['stage']==name)
                           for name in sorted({row['stage'] for row in wall_stages})},
        wall_stage_calls=wall_stages,stages=rows,
        profile_note='Explicit perf_counter wall times cover native calls; cProfile may omit pybind scopes. Nested stage times must not be summed.')
    write_json_strict(output/'profile.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--dataset',type=Path)
    inputs.add_argument('--synthetic',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--scale',type=float,default=.5)
    parser.add_argument('--width',type=int,default=320,help='Synthetic input width')
    parser.add_argument('--height',type=int,default=240,help='Synthetic input height')
    parser.add_argument('--profile-current',action='store_true',help='Also profile production refinement and reconstruction')
    args=parser.parse_args()
    if not 1<=args.repeats<=10:parser.error('repeats must be 1 to 10')
    output=args.output.resolve()
    if output.exists():parser.error('output already exists; choose a new directory')
    if args.dataset and (output==args.dataset.resolve() or args.dataset.resolve() in output.parents):
        parser.error('output must be outside the input dataset')
    output.mkdir(parents=True,exist_ok=False)
    dataset=args.dataset or generate(output/'synthetic_inputs',width=args.width,height=args.height)
    reports=[]
    for repeat in range(args.repeats):
        report=compare(dataset,output/f'run_{repeat+1}',scale=args.scale,progress=lambda s:print(s,flush=True))
        reports.append(report)
    frames=[row for report in reports for row in report['frames']]
    summary=dict(schema='organized-rgbd-experiment-summary-v1',dataset=str(dataset),synthetic=args.synthetic,
        repeats=args.repeats,scale=args.scale,source_unchanged=all(r['source_unchanged'] for r in reports),
        median_normal_timing_s={name:statistics.median(row['timing_s'][name] for row in frames)
            for name in ('integral_normals','kd_normals','edge_classification','fit_integral_normals','normal_space_sampling')},
        normal_coverage_range=[min(row['normal_coverage'] for row in frames),max(row['normal_coverage'] for row in frames)],
        normal_p95_difference_deg_vs_kd=[row['normal_difference_vs_kd']['p95_deg'] for row in reports[0]['frames']],
        frame_diagnostics=reports[0]['frames'],methods={},published=False)
    for method in ('baseline_icp','projective','normal_space_icp'):
        rows=[row for report in reports for pair in report['pairs'] for row in pair['methods'] if row['method']==method]
        summary['methods'][method]=dict(attempts=len(rows),passed=sum(row['would_pass_existing_gate'] for row in rows),
            median_pair_elapsed_s=statistics.median(r['elapsed_s'] for r in rows) if rows else None,
            rejection_reasons=sorted({reason for r in rows for reason in r.get('reasons',[])}),
            all_view_decisions=[r['all_view_decisions'][method] for r in reports])
    summary['timing_scope']='Already-imported process; frame preparation and pair solve/validation are separate. Not acquisition/fusion/route/save total.'
    summary['normal_space_extra_preparation']='Fit integral normals and sampling are reported separately; pair time alone is not total method cost.'
    summary['analytic_normal_comparison']=analytic_normal_comparison()
    if args.profile_current:summary['current_profile']=profile_current(dataset,output/'current_profile')
    write_json_strict(output/'summary.json',summary)
    print(output/'summary.json',flush=True)


if __name__=='__main__':main()
