"""Count every original point; detailed timings are separate from the report."""
from pathlib import Path
import argparse,json,sys,time
from contextlib import nullcontext


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--oriented-experiment',action='store_true')
    parser.add_argument('--csr-experiment',action='store_true')
    parser.add_argument('--native-complete-experiment',action='store_true')
    parser.add_argument('--baseline-queries',action='store_true')
    parser.add_argument('--native-pairs',type=int)
    parser.add_argument('--code-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--no-detail',action='store_true')
    args=parser.parse_args()
    sys.path.insert(0,str(args.code_root.resolve()))
    from femto.geometry.fusion import read_geometry_safe
    from femto.geometry.cloud_layer_consistency import count_second_sheet_points
    from femto.performance_detail import collect
    from femto.operation_diagnostics import record_operation
    if args.oriented_experiment:
        from functools import partial
        from femto.geometry import cloud_layer_batches
        cloud_layer_batches.negative_bounds=partial(cloud_layer_batches.negative_bounds,oriented=True)
    if args.baseline_queries:
        from femto.geometry import cloud_layer_batches
        cloud_layer_batches._complete_radius_counter=lambda p,n,legacy,check:legacy
    if args.csr_experiment:
        from femto.geometry import cloud_layer_batches
        from femto.geometry.cloud_csr_research import CellRadiusCounter
        cloud_layer_batches._complete_radius_counter=lambda p,n,legacy,check:CellRadiusCounter(p,n,legacy,check=check)
    if args.native_complete_experiment:
        from femto.geometry import cloud_layer_batches
        from femto.geometry.cloud_radius_batches_research import NativeRadiusCounter
        cloud_layer_batches._complete_radius_counter=lambda p,n,legacy,check:NativeRadiusCounter(p,n,legacy,check=check)
    if args.native_pairs is not None:
        from femto.geometry import cloud_layer_batches
        from femto.geometry.cloud_radius_batches import NativeRadiusCounter
        cloud_layer_batches._complete_radius_counter=lambda p,n,legacy,check:NativeRadiusCounter(p,n,legacy,check=check,max_pairs=args.native_pairs)
    args.output.mkdir(parents=True,exist_ok=False)
    from benchmark_observations import Observation,source_identity
    environment=source_identity(args.code_root,args.output)
    cloud=read_geometry_safe(args.source/'scene_fused_points_m.ply',mesh=False)
    report=json.loads((args.source/'reconstruction_report.json').read_text(encoding='utf-8'))
    with Observation() as observation,(nullcontext() if args.no_detail else collect()) as detail,record_operation('cloud.consistency',root=args.output/'diagnostics'):
        start=time.perf_counter();count=count_second_sheet_points(cloud);seconds=time.perf_counter()-start
    expected=round(report['metrics']['second_sheet_fraction']*len(cloud.points))
    result=dict(count=count,expected=expected,seconds=seconds,same=count==expected,
                oriented_experiment=args.oriented_experiment,csr_experiment=args.csr_experiment,
                native_complete_experiment=args.native_complete_experiment,detail=detail.snapshot() if detail is not None else None)
    result['baseline_queries']=args.baseline_queries
    result['native_pairs']=args.native_pairs
    result.update(environment=environment,resources=observation.result)
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return 0 if count==expected else 1


if __name__=='__main__':raise SystemExit(main())
