"""Same-frontier real target replay; separate from full generation/save timing."""
from pathlib import Path
import argparse,json,sys,time
from contextlib import nullcontext


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--code-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--detail',action='store_true')
    args=parser.parse_args();sys.path.insert(0,str(args.code_root.resolve()))
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from benchmark_observations import Observation,source_identity
    from femto.performance_detail import collect
    from femto.planning import target_surface as target
    from femto.planning.planning_revision import array_record
    environment=source_identity(args.code_root,args.output)
    native=Path(json.loads((args.run/'result.json').read_text(encoding='utf-8'))['native'])
    route=json.loads((native/'trajectory.json').read_text(encoding='utf-8'))
    metadata=json.loads((native/'target_surface.json').read_text(encoding='utf-8'))
    with np.load(native/'target_surface.npz',allow_pickle=False) as asset:
        arrays={name:asset[name] for name in asset.files}
    inputs={name:arrays[name] for name in ('vertices','faces','region_faces','depth','image_valid','mask','quality_classes','face_quality')}
    expected={name:array_record(value) for name,value in arrays.items()}
    with Observation() as observation,(collect() if args.detail else nullcontext()) as detail:
        started=time.perf_counter()
        actual,metrics=target._construct(inputs,route['planning_context']['revision'],
            target.TargetBudget(**metadata['budget']),lambda:None,frontier=metadata['metrics'])
        seconds=time.perf_counter()-started
    same=metrics==metadata['metrics'] and {name:array_record(value) for name,value in actual.items()}==expected
    result=dict(passed=same,seconds=seconds,metrics=metrics,environment=environment,resources=observation.result,
        source=str(native),scope='fixed_frontier_target_replay_only',detail=detail.snapshot() if detail is not None else None)
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=same,seconds=seconds,output=str(args.output)),ensure_ascii=False))
    return 0 if same else 1


if __name__=='__main__':raise SystemExit(main())
