"""A/B only the depth preparation, replaying the same real saved target frontier."""
from pathlib import Path
import argparse,json,sys,time
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--code-root',required=True,type=Path,help='Frozen trial implementation, e.g. outputs/equivalent_speed_20261103/candidate_v1')
    args=parser.parse_args()
    root=args.code_root.resolve();sys.path.insert(0,str(root))
    import numpy as np
    from benchmark_observations import Observation,source_identity
    from femto.planning import target_surface as target
    from femto.planning.planning_revision import array_record
    from femto.performance_detail import collect
    from inspect import signature
    if 'depth_ranges' not in signature(target._pixel_target_children).parameters:
        raise ValueError('Specify frozen trial code; the depth experiment is not enabled in production')
    args.output.mkdir(parents=True,exist_ok=False)
    environment=source_identity(root,args.output)
    report=json.loads((args.run/'result.json').read_text(encoding='utf-8'))
    native=Path(report['native'])
    route=json.loads((native/'trajectory.json').read_text(encoding='utf-8'))
    metadata=json.loads((native/'target_surface.json').read_text(encoding='utf-8'))
    with np.load(native/'target_surface.npz',allow_pickle=False) as asset:arrays={name:asset[name] for name in asset.files}
    source_names=('vertices','faces','region_faces','depth','image_valid','mask','quality_classes','face_quality')
    inputs={name:arrays[name] for name in source_names}
    expected={name:array_record(value) for name,value in arrays.items()}
    original=target._PixelClipBatches.__init__
    def disable(instance,*args,**kwargs):
        original(instance,*args,**kwargs);instance.current_depth_ranges=None
    rows=[]
    # Reverse the order on the second pair; this is still one process, not an
    # OS cold-cache claim. Every construction creates its own bounded caches.
    for enabled in (False,True,True,False):
        with Observation() as observation,collect() as detail,patch.object(target._PixelClipBatches,'__init__',original if enabled else disable):
            started=time.perf_counter()
            result,metrics=target._construct(inputs,route['planning_context']['revision'],target.TargetBudget(**metadata['budget']),lambda:None,frontier=metadata['metrics'])
            seconds=time.perf_counter()-started
        same=({name:array_record(value) for name,value in result.items()}==expected and metrics==metadata['metrics'])
        rows.append(dict(depth_preparation=enabled,seconds=seconds,same=same,resources=observation.result,detail=detail.snapshot()))
        print(json.dumps(dict(enabled=enabled,seconds=seconds,same=same)),flush=True)
    result=dict(environment=environment,source=str(native),scope='fixed_frontier_target_rebuild_only',
                same_input_and_geometry=True,rows=rows,passed=all(row['same'] for row in rows))
    (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
