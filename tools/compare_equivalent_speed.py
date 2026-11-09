"""Verify geometric results separately from run locations and creation times."""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_artifacts(root,result):
    native=Path(result['native'])
    route=read(native/'trajectory.json')
    source=route['source'];manifest_path=Path(source['region_manifest'])
    assert manifest_path.is_relative_to(root/'model'/'planning_inputs')
    assert manifest_path.parent==Path(source['region_export_dir'])
    assert sha(manifest_path)==source['region_manifest_sha256']
    manifest=read(manifest_path)
    model_mesh=str(root/'model'/'scene_tsdf_mesh_m.ply')
    assert route['data_layers']['source_mesh']==model_mesh
    assert all(value==model_mesh for value in (manifest['source_mesh'],manifest['region']['source_mesh'],manifest['data_layers']['source_mesh']))
    normalized=deepcopy(route);normalized_manifest=deepcopy(manifest)
    normalized['data_layers']['source_mesh']='<RUN_MODEL>'
    normalized['source']['region_manifest']='<VERIFIED_REGION_MANIFEST>'
    normalized['source']['region_export_dir']='<REGION_EXPORT>'
    normalized['source']['region_manifest_sha256']='<HASH_VERIFIED_ABOVE>'
    for node in (normalized_manifest,normalized_manifest['region'],normalized_manifest['data_layers']):node['source_mesh']='<RUN_MODEL>'
    # These exact fields are display timestamps, not geometry/replay inputs.
    for field in ('created_at','updated_at'):
        assert isinstance(normalized_manifest['region'][field],str)
        normalized_manifest['region'][field]='<RUN_TIME>'
    return normalized,normalized_manifest,route['result_sha256']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();results=[];artifacts=[]
    for folder in args.runs:
        root=folder.resolve();result=read(root/'result.json')
        assert result['status']=='passed',root
        assert result['environment']['code_tree_sha256']==sha(root/'source_manifest.json')
        artifacts.append(verified_artifacts(root,result));results.append(result)
    first=results[0]
    comparisons=[]
    for folder,result,artifact in zip(args.runs,results,artifacts):
        equal={key:result[key]==first[key] for key in ('planning_mesh_sha256','partition','source_surface_check','equivalence','source_file_sha256')}
        equal['native_route_after_explicit_metadata_normalization']=artifact[0]==artifacts[0][0]
        equal['native_region_after_explicit_metadata_normalization']=artifact[1]==artifacts[0][1]
        equal['native_result_sha256']=artifact[2]==artifacts[0][2]
        if result.get('includes_reconstruction'):
            equal['rebuilt_source_bytes_and_metrics']=all(result['reconstruction_equivalence'][key] for key in ('metrics_equal','mesh_bytes_equal','cloud_bytes_equal'))
        comparisons.append(dict(run=str(folder),equal=equal,timings=result['timings'],resources=result['resources'],
            wall_seconds=result['wall_seconds'],save_complete_wall_seconds=result['save_complete_wall_seconds'],
            detailed_instrumentation=result['detailed_instrumentation'],scope=result['scope'],
            code_tree_sha256=result['environment']['code_tree_sha256']))
    report=dict(passed=all(all(row['equal'].values()) for row in comparisons),comparisons=comparisons,
        ignored_metadata_only=['run model path','verified region export directory and manifest path',
            'independently verified manifest bytes SHA-256','region.created_at','region.updated_at'],
        budget_and_failure_evidence='Separate regression tests; time-budget frontier invariance is not claimed')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],runs=len(results),output=str(args.output)),ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
