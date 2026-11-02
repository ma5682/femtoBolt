"""Continue a saved target within unchanged per-page limits and replay all pages."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from femto.io_utils import load_json, write_json_strict
from femto.planning import target_surface as target, target_paging as paging


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    source=args.workspace.resolve();output=args.output.resolve()
    if output.exists() or source==output or source in output.parents:
        raise ValueError('Use a new separate output directory')
    output.mkdir(parents=True,exist_ok=False)
    project=load_json(source/'project.json');plan=project['plan'];context=plan['planning_context']
    parameters=plan['surface_trajectory']['planner']['parameters']
    quality,domain=parameters['quality_evidence'],parameters['effective_domain']
    paths=[source/name for name in ('project.json',target.MANIFEST,target.ASSET)]
    def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before=hashes();write_json_strict(output/'source_start.json',before)
    started=time.monotonic()
    ledger=target.read(source,context['target'],context['revision'],evidence=quality,domain=domain)
    bundle=paging._seal((ledger,),context['revision'])
    result=paging.resume(bundle,context['revision'],evidence=quality,domain=domain)
    paging.write(output/'pages',result)
    restored=paging.read(output/'pages',context['revision'],evidence=quality,domain=domain)
    report=dict(scope='saved_real_target_page_continuation_not_coverage_certificate',
        source_unchanged=before==hashes(),independent_read_equal=restored.metadata==result.metadata,
        old_summary=bundle.metadata['summary'],new_summary=result.metadata['summary'],
        pages=result.metadata['pages'],cumulative_work=result.metadata['cumulative_work'],
        seconds=time.monotonic()-started)
    write_json_strict(output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if not report['source_unchanged'] or not report['independent_read_equal']:raise RuntimeError('Target paging verification failed')


if __name__=='__main__':main()
