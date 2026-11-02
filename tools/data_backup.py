"""Cold asset/database backups and explicit non-overwriting recovery."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from femto import data_backup as storage
from femto.pipeline.data_maintenance import default_roots,inspect_references,backup_workspace


def mappings(items,*,resolve=True):
    result={}
    for item in items:
        name,separator,value=item.partition('=')
        if not separator or not value or name in result:raise ValueError('数据根应为不重复的 name=directory')
        result[name]=Path(value).resolve() if resolve else Path(value).absolute()
    return result


def targets(workspace,names,explicit):
    result={name:workspace/name for name in names if name in ('outputs','datasets','photos','models')}
    result.update(explicit)
    if set(result)!=set(names):
        raise ValueError('请使用 --to name=directory 明确指定每个外部数据根的空恢复目录')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=ROOT)
    parser.add_argument('--root',action='append',default=[],help='备份数据根 name=directory；重复提供')
    parser.add_argument('--to',action='append',default=[],help='空恢复目标 name=directory；不改写来源路径')
    parser.add_argument('--allow-existing-missing',action='store_true',help='明确记录备份前已缺失的资产；仍拒绝遗漏现存外部资产')
    commands=parser.add_subparsers(dest='operation',required=True)
    commands.add_parser('inspect')
    commands.add_parser('disk-preview')
    commands.add_parser('backup').add_argument('destination',type=Path)
    commands.add_parser('verify').add_argument('bundle',type=Path)
    commands.add_parser('restore').add_argument('bundle',type=Path)
    commands.add_parser('recover').add_argument('journal',type=Path)
    commands.add_parser('abort').add_argument('journal',type=Path)
    args=parser.parse_args(argv)
    try:
        workspace=args.workspace.resolve();roots=mappings(args.root,resolve=args.operation!='disk-preview') if args.root else default_roots(workspace)
        explicit=mappings(args.to)
        if args.operation=='inspect':
            report=inspect_references(workspace)
            report['roots']={n:str(p) for n,p in roots.items()}
            report['outside_roots']=[p for p in report['references'] if not any(Path(p).is_relative_to(root) for root in roots.values())]
            report['ready']=not report['missing'] and not report['outside_roots'] and bool(roots)
            result=report
        elif args.operation=='disk-preview':
            from femto.pipeline.disk_inventory import inspect_disk
            result=inspect_disk(roots)
        elif args.operation=='backup':
            value=backup_workspace(workspace,args.destination,roots=roots,
                allow_existing_missing=args.allow_existing_missing,
                progress=lambda message:print(message,file=sys.stderr,flush=True))
            result=dict(state='backed_up',destination=value['destination'],
                        roots=list(value['manifest']['roots']),reference_count=len(value['manifest']['reference_audit']['references']),
                        source_assets_complete=value['manifest']['reference_audit']['source_assets_complete'],
                        unavailable_reference_count=len(value['manifest']['reference_audit']['unavailable']))
        elif args.operation=='verify':
            manifest=storage.verify(args.bundle)
            result=dict(state='verified',roots=list(manifest['roots']),paths_rebased=False,
                        source_assets_complete=manifest.get('reference_audit',{}).get('source_assets_complete',True),
                        unavailable_reference_count=len(manifest.get('reference_audit',{}).get('unavailable',[])))
        elif args.operation=='restore':
            manifest=storage.verify(args.bundle)
            result=storage.restore_new(args.bundle,targets(workspace,manifest['roots'],explicit),workspace=workspace)
        else:
            state=json.loads(args.journal.read_text('utf-8'))
            destinations=targets(workspace,state['destinations'],explicit)
            function=storage.recover_restore if args.operation=='recover' else storage.abort_restore
            result=function(args.journal,destinations,workspace=workspace)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 2 if args.operation=='disk-preview' and not result['scan_complete'] else 0
    except Exception as error:
        print(json.dumps(dict(state='failed',error=str(error)),ensure_ascii=False),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
