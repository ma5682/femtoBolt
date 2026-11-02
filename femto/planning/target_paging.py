"""Immutable, bounded target pages sharing one original target and revision.

Each page obeys TargetBudget; cumulative work is recorded instead of pretending
the old result was a fresh target. Pages are not coverage certificates.
"""
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path

import numpy as np

from femto.planning import target_surface as target, planning_revision as revision_api

SCHEMA='femto-original-target-pages-v1'
MAX_PAGES=128
SOURCE_NAMES=('vertices','faces','region_faces','depth','image_valid','mask','quality_classes','face_quality')


@dataclass(frozen=True)
class TargetPages:
    metadata: dict
    pages: tuple
    pixel_status: np.ndarray


def _seal(pages, revision):
    if not 1<=len(pages)<=MAX_PAGES:raise ValueError('原目标分块数量超过预算')
    first=pages[0];status=first.arrays['pixel_status'].copy()
    cursor=0;units=0
    cumulative={name:0 for name in ('clip_count','pair_checks','exact_clip_fallbacks','candidate_count','indexed_faces')}
    records=[]
    if sum(value.nbytes for page in pages for value in page.arrays.values())>target.MAX_ASSET_BYTES:
        raise ValueError('原目标分块累计数组超过512 MiB预算')
    for ordinal,page in enumerate(pages):
        metrics=page.metadata['metrics']
        if page.metadata['computation_sha256']!=revision['sha256'] or page.metadata['target_id']!=revision['target_id']:
            raise ValueError('原目标分块混用计算修订')
        if metrics.get('page_start_group',0)!=cursor:raise ValueError('原目标分块前沿不连续')
        end=metrics['processed_pixel_groups']
        if type(end) is not int or end<cursor or (ordinal and end==cursor):
            raise ValueError('原目标续算没有推进')
        if ordinal:
            previous=pages[ordinal-1].metadata['metrics']
            if (previous['stop_reason'] not in ('unit_budget','pair_budget','time_budget')
                    or previous['candidate_pixel_groups']!=metrics['candidate_pixel_groups']):
                raise ValueError('原目标分块不具备完整且相同的候选索引')
            for name in SOURCE_NAMES:
                if first.metadata['arrays'][name]!=page.metadata['arrays'][name]:
                    raise ValueError('原目标分块输入发生变化：'+name)
            if (first.metadata['quality_evidence_sha256']!=page.metadata['quality_evidence_sha256']
                    or first.metadata['effective_domain_sha256']!=page.metadata['effective_domain_sha256']):
                raise ValueError('原目标分块质量来源发生变化')
            update=page.arrays['pixel_status']!=9
            status[update]=page.arrays['pixel_status'][update]
        records.append(dict(folder=f'page_{ordinal:03}',sha256=page.metadata['sha256'],
            start_group=cursor,end_group=end,leaf_count=len(page.arrays['area_m2'])))
        units+=len(page.arrays['area_m2']);cursor=end
        for name in cumulative:cumulative[name]+=metrics[name]
    unknown=first.arrays['mask'] & (status!=1)
    summary=dict(leaf_count=units,processed_pixel_groups=cursor,
        candidate_pixel_groups=pages[-1].metadata['metrics']['candidate_pixel_groups'],
        selected_pixel_count=int(first.arrays['mask'].sum()),
        unlocated_pixel_count=int(unknown.sum()),
        unknown_by_reason={name:int((status==code).sum()) for code,name in target.PIXEL_STATES.items() if code>1},
        total_target_area_m2=None if unknown.any() else sum(float(p.arrays['area_m2'].sum()) for p in pages),
        coverage_status='not_evaluated')
    metadata=dict(schema=SCHEMA,target_id=revision['target_id'],computation_sha256=revision['sha256'],
        pages=records,cumulative_work=cumulative,summary=summary,
        pixel_status=revision_api.array_record(status),execution_ready=False,physical_accuracy_verified=False)
    metadata['sha256']=revision_api.digest(metadata);status.setflags(write=False)
    return TargetPages(metadata,tuple(pages),status)


def begin(ledger,revision,*,evidence,domain,check=lambda:None):
    target.validate(ledger,revision,evidence=evidence,domain=domain,check=check)
    if ledger.metadata['metrics'].get('page_start_group',0):raise ValueError('首块必须从原目标起点开始')
    return _seal((ledger,),revision)


def validate(bundle,revision,*,evidence,domain,check=lambda:None):
    for page in bundle.pages:
        target.validate(page,revision,evidence=evidence,domain=domain,check=check)
    rebuilt=_seal(bundle.pages,revision)
    if rebuilt.metadata!=bundle.metadata or not np.array_equal(rebuilt.pixel_status,bundle.pixel_status):
        raise ValueError('原目标分块清单或累计结果无法重现')
    return bundle


def resume(bundle,revision,*,evidence,domain,budget=None,check=lambda:None):
    validate(bundle,revision,evidence=evidence,domain=domain,check=check)
    last=bundle.pages[-1];metrics=last.metadata['metrics']
    if metrics['stop_reason'] not in ('unit_budget','pair_budget','time_budget'):
        raise ValueError('当前停止原因不能按像素组续算，请检查候选索引或输入')
    budget=(budget or target.TargetBudget()).checked()
    inputs={name:last.arrays[name] for name in SOURCE_NAMES}
    arrays,counts=target._construct(inputs,revision,budget,check,start_group=metrics['processed_pixel_groups'])
    page=target._seal(arrays,revision,budget,counts,evidence['sha256'],None if domain is None else domain['sha256'])
    check()
    return _seal(bundle.pages+(page,),revision)


def write(folder,bundle,check=lambda:None):
    """Write only to a fresh caller-owned snapshot, so cancellation preserves old pages."""
    from femto.io_utils import write_json_strict
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    for record,page in zip(bundle.metadata['pages'],bundle.pages):
        check();destination=folder/record['folder'];destination.mkdir()
        target.write(destination,page,check)
    check();write_json_strict(folder/'target_pages.json',bundle.metadata)


def read(folder,revision,*,evidence,domain,check=lambda:None):
    from femto.io_utils import load_json
    folder=Path(folder);path=folder/'target_pages.json'
    if path.stat().st_size>1024*1024:raise ValueError('原目标分块清单超过预算')
    metadata=load_json(path)
    if (metadata.get('schema')!=SCHEMA or not isinstance(metadata.get('pages'),list)
            or not 1<=len(metadata['pages'])<=MAX_PAGES
            or metadata.get('sha256')!=revision_api.digest({k:v for k,v in metadata.items() if k!='sha256'})):
        raise ValueError('原目标分块清单无效')
    pages=[];allocated=0
    for index,record in enumerate(metadata['pages']):
        check()
        if record['folder']!=f'page_{index:03}':raise ValueError('原目标分块路径无效')
        destination=folder/record['folder'];header=load_json(destination/target.MANIFEST)
        if header['sha256']!=record['sha256']:raise ValueError('原目标分块摘要无效')
        target.validate_metadata(header,revision)
        allocated+=sum(int(np.prod(item['shape']))*np.dtype(item['dtype']).itemsize for item in header['arrays'].values())
        if allocated>target.MAX_ASSET_BYTES:raise ValueError('原目标分块累计数组超过512 MiB预算')
        pages.append(target.read(destination,header,revision,evidence=evidence,domain=domain,check=check))
    result=_seal(tuple(pages),revision)
    if result.metadata!=metadata:raise ValueError('原目标分块累计结果与独立回读不一致')
    return result


def validate_headers(metadata,revision):
    """Validate allocation and contiguous construction ranges before loading arrays."""
    headers=metadata.get('construction_pages')
    if not isinstance(headers,list) or not 1<=len(headers)<=MAX_PAGES:
        raise ValueError('分块目标缺少有界构建记录')
    cursor=0;units=0;first=headers[0]
    for ordinal,header in enumerate(headers):
        if header.get('schema')!=target.SCHEMA:raise ValueError('分块构建记录不能递归嵌套')
        target.validate_metadata(header,revision)
        metrics=header['metrics'];end=metrics['processed_pixel_groups']
        if metrics.get('page_start_group',0)!=cursor or type(end) is not int or end<cursor or (ordinal and end==cursor):
            raise ValueError('分块构建前沿不连续或没有推进')
        if ordinal:
            if (headers[ordinal-1]['metrics']['stop_reason'] not in ('unit_budget','pair_budget','time_budget')
                    or metrics['candidate_pixel_groups']!=first['metrics']['candidate_pixel_groups']):
                raise ValueError('分块构建候选范围不一致')
            if any(header['arrays'][key]!=first['arrays'][key] for key in SOURCE_NAMES):
                raise ValueError('分块构建混用输入')
        if (header['quality_evidence_sha256']!=metadata['quality_evidence_sha256']
                or header['effective_domain_sha256']!=metadata['effective_domain_sha256']):
            raise ValueError('分块构建质量来源不一致')
        cursor=end;units+=header['arrays']['area_m2']['shape'][0]
    if units!=metadata['arrays']['area_m2']['shape'][0]:raise ValueError('分块单元数量与合并资产不一致')
    if any(metadata['arrays'][key]!=first['arrays'][key] for key in SOURCE_NAMES):
        raise ValueError('合并原目标改变源输入')
    total=sum(int(np.prod(row['shape']))*np.dtype(row['dtype']).itemsize for row in metadata['arrays'].values())
    if total>target.MAX_ASSET_BYTES:raise ValueError('合并原目标数组超过512 MiB预算')


def materialize(bundle,revision):
    """Expose the complete target to existing geometry, archive and display APIs."""
    checked=_seal(bundle.pages,revision)
    if checked.metadata!=bundle.metadata:raise ValueError('原目标分块清单已变化')
    arrays=dict(bundle.pages[0].arrays)
    arrays['pixel_status']=bundle.pixel_status
    for key in target._LEAVES:arrays[key]=np.concatenate([page.arrays[key] for page in bundle.pages])
    metrics=deepcopy(bundle.pages[-1].metadata['metrics']);metrics.pop('page_start_group',None)
    for key in ('candidate_count','indexed_faces','clip_count','pair_checks','exact_clip_fallbacks','occluded_polygons'):
        metrics[key]=sum(page.metadata['metrics'][key] for page in bundle.pages)
    metrics['page_count']=len(bundle.pages)
    metrics['maximum_subdivision_depth']=max(page.metadata['metrics']['maximum_subdivision_depth'] for page in bundle.pages)
    first=bundle.pages[0].metadata
    return target._seal(arrays,revision,target.TargetBudget(**first['budget']),metrics,
        first['quality_evidence_sha256'],first['effective_domain_sha256'],
        construction_pages=[page.metadata for page in bundle.pages])


def reproduce_materialized(ledger,revision,*,evidence,domain,check=lambda:None):
    """Independently rebuild every bounded page, then compare the complete asset."""
    pages=[]
    inputs={key:ledger.arrays[key] for key in SOURCE_NAMES}
    for header in ledger.metadata['construction_pages']:
        check();budget=target.TargetBudget(**header['budget'])
        arrays,counts=target._construct(inputs,revision,budget,check,frontier=header['metrics'])
        page=target._seal(arrays,revision,budget,counts,evidence['sha256'],None if domain is None else domain['sha256'])
        if page.metadata!=header:raise ValueError('原目标分块不能由当前输入重现')
        pages.append(page)
    bundle=_seal(tuple(pages),revision)
    rebuilt=materialize(bundle,revision)
    if rebuilt.metadata!=ledger.metadata:raise ValueError('合并原目标单元或累计前沿不能重现')
    return bundle


def rebind_headers(ledger,revision):
    return rebind_header_records(ledger.metadata,revision)


def rebind_header_records(metadata,revision):
    headers=deepcopy(metadata.get('construction_pages'))
    if headers is None:return None
    for header in headers:
        header.update(target_id=revision['target_id'],computation_sha256=revision['sha256'])
        header['sha256']=revision_api.digest({k:v for k,v in header.items() if k!='sha256'})
    return headers
