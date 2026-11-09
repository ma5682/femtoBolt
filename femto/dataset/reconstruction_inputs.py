"""Versioned file evidence for the actual inputs of a local reconstruction.

Records contain hashes and references, never copies of the raw RGB-D files.
Old models without this optional record retain their original replay contract.
"""
from pathlib import Path
import hashlib
import json

from femto.dataset.local_reference import CALIBRATION_FILES
from femto.dataset.provenance import value_hash
from femto.io_utils import sha256_file
from femto.operation_diagnostics import diagnosed

SCHEMA='femto-local-reconstruction-inputs-v1'
MAX_FILES=4096
MAX_INDEX_BYTES=16*1024**2


class InputSourceChanged(ValueError):
    def __init__(self,path):
        self.path=Path(path)
        super().__init__(f'重建输入文件与保存的指纹不一致：{path}')


@diagnosed('reconstruction.input_fingerprints')
def snapshot(dataset,check=lambda:None):
    dataset=Path(dataset).resolve()
    index=dataset/'frames.jsonl'
    if index.stat().st_size>MAX_INDEX_BYTES:raise ValueError('重建帧索引超过读取预算')
    content=index.read_bytes()
    rows=[json.loads(line) for line in content.decode('utf-8-sig').splitlines() if line.strip()]
    if not rows or len(rows)*3+len(CALIBRATION_FILES)+2>MAX_FILES:
        raise ValueError('重建输入文件数量超过核验预算')
    paths={dataset/'manifest.json',index}
    paths.update(dataset/'calibration'/name for name in CALIBRATION_FILES)
    frames=[]
    for row in rows:
        check()
        frame=row['frame']
        if type(frame) is not int or frame<0 or frame in frames:raise ValueError('重建帧标识无效或重复')
        frames.append(frame)
        paths.update((dataset/row[key]).resolve() for key in ('rgb','depth','pose'))
        support=row.get('local_temporal_support')
        if support:paths.add((dataset/support['mask']).resolve())
    if len(paths)>MAX_FILES:raise ValueError('重建输入文件数量超过核验预算')
    files={str(path):sha256_file(path,check) for path in sorted(paths)}
    if files[str(index)]!=hashlib.sha256(content).hexdigest():raise InputSourceChanged(index)
    result=dict(schema=SCHEMA,source_dataset=str(dataset),frame_ids=frames,files=files,
                scope='admitted_rgb_depth_pose_calibration_index_and_applied_support_masks')
    result['sha256']=value_hash(result)
    return result


@diagnosed('reconstruction.verify_input_fingerprints')
def verify(record,check=lambda:None):
    if not isinstance(record,dict):raise ValueError('重建输入指纹记录无效')
    unsigned={key:value for key,value in record.items() if key!='sha256'}
    files=record.get('files')
    if (record.get('schema')!=SCHEMA or record.get('sha256')!=value_hash(unsigned)
            or not isinstance(files,dict) or not 1<=len(files)<=MAX_FILES):
        raise ValueError('重建输入指纹版本、摘要或文件数量无效')
    for name,expected in files.items():
        check()
        if (not isinstance(name,str) or not Path(name).is_absolute()
                or not isinstance(expected,str) or len(expected)!=64
                or any(char not in '0123456789abcdef' for char in expected)):
            raise ValueError('重建输入指纹路径或摘要无效')
        if sha256_file(name,check)!=expected:raise InputSourceChanged(name)
    check()
