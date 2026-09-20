"""Versioned research measurement records. Never clinical or robot acceptance."""
import argparse,hashlib,json,os,platform,tempfile,time,uuid
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

PURPOSES=('optical_range','axial_depth','housing_distance','two_point_length','plane_residual','mapping_reprojection')
KINDS=('synthetic','historical_replay','physical_test')

def hashed(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def finite(value):return type(value) in (int,float) and np.isfinite(value)
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def template():
    return dict(schema='femto-research-protocol-v1',test_id='replace-with-test-id',purpose='two_point_length',evidence_kind='physical_test',
        target_id=None,operator=None,measurement_definition=None,repetition_scope='independent_capture',coordinate_frame='world_rdf',unit='metre',
        reference=dict(value=None,unit='mm',uncertainty_bound=None,uncertainty_method=None,datum='two_physical_endpoints',instrument=None,reference_object=None),
        conditions=dict(distance_mm=None,angle_deg=None,material=None,camera_mode=None,environment=None,placement_procedure=None),
        tolerance=None,algorithm=dict(name=None,version=None,parameters={}),
        assets=[dict(role='raw',path='REPLACE_RAW_FILE'),dict(role='calibration',path='REPLACE_CALIBRATION_FILE'),dict(role='algorithm',path='REPLACE_ALGORITHM_SOURCE_FILE')],
        notes='所有空值保留未验证；每次尝试均录入，失败也保留。不据此设置固定补偿。')

def validate_spec(spec):
    if spec.get('schema')!='femto-research-protocol-v1' or spec.get('purpose') not in PURPOSES or spec.get('evidence_kind') not in KINDS:raise ValueError('Unknown protocol, purpose or evidence kind')
    if not isinstance(spec.get('test_id'),str) or not spec['test_id'].strip():raise ValueError('test_id required')
    if spec.get('coordinate_frame')!='world_rdf' or spec.get('unit')!='metre':raise ValueError('Only W0/RDF metre observations are supported')
    if spec.get('repetition_scope') not in ('same_data','independent_capture'):raise ValueError('Unknown repetition scope')
    for key in ('reference','conditions','algorithm'):
        if not isinstance(spec.get(key),dict):raise ValueError(key+' object required')
    expected_unit='pixel' if spec['purpose']=='mapping_reprojection' else 'mm'
    if spec['reference'].get('unit')!=expected_unit:raise ValueError('Reference unit must be '+expected_unit)
    for value in (spec.get('tolerance'),spec['reference'].get('uncertainty_bound')):
        if value is not None and (not finite(value) or value<0):raise ValueError('Invalid tolerance or uncertainty')
    if spec['reference'].get('value') is not None and not finite(spec['reference']['value']):raise ValueError('Invalid reference')
    for key in ('distance_mm','angle_deg'):
        if spec['conditions'].get(key) is not None and not finite(spec['conditions'][key]):raise ValueError('Invalid condition '+key)
    if not isinstance(spec['algorithm'].get('parameters'),dict):raise ValueError('Algorithm parameters object required')
    if spec['conditions'].get('distance_mm') is not None and spec['conditions']['distance_mm']<0:raise ValueError('Negative physical distance')
    hashed(spec)

def measure(purpose,observation):
    def points(key,count=None):
        p=np.asarray(observation[key],float)
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or (count is not None and len(p)!=count):raise ValueError('Invalid '+key)
        return p
    details={}
    if purpose in ('optical_range','axial_depth'):
        p=points('xyz_m',1)[0]
        if p[2]<=0:raise ValueError('Point must be in front of reference optical origin')
        value=float((np.linalg.norm(p) if purpose=='optical_range' else p[2])*1000)
    elif purpose=='housing_distance':
        value=observation['manual_distance_mm']
        if not finite(value) or value<0:raise ValueError('Invalid housing ruler reading')
        details['scope']='manual_housing_datum_only; no_conversion_to_optical_origin_or_fixed_compensation'
    elif purpose=='two_point_length':
        p=points('xyz_m',2);value=float(np.linalg.norm(p[1]-p[0])*1000)
    elif purpose=='plane_residual':
        p=points('xyz_m')
        if len(p)<4:raise ValueError('Plane fit needs at least four points')
        center=p.mean(axis=0);_,singular,vectors=np.linalg.svd(p-center,full_matrices=False)
        if singular[1]<1e-9:raise ValueError('Collinear/degenerate plane samples')
        residual=(p-center)@vectors[-1]*1000;value=float(np.sqrt(np.mean(residual**2)))
        details=dict(fitted_center_m=center.tolist(),fitted_normal=vectors[-1].tolist(),p95_abs_residual_mm=float(np.percentile(np.abs(residual),95)),sample_count=len(p),scope='internal_fitted_plane_flatness_not_absolute_position_accuracy')
    else:
        p=points('xyz_m',1)[0];k=np.asarray(observation['intrinsic'],float);pixel=np.asarray(observation['pixel'],float)
        if k.shape!=(3,3) or pixel.shape!=(2,) or not np.isfinite(k).all() or not np.isfinite(pixel).all() or p[2]<=0 or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):raise ValueError('Invalid reference-camera mapping input')
        projected=(k@p);uv=projected[:2]/projected[2];value=float(np.linalg.norm(uv-pixel))
        details=dict(projected_pixel=uv.tolist(),scope='reference_camera_reprojection_consistency_not_independent_3D_accuracy')
    return dict(value=value,unit='pixel' if purpose=='mapping_reprojection' else 'mm',details=details)

def evidence(spec,base,check):
    assets=[]
    if not isinstance(spec.get('assets'),list) or len(spec['assets'])>20000:raise ValueError('Invalid asset list / exceeds 20000 files')
    seen=set()
    for item in spec['assets']:
        check();role=item['role']
        if role not in ('raw','calibration','algorithm','geometry','reference_evidence','source_metadata'):raise ValueError('Unknown asset role')
        p=(base/item['path']).resolve()
        if (role,str(p)) in seen:raise ValueError('Duplicate asset declaration')
        seen.add((role,str(p)));row=dict(role=role,path=str(p),status='missing');assets.append(row)
        if p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):check();h.update(block)
            row.update(status='present',sha256=h.hexdigest(),size=p.stat().st_size)
    return assets

def protocol_key(spec):
    return hashed({k:spec.get(k) for k in ('test_id','purpose','evidence_kind','target_id','operator','measurement_definition','reference','conditions','repetition_scope','coordinate_frame','unit','tolerance')})

def assess(spec,assets):
    missing=[]
    for k in ('target_id','operator','measurement_definition'):
        if not spec.get(k):missing.append(k)
    for k in ('instrument','reference_object','uncertainty_method','datum'):
        if not spec['reference'].get(k):missing.append('reference.'+k)
    if spec['reference'].get('uncertainty_bound') is None:missing.append('reference.uncertainty_bound')
    for k in ('distance_mm','angle_deg','material','camera_mode','environment','placement_procedure'):
        if spec['conditions'].get(k) in (None,''):missing.append('conditions.'+k)
    for k in ('name','version'):
        if not spec['algorithm'].get(k):missing.append('algorithm.'+k)
    for role in ('raw','calibration','algorithm'):
        if not any(a['role']==role and a['status']=='present' for a in assets):missing.append('assets.'+role)
    missing.extend('missing_asset:'+a['path'] for a in assets if a['status']!='present')
    if spec['evidence_kind']=='physical_test' and not any(a['role']=='reference_evidence' and a['status']=='present' for a in assets):missing.append('physical_reference_evidence')
    required={'optical_range':'optical_origin','axial_depth':'optical_axial','housing_distance':'housing_mark','two_point_length':'two_physical_endpoints','plane_residual':'fitted_plane','mapping_reprojection':'reference_camera_pixel'}
    if spec['reference'].get('datum')!=required[spec['purpose']]:missing.append('datum_mismatch_no_conversion_applied')
    return missing

def record(spec_path,observation_path,parent,trial_id,status='measured',reason=None,elapsed_s=None,check=lambda:None,progress=lambda stage:None):
    start=time.perf_counter();check();spec=read(spec_path);validate_spec(spec)
    tool_identity=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if status not in ('measured','failed','cancelled'):raise ValueError('Unknown attempt status')
    if not trial_id or (status!='measured' and not reason):raise ValueError('trial_id and failure/cancel reason required')
    if elapsed_s is not None and (not finite(elapsed_s) or elapsed_s<0):raise ValueError('Invalid elapsed time')
    progress('记录原始数据、标定和算法文件指纹')
    assets=evidence(spec,Path(spec_path).resolve().parent,check);observation=read(observation_path) if observation_path else None
    result=None
    if status=='measured':
        if not isinstance(observation,dict):raise ValueError('Measured attempt requires an observations JSON object')
        if observation.get('fixture_only') and spec['evidence_kind']!='synthetic':raise ValueError('Synthetic example observations cannot be recorded as physical/history')
        result=measure(spec['purpose'],observation)
    missing=assess(spec,assets)
    def roles_digest(roles):return hashed(sorted((a['role'],a['sha256']) for a in assets if a['role'] in roles and a['status']=='present'))
    raw_ok=any(a['role']=='raw' and a['status']=='present' for a in assets) and not any(a['role']=='raw' and a['status']!='present' for a in assets)
    calibration_ok=any(a['role']=='calibration' and a['status']=='present' for a in assets) and not any(a['role']=='calibration' and a['status']!='present' for a in assets)
    record=dict(schema='femto-research-attempt-v1',record_id=uuid.uuid4().hex,created_at=datetime.now(timezone.utc).isoformat(),protocol=spec,
        trial_id=trial_id,status=status,reason=reason,observation=observation,result=result,assets=assets,
        protocol_key=protocol_key(spec),raw_identity=roles_digest(('raw',)) if raw_ok else None,
        calibration_identity=roles_digest(('calibration',)) if calibration_ok else None,algorithm_identity=hashed(dict(declaration=spec['algorithm'],files=roles_digest(('algorithm',)),measurement_tool=tool_identity,python=platform.python_version(),numpy=np.__version__)),
        missing_conditions=missing,verification_status='unverified',physical_accuracy_verified=False,execution_ready=False,
        processing_elapsed_s=elapsed_s,recording_elapsed_s=time.perf_counter()-start,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()),
        limitations=['Only listed attempts form the denominator; never infer unrecorded attempts.', 'Raw hashes identify current bytes, not proof of original capture authenticity.', 'No fixed offset inferred or applied.'])
    root=Path(parent).resolve();root.mkdir(parents=True,exist_ok=True);target=root/record['record_id']
    with tempfile.TemporaryDirectory(prefix='.pending_',dir=root) as tmp:
        progress('封存算法代码与测试输入')
        folder=Path(tmp);(folder/'algorithm_snapshot').mkdir();total=0
        for i,a in enumerate(assets):
            if a['role']=='algorithm' and a['status']=='present':
                check();data=Path(a['path']).read_bytes();total+=len(data)
                if total>20000000:raise ValueError('Algorithm snapshot exceeds 20MB')
                if hashlib.sha256(data).hexdigest()!=a['sha256']:raise ValueError('Algorithm changed during recording')
                name=f'algorithm_snapshot/{i}_{Path(a["path"]).name}';(folder/name).write_bytes(data);a['snapshot']=name
        progress('复核文件未在录入期间改变')
        # Recheck every asset before publication, so a moving input cannot look sealed.
        if [(a['role'],a['path'],a.get('sha256')) for a in evidence(spec,Path(spec_path).resolve().parent,check)]!=[(a['role'],a['path'],a.get('sha256')) for a in assets]:raise ValueError('Evidence changed during recording')
        tool=Path(__file__).read_bytes()
        if hashlib.sha256(tool).hexdigest()!=tool_identity:raise ValueError('Measurement tool changed during recording')
        (folder/'measurement_tool.py').write_bytes(tool);record['measurement_tool_sha256']=tool_identity
        record['recording_elapsed_s']=time.perf_counter()-start;record['record_sha256']=hashed(record);write(folder/'record.json',record)
        for p in folder.rglob('*'):
            if p.is_file():
                with p.open('r+b') as f:os.fsync(f.fileno())
        check()
        if folder.resolve().parent!=root or target.parent!=root:raise ValueError('Output path mismatch')
        progress('发布完整测试记录')
        os.replace(folder,target)
    return target

def load_record(path):
    r=read(path);checksum=r.pop('record_sha256',None)
    if r.get('schema')!='femto-research-attempt-v1' or checksum!=hashed(r):raise ValueError('Record checksum/schema mismatch')
    if hashlib.sha256((Path(path).parent/'measurement_tool.py').read_bytes()).hexdigest()!=r['measurement_tool_sha256']:raise ValueError('Measurement tool snapshot mismatch')
    validate_spec(r['protocol'])
    if r['status']=='measured' and r['result']!=measure(r['protocol']['purpose'],r['observation']):raise ValueError('Metric recomputation mismatch')
    for a in r['assets']:
        if a.get('snapshot'):
            p=(Path(path).parent/a['snapshot']).resolve()
            if Path(path).parent.resolve() not in p.parents or hashlib.sha256(p.read_bytes()).hexdigest()!=a['sha256']:raise ValueError('Algorithm snapshot mismatch')
    return r

def summary(paths):
    records=[load_record(p) for p in paths];ids=[r['record_id'] for r in records]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate record supplied')
    groups={}
    for r in records:groups.setdefault((r['protocol_key'],r['algorithm_identity'],r['calibration_identity']),[]).append(r)
    output=[]
    for (protocol,algorithm,calibration),items in sorted(groups.items(),key=lambda item:str(item[0])):
        measured=[r for r in items if r['status']=='measured'];values=np.array([r['result']['value'] for r in measured]);spec=items[0]['protocol'];ref=spec['reference'].get('value')
        errors=values-ref if ref is not None and not any('datum_mismatch_no_conversion_applied' in r['missing_conditions'] for r in items) else None
        capture_means={}
        for r in measured:
            if r['raw_identity']:capture_means.setdefault(r['raw_identity'],[]).append(r['result']['value'])
        means=np.array([np.mean(v) for v in capture_means.values()]);independent=len(means)
        output.append(dict(protocol_key=protocol,algorithm_identity=algorithm,calibration_identity=calibration,test_id=spec['test_id'],purpose=spec['purpose'],evidence_kind=spec['evidence_kind'],unit=items[0]['result']['unit'] if items[0]['result'] else ('pixel' if spec['purpose']=='mapping_reprojection' else 'mm'),
            attempt_count=len(items),measured_count=len(measured),failure_rate=sum(r['status']=='failed' for r in items)/len(items),cancelled_count=sum(r['status']=='cancelled' for r in items),
            mean=float(values.mean()) if len(values) else None,sample_stddev=float(values.std(ddof=1)) if len(values)>1 else None,
            distinct_raw_capture_count=independent,recapture_stddev=float(means.std(ddof=1)) if independent>1 and spec['repetition_scope']=='independent_capture' else None,
            signed_bias=float(errors.mean()) if errors is not None and len(errors) else None,rmse=float(np.sqrt(np.mean(errors**2))) if errors is not None and len(errors) else None,
            p95_abs_error=float(np.percentile(np.abs(errors),95)) if errors is not None and len(errors) else None,
            reference_uncertainty_bound=spec['reference'].get('uncertainty_bound'),tolerance=spec.get('tolerance'),
            numeric_outcomes=[numeric_outcome(r) for r in measured],
            processing_elapsed_s=[r['processing_elapsed_s'] for r in items],unverified_conditions=sorted(set(x for r in items for x in r['missing_conditions'])),
            verification_status='unverified',physical_accuracy_verified=False))
    return dict(schema='femto-research-summary-v1',groups=output,physical_accuracy_verified=False,interpretation='Sample spread is not accuracy; raw-identity deduplication is conservative and does not verify independence.')

def compare(before_paths,after_paths):
    before=[load_record(p) for p in before_paths];after=[load_record(p) for p in after_paths]
    def index(records):
        d={}
        for r in records:
            key=(r['protocol_key'],r['trial_id'],r['raw_identity'],r['calibration_identity'])
            if key in d:raise ValueError('Duplicate paired trial; one record per version/trial required')
            d[key]=r
        return d
    a,b=index(before),index(after);pairs=[]
    for k in sorted(a.keys()&b.keys(),key=str):
        x,y=a[k],b[k]
        if not x['raw_identity'] or not x['calibration_identity']:continue
        pairs.append(dict(trial_id=x['trial_id'],purpose=x['protocol']['purpose'],evidence_kind=x['protocol']['evidence_kind'],unit=x['result']['unit'] if x['result'] else y['result']['unit'] if y['result'] else None,before_record=x['record_id'],after_record=y['record_id'],before_status=x['status'],after_status=y['status'],
            delta_value=y['result']['value']-x['result']['value'] if x['result'] and y['result'] else None,
            delta_processing_s=y['processing_elapsed_s']-x['processing_elapsed_s'] if x['processing_elapsed_s'] is not None and y['processing_elapsed_s'] is not None else None,
            before_algorithm=x['algorithm_identity'],after_algorithm=y['algorithm_identity']))
    return dict(schema='femto-research-comparison-v1',pairs=pairs,before=summary(before_paths),after=summary(after_paths),
        unpaired_before=len(a)-len(pairs),unpaired_after=len(b)-len(pairs),physical_accuracy_verified=False,
        limitations=['Pairing requires identical purpose, reference, conditions, evidence kind, trial, raw bytes and calibration.', 'A changed value or faster run is not automatically an improvement.'])

def numeric_outcome(r):
    ref=r['protocol']['reference'];limit=r['protocol'].get('tolerance');value=ref.get('value');uncertainty=ref.get('uncertainty_bound')
    if r['missing_conditions'] or value is None or uncertainty is None or limit is None:return 'unverified'
    error=abs(r['result']['value']-value)
    return 'within_entered_interval_only' if error+uncertainty<=limit else 'outside_entered_interval' if error-uncertainty>limit else 'indeterminate_reference_uncertainty'


def dataset_assets(dataset):
    dataset=Path(dataset).resolve();entries=[dict(role='source_metadata',path=str(dataset/n)) for n in ('manifest.json','frames.jsonl')]
    frames=[json.loads(s) for s in (dataset/'frames.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    for f in frames:
        entries.extend(dict(role='raw',path=str((dataset/f[k]).resolve())) for k in ('rgb','depth'))
        if f.get('pose'):entries.append(dict(role='calibration',path=str((dataset/f['pose']).resolve())))
    entries.extend(dict(role='calibration',path=str(p)) for p in sorted((dataset/'calibration').glob('*.json')))
    return entries


def publish_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Hard-link the complete, flushed temporary file; fails atomically if target exists.
    fd,name=tempfile.mkstemp(prefix='.pending_',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.link(name,path)
    finally:Path(name).unlink(missing_ok=True)


def verify_assets(path):
    r=load_record(path);result=[]
    for a in r['assets']:
        p=Path(a['path']);current=None
        if p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            current=h.hexdigest()
        result.append(dict(path=str(p),role=a['role'],status='missing' if current is None else 'matches_record' if current==a.get('sha256') else 'changed',recorded_sha256=a.get('sha256'),current_sha256=current))
    return dict(schema='femto-research-asset-audit-v1',record_id=r['record_id'],assets=result,
        original_inputs_available_and_unchanged=all(a['status']=='matches_record' for a in result if a['role']!='algorithm'),
        note='Changed current algorithm files do not destroy sealed snapshots. This is an asset audit, not accuracy verification.',physical_accuracy_verified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('init');p.add_argument('output',type=Path);p.add_argument('--dataset',type=Path)
    p=sub.add_parser('record');p.add_argument('spec',type=Path);p.add_argument('--observations',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--trial',required=True);p.add_argument('--status',default='measured',choices=('measured','failed','cancelled'));p.add_argument('--reason');p.add_argument('--elapsed',type=float)
    p=sub.add_parser('summary');p.add_argument('records',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('compare');p.add_argument('--before',nargs='+',required=True,type=Path);p.add_argument('--after',nargs='+',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('verify-assets');p.add_argument('record',type=Path);p.add_argument('--output',required=True,type=Path)
    a=parser.parse_args()
    if a.command=='record':print(record(a.spec,a.observations,a.output,a.trial,a.status,a.reason,a.elapsed,progress=lambda text:print(text,flush=True)));return
    if a.command=='init':
        result=template()
        if a.dataset:
            result['evidence_kind']='historical_replay';result['repetition_scope']='same_data'
            result['assets']=dataset_assets(a.dataset)+[dict(role='algorithm',path=str(Path(__file__).resolve()))]
    elif a.command=='verify-assets':result=verify_assets(a.record)
    elif a.command=='summary':result=summary(a.records)
    else:result=compare(a.before,a.after)
    publish_json(a.output,result)
    print(a.output)

if __name__=='__main__':main()
