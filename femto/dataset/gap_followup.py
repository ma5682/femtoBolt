"""A bounded provenance link, never permission to inherit old coordinates or coverage."""
import hashlib
import json

SCHEMA='femto-gap-followup-v2'
STATE='new_model_requires_review_selection_route_and_coverage'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,
        allow_nan=False).encode('utf-8')).hexdigest()


def checked(value,task_id):
    fields={'schema','task_id','origin_target_id','origin_computation_sha256','origin_plan_sha256',
        'origin_geometry_sha256','mask_sha256','pixel_count','reference_bounds','reason','pixel_reason',
        'evidence_directory','next_state','sha256'}
    if not isinstance(value,dict) or set(value)!=fields or value.get('schema')!=SCHEMA:
        raise ValueError('缺口后续请求格式无效')
    if value['task_id']!=task_id or value['next_state']!=STATE:
        raise ValueError('缺口后续请求与当前任务不一致')
    hashes=('origin_target_id','origin_computation_sha256','origin_plan_sha256','origin_geometry_sha256','mask_sha256')
    for key in hashes:
        sha=value[key]
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('缺口后续请求来源摘要无效')
    bounds=value['reference_bounds']
    if (type(value['pixel_count']) is not int or not 0<value['pixel_count']<=8_388_608
            or not isinstance(bounds,list) or len(bounds)!=4 or any(type(n) is not int or n<0 or n>8_388_608 for n in bounds)
            or not bounds[0]<bounds[2] or not bounds[1]<bounds[3]
            or type(value['reason']) is not int or value['reason'] not in (2,8)
            or (value['pixel_reason'] is not None and (type(value['pixel_reason']) is not int or not 2<=value['pixel_reason']<=10))
            or not isinstance(value['evidence_directory'],str) or not 1<=len(value['evidence_directory'])<=2048):
        raise ValueError('缺口后续请求范围无效')
    if value['sha256']!=digest({k:v for k,v in value.items() if k!='sha256'}):
        raise ValueError('缺口后续请求摘要不一致')
    return json.loads(json.dumps(value,ensure_ascii=False))
