"""Bounded proposals to put old proof witnesses on candidate scan rows.

Research only. The caller independently replays the old proof and checks both
routes and target first. A proposal is neither a geometric acceptance nor a
coverage proof. Its final route needs fresh whole-edge, point, selection and
coverage validation before any other use.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from femto.planning import coverage_certificate as coverage
from femto.planning.coverage_domain import scan_endpoints
from femto.planning.planning_revision import array_record, digest
from femto.planning.surface_trajectory import ScanSegment

VERSION = 'research-witness-preserving-scan-bend-v1'
EDGE_INSERTION_VERSION = 'research-witness-preserving-edge-insertion-v1'
MAX_PENDING = 5000
MAX_WITNESSES = 256
MAX_POINTS = 200_000
MAX_PROPOSAL_POINTS = 400_000
MAX_COMPARISONS = 2_000_000
MAX_PROPOSALS = 16
MAX_PROOFS = 2_000_000


def combine_replacements(segments, proposals, *, check=lambda: None):
    """Combine two detached single-point proposals; certify nothing.

    Each member must alter only its declared interior point. The caller must
    rederive both from the original proof, validate the complete combined
    geometry and motion, and independently reprove coverage before use.
    """
    if (not isinstance(segments,list) or not segments
            or any(not isinstance(item,ScanSegment) for item in segments)
            or not isinstance(proposals,list) or len(proposals)!=2):
        raise ValueError('联合见证点需要两条独立替换候选')
    if sum(len(item.points) for item in segments)>MAX_POINTS:
        raise ValueError('联合见证点源扫描点超过预算')
    check();before=digest([item.to_dict() for item in segments])
    combined=deepcopy(segments);used=set();members=[]
    for proposal in proposals:
        check()
        if not isinstance(proposal,dict) or 'edge' in proposal or 'placement' in proposal:
            raise ValueError('联合见证点只支持内部点替换')
        index,point=proposal.get('segment'),proposal.get('point')
        changed=proposal.get('segments')
        if (type(index) is not int or not 0<=index<len(segments)
                or type(point) is not int or not 1<=point<len(segments[index].points)-1
                or (index,point) in used or not isinstance(changed,list) or len(changed)!=len(segments)
                or any(not isinstance(item,ScanSegment) for item in changed)
                or any(len(a.points)!=len(b.points) for a,b in zip(changed,segments))):
            raise ValueError('联合见证点索引冲突或源扫描不一致')
        expected=deepcopy(segments)
        expected[index].points[point]=deepcopy(changed[index].points[point])
        if digest([item.to_dict() for item in expected])!=digest([item.to_dict() for item in changed]):
            raise ValueError('单点见证候选包含未声明的改动')
        if expected[index].points[point]==segments[index].points[point]:
            raise ValueError('单点见证候选没有发生替换')
        combined[index].points[point]=deepcopy(expected[index].points[point]);used.add((index,point))
        members.append({key:deepcopy(proposal[key]) for key in ('seed','segment','point','distance_m','pending_leaves')})
    check()
    if digest([item.to_dict() for item in segments])!=before:
        raise ValueError('联合见证点期间源扫描发生变化')
    return dict(segments=combined,members=members,candidate_only=True,coverage_claim=False,execution_ready=False)


def propose(old_trajectory, segments, original_faces, certificate, pending_leaves, *,
            representative_faces=None, max_distance_m=.003, max_proposals=MAX_PROPOSALS,
            edge_insert=False, check=lambda: None):
    """Propose separate point replacements, with no installation or acceptance.

    Candidates stay inside the witness's block and row. By default, only interior
    scan points with adjacent scan edges may be replaced. Explicit edge insertion
    instead proposes inserting a witness into a nonzero scan edge, preserving its
    endpoints. Same numeric row IDs are
    a search heuristic, not proof that the two rows represent the same surface.
    Output point/segment arrays are detached. The caller must verify the
    original proof, both source routes and the target before this function.
    """
    check(); coverage._checked_assets(certificate, check)
    if (not isinstance(segments, list) or not segments or any(not isinstance(x, ScanSegment) for x in segments)
            or not isinstance(pending_leaves, (list, tuple, np.ndarray)) or len(pending_leaves) > MAX_PENDING
            or type(max_proposals) is not int or not 1 <= max_proposals <= MAX_PROPOSALS
            or type(edge_insert) is not bool
            or isinstance(max_distance_m, bool) or not isinstance(max_distance_m, (int, float))
            or not math.isfinite(max_distance_m) or not 0 < max_distance_m <= .01):
        raise ValueError('见证点局部调整参数无效或超过预算')
    pending = []
    for value in pending_leaves:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError('待确认旧单元索引无效')
        pending.append(int(value))
    if len(set(pending)) != len(pending):
        raise ValueError('待确认旧单元索引重复')
    arrays = certificate.arrays
    if len(arrays['proof_leaf']) > MAX_PROOFS: raise ValueError('旧证据见证超过预算')
    if any(i < 0 or i >= len(arrays['root']) or arrays['status'][i] != 1 for i in pending):
        raise ValueError('待确认单元不是原已确认单元')
    faces = np.asarray(original_faces)
    if faces.ndim != 1 or faces.dtype.kind not in 'iu':
        raise ValueError('原网格面映射无效')
    representatives = None if representative_faces is None else np.asarray(representative_faces)
    if representatives is not None and (representatives.ndim != 1 or representatives.dtype.kind not in 'iu'):
        raise ValueError('原面代表映射无效')
    source_snapshot = digest([item.to_dict() for item in old_trajectory.ordered_items])
    segment_snapshot = digest([item.to_dict() for item in segments])
    face_snapshot = array_record(faces)
    representative_snapshot = None if representatives is None else array_record(representatives)
    records = scan_endpoints(old_trajectory, faces)
    if digest(records) != certificate.metadata['inputs']['seeds_sha256']:
        raise ValueError('旧路线扫描点与证据不一致')
    source = {item.segment_index: item for item in old_trajectory.ordered_items if isinstance(item, ScanSegment)}
    if len(source) != sum(isinstance(item, ScanSegment) for item in old_trajectory.ordered_items):
        raise ValueError('旧路线扫描段索引重复')
    selected = set(pending); witnesses = {}
    for i, (leaf, seed) in enumerate(zip(arrays['proof_leaf'], arrays['proof_seed'])):
        if i % 1024 == 0: check()
        if int(leaf) not in selected: continue
        seed = int(seed)
        if not 0 <= seed < len(arrays['seed_input_index']):
            raise ValueError('旧证据见证索引无效')
        index = int(arrays['seed_input_index'][seed])
        if not 0 <= index < len(records): raise ValueError('旧证据扫描点索引无效')
        record = records[index]; face = record['source_face']
        if representatives is not None:
            if not 0 <= face < len(representatives): raise ValueError('扫描点原面代表映射无效')
            face = int(representatives[face])
        if int(arrays['seed_face'][seed]) != face or not np.array_equal(arrays['seed_xyz'][seed], record['xyz_m']):
            raise ValueError('旧证据见证与扫描点不一致')
        item = source.get(record['segment_index'])
        edge = record['edge_index']; endpoint = record['endpoint']
        if item is None or endpoint not in (0, 1) or not 0 <= edge < len(item.points)-1:
            raise ValueError('旧见证扫描边无效')
        point = item.points[edge+endpoint]
        witnesses.setdefault(seed, (item.block_index, deepcopy(point), set()))[2].add(int(leaf))
        if len(witnesses) > MAX_WITNESSES: raise ValueError('旧见证候选超过预算')
    if set().union(*(value[2] for value in witnesses.values())) != selected:
        raise ValueError('待确认单元缺少原扫描点见证')
    total_points = sum(len(segment.points) for segment in segments)
    if total_points > MAX_POINTS: raise ValueError('扫描点候选超过预算')
    candidates = []; visited = 0
    for i, segment in enumerate(segments):
        check()
        if edge_insert:
            for j, (a, b) in enumerate(zip(segment.points, segment.points[1:])):
                visited += 1
                if visited % 1024 == 0: check()
                if visited > MAX_POINTS: raise ValueError('扫描点候选超过预算')
                if (b.get('role') not in (None, 'scan') or a.get('row_index') != b.get('row_index')): continue
                left, right = np.asarray(a.get('xyz_m'), dtype=float), np.asarray(b.get('xyz_m'), dtype=float)
                if left.shape != (3,) or right.shape != (3,) or not np.isfinite([left, right]).all():
                    raise ValueError('扫描点坐标无效')
                delta = right-left; squared = float(delta@delta)
                if squared > 0: candidates.append((i, j, segment.block_index, a['row_index'], left, delta, squared))
            continue
        for j in range(1, len(segment.points)-1):
            visited += 1
            if visited % 1024 == 0: check()
            if visited > MAX_POINTS: raise ValueError('扫描点候选超过预算')
            a, point, b = segment.points[j-1:j+2]
            if (point.get('role') not in (None, 'scan') or b.get('role') not in (None, 'scan')
                    or a.get('row_index') != point.get('row_index') or point.get('row_index') != b.get('row_index')):
                continue
            xyz = np.asarray(point.get('xyz_m'), dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all(): raise ValueError('扫描点坐标无效')
            candidates.append((i, j, segment.block_index, point['row_index'], xyz))
    if len(witnesses)*len(candidates) > MAX_COMPARISONS:
        raise ValueError('见证点局部调整比较超过预算')
    proposals = []
    for seed, (block, point, leaves) in sorted(witnesses.items()):
        check(); xyz = np.asarray(point['xyz_m'], dtype=float)
        options = []
        for index, candidate in enumerate(candidates):
            if index % 1024 == 0: check()
            if edge_insert:
                i, j, candidate_block, row, left, delta, squared = candidate
                if candidate_block != block or row != point['row_index']: continue
                t = float((xyz-left)@delta/squared)
                if 0 < t < 1: options.append((float(np.linalg.norm(xyz-(left+t*delta))), i, j, t))
            else:
                i, j, candidate_block, row, value = candidate
                if candidate_block == block and row == point['row_index']:
                    options.append((float(np.linalg.norm(value-xyz)), i, j))
        if not options: continue
        option = min(options);distance, index, position = option[:3]
        if distance > max_distance_m: continue
        original = segments[index].points[position+1 if edge_insert else position]
        replacement = deepcopy(point)
        replacement['row_index'] = original['row_index']
        replacement['route_index'] = original.get('route_index')
        replacement['role'] = 'scan'; replacement.pop('revisit', None)
        if (len(proposals)+1)*(total_points+int(edge_insert)) > MAX_PROPOSAL_POINTS:
            raise ValueError('见证点候选复制超过预算')
        changed = deepcopy(segments)
        if edge_insert: changed[index].points.insert(position+1, replacement)
        else: changed[index].points[position] = replacement
        proposal = dict(seed=seed, pending_leaves=sorted(leaves), segment=index,
                        point=position+1 if edge_insert else position, distance_m=distance, segments=changed)
        if edge_insert: proposal.update(placement='edge_insertion', edge=position, projection_fraction=option[3])
        proposals.append(proposal)
        if len(proposals) >= max_proposals: break
    if (digest([item.to_dict() for item in old_trajectory.ordered_items]) != source_snapshot
            or digest([item.to_dict() for item in segments]) != segment_snapshot or array_record(faces) != face_snapshot
            or (representatives is not None and array_record(representatives) != representative_snapshot)):
        raise ValueError('见证点局部调整期间输入发生变化')
    coverage._checked_assets(certificate, check); check()
    report = dict(version=EDGE_INSERTION_VERSION if edge_insert else VERSION, pending_leaves=len(pending), witnesses=len(witnesses),
                  eligible_points=len(candidates), proposals=len(proposals),
                  proposal_limit_reached=len(proposals) >= max_proposals,
                  semantics='proposals_only_require_geometry_and_coverage_replay')
    if edge_insert: report['placement'] = 'edge_insertion'
    return proposals, report
