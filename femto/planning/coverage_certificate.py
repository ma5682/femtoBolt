"""Versioned nominal coverage proofs, independent of route generation.

Every confirmed cell has one inner-ball witness and a positive-shared-subedge
path to that ball's exact surface anchor. Unknowns and budget leftovers retain
their original area weights. A proof concerns the declared numerical model.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import math
import time
from typing import Callable

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning import footprint_model
from femto.planning.coverage_domain import CoverageDomain
from femto.planning.planning_revision import array_record, digest
from femto.operation_diagnostics import diagnosed

SCHEMA = 'femto-model-surface-coverage-prototype-v1'
ALGORITHM = 'exact-anchor-ball-subedge-coverage-v1'
CURRENT_ALGORITHM = 'layered-anchor-ball-subedge-coverage-v2'
INDEXED_ALGORITHM = 'indexed-anchor-ball-subedge-coverage-v3'
REPROOF_ALGORITHM = 'retained-partition-anchor-ball-coverage-v3'
INDEXED_PARTIAL_REPROOF_ALGORITHM = 'indexed-partial-anchor-ball-reproof-v6'
RESEARCH_FRACTION_REPROOF_ALGORITHM = 'research-dyadic-anchor-ball-reproof-v1'
RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM = 'research-paged-dyadic-anchor-ball-reproof-v1'
RESEARCH_FRACTION_ALGORITHMS = (RESEARCH_FRACTION_REPROOF_ALGORITHM, RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM)
RESIDUAL_ALGORITHM = 'residual-partition-anchor-ball-coverage-v4'
INTERLEAVED_ALGORITHM = 'interleaved-partition-anchor-ball-coverage-v5'
RECALCULATION_ALGORITHM = 'retained-first-anchor-ball-coverage-v1'
PAGED_ALGORITHM = 'paged-target-anchor-ball-coverage-v1'
PAGED_RECALCULATION_ALGORITHM = 'paged-target-retained-first-coverage-v1'
GROUPED_RECALCULATION_ALGORITHM = 'paged-target-retained-first-grouped-coverage-v2'
PAGED_RECALCULATION_ALGORITHMS = (PAGED_RECALCULATION_ALGORITHM,GROUPED_RECALCULATION_ALGORITHM)
RECALCULATION_ALGORITHMS = (RECALCULATION_ALGORITHM,*PAGED_RECALCULATION_ALGORITHMS)
P2_ALGORITHMS = (CURRENT_ALGORITHM, INDEXED_ALGORITHM, *RECALCULATION_ALGORITHMS, PAGED_ALGORITHM)
REPROOF_ALGORITHMS = (REPROOF_ALGORITHM, RESIDUAL_ALGORITHM, INTERLEAVED_ALGORITHM)
GUIDED_ALGORITHMS = (*REPROOF_ALGORITHMS, INDEXED_PARTIAL_REPROOF_ALGORITHM,
                     *RESEARCH_FRACTION_ALGORITHMS, *RECALCULATION_ALGORITHMS)
SUPPORTED_ALGORITHMS = (ALGORITHM, CURRENT_ALGORITHM, INDEXED_ALGORITHM, PAGED_ALGORITHM, *GUIDED_ALGORITHMS)
STATES = {1:'confirmed', 2:'quality_excluded_or_pending', 3:'outside_inner_seed_balls',
          4:'ball_connectivity_unconfirmed', 5:'subdivision_limit',
          6:'target_construction_unconfirmed', 7:'budget_unprocessed'}
MAX_ARRAY_BYTES = 256*1024**2
DTYPES = dict(root='<i8', depth='|u1', code='<u4', status='|u1', witness='<i8',
              proof_leaf='<i8', proof_seed='<i8', proof_parent='<i8',
              seed_face='<i8', seed_input_index='<i8', seed_xyz='<f8', seed_inner_radius='<f8')


@dataclass(frozen=True)
class CoverageBudget:
    max_added_cells: int = 200_000
    max_depth: int = 4
    max_seeds: int = 20_000
    max_comparisons: int = 1_000_000
    max_adjacencies: int = 1_000_000
    max_proofs: int = 1_000_000
    seconds: float = 45.

    def checked(self) -> CoverageBudget:
        caps = asdict(CoverageBudget()); caps['max_depth'] = 8
        for key, cap in caps.items():
            value = getattr(self, key)
            if key == 'seconds':
                if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= cap:
                    raise ValueError('覆盖时间预算无效')
            elif type(value) is not int or not 0 <= value <= cap:
                raise ValueError('覆盖预算无效：'+key)
        return self


@dataclass(frozen=True)
class Certificate:
    metadata: dict
    arrays: dict[str, np.ndarray]

    @property
    def summary(self) -> dict:return self.metadata['summary']


class _BudgetStop(Exception):pass


def reproof_algorithm_for_policy(value):
    """Dispatch saved policies exactly; changing defaults never migrates a save."""
    from femto.planning.coverage_reproof import POLICY as v3
    from femto.planning.coverage_residual import POLICY as v4
    from femto.planning.coverage_interleaved import POLICY as v5
    for algorithm, policy in ((REPROOF_ALGORITHM, v3), (RESIDUAL_ALGORITHM, v4), (INTERLEAVED_ALGORITHM, v5)):
        if isinstance(value, dict) and value == policy and digest(value) == digest(policy):return algorithm
    raise ValueError('覆盖复算调度策略无效或版本不一致')


class _Meter:
    def __init__(self, budget, check, frontier):
        self.budget, self.check, self.frontier = budget, check, frontier
        self.started = time.monotonic()
        self.steps = 0
        self.counts = dict(comparisons=0, adjacencies=0, proofs=0, exact_fallbacks=0,
                           roots_reconstructed=0, added_cells=0, offered_seeds=0,
                           ignored_connectors=0, unbound_seeds=0, spatial_candidates=0)
        self.flags = set()
        self.phase = 'seeds'; self.stop_reason = 'complete'

    def touch(self, kind=None):
        if self.steps % 32 == 0:self.check()
        stopped = (self.frontier is not None and self.frontier['stop_reason'] == 'time_budget'
                   and self.steps >= self.frontier['steps'])
        stopped |= self.frontier is None and time.monotonic()-self.started >= self.budget.seconds
        reason = 'time_budget' if stopped else None
        if kind and self.counts[kind] >= getattr(self.budget, 'max_'+kind):reason = kind+'_budget'
        if reason:
            self.stop_reason = reason; self.flags.add(reason)
            raise _BudgetStop()
        self.steps += 1
        if kind:self.counts[kind] += 1

    def touch_many(self, count, kind):
        """Charge a new vector batch with the same count/frontier limits.

        Used by the v3 scheduler only. At most 31 accounting entries separate
        wall-time/cancellation checks; old v1/v2 call sequences stay unchanged.
        A partial reservation never installs any result from that batch.
        """
        while count:
            self.touch(kind); count -= 1
            amount = min(count, (-self.steps)%32, getattr(self.budget, 'max_'+kind)-self.counts[kind])
            if self.frontier is not None and self.frontier['stop_reason'] == 'time_budget':
                amount = min(amount, max(0, self.frontier['steps']-self.steps))
            self.steps += amount; self.counts[kind] += amount; count -= amount


@dataclass(slots=True)
class _Node:
    root: int
    depth: int = 0
    code: int = 0
    active: bool = True
    status: int = 7
    bary: tuple | None = None
    good: tuple = ()
    witness: int = -1


def _distance2(point, triangle):
    """Exact point-to-triangle distance, only for pruning uncontained cells."""
    a, b, c = triangle
    ab, ac, ap = exact.subtract(b, a), exact.subtract(c, a), exact.subtract(point, a)
    aa, bb, cc = exact.dot(ab, ab), exact.dot(ab, ac), exact.dot(ac, ac)
    determinant = aa*cc-bb*bb
    if determinant > 0:
        d, e = exact.dot(ap, ab), exact.dot(ap, ac)
        u, v = (d*cc-e*bb)/determinant, (e*aa-d*bb)/determinant
        if u >= 0 and v >= 0 and u+v <= 1:
            delta = exact.subtract(point, exact.xyz((1-u-v, u, v), triangle))
            return exact.dot(delta, delta)
    distances = []
    for p, q in zip(triangle, triangle[1:]+triangle[:1]):
        direction, offset = exact.subtract(q, p), exact.subtract(point, p)
        length = exact.dot(direction, direction)
        t = max(0, min(1, exact.dot(offset, direction)/length)) if length else 0
        delta = tuple(x-t*d for x, d in zip(offset, direction))
        distances.append(exact.dot(delta, delta))
    return min(distances)


def _grid_bin(value, radius):
    ratio = float(value)/radius
    return math.floor(ratio) if math.isfinite(ratio) else exact.Q(float(value))//exact.Q(radius)


class _Engine:
    def __init__(self, domain, records, model, budget, check, frontier):
        self.domain, self.records, self.model, self.budget = domain, records, model, budget
        self.meter = _Meter(budget, check, frontier)
        self.nodes = self._initial_partition()
        self.seeds = []; self.grid = defaultdict(list); self.candidates = {}
        self.roots = defaultdict(dict); self.neighbors = defaultdict(set)
        self.proofs = []
        self.radius = footprint_model.search_radius(model)

    def _initial_partition(self):
        if self.domain.binding.get('target_page_units') is not None:
            nodes=[]
            for index in range(len(self.domain.areas)):
                if index%1024==0:self.meter.check()
                nodes.append(_Node(index,status=2 if self.domain.quality[index] else 7))
            return nodes
        return [_Node(index, status=2 if self.domain.quality[index] else 7) for index in range(len(self.domain.areas))]

    def leaf_arrays(self):
        active = [(i, node) for i, node in enumerate(self.nodes) if node.active]
        active.sort(key=lambda item:(item[1].root, item[1].code/(4**item[1].depth), item[1].depth))
        remap = {number:index for index, (number, _) in enumerate(active)}
        values = {key:[getattr(node, key) for _, node in active] for key in ('root','depth','code','status','witness')}
        return values, remap

    @diagnosed('coverage.seed_binding')
    def seed(self):
        for index, record in enumerate(self.records):
            self.meter.touch()
            if not isinstance(record, dict) or record.get('role') not in ('scan', 'connector'):
                raise ValueError('覆盖种子必须声明扫描或连接角色')
            if record['role'] == 'connector':
                self.meter.counts['ignored_connectors'] += 1; continue
            self.meter.counts['offered_seeds'] += 1
            if len(self.seeds) >= self.budget.max_seeds:
                self.meter.flags.add('seed_budget'); continue
            face = record.get('source_face'); point = np.asarray(record.get('xyz_m'), float)
            if (type(face) is not int or not 0 <= face < len(self.domain.faces)
                    or point.shape != (3,) or not np.isfinite(point).all()):
                raise ValueError('覆盖种子坐标或支持面索引无效')
            face = int(self.domain.representative_faces[face])
            anchor = exact.anchor_ball(point, self.domain.triangle(face), footprint_model.seed_radius(self.model, record))
            if anchor is None:
                self.meter.counts['unbound_seeds'] += 1; continue
            bary, center, inner = anchor
            item = dict(face=face, input=index, point=point, bary=bary, center=center,
                        inner=inner, radius2=exact.Q(inner)**2, bounds=exact.bounds(center))
            number = len(self.seeds); self.seeds.append(item)
            self.grid[tuple(_grid_bin(x, self.radius) for x in point)].append(number)
        self.points = np.asarray([s['point'] for s in self.seeds]).reshape(-1, 3)
        self.centers_low = np.asarray([s['bounds'][0] for s in self.seeds]).reshape(-1, 3)
        self.centers_high = np.asarray([s['bounds'][1] for s in self.seeds]).reshape(-1, 3)

    def candidates_for(self, face):
        if face in self.candidates:return self.candidates[face]
        tri = self.domain.vertices[self.domain.faces[face]]
        low = np.nextafter(tri.min(axis=0)-self.radius, -np.inf)
        high = np.nextafter(tri.max(axis=0)+self.radius, np.inf)
        lo = [_grid_bin(x, self.radius) for x in low]
        hi = [_grid_bin(x, self.radius) for x in high]
        count = math.prod(b-a+1 for a, b in zip(lo, hi))
        if count > 4096:
            found = [i for i, seed in enumerate(self.seeds) if np.all(seed['point'] >= low) and np.all(seed['point'] <= high)]
        else:
            found = []
            for x in range(lo[0], hi[0]+1):
                for y in range(lo[1], hi[1]+1):
                    for z in range(lo[2], hi[2]+1):found.extend(self.grid.get((x, y, z), ()))
            found.sort()
        self.candidates[face] = tuple(found)
        return self.candidates[face]

    def cells(self):
        self.meter.phase = 'cells'
        for root in range(len(self.domain.areas)):
            self.meter.touch()
            if self.domain.quality[root]:continue
            face = int(self.domain.source_faces[root])
            candidates = self.candidates_for(face)
            if not candidates:self.nodes[root].status = 3; continue
            bary = self.domain.root(root)
            if bary is None:self.nodes[root].status = 6; continue
            self.meter.counts['roots_reconstructed'] += 1
            self.nodes[root].bary = bary
            queue = [root]
            while queue:
                number = queue.pop()
                self._cell(number, face, candidates, queue)

    def _cell(self, number, face, candidates, queue):
        self.meter.touch()
        node = self.nodes[number]
        tri, bounds = self.domain.cell(face, node.bary)
        low = np.nextafter(bounds[0].min(axis=0)-self.radius, -np.inf)
        high = np.nextafter(bounds[1].max(axis=0)+self.radius, np.inf)
        candidates = np.asarray(candidates, np.int64)
        self.meter.counts['spatial_candidates'] += len(candidates)
        candidates = candidates[np.all(self.points[candidates] >= low, axis=1) & np.all(self.points[candidates] <= high, axis=1)]
        for seed_id in candidates:
            self.meter.touch('comparisons')
        contained, fallbacks = exact.contains_balls(tri, tuple(self.seeds[i]['center'] for i in candidates),
            [self.seeds[i]['inner'] for i in candidates], triangle_bounds=bounds,
            center_bounds=(self.centers_low[candidates], self.centers_high[candidates]))
        self.meter.counts['exact_fallbacks'] += fallbacks
        good = tuple(int(i) for i in candidates[contained])
        if good:
            node.good = good; node.status = 4
            # Root proofs are already complete without global adjacency. Retain
            # them immediately, including when a later budget stops traversal.
            for seed_id in good:
                seed = self.seeds[seed_id]
                if face == seed['face'] and exact.contains_point(node.bary, seed['bary']):
                    witness = self._proof(number, seed_id, -1)
                    self.roots[seed_id][number] = witness
            return
        possible = False
        for seed_id in candidates:
            self.meter.touch('comparisons')
            seed = self.seeds[seed_id]
            if _distance2(seed['center'], tri) <= seed['radius2']:
                possible = True; break
        if not possible:node.status = 3; node.bary = None; return
        if node.depth >= self.budget.max_depth:
            node.status = 5; node.bary = None; self.meter.flags.add('depth_limit'); return
        if self.meter.counts['added_cells']+3 > self.budget.max_added_cells:
            node.status = 7; node.bary = None; self.meter.flags.add('cell_budget'); return
        first = len(self.nodes)
        self.nodes.extend(_Node(node.root, node.depth+1, 4*node.code+child, bary=part)
                          for child, part in enumerate(exact.subdivide(node.bary)))
        node.active = False; node.bary = None
        self.meter.counts['added_cells'] += 3
        queue.extend(range(first+3, first-1, -1))

    def topology(self):
        self.meter.phase = 'topology'
        edges = defaultdict(list)
        for number, node in enumerate(self.nodes):
            if not node.active or not node.good:continue
            self.meter.touch()
            face = int(self.domain.source_faces[node.root]); ids = tuple(int(x) for x in self.domain.faces[face])
            for a, b in zip(node.bary, node.bary[1:]+node.bary[:1]):
                edge = exact.edge_interval(face, ids, a, b)
                if edge is not None:edges[edge[0]].append((edge[1], edge[2], number))
        for group in edges.values():
            active = []
            for low, high, number in sorted(group):
                self.meter.touch()
                active = [(end, other) for end, other in active if end > low]
                for end, other in active:
                    self.meter.touch('adjacencies')
                    if number != other and min(end, high) > low:
                        self.neighbors[number].add(other); self.neighbors[other].add(number)
                active.append((high, number))

    def _proof(self, number, seed_id, parent):
        self.meter.touch('proofs')
        node = self.nodes[number]
        witness = len(self.proofs); self.proofs.append((number, seed_id, parent))
        if node.witness < 0:node.witness = witness; node.status = 1
        return witness

    def connect(self):
        self.meter.phase = 'connectivity'
        for seed_id in range(len(self.seeds)):
            visited = set(self.roots[seed_id]); queue = deque(sorted(self.roots[seed_id].items()))
            while queue:
                self.meter.touch()
                number, parent = queue.popleft()
                for other in sorted(self.neighbors[number]):
                    if other not in visited and seed_id in self.nodes[other].good:
                        witness = self._proof(other, seed_id, parent)
                        visited.add(other); queue.append((other, witness))

    def run(self):
        self.meter.check()
        try:
            self.seed(); self.cells(); self.topology(); self.connect()
        except _BudgetStop:
            pass
        self.meter.check()


@diagnosed('coverage.partition_arrays')
def _arrays(engine):
    values, remap = engine.leaf_arrays()
    values.update(proof_leaf=[remap[item[0]] for item in engine.proofs],
                  proof_seed=[item[1] for item in engine.proofs], proof_parent=[item[2] for item in engine.proofs],
                  seed_face=[seed['face'] for seed in engine.seeds],
                  seed_input_index=[seed['input'] for seed in engine.seeds],
                  seed_xyz=np.asarray([seed['point'] for seed in engine.seeds]).reshape(-1, 3),
                  seed_inner_radius=[seed['inner'] for seed in engine.seeds])
    result = {key:np.ascontiguousarray(value, dtype=DTYPES[key]) for key, value in values.items()}
    if sum(value.nbytes for value in result.values()) > MAX_ARRAY_BYTES:raise ValueError('覆盖数组超过 256 MiB 预算')
    for value in result.values():value.setflags(write=False)
    return result


@diagnosed('coverage.area_summary')
def _summary(domain, arrays, budget, metrics):
    denominator = 4**budget.max_depth
    weights = np.left_shift(np.int64(1), 2*(budget.max_depth-arrays['depth'].astype(np.int64)))
    sums = np.zeros(len(domain.areas), np.int64); np.add.at(sums, arrays['root'], weights)
    if np.any(sums != denominator):raise ValueError('覆盖细分没有完整保留原目标分区')
    certified = np.zeros(len(domain.areas), np.int64)
    mask = arrays['status'] == 1
    np.add.at(certified, arrays['root'][mask], weights[mask])
    total = math.fsum(float(v) for v in domain.areas)
    covered = math.fsum(float(area)*(int(weight)/denominator) for area, weight in zip(domain.areas, certified))
    complete = (len(domain.areas) > 0 and bool(np.all(certified == denominator))
                and not metrics['limits_reached'] and domain.binding['original_target_complete']
                and domain.binding['unknown_pixel_count'] == 0 and not np.any(domain.quality))
    status = ('model_coverage_complete' if complete else 'calculation_incomplete' if metrics['limits_reached']
              else 'partially_confirmed' if covered else 'no_confirmed_area')
    return dict(status=status, original_cell_count=len(domain.areas), leaf_count=len(arrays['root']),
                confirmed_leaf_count=int(mask.sum()), located_nominal_area_m2=total,
                confirmed_nominal_area_m2=covered, unconfirmed_nominal_area_m2=total-covered,
                confirmed_fraction_of_located_nominal_area=covered/total if total else None,
                unlocated_pixel_count=domain.binding['unknown_pixel_count'], unlocated_area_m2=None,
                total_original_target_area_m2=None if domain.binding['unknown_pixel_count'] else total,
                original_target_coverage_complete=complete,
                reasons={name:int(np.count_nonzero(arrays['status'] == code)) for code, name in STATES.items()},
                scope=domain.binding['scope'], execution_ready=False, physical_accuracy_verified=False)


@diagnosed('coverage.evaluate')
def evaluate(domain: CoverageDomain, seeds: list[dict], model: dict, *, route_binding: dict | None = None,
             budget: CoverageBudget | None = None, check: Callable[[], None] = lambda: None,
             algorithm: str = ALGORITHM, _frontier: dict | None = None,
             guide: Certificate | None = None) -> Certificate:
    """Prove a subset of nominal coverage; the retained route is never changed."""
    check()
    footprint_model.validate(model)
    budget = (budget or CoverageBudget()).checked()
    paged=checked_page_binding(domain.binding)
    if paged:
        from femto.planning import target_surface
        if (domain.ledger is None or domain.revision is None
                or domain.ledger.metadata.get('schema')!=target_surface.PAGED_SCHEMA
                or domain.binding['target_sha256']!=domain.ledger.metadata['sha256']):
            raise ValueError('分块覆盖必须关联实际分块目标资产')
        target_surface.validate_metadata(domain.ledger.metadata,domain.revision)
    if len(domain.areas) > 200_000 and not paged:raise ValueError('原目标单元超过覆盖原型输入预算')
    if paged and sum(domain.binding['target_page_units'])!=len(domain.areas):raise ValueError('覆盖分块单元与完整目标不一致')
    if algorithm in (PAGED_ALGORITHM,*PAGED_RECALCULATION_ALGORITHMS) and not paged:raise ValueError('分块覆盖需要完整分块目标')
    if paged and algorithm not in (PAGED_ALGORITHM,*PAGED_RECALCULATION_ALGORITHMS,*REPROOF_ALGORITHMS,RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM):
        raise ValueError('分块目标必须使用声明分块输入的覆盖版本')
    if algorithm == RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM and not paged:
        raise ValueError('分块研究分数位置复算需要完整分块目标')
    if not isinstance(seeds, list) or len(seeds) > 400_000:raise ValueError('覆盖种子输入超过预算')
    if algorithm not in SUPPORTED_ALGORITHMS:raise ValueError('覆盖算法版本不受支持')
    if algorithm not in GUIDED_ALGORITHMS and guide is not None:raise ValueError('此覆盖版本不接受引导分区')
    binding = dict(domain=domain.binding, seeds_sha256=digest(seeds), footprint=model,
                   route=route_binding or dict(schema='declared-model-seeds-v1', path_acceptance_verified=False),
                   algorithm=algorithm, budget=asdict(budget))
    if route_binding and domain.revision is not None and route_binding.get('computation_sha256') != domain.revision['sha256']:
        raise ValueError('覆盖路线与目标计算修订不一致')
    engine_type = _Engine; extra = {}
    if algorithm == INDEXED_ALGORITHM:
        from femto.planning.coverage_indexed import IndexedLayeredEngine, POLICY
        engine_type = IndexedLayeredEngine
        binding['search_policy'] = dict(POLICY)
    elif algorithm == PAGED_ALGORITHM:
        from femto.planning.coverage_schedule import LayeredEngine, POLICY
        engine_type = LayeredEngine
        binding['search_policy'] = dict(POLICY)
    elif algorithm == CURRENT_ALGORITHM:
        from femto.planning.coverage_schedule import LayeredEngine, POLICY
        engine_type = LayeredEngine
        binding['search_policy'] = dict(POLICY)
    elif algorithm in GUIDED_ALGORITHMS:
        if algorithm == GROUPED_RECALCULATION_ALGORITHM:
            from femto.planning.coverage_grouped import GroupedRecalculationEngine as ReproofEngine, POLICY
        elif algorithm in RECALCULATION_ALGORITHMS:
            from femto.planning.coverage_recalculation import RecalculationEngine as ReproofEngine, POLICY
        elif algorithm == INTERLEAVED_ALGORITHM:
            from femto.planning.coverage_interleaved import InterleavedEngine as ReproofEngine, POLICY
        elif algorithm == INDEXED_PARTIAL_REPROOF_ALGORITHM:
            from femto.planning.coverage_indexed_reproof import IndexedPartialReproofEngine as ReproofEngine, POLICY
        elif algorithm in RESEARCH_FRACTION_ALGORITHMS:
            from femto.planning.coverage_indexed_reproof import IndexedPartialReproofEngine as ReproofEngine
            if algorithm == RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM:
                from femto.planning.fraction_paged_reproof_policy import POLICY
            else:
                from femto.planning.fraction_reproof_policy import POLICY
        elif algorithm == REPROOF_ALGORITHM:
            from femto.planning.coverage_reproof import ReproofEngine, POLICY
        else:
            from femto.planning.coverage_residual import ResidualEngine as ReproofEngine, POLICY
        if not isinstance(guide, Certificate):raise ValueError('覆盖复算需要前一轮完整引导证据')
        _checked_assets(guide, check)
        def identity(value):
            return {k:v for k,v in value.items() if k not in ('sha256', 'target_sha256', 'computation_sha256')}
        if (algorithm in RESEARCH_FRACTION_ALGORITHMS
                and model.get('schema') != 'femto-research-nominal-dyadic-ball-v1'):
            raise ValueError('研究分数位置复算需要专用作用模型')
        expected_guide_model = (footprint_model.nominal(model['effective_width_mm'])
                                if algorithm in RESEARCH_FRACTION_ALGORITHMS else model)
        if (guide.metadata['inputs']['footprint'] != expected_guide_model
                or identity(guide.metadata['inputs']['domain']) != identity(domain.binding)):
            raise ValueError('覆盖引导不能跨原目标或作用模型使用')
        engine_type = ReproofEngine; extra['guide'] = guide
        binding.update(search_policy=dict(POLICY), guide_sha256=guide.metadata['sha256'])
    engine = engine_type(domain, seeds, model, budget, check, _frontier, **extra); engine.run()
    arrays = _arrays(engine)
    meter = engine.meter
    metrics = dict(meter.counts, steps=meter.steps, stop_phase=meter.phase,
                   stop_reason=meter.stop_reason, limits_reached=sorted(meter.flags))
    metadata = dict(schema=SCHEMA, algorithm=algorithm, coordinate_frame='world_rdf', unit='metre',
                    coverage_computation_sha256=digest(binding), inputs=binding,
                    area_semantics='nominal_root_weights_not_a_certified_physical_or_roundoff_area_bound',
                    proof_rule='one_inner_ball_and_positive_shared_subedge_path_to_exact_anchor',
                    states={str(key):value for key,value in STATES.items()}, metrics=metrics,
                    summary=_summary(domain, arrays, budget, metrics),
                    arrays={key:array_record(value) for key, value in sorted(arrays.items())},
                    execution_ready=False, physical_accuracy_verified=False)
    metadata['sha256'] = digest(metadata)
    if algorithm in RECALCULATION_ALGORITHMS:
        from femto.planning.coverage_recalculation import RecalculatedCertificate, flat_hint, require_guide
        result = RecalculatedCertificate(metadata, arrays, flat_hint(guide))
        require_guide(result, check)
        return result
    return Certificate(metadata, arrays)


def checked_page_binding(binding):
    units=binding.get('target_page_units')
    if units is None:return False
    if (binding.get('target_construction_schema')!='femto-original-target-surface-paged-v1'
            or not isinstance(units,list) or not 1<=len(units)<=128
            or any(type(n) is not int or not 0<=n<=200_000 for n in units)):
        raise ValueError('覆盖分块输入记录无效')
    # A complete materialized target already obeys its 512 MiB asset cap.
    # Coverage output and all proof/search budgets retain their existing caps.
    return True


def validate_metadata(metadata: dict) -> CoverageBudget:
    """Check all sizes before an archive reader allocates numerical arrays."""
    if (not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA or metadata.get('algorithm') not in SUPPORTED_ALGORITHMS
            or metadata.get('sha256') != digest({key:value for key, value in metadata.items() if key != 'sha256'})
            or metadata.get('coordinate_frame') != 'world_rdf' or metadata.get('unit') != 'metre'
            or metadata.get('execution_ready') is not False or metadata.get('physical_accuracy_verified') is not False):
        raise ValueError('覆盖证据版本、摘要或语义无效')
    try:
        inputs = metadata['inputs']
        if inputs['algorithm'] != metadata['algorithm'] or metadata['coverage_computation_sha256'] != digest(inputs):
            raise ValueError('覆盖计算输入摘要不一致')
        footprint_model.validate(inputs['footprint'])
        paged=checked_page_binding(inputs['domain'])
        if paged and metadata['algorithm'] in (INDEXED_PARTIAL_REPROOF_ALGORITHM, RESEARCH_FRACTION_REPROOF_ALGORITHM):
            raise ValueError('索引部分复算暂不支持分块目标')
        if metadata['algorithm'] in (PAGED_ALGORITHM,*PAGED_RECALCULATION_ALGORITHMS) and not paged:
            raise ValueError('分块覆盖缺少分块目标记录')
        if metadata['algorithm'] == RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM and not paged:
            raise ValueError('分块研究分数位置复算缺少分块目标记录')
        if metadata['algorithm']==PAGED_ALGORITHM:
            from femto.planning.coverage_schedule import POLICY
            if inputs.get('search_policy')!=POLICY:raise ValueError('分块覆盖调度策略不一致')
        if metadata['algorithm'] == INDEXED_ALGORITHM:
            from femto.planning.coverage_indexed import checked_policy
            checked_policy(inputs.get('search_policy'))
        if metadata['algorithm'] in GUIDED_ALGORITHMS:
            if metadata['algorithm'] in RECALCULATION_ALGORITHMS:
                if metadata['algorithm'] == GROUPED_RECALCULATION_ALGORITHM:
                    from femto.planning.coverage_grouped import checked_policy
                else:
                    from femto.planning.coverage_recalculation import checked_policy
                checked_policy(inputs.get('search_policy'))
                if inputs['footprint']['schema'] != footprint_model.SCHEMA:
                    raise ValueError('原集合优先复算只支持固定名义模型')
            elif metadata['algorithm'] == INDEXED_PARTIAL_REPROOF_ALGORITHM:
                from femto.planning.coverage_indexed_reproof import checked_policy
                checked_policy(inputs.get('search_policy'))
            elif metadata['algorithm'] in RESEARCH_FRACTION_ALGORITHMS:
                if metadata['algorithm'] == RESEARCH_PAGED_FRACTION_REPROOF_ALGORITHM:
                    from femto.planning.fraction_paged_reproof_policy import checked_policy
                else:
                    from femto.planning.fraction_reproof_policy import checked_policy
                checked_policy(inputs.get('search_policy'))
                if inputs['footprint']['schema'] != 'femto-research-nominal-dyadic-ball-v1':
                    raise ValueError('研究分数位置复算需要专用作用模型')
            elif reproof_algorithm_for_policy(inputs.get('search_policy')) != metadata['algorithm']:
                raise ValueError('覆盖复算算法与策略版本不一致')
            hint = inputs.get('guide_sha256')
            if not isinstance(hint, str) or len(hint) != 64 or any(c not in '0123456789abcdef' for c in hint):
                raise ValueError('覆盖复算缺少绑定的引导证据摘要')
        budget = CoverageBudget(**inputs['budget']).checked()
        records = metadata['arrays']
        if set(records) != set(DTYPES):raise ValueError('覆盖数组清单不完整')
        total = 0
        for key, dtype in DTYPES.items():
            record = records[key]; shape = record['shape']; sha = record['sha256']
            roots=sum(inputs['domain']['target_page_units']) if paged else 200_000
            cap = budget.max_proofs if key.startswith('proof_') else budget.max_seeds if key.startswith('seed_') else (roots+budget.max_added_cells if paged else 400_000)
            if (set(record) != {'shape', 'dtype', 'sha256'} or record['dtype'] != dtype
                    or not isinstance(shape, list) or len(shape) != (2 if key == 'seed_xyz' else 1)
                    or any(type(v) is not int or v < 0 for v in shape) or shape[0] > cap
                    or (key == 'seed_xyz' and shape[1] != 3)
                    or not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)):
                raise ValueError('覆盖数组尺寸、类型或摘要无效：'+key)
            total += math.prod(shape)*np.dtype(dtype).itemsize
        if total > MAX_ARRAY_BYTES:raise ValueError('覆盖数组超过 256 MiB 预算')
        for keys in (('root', 'depth', 'code', 'status', 'witness'), ('proof_leaf', 'proof_seed', 'proof_parent'),
                     ('seed_face', 'seed_input_index', 'seed_xyz', 'seed_inner_radius')):
            if len({records[key]['shape'][0] for key in keys}) != 1:raise ValueError('覆盖关联数组长度不一致')
        metrics = metadata['metrics']
        if (type(metrics['steps']) is not int or not 0 <= metrics['steps'] <= 10_000_000
                or metrics['stop_reason'] not in ('complete', 'time_budget', 'comparisons_budget', 'adjacencies_budget', 'proofs_budget')):
            raise ValueError('覆盖复算边界无效')
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('覆盖清单缺少有效结构') from error
    return budget


def _checked_assets(certificate: Certificate, check: Callable[[], None]) -> CoverageBudget:
    """Structural checks only; this does not validate any coverage claim."""
    metadata = certificate.metadata
    budget = validate_metadata(metadata)
    if set(certificate.arrays) != set(DTYPES):raise ValueError('覆盖数组清单不完整')
    for key, value in certificate.arrays.items():
        check()
        if not isinstance(value, np.ndarray) or value.dtype.str != DTYPES[key] or array_record(value) != metadata['arrays'].get(key):
            raise ValueError('覆盖数组损坏：'+key)
    return budget


@diagnosed('coverage.replay')
def validate(certificate: Certificate, domain: CoverageDomain, seeds: list[dict], model: dict, *,
             route_binding: dict | None = None, check: Callable[[], None] = lambda: None,
             guide: Certificate | None = None) -> Certificate:
    """Recompute using caller-supplied authoritative inputs, never archive claims."""
    metadata = certificate.metadata
    budget = _checked_assets(certificate, check)
    if metadata['algorithm'] in RECALCULATION_ALGORITHMS:
        from femto.planning.coverage_recalculation import require_guide
        actual = require_guide(certificate, check)
        if guide is not None and guide.metadata != actual.metadata:
            raise ValueError('原集合优先复算提供了不一致的引导')
        guide = actual
    rebuilt = evaluate(domain, seeds, model, route_binding=route_binding,
                       budget=budget, check=check,
                       algorithm=metadata['algorithm'], _frontier=metadata['metrics'], guide=guide)
    if rebuilt.metadata != metadata:raise ValueError('覆盖归属、连通证据、分区或面积无法重现')
    return certificate
