"""Guidance is scheduling input, never inherited confirmation or connectivity."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
import unittest
from unittest.mock import patch
from weakref import ref

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning import coverage_certificate as coverage, coverage_partition, footprint_model
from femto.planning.coverage_domain import CoverageDomain
from test_coverage_certificate import triangle_domain, seed, reseal
from test_exact_recomputation import edge_reference, orientation


class CoverageReproofTests(unittest.TestCase):
    def setUp(self):
        self.domain = triangle_domain()
        self.model = footprint_model.nominal(125)
        self.records = [seed()]
        self.budget = coverage.CoverageBudget(max_depth=3)
        self.old = coverage.evaluate(self.domain, self.records, self.model,
            budget=self.budget, algorithm=coverage.CURRENT_ALGORITHM)

    def evaluate(self, records=None, guide=None, domain=None, budget=None):
        return coverage.evaluate(self.domain if domain is None else domain,
            self.records if records is None else records, self.model,
            budget=self.budget if budget is None else budget, algorithm=coverage.REPROOF_ALGORITHM,
            guide=self.old if guide is None else guide)

    def test_new_seed_gains_area_and_keeps_original_partition_set(self):
        records = self.records+[seed((.125, 0, 0))]
        result = self.evaluate(records)
        comparison = coverage_partition.compare(self.old, result, self.domain.areas)
        self.assertTrue(comparison['retained_subset'])
        self.assertEqual(comparison['lost_weight_units'], 0)
        self.assertGreater(comparison['gained_weight_units'], 0)
        self.assertEqual(result.metadata['metrics']['retained_leaves_rechecked'], self.old.summary['confirmed_leaf_count'])
        self.assertGreater(result.metadata['metrics']['hint_proofs_rechecked'], 0)
        coverage.validate(result, self.domain, records, self.model, guide=self.old)

    def test_reproof_caches_release_engine_without_cyclic_collection(self):
        from femto.planning.coverage_reproof import ReproofEngine
        engine = ReproofEngine(self.domain, self.records, self.model, self.budget, lambda: None, None, guide=self.old)
        engine.run()
        self.assertGreater(engine._geometry.cache_info().currsize, 0)
        weak = ref(engine)
        del engine
        self.assertIsNone(weak())

    def test_budget_before_global_exclusions_still_rechecks_new_anchor_and_keeps_original_set(self):
        from femto.planning.coverage_reproof import ReproofEngine
        vertices = [[float(i), 0, 0] for i in range(64) for _ in range(3)]
        for i in range(64):
            vertices[3*i+1][0] += .01; vertices[3*i+2][1] = .01
        domain = CoverageDomain.mesh(vertices, np.arange(192).reshape(-1,3))
        old = coverage.evaluate(domain, self.records, self.model, algorithm=coverage.CURRENT_ALGORITHM)
        records = self.records+[seed((63.,0,0),63)]
        frontier = []
        outside = ReproofEngine._outside_faces
        def observe(engine):
            frontier.append(engine.meter.steps)
            self.assertEqual(engine.nodes[63].status, 1)
            return outside(engine)
        with patch.object(ReproofEngine, '_outside_faces', observe):
            self.evaluate(records, domain=domain, guide=old)
        limited = coverage.evaluate(domain, records, self.model, budget=self.budget,
            algorithm=coverage.REPROOF_ALGORITHM, guide=old,
            _frontier=dict(stop_reason='time_budget',steps=frontier[0]))
        comparison = coverage_partition.compare(old, limited, domain.areas)
        self.assertEqual(comparison['lost_weight_units'], 0)
        self.assertGreater(comparison['gained_weight_units'], 0)
        self.assertEqual(limited.metadata['metrics']['outside_roots_rechecked'], 0)
        self.assertTrue(np.all(limited.arrays['status'][1:63] == 7))
        coverage_partition.intervals(limited, len(domain.areas))
        coverage.validate(limited, domain, records, self.model, guide=old)

    def test_interrupted_new_ball_link_preserves_already_rechecked_witness(self):
        from femto.planning.coverage_reproof import ReproofEngine
        engine = ReproofEngine(self.domain,self.records,self.model,self.budget,lambda:None,None,guide=self.old)
        engine.seed(); engine._prepare(); engine._pixel_bounds(); self.assertTrue(engine._reprove())
        number = int(engine.old_confirmed[0])
        witness = engine.nodes[number].witness
        with patch.object(engine,'_link',side_effect=coverage._BudgetStop), self.assertRaises(coverage._BudgetStop):
            engine._visit(number)
        arrays, remap = engine.leaf_arrays()
        self.assertEqual(arrays['status'][remap[number]],1)
        self.assertEqual(arrays['witness'][remap[number]],witness)

    def test_lazy_partition_serializes_like_all_eager_nodes_at_budget_and_subdivision(self):
        from femto.planning.coverage_reproof import ReproofEngine
        for seconds in (0., 45.):
            budget = replace(self.budget, seconds=seconds)
            engine = ReproofEngine(self.domain, self.records+[seed((.125,0,0))], self.model,
                                   budget, lambda: None, None, guide=self.old)
            engine.run()
            actual, remap = engine.leaf_arrays()
            expected, reference = coverage._Engine.leaf_arrays(engine)
            for key in actual:np.testing.assert_array_equal(actual[key], expected[key])
            for key, value in reference.items():self.assertEqual(remap[key], value)

    def test_every_proof_has_independent_fraction_containment_and_source_connectivity(self):
        records = self.records+[seed((.125, 0, 0))]
        result = self.evaluate(records); a = result.arrays
        cells = [exact.descend(exact.IDENTITY, int(d), int(c)) for d, c in zip(a['depth'], a['code'])]
        for i, (leaf, seed_id, parent) in enumerate(zip(a['proof_leaf'], a['proof_seed'], a['proof_parent'])):
            face = int(self.domain.source_faces[a['root'][leaf]])
            tri = tuple(tuple(sum(p[j]*self.domain.triangle(face)[j][axis] for j in range(3)) for axis in range(3))
                        for p in cells[leaf])
            record = records[int(a['seed_input_index'][seed_id])]
            bary, center, inner = exact.anchor_ball(record['xyz_m'], self.domain.triangle(int(a['seed_face'][seed_id])), .0625)
            self.assertEqual(a['seed_inner_radius'][seed_id], inner)
            for point in tri:
                self.assertLessEqual(sum((x-y)**2 for x, y in zip(point, center)), Q(inner)**2)
            if parent < 0:
                self.assertEqual(face, a['seed_face'][seed_id])
                signs = [orientation(left, right, bary) for left, right in zip(cells[leaf], cells[leaf][1:]+cells[leaf][:1])]
                self.assertTrue(all(v >= 0 for v in signs) or all(v <= 0 for v in signs))
            else:
                self.assertLess(parent, i); self.assertEqual(seed_id, a['proof_seed'][parent])
                other = a['proof_leaf'][parent]
                edges = []
                for number in (leaf, other):
                    f = int(self.domain.source_faces[a['root'][number]])
                    ids = tuple(int(v) for v in self.domain.faces[f]); cell = cells[number]
                    edges.append([edge_reference(f, ids, x, y) for x, y in zip(cell, cell[1:]+cell[:1])])
                self.assertTrue(any(x and y and x[0] == y[0] and min(x[2], y[2]) > max(x[1], y[1])
                                    for x in edges[0] for y in edges[1]))

    def test_removed_witness_seed_does_not_reuse_old_confirmation(self):
        result = self.evaluate([])
        self.assertEqual(result.summary['confirmed_leaf_count'], 0)
        self.assertGreater(result.metadata['metrics']['unbound_hint_witnesses'], 0)
        self.assertFalse(coverage_partition.compare(self.old, result, self.domain.areas)['retained_subset'])

    def test_outside_classification_is_recomputed_and_new_ball_invalidates_it(self):
        domain = CoverageDomain.mesh([[0,0,0],[.01,0,0],[0,.01,0],
            [1,0,0],[1.01,0,0],[1,.01,0]], [[0,1,2],[3,4,5]])
        old = coverage.evaluate(domain, self.records, self.model, algorithm=coverage.CURRENT_ALGORITHM)
        self.assertEqual(old.arrays['status'].tolist(), [1,3])
        unchanged = self.evaluate(domain=domain, guide=old)
        self.assertEqual(unchanged.arrays['status'].tolist(), [1,3])
        self.assertEqual(unchanged.metadata['metrics']['outside_roots_rechecked'], 1)
        records = self.records+[dict(source_face=1, xyz_m=[1.,0.,0.], role='scan')]
        changed = self.evaluate(records, domain=domain, guide=old)
        self.assertEqual(changed.arrays['status'].tolist(), [1,1])
        self.assertEqual(changed.metadata['metrics']['outside_roots_rechecked'], 0)
        coverage.validate(changed, domain, records, self.model, guide=old)

    def test_outside_aabb_never_excludes_a_face_intersecting_a_ball_at_boundary(self):
        radius = .0625
        domain = CoverageDomain.mesh([[0,0,0],[.01,0,0],[0,.01,0],
            [radius,0,0],[radius,.01,0],[radius,0,.01]], [[0,1,2],[3,4,5]])
        old = coverage.evaluate(domain, self.records, self.model, budget=self.budget, algorithm=coverage.CURRENT_ALGORITHM)
        result = self.evaluate(domain=domain, guide=old)
        self.assertEqual(result.metadata['metrics']['outside_roots_rechecked'], 0)

    def test_pixel_target_prism_and_grid_paths_match_exact_geometric_truth(self):
        import open3d as o3d
        import test_target_surface
        from test_mesh_slice_planner import grid_mesh
        from femto.planning import target_surface, surface_workspace
        from femto.planning.effective_domain import build
        fixture = test_target_surface.TargetSurfaceTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        mesh = grid_mesh(width=.03,height=.03,cell=.005)
        native = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),o3d.utility.Vector3iVector(mesh.faces))
        mask = np.zeros((100,100),bool); mask[47:54,47:54] = True
        source, region = fixture.fixture(mesh=native,mask=mask)
        target, revision, evidence = fixture.build(source,region)
        quality = build(surface_workspace.canonical_region_mesh(source.mesh,region),region,evidence,5)
        target = target_surface.with_quality(target,revision,region,evidence,quality)
        domain = CoverageDomain.target(target,revision,evidence=evidence,effective_domain=quality)
        eligible = np.flatnonzero(domain.quality == 0)
        def record(root):
            face = int(domain.source_faces[root])
            triangle = [exact.xyz(p,domain.triangle(face)) for p in domain.root(int(root))]
            point = [float(sum(p[axis] for p in triangle)/3) for axis in range(3)]
            return seed(point,face)
        records = [record(eligible[0])]; model = footprint_model.nominal(5.)
        budget = coverage.CoverageBudget(max_depth=2)
        old = coverage.evaluate(domain,records,model,budget=budget,algorithm=coverage.CURRENT_ALGORITHM)
        records.append(record(eligible[-1]))
        result = coverage.evaluate(domain,records,model,budget=budget,algorithm=coverage.REPROOF_ALGORITHM,guide=old)
        self.assertTrue(coverage_partition.compare(old,result,domain.areas)['retained_subset'])
        self.assertGreater(result.metadata['metrics']['outside_index_bins_visited'],0)
        self.assertGreater(result.metadata['metrics']['prism_contained_pairs'],0)
        coverage.validate(result,domain,records,model,guide=old)
        balls = [exact.anchor_ball(r['xyz_m'],domain.triangle(r['source_face']),.0025) for r in records]
        arrays = result.arrays
        self.assertGreater(result.metadata['metrics']['shared_endpoint_pairs'],0)
        cells = {}
        def cell_at(index):
            if index not in cells:
                root = int(arrays['root'][index]); face = int(domain.source_faces[root])
                cell = exact.descend(domain.root(root),int(arrays['depth'][index]),int(arrays['code'][index]))
                cells[index] = face, cell
            return cells[index]
        # The shortcut must retain the same positive-subedge and containment
        # truth on actual pixel roots; neither the integer key nor its result
        # is used by this Fraction reference.
        for leaf, seed_id, parent in zip(arrays['proof_leaf'], arrays['proof_seed'], arrays['proof_parent']):
            face, cell = cell_at(leaf)
            ball = balls[int(arrays['seed_input_index'][seed_id])]
            for point in cell:
                xyz = tuple(sum(point[j]*domain.triangle(face)[j][axis] for j in range(3)) for axis in range(3))
                self.assertLessEqual(sum((x-y)**2 for x,y in zip(xyz,ball[1])),Q(ball[2])**2)
            if parent >= 0:
                other_face, other_cell = cell_at(arrays['proof_leaf'][parent])
                edges = []
                for f, vertices in ((face,cell),(other_face,other_cell)):
                    ids = tuple(int(v) for v in domain.faces[f])
                    edges.append([edge_reference(f,ids,x,y) for x,y in zip(vertices,vertices[1:]+vertices[:1])])
                self.assertTrue(any(x and y and x[0]==y[0] and min(x[2],y[2])>max(x[1],y[1])
                                    for x in edges[0] for y in edges[1]))
            else:
                self.assertEqual(face, arrays['seed_face'][seed_id])
                signs = [orientation(x,y,ball[0]) for x,y in zip(cell,cell[1:]+cell[:1])]
                self.assertTrue(all(v>=0 for v in signs) or all(v<=0 for v in signs))
        for index in np.flatnonzero(arrays['status'] == 3):
            root = int(arrays['root'][index]); face = int(domain.source_faces[root])
            cell = exact.descend(domain.root(root),int(arrays['depth'][index]),int(arrays['code'][index]))
            triangle = tuple(exact.xyz(p,domain.triangle(face)) for p in cell)
            for ball in balls:
                if ball is not None:self.assertGreater(coverage._distance2(ball[1],triangle),Q(ball[2])**2)

    def test_each_budget_keeps_all_original_area_and_replays(self):
        for limits in (dict(seconds=0), dict(max_seeds=0), dict(max_comparisons=0),
                       dict(max_adjacencies=0), dict(max_proofs=0), dict(max_added_cells=0)):
            with self.subTest(limits=limits):
                result = self.evaluate(budget=replace(self.budget, **limits))
                coverage_partition.intervals(result, len(self.domain.areas))
                self.assertEqual(result.summary['located_nominal_area_m2'], self.old.summary['located_nominal_area_m2'])
                self.assertFalse(result.summary['original_target_coverage_complete'])
                coverage.validate(result, self.domain, self.records, self.model, guide=self.old)

    def test_batch_accounting_keeps_exact_caps_and_partial_replay_frontier(self):
        for stop in (0,1,31,32,33,75):
            for kind in ('comparisons','adjacencies','proofs'):
                budget = replace(self.budget, **{'max_'+kind:stop})
                meter = coverage._Meter(budget, lambda: None, None)
                with self.assertRaises(coverage._BudgetStop):meter.touch_many(100, kind)
                self.assertEqual(meter.counts[kind], stop)
                self.assertEqual(meter.steps, stop)
                self.assertEqual(meter.stop_reason, kind+'_budget')
            meter = coverage._Meter(self.budget, lambda: None, dict(stop_reason='time_budget', steps=stop))
            with self.assertRaises(coverage._BudgetStop):meter.touch_many(100, 'comparisons')
            self.assertEqual(meter.steps, stop)
            self.assertEqual(meter.counts['comparisons'], stop)
            self.assertEqual(meter.stop_reason, 'time_budget')

    def test_time_stop_during_preparation_or_reproof_has_replayable_frontier(self):
        for seconds in (1., 6., 20., 40.):
            with self.subTest(seconds=seconds):
                with patch('femto.planning.coverage_certificate.time.monotonic', side_effect=range(100000)):
                    result = self.evaluate(budget=replace(self.budget, seconds=seconds))
                self.assertEqual(result.metadata['metrics']['stop_reason'], 'time_budget')
                coverage_partition.intervals(result, len(self.domain.areas))
                coverage.validate(result, self.domain, self.records, self.model, guide=self.old)

    def test_guide_from_another_model_or_target_is_refused(self):
        with self.assertRaisesRegex(ValueError, '原目标|作用模型'):
            coverage.evaluate(self.domain, self.records, footprint_model.nominal(250),
                algorithm=coverage.REPROOF_ALGORITHM, guide=self.old)
        with self.assertRaisesRegex(ValueError, '原目标'):
            self.evaluate(domain=triangle_domain(.25))

    def test_rehashed_false_parent_cannot_connect_distinct_layers(self):
        domain = CoverageDomain.mesh([[0, 0, 0], [.01, 0, 0], [0, .01, 0],
                                      [0, 0, .001], [.01, 0, .001], [0, .01, .001]], [[0, 1, 2], [3, 4, 5]])
        old = coverage.evaluate(domain, self.records, self.model, algorithm=coverage.CURRENT_ALGORITHM)
        self.assertEqual(old.arrays['status'].tolist(), [1, 4])
        arrays = {k: v.copy() for k, v in old.arrays.items()}
        arrays['status'][1] = 1; arrays['witness'][1] = 1
        for key, value in (('proof_leaf', 1), ('proof_seed', 0), ('proof_parent', 0)):
            arrays[key] = np.append(arrays[key], np.int64(value))
        forged = reseal(old, arrays=arrays)
        result = self.evaluate(guide=forged, domain=domain)
        self.assertEqual(result.arrays['status'][1], 7)
        self.assertEqual(result.metadata['metrics']['hint_connectivity_failures'], 1)

    def test_rehashed_false_containment_is_rechecked(self):
        arrays = {k:v.copy() for k,v in self.old.arrays.items()}
        unknown = int(np.flatnonzero(arrays['status'] != 1)[-1])
        proof = len(arrays['proof_leaf'])
        arrays['status'][unknown] = 1; arrays['witness'][unknown] = proof
        for key, value in (('proof_leaf', unknown), ('proof_seed', 0), ('proof_parent', 0)):
            arrays[key] = np.append(arrays[key], np.int64(value))
        result = self.evaluate(guide=reseal(self.old, arrays=arrays))
        self.assertNotEqual(result.arrays['status'][unknown], 1)
        self.assertGreater(result.metadata['metrics']['hint_containment_failures'], 0)

    def test_cycles_missing_roots_and_wrong_witness_links_are_rejected(self):
        for key, change in (('proof_parent', lambda a: a.__setitem__(0, 0)),
                            ('root', lambda a: a.__setitem__(0, 1)),
                            ('witness', lambda a: a.__setitem__(int(np.flatnonzero(a >= 0)[0]), 99999))):
            arrays = {k:v.copy() for k,v in self.old.arrays.items()}; change(arrays[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.evaluate(guide=reseal(self.old, arrays=arrays))

    def test_standalone_validation_requires_the_bound_guide(self):
        result = self.evaluate()
        with self.assertRaisesRegex(ValueError, '引导证据'):
            coverage.validate(result, self.domain, self.records, self.model)
        metadata = deepcopy(result.metadata); metadata['inputs']['guide_sha256'] = '0'*64
        from femto.planning.planning_revision import digest
        metadata['coverage_computation_sha256'] = digest(metadata['inputs'])
        with self.assertRaises(ValueError):
            coverage.validate(reseal(result, metadata=metadata), self.domain, self.records, self.model, guide=self.old)

    def test_current_quality_exclusion_overrides_all_guide_claims(self):
        d = self.domain
        domain = CoverageDomain(d.vertices, d.faces, d.source_faces, d.areas, np.ones(len(d.areas), 'u1'), d.binding)
        result = self.evaluate(domain=domain)
        self.assertEqual(result.summary['confirmed_leaf_count'], 0)
        self.assertTrue(np.all(result.arrays['status'] == 2))

    def test_cancel_returns_no_certificate_and_preserves_guide_arrays(self):
        before = {k:v.copy() for k,v in self.old.arrays.items()}
        cancelled = False
        original = exact.contains_ball_pairs
        def reached_batch(*args, **kwargs):
            nonlocal cancelled
            result = original(*args, **kwargs)
            cancelled = True
            return result
        def cancel():
            if cancelled:raise RuntimeError('cancel during reproof')
        with patch.object(exact, 'contains_ball_pairs', side_effect=reached_batch), self.assertRaisesRegex(RuntimeError, 'cancel during reproof'):
            coverage.evaluate(self.domain, self.records, self.model, algorithm=coverage.REPROOF_ALGORITHM,
                guide=self.old, budget=self.budget, check=cancel)
        for key, value in before.items():np.testing.assert_array_equal(value, self.old.arrays[key])


if __name__ == '__main__': unittest.main()
