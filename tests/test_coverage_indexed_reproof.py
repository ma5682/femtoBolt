"""Versioned partial-guide indexing never treats a hint as a proof."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
import unittest

import numpy as np

from femto.geometry import exact_surface as exact
from femto.planning import coverage_certificate as coverage,footprint_model
from femto.planning.planning_revision import digest
from femto.planning.coverage_domain import CoverageDomain
from test_coverage_certificate import triangle_domain,seed,reseal
from test_exact_recomputation import edge_reference


class IndexedReproofTests(unittest.TestCase):
    def setUp(self):
        self.domain=triangle_domain();self.model=footprint_model.nominal(125)
        self.records=[seed()];self.budget=coverage.CoverageBudget(max_depth=3)
        self.old=coverage.evaluate(self.domain,self.records,self.model,budget=self.budget,
                                   algorithm=coverage.CURRENT_ALGORITHM)

    def run_version(self,records,budget=None):
        return coverage.evaluate(self.domain,records,self.model,budget=budget or self.budget,
            algorithm=coverage.INDEXED_PARTIAL_REPROOF_ALGORITHM,guide=self.old)

    def test_removed_witness_never_inherits_old_confirmations(self):
        self.assertGreater(self.old.summary['confirmed_leaf_count'],0)
        result=self.run_version([])
        self.assertEqual(result.summary['confirmed_leaf_count'],0)
        self.assertFalse(result.summary['original_target_coverage_complete'])
        coverage.validate(result,self.domain,[],self.model,guide=self.old)

    def test_new_seed_is_checked_even_when_old_seed_is_missing(self):
        records=[seed((.125,0,0))]
        result=self.run_version(records)
        self.assertGreater(result.summary['confirmed_leaf_count'],0)
        self.assertIn('retained_proofs_unconfirmed',result.metadata['metrics']['limits_reached'])
        coverage.validate(result,self.domain,records,self.model,guide=self.old)

    def test_every_emitted_proof_is_contained_and_connected(self):
        records=self.records+[seed((.125,0,0))]
        result=self.run_version(records);a=result.arrays
        cells=[exact.descend(exact.IDENTITY,int(d),int(c)) for d,c in zip(a['depth'],a['code'])]
        for i,(leaf,seed_id,parent) in enumerate(zip(a['proof_leaf'],a['proof_seed'],a['proof_parent'])):
            face=int(self.domain.source_faces[a['root'][leaf]])
            tri=tuple(exact.xyz(p,self.domain.triangle(face)) for p in cells[leaf])
            record=records[int(a['seed_input_index'][seed_id])]
            bary,center,inner=exact.anchor_ball(record['xyz_m'],self.domain.triangle(int(a['seed_face'][seed_id])),.0625)
            self.assertTrue(all(sum((x-y)**2 for x,y in zip(point,center))<=Q(inner)**2 for point in tri))
            if parent<0:
                self.assertEqual(face,a['seed_face'][seed_id])
                self.assertTrue(exact.contains_point(cells[leaf],bary))
            else:
                self.assertLess(parent,i)
                self.assertEqual(a['proof_seed'][parent],seed_id)
                other=int(a['proof_leaf'][parent]);edges=[]
                for number in (int(leaf),other):
                    f=int(self.domain.source_faces[a['root'][number]])
                    ids=tuple(int(v) for v in self.domain.faces[f]);cell=cells[number]
                    edges.append([edge_reference(f,ids,x,y) for x,y in zip(cell,cell[1:]+cell[:1])])
                self.assertTrue(any(x and y and x[0]==y[0] and min(x[2],y[2])>max(x[1],y[1])
                                    for x in edges[0] for y in edges[1]))
        coverage.validate(result,self.domain,records,self.model,guide=self.old)

    def test_budget_exit_keeps_whole_partition_and_replays(self):
        from femto.planning.coverage_partition import intervals
        for seconds in (0.,45.):
            result=self.run_version(self.records+[seed((.125,0,0))],replace(self.budget,seconds=seconds))
            intervals(result,len(self.domain.areas))
            coverage.validate(result,self.domain,self.records+[seed((.125,0,0))],self.model,guide=self.old)

    def test_adjacent_and_disconnected_faces_only_emit_supported_proofs(self):
        vertices=[[0,0,0],[.125,0,0],[0,.125,0],[.125,.125,0],
                  [1,0,0],[1.125,0,0],[1,.125,0]]
        domain=CoverageDomain.mesh(vertices,[[0,1,2],[1,3,2],[4,5,6]])
        old=coverage.evaluate(domain,[seed()],self.model,budget=self.budget,algorithm=coverage.CURRENT_ALGORITHM)
        records=[seed(),seed((.125,.125,0),1),seed((1.,0,0),2)]
        result=coverage.evaluate(domain,records,self.model,budget=self.budget,
                                 algorithm=coverage.INDEXED_PARTIAL_REPROOF_ALGORITHM,guide=old)
        coverage.validate(result,domain,records,self.model,guide=old)
        a=result.arrays;cells=[exact.descend(exact.IDENTITY,int(d),int(c)) for d,c in zip(a['depth'],a['code'])]
        for i,(leaf,sid,parent) in enumerate(zip(a['proof_leaf'],a['proof_seed'],a['proof_parent'])):
            face=int(domain.source_faces[a['root'][leaf]])
            _,center,inner=exact.anchor_ball(records[int(a['seed_input_index'][sid])]['xyz_m'],
                                            domain.triangle(int(a['seed_face'][sid])),.0625)
            self.assertTrue(all(sum((x-y)**2 for x,y in zip(exact.xyz(p,domain.triangle(face)),center))<=Q(inner)**2
                                for p in cells[leaf]))
            if parent>=0:
                self.assertLess(parent,i);self.assertEqual(a['proof_seed'][parent],sid)
                paths=[]
                for number in (int(leaf),int(a['proof_leaf'][parent])):
                    f=int(domain.source_faces[a['root'][number]]);cell=cells[number]
                    paths.append([edge_reference(f,tuple(int(v) for v in domain.faces[f]),x,y)
                                  for x,y in zip(cell,cell[1:]+cell[:1])])
                self.assertTrue(any(x and y and x[0]==y[0] and min(x[2],y[2])>max(x[1],y[1])
                                    for x in paths[0] for y in paths[1]))

    def test_restricted_seed_search_does_not_poison_full_candidate_cache(self):
        from femto.planning.coverage_indexed_reproof import IndexedPartialReproofEngine
        records=self.records+[seed((.125,0,0))]
        engine=IndexedPartialReproofEngine(self.domain,records,self.model,self.budget,lambda:None,None,guide=self.old)
        engine.seed();engine._prepare();engine._pixel_bounds();engine._reprove()
        engine._new_neighborhoods(anchor_only=True)
        self.assertIsNone(engine.limited_seeds)
        self.assertEqual(engine._candidate_ids.cache_info().currsize,0)
        np.testing.assert_array_equal(engine._candidate_ids(0),np.asarray(engine.candidates_for(0),np.int64))

    def test_tampered_policy_cannot_replay_under_new_version(self):
        result=self.run_version(self.records+[seed((.125,0,0))])
        metadata=deepcopy(result.metadata);metadata['inputs']['search_policy']['face_selection']='unsafe'
        metadata['coverage_computation_sha256']=digest(metadata['inputs'])
        forged=reseal(result,metadata=metadata)
        with self.assertRaises(ValueError):coverage.validate_metadata(forged.metadata)

    def test_unimplemented_paged_input_is_rejected_before_asset_allocation(self):
        result=self.run_version(self.records+[seed((.125,0,0))])
        metadata=deepcopy(result.metadata);domain=metadata['inputs']['domain']
        domain['target_construction_schema']='femto-original-target-surface-paged-v1'
        domain['target_page_units']=[1]
        metadata['coverage_computation_sha256']=digest(metadata['inputs'])
        forged=reseal(result,metadata=metadata)
        with self.assertRaisesRegex(ValueError,'不支持分块目标'):
            coverage.validate_metadata(forged.metadata)
