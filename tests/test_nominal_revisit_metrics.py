"""Exact nominal scores under permutation, mutation, boundary ties and fallback."""
from copy import deepcopy
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch
import numpy as np
from femto.planning import nominal_revisit_metrics as fast,balanced_visit_improvement as old


def point(xyz,row,role='scan',**extra):
    return dict(xyz_m=[float(v) for v in xyz],row_index=row,role=role,**extra)


class NominalRevisitMetricsTests(unittest.TestCase):
    def test_random_candidates_are_exact_and_do_not_mutate(self):
        rng=np.random.default_rng(1831)
        with fast.operation():
            for trial in range(100):
                rows=[]
                for r in range(5):
                    xyz=rng.normal(0,.006,(4,3))
                    rows.append([point(p,r) for p in xyz])
                for order in (range(5),rng.permutation(5)):
                    chain=[]
                    for r in order:
                        r=int(r)
                        if chain:
                            for k in range(3):
                                chain.append(point(rng.normal(0,.006,3),r,'connector',revisit=k==0 and trial%3==0))
                        chain.extend(rows[r])
                    chains=[(0,chain)];original=deepcopy(chains)
                    self.assertEqual(fast.measure(chains,.01,lambda:None),old._legacy_metrics(chains,.01,lambda:None))
                    self.assertEqual(chains,original)

    def test_exact_radius_duplicate_rows_and_existing_marks(self):
        radius=.005
        for x in (np.nextafter(radius,0.),radius,np.nextafter(radius,np.inf)):
            chain=[point([0,0,0],0),point([0,0,0],1,'connector'),point([.02,0,0],1),
                   point([x,0,0],2),point([x,0,0],2),point([0,0,0],2,'connector',revisit=True)]
            for permutation in (chain,list(reversed(chain[2:]))+chain[:2]):
                # The second arrangement starts with a connector and must use
                # the retained path rather than silently changing its meaning.
                chains=[(0,permutation)]
                with fast.operation():
                    self.assertEqual(old._metrics(chains,.01,lambda:None),old._legacy_metrics(chains,.01,lambda:None))

    def test_changed_scan_coordinates_rows_roles_and_spacing_invalidate(self):
        chains=[(0,[point([0,0,0],0),point([.001,0,0],1,'connector'),point([.02,0,0],1),point([.001,0,0],2)])]
        with fast.operation():
            for key,value in (('xyz_m',[.03,0.,0.]),('row_index',1),('role','connector')):
                self.assertEqual(old._metrics(chains,.01,lambda:None),old._legacy_metrics(chains,.01,lambda:None))
                chains[0][1][-1][key]=value
                for spacing in (.01,.1):
                    self.assertEqual(old._metrics(chains,spacing,lambda:None),old._legacy_metrics(chains,spacing,lambda:None))

    def test_union_index_excludes_removed_scan_heads_on_every_candidate(self):
        base=[point([0,0,0],0),point([.001,0,0],1,'connector'),point([.02,0,0],1)]
        extra=point([.001,0,0],2)
        with fast.operation():
            populated=fast.measure([(0,base+[extra])],.01,lambda:None)
            tree=fast._active.get().tree
            removed=fast.measure([(0,base)],.01,lambda:None)
            self.assertIs(tree,fast._active.get().tree)
            self.assertGreater(populated['revisit_connector_m'],0.)
            self.assertEqual(removed['revisit_connector_m'],0.)
            self.assertEqual(populated,fast.measure([(0,base+[extra])],.01,lambda:None))

    def test_complete_relocation_decisions_match_full_scorer(self):
        from test_budgeted_relocation_research import BudgetedRelocationTests
        from femto.planning import budgeted_relocation_research as relocation
        fixture=BudgetedRelocationTests('test_adopted_candidate_is_fully_replayed_and_bounded');fixture.setUp()
        with patch.object(old,'_metrics',side_effect=old._legacy_metrics):
            expected=relocation.propose(fixture.graph,fixture.segments,fixture.visits,fixture.links,.01,**fixture.options)
        fixture.setUp()
        with fast.operation():
            actual=relocation.propose(fixture.graph,fixture.segments,fixture.visits,fixture.links,.01,**fixture.options)
            self.assertIsNotNone(fast._active.get().tree)
        self.assertEqual(actual,expected)

    def test_query_and_union_limits_bound_memory_without_changing_scores(self):
        base=[point([0,0,0],0),point([.001,0,0],1,'connector'),point([.02,0,0],1)]
        with patch.object(fast,'MAX_SCAN_POINTS',3),patch.object(fast,'MAX_QUERIES',0),fast.operation():
            for x in (.001,.002,.003):
                chains=[(0,base+[point([x,0,0],2)])]
                self.assertEqual(fast.measure(chains,.01,lambda:None),old._legacy_metrics(chains,.01,lambda:None))
                self.assertLessEqual(len(fast._active.get().points),3)
                self.assertEqual(len(fast._active.get().queries),0)

    def test_budget_fallback_and_no_cross_operation_or_thread_reuse(self):
        chains=[(0,[point([0,0,0],0),point([.001,0,0],1,'connector'),point([.02,0,0],1),point([.001,0,0],2)])]
        self.assertIsNone(fast.measure(chains,.01,lambda:None))
        with fast.operation():
            state=fast._active.get()
            self.assertEqual(old._metrics(chains,.01,lambda:None),old._legacy_metrics(chains,.01,lambda:None))
            with fast.operation():self.assertIs(fast._active.get(),state)
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertIsNone(pool.submit(copy_context().run,fast.measure,chains,.01,lambda:None).result())
            with patch.object(fast,'MAX_SCAN_POINTS',1):self.assertIsNone(fast.measure(chains,.01,lambda:None))
        self.assertFalse(state.active);self.assertIsNone(state.tree)
        with fast.operation():self.assertIsNot(fast._active.get(),state)

    def test_unsupported_types_and_cancellation_preserve_old_behavior(self):
        chains=[(0,[point([0,0,0],0),point([.001,0,0],1,'connector'),point([.02,0,0],1)])]
        chains[0][1][0]['xyz_m']=[0,0,0]
        with fast.operation():self.assertIsNone(fast.measure(chains,.01,lambda:None))
        chains[0][1][0]['xyz_m']=[0.,0.,0.]
        def cancelled():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'),fast.operation():fast.measure(chains,.01,cancelled)
        self.assertIsNone(fast._active.get())

    def test_fallback_does_not_consume_unsupported_iterators(self):
        points=[point([0,0,0],0),point([.001,0,0],1,'connector'),point([.02,0,0],1)]
        expected=old._legacy_metrics([(0,points)],.01,lambda:None)
        with fast.operation():
            source=iter([(0,points)])
            self.assertIsNone(fast.measure(source,.01,lambda:None))
            self.assertEqual(old._metrics(source,.01,lambda:None),expected)
            self.assertEqual(old._metrics([iter((0,points))],.01,lambda:None),expected)


if __name__=='__main__':unittest.main()
