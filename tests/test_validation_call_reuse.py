"""One validation call checks its evidence, effective domain and selection once.

Every removed computation repeated an earlier one with the same objects in the
same call, so verdicts and the first raised error must stay unchanged.
"""
from copy import deepcopy
from contextlib import ExitStack
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

import test_lazy_surface_graph as lazy_fixture
from femto.planning import continuous_quality as continuous, quality_constrained as quality
from femto.planning import effective_domain, selection_blocks


class ValidationCallReuseTests(TestCase):
    def setUp(self):
        fixture = lazy_fixture.LazySurfaceTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture

    def plan(self, split=False, routing=continuous.LAZY_ROUTING):
        candidate, mesh, region, record, domain = self.fixture.inputs(split)
        return continuous.replan(candidate, mesh, region, record, 5, domain=domain, routing=routing), mesh, region

    def count(self, function):
        stack = ExitStack()
        calls = {}
        for owner, name in function:
            original = getattr(owner, name)
            def spy(*args, _name=name, _original=original, **kwargs):
                calls[_name] = calls.get(_name, 0) + 1
                return _original(*args, **kwargs)
            stack.enter_context(patch.object(owner, name, spy))
        return stack, calls

    def test_one_evidence_check_domain_rebuild_and_assignment_per_validation(self):
        for split in (False, True):
            with self.subTest(split=split):
                plan, mesh, region = self.plan(split)
                stack, calls = self.count([(quality, 'checked_evidence'), (effective_domain, 'build'),
                                           (selection_blocks, 'selection_face_assignment')])
                with stack:
                    continuous.validate(plan, mesh, region)
                self.assertEqual(calls, {'checked_evidence': 1, 'build': 1, 'selection_face_assignment': 1})

    def test_replan_validates_the_domain_once_and_constrain_is_unchanged(self):
        candidate, mesh, region, record, domain = self.fixture.inputs()
        stack, calls = self.count([(effective_domain, 'build')])
        with stack:
            continuous.replan(candidate, mesh, region, record, 5, domain=domain)
        self.assertEqual(calls, {'build': 1})
        allowed = effective_domain.validate(domain, record, 5, mesh, region)
        fresh = quality.constrain(candidate, mesh, region, record, 5, domain=domain)
        reused = quality.constrain(candidate, mesh, region, record, 5, domain=domain, _allowed_faces=allowed)
        self.assertEqual(fresh.to_dict(), reused.to_dict())

    def recomputing(self):
        """Disable every call-local shortcut, including the lower-level ones."""
        stack = ExitStack()
        for name in ('evidence', 'allowed', 'selection_labels'):
            stack.enter_context(patch.object(quality._ValidationShare, name, lambda *args, **kwargs: None))
        for owner, name, keyword in (
                (effective_domain, 'build', '_classes'),
                (effective_domain, 'validate', '_classes'),
                (quality.ReferenceEnvelope, '__init__', '_evidence'),
                (quality, 'constrain', '_allowed_faces')):
            original = getattr(owner, name)
            def recompute(*args, _original=original, _keyword=keyword, **kwargs):
                kwargs.pop(_keyword, None)
                return _original(*args, **kwargs)
            stack.enter_context(patch.object(owner, name, recompute))
        return stack

    def outcome(self, plan, mesh, region):
        try:
            continuous.validate(plan, mesh, region)
        except Exception as error:
            return type(error), str(error)
        return None

    def test_verdicts_and_first_errors_match_full_recomputation(self):
        for routing in (continuous.LAZY_ROUTING, continuous.ORDERED_ROUTING):
            plan, mesh, region = self.plan(True, routing)
            params = plan.planner['parameters']
            cases = {'unchanged': lambda p, r: None}

            def move_point(p, r):
                segment = next(i for i in p.ordered_items if hasattr(i, 'points') and len(i.points) > 2)
                segment.points[1]['xyz_m'] = [v + .002 for v in segment.points[1]['xyz_m']]
            def domain_count(p, r):
                p.planner['parameters']['effective_domain']['eligible_face_count'] += 1
            def resigned_domain_area(p, r):
                domain = p.planner['parameters']['effective_domain']
                domain['eligible_area_m2'] += .001
                domain['sha256'] = quality._hash(domain)
            def evidence_runs(p, r):
                p.planner['parameters']['quality_evidence']['residual_um_runs'][0] += 1
            def width(p, r):
                p.planner['parameters']['effective_width_mm'] = 6.
            def selection(p, r):
                r['annotation_id'] = r['annotation_id'] + '-changed'
                r['triangle_indices'] = list(r['triangle_indices'])[:-1]
            cases.update(move_point=move_point, domain_count=domain_count,
                         resigned_domain_area=resigned_domain_area, evidence_runs=evidence_runs,
                         width=width, selection=selection)
            for name, tamper in cases.items():
                with self.subTest(routing=routing, case=name):
                    left, left_region = deepcopy(plan), deepcopy(region)
                    tamper(left, left_region)
                    right, right_region = deepcopy(left), deepcopy(left_region)
                    reused = self.outcome(left, mesh, left_region)
                    with self.recomputing():
                        recomputed = self.outcome(right, mesh, right_region)
                    self.assertEqual(reused, recomputed)
                    if name == 'unchanged':self.assertIsNone(reused)
                    else:self.assertIsNotNone(reused)
            self.assertIs(plan.planner['parameters'], params)

    def test_full_recomputation_exercises_the_original_validation_work(self):
        plan, mesh, region = self.plan()
        stack, calls = self.count([(quality, 'checked_evidence'), (effective_domain, 'build'),
                                   (selection_blocks, 'selection_face_assignment')])
        with stack, self.recomputing():
            continuous.validate(plan, mesh, region)
        self.assertEqual(calls, {'checked_evidence': 8, 'build': 2, 'selection_face_assignment': 2})

    def test_separate_calls_recheck_mutated_objects_and_recover_after_cancellation(self):
        plan, mesh, region = self.plan()
        stack, calls = self.count([(effective_domain, 'build')])
        with stack:
            continuous.validate(plan, mesh, region)
            continuous.validate(plan, mesh, region)
        self.assertEqual(calls['build'], 2)

        domain = plan.planner['parameters']['effective_domain']
        original = deepcopy(domain)
        domain['eligible_area_m2'] += .001
        domain['sha256'] = quality._hash(domain)
        changed = self.outcome(plan, mesh, region)
        with self.recomputing():
            self.assertEqual(changed, self.outcome(plan, mesh, region))
        self.assertIsNotNone(changed)
        domain.clear()
        domain.update(original)

        polls = 0
        def cancel():
            nonlocal polls
            polls += 1
            if polls == 3:
                raise RuntimeError('operator cancelled')
        with self.assertRaisesRegex(RuntimeError, 'operator cancelled'):
            continuous.validate(plan, mesh, region, check=cancel)
        continuous.validate(plan, mesh, region)

    def test_share_hands_out_isolated_copies_only_for_the_same_objects(self):
        record, region, mesh, domain = {}, {}, object(), {}
        arrays = (np.arange(6, dtype=np.int32).reshape(2, 3), np.zeros((2, 3), np.int32))
        share = quality._ValidationShare(record, region, arrays)
        first = share.evidence(record, region)
        first[0][:] = -1
        np.testing.assert_array_equal(share.evidence(record, region)[0], arrays[0])
        self.assertIsNone(share.evidence(dict(record), region))
        self.assertIsNone(share.evidence(record, dict(region)))
        allowed = np.array([True, False, True])
        share.remember_allowed(domain, 5., mesh, region, allowed)
        allowed[:] = False
        got = share.allowed(domain, 5., mesh, region)
        np.testing.assert_array_equal(got, [True, False, True])
        got[:] = False
        np.testing.assert_array_equal(share.allowed(domain, 5., mesh, region), [True, False, True])
        for other in ((dict(domain), 5., mesh, region), (domain, 6., mesh, region),
                      (domain, 5., object(), region), (domain, 5., mesh, dict(region))):
            self.assertIsNone(share.allowed(*other))
        labels = np.ones((2, 3), np.int32)
        share.remember_selection(region, mesh, labels)
        copied = share.selection_labels(region, mesh)
        copied[:] = 0
        np.testing.assert_array_equal(share.selection_labels(region, mesh), np.ones((2, 3)))
        self.assertIsNone(share.selection_labels(dict(region), mesh))


if __name__ == '__main__':
    main()
