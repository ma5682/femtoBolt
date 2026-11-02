"""Analytic P4 constructions: gaps, seams, folds and exact point provenance."""
from fractions import Fraction as Q
import unittest

import numpy as np

from femto.geometry.surface_support import MeshSupport, NumericalLimit, fraction


def strips(gap=0., *, scale=1.):
    vertices=np.array([[-1,0,0],[0,0,0],[0,1,0],[-1,1,0],
        [gap,0,0],[1,0,0],[1,1,0],[gap,1,0]],float)*scale
    faces=np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7]])
    return MeshSupport(vertices,faces)


class SurfaceSupportTests(unittest.TestCase):
    def test_positive_microgap_has_exact_nonzero_rejected_interval(self):
        gap=2.**-40;mesh=strips(gap)
        a,b=mesh.closest([-.5,.25,0],0),mesh.closest([.5,.25,0],2)
        result=mesh.chord(a,b)
        self.assertEqual(result.status,'rejected')
        self.assertEqual(result.reason,'positive_support_gap')
        self.assertEqual(result.gaps,((Q(1,2),Q(1,2)+Q(gap)),))

    def test_shared_seam_intervals_meet_exactly_without_tolerance(self):
        mesh=strips()
        result=mesh.chord(mesh.closest([-.5,.25,0],0),mesh.closest([.5,.25,0],2))
        self.assertEqual(result.status,'passed')
        self.assertEqual(result.gaps,())
        self.assertTrue(any(hi==Q(1,2) for _,hi,_ in result.intervals))
        self.assertTrue(any(lo==Q(1,2) for lo,_,_ in result.intervals))

    def test_near_degenerate_but_nonzero_face_is_not_erased_by_an_epsilon(self):
        mesh=MeshSupport([[0,0,0],[1,0,0],[0,2.**-600,0]],[[0,1,2]])
        a=mesh.point(0,(Q(1,2),Q(1,4),Q(1,4)))
        b=mesh.point(0,(Q(1,4),Q(1,2),Q(1,4)))
        self.assertEqual(mesh.chord(a,b,candidate_faces=[0]).status,'passed')

    def test_fold_chord_is_rejected_but_constructed_seam_pieces_pass(self):
        mesh=MeshSupport([[-1,0,0],[0,0,0],[0,1,0],[1,0,1],[1,1,1]],
                         [[0,1,2],[1,3,4],[1,4,2]])
        a=mesh.point(0,(Q(1,2),Q(1,4),Q(1,4)))
        b=mesh.point(1,(Q(1,2),Q(1,4),Q(1,4)))
        seam_a=mesh.point(0,(Q(0),Q(3,4),Q(1,4)))
        seam_b=mesh.point(2,(Q(3,4),Q(0),Q(1,4)))
        self.assertEqual(mesh.resolve(seam_a),mesh.resolve(seam_b))
        self.assertEqual(mesh.chord(a,b).status,'rejected')
        self.assertEqual(mesh.chord(a,seam_a).status,'passed')
        self.assertEqual(mesh.chord(seam_b,b).status,'passed')

    def test_zero_length_on_surface_is_valid_not_a_fake_positive_length_scan(self):
        mesh=strips();a=mesh.closest([-.5,.25,0],0)
        self.assertEqual(mesh.chord(a,a).status,'passed')

    def test_vertex_crossing_has_a_complete_exact_union(self):
        mesh=MeshSupport([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],
                         [[0,1,2],[0,2,3],[0,3,4],[0,4,1]])
        a=mesh.point(2,(Q(1,2),Q(1,4),Q(1,4)))
        b=mesh.point(0,(Q(1,2),Q(1,4),Q(1,4)))
        result=mesh.chord(a,b)
        self.assertEqual(result.status,'passed')
        self.assertEqual(result.gaps,())

    def test_extreme_scales_preserve_seams_and_positive_gaps(self):
        for scale in (2.**-400,1.,2.**400):
            for gap in (0.,2.**-40):
                with self.subTest(scale=scale,gap=gap):
                    mesh=strips(gap,scale=scale)
                    a=mesh.closest(np.array([-.5,.25,0])*scale,0)
                    b=mesh.closest(np.array([.5,.25,0])*scale,2)
                    self.assertEqual(mesh.chord(a,b).status,'passed' if gap==0 else 'rejected')

    def test_constructed_thirds_resolve_exactly_despite_rounded_display(self):
        mesh=MeshSupport([[0,0,0],[1,0,1],[0,1,1]],[[0,1,2]])
        point=mesh.point(0,(Q(1,3),Q(1,3),Q(1,3)))
        self.assertEqual(mesh.resolve(point),(Q(1,3),Q(1,3),Q(2,3)))
        display,error=mesh.display(point)
        self.assertGreater(error,0)
        exact=mesh.resolve(point)
        self.assertGreaterEqual(Q(error),sum(abs(x-Q(y)) for x,y in zip(exact,display)))
        self.assertEqual(mesh.chord(point,point).status,'passed')

    def test_shared_edge_record_preserves_source_ids_and_exact_parameter(self):
        mesh=MeshSupport([[0,0,0],[1,0,0],[0,1,0],[0,-1,1]],[[0,1,2],[1,0,3]])
        left=mesh.point(0,(Q(2,3),Q(1,3),Q(0)))
        right=mesh.point(1,(Q(1,3),Q(2,3),Q(0)))
        self.assertEqual(left['edge'],[0,1]);self.assertEqual(left['edge_parameter'],[1,3])
        self.assertEqual(mesh.resolve(left),mesh.resolve(right))

    def test_projection_is_an_explicit_new_candidate_not_a_proof_for_old_xyz(self):
        mesh=strips();record=mesh.closest([-.5,.25,.000001],0)
        self.assertEqual(mesh.resolve(record)[2],0)
        with self.assertRaisesRegex(ValueError,'missing_constructed_point'):
            mesh.chord({'xyz_m':[-.5,.25,.000001]},record)

    def test_point_metadata_cannot_forge_edge_or_barycentric_provenance(self):
        mesh=strips();a=mesh.point(0,(Q(0),Q(1,2),Q(1,2)))
        changed=dict(a,edge=[0,7])
        with self.assertRaisesRegex(ValueError,'inconsistent'):mesh.resolve(changed)
        changed=dict(a,barycentric=[[1,2],[1,2],[1,2]])
        with self.assertRaisesRegex(ValueError,'outside'):mesh.resolve(changed)

    def test_noncanonical_and_unbounded_rationals_are_not_silently_rounded(self):
        for value in ([2,4],[1,0],[True,1],[1<<4097,1]):
            with self.subTest(value=str(value)[:30]),self.assertRaises(NumericalLimit):fraction(value)

    def test_zero_budget_returns_unconfirmed_for_otherwise_valid_construction(self):
        mesh=strips();mesh.max_face_checks=0
        a=mesh.closest([-.5,.25,0],0)
        result=mesh.chord(a,a)
        self.assertEqual(result.status,'unconfirmed');self.assertEqual(result.reason,'support_face_budget')

    def test_cancellation_is_not_downgraded_to_a_numeric_unknown(self):
        mesh=strips();a=mesh.closest([-.5,.25,0],0)
        def stop():raise RuntimeError('cancelled')
        mesh.check=stop
        with self.assertRaisesRegex(RuntimeError,'cancelled'):mesh.chord(a,a)

    def test_disabled_faces_cannot_complete_a_gap(self):
        mesh=strips();a=mesh.closest([-.5,.25,0],0);b=mesh.closest([.5,.25,0],2)
        mesh.allowed[2:]=False
        self.assertEqual(mesh.chord(a,b).status,'rejected')

    def test_degenerate_input_does_not_create_a_fake_surface_point(self):
        mesh=MeshSupport([[0,0,0],[1,0,0],[2,0,0]],[[0,1,2]])
        with self.assertRaisesRegex(ValueError,'degenerate'):mesh.closest([1,0,0],0)
