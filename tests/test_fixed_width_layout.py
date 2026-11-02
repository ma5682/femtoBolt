"""Transverse placement checks; these do not certify swept-area coverage."""
import unittest
import numpy as np
from unittest.mock import patch

from femto.planning.fixed_width_layout import metric_positions, row_positions, summary,Layout,request,RESIDUAL_VERSION,REPAIR_PATCH_VERSION,OPTIMIZED_PATCH_VERSION,retained_steps


class FixedWidthPlacementTests(unittest.TestCase):
    def test_budget_fallback_is_visible_in_layout_summary(self):
        record=dict(request=dict(effective_width_mm=5,version=OPTIMIZED_PATCH_VERSION),fragments=[
            dict(row_count=4,actual_plane_step_mm=4,patches=[
                dict(alignment_optimization=dict(strategy='prior_greedy_budget_fallback'))])])
        self.assertIn('1 个分区因计算预算',summary(record))

    def test_malformed_row_shift_is_rejected_as_invalid_input(self):
        spec=request(5,20,version=REPAIR_PATCH_VERSION)
        shift=dict(block_index=0,fragment_face_sha256='a'*64,patch_seed_face=0,position_m=.01,shift_m=.00025)
        for value in ('not a number',None,True,float('nan'),float('inf')):
            with self.subTest(value=value):
                bad=dict(shift,position_m=value)
                with self.assertRaises(ValueError):Layout(spec,{},row_shifts=[bad])

    def test_retained_spacing_exposes_missing_row_without_cross_block_distance(self):
        record=dict(fragments=[dict(row_count=4,block_index=i,first_row_index=i*4,
            origin_m=[0,0,0],transverse_axis=[1,0,0]) for i in range(2)])
        def point(row,x,role='scan'):return dict(row_index=row,xyz_m=[x,0,1],role=role)
        trajectory=dict(ordered_items=[
            dict(kind='scan_segment',block_index=0,points=[point(0,.005),point(2,.021),point(2,.02,'connector'),point(3,.029)]),
            dict(kind='scan_segment',block_index=1,points=[point(4,.1),point(5,.108)])])
        np.testing.assert_allclose(retained_steps(record,trajectory),[16,8,8])

    def test_existing_local_residual_moves_edges_asymmetrically_without_width_change(self):
        class Reference:
            global_residual=.002
            def probe(self,point,half):return False,(.001 if point[0]<.017 else .002),None
        vertices=np.array([[0,0,1],[.034,0,1],[.034,.05,1],[0,.05,1]])
        faces=np.array([[0,1,2],[0,2,3]])
        with patch('femto.planning.quality_constrained.ReferenceEnvelope',return_value=Reference()):
            layout=Layout(request(10,20,version=RESIDUAL_VERSION),{})
        rows=layout.positions(vertices,faces,np.array([0,1]),np.zeros(3),np.array([1.,0,0]),0,0,lambda:None)
        self.assertAlmostEqual(rows[0],.006)
        self.assertAlmostEqual(rows[-1],.027)
        self.assertEqual(layout.fragments[0]['center_insets_mm'],[6.,7.])
        self.assertEqual(layout.spec['effective_width_mm'],10)
        self.assertTrue(np.all(np.diff(rows)<=.008))

    def test_summary_distinguishes_proposals_from_retained_rows(self):
        text=summary(dict(request=dict(effective_width_mm=5),fragments=[
            dict(row_count=10,actual_plane_step_mm=4,minimum_plane_step_mm=2)]),7)
        self.assertIn('候选排布 10 行',text)
        self.assertIn('保留 7 行',text)
        self.assertIn('2.00～4.00',text)

    def test_ideal_rectangle(self):
        rows, report = row_positions(0, .034, 10, 20)
        np.testing.assert_allclose(rows, [.005, .013, .021, .029], atol=1e-14)
        self.assertEqual(report['row_count'], 4)

    def test_narrow_area_never_shrinks_width(self):
        rows, report = row_positions(0, .009, 10, 20)
        self.assertEqual(len(rows), 0)
        self.assertEqual(report['status'], 'narrower_than_fixed_width')
        self.assertEqual(report['nominal_inset_mm'], 5)

    def test_margin_changes_both_edges(self):
        rows, _ = row_positions(0, .034, 10, 20, margin_mm=1)
        self.assertAlmostEqual(rows[0], .006)
        self.assertAlmostEqual(rows[-1], .028)
        self.assertTrue(np.all(np.diff(rows) <= .008))

    def test_local_slope_does_not_shrink_every_row(self):
        # 30 mm flat + 2 mm with slope factor 0.25: metric length = 38 mm.
        rows, step = metric_positions(0, .032, [[0, .032], [.015, .017]], [1, .25], .008)
        self.assertEqual(len(rows), 6)
        self.assertAlmostEqual(step, .038/5)
        def distance(x):
            return x+3*min(max(x-.015, 0), .002)
        np.testing.assert_allclose(np.diff([distance(x) for x in rows]), step)
        self.assertAlmostEqual(rows[0], 0)
        self.assertAlmostEqual(rows[-1], .032)

    def test_triangle_order_does_not_change_positions(self):
        spans = np.array([[0, .04], [.01, .02], [.015, .03]])
        factors = np.array([1, .5, .8])
        rows, step = metric_positions(0, .04, spans, factors, .008)
        other, other_step = metric_positions(0, .04, spans[::-1], factors[::-1], .008)
        np.testing.assert_array_equal(rows, other)
        self.assertEqual(step, other_step)

    def test_absent_surface_interval_rejected(self):
        with self.assertRaisesRegex(ValueError, '无表面'):
            metric_positions(0, .04, [[0, .01], [.02, .04]], [1, 1], .008)

    def test_vertical_face_without_transverse_extent_does_not_explode(self):
        rows, _ = metric_positions(0, .04, [[0, .04], [.02, .02]], [1, 0], .008)
        self.assertEqual(len(rows), 6)


if __name__ == '__main__':
    unittest.main()
