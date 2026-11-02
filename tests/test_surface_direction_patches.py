import unittest
import numpy as np
from femto.planning.surface_direction_patches import partition,frame,rows


def folded_grid(angle=75.):
    # Two panels share their seam vertex IDs. Right panel rises by angle.
    vertices=[]
    for y in range(5):
        for x in range(5):
            t=np.deg2rad(angle)
            vertices.append([x if x<=2 else 2+(x-2)*np.cos(t), y,
                             0 if x<=2 else (x-2)*np.sin(t)])
    faces=[]
    for y in range(4):
        for x in range(4):
            i=y*5+x;faces.extend([[i,i+1,i+6],[i,i+6,i+5]])
    return np.asarray(vertices,float),np.asarray(faces,int)


class DirectionPatchTests(unittest.TestCase):
    def test_flat_surface_is_one_patch_with_no_artificial_seam(self):
        v,f=folded_grid(0)
        patches=partition(v,f,np.arange(len(f)))
        self.assertEqual(len(patches),1)
        self.assertEqual(patches[0]['seam_edges'],[])
        self.assertAlmostEqual(frame(v,f,patches[0])['minimum_transverse_metric_factor'],1.)

    def test_steep_panel_is_local_and_shared_seam_is_not_an_outer_edge(self):
        v,f=folded_grid()
        patches=partition(v,f,np.arange(len(f)))
        self.assertEqual(len(patches),2)
        self.assertEqual(sorted(i for p in patches for i in p['face_ids']),list(range(len(f))))
        self.assertEqual(patches[0]['seam_edges'],patches[1]['seam_edges'])
        self.assertEqual(len(patches[0]['seam_edges']),4)
        for p in patches:
            self.assertFalse(set(map(tuple,p['seam_edges'])) & set(map(tuple,p['outer_edges'])))
            local=frame(v,f,p)
            self.assertGreater(local['minimum_transverse_metric_factor'],.999)
            axes=np.array([local[k] for k in ('e1','e2','n')])
            np.testing.assert_allclose(axes@axes.T,np.eye(3),atol=1e-12)

    def test_face_order_does_not_change_partition_and_cancellation_propagates(self):
        v,f=folded_grid()
        self.assertEqual(partition(v,f,np.arange(len(f))),partition(v,f,np.arange(len(f))[::-1]))
        def cancel():raise RuntimeError('cancel')
        with self.assertRaisesRegex(RuntimeError,'cancel'):partition(v,f,np.arange(len(f)),cancel)

    def test_coincident_disconnected_panels_do_not_join(self):
        v,f=folded_grid(0)
        vv=np.concatenate([v,v]);ff=np.concatenate([f,f+len(v)])
        patches=partition(vv,ff,np.arange(len(ff)))
        self.assertEqual(len(patches),2)
        self.assertTrue(all(not p['seam_edges'] for p in patches))

    def test_artificial_seam_is_not_inset_like_real_outer_boundary(self):
        v,f=folded_grid();v*=.01
        patches=partition(v,f,np.arange(len(f)))
        touched=0
        for p in patches:
            decision=rows(v,f,p,5,20,None)
            for is_seam,inset in zip(decision['artificial_seam_sides'],decision['insets_mm']):
                self.assertEqual(inset,0. if is_seam else 2.5)
                touched+=int(is_seam)
            self.assertTrue(decision['positions_m'])
            if len(decision['positions_m'])>1:
                self.assertLessEqual(max(np.diff(decision['positions_m'])),.004+1e-12)
        self.assertGreater(touched,0)

    def test_accumulated_curvature_splits_even_without_a_sharp_neighbor_fold(self):
        angles=np.deg2rad(np.arange(13)*10.)
        v=np.array([[np.sin(a),y,np.cos(a)] for y in (0.,.5,1.) for a in angles])
        f=[]
        for y in range(2):
            for x in range(12):
                i=y*13+x;f.extend([[i,i+1,i+14],[i,i+14,i+13]])
        f=np.array(f)
        patches=partition(v,f,np.arange(len(f)))
        self.assertGreater(len(patches),1)
        self.assertEqual(sorted(i for p in patches for i in p['face_ids']),list(range(len(f))))
