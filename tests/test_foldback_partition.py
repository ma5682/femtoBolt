"""Geometric research checks; the experimental splitter is not a planner version."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from femto.planning import foldback_partition as foldback
from femto.planning.surface_direction_patches import partition
from test_surface_direction_patches import folded_grid


class FoldbackPartitionTests(unittest.TestCase):
    def test_splits_foldback_without_changing_face_ownership_or_winding(self):
        for reverse in (False, True):
            for angle in (0, 70, 80, 160, 180):
                with self.subTest(reverse=reverse, angle=angle):
                    vertices, faces = folded_grid(angle)
                    if reverse:
                        faces[::2] = faces[::2, ::-1]
                    original = partition(vertices, faces, np.arange(len(faces)))
                    before = copy.deepcopy(original)
                    result = foldback.split(vertices, faces, original)
                    self.assertEqual(original, before)
                    self.assertEqual(sorted(i for p in result for i in p['face_ids']), list(range(len(faces))))
                    self.assertEqual(len(result), len(original) if angle < 120 else 2)
                    if angle >= 160:
                        self.assertEqual(len(original), 1)
                        self.assertEqual(result[0]['seam_edges'], result[1]['seam_edges'])
                        self.assertEqual(len(result[0]['seam_edges']), 4)
                    else:
                        self.assertEqual(result, original)
                        result[0]['face_ids'].clear()
                        self.assertEqual(original, before)

    def test_rejects_bad_inputs_and_budget_and_propagates_cancel(self):
        vertices, faces = folded_grid()
        original = partition(vertices, faces, np.arange(len(faces)))
        for mutation in ('duplicate', 'empty', 'bad_seed', 'bad_vertex'):
            with self.subTest(mutation=mutation):
                parts = copy.deepcopy(original)
                mesh = faces.copy()
                if mutation == 'duplicate': parts[0]['face_ids'].append(parts[0]['face_ids'][0])
                if mutation == 'empty': parts[0]['face_ids'] = []
                if mutation == 'bad_seed': parts[0]['seed_face'] = -1
                if mutation == 'bad_vertex': mesh[0, 0] = len(vertices)
                with self.assertRaises(ValueError): foldback.split(vertices, mesh, parts)
        with patch.object(foldback, 'MAX_FACES', 1):
            with self.assertRaisesRegex(ValueError, '预算'): foldback.split(vertices, faces, original)
        def cancel(): raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'): foldback.split(vertices, faces, original, cancel)

    def test_rejects_fold_that_can_be_bypassed_in_same_patch(self):
        vertices = np.array([[-.9208, .0572, -.0813], [-.8753, .2827, .7053],
                             [.1859, -.4798, .6798], [.0190, .0218, .5061]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1]])
        parts = [dict(face_ids=[0, 1, 2], seed_face=0, seam_edges=[], outer_edges=[], area_m2=1.)]
        with self.assertRaisesRegex(ValueError, '绕回'):
            foldback.split(vertices, faces, parts)

    def test_nonmanifold_edges_and_coincident_disconnected_surfaces_do_not_join(self):
        vertices, faces = folded_grid(0)
        vertices = np.concatenate([vertices, vertices])
        faces = np.concatenate([faces, faces + len(vertices)//2])
        original = partition(vertices, faces, np.arange(len(faces)))
        self.assertEqual(len(foldback.split(vertices, faces, original)), 2)
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.]])
        faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
        parts = [dict(face_ids=[0, 1, 2], seed_face=0, seam_edges=[], outer_edges=[], area_m2=1.)]
        result = foldback.split(vertices, faces, parts)
        self.assertEqual(len(result), 3)
        self.assertTrue(all([0, 1] in p['outer_edges'] for p in result))


if __name__ == '__main__':
    unittest.main()
