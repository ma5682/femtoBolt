import unittest
import numpy as np
from square_brush import stroke, mask_polygon, polygon_mask


class SquareBrushTests(unittest.TestCase):
    def test_holes_components_and_thin_pixels_roundtrip(self):
        mask=np.zeros((80,100),np.uint8)
        mask[5:45,5:45]=1;mask[15:35,15:35]=0
        mask[55:70,60:90]=1;mask[70,90]=1;mask[75,95]=1
        polygon=mask_polygon(mask)
        np.testing.assert_array_equal(polygon_mask(polygon,(100,80)),mask)

    def test_random_masks_roundtrip(self):
        rng=np.random.default_rng(42)
        for _ in range(20):
            mask=(rng.random((8,8))>.5).astype(np.uint8)
            np.testing.assert_array_equal(polygon_mask(mask_polygon(mask),(8,8)),mask)

    def test_fast_drag_and_erase_do_not_leave_gaps(self):
        mask=np.zeros((120,120),np.uint8)
        stroke(mask,(10,10),(100,100),12)
        self.assertTrue(all(mask[i,i] for i in range(10,101)))
        stroke(mask,(50,50),(70,70),8,erase=True)
        self.assertTrue(mask[20,20]);self.assertFalse(mask[60,60])

    def test_clips_at_image_and_empty_mask(self):
        mask=np.zeros((20,20),np.uint8)
        stroke(mask,(-10,0),(5,0),10)
        self.assertTrue(mask[0,0]);self.assertFalse(mask[19,19])
        stroke(mask,(0,0),(20,20),100,erase=True)
        self.assertEqual(mask_polygon(mask),[])

    def test_complexity_is_rejected_without_changing_mask(self):
        mask=np.indices((50,50)).sum(axis=0)%2
        before=mask.copy()
        with self.assertRaises(ValueError):mask_polygon(mask)
        np.testing.assert_array_equal(mask,before)

    def test_long_diagonal_stroke_fits_polygon_budget(self):
        mask=np.zeros((1000,1000),np.uint8)
        stroke(mask,(30,30),(900,800),40)
        polygon=mask_polygon(mask)
        self.assertLessEqual(len(polygon),512)
        from fixed_rgb_geometry import polygon_contains
        self.assertTrue(polygon_contains(np.linspace([30,30],[900,800],100),polygon).all())
