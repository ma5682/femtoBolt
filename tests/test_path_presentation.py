import unittest
import numpy as np
from femto.planning.path_presentation import render_diagram


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.full((120,240,3),100,np.uint8)
        self.ids=np.zeros((120,240),int)
        self.region={'triangle_indices':[0]}
        self.plan={'row_step_px':10,'segments':[
            [{'pixel':[20,30],'xyz_m':[0,0,.5]}, {'pixel':[80,30],'xyz_m':[.06,0,.5]}],
            [{'pixel':[160,80],'xyz_m':[.16,.08,.5]}, {'pixel':[220,80],'xyz_m':[.22,.08,.5]}]]}

    def render(self,**kwargs):return np.asarray(render_diagram(self.rgb,self.ids,self.region,self.plan,**kwargs))

    def test_markers_no_segment_connector(self):
        image=self.render()
        np.testing.assert_array_equal(image[155,20],[53,234,135])
        np.testing.assert_array_equal(image[155,80],[255,87,101])
        np.testing.assert_array_equal(image[180,120],[100,100,100])

    def test_visibility_and_single_segment(self):
        image=self.render(show_path=False)
        np.testing.assert_array_equal(image[125:],self.rgb)
        image=self.render(show_region=False,show_path=False)
        self.assertTrue((image[125:]==24).all())
        image=self.render(segment_index=1)
        np.testing.assert_array_equal(image[155,20],[100,100,100])
        np.testing.assert_array_equal(image[205,160],[53,234,135])


if __name__=='__main__':unittest.main()
