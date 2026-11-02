import unittest,tempfile,json,csv,struct
from pathlib import Path
import numpy as np
from tests import test_serpentine_path as fixtures
from femto.planning.serpentine_path import serpentine_plan
from femto.planning.region_path import export_path,load_path
from femto.planning.route_motion import MotionPlayer,statistics
from femto.planning.path_presentation import render_diagram

class RouteMotionTests(unittest.TestCase):
    def test_eight_orders_roundtrip_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mesh,store,region,projection=fixtures.SerpentineTests().setup_surface(root)
            for direction in ('horizontal','vertical'):
                for corner in ('top_left','top_right','bottom_left','bottom_right'):
                    with self.subTest(direction=direction,corner=corner):
                        plan=serpentine_plan(projection,mesh,region,8,20,direction=direction,start_corner=corner)
                        self.assertEqual(len(plan['segments']),1)
                        chain=plan['segments'][0];first=chain[0]['pixel']
                        self.assertEqual(first[0]>40,'right' in corner);self.assertEqual(first[1]>40,'bottom' in corner)
                        axis=0 if direction=='horizontal' else 1;signs=[];row=None
                        for a,b in zip(chain,chain[1:]):
                            if b['motion_kind']=='scan':
                                self.assertEqual(a['pixel'][1-axis],b['pixel'][1-axis])
                                if row!=b['pixel'][1-axis]:
                                    row=b['pixel'][1-axis];signs.append(np.sign(b['pixel'][axis]-a['pixel'][axis]))
                        expected=-1 if ('right' in corner if axis==0 else 'bottom' in corner) else 1
                        self.assertEqual(signs,[expected*(-1)**i for i in range(len(signs))])
                        folder=export_path(plan,store,root/'exports',region)
                        self.assertEqual(load_path(folder/'path.json',store,projection,region)['segments'],chain and plan['segments'])
                        rows=list(csv.DictReader((folder/'path.csv').read_text(encoding='utf-8-sig').splitlines()))
                        self.assertEqual(len(rows),len(chain))
                        self.assertAlmostEqual(float(rows[-1]['route_distance_m']),statistics(plan)['total_length_m'])
                        glb=(folder/'region_with_path.glb').read_bytes();length=struct.unpack_from('<I',glb,12)[0]
                        document=json.loads(glb[20:20+length]);extras=document['scenes'][0]['extras']
                        self.assertFalse(extras['animation_included']);self.assertEqual(extras['route_order'],plan['route_order'])
                        self.assertEqual(extras['chains'][0]['edge_motion_kinds'],[s['motion_kind'] for s in chain[1:]])
                        image=render_diagram(np.zeros((85,85,3),np.uint8),projection.ids,region,plan)
                        self.assertEqual(image.height,85+265)
                        self.assertTrue(plan['start_selection']['adjusted'])
    def test_version_binding_protects_route_semantics(self):
        import copy
        from femto.planning.planning_state import validate
        with tempfile.TemporaryDirectory() as tmp:
            mesh,store,region,projection=fixtures.SerpentineTests().setup_surface(Path(tmp))
            plan=serpentine_plan(projection,mesh,region,8,20,direction='vertical',start_corner='bottom_right')
            for key in ('start_selection','chain_breaks','motion_demo','route_order'):
                changed=copy.deepcopy(plan);changed[key]={'tampered':True}
                with self.subTest(key=key),self.assertRaises(ValueError):validate(changed,mesh,region)
            region=dict(region,name='metadata only',note='keeps path');validate(plan,mesh,region)

    def test_obstacle_and_break(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh,store,region,projection=fixtures.SerpentineTests().setup_surface(Path(tmp))
            region=dict(region,exclusion_polygons=[[[30,30],[45,30],[45,45],[30,45]]])
            for direction in ('horizontal','vertical'):
                plan=serpentine_plan(projection,mesh,region,4,20,boundary_margin_mm=0,direction=direction,start_corner='bottom_right')
                self.assertEqual(len(plan['segments']),1)
                for p in plan['segments'][0]:self.assertFalse(30<=p['pixel'][0]<=45 and 30<=p['pixel'][1]<=45)
            region['exclusion_polygons']=[[[30,0],[45,0],[45,80],[30,80]]]
            plan=serpentine_plan(projection,mesh,region,4,20,boundary_margin_mm=0,direction='horizontal',start_corner='top_left')
            self.assertEqual(len(plan['chain_breaks']),len(plan['segments'])-1)
            self.assertIsNone(statistics(plan)['complete_route_time_s'])
            for gap in plan['chain_breaks']:self.assertFalse(gap['transition_planned'])
    def test_playback_scrub_and_explicit_break(self):
        def p(x,y,kind='scan'):return dict(xyz_m=[x,y,0],pixel=[x*100,y*100],motion_kind=kind)
        plan=dict(segments=[[p(0,0),p(.02,0),p(.02,.02,'connector')],[p(.1,.1),p(.12,.1)]])
        player=MotionPlayer(plan);stats=statistics(plan)
        self.assertAlmostEqual(stats['scan_length_m'],.04);self.assertAlmostEqual(stats['connector_length_m'],.02)
        player.seek(.5);pos=player.position()
        np.testing.assert_allclose(pos['xyz_m'],[.02,.01,0]);self.assertEqual(pos['motion_kind'],'connector')
        np.testing.assert_allclose(pos['direction_W0'],[0,1,0]);self.assertFalse(player.running)
        player.play();player.advance(1000);self.assertTrue(player.at_break);self.assertEqual(player.chain,0)
        player.advance(1000);self.assertEqual(player.chain,0)
        self.assertTrue(player.play());self.assertEqual(player.chain,1);self.assertEqual(player.distance,0)
        player.pause();player.set_speed(40);player.select_chain(0);player.play();player.advance(.25)
        self.assertAlmostEqual(player.distance,.01);player.pause();player.advance(10);self.assertAlmostEqual(player.distance,.01)
        player.reset();self.assertEqual(player.position()['progress'],0)

class MotionWindowTests(unittest.TestCase):
    """Placeholder for the retired motion-window tests.

    Motion playback is now covered by the backend ``MotionPlayer`` tests
    above and the Qt presentation tests.  Keep the class name for historical
    test selection compatibility while making the module valid Python after
    removing the obsolete widget fixture.
    """

    pass


if __name__ == '__main__':
    unittest.main()
