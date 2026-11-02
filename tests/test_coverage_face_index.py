"""Grid face proposals must match a full outward-box search."""
import unittest
import numpy as np

from femto.planning.coverage_face_index import relevant_faces
from femto.planning.coverage_certificate import _Engine,CoverageBudget
from femto.planning.coverage_domain import CoverageDomain
from femto.planning.footprint_model import nominal


class CoverageFaceIndexTests(unittest.TestCase):
    def test_existing_grid_matches_full_search_on_random_surface_and_seed_subsets(self):
        rng=np.random.default_rng(107)
        origins=rng.uniform(-.08,.08,(80,3))
        vertices=np.concatenate([origins,origins+[.008,0,0],origins+[0,.008,0]])
        faces=np.array([[i,i+80,i+160] for i in range(80)],dtype='<i8')
        domain=CoverageDomain.mesh(vertices,faces)
        records=[dict(source_face=i,xyz_m=domain.vertices[domain.faces[i,0]].tolist(),role='scan') for i in range(80)]
        engine=_Engine(domain,records,nominal(20),CoverageBudget(),lambda:None,None);engine.seed()
        for ids in (set(range(len(engine.points))),set(range(0,len(engine.points),2))):
            def indexed(face):return tuple(i for i in engine.candidates_for(face) if i in ids)
            charges=[]
            got=relevant_faces(domain.vertices,domain.faces,domain.source_faces,engine.points,engine.radius,
                               indexed,charge=charges.append)
            expected=[]
            for face in domain.source_faces:
                tri=domain.vertices[domain.faces[face]]
                low=np.nextafter(tri.min(axis=0)-engine.radius,-np.inf)
                high=np.nextafter(tri.max(axis=0)+engine.radius,np.inf)
                if any(np.all(engine.points[i]>=low) and np.all(engine.points[i]<=high) for i in ids):expected.append(int(face))
            self.assertEqual(got.tolist(),expected)
            self.assertEqual(len(charges),len(domain.source_faces))
            self.assertLessEqual(sum(charges),len(domain.source_faces)*len(ids))

    def test_boundary_inclusion_and_empty_seed_set(self):
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces=np.array([[0,1,2]])
        points=np.array([[1.1,0.,0.],[-.1,0.,0.],[2.,2.,0.]])
        self.assertEqual(relevant_faces(vertices,faces,np.array([0]),points,.1,lambda face:(0,1,2)).tolist(),[0])
        self.assertEqual(relevant_faces(vertices,faces,np.array([0]),np.empty((0,3)),.1,lambda face:()).tolist(),[])

    def test_bad_indices_or_cancellation_fail_closed(self):
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces=np.array([[0,1,2]]);points=np.array([[0.,0.,0.]])
        with self.assertRaises(ValueError):relevant_faces(vertices,faces,np.array([0]),points,.1,lambda face:(2,))
        with self.assertRaises(ValueError):relevant_faces(vertices,faces,np.array([0]),points,.1,lambda face:(.5,))
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            relevant_faces(vertices,faces,np.array([0]),points,.1,lambda face:(0,),check=cancel)
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            relevant_faces(vertices,faces,np.array([0]),points,.1,lambda face:(0,),charge=lambda n:cancel())
