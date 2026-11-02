"""Planning-only split retains original faces and actual shared-edge topology."""
from collections import Counter,defaultdict,deque
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from test_mesh_slice_planner import grid_mesh
from femto.planning import spatial_patch_split,surface_direction_patches


class SpatialPatchSplitTests(unittest.TestCase):
    def setUp(self):
        mesh=grid_mesh(width=.08,height=.04,cell=.01)
        self.vertices=np.asarray(mesh.vertices);self.faces=np.asarray(mesh.faces)
        parts=surface_direction_patches.partition(self.vertices,self.faces,np.arange(len(self.faces)))
        self.assertEqual(len(parts),1);self.part=parts[0]

    def test_preserves_faces_area_edges_and_connected_pieces(self):
        source=deepcopy(self.part);vertices=self.vertices.copy();faces=self.faces.copy()
        parts=spatial_patch_split.split(vertices,faces,self.part,[1.,0.,0.],2)
        self.assertGreater(len(parts),1)
        self.assertEqual(sorted(i for p in parts for i in p['face_ids']),source['face_ids'])
        self.assertAlmostEqual(sum(p['area_m2'] for p in parts),source['area_m2'])
        self.assertEqual({tuple(e) for p in parts for e in p['outer_edges']},set(map(tuple,source['outer_edges'])))
        seams=Counter(tuple(e) for p in parts for e in p['seam_edges'])
        self.assertTrue(seams);self.assertTrue(all(count==2 for count in seams.values()))
        for part in parts:
            attached=defaultdict(list)
            for fid in part['face_ids']:
                face=faces[fid]
                for a,b in zip(face,np.roll(face,-1)):attached[tuple(sorted((int(a),int(b))))].append(fid)
            neighbors=defaultdict(set)
            for members in attached.values():
                if len(members)==2:
                    a,b=members;neighbors[a].add(b);neighbors[b].add(a)
            seen={part['face_ids'][0]};queue=deque(seen)
            while queue:
                for nxt in neighbors[queue.popleft()]-seen:seen.add(nxt);queue.append(nxt)
            self.assertEqual(seen,set(part['face_ids']))
        self.assertEqual(self.part,source)
        self.assertTrue(np.array_equal(vertices,self.vertices));self.assertTrue(np.array_equal(faces,self.faces))
        self.assertEqual(parts,spatial_patch_split.split(vertices,faces,self.part,[1.,0.,0.],2))

    def test_nonmanifold_outer_edge_never_becomes_adjacency(self):
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.]])
        faces=np.array([[0,1,2],[1,0,3],[0,1,4]])
        outer=sorted({tuple(sorted((int(a),int(b)))) for face in faces for a,b in zip(face,np.roll(face,-1))})
        part=dict(face_ids=[0,1,2],seed_face=0,seam_edges=[],outer_edges=[list(e) for e in outer])
        result=spatial_patch_split.split(vertices,faces,part,[1.,0.,0.],2)
        self.assertEqual(len(result),3)
        self.assertTrue(all([0,1] in p['outer_edges'] and [0,1] not in p['seam_edges'] for p in result))

    def test_invalid_edges_limits_and_cancel_fail_closed(self):
        bad=deepcopy(self.part);bad['face_ids'].append(bad['face_ids'][0])
        with self.assertRaises(ValueError):spatial_patch_split.split(self.vertices,self.faces,bad,[1,0,0],2)
        bad=deepcopy(self.part);bad['outer_edges'].pop()
        with self.assertRaises(ValueError):spatial_patch_split.split(self.vertices,self.faces,bad,[1,0,0],2)
        for axis,bins in (([0,0,0],2),([1,0,0],1),([1,0,0],True)):
            with self.subTest(axis=axis,bins=bins),self.assertRaises(ValueError):
                spatial_patch_split.split(self.vertices,self.faces,self.part,axis,bins)
        with patch.object(spatial_patch_split,'MAX_PIECES',0),self.assertRaisesRegex(ValueError,'超过预算'):
            spatial_patch_split.split(self.vertices,self.faces,self.part,[1,0,0],2)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            spatial_patch_split.split(self.vertices,self.faces,self.part,[1,0,0],2,cancel)
