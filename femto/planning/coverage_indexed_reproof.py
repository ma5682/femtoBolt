"""Opt-in indexed partial reproof; every positive proof is recomputed.

The spatial index and nearest-neighbor ranking select work only. Unreplayed
guide claims are never installed, and interrupted work stays unconfirmed.
Retained older algorithms and their saved policies remain unchanged.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d

from femto.planning.coverage_reproof import ReproofEngine, POLICY as REPROOF_POLICY
from femto.planning.coverage_schedule import POLICY as LAYERED_POLICY
from femto.planning.coverage_face_index import relevant_faces
from femto.planning.planning_revision import digest

POLICY=dict(REPROOF_POLICY,
    schedule='reprove_present_paths_then_current_anchor_prefix_exclusions_and_indexed_current_neighborhoods',
    witnesses='present_current_seed_old_leaf_first_proof_with_parent_closure',
    unbound_guide_witnesses='remain_unconfirmed_continue_current_scan_search',
    face_selection='conservative_seed_grid_then_outward_expanded_face_box',
    face_pair_accounting='charge_each_grid_candidate_before_box_comparison',
    new_scan_order='anchor_faces_then_kdtree_nearest_seed_then_leaf_index',
    candidate_cache='bypass_face_only_cache_when_seed_subset_is_active',
    interrupted_visit='keep_current_rechecked_witness_and_leave_partial_new_search_unconfirmed')


def checked_policy(value):
    if not isinstance(value,dict) or value!=POLICY or digest(value)!=digest(POLICY):
        raise ValueError('索引覆盖复算调度策略无效或版本不一致')
    return value


class IndexedPartialReproofEngine(ReproofEngine):
    def __init__(self,*args,guide):
        super().__init__(*args,guide=guide)
        self.meter.counts['grid_face_pairs']=0

    def _choose_paths(self,confirmed):
        """Unmatched guide witnesses supply no proof and stay unconfirmed."""
        a=self.guide.arrays;wanted=np.zeros(len(a['proof_leaf']),bool);remaining=confirmed.copy()
        for proof in range(len(a['proof_leaf'])):
            if proof%128==0:self.meter.touch()
            seed=int(a['proof_seed'][proof]);leaf=int(a['proof_leaf'][proof])
            if self.seed_remap[seed]<0 or not remaining[leaf]:continue
            remaining[leaf]=False;current=proof
            while current>=0 and not wanted[current]:
                self.meter.touch();wanted[current]=True;current=int(a['proof_parent'][current])
        return np.flatnonzero(wanted)

    def _new_neighborhoods(self,*,anchor_only=False):
        if not self.new_seeds:return
        self.limited_seeds=set(self.new_seeds)
        previous_candidate_ids=self._candidate_ids
        # A face-only cache cannot be shared by restricted and unrestricted
        # seed sets. It could otherwise silently omit valid later candidates.
        self._candidate_ids=lambda face:np.asarray(self.candidates_for(face),np.int64)
        try:
            self.meter.phase='new_anchor_prefix' if anchor_only else 'new_scan_neighborhoods'
            anchor_faces=np.zeros(len(self.domain.faces),bool)
            anchor_faces[[self.seeds[i]['face'] for i in self.new_seeds]]=True
            relevant=set(np.flatnonzero(anchor_faces))
            faces=np.empty(0,np.int64) if anchor_only else np.unique(self.domain.source_faces)
            def charge(count):
                self.meter.touch_many(count,'comparisons');self.meter.counts['grid_face_pairs']+=count
            relevant.update(relevant_faces(self.domain.vertices,self.domain.faces,faces,self.points,self.radius,
                self.candidates_for,charge=charge,check=self.meter.touch))
            roots_by_leaf=np.r_[self.nodes.roots,np.asarray([n.root for n in self.nodes.extra],np.int64)]
            faces_by_leaf=self.domain.source_faces[roots_by_leaf]
            flags=np.zeros(len(self.domain.faces),bool);flags[list(relevant)]=True
            eligible=flags[faces_by_leaf]&~np.isin(self.nodes.root_status[roots_by_leaf],(2,3))
            if self._new_visited:eligible[list(self._new_visited)]=False
            numbers=np.flatnonzero(eligible);roots=roots_by_leaf[numbers]
            centres=(self.domain.vertices[self.domain.faces[self.domain.source_faces[roots]]].mean(axis=1)
                     if self.prism_bounds is None else self.prism_bounds[0][roots]/2+self.prism_bounds[1][roots]/2)
            cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(self.points[list(self.new_seeds)])
            tree=o3d.geometry.KDTreeFlann(cloud);distances=np.empty(len(numbers),float)
            for i,point in enumerate(centres):
                if i%128==0:self.meter.touch()
                count,_,distance=tree.search_knn_vector_3d(point,1)
                distances[i]=distance[0] if count else np.inf
            numbers=numbers[np.lexsort((numbers,distances,~anchor_faces[faces_by_leaf[numbers]]))]
            if anchor_only:numbers=numbers[:POLICY['new_anchor_leaf_budget']]
            for position,value in enumerate(numbers):
                if position%128==0:self.meter.touch()
                number=int(value);self.meter.counts['new_scan_leaves_visited']+=1
                if anchor_only:self.meter.counts['new_anchor_leaves_visited']+=1
                self._visit(number);self._new_visited.add(number)
                if position%LAYERED_POLICY['connectivity_batch']==0:self._drain(LAYERED_POLICY['proof_batch'])
            self._drain()
            if not anchor_only:self._subdivide()
        finally:
            self.limited_seeds=None;self._candidate_ids=previous_candidate_ids

    def cells(self):
        if not self._prepare():return
        self._pixel_bounds()
        self._reprove()
        self._new_neighborhoods(anchor_only=True)
        self._outside_faces();self._outside_pixels();self._new_neighborhoods()
        self.pending.clear()
        for number in range(len(self.nodes)):
            self.meter.phase='remaining_original_cells';self.meter.touch()
            node=self.nodes[number]
            if not node.active or node.status in (1,2,3,6):continue
            self.meter.counts['original_cells_visited']+=1;self._visit(number)
            if number%LAYERED_POLICY['connectivity_batch']==0:self._drain(LAYERED_POLICY['proof_batch'])
        self._drain();self._subdivide()
