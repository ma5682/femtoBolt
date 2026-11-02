"""Stage-local ray evidence for missing measured surfaces, without relaxing gates."""
from __future__ import annotations

import numpy as np

STAGES = ('tsdf_extracted', 'topology_cleaned', 'initial_components_filtered',
          'measured_support_filtered', 'final_components_filtered')




class GapProbe:
    def __init__(self, pixels, intrinsic, reference_depth_mm, *, check=lambda: None):
        pixels = np.asarray(pixels)
        k = np.asarray(intrinsic, float)
        depth = np.asarray(reference_depth_mm)
        if (pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.dtype.kind not in 'iu'
                or len(pixels) > 8_388_608 or k.shape != (3, 3) or not np.isfinite(k).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0 or depth.ndim != 2):
            raise ValueError('Invalid gap probe inputs')
        if (np.any(pixels < 0) or np.any(pixels[:, 0] >= depth.shape[1])
                or np.any(pixels[:, 1] >= depth.shape[0])):
            raise ValueError('Gap pixels outside reference image')
        self.pixels = pixels.copy()
        self.depth = depth[pixels[:, 1], pixels[:, 0]].astype(float) / 1000
        self.rays = np.zeros((len(pixels), 6), np.float32)
        self.rays[:, 3:5] = (pixels - [k[0, 2], k[1, 2]]) / [k[0, 0], k[1, 1]]
        self.rays[:, 5] = 1
        self.check, self.depths, self.triangles = check, {}, {}

    def __call__(self, stage, mesh):
        import open3d as o3d
        self.check()
        if len(self.depths) >= len(STAGES) or stage != STAGES[len(self.depths)]:
            raise ValueError('Reconstruction stage order changed')
        self.triangles[stage] = len(mesh.triangles)
        values = np.full(len(self.pixels), np.inf, np.float32)
        if len(mesh.triangles):
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            for start in range(0, len(values), 32768):
                self.check()
                values[start:start+32768] = scene.cast_rays(
                    o3d.core.Tensor(self.rays[start:start+32768]))['t_hit'].numpy()
        self.depths[stage] = values

    def report(self):
        if tuple(self.depths) != STAGES:
            raise ValueError('Incomplete reconstruction stage evidence')
        # For final no-hit pixels, identify the last stage where a hit existed.
        # This also handles a previously occluded back surface becoming visible.
        cause = np.zeros(len(self.pixels), np.uint8)
        for index, stage in enumerate(STAGES[:-1], 1):
            cause[np.isfinite(self.depths[stage])] = index
        final_missing = ~np.isfinite(self.depths[STAGES[-1]])
        cause[~final_missing] = 255
        labels = {0: 'no_hit_at_tsdf_extraction', 1: 'topology_cleanup',
                  2: 'initial_component_filter', 3: 'measured_support_filter',
                  4: 'final_component_filter', 255: 'surface_present_in_replay'}
        return dict(schema='femto-reconstruction-gap-audit-v1',
            scope='fixed_reference_pixel_first_hit_stage_evidence',
            requested_pixel_count=len(cause),
            counts={label: int((cause == code).sum()) for code, label in labels.items()},
            stages=[dict(stage=stage, triangle_count=self.triangles[stage],
                hit_pixels=int(np.isfinite(self.depths[stage]).sum()),
                reference_consistent_pixels=int((np.isfinite(self.depths[stage]) &
                    (np.abs(self.depths[stage]-self.depth) <= .01)).sum())) for stage in STAGES],
            recommended_actions=[dict(action=action,pixel_count=count,guidance=note)
                for action,count,note in (
                    ('inspect_observation_consistency',int((cause==0).sum()),
                     '先核对接纳视角的深度与记录位姿一致性，再比较TSDF尺度；不要仅靠补洞消除缺面。'),
                    ('inspect_same_view_face_support',int((cause==3).sum()),
                     '检查同一视角下三角形三顶点及面心的实测支持；参考像素有深度不足以恢复被拒绝的面。'),
                    ('inspect_component_noise',int(((cause==2)|(cause==4)).sum()),
                     '检查过滤的孤立分量与噪声来源；先区分过滤损失和缺少观测，再决定重建或补采。')) if count],
            interpretation='filter_stage_attribution_not_physical_accuracy_or_root_cause_of_TSDF_loss',
            physical_accuracy_verified=False, execution_ready=False), cause
