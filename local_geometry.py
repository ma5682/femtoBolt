"""Conservative observed-depth support for local TSDF annotation geometry."""
import cv2
import numpy as np
import open3d as o3d
from scene_reconstruction import _read_image, _prepare_rgbd


def filter_small_components(mesh):
    """Reapply the existing component rule after later geometry filters."""
    before=len(mesh.triangles)
    if not before:return dict(triangles_before=0,triangles_after=0,removed_triangles=0,removed_components=0,threshold=0)
    labels,counts,_=mesh.cluster_connected_triangles();counts=np.asarray(counts);labels=np.asarray(labels)
    threshold=max(30,int(counts.max()*.002)) if len(counts) else 30
    keep=counts>=threshold
    mesh.remove_triangles_by_mask(~keep[labels]);mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals()
    return dict(triangles_before=before,triangles_after=int(len(mesh.triangles)),
        removed_triangles=int(before-len(mesh.triangles)),removed_components=int((~keep).sum()),threshold=int(threshold),
        rule='component_triangles >= max(30, 0.2% of largest component)')


def retain_observed_triangles(mesh, dataset, frames, poses, maps, minimum, maximum, tolerance=.01,
                              prepared_depths=None):
    """All corners and centroid must be supported by one source view.

    This is a depth-consistency gate, not a physical-accuracy certificate.
    Missing samples never count as support; no depth interpolation is used.
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    supported = np.zeros(len(triangles), dtype=bool)
    best_worst=np.full(len(triangles),np.inf,float)
    ever_observed=np.zeros(len(triangles),bool)
    jump_touched=np.zeros(len(triangles),bool)
    front_supported=np.zeros(len(triangles),bool)
    grazing_supported=np.zeros(len(triangles),bool)
    mx, my, k, _, _ = maps
    if prepared_depths is not None and len(prepared_depths) != len(frames):
        raise ValueError('Prepared depth count does not match local frames')
    for ordinal,(frame, pose) in enumerate(zip(frames, poses)):
        if prepared_depths is None:
            color = _read_image(dataset/frame['rgb'], cv2.IMREAD_COLOR)
            depth = _read_image(dataset/frame['depth'], cv2.IMREAD_UNCHANGED)
            from local_temporal_support import filtered_depth
            depth=filtered_depth(dataset,frame,depth)
            _, _, _, depth = _prepare_rgbd(color, depth, mx, my, k, 1., minimum, maximum)
        else:
            depth=np.asarray(prepared_depths[ordinal])
        jump=np.zeros(depth.shape,bool)
        horizontal=(depth[:,1:]>0)&(depth[:,:-1]>0)&(np.abs(depth[:,1:].astype(np.int32)-depth[:,:-1].astype(np.int32))>30)
        vertical=(depth[1:]>0)&(depth[:-1]>0)&(np.abs(depth[1:].astype(np.int32)-depth[:-1].astype(np.int32))>30)
        # Mark both endpoints of every discontinuity.  A triangle touching
        # either side of a foreground/background edge belongs in the same
        # diagnostic bucket.
        jump[:,1:]|=horizontal;jump[:,:-1]|=horizontal
        jump[1:]|=vertical;jump[:-1]|=vertical
        inverse = np.linalg.inv(pose)
        for start in range(0, len(triangles), 50000):
            ids = np.arange(start, min(start+50000, len(triangles)))
            corners = vertices[triangles[ids]]
            sample = np.concatenate((corners, corners.mean(axis=1)[:, None, :]), axis=1)
            camera = sample @ inverse[:3, :3].T + inverse[:3, 3]
            z = camera[..., 2]
            u = np.rint(camera[..., 0]*k[0, 0]/np.maximum(z, 1e-9)+k[0, 2]).astype(int)
            v = np.rint(camera[..., 1]*k[1, 1]/np.maximum(z, 1e-9)+k[1, 2]).astype(int)
            valid = (z > 0) & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
            observed = depth[np.clip(v, 0, depth.shape[0]-1), np.clip(u, 0, depth.shape[1]-1)]/1000.
            observed_valid=valid&(observed>0)
            residual=np.abs(observed-z)
            complete=observed_valid.all(axis=1)
            face_camera=camera[:,:3]
            normal=np.cross(face_camera[:,1]-face_camera[:,0],face_camera[:,2]-face_camera[:,0])
            center=face_camera.mean(axis=1)
            cosine=-np.einsum('ij,ij->i',normal,center)/np.maximum(np.linalg.norm(normal,axis=1)*np.linalg.norm(center,axis=1),1e-12)
            front_supported[ids]|=complete&(cosine>0)
            grazing_supported[ids]|=complete&(cosine>0)&(cosine<=.15)
            ever_observed[ids]|=complete
            if np.any(complete):best_worst[ids[complete]]=np.minimum(best_worst[ids[complete]],residual[complete].max(axis=1))
            touched=jump[np.clip(v,0,depth.shape[0]-1),np.clip(u,0,depth.shape[1]-1)]&valid
            jump_touched[ids]|=touched.any(axis=1)
            valid=observed_valid&(residual<=tolerance)
            supported[ids] |= valid.all(axis=1)
    before = len(triangles)
    mesh.remove_triangles_by_mask(~supported)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    removed=~supported
    finite=best_worst[removed&np.isfinite(best_worst)]*1000
    histogram=dict(mm_10_to_15=int(((finite>10)&(finite<=15)).sum()),
        mm_15_to_20=int(((finite>15)&(finite<=20)).sum()),
        mm_20_to_30=int(((finite>20)&(finite<=30)).sum()),mm_over_30=int((finite>30).sum()),
        never_fully_observed=int((removed&~ever_observed).sum()))
    return dict(rule='all_vertices_and_centroid_supported_in_one_raw_view', tolerance_m=tolerance,
                triangles_before=before, triangles_after=int(supported.sum()),
                rejected_best_worst_residual_histogram=histogram,
                depth_jump_edge_triangles=dict(removed=int((removed&jump_touched).sum()),
                                               retained=int((supported&jump_touched).sum()),threshold_mm=30),
                orientation_diagnostics=dict(retained_all_views_backface=int((supported&~front_supported).sum()),
                    retained_grazing_frontface=int((supported&grazing_supported).sum()),frontface_not_used_as_gate=True),
                reused_prepared_depth=prepared_depths is not None,
                warning='Depth consistency only; not independently validated physical accuracy')
