"""One admitted RGB-D view: calibrated local pixel surface, without TSDF."""
from types import SimpleNamespace

from femto.geometry.rgbd_frames import prepare_rgbd_arrays
from femto.geometry.single_frame_mesh import build_pixel_mesh_and_cloud
from femto.geometry.fusion import write_point_cloud_safe
from femto.operation_diagnostics import diagnosed


@diagnosed('reconstruction.pixel_surface')
def prepare_surface(color, depth, maps, pose, minimum, maximum, region_filter, frame_id, output,
                    *, prepared_depth=None):
    mx,my,k=maps
    if prepared_depth is None:
        bgr, measured, k = prepare_rgbd_arrays(color,depth,mx,my,k,1.,minimum,maximum,
            depth_filter=lambda d:region_filter.filter(d,pose,k,frame_id))
    else:
        import cv2
        bgr=cv2.remap(color,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        measured=prepared_depth
    # Pixels have already been undistorted once by prepare_rgbd_arrays. Zero
    # distortion here preserves exactly those calibrated rays and RGB pixels.
    intr=SimpleNamespace(width=measured.shape[1],height=measured.shape[0],
        fx=k[0,0],fy=k[1,1],cx=k[0,2],cy=k[1,2])
    distortion=dict(model='BROWN_CONRADY_K6',**{key:0. for key in ('k1','k2','p1','p2','k3','k4','k5','k6')})
    mesh,_,report,cloud=build_pixel_mesh_and_cloud(measured,bgr,intr,distortion,pose,
        depth_min_mm=minimum*1000,depth_max_mm=maximum*1000)
    report.update(input_pixel_frame='undistorted_reference_rgb',tsdf_integrations=0)
    initial=cloud.select_by_index(list(range(0,len(cloud.points),16)))
    if not write_point_cloud_safe(output/'initial_observed_points_m.ply',initial):
        raise RuntimeError('Initial observed cloud write failed')
    return cloud,mesh,initial,measured,report
