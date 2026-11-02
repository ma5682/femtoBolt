"""Read-only, reference-aligned rendering of acquired geometry in metres."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=4)
def hud_font(size=13):
    for name in ('C:/Windows/Fonts/msyh.ttc', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def grid_interval(span):
    """Readable metric interval, shared by the grid and the scale legend."""
    raw=max(float(span)/8, .00001)
    decade=10.**np.floor(np.log10(raw))
    return next(value*decade for value in (1,2,5,10) if value*decade >= raw)


def model_framing(points):
    """Fit the main observed extent without changing or filtering geometry."""
    points = np.asarray(points)
    if len(points) >= 100:
        low, high = np.percentile(points, [1, 99], axis=0)
        # Leave room around the robust extent for the original outer edges.
        span = max(float(np.max(high[:2]-low[:2])) * 1.17, .01)
    else:
        low, high = points.min(axis=0), points.max(axis=0)
        span = max(float(np.max(high[:2]-low[:2])), .01)
    return (low+high)*.5, span


def current_camera_pose(record, *, active, finishing, state, now):
    from femto.dataset.scene_dataset_validator import validate_pose
    if not active or finishing or state != 'tracking' or not record:
        return None
    age=now-record.get('received_monotonic', -1e12)
    if record.get('state') != 'tracking' or not 0 <= age <= 2:
        return None
    pose=record.get('T_WC')
    if pose is None or validate_pose(pose):
        return None
    return np.asarray(pose, dtype=float)


def read_review_surface(result):
    """Return bounded *original* triangles for display; never fill holes."""
    from femto.geometry.fusion import read_geometry_safe
    path=Path(result)/'reconstruction'/'scene_tsdf_mesh_m.ply'
    if not path.is_file() or path.stat().st_size > 128*1024*1024:
        return None
    mesh=read_geometry_safe(path, mesh=True)
    vertices=np.asarray(mesh.vertices)
    faces=np.asarray(mesh.triangles)
    if not len(faces) or not np.isfinite(vertices).all():
        return None
    if faces.min()<0 or faces.max()>=len(vertices):
        raise ValueError('Invalid mesh vertex index')
    # Modest models render every original face. Very large models use an
    # explicitly labelled edge sample, never an invented surface substitute.
    stride=1 if len(faces)<=120000 else (len(faces)+11999)//12000
    selected_faces = faces[::stride]
    vertex_colors = np.asarray(getattr(mesh, 'vertex_colors', []))
    face_colors = None
    if vertex_colors.shape == vertices.shape and np.isfinite(vertex_colors).all():
        face_colors = np.clip(vertex_colors[selected_faces].mean(axis=1), 0., 1.).copy()
    return dict(triangles=vertices[selected_faces].copy(), colors=face_colors,
                sampled=stride>1, total=len(faces))


def sample_aligned_frame_cloud(depth_mm, color_bgr, intrinsics, distortion,
                               minimum_mm, maximum_mm, *, max_points=35000):
    """Bounded K6-corrected RGB-D preview in camera frame C; never fusion input."""
    depth = np.asarray(depth_mm)
    color = np.asarray(color_bgr)
    if (depth.ndim != 2 or color.shape != (*depth.shape, 3)
            or depth.shape != (intrinsics.height, intrinsics.width)):
        raise ValueError('实时点云要求 RGB、深度与标定尺寸一致')
    if 'BROWN_CONRADY_K6' not in str(distortion.get('model', '')):
        raise ValueError('实时点云缺少 RGB K6 畸变标定')
    if (not np.isfinite([intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                         minimum_mm, maximum_mm]).all() or intrinsics.fx <= 0
            or intrinsics.fy <= 0 or minimum_mm <= 0 or maximum_mm <= minimum_mm):
        raise ValueError('实时点云标定或深度范围无效')
    coefficients = np.asarray([distortion[key] for key in
        ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')], dtype=np.float64)
    if not np.isfinite(coefficients).all():
        raise ValueError('实时点云畸变参数无效')
    stride = max(1, int(np.ceil(np.sqrt(depth.size / max_points))))
    while (((depth.shape[0] + stride - 1) // stride)
           * ((depth.shape[1] + stride - 1) // stride)) > max_points:
        stride += 1
    sampled = depth[::stride, ::stride]
    valid = np.isfinite(sampled) & (sampled >= minimum_mm) & (sampled <= maximum_mm)
    rows, columns = np.nonzero(valid)
    if not len(rows):
        return np.empty((0, 3)), np.empty((0, 3))
    rows *= stride
    columns *= stride
    pixels = np.column_stack((columns, rows)).astype(np.float64).reshape(-1, 1, 2)
    matrix = np.array([[intrinsics.fx, 0, intrinsics.cx],
                       [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]], dtype=np.float64)
    rays = cv2.undistortPoints(pixels, matrix, coefficients).reshape(-1, 2)
    metres = depth[rows, columns].astype(np.float64) / 1000
    points = np.column_stack((rays * metres[:, None], metres))
    finite = np.isfinite(points).all(axis=1)
    colors = color[rows, columns, ::-1].astype(np.float64) / 255
    return points[finite], colors[finite]


def camera_oblique_points(points):
    """Rotate only display coordinates so depth is visible in the single-frame preview."""
    yaw, pitch = np.deg2rad((25., 18.))
    x, y, z = np.asarray(points).T
    side = np.cos(yaw) * x + np.sin(yaw) * z
    depth = -np.sin(yaw) * x + np.cos(yaw) * z
    return np.column_stack((side, np.cos(pitch) * y - np.sin(pitch) * depth,
                            np.sin(pitch) * y + np.cos(pitch) * depth))


def render_scene(width, height, points=None, colors=None, *, center=None, span=1., zoom=1.,
                 pan=(0.,0.), surface=None, camera=None, highlight=None, reference=False,
                 live_frame=False):
    """Render the fixed XYZ front/orthographic view used by model inspection.

    ``pan`` remains in the signature for callers that persisted the old view
    state, but the inspection viewport is intentionally centered and cannot be
    translated.  This keeps the model plane aligned to the reference frame;
    only ``zoom`` changes the display scale.
    """
    width,height=max(100,int(width)),max(100,int(height))
    image=Image.new('RGB',(width,height),'#080d13')
    draw=ImageDraw.Draw(image)
    populated=points is not None and len(points)>0 and center is not None
    if not populated:
        for x in range(0,width,44):draw.line((x,0,x,height),fill='#101b28')
        for y in range(0,height,44):draw.line((0,y,width,y),fill='#101b28')
        draw.text((24,24),'实时点云 · 等待有效深度' if live_frame else '三维视口 · 等待采集数据',font=hud_font(17),fill='#c7d7e4')
        draw.text((24,58),'请检查相机连接、对齐深度与有效距离。' if live_frame else '建立参考位置后显示真实采样；已有模型可从历史记录打开。',font=hud_font(),fill='#8295a8')
        return image
    center=np.asarray(center)
    scale=min(width,height)*.82*zoom/max(float(span),.01)
    # The inspection view is locked to the reference XYZ front projection.
    # Keep accepting ``pan`` for compatibility with older callers, but never
    # apply it: model placement must not drift after a right-button drag.
    offset=np.array([width/2,height/2])
    def project(p):return (np.asarray(p)[...,:2]-center[:2])*scale+offset
    interval=grid_interval(span/zoom)
    for axis,extent in ((0,width),(1,height)):
        lower=center[axis]-offset[axis]/scale
        upper=lower+extent/scale
        for value in np.arange(np.ceil(lower/interval)*interval,upper+interval/2,interval):
            pixel=(value-center[axis])*scale+offset[axis]
            line=(pixel,0,pixel,height) if axis==0 else (0,pixel,width,pixel)
            draw.line(line, fill='#142433')
    p=np.asarray(points)
    xy=project(p)
    cols=(np.clip(np.asarray(colors)*255,0,255).astype(np.uint8) if colors is not None and len(colors)==len(p)
          else np.tile([138,179,195],(len(p),1)))
    if highlight is not None and len(highlight)==len(p):
        cols[np.asarray(highlight,dtype=bool)]=[6,214,160]
    for i in np.argsort(p[:,2])[::-1]:
        x,y=xy[i]
        if np.isfinite(x+y) and 0<=x<width and 0<=y<height:
            draw.ellipse((x-1,y-1,x+1,y+1),fill=tuple(cols[i]))
    if surface is not None:
        triangles=surface['triangles']
        projected=project(triangles)
        normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        lights=.25+.65*np.abs(normals[:,2])/np.maximum(np.linalg.norm(normals,axis=1),1e-12)
        visible=(np.isfinite(projected).all(axis=(1,2)) &
                 (projected[:,:,0].max(axis=1)>=0) & (projected[:,:,0].min(axis=1)<=width) &
                 (projected[:,:,1].max(axis=1)>=0) & (projected[:,:,1].min(axis=1)<=height))
        polygons=projected.tolist()
        face_colors = surface.get('colors')
        if (face_colors is not None and np.shape(face_colors) == (len(triangles), 3)
                and np.isfinite(face_colors).all()):
            # Keep the saved RGB appearance; mild lighting reveals shape
            # without replacing it with the geometry-only fallback tint.
            shading = .8 + .2*np.abs(normals[:,2])/np.maximum(np.linalg.norm(normals,axis=1),1e-12)
            shades = (np.clip(face_colors,0.,1.)*255*shading[:,None]).astype(np.uint8).tolist()
        else:
            shades=(lights[:,None]*np.array([130,169,190])).astype(np.uint8).tolist()
        # Sampled meshes show original edges only. Filling sampled faces could
        # make omissions from the display budget look like actual mesh holes.
        for i in np.argsort(triangles[:,:,2].mean(axis=1))[::-1]:
            if not visible[i]:continue
            polygon=[tuple(point) for point in polygons[i]]
            if surface['sampled']:
                draw.line(polygon+[polygon[0]],fill='#315666',width=1)
            else:
                draw.polygon(polygon,fill=tuple(shades[i]))
    # This is a projected reference marker, not a tool/focal-point location.
    if reference:
        x,y=project([0.,0.,0.])
        if 20<x<width-100 and 60<y<height-50:
            draw.line((x-7,y,x+7,y),fill='#55b3ee',width=1)
            draw.line((x,y-7,x,y+7),fill='#55b3ee',width=1)
        draw.text((max(24,width-240),height-51),'十字：参考原点投影',font=hud_font(11),fill='#8dabbf')
    if camera is not None:
        x,y=project(camera[:3,3])
        if 10<x<width-10 and 40<y<height-30:
            draw.rectangle((x-7,y-6,x+7,y+6),outline='#268bda',width=2)
            draw.text((min(x+12,width-140),max(42,y-22)),'相机位置 · 投影',font=hud_font(11),fill='#75b8ea')
    draw.rectangle((12,12,460 if live_frame else 310,38),fill='#111d29')
    label='实时单帧点云 · 相机坐标 C · 斜视投影' if live_frame else (
        '表面网格' if surface and not surface['sampled'] else '网格边线抽样 + 点云' if surface else '已采点云') + ' · XYZ 固定正视 / 正交投影'
    draw.text((22,17),label,font=hud_font(12),fill='#d0deea')
    if width>=550 and not live_frame:
        x,y=width-80,64
        draw.line((x,y,x+34,y),fill='#75b8ea',width=2)
        draw.polygon([(x+34,y),(x+28,y-3),(x+28,y+3)],fill='#75b8ea')
        draw.line((x,y,x,y+34),fill='#06d6a0',width=2)
        draw.polygon([(x,y+34),(x-3,y+28),(x+3,y+28)],fill='#06d6a0')
        draw.text((x+37,y-7),'X',font=hud_font(11),fill='#75b8ea')
        draw.text((x-4,y+37),'Y',font=hud_font(11),fill='#06d6a0')
        draw.text((width-142,20),'参考方向 · 非比例',font=hud_font(11),fill='#8295a8')
    scale_width=int(interval*scale)
    y=height-28
    draw.line((24,y,24+scale_width,y),fill='#aec4d5',width=2)
    draw.text((24,y-22),f'{interval*1000:g} mm / 格',font=hud_font(12),fill='#aec4d5')
    footer = ('当前 RGB-D 帧 · 仅显示预览 · 未配准 / 未累计' if live_frame else
              'XYZ 正视 / 正交投影 · 仅滚轮缩放 · 缺失区域尚未评估')
    draw.text((max(24,width-(390 if live_frame else 370)),height-27),footer,font=hud_font(11),fill='#8295a8')
    return image
