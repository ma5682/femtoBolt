"""Read-only, point-local raw observation evidence, separate from texture and fusion."""
from __future__ import annotations
from typing import Any, Callable

from femto.paths import OUTPUTS_ROOT
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from femto.io_utils import load_json, load_jsonl
from femto.dataset.scene_dataset_validator import validate_pose


def digest(path: str | Path) -> str:return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_view(xyz: Any,pose: Any,k: Any,depth: Any,minimum: float,maximum: float) -> dict[str, Any]:
    inverse=np.linalg.inv(pose);camera=np.asarray(xyz)@inverse[:3,:3].T+inverse[:3,3];z=camera[:,2]
    uv=camera[:,:2]/np.maximum(z[:,None],1e-12)*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
    pix=np.rint(uv).astype(int);inside=(z>0)&(pix[:,0]>=0)&(pix[:,0]<depth.shape[1])&(pix[:,1]>=0)&(pix[:,1]<depth.shape[0])
    raw=depth[np.clip(pix[:,1],0,depth.shape[0]-1),np.clip(pix[:,0],0,depth.shape[1]-1)]/1000.
    valid=inside&(raw>=minimum)&(raw<=maximum)&(raw>0)
    return dict(pixel=uv[0].tolist(),observed_depth_m=float(raw[0]) if valid[0] else None,
        projected_depth_m=float(z[0]),signed_residual_mm=float((raw[0]-z[0])*1000) if valid[0] else None,
        all_observed=bool(valid.all()),max_abs_residual_m=float(np.abs(raw-z).max()) if valid.all() else None)


def diagnose(model: Any,point: Any,check: Callable[..., Any]=lambda:None,progress: Callable[..., Any]=lambda stage:None) -> dict[str, Any]:
    check();progress('核对局部模型、参考系与来源')
    xyz=point['point_world_m'];tid=int(point['triangle_index']);mesh=model.mesh
    tri=np.asarray(mesh.vertices)[np.asarray(mesh.triangles)[tid]];samples=np.vstack([xyz,tri,tri.mean(axis=0)])
    report=dict(schema='femto-local-geometry-diagnostics-v1',created_at=datetime.now(timezone.utc).isoformat(),
        coordinate_frame='world_rdf',unit='metre',geometry_sha256=model.store.digest,
        annotation_id=point.get('annotation_id'),triangle_index=tid,xyz_m=xyz,view_reference=model.view_reference,
        observations=[],warnings=[],source_status='unknown',support_kind='source_unknown',support_frame_count=None,
        support_definition='current_raw_point_plus_three_vertices_and_centroid_agree_in_same_view; not_TSDF_weight',
        historical_asset_integrity='capture_time_content_hashes_not_available; current_read_hashes_recorded',
        physical_accuracy_verified=False,execution_ready=False)
    uv=model.projection.project([xyz])[0];x,y=np.rint(uv).astype(int)
    depth=model.projection.depth;ref=float(depth[y,x])/1000 if 0<=y<depth.shape[0] and 0<=x<depth.shape[1] else 0.
    report['reference_depth_missing']=not model.projection.minimum<=ref<=model.projection.maximum
    report['reference_pixel']=uv.tolist()
    try:
        dataset=Path(model.store.origin_reference['source_dataset']);local=load_json(model.reconstruction/'local_geometry.json')
        reconstruction=load_json(model.reconstruction/'reconstruction_report.json');manifest=load_json(dataset/'manifest.json')
        frames=load_jsonl(dataset/'frames.jsonl');ids=[f['frame'] for f in frames]
        if not 1<=len(frames)<=128 or len(set(ids))!=len(ids):raise ValueError('帧数或重复ID无效（预算128帧）')
        if Path(reconstruction['source_dataset']).resolve()!=dataset.resolve() or [f['frame'] for f in reconstruction['frames']]!=ids:raise ValueError('融合报告与来源帧列表不匹配')
        from femto.dataset.local_reference import calibration_identity
        proof=local['reference_provenance'];calibration=calibration_identity(dataset)
        if proof['calibration']!=calibration or local['session_id']!=model.view_reference['session_id']:raise ValueError('标定或会话不匹配')
        tolerance=local['support']['tolerance_m']
        if not isinstance(tolerance,(int,float)) or not np.isfinite(tolerance) or tolerance<=0:raise ValueError('缺少已记录的支持门限')
        report['recorded_support_tolerance_m']=tolerance
        tracking_path=Path(local['tracking_report']);tracking=load_json(tracking_path);record_path=tracking_path.parent/'live_tracking.jsonl'
        records={r['frame_id']:r for r in load_jsonl(record_path)}
        raw_dataset=tracking_path.parent.parent;raw_frames={r['frame']:r for r in load_jsonl(raw_dataset/'frames.jsonl')}
        if calibration_identity(raw_dataset)!=calibration:raise ValueError('原始会话标定不匹配')
        if tracking['session_id']!=local['session_id'] or tracking['reference_frame_id']!=local['reference_frame_id']:raise ValueError('跟踪来源与模型原点不匹配')
        from femto.io_utils import imread_unicode
        from femto.geometry.rgbd_frames import distortion_maps
        mx,my,k,w,h=distortion_maps(dataset)
        report['source_hashes']={str(p.resolve()):digest(p) for p in (dataset/'manifest.json',dataset/'frames.jsonl',tracking_path,record_path)}
        for frame in frames:
            check();progress('核对原始观测帧 '+str(frame['frame']))
            item=dict(frame_id=frame['frame'],status='unknown',supports=False);report['observations'].append(item)
            try:
                pose_path=dataset/frame['pose'];pose=load_json(pose_path);record=records[frame['frame']]
                t=np.asarray(pose['T_WC'],float)
                if (pose.get('frame')!=frame['frame'] or pose.get('capture_timestamp_us')!=frame.get('camera_timestamp_us') or
                    record.get('session_id')!=local['session_id'] or record.get('calibration_sha256')!=calibration['sha256'] or
                    not frame.get('pose_valid') or not pose.get('pose_valid') or validate_pose(t) or
                    pose.get('session_id')!=local['session_id'] or pose.get('reference_frame_id')!=local['reference_frame_id'] or
                    record.get('state')!='tracking' or not record.get('selected_for_fusion') or
                    record.get('reference_frame_id')!=local['reference_frame_id'] or frame['frame'] not in tracking['accepted_frames']):raise ValueError('位姿/原点/接纳记录不一致')
                if not np.allclose(t,np.asarray(record['T_WC']),rtol=0,atol=1e-9):raise ValueError('位姿与在线记录不同；本诊断尚未独立核验精配准修正来源')
                rgb_path=dataset/frame['rgb'];depth_path=dataset/frame['depth']
                original=raw_frames[frame['frame']]
                if (rgb_path.resolve()!=(raw_dataset/original['rgb']).resolve() or depth_path.resolve()!=(raw_dataset/original['depth']).resolve() or
                    frame.get('camera_timestamp_us')!=record.get('capture_timestamp_us') or frame.get('camera_timestamp_us')!=original.get('camera_timestamp_us')):raise ValueError('RGB/Depth路径或采集时间与原始会话不一致')
                raw=imread_unicode(depth_path,cv2.IMREAD_UNCHANGED)
                if raw.shape!=(h,w) or raw.dtype!=np.uint16:raise ValueError('原始深度尺寸或单位无效')
                raw=cv2.remap(raw,mx,my,cv2.INTER_NEAREST)
                item.update(sample_view(samples,t,k,raw,model.projection.minimum,model.projection.maximum))
                item.update(status='checked',supports=item['all_observed'] and item['max_abs_residual_m']<=tolerance,
                    pose_path=str(pose_path.resolve()),pose_sha256=digest(pose_path),depth_path=str(depth_path.resolve()),depth_sha256=digest(depth_path),rgb_path=str(rgb_path.resolve()),rgb_sha256=digest(rgb_path))
                # Current file hashes enable later comparison, not retrospective capture integrity.
                if frame['frame']==local['reference_frame_id']:
                    if item['depth_sha256']!=model.view_reference['depth_sha256'] or item['rgb_sha256']!=model.view_reference['rgb_sha256']:raise ValueError('参考RGB/深度指纹改变')
            except (OSError,ValueError,KeyError,TypeError) as e:item.update(status='unknown',supports=False,reason=str(e))
        known=[r for r in report['observations'] if r['status']=='checked'];count=sum(r['supports'] for r in known)
        report.update(source_status='checked' if len(known)==len(frames) else 'partial',support_frame_count=count,
            support_kind=('multi_frame_supported' if count>=2 else 'single_frame_supported' if count==1 else 'no_consistent_raw_support') if len(known)==len(frames) else 'source_unknown',
            checked_frame_count=len(known),total_frame_count=len(frames))
    except (OSError,ValueError,KeyError,TypeError) as e:report['warnings'].append(str(e))
    # Multiple intersections are geometry evidence, not proof of TSDF ghosting:
    # a valid thick object can have front and back surfaces on the same ray.
    check();progress('分离几何与外观诊断')
    hits=[]
    if hasattr(model.projection,'scene'):
        import open3d as o3d
        direction=np.r_[((uv-[model.projection.k[0,2],model.projection.k[1,2]])/[model.projection.k[0,0],model.projection.k[1,1]]),1.]
        start=np.zeros(3)
        for _ in range(4):
            check();z=float(model.projection.scene.cast_rays(o3d.core.Tensor(np.r_[start,direction][None].astype(np.float32)))['t_hit'].numpy()[0])
            if not np.isfinite(z):break
            at=start+z*direction;hits.append(float(at[2]));start=at+direction*.0001
    report['diagnosis']=dict(geometry_ray_layers=dict(depths_m=hits,probe_limit=4,ray_restart_epsilon_m=.0001,
        status='multiple_intersections_observed' if len(hits)>1 else 'no_multiple_intersections_in_probe',cause='unknown; back_surface_or_real_layers_or_reconstruction_artifact'),
        fusion_ghosting=dict(status='not_established',evidence='per_view_signed_depth_residuals_only; no causal attribution'),
        texture_alignment=dict(status='not_measured',evidence='RGB projection comparison only; appearance must not establish geometry support'))
    report['limitations']=['支持帧数为当前原始观测与模型的一致性，不是TSDF逐体素贡献、独立视角数量或精度。',
        '原深度使用最近邻去畸变，不补洞；读取当前源文件并记录哈希，旧记录缺采集时内容哈希，无法追溯证明当时字节未变。',
        '只检查选中点、所在面三顶点和面心；双交点也可能是物体前后表面，不自动归因融合重影。',
        '精配准位姿与在线记录不同时标来源未知，不替换为在线位姿，也不以产品10–20mm目标设门限。']
    return report


def render_comparison(model: Any,report: Any) -> Any:
    from PIL import Image,ImageDraw,ImageFont
    x,y=np.rint(report['reference_pixel']).astype(int);h,w=model.rgb.shape[:2]
    x0=max(0,min(w-96,x-48));y0=max(0,min(h-96,y-48));x1=min(w,x0+96);y1=min(h,y0+96)
    raw=model.projection.depth[y0:y1,x0:x1].astype(float)/1000;z=model.projection.z[y0:y1,x0:x1]
    measured=np.r_[raw.ravel(),z.ravel()];measured=measured[np.isfinite(measured)&(measured>=model.projection.minimum)&(measured<=model.projection.maximum)]
    lo=float(measured.min()) if len(measured) else model.projection.minimum
    hi=max(lo+.001,float(measured.max())) if len(measured) else model.projection.maximum
    def colors(values):
        values=np.asarray(values);valid=np.isfinite(values)&(values>=model.projection.minimum)&(values<=model.projection.maximum)
        scaled=np.nan_to_num((values-lo)/(hi-lo)*255,nan=0,posinf=0)
        a=cv2.applyColorMap(np.clip(scaled,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)[:,:,::-1];a[~valid]=0;return a
    rgb=model.rgb[y0:y1,x0:x1];overlay=rgb.copy();mask=model.projection.ids[y0:y1,x0:x1]>=0
    edges=cv2.Canny(mask.astype(np.uint8)*255,50,100)>0;overlay[edges]=[0,255,255]
    # Face boundaries reveal reprojection without using photo texture as geometry.
    ids=model.projection.ids[y0:y1,x0:x1];edge=np.zeros(mask.shape,bool);edge[:,1:]|=ids[:,1:]!=ids[:,:-1];edge[1:]|=ids[1:]!=ids[:-1];overlay[edge&mask]=[0,220,180]
    canvas=Image.new('RGB',(800,870),'#14202b');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    names=['参考RGB（去畸变）','原始深度（最近邻去畸变）','融合表面深度（非纹理）','网格边界投影到参考RGB']
    marker={'multi_frame_supported':'#25f08b','single_frame_supported':'#ffb347'}.get(report['support_kind'],'#de69ff')
    for i,a in enumerate((rgb,colors(raw),colors(z),overlay)):
        ox=(i%2)*400;oy=(i//2)*420;canvas.paste(Image.fromarray(a).resize((400,390)),(ox,oy+30));draw.text((ox+8,oy+5),names[i],font=font,fill='white')
        px=ox+(x-x0)/max(1,x1-x0)*400;py=oy+30+(y-y0)/max(1,y1-y0)*390
        draw.line((px-9,py,px+9,py),fill=marker,width=2);draw.line((px,py-9,px,py+9),fill=marker,width=2)
    draw.text((8,843),f'同像素；深度共用局部色标 {lo*1000:.1f}–{hi*1000:.1f}mm；黑色无有效值。',font=font,fill='white')
    return canvas


def show_diagnostics(parent: Any, model: Any, report: Any) -> Any:
    """Open the local geometry evidence report in a Qt dialog."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTextBrowser, QVBoxLayout
    image = render_comparison(model, report)
    pixels = np.ascontiguousarray(np.asarray(image.convert("RGB")))
    qimage = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format.Format_RGB888).copy()
    window = QDialog(parent); window.setWindowTitle("同位置几何来源诊断"); window.resize(1150, 850)
    layout = QHBoxLayout(window); preview = QLabel(); preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
    preview.setPixmap(QPixmap.fromImage(qimage).scaled(640, 696, Qt.KeepAspectRatio, Qt.SmoothTransformation)); layout.addWidget(preview)
    side = QVBoxLayout(); layout.addLayout(side, 1)
    labels={'multi_frame_supported':'多帧观测支持','single_frame_supported':'单帧观测支持','source_unknown':'来源未知/核验不完整','no_consistent_raw_support':'无一致原深度支持'}
    summary=labels.get(report.get('support_kind'),'来源未知')+'；已核对支持帧数 '+str(report.get('support_frame_count','—'))+'（不是精度）\n参考深度：'+('缺失/量程外' if report.get('reference_depth_missing') else '存在')
    side.addWidget(QLabel(summary)); save_button=QPushButton('保存诊断 JSON 与对照图'); side.addWidget(save_button)
    text=QTextBrowser(); text.setPlainText(json.dumps(report,ensure_ascii=False,indent=2)); side.addWidget(text,1)
    def save():
        try: folder=save_diagnostics(report, render_comparison(model, report)); QMessageBox.information(window,'诊断已保存',str(folder))
        except (OSError,ValueError) as error: QMessageBox.critical(window,'诊断保存失败',str(error))
    save_button.clicked.connect(save); window.show(); return window


def save_diagnostics(report: Any,image: Any,parent: str | Path | None=None) -> Any:
    import tempfile,os,uuid
    root=Path(parent) if parent else OUTPUTS_ROOT/'local_geometry_diagnostics'
    root.mkdir(parents=True,exist_ok=True);root=root.resolve();target=root/('diagnostic_'+uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(prefix='.pending_',dir=root) as tmp:
        folder=Path(tmp);(folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');image.save(folder/'comparison.png')
        (folder/'files.json').write_text(json.dumps({p.name:digest(p) for p in folder.iterdir()},indent=2),encoding='utf-8')
        for file in folder.iterdir():
            with file.open('r+b') as stream:os.fsync(stream.fileno())
        if folder.resolve().parent!=root or target.parent!=root:raise ValueError('Diagnostic publication path mismatch')
        os.replace(folder,target)
    return target
