"""Single RGB-D Gaussian fitting in the observed camera frame (not multiview)."""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image

from femto.io_utils import load_json, write_json

from femto.paths import REPO_ROOT as ROOT
from femto.dataset.single_frame_source import resolve_source
MARKER = ROOT / 'outputs' / 'latest_single_gs.txt'
SCHEMA = 'femto-single-rgbd-gs-v1'


def prepare(photo: Path, output: Path):
    """Rectify one saved photograph, preserving all original capture bytes."""
    import cv2  # capture runtime only
    from femto.depth.depth_quality import process_depth_quality
    photo, manifest, source_info = resolve_source(photo)
    if manifest.get('alignment') != 'depth_to_color':
        raise ValueError('Select a saved, depth-to-color aligned Femto RGB-D photo')
    files = [photo / manifest['files'][key] for key in ('color', 'depth')]
    files += [photo / 'calibration' / name for name in ('color_intrinsic.json', 'distortion.json')]
    files.append(photo / 'manifest.json')
    if source_info['source_kind'] == 'fixed_keyframe':
        files.extend([photo/'frames.jsonl', photo/'keyframe_report.json'])
    temporal_frame = None
    if manifest.get('temporal_fusion'):
        temporal_frame = json.loads((photo/'frames.jsonl').read_text(encoding='utf-8').strip())
        files.append(photo/temporal_frame['depth_confidence'])
    for path in files:
        path.resolve().relative_to(photo)
    hashes = {str(p.relative_to(photo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    intr = load_json(files[2])
    distortion = load_json(files[3])['rgb']
    if 'BROWN_CONRADY_K6' not in distortion.get('model', ''):
        raise ValueError('Unsupported RGB distortion model')
    K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]], dtype=float)
    if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError('Invalid camera intrinsics')
    size = (int(intr['width']), int(intr['height']))
    with Image.open(files[0]) as im:
        rgb = np.asarray(im.convert('RGB'))
    with Image.open(files[1]) as im:
        depth = np.asarray(im)
    if rgb.shape[:2] != size[::-1] or depth.shape != size[::-1] or depth.dtype != np.uint16:
        raise ValueError('RGB-D dimensions/type do not match the calibration')
    coeff = np.array([distortion[k] for k in ('k1','k2','p1','p2','k3','k4','k5','k6')])
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, coeff, size, 0, size)
    mx, my = cv2.initUndistortRectifyMap(K, coeff, None, new_K, size, cv2.CV_32FC1)
    rgb = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    temporal_confidence = None
    if temporal_frame is not None:
        from femto.depth.temporal_depth_layer import load_temporal_confidence
        temporal_confidence = cv2.remap(load_temporal_confidence(photo, temporal_frame, depth),
                                       mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    depth = cv2.remap(depth, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    low, high = manifest['image']['depth_range_m']
    depth = np.where((depth >= low*1000) & (depth <= high*1000), depth, 0).astype(np.uint16)
    if np.count_nonzero(depth) < 100:
        raise ValueError('Too few valid depth pixels for single-frame Gaussian fitting')
    output = Path(output).resolve()
    if output == photo or photo in output.parents:
        raise ValueError('Derived output must be outside the source photo directory')
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(rgb).save(output / 'source_rectified.png')
    Image.fromarray(depth).save(output / 'depth_rectified_mm.png')
    if temporal_confidence is None:
        _, _, depth_quality, depth_quality_paths = process_depth_quality(
            depth, min_range_mm=float(low)*1000, max_range_mm=float(high)*1000,
            output=output/'derived'/'depth_quality', rgb_bgr=rgb[:, :, ::-1])
    else:
        temporal_confidence[depth == 0] = 0
        quality_dir = output/'derived'/'depth_quality'
        quality_dir.mkdir(parents=True)
        Image.fromarray(depth).save(quality_dir/'filled_depth_mm.png')
        np.save(quality_dir/'confidence.npy', temporal_confidence, allow_pickle=False)
        depth_quality = dict(policy='preserved_temporal_confidence',
                             temporal_fusion=manifest['temporal_fusion'])
        depth_quality_paths = dict(filled_depth=str(quality_dir/'filled_depth_mm.png'),
                                   confidence=str(quality_dir/'confidence.npy'))
    lineage = dict(schema=SCHEMA, source_photo=str(photo), source_hashes=hashes,
                   **source_info,
                   source_fingerprint=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                   K=new_K.tolist(), width=size[0], height=size[1],
                   coordinate_system='camera-local RDF metres; +X right, +Y down, +Z forward',
                   external_pose_valid=False, T_WC=None, fixed_view=True,
                   scale_validated=False, unseen_surfaces_reconstructed=False,
                   rectification='RGB linear, aligned uint16 depth nearest; rational Brown-Conrady K6',
                   depth_quality=depth_quality,
                   depth_quality_files={key: str(Path(value).relative_to(output)) for key, value in depth_quality_paths.items()},
                   depth_training_policy='continuous confidence preserved; conservative repair <=0.3; unknown=0.0')
    write_json(output / 'single_input.json', lineage)
    return output


@dataclass(frozen=True)
class SingleConfig:
    iterations: int = 500
    max_gaussians: int = 300000
    training_width: int = 1920
    max_cuda_reserved_mib: int = 4096
    min_cuda_free_mib: int = 1536
    checkpoint_every: int = 100


def seed_arrays(depth, rgb, K, cap, confidence=None):
    if cap < 1:
        raise ValueError('Point cap must be positive')
    h, w = depth.shape
    stride = max(1, math.ceil(math.sqrt(h*w/cap)))
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    while xx.size > cap:
        stride += 1
        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[yy, xx].astype(np.float32)*.001
    valid = (z > 0) & np.isfinite(z)
    if confidence is not None:
        confidence=np.asarray(confidence,dtype=np.float32)
        if confidence.shape!=depth.shape or not np.isfinite(confidence).all() or np.any((confidence<0)|(confidence>1)):
            raise ValueError('Invalid continuous depth confidence')
        valid &= confidence[yy,xx]>0
    x, y, z = xx[valid], yy[valid], z[valid]
    means = np.column_stack(((x-K[0,2])*z/K[0,0], (y-K[1,2])*z/K[1,1], z)).astype(np.float32)
    scales = np.column_stack((z/K[0,0]*stride*.65, z/K[1,1]*stride*.65, z*.0005)).astype(np.float32)
    result = means, rgb[y, x].astype(np.float32)/255, scales, stride
    if confidence is None:
        return result
    confidence = np.asarray(confidence, dtype=np.float32)
    if confidence.shape != depth.shape:
        raise ValueError('Depth confidence dimensions do not match depth')
    return (*result, confidence[yy[valid], xx[valid]])


def fit(output: Path, config=SingleConfig(), *, publish=True):
    import torch
    from gsplat.rendering import rasterization
    from femto.gsplat.scene_gsplat_guard import guard_cuda, ResourceLimitError, cuda_memory_snapshot
    from femto.gsplat.scene_gsplat_export import write_trained_gaussian_ply
    output = Path(output).resolve()
    meta = load_json(output/'single_input.json')
    if meta.get('schema') != SCHEMA or not meta.get('fixed_view'):
        raise ValueError('Not a single-frame input')
    if (output/'single_report.json').exists():
        raise FileExistsError('Completed single-frame model already exists; use a new output')
    if not (1 <= config.iterations <= 2000 and 100 <= config.max_gaussians <= 300000
            and 160 <= config.training_width <= 1920 and config.checkpoint_every > 0
            and config.max_cuda_reserved_mib > 0 and config.min_cuda_free_mib > 0):
        raise ValueError('Single-frame safety bounds: 1..2000 steps, 100..300000 points, 160..1920 pixels')
    write_json(output/'single_config.json', asdict(config))
    device = torch.device('cuda:0')
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = min(config.max_cuda_reserved_mib*2**20/total, 1-config.min_cuda_free_mib*2**20/total)
        if fraction <= 0:
            raise ResourceLimitError('Insufficient total GPU memory')
        torch.cuda.set_per_process_memory_fraction(min(fraction, 1), device)
        guard_cuda(device, config)
        with Image.open(output/'source_rectified.png') as im: rgb = np.asarray(im.convert('RGB'))
        with Image.open(output/'derived'/'depth_quality'/'filled_depth_mm.png') as im: depth = np.asarray(im)
        confidence = np.load(output/'derived'/'depth_quality'/'confidence.npy').astype(np.float32)
        K = np.array(meta['K'], dtype=np.float32)
        xyz, colors, scales, stride, seed_confidence = seed_arrays(
            depth, rgb, K, config.max_gaussians, confidence
        )
        if len(xyz) < 100: raise ValueError('Too few valid sampled depth points')
        means = torch.tensor(xyz, device=device)  # geometry is measured and frozen
        quats = torch.zeros((len(xyz),4), device=device); quats[:,0] = 1
        base = torch.tensor(scales, device=device)
        log_scales = torch.nn.Parameter(base.log())
        sh0 = torch.nn.Parameter(torch.tensor((colors[:,None,:]-.5)/.28209479177387814, device=device))
        seed_confidence_tensor = torch.tensor(seed_confidence, device=device)
        confidence_opacity_logit = torch.logit(seed_confidence_tensor.clamp(1e-6, 1-1e-6))
        opacity = torch.nn.Parameter(torch.where(
            seed_confidence_tensor >= .999,
            torch.full((len(xyz),), 2., device=device),
            confidence_opacity_logit.clamp(max=2.),
        ))
        optim = torch.optim.Adam([{'params':[sh0], 'lr':.015}, {'params':[opacity], 'lr':.015},
                                  {'params':[log_scales], 'lr':.003}])
        w = min(config.training_width, meta['width']); h = round(w*meta['height']/meta['width'])
        Kt = K.copy(); Kt[0] *= w/meta['width']; Kt[1] *= h/meta['height']
        target = torch.tensor(np.asarray(Image.fromarray(rgb).resize((w,h), Image.Resampling.BILINEAR)).copy(), device=device).float()/255
        confidence_training = torch.tensor(
            np.asarray(Image.fromarray(confidence).resize((w,h), Image.Resampling.NEAREST)).copy(),
            device=device,
        )
        mask = confidence_training > 0
        if not mask.any(): raise ValueError('No valid depth remains at training resolution')
        views = torch.eye(4, device=device)[None]  # local sensor frame, never a fabricated external pose
        Ks = torch.tensor(Kt, device=device)[None]
        started = time.monotonic()
        def render():
            guard_cuda(device, config)
            return rasterization(means=means, quats=quats, scales=log_scales.exp(), opacities=opacity.sigmoid(),
                colors=sh0, viewmats=views, Ks=Ks, width=w, height=h, sh_degree=0,
                packed=True, rasterize_mode='antialiased', render_mode='RGB')[:2]
        with (output/'training_metrics.jsonl').open('w', encoding='utf-8') as log:
            for step in range(1,config.iterations+1):
                optim.zero_grad(set_to_none=True)
                rendered, alpha = render()
                rgb_error = (rendered[0]-target).abs().mean(dim=-1)
                loss = ((rgb_error * confidence_training).sum() / confidence_training.sum().clamp_min(1e-12)
                        + .1*((alpha[0,:,:,0]-mask.float()).abs()*confidence_training).sum()
                        / confidence_training.sum().clamp_min(1e-12))
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite single-frame loss')
                loss.backward(); optim.step()
                with torch.no_grad():
                    log_scales.clamp_(min=(base*.5).log(), max=(base*1.5).log())
                    opacity.clamp_(-6, 6)
                    repaired_seed = seed_confidence_tensor < .999
                    opacity[repaired_seed] = torch.minimum(opacity[repaired_seed], confidence_opacity_logit[repaired_seed])
                memory = guard_cuda(device, config)
                if step == 1 or step % 10 == 0 or step == config.iterations:
                    record = dict(step=step,total=config.iterations,gaussians=len(xyz),max_gaussians=config.max_gaussians,
                                  loss=float(loss.detach()), elapsed_s=time.monotonic()-started, **memory)
                    line = json.dumps(record); print(line, flush=True); log.write(line+'\n'); log.flush(); os.fsync(log.fileno())
                if step % config.checkpoint_every == 0 or step == config.iterations:
                    from femto.gsplat.scene_gsplat_trainer import _to_cpu
                    payload = dict(schema=SCHEMA, step=step, source_fingerprint=meta['source_fingerprint'],
                                   means=means.cpu(), sh0=sh0.detach().cpu(), log_scales=log_scales.detach().cpu(),
                                   opacity_logits=opacity.detach().cpu(), optimizer=_to_cpu(optim.state_dict()))
                    with (output/'checkpoint.tmp').open('wb') as f:
                        torch.save(payload,f); f.flush(); os.fsync(f.fileno())
                    (output/'checkpoint.tmp').replace(output/'checkpoint.pt')
                time.sleep(.01)
        with torch.no_grad():
            image, _ = render()
            mse = ((image[0][mask]-target[mask])**2).mean().item()
            Image.fromarray((image[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)).save(output/'render.png')
        model = output/'model.ply'
        write_trained_gaussian_ply(model, means=xyz, sh0=sh0.detach().cpu().numpy(), sh_rest=np.zeros((len(xyz),0,3),np.float32),
            opacity_logits=opacity.detach().cpu().numpy(), log_scales=log_scales.detach().cpu().numpy(),
            quaternions_wxyz=quats.cpu().numpy(), comment='Single RGB-D fixed-view fit; measured camera-local geometry; unvalidated scale')
        report = dict(**meta, completed=True, model_ply=str(model), model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                      iterations=config.iterations, gaussian_count=len(xyz), seed_stride=stride,
                      repaired_gaussian_count=(None if meta.get('depth_quality',{}).get('temporal_fusion') else int(np.sum(seed_confidence < .999))),
                      low_confidence_gaussian_count=int(np.sum(seed_confidence < .999)),
                      confidence_policy='continuous source confidence weights loss and caps opacity; repairs <=0.3; unknown omitted',
                      training_resolution=[w,h], same_view_psnr=-10*math.log10(max(mse,1e-12)),
                      evaluation_scope='same training view only; no held-out/new-view validation',
                      multiview_accepted=False, geometry_optimized=False, densification=False)
        write_json(output/'single_report.json', report)
        if publish: MARKER.write_text(str(output),encoding='utf-8')
        return report
    except (ResourceLimitError, torch.cuda.OutOfMemoryError) as error:
        try: memory = cuda_memory_snapshot(device)
        except Exception: memory = {}
        write_json(output/'resource_stop.json', dict(error=str(error), memory=memory, checkpoint=str(output/'checkpoint.pt')))
        raise


def launch(photo, *, iterations=500, on_progress=None):
    """Called in capture runtime; CUDA child owns training allocations."""
    import subprocess
    from femto.gsplat.gsplat_backend import runtime_environment, runtime_python
    output = ROOT/'outputs'/('single_gs_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000))
    prepare(Path(photo), output)
    command = [str(runtime_python()), '-m', 'femto.gsplat.scene_single_gs', 'train', str(output), '--iterations', str(iterations)]
    with (output/'trainer_stdout.log').open('w',encoding='utf-8') as log:
        process = subprocess.Popen(command,cwd=ROOT,env=runtime_environment(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        for line in process.stdout:
            log.write(line); log.flush()
            try: record=json.loads(line)
            except ValueError: continue
            if on_progress and 'step' in record: on_progress(record)
        if process.wait() != 0: raise RuntimeError(f'Single-frame fitting failed; see {output / "trainer_stdout.log"}')
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['train','build'])
    parser.add_argument('path',type=Path)
    parser.add_argument('--iterations',type=int,default=500)
    args=parser.parse_args()
    if args.mode=='build': print(launch(args.path,iterations=args.iterations))
    else: fit(args.path,SingleConfig(iterations=args.iterations))
