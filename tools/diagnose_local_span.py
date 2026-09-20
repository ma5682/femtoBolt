"""Read-only sparse RGB-D PnP cross-check; not an alternate production pose."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/ scripts import the femto package from the repository root
from femto.io_utils import imread_unicode as _read_image
from femto.geometry.scene_reconstruction import _distortion_maps
from femto.tracking.moving_quality import rotation_deg


def diagnose(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(p for p in source.rglob('*') if p.is_file() and 'derived' not in p.relative_to(source).parts)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows = [json.loads(line) for line in (source/'derived/live_tracking.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    selected = [r for r in rows if r.get('selected_for_fusion')]
    mx, my, k, width, height = _distortion_maps(source)
    sift = cv2.SIFT_create(nfeatures=6000)
    prepared = {}
    for r in selected:
        i = r['frame_id']
        rgb = cv2.remap(_read_image(source/f'rgb/{i:06}.png', cv2.IMREAD_COLOR), mx, my, cv2.INTER_LINEAR)
        dep = cv2.remap(_read_image(source/f'depth/{i:06}.png', cv2.IMREAD_UNCHANGED), mx, my, cv2.INTER_NEAREST)
        kp, desc = sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), None)
        prepared[i] = kp, desc, dep
    pairs = []
    cv2.setRNGSeed(20260915)
    for ai in range(len(selected)):
        for bi in range(ai+1, len(selected)):
            a, b = selected[ai], selected[bi]
            ka, da, depth = prepared[a['frame_id']]
            kb, db, _ = prepared[b['frame_id']]
            forward = cv2.BFMatcher().knnMatch(da, db, k=2)
            reverse = cv2.BFMatcher().knnMatch(db, da, k=2)
            reverse_good = {(m.trainIdx, m.queryIdx) for m,n in reverse if m.distance < .7*n.distance}
            matches = [m for m,n in forward if m.distance < .7*n.distance and (m.queryIdx,m.trainIdx) in reverse_good]
            obj, uv = [], []
            for m in matches:
                x,y = ka[m.queryIdx].pt
                ix,iy = int(round(x)),int(round(y))
                if not (1 <= ix < width-1 and 1 <= iy < height-1):continue
                neighborhood = depth[iy-1:iy+2,ix-1:ix+2]
                if neighborhood.min()<250 or neighborhood.max()>2210 or np.ptp(neighborhood)>20:continue
                z = float(depth[iy,ix])/1000
                obj.append([(x-k[0,2])*z/k[0,0], (y-k[1,2])*z/k[1,1], z])
                uv.append(kb[m.trainIdx].pt)
            item = dict(source_frame=a['frame_id'], target_frame=b['frame_id'], correspondences=len(obj))
            if len(obj)>=12:
                obj,uv = np.asarray(obj),np.asarray(uv)
                ok, rv, tv, inliers = cv2.solvePnPRansac(obj,uv,k,None,iterationsCount=2000,
                    reprojectionError=2.5,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
                if ok and inliers is not None and len(inliers)>=12:
                    ids=inliers[:,0]
                    rv,tv=cv2.solvePnPRefineLM(obj[ids],uv[ids],k,None,rv,tv)
                    t=np.eye(4);t[:3,:3]=cv2.Rodrigues(rv)[0];t[:3,3]=tv[:,0]
                    predicted=np.linalg.inv(np.asarray(b['T_WC']))@np.asarray(a['T_WC'])
                    diff=np.linalg.inv(t)@predicted
                    pixels=cv2.projectPoints(obj[ids],rv,tv,k,None)[0][:,0]
                    item.update(inliers=len(ids),pnp_translation_m=float(np.linalg.norm(tv)),
                        online_translation_m=float(np.linalg.norm(predicted[:3,3])),
                        translation_disagreement_mm=float(np.linalg.norm(diff[:3,3])*1000),
                        rotation_disagreement_deg=rotation_deg(diff),
                        reprojection_median_px=float(np.median(np.linalg.norm(pixels-uv[ids],axis=1))),
                        T_target_source=t.tolist())
            pairs.append(item)
    report=dict(source=str(source.resolve()),method='mutual SIFT + source raw depth + PnP RANSAC, independent of dense odometry',
        limitation='Same sensor evidence, no physical ground truth; dynamic screen and scene motion may affect both methods.',
        frames=[dict(frame_id=r['frame_id'],xyz_mm=r.get('xyz_mm'),step_m=r.get('step_m'),
                     accepted=r.get('selected_for_fusion')) for r in rows],pairs=pairs,
        source_hashes=hashes,raw_unchanged=all(hashlib.sha256((source/p).read_bytes()).hexdigest()==h for p,h in hashes.items()))
    (output/'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(pairs=pairs,raw_unchanged=report['raw_unchanged']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
    diagnose(a.source,a.output)
