"""Display-only pieces of selected source faces; no planning geometry is edited."""
import cv2
import numpy as np

MAX_CLIP_PAIRS = 20000
MAX_MASK_RECTANGLES = 100000


def _rectangles(mask):
    """Exact disjoint pixel rectangles, merging equal runs on adjacent rows."""
    active, result = {}, []
    for y in range(len(mask)+1):
        row = mask[y] if y < len(mask) else np.zeros(mask.shape[1], np.uint8)
        ends = np.flatnonzero(np.diff(np.pad(row.astype(np.int8), (1, 1))))
        runs = set(zip(ends[::2].tolist(), ends[1::2].tolist()))
        for run in active.keys()-runs:
            result.append((run[0]-.5, active[run]-.5, run[1]-.5, y-.5))
        active = {run: active.get(run, y) for run in runs}
        if len(result)+len(active) > MAX_MASK_RECTANGLES:
            raise ValueError('选区三维显示超出预算，请返回参考图核对')
    return np.asarray(result, float).reshape(-1, 4)


def selected_surface(vertices, faces, colors, ids, region):
    """Crop display cells and retain original face IDs for world-coordinate picks.

    Whole interior faces retain their vertices. Only boundary faces are clipped
    against the painted pixel rectangles. Holes and disconnected runs stay empty.
    This is a visual derivative, never an area ledger or a route support proof.
    """
    intent = region.get('selection_intent')
    if not intent:
        used, inverse = np.unique(faces[ids], return_inverse=True)
        return vertices[used].copy(), inverse.reshape(-1, 3), (None if colors is None else colors[used].copy()), ids.copy()
    from femto.annotation.selection_intent import intent_mask
    from femto.geometry.pixel_clip import clip_triangle_planes
    mask = intent_mask(intent).astype(np.uint8)
    k = np.asarray(region['view_reference']['undistorted_intrinsic'], float)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0,0], k[1,1]) <= 0:
        raise ValueError('选区显示缺少有效参考内参')
    triangles = vertices[faces[ids]]
    front = (triangles[:,:,2] > 0).all(axis=1)
    ids, triangles = ids[front], triangles[front]
    uv = triangles[:,:,:2]/triangles[:,:,2:3]*[k[0,0],k[1,1]]+k[:2,2]
    low, high = uv.min(axis=1), uv.max(axis=1)
    h, w = mask.shape
    # Integral counts only classify whole bounding boxes; mixed faces are clipped.
    lo = np.floor(np.clip(low+.5, [0,0], [w,h])).astype(int)
    hi = np.floor(np.clip(high+.5, [-1,-1], [w-1,h-1])).astype(int)+1
    integral = cv2.integral(mask, sdepth=cv2.CV_32S)
    counts = integral[hi[:,1],hi[:,0]]-integral[lo[:,1],hi[:,0]]-integral[hi[:,1],lo[:,0]]+integral[lo[:,1],lo[:,0]]
    inside = ((low >= [-.5,-.5]).all(axis=1) & (high < [w-.5,h-.5]).all(axis=1)
              & (counts == np.prod(hi-lo, axis=1)) & (counts > 0))
    whole = ids[inside]
    used, inverse = np.unique(faces[whole], return_inverse=True)
    vparts, fparts, iparts = [vertices[used].copy()], [inverse.reshape(-1,3)], [whole.copy()]
    cparts = [] if colors is None else [colors[used].copy()]
    offset, tested = len(used), 0
    mixed = np.flatnonzero(~inside & (counts > 0) & (hi > lo).all(axis=1))
    rectangles = _rectangles(mask) if len(mixed) else np.empty((0,4))
    for index in mixed:
        bounds = rectangles[(rectangles[:,0] < high[index,0]) & (rectangles[:,2] > low[index,0])
                            & (rectangles[:,1] < high[index,1]) & (rectangles[:,3] > low[index,1])]
        tested += len(bounds)
        if tested > MAX_CLIP_PAIRS:
            raise ValueError('选区三维显示超出预算，请返回参考图核对')
        face = int(ids[index])
        for x0, y0, x1, y1 in bounds:
            planes = np.array([[k[0,0],0,k[0,2]-x0],[-k[0,0],0,x1-k[0,2]],
                               [0,k[1,1],k[1,2]-y0],[0,-k[1,1],y1-k[1,2]]])
            bary, _ = clip_triangle_planes(triangles[index], planes)
            if len(bary) < 3:
                continue
            points = bary @ triangles[index]
            local = np.column_stack((np.zeros(len(bary)-2,int),np.arange(1,len(bary)-1),np.arange(2,len(bary))))
            vparts.append(points); fparts.append(local+offset)
            iparts.append(np.full(len(local), face, dtype=np.int64))
            if colors is not None:
                cparts.append(np.rint(bary @ colors[faces[face]]).clip(0,255).astype(np.uint8))
            offset += len(points)
    return np.concatenate(vparts), np.concatenate(fparts), (np.concatenate(cparts) if colors is not None else None), np.concatenate(iparts)
