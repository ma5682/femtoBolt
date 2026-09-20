"""Square strokes in reference-image pixels; bounded legacy polygon storage."""
import cv2
import numpy as np


def polygon_mask(polygon, size):
    from fixed_rgb_geometry import polygon_contains
    mask = np.zeros((size[1], size[0]), np.uint8)
    if polygon:
        points=np.asarray(polygon,float)
        left,top=np.clip(np.ceil(points.min(axis=0)).astype(int),0,size)
        right,bottom=np.clip(np.floor(points.max(axis=0)).astype(int)+1,0,size)
        for row in range(top,bottom,64):
            yy,xx=np.mgrid[row:min(row+64,bottom),left:right]
            mask[row:row+len(yy),left:right]=polygon_contains(np.c_[xx.ravel(),yy.ravel()],polygon).reshape(xx.shape)
    return mask


def stroke(mask, start, end, size, erase=False):
    """Sweep an axis-aligned square continuously, including between mouse events."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    half = max(1., float(size))/2
    corners = np.array([[-half,-half],[half,-half],[half,half],[-half,half]])
    hull = cv2.convexHull(np.rint(np.r_[a+corners,b+corners]*16).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 0 if erase else 1, shift=4)


def mask_polygon(mask, limit=512):
    """Even/odd contour with doubled bridges; never silently drop components."""
    rows=np.flatnonzero(np.any(mask,axis=1));cols=np.flatnonzero(np.any(mask,axis=0))
    if not len(rows):return []
    left,top=int(cols[0]),int(rows[0]);right,bottom=int(cols[-1])+1,int(rows[-1])+1
    if left or top or right<mask.shape[1] or bottom<mask.shape[0]:
        polygon=mask_polygon(mask[top:bottom,left:right],limit)
        return [[x+left,y+top] for x,y in polygon]
    # Trace pixel-cell edges at half-pixels so edits/recovery never shrink masks.
    m=mask.astype(bool);p=np.pad(m,1);edges={};count=0
    for exposed,dx,dy,ex,ey in (
        (m & ~p[:-2,1:-1],0,0,1,0),
        (m & ~p[1:-1,2:],1,0,1,1),
        (m & ~p[2:,1:-1],1,1,0,1),
        (m & ~p[1:-1,:-2],0,1,0,0)):
        ys,xs=np.nonzero(exposed);count+=len(xs)
        if count>500000:raise ValueError('涂抹边界过于复杂，本笔未应用；请分成多个区域。')
        for x,y in zip(xs.tolist(),ys.tolist()):
            edges.setdefault((x+dx,y+dy),[]).append((x+ex,y+ey))
    loops=[]
    while edges:
        start=next(iter(edges));current=start;previous=None;loop=[]
        while True:
            loop.append(current);choices=edges[current]
            if len(choices)>1 and previous is not None:
                vx,vy=current[0]-previous[0],current[1]-previous[1]
                nxt=max(choices,key=lambda q:vx*(q[1]-current[1])-vy*(q[0]-current[0]))
            else:nxt=choices[0]
            choices.remove(nxt)
            if not choices:del edges[current]
            previous,current=current,nxt
            if current==start:break
        points=np.asarray(loop,float)-.5
        before=points-np.roll(points,1,axis=0);after=np.roll(points,-1,axis=0)-points
        loops.append(points[(before[:,0]*after[:,1]-before[:,1]*after[:,0])!=0])
    def join(parts):
        result=[]
        for loop in parts:
            if not result:result=loop.tolist()+[loop[0].tolist()]
            else:result += [loop[0].tolist()]+loop[1:].tolist()+[loop[0].tolist(),result[0]]
        return result
    result=join(loops)
    if len(result)>limit:
        simplified=[]
        for loop in loops:
            candidate=cv2.approxPolyDP(loop.astype(np.float32),1.,True).reshape(-1,2)
            simplified.append(candidate if len(candidate)>=3 else loop)
        # One-pixel simplification removes staircase corners on long drags.
        # Keep every component and hole; never replace with a convex hull.
        result=join(simplified)
    if len(result)>limit:
        raise ValueError('涂抹边界过于复杂，本笔未应用；请用较大方块或分成多个区域。')
    return result
