"""Conservative boundary clearance for demo scan footprints; not contact validation."""
import numpy as np
from femto.planning.region_coverage import point_segment_distance


def constrain_path(projection, mesh, region, plan, width_mm, margin_mm, check=lambda:None):
    vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles)[region['triangle_indices']]
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    edges,counts=np.unique(edges,axis=0,return_counts=True)
    if np.any(counts>2):raise ValueError('Non-manifold region boundary')
    boundary=vertices[edges[counts==1]]
    if not len(boundary):raise ValueError('Region has no checkable boundary')
    radius=(width_mm/2+margin_mm)/1000
    # Distance to all boundary segments is conservative on folded surfaces.
    # Midpoint clearance minus half chord bounds clearance for the whole chord.
    pairs=[(left,right) for segment in plan['segments'] for left,right in zip(segment,segment[1:])]
    mids=np.asarray([(np.asarray(a['xyz_m'])+b['xyz_m'])/2 for a,b in pairs]).reshape(-1,3)
    half=np.asarray([np.linalg.norm(np.asarray(a['xyz_m'])-b['xyz_m'])/2 for a,b in pairs])
    if len(pairs)*len(boundary)>100_000_000:
        from femto.planning.boundary_clearance import boundary_clearance_mask
        accepted=boundary_clearance_mask(mids,half,boundary,radius,check)
    else:
        # Keep the old arithmetic/order for already supported small requests.
        clearance=np.full(len(mids),np.inf)
        for start in range(0,len(boundary),128):
            check()
            for a,b in boundary[start:start+128]:clearance=np.minimum(clearance,point_segment_distance(mids,a,b))
        accepted=clearance-half>=radius+1e-9
    visibility_checked=hasattr(projection,'visible')
    if visibility_checked and len(mids):
        accepted &= projection.visible(mids,check=check)
        endpoints=np.asarray([p['xyz_m'] for pair in pairs for p in pair])
        accepted &= projection.visible(endpoints,check=check).reshape(-1,2).all(axis=1)
    polygon=region.get('screen_polygon');polygon_checked=False
    if polygon and len(mids):
        from femto.annotation.fixed_rgb_geometry import polygon_contains
        # Entire 3D ball containing a swept footprint is bounded in image space.
        if not hasattr(projection,'k'):raise ValueError('Camera intrinsics required for polygon footprint constraint')
        uv=projection.project(mids);r=radius+half;z=mids[:,2]
        valid=z>r
        bound=np.full(len(mids),np.inf)
        bound[valid]=max(projection.k[0,0],projection.k[1,1])*r[valid]/(z[valid]-r[valid])*(1+np.linalg.norm(mids[valid,:2],axis=1)/z[valid])
        polygon=np.asarray(polygon,float);distance=np.full(len(mids),np.inf)
        for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
            distance=np.minimum(distance,point_segment_distance(uv,a,b))
        accepted &= valid & polygon_contains(uv,polygon) & (distance>=bound)
        for forbidden in region.get('exclusion_polygons',[]):
            forbidden=np.asarray(forbidden,float);distance=np.full(len(mids),np.inf)
            for a,b in zip(forbidden,np.roll(forbidden,-1,axis=0)):
                distance=np.minimum(distance,point_segment_distance(uv,a,b))
            accepted &= ~polygon_contains(uv,forbidden) & (distance>=bound)
        polygon_checked=True
    output=[];index=0
    for segment in plan['segments']:
        current=[]
        for left,right in zip(segment,segment[1:]):
            if accepted[index]:
                if not current:current=[left]
                current.append(right)
            elif current:output.append(current);current=[]
            index+=1
        if current:output.append(current)
    plan['segments']=output
    return dict(schema='femto-footprint-boundary-v1',width_mm=width_mm,margin_mm=margin_mm,
        candidate_edges=len(pairs),retained_edges=int(accepted.sum()),rejected_edges=int((~accepted).sum()),
        boundary_edge_count=len(boundary),polygon_footprint_checked=polygon_checked,
        visibility_samples_checked=visibility_checked,
        method='euclidean_boundary_lower_bound_for_entire_chord_and_projected_enclosing_ball',
        contact_footprint_verified=False,transitions_planned=False,
        limitations='Conservative simulated radius; not real tool contact geometry. Visibility sampled at endpoints and midpoints, not continuously certified.')




from femto.planning.report_catalog import coverage_report_text
