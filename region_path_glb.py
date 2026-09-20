"""Portable GLB: selected source mesh plus separate yellow path display mesh."""
import hashlib
import numpy as np
import trimesh
from scene_annotations import region_mesh


def path_tubes(plan, radius=.00015):
    vertices=[];faces=[];colors=[];offset=0
    # Four-sided tubes are display geometry only; no segment-to-segment joins.
    for segment in plan['segments']:
        xyz=np.asarray([p['xyz_m'] for p in segment]);a,b=xyz[:-1],xyz[1:]
        delta=b-a;length=np.linalg.norm(delta,axis=1);keep=length>1e-12
        edge_colors=np.asarray([[255,145,56,255] if sample.get('motion_kind')=='connector' else [255,215,30,255] for sample in segment[1:]],dtype=np.uint8)[keep]
        colors.append(np.repeat(edge_colors,8,axis=0))
        a,b,delta,length=a[keep],b[keep],delta[keep],length[keep]
        if not len(a):continue
        direction=delta/length[:,None]
        axis=np.tile([0.,0.,1.],(len(a),1));axis[np.abs(direction[:,2])>.9]=[1.,0.,0.]
        u=np.cross(direction,axis);u/=np.linalg.norm(u,axis=1)[:,None]
        v=np.cross(direction,u)
        ring=np.stack([u,v,-u,-v],axis=1)*radius
        points=np.concatenate([a[:,None,:]+ring,b[:,None,:]+ring],axis=1)
        template=[]
        for i in range(4):
            j=(i+1)%4;template.extend([[i,j,4+i],[j,4+j,4+i]])
        template.extend([[0,2,1],[0,3,2],[4,5,6],[4,6,7]])
        triangles=np.asarray(template)[None,:,:]+np.arange(len(a))[:,None,None]*8+offset
        vertices.append(points.reshape(-1,3));faces.append(triangles.reshape(-1,3));offset+=len(a)*8
    if not vertices:raise ValueError('No path display geometry')
    mesh=trimesh.Trimesh(vertices=np.concatenate(vertices),faces=np.concatenate(faces),process=False)
    mesh.visual.vertex_colors=np.concatenate(colors)
    return mesh


def export_scene(folder,mesh,region,plan,source=None):
    allowed=plan.get('exclusions',{}).get('allowed_triangle_indices',region['triangle_indices'])
    selected=region_mesh(mesh,allowed)
    colors=(np.asarray(selected.vertex_colors)*255).astype(np.uint8) if selected.has_vertex_colors() else None
    surface=trimesh.Trimesh(vertices=np.asarray(selected.vertices),faces=np.asarray(selected.triangles),
                            vertex_colors=colors,process=False)
    metadata=dict(appearance='vertex_colors_without_photo_texture')
    if source is not None:
        from region_texture import textured_region
        textured,metadata=textured_region(mesh,allowed,source)
        if textured is not None:surface=textured
    transform=np.diag([1.,-1.,-1.,1.]);scene=trimesh.Scene()
    scene.add_geometry(surface,node_name='selected_region_W0',geom_name='selected_region',transform=transform)
    excluded=plan.get('exclusions',{}).get('removed_triangle_indices',[])
    if excluded:
        blocked=region_mesh(mesh,excluded)
        display=trimesh.Trimesh(vertices=np.asarray(blocked.vertices),faces=np.asarray(blocked.triangles),process=False)
        display.visual.vertex_colors=[225,45,55,255]
        scene.add_geometry(display,node_name='forbidden_region_W0',geom_name='forbidden_region',transform=transform)
    scene.add_geometry(path_tubes(plan),node_name='path_display_W0',geom_name='path_display',transform=transform)
    for name,marker in direction_markers(plan).items():
        scene.add_geometry(marker,node_name=name+'_W0',geom_name=name,transform=transform)
    from route_motion import statistics
    scene.metadata.update(route_schema='femto-static-route-v2',static_route=True,animation_included=False,
        plan_binding=plan.get('plan_binding'),route_order=plan.get('route_order'),
        start_selection=plan.get('start_selection'),chain_breaks=plan.get('chain_breaks',[]),
        statistics=statistics(plan),chains=[dict(index=i,sample_count=len(c),start_xyz_m=c[0]['xyz_m'],
            end_xyz_m=c[-1]['xyz_m'],edge_motion_kinds=[p.get('motion_kind','scan') for p in c[1:]]) for i,c in enumerate(plan['segments'])])
    if plan.get('surface_mode'):
        scene.metadata.update(surface_mode=plan['surface_mode'],spacing_semantics=plan['spacing_semantics'],
            spacing_report_status=plan['spacing_report']['status'],coverage_parameters=plan['coverage_parameters'])
    data=scene.export(file_type='glb');name='region_with_path.glb';(folder/name).write_bytes(data)
    return dict(file=name,sha256=hashlib.sha256(data).hexdigest(),unit='metre',
        glb_scene_from_W0=transform.tolist(),W0_from_glb_scene=transform.tolist(),
        path_display_radius_m=.00015,path_display_semantics='yellow_scan_orange_connector_display_only',static_route=True,animation_included=False,
        region_semantics='selected_source_triangles',texture_metadata=metadata,
        markers='green_start_red_end_yellow_direction_per_segment_display_only')


def direction_markers(plan):
    groups={'segment_starts':[],'segment_ends':[],'segment_directions':[]}
    for segment in plan['segments']:
        xyz=np.asarray([sample['xyz_m'] for sample in segment])
        for name,point in [('segment_starts',xyz[0]),('segment_ends',xyz[-1])]:
            ball=trimesh.creation.icosphere(subdivisions=0,radius=.0006)
            ball.apply_translation(point);groups[name].append(ball)
        delta=np.diff(xyz,axis=0);length=np.linalg.norm(delta,axis=1)
        if length.sum()>1e-12:
            index=min(int(np.searchsorted(np.cumsum(length),length.sum()/2)),len(length)-1)
            if length[index]>1e-12:
                arrow=trimesh.creation.cone(radius=.0005,height=.0015,sections=6)
                rotation=trimesh.geometry.align_vectors([0,0,1],delta[index]/length[index])
                arrow.apply_transform(rotation);arrow.apply_translation((xyz[index]+xyz[index+1])/2)
                groups['segment_directions'].append(arrow)
    colors={'segment_starts':[40,230,110,255],'segment_ends':[255,70,80,255],'segment_directions':[255,215,30,255]}
    result={}
    for name,meshes in groups.items():
        if meshes:
            combined=trimesh.util.concatenate(meshes);combined.visual.vertex_colors=colors[name];result[name]=combined
    return result
