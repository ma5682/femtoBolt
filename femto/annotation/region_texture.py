"""Preserve per-corner source UVs without changing selected triangle positions."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh


def source_vertex_normals(vertices, faces):
    """Area weighted normals before UV seam duplication; positions unchanged."""
    normals=np.zeros_like(vertices,dtype=float)
    area=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    for corner in range(3):np.add.at(normals,faces[:,corner],area)
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-15)
    return normals


def textured_region(mesh, ids, source):
    source=Path(source);folder=source.parent
    uv_path=folder/'mesh_triangle_uv.npy';image_path=folder/'texture_multiview.png'
    if not uv_path.exists() and not image_path.exists():return None,dict(appearance='vertex_colors_without_photo_texture')
    if source.name!='scene_tsdf_mesh_m.ply':
        return None,dict(appearance='vertex_colors_without_photo_texture',reason='unsupported_texture_source_mesh')
    if not uv_path.is_file() or not image_path.is_file():raise ValueError('Incomplete source texture assets')
    uv=np.load(uv_path,allow_pickle=False)
    faces=np.asarray(mesh.triangles);vertices=np.asarray(mesh.vertices)
    if uv.shape!=(len(faces),3,2) or not np.isfinite(uv).all():
        raise ValueError('Source texture UV does not match mesh triangles')
    ids=np.asarray(sorted(set(ids)),dtype=int)
    with Image.open(image_path) as image:atlas=image.convert('RGB').copy()
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=atlas,metallicFactor=0,
                                                roughnessFactor=1,doubleSided=True)
    visual=trimesh.visual.texture.TextureVisuals(uv=uv[ids].reshape(-1,2),material=material)
    result=trimesh.Trimesh(vertices=vertices[faces[ids]].reshape(-1,3),
        faces=np.arange(len(ids)*3).reshape(-1,3),visual=visual,process=False,
        vertex_normals=source_vertex_normals(vertices,faces)[faces[ids]].reshape(-1,3))
    return result,dict(appearance='source_rgb_atlas_with_original_corner_uv',
        texture_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        uv_sha256=hashlib.sha256(uv_path.read_bytes()).hexdigest(),
        source_triangle_indices=ids.tolist(),texture_embedded=True,
        normal_policy='area_weighted_source_topology_normals_preserved_across_uv_seams',
        limitations='Original texture seams and fallback colors remain; appearance is not geometric accuracy.')
