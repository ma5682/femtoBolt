"""Isolated, read-only preview of exported mesh assets."""
import sys
from pathlib import Path


def ensure_output_streams():
    """pythonw has no console streams; native viewer callbacks may print."""
    if sys.stdout is not None and sys.stderr is not None:return None
    import os
    try:
        folder=Path(__file__).parent/'Log';folder.mkdir(exist_ok=True)
        stream=(folder/'export_preview.log').open('a',encoding='utf-8',buffering=1)
    except OSError:
        stream=open(os.devnull,'w',encoding='utf-8')
    if sys.stdout is None:sys.stdout=stream
    if sys.stderr is None:sys.stderr=stream
    return stream


if __name__=='__main__':ensure_output_streams()

import numpy as np
import open3d as o3d
import trimesh


def load_preview(path):
    path=Path(path)
    if path.suffix.lower() not in ('.glb','.ply'):raise ValueError('Preview supports GLB / PLY only')
    scene=trimesh.load(path,force='scene',process=False);result=[]
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node];mesh=scene.geometry[name]
        if not isinstance(mesh,trimesh.Trimesh):continue
        vertices=trimesh.transform_points(mesh.vertices,transform)
        if path.suffix.lower()=='.ply':vertices=vertices*np.array([1.,-1.,-1.])
        geometry=o3d.geometry.TriangleMesh()
        geometry.vertices=o3d.utility.Vector3dVector(vertices)
        geometry.triangles=o3d.utility.Vector3iVector(np.asarray(mesh.faces))
        if mesh.visual.kind=='texture':
            material=mesh.visual.material
            texture=getattr(material,'baseColorTexture',None)
            if texture is None:texture=getattr(material,'image',None)
            uv=getattr(mesh.visual,'uv',None)
            if texture is None or uv is None:raise ValueError('Incomplete GLB texture material')
            geometry.triangle_uvs=o3d.utility.Vector2dVector(np.asarray(uv)[mesh.faces].reshape(-1,2))
            # Trimesh UVs use bottom-left origin; legacy Open3D uploads image
            # rows directly. Flip the display image, not the exported UVs.
            geometry.textures=[o3d.geometry.Image(np.asarray(texture.convert('RGB'))[::-1].copy())]
            geometry.triangle_material_ids=o3d.utility.IntVector(np.zeros(len(mesh.faces),dtype=np.int32))
        elif hasattr(mesh.visual,'vertex_colors'):
            geometry.vertex_colors=o3d.utility.Vector3dVector(np.asarray(mesh.visual.vertex_colors)[:,:3]/255.)
        # Preserve exported smooth normals across duplicated UV vertices.
        normal_transform=np.linalg.inv(transform[:3,:3]).T
        normals=np.asarray(mesh.vertex_normals) @ normal_transform.T
        if path.suffix.lower()=='.ply':normals*=np.array([1.,-1.,-1.])
        normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-15)
        geometry.vertex_normals=o3d.utility.Vector3dVector(normals)
        geometry.compute_triangle_normals();result.append(geometry)
    if not result:raise ValueError('No mesh geometry in export')
    return result


if __name__=='__main__':
    viewer=None
    try:
        geometries=load_preview(sys.argv[1])
        viewer=o3d.visualization.Visualizer()
        if not viewer.create_window(window_name='Export preview · '+Path(sys.argv[1]).name,width=1100,height=760):
            raise RuntimeError('Cannot create Open3D preview window')
        for geometry in geometries:viewer.add_geometry(geometry)
        viewer.get_render_option().mesh_show_back_face=True
        viewer.get_render_option().light_on=False
        viewer.run()
    except Exception as error:
        import traceback
        traceback.print_exc()
        import tkinter as tk
        from tkinter import messagebox
        root=tk.Tk();root.withdraw();messagebox.showerror('导出预览失败',str(error));root.destroy()
    finally:
        if viewer is not None:viewer.destroy_window()
