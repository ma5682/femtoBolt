"""Display-only tangent ribbons in metres, never used as coverage evidence.

Each accepted scan chord gets a transverse half-width in its source face plane.
At curved faces the ribbons may overlap or leave display seams. They are not
clipped to the selection (which would conceal apparent overflow), and do not
assert surface contact or confirmed coverage. Connectors carry no effect claim.
"""
import numpy as np


def ribbons(plan, mesh, intrinsic):
    width=plan['treatment_head']['effective_treatment_width_mm']/1000
    vertices=np.asarray(mesh.vertices)
    faces=np.asarray(mesh.faces)
    k=np.asarray(intrinsic,dtype=float)
    result=[]
    for chain in plan['segments']:
        for left,right in zip(chain,chain[1:]):
            if right.get('motion_kind',right.get('role','scan'))!='scan':continue
            a,b=np.asarray(left['xyz_m']),np.asarray(right['xyz_m'])
            tri=vertices[faces[left['triangle_index']]]
            normal=np.cross(tri[1]-tri[0],tri[2]-tri[0])
            side=np.cross(normal,b-a);length=np.linalg.norm(side)
            if not np.isfinite(length) or length<=1e-15:continue
            side=side/length*(width/2)
            quad=np.array([a+side,b+side,b-side,a-side])
            projected=quad@k.T
            if not np.isfinite(projected).all() or np.any(projected[:,2]<=0):continue
            result.append(projected[:,:2]/projected[:,2,None])
    return result
