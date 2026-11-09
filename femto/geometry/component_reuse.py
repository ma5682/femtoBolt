"""Operation-owned component counts bound to the actual post-filter mesh."""
from hashlib import sha256
from threading import get_ident
import numpy as np
from femto.performance_detail import observed, count


def _fingerprint(mesh):
    digest=sha256()
    for value in (mesh.vertices,mesh.triangles):
        array=np.asarray(value)
        digest.update(str((array.shape,array.dtype.str)).encode('ascii'))
        if array.size:digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return digest.digest()


class ComponentReuse:
    """Only the component filter populates this transient receipt.

    Identity limits the lifetime; the full content digest detects mutations
    through Open3D and NumPy aliases. Counts are never accepted from a report.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self._mesh=None
        self._digest=None
        self._counts=()
        self._thread=None

    @observed('mesh.component_bind')
    def _remember_filtered(self,mesh,counts):
        self.clear()
        try:
            values=tuple(int(value) for value in counts)
            if sum(values)!=len(mesh.triangles) or any(value<=0 for value in values):return
            digest=_fingerprint(mesh)
        except MemoryError:
            return
        self._mesh,self._digest,self._counts,self._thread=mesh,digest,values,get_ident()

    @observed('mesh.component_match')
    def matching_counts(self,mesh):
        if mesh is not self._mesh or self._thread!=get_ident():return None
        try:
            matches=_fingerprint(mesh)==self._digest
        except MemoryError:
            matches=False
        if not matches:
            self.clear()
            count('mesh.component_reuse',changed=1)
            return None
        count('mesh.component_reuse',hits=1)
        return self._counts
