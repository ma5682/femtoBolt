"""Graph-owned immutable residual preparation, never cached edge decisions."""
import numpy as np
from femto.performance_detail import count

MAX_SNAPSHOT_BYTES=16*1024*1024
TILE=16


class ResidualTileSnapshot:
    def __init__(self,values):
        self.shape=values.shape
        self.dtype=values.dtype
        # bytes owns the storage: neither setflags nor an existing mutable view
        # can change this snapshot. A read-only flag on the caller is not used.
        self.values=np.frombuffer(values.tobytes(order='C'),dtype=values.dtype).reshape(values.shape)
        self._tiles=None

    def get(self,values):
        if (values is not self.values or values.shape!=self.shape
                or values.dtype!=self.dtype or not values.flags.c_contiguous):return None
        if self._tiles is None:
            rows=np.maximum.reduceat(values,np.arange(0,self.shape[0],TILE),axis=0)
            tiles=np.maximum.reduceat(rows,np.arange(0,self.shape[1],TILE),axis=1)
            self._tiles=np.frombuffer(tiles.tobytes(order='C'),dtype=tiles.dtype).reshape(tiles.shape)
            count('route.residual_work',tile_builds=1,tile_input_values=values.size)
        else:
            count('route.residual_work',shared_tile_hits=1)
        return self._tiles


def bind(reference):
    values=reference.residual
    if type(values) is not np.ndarray or values.dtype!=np.dtype('int32') or values.nbytes>MAX_SNAPSHOT_BYTES:return
    if reference._shared_residual_tiles is not None and reference._shared_residual_tiles.values is values:return
    try:snapshot=ResidualTileSnapshot(values)
    except MemoryError:return
    reference.residual=snapshot.values
    reference._shared_residual_tiles=snapshot


def bind_task(reference):
    """Reuse immutable values/tiles, never the per-envelope logical budget."""
    from femto.planning.task_preparation import array_key, lookup, remember
    values=reference.residual
    if type(values) is not np.ndarray or values.dtype!=np.dtype('int32') or values.nbytes>MAX_SNAPSHOT_BYTES:return
    fingerprint=array_key(values)
    if fingerprint is None:return
    key=(fingerprint,TILE,MAX_SNAPSHOT_BYTES)
    snapshot=lookup('residual_tiles',key)
    if snapshot is None:
        try:snapshot=ResidualTileSnapshot(values)
        except MemoryError:return
        # Include the lazy tile allocation in the retained-memory budget.
        size=values.nbytes+((values.shape[0]+TILE-1)//TILE)*((values.shape[1]+TILE-1)//TILE)*values.dtype.itemsize
        remember('residual_tiles',key,snapshot,size)
    reference.residual=snapshot.values
    reference._shared_residual_tiles=snapshot
