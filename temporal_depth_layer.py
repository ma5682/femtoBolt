"""Shared, renderer-independent temporal depth/confidence intermediate layer."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class TemporalDepthLayer:
    fused_depth_mm: np.ndarray
    fused_confidence: np.ndarray
    source_count: np.ndarray
    temporal_variance: np.ndarray
    original_support_count: np.ndarray
    filled_support_count: np.ndarray
    selected_original: np.ndarray

    def validate(self):
        d,c=self.fused_depth_mm,self.fused_confidence
        arrays=(c,self.source_count,self.temporal_variance,self.original_support_count,
                self.filled_support_count,self.selected_original)
        if d.ndim!=2 or d.dtype!=np.uint16 or any(a.shape!=d.shape for a in arrays):
            raise ValueError('Temporal intermediate layer dimensions/types mismatch')
        if any(not np.isfinite(a).all() for a in arrays) or np.any((c<0)|(c>1)):
            raise ValueError('Invalid temporal values')
        if np.any((d>0)!=(c>0)) or np.any((d>0)&(self.source_count==0)):
            raise ValueError('Temporal confidence lacks depth/support')
        if np.any(self.source_count!=self.original_support_count+self.filled_support_count):
            raise ValueError('Temporal provenance support mismatch')
        if np.any(c[self.selected_original==0]>.300001) or np.any(self.temporal_variance<0):
            raise ValueError('Repaired depth promoted or invalid variance')
        return self


def load_temporal_layer(directory):
    from PIL import Image
    root=Path(directory)
    with Image.open(root/'fused_depth_mm.png') as im: depth=np.asarray(im).copy()
    names=('fused_confidence','source_count','temporal_variance','original_support_count',
           'filled_support_count','selected_original')
    return TemporalDepthLayer(depth,*(np.load(root/(name+'.npy'),allow_pickle=False) for name in names)).validate()
