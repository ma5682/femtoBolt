"""Circular brush swept in original reference pixels, including fast drags."""
import math
import numpy as np


def stroke(mask, start, end, size, erase=False):
    a, b = np.asarray(start, float), np.asarray(end, float)
    radius = max(1., float(size))/2
    left, top = np.maximum(0, np.floor(np.minimum(a, b)-radius)).astype(int)
    right = min(mask.shape[1], math.ceil(max(a[0], b[0])+radius)+1)
    bottom = min(mask.shape[0], math.ceil(max(a[1], b[1])+radius)+1)
    delta = b-a
    length2 = float(delta @ delta)
    # Row batches bound scratch memory even for long diagonal strokes.
    for y in range(top, bottom, 64):
        yy, xx = np.mgrid[y:min(y+64, bottom), left:right]
        t = np.clip(((xx-a[0])*delta[0]+(yy-a[1])*delta[1])/length2, 0, 1) if length2 else 0.
        inside = (xx-a[0]-t*delta[0])**2+(yy-a[1]-t*delta[1])**2 <= radius**2
        mask[y:y+len(yy), left:right][inside] = 0 if erase else 1
