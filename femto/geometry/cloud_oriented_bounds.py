"""Experimental directed-rounding envelopes; not enabled in the default chain.

Centers and basis vectors are arbitrary *stored* finite floats, not fitted
geometry. Bounds cover every observed point in a cell. Every elementary bound
operation rounds outward. The final allowance bounds the original three
subtractions, products and two additions, using monotone ULP upper bounds.
"""
import numpy as np


def down(value):return np.nextafter(value,-np.inf)
def up(value):return np.nextafter(value,np.inf)


def dot_interval(left,right,coefficient):
    """Enclose exact (left-right) dot coefficient on stored float inputs."""
    lo=down(left-right);hi=up(left-right)
    a=lo*coefficient;b=hi*coefficient
    low=down(np.minimum(a,b));high=up(np.maximum(a,b))
    return down(down(low[...,0]+low[...,1])+low[...,2]),up(up(high[...,0]+high[...,1])+high[...,2])


def positive_sum(value):
    return up(up(value[...,0]+value[...,1])+value[...,2])


def ulp_upper(magnitude):
    # One ULP (not half an ULP) bounds correctly rounded operations throughout
    # [-magnitude,magnitude], including gradual underflow and binade endpoints.
    return up(magnitude)-magnitude


class OrientedBounds:
    def __init__(self,points,normals,inverse,low,high,check=lambda:None):
        cells=len(low);counts=np.bincount(inverse,minlength=cells)
        self.low,self.high=low,high
        # The midpoint/average only chooses a coordinate system. Their rounding
        # is immaterial to the identity once these stored floats are fixed.
        with np.errstate(all='ignore'):
            self.center=low+(high-low)*.5
            self.basis=np.column_stack([np.bincount(inverse,weights=normals[:,k],minlength=cells)/counts for k in range(3)])
            self.delta=np.maximum(np.abs(down(low-self.center)),np.abs(up(high-self.center)))
            self.hlow=np.full(cells,np.inf);self.hhigh=np.full(cells,-np.inf)
            for start in range(0,len(points),8192):
                check();ids=inverse[start:start+8192]
                lower,upper=dot_interval(points[start:start+8192],self.center[ids],self.basis[ids])
                np.minimum.at(self.hlow,ids,lower);np.maximum.at(self.hhigh,ids,upper)

    def negative(self,p,n,ids,height):
        with np.errstate(all='ignore'):
            center=self.center[ids];basis=self.basis[ids]
            difference=np.maximum(np.abs(down(n-basis)),np.abs(up(n-basis)))
            remainder=positive_sum(up(self.delta[ids]*difference))
            lower,upper=dot_interval(center,p,n)
            lower=down(down(self.hlow[ids]+lower)-remainder)
            upper=up(up(self.hhigh[ids]+upper)+remainder)
            maximum=np.maximum(np.abs(lower),np.abs(upper))
            # Bound errors in the *legacy* floating dot, separately from the
            # exact real interval above. A float64 dot of 3 components performs
            # at most 3 products + 2 sums (FMA only reduces that error bound).
            delta=np.maximum(np.abs(down(self.low[ids]-p)),np.abs(up(self.high[ids]-p)))
            subtraction_error=ulp_upper(delta)
            magnitude=up(up(delta+subtraction_error)*np.abs(n))
            product_error=ulp_upper(magnitude)
            sum_magnitude=positive_sum(up(magnitude+product_error))
            error=up(positive_sum(up(subtraction_error*np.abs(n)))+positive_sum(product_error))
            error=up(error+up(2*ulp_upper(sum_magnitude)))
            bound=up(maximum+error)
        return np.isfinite(bound)&(bound<=height)
