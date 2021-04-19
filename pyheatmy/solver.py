from numba import njit
from numpy import ones, zeros
import numpy as np

@njit
def solver(a, b, c, d):
    nf = len(d)

    ac = a.astype(np.float32)
    bc = b.astype(np.float32)
    cc = c.astype(np.float32)
    dc = d.astype(np.float32)
    
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

@njit
def tri_product(a,b,c,d):
    n = len(d)
    res = zeros(n)
    
    ac = a.astype(np.float32)
    bc = b.astype(np.float32)
    cc = c.astype(np.float32)
    dc = d.astype(np.float32)
    
    res[0] = dc[0]*bc[0] + dc[1]*cc[0]
    res[n-1] = dc[n-1]*bc[n-1] + dc[n-2]*ac[n-2]
    
    for ix in range(1,n-1):
        res[ix] = ac[ix-1]*dc[ix-1] + bc[ix]*dc[ix] + cc[ix]*dc[ix+1]
        
    return res

#Pour forcer la compilation à l'init
solver(.1*ones(1),ones(2),-ones(1),ones(2))
tri_product(ones(1),ones(2),ones(1),ones(2))

__all__ = ["solver", "tri_product"]
import numpy as np
print(tri_product(np.array([-1, 4]),np.array([1,2,-6]),np.array([2,5]),np.array([ 1.0454545,0.47727275, -0.18181819])))