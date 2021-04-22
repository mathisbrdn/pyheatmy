from numpy import float32, full
from numba import njit

from .solver import solver, tri_product

LAMBDA_W = .6071
RHO_W = 1000
C_W = 4185

PARAM_LIST = (
    "moinslog10K",
    "n",
    "lambda_s",
    "rhos_cs",
)

@njit()
def compute_next_temp(moinslog10K, n, lambda_s, rhos_cs, dt, dz, temp_prev, H, H_prev, t0, tn, alpha = .7):
    N = H_prev.size
    H = H.astype(float32)
    H_prev = H_prev.astype(float32)
    temp_prev = temp_prev.astype(float32)

    rho_mc_m = n*RHO_W*C_W + (1-n)*rhos_cs
    K = 10.**-moinslog10K
    lambda_m = (n*(LAMBDA_W)**.5 + (1.-n)*(lambda_s)**.5)**2

    ke = lambda_m/rho_mc_m
    ae = RHO_W*C_W*K/rho_mc_m
    dH = (H_prev[1:]-H_prev[:-1])/dz

    a = (-ke/dz**2 + dH*(.5*ae/dz)) * (1-alpha)
    b = full(N, (1-alpha)*2*ke/dz**2 - 1/dt, dtype = float32)
    c = (-ke/dz**2 - dH*(.5*ae/dz)) * (1-alpha)

    lim = tri_product(a, b, c, temp_prev)
    lim[0], lim[-1] = t0, tn

    dH = (H[1:]-H[:-1])/dz

    a = (ke/dz**2-dH*(.5*ae/dz)) * alpha
    a[-1] = 0.
    b = full(N, -alpha*2*ke/dz**2 - 1/dt, dtype = float32)
    b[0], b[-1] = 1., 1.
    c = (ke/dz**2+dH*(.5*ae/dz)) * alpha
    c[0] = 0.

    return solver(a, b, c, lim)

@njit
def compute_next_h(K, Ss, dt, dz, H_prev, H0, Hn, alpha = .7):
    N = H_prev.size
    H_prev = H_prev.astype(float32)

    k = K*(1-alpha)/dz**2

    a = full(N-1, -k ,dtype = float32)
    b = full(N, 2*k-Ss/dt,dtype = float32)
    c = full(N-1, -k ,dtype = float32)

    lim = tri_product(a, b, c, H_prev)

    lim[0], lim[-1] = H0, Hn

    k = K*alpha/dz**2

    a = full(N-1, k ,dtype = float32)
    a[-1] = 0
    b = full(N, -2*k-Ss/dt, dtype = float32)
    b[0], b[-1] = 1, 1
    c = full(N-1, k ,dtype = float32)
    c[0] = 0

    return solver(a, b, c, lim)
