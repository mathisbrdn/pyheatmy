from numpy import ones, float32
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

#@njit()
def compute_next_temp(param, dt, dz, temp_prev, H, H_prev, t0, tn, alpha = .7):
    N = H_prev.size
    H = H.astype(float32)
    H_prev = H_prev.astype(float32)
    temp_prev = temp_prev.astype(float32)

    rho_mc_m = param.n*RHO_W*C_W + (1-param.n)*param.rhos_cs
    K = 10**-param.moinslog10K
    lambda_m = (param.n*(LAMBDA_W)**.5 + (1.-param.n)*(param.lambda_s)**.5)**2

    ke = lambda_m/rho_mc_m
    ae = RHO_W*C_W*K/rho_mc_m
    dH = (H_prev[1:]-H_prev[:-1])/dz

    a = (-ke/dz**2 + .5*ae*dH/dz) * (1-alpha)
    b = 2*ke/dz**2 * ones(N, dtype = float32) * (1-alpha) - 1/dt
    c = (-ke/dz**2 - .5*ae*dH/dz) * (1-alpha)

    lim = tri_product(a, b, c, temp_prev)
    lim[0], lim[-1] = t0, tn

    dH = (H[1:]-H[:-1])/dz

    a = (ke/dz**2-.5*ae*dH/dz) * alpha
    a[-1] = 0.
    b = -2*ke/dz**2 * ones(N, dtype = float32) * alpha - 1/dt
    b[0], b[-1] = 1., 1.
    c = (ke/dz**2+.5*ae*dH/dz) * alpha
    c[0] = 0.
    
    return solver(a, b, c, lim)

@njit
def compute_next_h(K, Ss, dt, dz, H_prev, H0, Hn, alpha = .7):
    N = H_prev.size
    H_prev = H_prev.astype(float32)

    k = K*(1-alpha)/dz**2

    a = -k * ones(N-1, dtype = float32)
    b = (2*k-Ss/dt) * ones(N, dtype = float32)
    c = -k * ones(N-1, dtype = float32)

    lim = tri_product(a, b, c, H_prev)

    lim[0], lim[-1] = H0, Hn

    k = K*alpha/dz**2

    a = k * ones(N-1, dtype = float32)
    a[-1] = 0
    b = (-2*k-Ss/dt) * ones(N, dtype = float32)
    b[0], b[-1] = 1, 1
    c = k * ones(N-1, dtype = float32)
    c[0] = 0

    return solver(a, b, c, lim)


__all__ = [
    "LAMBDA_W",
    "RHO_W",
    "C_W",
    "PARAM_LIST",
    "compute_next_temp",
    "compute_next_h"
]
