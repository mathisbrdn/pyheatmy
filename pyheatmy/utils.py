from numpy import eye, zeros, exp
from numpy.linalg import solve

LAMBDA_W = .6071
RHO_W = 1000
C_W = 4185

PARAM_LIST = (
    "moinslog10K",
    "n",
    "lambda_s",
    "rhos_cs",
)

def compute_acceptance(actual_energy: float, prev_energy: float, actual_sigma: float, prev_sigma: float):
    return min(1, (actual_sigma/prev_sigma)**-4*exp(prev_energy-actual_energy))

def compute_next_temp(param, dt, dz, temp_prev, H, H_prev, t0, tn, alpha = .7):
    N = H.size 

    rho_mc_m = param.n*RHO_W*C_W + (1-param.n)*param.rhos_cs
    K = 10**-param.moinslog10K
    lambda_m = (param.n*(LAMBDA_W)**.5+(1-param.n)*(param.lambda_s)**.5)**2

    ke = lambda_m/rho_mc_m
    ae = RHO_W*C_W*K/rho_mc_m
    dH = (H_prev[1:]-H_prev[:-1])/dz

    a = -ke/dz**2+.5*ae*dH/dz
    b = 2*ke/dz**2
    c = -ke/dz**2-.5*ae*dH/dz

    #diag = [
    #    a * (1-alpha),
    #    b * ones(N) * (1-alpha) - 1/dt,
    #    c * (1-alpha)
    #]
    #B = diags(diag, [-1, 0, 1], (N, N)).toarray()
    #B[0,0], B[0,1], B[-1,-1], B[-1,-2] = (0,0,0,0)

    B=zeros((N,N))
    for k in range(1,N-1):
        B[k,k-1] = a[k-1]*(1-alpha)
        B[k,k] = b*(1-alpha) - 1/dt
        B[k,k+1] = c[k-1]*(1-alpha)
    
    lim = B @ temp_prev
    
    lim[0], lim[-1] = t0, tn

    dH = (H[1:]-H[:-1])/dz

    a = ke/dz**2-.5*ae*dH/dz
    b = -2*ke/dz**2
    c = ke/dz**2+.5*ae*dH/dz

    #diag = [
    #    a * alpha,
    #    b * ones(N) * alpha - 1/dt,
    #    c * alpha
    #]

    #A = diags(diag, [-1, 0, 1], (N, N)).toarray()
    
    #A[0,0], A[0,1], A[-1,-1], A[-1,-2] = (1,0,1,0)
    A = eye(N)
    for k in range(1,N-1):
        A[k,k-1] = a[k-1]*alpha
        A[k,k] = b*alpha - 1/dt
        A[k,k+1] = c[k-1]*alpha

    return solve(A, lim)

def compute_next_h(K, Ss, dt, dz, H_prev, H0, Hn, alpha = .7):
    N = H_prev.size

    a = K*(1-alpha)/dz**2

    #diag = [
    #    -a * ones(N-1),
    #    (2*a-Ss/dt) * ones(N),
    #    -a * ones(N-1)
    #]
    #B = diags(diag, [-1, 0, 1], (N, N)).toarray()
    #B[0,0], B[0,1], B[-1,-1], B[-1,-2] = (0,0,0,0)
    
    B=zeros((N,N))
    for k in range(1,N-1):
        B[k,k-1] = -a
        B[k,k] = (2*a-Ss/dt)
        B[k,k+1] = -a

    lim = B @ H_prev
    lim[0], lim[-1] = H0, Hn
    
    b = K*alpha/dz**2

    #diag = [
    #    b * ones(N-1),
    #    (-2*b-Ss/dt) * ones(N),
    #    b * ones(N-1)
    #]
    
    #A = diags(diag, [-1, 0, 1], (N, N)).toarray()
    A=eye(N)
    for k in range(1,N-1):
        A[k,k-1] = b
        A[k,k] = (-2*b-Ss/dt)
        A[k,k+1] = b
    
    
    #A[0,0], A[0,1], A[-1,-1], A[-1,-2] = (1,0,1,0)

    return solve(A, lim)

__all__ = [
    "LAMBDA_W",
    "RHO_W",
    "C_W",
    "PARAM_LIST",
    "compute_acceptance",
    "compute_next_temp",
    "compute_next_h"
]
