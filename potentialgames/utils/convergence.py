import numpy as np

def compute_beta(A: int, N: int, delta: float, epsilon: float, symmetric: bool = False, use_noisy_utility: bool = False):
    if symmetric:
        return 1 / max(epsilon, delta) * (A * np.log(N) - np.log(epsilon))
    return 1 / max(epsilon, delta) * (N * np.log(A) - np.log(epsilon))

def compute_t(A: int, N: int, delta: float, epsilon: float, beta: float, symmetric:bool=False, use_noisy_utility: bool = False):
    if use_noisy_utility:
        return np.log(N**1.5*A**3) + N + beta*(1+1/beta)*(N+3) * np.log(-2*np.log(epsilon))
    return np.log(N**2*A**5) + (1/max(epsilon, delta))*N*np.log(A/epsilon)