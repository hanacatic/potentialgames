import numpy as np

"""Utility functions for computing convergence parameters like beta and T based on the theoretical bounds from the paper Maddux et al (2026)."""


def compute_beta(A: int, N: int, delta: float, epsilon: float, symmetric: bool = False, use_noisy_utility: bool = False):
    """
    Compute the beta temperature parameter.

    Args:
        A (int): Number of actions per player.
        N (int): Number of players.
        delta (float): Suboptimality gap.
        epsilon (float): Tolerance level for convergence.
        symmetric (bool, optional): Whether the game is symmetric. Defaults to False.
        use_noisy_utility (bool, optional): Whether noisy utility is used. Defaults to False.

    Returns:
        float: The computed beta value.
    """
    
    if symmetric:
        return 1 / delta * (A * np.log(N) + np.log(2-epsilon) - np.log(epsilon))
    
    return 1 / delta * (N * np.log(A) + np.log(2-epsilon) - np.log(epsilon))

def compute_t(A: int, N: int, delta: float, epsilon: float, beta: float, symmetric:bool=False, use_noisy_utility: bool = False):
    """
    Computes the convergence guarantee.

    Args:
        A (int): Number of actions per player.
        N (int): Number of players.
        delta (float): Suboptimality gap.
        epsilon (float): Tolerance level for convergence.
        beta (float): Beta temperature parameter.
        symmetric (bool, optional): Whether the game is symmetric. Defaults to False.
        use_noisy_utility (bool, optional): Whether noisy utility is used. Defaults to False.

    Returns:
        float: The computed number of iterations T for convergence.
    """
    
    if use_noisy_utility:    
        return np.log(N**1.5*A**3) + N + beta*(1+1/beta)*(N+3) * np.log(-2*np.log(epsilon))
    
    return np.log(N**2*A**5) + (1/max(epsilon, delta))*N*np.log(A/epsilon)