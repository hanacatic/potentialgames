import numpy as np
from itertools import permutations

rng = np.random.default_rng(seed = 2)


def rejection_sampling(prob_func, initial_profile, action_space, M = 1.0, iterations = 1000):
    
    current_profile = initial_profile
    
    for _ in range(iterations):
        proposed_profile = proposal_sampler(current_profile, action_space)
        
        acceptance_ratio = prob_func(proposed_profile)/(M*prob_func(current_profile))
        
        if np.random.uniform(0, 1) < acceptance_ratio:
            return proposed_profile
        
    return -1

def proposal_sampler(action_profile, action_space):
    return [np.random.randint(0, len(action_space[i]), size = 1) for i in range(len(action_space))]

def make_symmetric_nd(matrix):
    # Get the number of dimensions
    n_dims = matrix.ndim
    
    # Generate all possible permutations of the axis indices
    permutations_axes = list(permutations(range(n_dims)))
    
    # Initialize an accumulator with zeros
    symmetric_matrix = np.zeros_like(matrix, dtype=np.float64)
    
    # Sum up the matrix with all axis-permuted versions
    for perm in permutations_axes:
        symmetric_matrix += np.transpose(matrix, axes=perm)
    
    # Average by dividing by the number of permutations
    symmetric_matrix /= len(permutations_axes)
    
    return symmetric_matrix
