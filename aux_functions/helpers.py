import numpy as np
from itertools import permutations
from scipy.sparse import csc_matrix
# def metropolis_hastings(prob_func, initial_profile, iterations = 100):
# in this implementation it is not working because the application is discrete
    
#     current_profile = initial_profile
    
#     for i in range(iterations):
#         proposed_profile = proposal_sampler(current_profile)
        
#         acceptance_ratio = prob_func(proposed_profile)/prob_func(current_profile)
        
#         if np.random.uniform(0, 1) < acceptance_ratio:
#             return proposed_profile
        
#     return -1

# def proposal_sampler(action_profile):
#     return action_profile + np.random.normal(0, 0.1, size = action_profile.shape)

def rejection_sampling(prob_func, initial_profile, action_space_size, M = 1.0, iterations = 1000):
    
    current_profile = initial_profile
    
    for _ in range(iterations):
        proposed_profile = proposal_sampler(current_profile, action_space_size)
        
        acceptance_ratio = prob_func(proposed_profile)/(M*prob_func(current_profile))
        
        if np.random.uniform(0, 1) < acceptance_ratio:
            return proposed_profile
        
    return -1

def proposal_sampler(action_profile, action_space_size):
    return np.random.randint(0, action_space_size, size = action_profile.shape)

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

def sparse_col_vec_dot(csc_mat, csc_vec):
    # row numbers of vector non-zero entries
    v_rows = csc_vec.indices
    v_data = csc_vec.data
    # matrix description arrays
    m_dat = csc_mat.data
    m_ind = csc_mat.indices
    m_ptr = csc_mat.indptr
    # output arrays
    sizes = m_ptr.take(v_rows+1) - m_ptr.take(v_rows)
    sizes = np.concatenate(([0], np.cumsum(sizes)))
    data = np.empty((sizes[-1],), dtype=csc_mat.dtype)
    indices = np.empty((sizes[-1],), dtype=np.intp)
    indptr = np.zeros((2,), dtype=np.intp)

    for j in range(len(sizes)-1):
        slice_ = slice(*m_ptr[[v_rows[j] ,v_rows[j]+1]])
        np.multiply(m_dat[slice_], v_data[j], out=data[sizes[j]:sizes[j+1]])
        indices[sizes[j]:sizes[j+1]] = m_ind[slice_]
    indptr[-1] = len(data)
    
    ret = csc_matrix((data, indices, indptr),
                         shape=csc_vec.shape)
    ret.sum_duplicates()

    return ret