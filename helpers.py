import numpy as np
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
