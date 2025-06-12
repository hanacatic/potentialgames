from scripts import delta_experiment, epsilon_experiment, coverage_experiment

import cProfile
import sys

if __name__ == '__main__':
    
    delta_experiment(max_iter=25000, algorithm="fast_log_linear", n_exp=30)    
    delta_experiment(max_iter=25000, algorithm="fast_binary_log_linear", n_exp=30)
    delta_experiment(max_iter=25000, algorithm="log_linear", use_noisy_utility=True, n_exp=1)

    epsilon_experiment(max_iter=25000, algorithm="fast_log_linear", n_exp=30)    
    epsilon_experiment(max_iter=25000, algorithm="fast_binary_log_linear", n_exp=30)
    epsilon_experiment(max_iter=25000, algorithm="log_linear", use_noisy_utility=True, n_exp=30)

    coverage_experiment()