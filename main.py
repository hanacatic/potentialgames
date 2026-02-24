from scripts import (delta_experiment, epsilon_experiment, full_feedback_comparison, 
                     reduced_feedback_comparison, delta_estimation_experiment)

from potentialgames.utils import (visualise_delta_experiment, visualise_epsilon_experiment,
                                  visualise_full_feedback_comparison, visualise_reduced_feedback_comparison,
                                  visualise_delta_estimation_experiment, visualise_experiment)

import sys
import multiprocessing as mp

if __name__ == '__main__':
    
    mp.freeze_support()

    delta_experiment(max_iter=2000, algorithm="fast_log_linear", n_exp=30)    
    # delta_experiment(max_iter=2000, algorithm="fast_binary_log_linear", n_exp=30)
    # delta_experiment(max_iter=1500, algorithm="log_linear", use_noisy_utility=True, n_exp=30)

    # epsilon_experiment(max_iter=7000, algorithm="fast_log_linear", n_exp=30, n_processes=4)  
    # epsilon_experiment(max_iter=7000, algorithm="fast_binary_log_linear", n_exp=30, n_processes=4)
    # epsilon_experiment(max_iter=7000, algorithm="log_linear", use_noisy_utility=True, n_exp=30, n_processes=4)

    
    # full_feedback_comparison(algorithm="hedge", n_exp=30, max_iter=2000, verbose=True, n_processes=4)
    # full_feedback_comparison(algorithm="fast_log_linear", n_exp=30, max_iter=2000, verbose=True, n_processes=4)

    # reduced_feedback_comparison(algorithm="fast_binary_log_linear", n_exp=30, max_iter=5000, verbose=True, n_processes=4)
    # reduced_feedback_comparison(algorithm="exp3p", n_exp=30, max_iter=5000, verbose=True, n_processes=4)
    # reduced_feedback_comparison(algorithm="exponential_weight_with_annealing", n_exp=30, max_iter=5000, verbose=True, n_processes=4)
    
    # delta_estimation_experiment(algorithm="fast_log_linear", n_exp=30, max_iter=2000, verbose=True, easy_game=True, n_processes=4)
    # delta_estimation_experiment(algorithm="fast_log_linear", n_exp=30, max_iter=2000, verbose=True, n_processes=4)

    # visualise_delta_experiment("fast_binary_log_linear")
    # visualise_delta_experiment("fast_log_linear")
    # visualise_delta_experiment("log_linear_noisy")
    
    # visualise_epsilon_experiment("fast_log_linear", iter=4500)
    # visualise_epsilon_experiment("fast_binary_log_linear", iter=4500)
    # visualise_epsilon_experiment("log_linear_noisy", iter=4500)

    # visualise_full_feedback_comparison(iter=6000)
    # visualise_reduced_feedback_comparison(iter=5000)
    
    # visualise_delta_estimation_experiment("fast_log_linear", iter = 400)
    # visualise_delta_estimation_experiment("easy_fast_log_linear", iter = 400)
    
    # visualise_experiment("delta", "fast_log_linear")
    # visualise_experiment("delta", "fast_binary_log_linear")
    # visualise_experiment("delta", "log_linear_noisy")
    
    # visualise_experiment("epsilon", "fast_log_linear")
    # visualise_experiment("epsilon", "fast_binary_log_linear")
    # visualise_experiment("epsilon", "log_linear_noisy")
    
    # visualise_experiment("full_feedback_comparison", "comparison")
    # visualise_experiment("reduced_feedback_comparison", "comparison")
    
    # visualise_experiment("delta_est", "fast_log_linear")
    # visualise_experiment("delta_est", "easy_fast_log_linear")
    