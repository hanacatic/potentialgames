from scripts import (delta_experiment, epsilon_experiment, full_feedback_comparison, 
                     reduced_feedback_comparison, delta_estimation_experiment)

from potentialgames.utils import (visualise_delta_experiment, visualise_epsilon_experiment,
                                  visualise_full_feedback_comparison, visualise_reduced_feedback_comparison,
                                  visualise_delta_estimation_experiment, visualise_experiment)

import sys
import multiprocessing as mp


if __name__ == '__main__':
    
    mp.freeze_support()

    # delta_experiment(max_iter=2000, algorithm="fast_log_linear", n_exp=30, results_folder="results")    
    # delta_experiment(max_iter=2000, algorithm="fast_binary_log_linear", n_exp=30, results_folder="results")
    # delta_experiment(max_iter=1500, algorithm="log_linear", use_noisy_utility=True, n_exp=30, results_folder="results")

    # epsilon_experiment(max_iter=7000, algorithm="fast_log_linear", n_exp=30, n_processes=4, results_folder="results")  
    # epsilon_experiment(max_iter=7000, algorithm="fast_binary_log_linear", n_exp=30, n_processes=4, results_folder="results")
    # epsilon_experiment(max_iter=7000, algorithm="log_linear", use_noisy_utility=True, n_exp=30, n_processes=4, results_folder="results")

    # full_feedback_comparison(algorithm="hedge", n_exp=30, max_iter=2000, verbose=True, n_processes=4, results_folder="results")  
    # full_feedback_comparison(algorithm="fast_log_linear", n_exp=30, max_iter=2000, verbose=True, n_processes=4, results_folder="results")

    # reduced_feedback_comparison(algorithm="fast_binary_log_linear", n_exp=30, max_iter=5000, verbose=True, results_folder="results")  
    # reduced_feedback_comparison(algorithm="exp3p", n_exp=30, max_iter=5000, verbose=True, n_processes=4, results_folder="results")
    # reduced_feedback_comparison(algorithm="exponential_weight_with_annealing", n_exp=30, max_iter=5000, verbose=True, n_processes=4, results_folder="results")
    
    # delta_estimation_experiment(algorithm="fast_log_linear", n_exp=30, max_iter=2000, verbose=True, easy_game=True, results_folder="results")  
    # delta_estimation_experiment(algorithm="fast_log_linear", n_exp=30, max_iter=2000, verbose=True, n_processes=4, results_folder="results")

    # visualise_delta_experiment("fast_log_linear", iter=2000, data_folder="results", save_folder="results")
    # visualise_delta_experiment("fast_binary_log_linear", iter=2000, data_folder="results", save_folder="results")
    # visualise_delta_experiment("log_linear_noisy", iter=2000, data_folder="results", save_folder="results")
    
    # visualise_epsilon_experiment("fast_log_linear", iter=4500, data_folder="results", save_folder="results")
    # visualise_epsilon_experiment("fast_binary_log_linear", iter=4500, data_folder="results", save_folder="results")
    # visualise_epsilon_experiment("log_linear_noisy", iter=4500, data_folder="results", save_folder="results")

    # visualise_full_feedback_comparison(iter=6000, data_folder="results", save_folder="results")
    # visualise_reduced_feedback_comparison(iter=5000, data_folder="results", save_folder="results")
    
    # visualise_delta_estimation_experiment("fast_log_linear", iter = 400, data_folder="results", save_folder="results")
    # visualise_delta_estimation_experiment("easy_fast_log_linear", iter = 400, data_folder="results", save_folder="results")
    
    visualise_experiment("delta", "fast_log_linear", processed_data_folder="results/experiments", save_root="results/visualisations", save_file_name="delta_experiment_fast_log_linear")
    visualise_experiment("delta", "fast_binary_log_linear", processed_data_folder="results/experiments", save_root="results/visualisations", save_file_name="delta_experiment_fast_binary_log_linear")
    visualise_experiment("delta", "log_linear_noisy", processed_data_folder="results/experiments", save_root="results/visualisations", save_file_name="delta_experiment_log_linear_noisy")
    
    visualise_experiment("epsilon", "fast_log_linear", processed_data_folder="results/experiments", save_root="results/visualisations", save_file_name="epsilon_experiment_fast_log_linear")
    visualise_experiment("epsilon", "fast_binary_log_linear", processed_data_folder="results/experiments", save_root="results/visualisations", save_file_name="epsilon_experiment_fast_binary_log_linear")
    visualise_experiment("epsilon", "log_linear_noisy", processed_data_folder="results/experiments", save_root="results/visualisations", save_file_name="epsilon_experiment_log_linear_noisy")
    
    visualise_experiment("full_feedback_comparison", "comparison", processed_data_folder="results/experiments", save_root="results/visualisations", save_file_name="full_feedback_comparison_experiment")
    visualise_experiment("reduced_feedback_comparison", "comparison", processed_data_folder="results/experiments", save_root="results/visualisations", save_file_name="reduced_feedback_comparison_experiment")
    
    visualise_experiment("delta_est", "fast_log_linear", processed_data_folder="results/experiments", save_root="results/visualisations", save_file_name="delta_estimation_experiment_fast_log_linear")
    visualise_experiment("delta_est", "easy_fast_log_linear", processed_data_folder="results/experiments", save_root="results/visualisations", save_file_name="delta_estimation_experiment_easy_fast_log_linear")
    