import os
import pickle
import numpy as np
from lib.aux_functions.plot import *

def visualise_potential_history():
    
    network = "SiouxFalls"
    version = "no_actions_5"
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", network)
    
    log_linear_potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    mwu_potentials_path = os.path.join(root, "mwu_potentials.pckl")
    alpha_potentials_path = os.path.join(root, "alpha_best_potentials.pckl")
    
    with open(log_linear_potentials_path, 'rb') as f:
        potentials_history_log_linear = pickle.load(f)
    with open(mwu_potentials_path, 'rb') as f:
        potentials_history_mwu = pickle.load(f)
    with open(alpha_potentials_path, 'rb') as f:
        potentials_history_alpha_best = pickle.load( f)
        
    log_linear_objective_path = os.path.join(root, "log_linear_objective.pckl")
    mwu_objective_path = os.path.join(root, "mwu_objective.pckl")
    alpha_objective_path = os.path.join(root, "alpha_best_objective.pckl")
    
    with open(log_linear_objective_path, 'rb') as f:
        objective_history_log_linear = pickle.load(f)
    with open(mwu_objective_path, 'rb') as f:
        objective_history_mwu = pickle.load(f)
    with open(alpha_objective_path, 'rb') as f:
        objective_history_alpha_best = pickle.load( f)
        
    max_iter = len(potentials_history_alpha_best.T)
    eps = 0.1
    
    save = True
    folder = 'WEEK 11'
        
    mean_potential = np.zeros((4, max_iter))
    mean_potential[0] = np.mean(potentials_history_log_linear, 0)
    mean_potential[1] = np.mean(potentials_history_mwu, 0)
    mean_potential[2] = np.mean(potentials_history_alpha_best, 0)
    mean_potential[3] = (1-eps) * np.ones((1, max_iter))
    
    std = np.zeros((3,max_iter))
    std[0] = np.std(potentials_history_log_linear, 0)
    std[1] = np.std(potentials_history_mwu, 0)
    std[2] = np.std(potentials_history_alpha_best, 0)
    
    labels = ['Log linear learning', 'Multiplicative weight update', 'Alpha best response',  r'$\Phi(a^*) - \epsilon$']
    
    plot_lines_with_std(mean_potential, std, labels, plot_e_efficient = True, save = save, folder = folder, file_name="Comparison_5_actions")
    
    plt.show()
    list_labels = ['1', '2', '3', '4', '5', '..']
    plot_lines(potentials_history_log_linear, list_labels=list_labels)
    plt.show()
    plot_lines(potentials_history_mwu, list_labels=list_labels)
    plt.show()
    
    mean_objective = np.zeros((3, max_iter))
    mean_objective[0] = np.mean(objective_history_log_linear, 0)
    mean_objective[1] = np.mean(objective_history_mwu, 0)
    mean_objective[2] = np.mean(objective_history_alpha_best, 0)
    
    std = np.zeros((3,max_iter))
    std[0] = np.std(objective_history_log_linear, 0)
    std[1] = np.std(objective_history_mwu, 0)
    std[2] = np.std(objective_history_alpha_best, 0)
    print(mean_objective)
    plot_lines_with_std(mean_objective, std, labels, plot_e_efficient = False, save = save, folder = folder, file_name="Comparison_5_actions_average_congestion")
    plt.show()
    
    # print(mean_objective[2])
    # plt.plot(mean_objective[2])
    # plt.show()