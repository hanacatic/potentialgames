import os
import pickle
import numpy as np
from potentialgames.utils.plot import *

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
    folder = 'WEEK 12'
        
    mean_potential = np.zeros((4, max_iter))
    mean_potential[0] = np.mean(potentials_history_log_linear, 0)
    mean_potential[1] = np.mean(potentials_history_mwu, 0)
    mean_potential[2] = np.mean(potentials_history_alpha_best, 0)
    mean_potential[3] = (1-eps) * np.ones((1, max_iter))
    
    std = np.zeros((3,max_iter))
    std[0] = np.std(potentials_history_log_linear, 0)
    std[1] = np.std(potentials_history_mwu, 0)
    std[2] = np.std(potentials_history_alpha_best, 0)
    
    labels = ['Log linear learning', 'HEDGE', 'Alpha best response',  r'$\Phi(a^*) - \epsilon$']
    
    plot_lines_with_std(mean_potential, std, labels, plot_e_efficient = True, save = save, folder = folder, file_name="Comparison_" + network + "_" + version)
    
    # plt.show()
    # list_labels = ['1', '2', '3', '4', '5', '..']
    # plot_lines(potentials_history_log_linear, list_labels=list_labels)
    # plt.show()
    # plot_lines(potentials_history_mwu, list_labels=list_labels)
    # plt.show()
    
    mean_objective = np.zeros((3, max_iter))
    mean_objective[0] = np.mean(objective_history_log_linear, 0)
    mean_objective[1] = np.mean(objective_history_mwu, 0)
    mean_objective[2] = np.mean(objective_history_alpha_best, 0)
    
    std = np.zeros((3,max_iter))
    std[0] = np.std(objective_history_log_linear, 0)
    std[1] = np.std(objective_history_mwu, 0)
    std[2] = np.std(objective_history_alpha_best, 0)
    print(mean_objective)
    
    plot_lines_with_std(mean_objective, std, labels, plot_e_efficient = False, save = False, title = "Average congestion", folder = folder, file_name="Comparison_congestion_" +  network + "_" + version, legend = "upper right")
    plt.show()
    
    # print(mean_objective[2])
    # plt.plot(mean_objective[2])
    # plt.show()
    
def visualise_comparison_data():
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "secondNE", "comparison")
    
    # log_linear_potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    # mwu_potentials_path = os.path.join(root, "mwu_potentials.pckl")
    # alpha_potentials_path = os.path.join(root, "alpha_best_potentials.pckl")
    
    log_linear_potentials_path = os.path.join(root, "log_linear_potentials_noisy.pckl")
    mwu_potentials_path = os.path.join(root, "mwu_potentials_noisy.pckl")
    alpha_potentials_path = os.path.join(root, "alpha_best_potentials_noisy.pckl")
    
    with open(log_linear_potentials_path, 'rb') as f:
        potentials_history_log_linear = pickle.load(f)
    with open(mwu_potentials_path, 'rb') as f:
        potentials_history_mwu = pickle.load(f)
    with open(alpha_potentials_path, 'rb') as f:
        potentials_history_alpha_best = pickle.load( f)
        
        
    max_iter = len(potentials_history_alpha_best.T)
    eps = 0.1
    
    mean_potential = np.zeros((4, max_iter))
    mean_potential[0] = np.mean(potentials_history_log_linear, 0)
    mean_potential[1] = np.mean(potentials_history_mwu, 0)
    mean_potential[2] = np.mean(potentials_history_alpha_best, 0)
    mean_potential[3] = (1-eps) * np.ones((1, max_iter))
    
    std = np.zeros((3,max_iter))
    std[0] = np.std(potentials_history_log_linear, 0)
    std[1] = np.std(potentials_history_mwu, 0)
    std[2] = np.std(potentials_history_alpha_best, 0)
    
    labels = ['Log linear learning', 'HEDGE', 'Alpha best response',  r'$\Phi(a^*) - \epsilon$']
    
    plot_lines_with_std(mean_potential, std, labels, plot_e_efficient = True, title = "Expected potential")
    plt.show()
    
    list_labels = ['1', '2', '3', '4', '5','6', '7', '8', '9', '10', '..']
    
    # print(len(potentials_history_mwu))
    # mask = np.random.random_integers(250, len(potentials_history_mwu), 10)
    plot_lines(mean_potential, list_labels=list_labels)
    plt.show()
    # mask = np.arange(0, 10)
    # for i in range(int(len(potentials_history_mwu)/10)):
    #     plot_lines(potentials_history_mwu[i*10+mask], list_labels=list_labels, title = "Expected potential")
    #     plt.show()
    
def visualise_betas_data():
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "uniform", "betas")
    
    potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    
    with open(potentials_path, 'rb') as f:
        potentials_history = pickle.load(f)
    
    potentials_history = np.array(potentials_history)
    potentials_history = np.einsum('ijk->jik', potentials_history)
        
    max_iter = len(potentials_history[0][0])
    iter = max_iter
    eps = 0.1
    
    mean_potential = np.zeros((len(potentials_history), max_iter))
    std = np.zeros((len(potentials_history),max_iter))

    for i in range(len(potentials_history)):
        mean_potential[i] = np.mean(potentials_history[i], 0)
        std[i] = np.std(potentials_history[i], 0)
        print(mean_potential[i].shape)
    
    labels = [ r'$\frac{\beta_T}{2}$', r'$\frac{5\beta_T}{8}$', r'$\frac{6\beta_T}{8}$', r'$\frac{7\beta_T}{8}$', r'$\beta_T$', r'$\frac{9\beta_T}{8}$', r'$\frac{10\beta_T}{8}$', r'$\frac{11\beta_T}{8}$', r'$\frac{3\beta_T}{2}$', r'$\Phi(a^*) - \epsilon$']

    mask = np.arange(4,9)
    mask = np.append(mask, 9)
    
    print(len(labels))
    plot_potential_with_std(mean_potential[1], std[1])
    plt.show()
    # mean_potential[len(potentials_history)] = (1-eps) * np.ones((1, max_iter))
    
    plot_lines(mean_potential[mask], np.array(labels)[mask], iter, plot_e_efficient = True, title = "Expected Potential")

    plot_lines_with_std(mean_potential[mask], std[mask],  np.array(labels)[mask], iter, plot_e_efficient = True, title = "Expected potential")
    plt.show()
    
    list_labels = ['1', '2', '3', '4', '5','6', '7', '8', '9', '10', '..']
    
    print(len(potentials_history))
    mask = np.random.random_integers(0, len(potentials_history[8])-1, 10)
    plot_lines(potentials_history[8][mask], list_labels=list_labels, iter = max_iter)
    plt.show()
    
    # mask = np.arange(0, 10)
    # for i in range(int(len(potentials_history_mwu)/10)):
    #     plot_lines(potentials_history_mwu[i*10+mask], list_labels=list_labels)
    #     plt.show()
 
def visualise_deltas_data(folder_name, file_name, iter = 10, save = False, save_folder = None, save_file_name = None):
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", folder_name)
    
    potentials_path = os.path.join(root, file_name)
    
    with open(potentials_path, 'rb') as f:
        potentials_history = pickle.load(f)
    
    potentials_history = np.array(potentials_history)
    print(potentials_history.shape)
    # potentials_history = np.einsum('ijk->jik', potentials_history)
        
    max_iter = len(potentials_history[0][0])

    eps = 0.05
    
    mean_potential = np.zeros((len(potentials_history)+1, max_iter))
    std = np.zeros((len(potentials_history),max_iter))

    conv_idx = np.zeros(len(potentials_history)+1)
    
    for i in range(len(potentials_history)):
        mean_potential[i] = np.mean(potentials_history[i], 0)
        std[i] = np.std(potentials_history[i], 0)
        print(mean_potential[i].shape)
    
        conv_idx[i+1] = np.argwhere(mean_potential[i] > 1 - eps)[0]
    
    mean_potential[len(potentials_history)] = np.ones(max_iter)*(1-eps)
    conv_idx[0] = 1 - eps
    
    print(conv_idx)
    
    labels = [ r'$\Delta = 0.15$', r'$\Delta = 0.1$', r'$\Delta = 0.075$', r'$\Phi(a^*) - \epsilon$']
    
    plot_lines(mean_potential, labels, iter, plot_e_efficient = True)

    plot_lines_with_std(mean_potential, std,  labels, iter, save = save, folder = save_folder, file_name = save_file_name, plot_e_efficient = True, conv_idx = conv_idx, title = "Expected potential")
    plt.show()
    
def visualise_eps_data(folder_name, file_name, iter = 10, save = False, save_folder_name = None, save_file_name = None):
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", folder_name)
    
    potentials_path = os.path.join(root, file_name)
    
    with open(potentials_path, 'rb') as f:
        potentials_history = pickle.load(f)
    
    potentials_history = np.array(potentials_history)
    
    print(potentials_history.shape)
    # potentials_history = np.einsum('ijkh->jkih', potentials_history)
    # print(potentials_history.shape)

    max_iter = len(potentials_history[0][0])
    eps = [0.1, 0.05, 0.025, 0.01]
        
    mean_potential = np.zeros((len(potentials_history), max_iter))
    std = np.zeros(mean_potential.shape)
    print(mean_potential.shape)
    
    conv_idx = np.zeros(len(potentials_history)*2)

    for i in range(len(potentials_history)):
        mean_potential[i] = np.mean(potentials_history[i], 0)
        std[i] = np.std(potentials_history[i], 0)
        
        conv_idx[2*i] = np.argwhere(mean_potential[i] > 1 - eps[i])[0]

        conv_idx[2*i+1] = 1 - eps[i]
    
    labels = [ r'$\epsilon = 0.1$', r'$\epsilon = 0.05$', r'$\epsilon = 0.025$', r'$\epsilon = 0.01$']
    
    # plot_lines_eps_exp(mean_potential[i], labels, iter, title = "Expected potential")
    plot_lines_eps_exp_with_std(mean_potential, std, labels, iter, conv_idx = conv_idx, title = "Expected potential", save = save, folder = save_folder_name, file_name=save_file_name)
    plt.show()

def visualise_no_actions_data():
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "secondNE", "A")
    
    potentials_path = os.path.join(root, "log_linear_potentials_binary.pckl")
    
    with open(potentials_path, 'rb') as f:
        potentials_history = pickle.load(f)
    
    max_iter = len(potentials_history[0][0])
    iter = max_iter

    eps = 0.1
    
    mean_potential = np.zeros((len(potentials_history), max_iter))
    std = np.zeros((len(potentials_history)-1,max_iter))

    for i in range(len(potentials_history)-1):
        mean_potential[i] = np.mean(potentials_history[i], 0)
        std[i] = np.std(potentials_history[i], 0)
    
    mean_potential[len(potentials_history)-1] = (1-eps) * np.ones((1, max_iter))


    # print(potentials_history.shape)
    labels = [r'A=6', r'A=12', r'A=18', r'A=24', r'A=30', r'A=36', r'$\Phi(a^*) - \epsilon$']
    
    # plot_potential(potentials_history[0][0])
    plot_lines(mean_potential, labels, iter, plot_e_efficient = True, title = "Avergae potential")
    plt.show()

    plot_lines_with_std(mean_potential, std,  labels, iter, plot_e_efficient = True, title = "Average potential")
    plt.show()
    
def visualise_no_players_data():
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "secondNE", "comparison", "noisy_beta") #"N")
    
    potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    potentials_path = os.path.join(root, "log_linear_potentials_gammas.pckl")
    
    with open(potentials_path, 'rb') as f:
        potentials_history = pickle.load(f)
    
    max_iter = len(potentials_history[0][0])
    iter = 600000

    eps = 0.1
    
    mean_potential = np.zeros((len(potentials_history)+1, max_iter))
    std = np.zeros((len(potentials_history),max_iter))

    for i in range(len(potentials_history)):
        mean_potential[i] = np.mean(potentials_history[i], 0)
        std[i] = np.std(potentials_history[i], 0)
    
    mean_potential[len(potentials_history)] = (1-eps) * np.ones((1, max_iter))


    print(potentials_history.shape)
    # labels = [r'N=2', r'N=4', r'N=6', r'N=8', r'$\Phi(a^*) - \epsilon$']
    # [0.2, 0.1, 0.05, 0.01, 0.001, 0.0001]
    labels = [r'$\xi=0.2$', r'$\xi=0.1$', r'$\xi=0.05$', r'$\xi=0.01$', r'$\xi=0.001$', r'$\xi=0.0001$', r'$\Phi(a^*) - \epsilon$']

    plot_lines(mean_potential, labels, iter, plot_e_efficient = True, title = "Expected potential")
    plt.show()
    plot_lines_with_std(mean_potential, std,  labels, iter, plot_e_efficient = True, title = "Expected potential")
    plt.show()

def visualise_time_varying_data():
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "Diagonal", "comparison", "time_varying")
    
    potentials_path = os.path.join(root, "log_linear_2.pckl")
    potentials_t_path = os.path.join(root, "log_linear_t_2.pckl")
    potentials_tatrenko_path = os.path.join(root, "log_linear_tatarenko_2.pckl")
    
    with open(potentials_path, 'rb') as f:
        potentials_history = pickle.load(f)
    with open(potentials_t_path, 'rb') as f:
        potentials_history_t = pickle.load(f)
    with open(potentials_tatrenko_path, 'rb') as f:
        potentials_history_tatarenko = pickle.load(f)
        
    max_iter = len(potentials_history[0])
    mean_potential_history = np.zeros((3, max_iter))
    mean_potential_history[0] = np.mean(potentials_history_t, 0)
    mean_potential_history[1] = np.mean(potentials_history_tatarenko, 0)
    mean_potential_history[2] = np.mean(potentials_history, 0)
    
    std = np.zeros((3, max_iter))
    std[0] = np.std(potentials_history_t, 0)
    std[1] = np.std(potentials_history_tatarenko, 0)  
    std[2] = np.std(potentials_history, 0)
    labels = [ r'$\beta(t) = \beta_T\cdot\frac{a\cdot\log(t+a)}{1+a\cdot\log(t+a)}$',  r'$\beta(t) = (t+1)^n$', r'$\beta = \beta_T$']
    labels = [ r'$\beta(t) = \beta_T\cdot\frac{a\cdot\log(t+a)}{1+a\cdot\log(t+a)}$',  r'$\beta(t) =  \frac{b^2+1}{b}\log(t+1)$', r'$\beta = \beta_T$']
    
    plot_lines_with_std(mean_potential_history, std, labels)
    plt.show()

def visualise_identical_interest():
    
    root_one = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "OnePlateau", "uniform", "betas")
    root_two = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "uniform", "betas")
    root_three = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "Diagonal", "uniform", "betas")

    potentials_one_path = os.path.join(root_one, "log_linear_potentials.pckl")
    potentials_two_path = os.path.join(root_two, "log_linear_potentials.pckl")
    potentials_three_path = os.path.join(root_three, "log_linear_potentials.pckl")
    
    with open(potentials_one_path, 'rb') as f:
        potentials_history_one = pickle.load(f)
    with open(potentials_two_path, 'rb') as f:
        potentials_history_two = pickle.load(f)
    with open(potentials_three_path, 'rb') as f:
        potentials_history_three = pickle.load(f)
    
    potentials_history_one = np.array(potentials_history_one)
    potentials_history_one = np.einsum('ijk->jik', potentials_history_one)
    potentials_history_two = np.array(potentials_history_two)
    potentials_history_two = np.einsum('ijk->jik', potentials_history_two)
    potentials_history_three = np.array(potentials_history_three)
    potentials_history_three = np.einsum('ijk->jik', potentials_history_three)
    
    eps = 0.1
    max_iter = len(potentials_history_one[4][0])
    mean_potential_history = np.zeros((4, max_iter))
    mean_potential_history[0] = np.mean(potentials_history_one[4], 0)
    mean_potential_history[1] = np.mean(potentials_history_two[4], 0)
    mean_potential_history[2] = np.mean(potentials_history_three[4], 0)
    mean_potential_history[3] = (1-eps) * np.ones((1, max_iter))

    
    std = np.zeros((3,max_iter))
    std[0] = np.std(potentials_history_one[4], 0)
    std[1] = np.std(potentials_history_two[4], 0)
    std[2] = np.std(potentials_history_three[4], 0)
    
    labels = ['Game A', 'Game B', 'Game C',  r'$\Phi(a^*) - \epsilon$']
    
    plot_lines_with_std(mean_potential_history, std, labels, plot_e_efficient = True, title = "Expected potential")
    plt.show()

def visualise_trenches():
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "Diagonal", "secondNE", "trenches")
    
    potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    
    with open(potentials_path, 'rb') as f:
        potentials_history = pickle.load(f)
        
    max_iter = len(potentials_history[0][0])
    iter = max_iter

    eps = 0.1
    
    mean_potential = np.zeros((len(potentials_history)+1, max_iter))
    std = np.zeros((len(potentials_history),max_iter))

    for i in range(len(potentials_history)):
        mean_potential[i] = np.mean(potentials_history[i], 0)
        std[i] = np.std(potentials_history[i], 0)
    
    mean_potential[len(potentials_history)] = (1-eps) * np.ones((1, max_iter))


    print(potentials_history.shape)
    labels = [r'$\delta = 0.0$', r'$\delta = 0.1$', r'$\delta = 0.2$', r'$\delta = 0.3$', r'$\delta = 0.4$', r'$\Phi(a^*) - \epsilon$']
    
    # plot_lines(mean_potential, labels, iter, plot_e_efficient = True, title = "Expected potential")
    plot_potential(potentials_history[4][0])
    plt.show()
    plot_lines_with_std(mean_potential, std,  labels, iter, plot_e_efficient = True, title = "Expected potential")
    plt.show()
    
def visualise_lll_binary():
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "secondNE", "comparison", "noisy_beta") # "binary"
    
    potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    # potentials_binary_path = os.path.join(root, "log_linear_binary_potentials.pckl")
    potentials_binary_path = os.path.join(root, "log_linear_noisy_potentials.pckl")


    with open(potentials_path, 'rb') as f:
        potentials_history = pickle.load(f)
    with open(potentials_binary_path, 'rb') as f:
        potentials_history_binary = pickle.load(f) 
        
    max_iter = len(potentials_history[0])
    iter = max_iter

    eps = 0.1
    
    mean_potential = np.zeros((3, max_iter))
    mean_potential[0] = np.mean(potentials_history,0)
    mean_potential[1] = np.mean(potentials_history_binary, 0)
    mean_potential[2] = (1-eps) * np.ones((1, max_iter))

    std = np.zeros((2,max_iter))
    std[0] = np.std(potentials_history, 0)
    std[1] = np.std(potentials_history_binary, 0)

    # labels = ['Log linear learning', 'Log linear binary learning', r'$\Phi(a^*) - \epsilon$']
    # labels = [ r'$U_i(a_i, a_{-i})$', r'$U_i(a_i, a_{-i}) + \xi_i(a_i, a_{-i})$', r'$\Phi(a^*) - \epsilon$']
    labels = [ 'Log linear learning', 'Fixed-share log linear learning', r'$\Phi(a^*) - \epsilon$']
    
    plot_lines(mean_potential, labels, iter, plot_e_efficient = True, title = "Average potential")
    # plot_potential(potentials_history[4][0])
    plt.show()
    plot_lines_with_std(mean_potential, std,  labels, iter, plot_e_efficient = True, title = "Average potential")
    plt.show()

def visualise_coverage(read_folder, read_file_name_1, read_file_name_2, save, save_folder_name, save_file_name_1, save_file_name_2):
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", read_folder)
    
    potentials_path = os.path.join(root, read_file_name_1)
    potentials_modified_path = os.path.join(root, read_file_name_2)

    with open(potentials_path, 'rb') as f:
        potentials_history = pickle.load(f)
    with open(potentials_modified_path, 'rb') as f:
        potentials_history_modified = pickle.load(f) 
        
    max_iter = len(potentials_history[0][0])
    iter = 2000

    eps = 0.05
    
    print(len(potentials_history[0]))
    
    mean_potential = np.zeros((len(potentials_history)+1, max_iter))
    mean_potential_modified = np.zeros((len(potentials_history)+1, max_iter))
    std = np.zeros((len(potentials_history),max_iter))
    std_modified = np.zeros((len(potentials_history), max_iter))
    conv_idx = np.zeros(len(potentials_history)+1)
    conv_idx_modified = np.zeros(len(potentials_history)+1)
    
    print(potentials_history.shape)
    print(mean_potential.shape)
    for i in range(len(potentials_history)):
        mean_potential[i] = np.mean(potentials_history[i], 0)
        std[i] = np.std(potentials_history[i], 0)
        mean_potential_modified[i] = np.mean(potentials_history_modified[i], 0)
        std_modified[i] = np.std(potentials_history_modified[i], 0)
        conv_idx[i+1] = np.argwhere(mean_potential[i] > 1 - eps)[0]
        conv_idx_modified[i+1] = np.argwhere(mean_potential_modified[i] > 1 - eps)[0]
    
    mean_potential[len(potentials_history)] = np.ones(max_iter)*(1-eps)
    mean_potential_modified[len(potentials_history)] = np.ones(max_iter)*(1-eps)
    conv_idx[0] = 1 - eps
    conv_idx_modified[0] = 1 - eps
        
    # # std[0] = np.std(potentials_history, 0)
    # # std[1] = np.std(potentials_history_binary, 0)

    # # labels = ['Log linear learning', 'Log linear binary learning', r'$\Phi(a^*) - \epsilon$']
    # # labels = [ r'$U_i(a_i, a_{-i})$', r'$U_i(a_i, a_{-i}) + \xi_i(a_i, a_{-i})$', r'$\Phi(a^*) - \epsilon$']
    # labels = [ 'Log linear learning', 'Fixed-share log linear learning', r'$\Phi(a^*) - \epsilon$']
    labels = ['N=100', 'N=200', 'N=300', 'N=400', 'N=500', r'$\Phi(a^*) - \epsilon$']
    
    plot_lines(mean_potential, labels, iter, plot_e_efficient = True, title = "Average potential")
    # plot_potential(potentials_history[4][0])
    plt.show()
    plot_lines_with_std(mean_potential, std,  labels, iter,  title = "Expected potential value", plot_e_efficient=True, conv_idx=conv_idx, save = save, folder = save_folder_name, file_name=save_file_name_1)
    plt.show(block = False)
    plt.pause(10)
    plot_lines_with_std(mean_potential_modified, std_modified,  labels, iter, title = "Expected potential value", plot_e_efficient=True, conv_idx=conv_idx_modified, save = save, folder = save_folder_name, file_name=save_file_name_2)
    plt.show(block = False)
    plt.pause(10)

def visualise_exp3():
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "secondNE", "comparison", "binary", "exp3p_ewa")

    potentials_binary_path = os.path.join(root, "log_linear_binary_potentials.pckl")
    potentials_exp3_path = os.path.join(root, "log_linear_potentials.pckl")
    potentials_ewa_path = os.path.join(root, "log_linear_ewa_potentials.pckl")
    
    with open(potentials_binary_path, 'rb') as f:
        potentials_history_binary = pickle.load(f)
    with open(potentials_exp3_path, 'rb') as f:
        potentials_history_exp3 = pickle.load(f)
    with open(potentials_ewa_path, 'rb') as f:
        potentials_history_ewa = pickle.load(f)
    
    eps = 0.1
    max_iter = len(potentials_history_binary[0])
    mean_potential_history = np.zeros((4, max_iter))
    mean_potential_history[0] = np.mean(potentials_history_binary, 0)
    mean_potential_history[1] = np.mean(potentials_history_exp3, 0)
    mean_potential_history[2] = np.mean(potentials_history_ewa, 0)
    mean_potential_history[3] = (1-eps) * np.ones((1, max_iter))

    
    std = np.zeros((3,max_iter))
    std[0] = np.std(potentials_history_binary, 0)
    std[1] = np.std(potentials_history_exp3, 0)
    std[2] = np.std(potentials_history_ewa, 0)
    
    labels = ['Binary log linear learning', 'EXP3.P', 'EWA',  r'$\Phi(a^*) - \epsilon$']
    
    plot_lines_with_std(mean_potential_history, std, labels, plot_e_efficient = True, title = "Expected potential")
    plt.show()
    
    labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '..']

    plot_lines(potentials_history_ewa[0:10],labels)
    plt.show()
