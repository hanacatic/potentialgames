import numpy as np
from game import Game,IdenticalInterestGame, rng
from aux_functions.plot import *
from aux_functions.helpers import make_symmetric_nd
from scipy.sparse import csr_matrix
RATIONALITY = 100
EPS = 0.5e-1
    
def mu(action_profile):
    return 1.0/16.0

def beta_experiments(game, n_exp = 10, eps = 0.1, algorithm = "log_linear", save = False, folder = None, file_name = None, title = 'Average potential'): 
    
    beta_t = game.compute_beta(eps)
    print(beta_t)
    print(game.max_iter)
        
    plot_payoff(game.gameSetup.payoff_player_1, folder = folder, save = save)
    mean_potential_history = np.zeros((7, game.max_iter))
    betas = np.arange(beta_t/2, beta_t + beta_t/8.0, beta_t/8.0)
    
    for idx, beta in enumerate(betas):
        
        print(beta)
        
        potentials_history = np.zeros((n_exp, game.max_iter))
        
        for i in range(0, n_exp):
            
            potentials_history[i] = np.transpose(game.potentials_history)
            game.play(beta=beta)
            
            # game.gameSetup.reset_payoff_matrix()
            # game.reset_game()
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
    game = Game(game.gameSetup, algorithm = algorithm, mu=mu)
    game.set_initial_action_profile(game.initial_action_profile)
    
    potentials_history = np.zeros((n_exp, game.max_iter))

    for i in range(0, n_exp):
        
        game.play()
        potentials_history[i] = np.transpose(game.potentials_history)
   
    mean_potential_history[5] = np.mean(potentials_history, 0)

    mean_potential_history[6] = (1-EPS) * np.ones((1, game.max_iter))
    labels = [ r'$\frac{\beta_T}{2}$', r'$\frac{5\beta_T}{8}$', r'$\frac{6\beta_T}{8}$', r'$\frac{7\beta_T}{8}$', r'$\beta_T$', r'$\beta(t)$', r'$\Phi(a^*) - \epsilon$']
   
    plot_lines(mean_potential_history, labels, True, title, file_name = file_name, save = save, folder = folder)

    if not save:

        plt.show(block = False)
        plt.pause(60)
        plt.close()
        
def delta_experiments(game, deltas = [0.9, 0.75, 0.5, 0.25, 0.1],  n_exp = 10, eps = 0.1, save = False, folder = None, file_name = None, title = 'Average potential'):
       
    beta_t = game.compute_beta(eps)
    
    plot_payoff(game.gameSetup.payoff_player_1)
    
    mean_potential_history = np.zeros((6, game.max_iter))
    
    for idx, delta in enumerate(deltas):
               
        game.gameSetup.reset_payoff_matrix(delta)
        game.reset_game()
        
        beta_t = game.compute_beta(eps)
        
        print(delta)
        print(beta_t)
        plot_payoff(game.gameSetup.payoff_player_1)
        
        potentials_history = np.zeros((n_exp, game.max_iter))
        
        for i in range(0, n_exp):
            
            potentials_history[i] = np.transpose(game.potentials_history).copy()
            game.play(beta=beta_t)
            
            game.gameSetup.reset_payoff_matrix(delta)
            game.reset_game()
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
    mean_potential_history[5] = (1-eps) * np.ones((1, game.max_iter))
    labels = [ r'$\Delta = 0.9$', r'$\Delta = 0.75$', r'$\Delta = 0.5$', r'$\Delta = 0.25$', r'$\Delta = 0.1$', r'$\Phi(a^*) - \epsilon$']
   
    plot_lines(mean_potential_history, labels, True, title = title, folder = folder, save = save, file_name = file_name)

    if not save:

        plt.show(block = False)
        plt.pause(60)
        plt.close()

def epsilon_experiments(game, epsilons = [0.2, 0.1, 0.05, 0.01, 0.001], n_exp = 10, save = False, folder = None, file_name = None, title = 'Average potential'):
    
    mean_potential_history = np.zeros((10, game.max_iter))
    
    for idx, eps in enumerate(epsilons):
        
        beta = game.compute_beta(eps)
        print(beta)
        
        potentials_history = np.zeros((n_exp, game.max_iter))
        
        for i in range(0, n_exp):
            
            potentials_history[i] = np.transpose(game.potentials_history)
            game.play(beta=beta)
            
            # game.gameSetup.reset_payoff_matrix()
            # game.reset_game()
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
        mean_potential_history[idx + 5] = (1 - eps) * np.ones((1, game.max_iter))
        
    labels = [ r'$\epsilon = 0.2$', r'$\epsilon = 0.1$', r'$\epsilon = 0.05$', r'$\epsilon = 0.01$', r'$\epsilon = 0.001$']
   
    plot_lines_eps_exp(mean_potential_history, labels, True, title = title, folder = folder, save = save, file_name = file_name)

    if not save:
        plt.show(block = False)
        plt.pause(60)
        plt.close()

def beta_experiments_fast(game, eps = 0.1, save = False, folder = None, file_name = None, title = 'Expected potential value'): 
    
    beta_t = game.compute_beta(eps)
    print(beta_t)
    print(game.max_iter)
        
    plot_payoff(game.gameSetup.payoff_player_1, folder = folder, save = save, file_name = "Payoff matrix " + file_name)
    potential_history = np.zeros((6, game.max_iter))
    betas = np.arange(beta_t/2, beta_t + beta_t/8.0, beta_t/8.0)
    
    for idx, beta in enumerate(betas):
        print(beta)
        game.play(beta=beta)
        potential_history[idx] = np.transpose(game.expected_value)

            # game.gameSetup.reset_payoff_matrix()
            # game.reset_game()
            

    potential_history[5] = (1-EPS) * np.ones((1, game.max_iter))
    labels = [ r'$\frac{\beta_T}{2}$', r'$\frac{5\beta_T}{8}$', r'$\frac{6\beta_T}{8}$', r'$\frac{7\beta_T}{8}$', r'$\beta_T$', r'$\Phi(a^*) - \epsilon$']
   
    plot_lines(potential_history, labels, True, title, file_name = file_name, save = save, folder = folder)

    if not save:
        # plt.show()
        plt.show(block = False)
        plt.pause(20)
        plt.close()
 
def epsilon_experiments_fast(game, epsilons = [0.2, 0.1, 0.05, 0.01, 0.001], scale_factor = 1, save = False, folder = None, file_name = None, title = 'Expected potential value'):
    
    potentials_history = np.zeros((10, game.max_iter))
    
    plot_payoff(game.gameSetup.payoff_player_1, folder = folder, save = save, file_name = "Payoff matrix epsilon")

    for idx, eps in enumerate(epsilons):
        
        beta = game.compute_beta(eps)
        print(beta)
        print(game.compute_t(eps))
        game.play(beta = beta, scale_factor = scale_factor)
            
        potentials_history[idx] = np.transpose(game.expected_value)
                        
        potentials_history[idx + 5] = (1 - eps) * np.ones((1, game.max_iter))
    labels = [ r'$\epsilon = 0.2$', r'$\epsilon = 0.1$', r'$\epsilon = 0.05$', r'$\epsilon = 0.01$', r'$\epsilon = 0.001$']
    
    plot_lines_eps_exp(potentials_history, labels, True, title = title, folder = folder, save = save, file_name = file_name)

    if not save:
        plt.show()
        # plt.show(block = False)
        # plt.pause(20)
        # plt.close()

def delta_experiments_fast(game, deltas = [0.9, 0.75, 0.5, 0.25, 0.1], trench = None, eps = 0.1, save = False, folder = None, file_name = None, title = 'Average potential'):
       
    beta_t = game.compute_beta(eps)
    plot_payoff(game.gameSetup.payoff_player_1)
    potential_history = np.zeros((6, game.max_iter))
    
    for idx, delta in enumerate(deltas):
        
        print(delta)
        if trench is None:
            payoff_matrix = generate_two_plateau_payoff_matrix(delta)
        else:
            payoff_matrix = generate_one_plateau_payoff_matrix(delta = delta, no_actions = game.gameSetup.no_actions, trench = trench)

        game.gameSetup.set_payoff_matrix(payoff_matrix)
        game.reset_game()
        plot_payoff(game.gameSetup.payoff_player_1, save = save, folder = folder, file_name = "Payoff matrix " + file_name + str(delta) )
        beta_t = game.compute_beta(eps)
        print(beta_t)
        
        game.play(beta=beta_t)
            
            
        potential_history[idx] = np.transpose(game.expected_value)
    
    potential_history[5] = (1-eps) * np.ones((1, game.max_iter))
    labels = [ r'$\Delta = 0.9$', r'$\Delta = 0.75$', r'$\Delta = 0.5$', r'$\Delta = 0.25$', r'$\Delta = 0.1$', r'$\Phi(a^*) - \epsilon$']
   
    plot_lines(potential_history, labels, True, title = title, folder = folder, save = save, file_name = file_name)

    if not save:
        # plt.show()
        plt.show(block = False)
        plt.pause(20)
        plt.close()

def generate_two_value_payoff_matrix(delta = 0.25, no_actions = 6, no_players = 2):
    
    firstNE = np.array([1]*no_players)
    payoff = np.ones(size = [no_actions] * no_players) * (1-delta)
    
    payoff[tuple(firstNE)] = 1
    
    return payoff

def generate_two_plateau_payoff_matrix(delta = 0.25, no_actions = 6):
        
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions-2, no_actions-2])

    b = 1 - delta 
       
    payoff = (rng.random(size=np.array([no_actions, no_actions])) * 0.25 + 0.75) * 0.7 * (1-delta) # 0.25

    # payoff_firstNE = (rng.random(size=np.array([5, 5]))*0.6 + 0.4) * 0.75 * b
    payoff_firstNE= (rng.random(size=np.array([3, 3]))*0.1 + 0.9) * b
    
    # payoff_secondNE = (rng.random(size=np.array([5, 5]))*0.65 + 0.35) * 0.5 * (1 - delta)
    payoff_secondNE = (rng.random(size=np.array([3, 3]))*0.15 + 0.85) * (1-delta)

    payoff[0:3,0:3] = payoff_firstNE
    payoff[-3::,-3::] = payoff_secondNE
    
    payoff[firstNE[0], firstNE[1]] = 1
    payoff[secondNE[0], secondNE[1]] = 1 - delta
    
    return payoff

def generate_two_plateau_payoff_matrix_multi(delta = 0.25, no_actions = 4, no_players = 2):
        
    firstNE = tuple(np.zeros(no_players, dtype=int))     
    secondNE = tuple((3 * np.ones(no_players)).astype(int))  

    b = 1 - delta 
       
    payoff = (rng.random(size = [no_actions] * no_players) * 0.25 + 0.75) * 0.6 * (1-delta) # 0.25

    # payoff_firstNE = (rng.random(size=np.array([5, 5]))*0.6 + 0.4) * 0.75 * b
    payoff_firstNE= (rng.random(size=[2]*no_players)*0.1 + 0.9) * b
    
    # payoff_secondNE = (rng.random(size=np.array([5, 5]))*0.65 + 0.35) * 0.5 * (1 - delta)
    payoff_secondNE = (rng.random(size=[2]*no_players)*0.15 + 0.85) * (1-delta)

    slice_all_dims = tuple([slice(0, 2)] * no_players)
    payoff[slice_all_dims] = payoff_firstNE

    slice_last_two_dims = tuple([slice(-2, None)] * no_players)
    payoff[slice_last_two_dims] = payoff_secondNE
    
    payoff[firstNE] = 1
    payoff[secondNE] = 1 - delta
    
    return payoff


def generate_one_plateau_payoff_matrix(delta = 0.25, no_actions = 6, trench = None):
        
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions-2, no_actions-2])

    b = 1 - delta 
    if trench is None:
        trench = 0.5 * (1-delta)
        
    payoff = (rng.random(size=np.array([no_actions, no_actions])) *0.5 + 0.5) * trench

    payoff_firstNE= (rng.random(size=np.array([3, 3]))*0.2 + 0.8) * b
    
    payoff[0:3,0:3] = payoff_firstNE
    
    payoff[firstNE[0], firstNE[1]] = 1
    payoff[firstNE[0]+1, firstNE[1]+1] = 1 - delta
    
    return payoff
           
def custom_game_alg_experiments(delta = 0.25, eps = 0.1, n_exp = 50, max_iter = 10000):
    
    action_space = np.arange(0, 35)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    # secondNE = np.array([2,2])
    
    # initial_action_profile = np.array([2,2])
    
    # payoff_matrix = generate_one_plateau_payoff_matrix(delta, no_actions = no_actions, trench = 0.1)
    
    initial_action_profile = secondNE
    payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    game_log_linear = Game(gameSetup, algorithm = "log_linear_fast", max_iter = max_iter, mu=mu)
    game_log_linear.set_initial_action_profile(initial_action_profile)

    game_mwu = Game(gameSetup, algorithm = "multiplicative_weight", max_iter = max_iter, mu=mu)
    game_mwu.set_initial_action_profile(initial_action_profile)
    
    game_alpha_best = Game(gameSetup, algorithm = "alpha_best_response", max_iter = max_iter, mu=mu)
    game_alpha_best.set_initial_action_profile(initial_action_profile)

    potentials_history_log_linear = np.zeros((1, max_iter))
    potentials_history_mwu = np.zeros((n_exp, max_iter))
    potentials_history_alpha_best = np.zeros((1, max_iter))
    
    beta_t = game_log_linear.compute_beta(1e-1)
    
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, initial_action_profile[0]*game_log_linear.gameSetup.no_actions + initial_action_profile[1]] = 1
    game_log_linear.set_mu_matrix(mu_matrix)
    
    game_log_linear.play(beta = beta_t)
    game_alpha_best.play()
    
    potentials_history_log_linear[0] = np.transpose(game_log_linear.expected_value)
    potentials_history_alpha_best[0] = np.transpose(game_alpha_best.potentials_history)  

    for i in range(n_exp):
           
        game_mwu.play()
        
        potentials_history_mwu[i] = np.transpose(game_mwu.potentials_history)
        
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
    
    save = True
    folder = 'WEEK 7'
    setup = 'Comparison_two_plateau_secondNE_delta_' + str(delta) + '_maxiter_' + str(max_iter) + '_no_actions_' + str(no_actions) + '_3'
    plot_payoff(payoff_matrix, save = save, folder = folder, file_name = 'Payoff_matrix_' + setup)
    plot_lines_with_std(mean_potential, std, labels, plot_e_efficient = True, save = save, folder = folder, file_name = setup)
    
    if not save:
        plt.show(block = False)
        plt.pause(60)
        plt.close()
    
    # initial_action_profile = np.array([1,3])
    
    # # initial_action_profile = np.array([4,4])
    # game_log_linear.set_initial_action_profile(initial_action_profile)
    # game_mwu.set_initial_action_profile(initial_action_profile)
    # game_alpha_best.set_initial_action_profile(initial_action_profile)
    
    # mu_matrix = np.zeros([1, len(action_space)**no_players])
    # mu_matrix[0, initial_action_profile[0]*game_log_linear.gameSetup.no_actions + initial_action_profile[1]] = 1
    # game_log_linear.set_mu_matrix(mu_matrix)
    
    # beta_t = game_log_linear.compute_beta(1e-1)
    
    # game_log_linear.play(beta = beta_t)
    # game_alpha_best.play()
    
    # potentials_history_log_linear[0] = np.transpose(game_log_linear.expected_value)
    # potentials_history_alpha_best[0] = np.transpose(game_alpha_best.potentials_history)  

    # for i in range(n_exp):
           
    #     game_mwu.play()
        
    #     potentials_history_mwu[i] = np.transpose(game_mwu.potentials_history)
    
    # mean_potential[0] = np.mean(potentials_history_log_linear, 0)
    # mean_potential[1] = np.mean(potentials_history_mwu, 0)
    # mean_potential[2] = np.mean(potentials_history_alpha_best, 0)
    # mean_potential[3] = (1-eps) * np.ones((1, max_iter)) 

    # std[0] = np.std(potentials_history_log_linear, 0)
    # std[1] = np.std(potentials_history_mwu, 0)
    # std[2] = np.std(potentials_history_alpha_best, 0)
        
    # setup = 'Comparison_two_plateau_trench_delta_' + str(delta) + '_maxiter_' + str(max_iter) + '_no_actions_' + str(no_actions)
    # # plot_payoff(payoff_matrix, 'Payoff_matrix_' + setup)
    # plot_lines_with_std(mean_potential, std, labels, plot_e_efficient = True, save = save, folder = folder, file_name = setup)
    
    # if not save:
    #     plt.show(block = False)
    #     plt.pause(60)
    #     plt.close()
    
    # initial_action_profile = np.array([0,0])
    # game_log_linear.set_initial_action_profile(initial_action_profile)
    # game_mwu.set_initial_action_profile(initial_action_profile)
    # game_alpha_best.set_initial_action_profile(initial_action_profile)
    
    # mu_matrix = np.zeros([1, len(action_space)**no_players])
    # mu_matrix[0, initial_action_profile[0]*game_log_linear.gameSetup.no_actions + initial_action_profile[1]] = 1
    # game_log_linear.set_mu_matrix(mu_matrix)
    
    # beta_t = game_log_linear.compute_beta(1e-1)
    
    # game_log_linear.play(beta = beta_t)
    # game_alpha_best.play()
    
    # potentials_history_log_linear[0] = np.transpose(game_log_linear.expected_value)
    # potentials_history_alpha_best[0] = np.transpose(game_alpha_best.potentials_history)  
    
    # for i in range(n_exp):
           
    #     game_mwu.play()
        
    #     potentials_history_mwu[i] = np.transpose(game_mwu.potentials_history)
    
    # mean_potential[0] = np.mean(potentials_history_log_linear, 0)
    # mean_potential[1] = np.mean(potentials_history_mwu, 0)
    # mean_potential[2] = np.mean(potentials_history_alpha_best, 0)
    # mean_potential[3] = (1-eps) * np.ones((1, max_iter)) 

    # std[0] = np.std(potentials_history_log_linear, 0)
    # std[1] = np.std(potentials_history_mwu, 0)
    # std[2] = np.std(potentials_history_alpha_best, 0)
    
    # setup = 'Comparison_two_plateau_plateau_delta_' + str(delta) + '_maxiter_' + str(max_iter) + '_no_actions_' + str(no_actions)
    # # plot_payoff(payoff_matrix, 'Payoff_matrix_' + setup)
    # plot_lines_with_std(mean_potential, std, labels, plot_e_efficient = True, save = save, folder = folder, file_name = setup)
    
    # if not save:
    #     plt.show(block = False)
    #     plt.pause(60)
    #     plt.close()

def compare_log_linear_t(delta = 0.25, max_iter = 200000):
    action_space = [0, 1, 2, 3, 4, 5]
    no_players = 2 
    
    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    
    payoff_matrix = generate_two_plateau_payoff_matrix(delta = delta, no_actions = len(action_space))
            
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta, payoff_matrix = payoff_matrix)
    
    game_t = Game(gameSetup, algorithm = "log_linear_t",  max_iter = max_iter, mu=mu)
    game_t.set_initial_action_profile(secondNE)

    game_t.set_initial_action_profile(np.array([1,4]))
    
    game_tatarenko = Game(gameSetup, algorithm = "log_linear_tatarenko",  max_iter = max_iter, mu=mu)
    game_tatarenko.set_initial_action_profile(secondNE)

    potentials_history_t = np.zeros((10, game_t.max_iter))
    potentials_history_tatarenko = np.zeros((10, game_tatarenko.max_iter))

    beta_t = game_t.compute_beta(1e-1)
    
    for i in range(10):
        game_t.play(beta = beta_t)
        game_tatarenko.play()
        potentials_history_t[i] = np.transpose(game_t.potentials_history).copy()
        potentials_history_tatarenko[i] = np.transpose(game_tatarenko.potentials_history).copy()

    mean_potential_history = np.zeros((2, max_iter))
    mean_potential_history[0] = np.mean(potentials_history_t, 0)
    mean_potential_history[1] = np.mean(potentials_history_tatarenko, 0)
    
    std = np.zeros((2, max_iter))
    std[0] = np.std(potentials_history_t, 0)
    std[1] = np.std(potentials_history_tatarenko, 0)  
    labels = [ r'$\beta(t) = \beta_T\cdot\frac{a\cdot\log(t+1)}{1+a\cdot\log(t+1)}$',  r'$\beta(t) = (t+1)^n$']
    
    save = True
    folder = 'WEEK 6'
    setup = 'Comparison_log_linear_t_two_plateau_secondNE'
    plot_payoff(game_t.gameSetup.payoff_player_1, save = save, folder = folder, file_name = 'Payoff matrix' + setup)
    plot_lines_with_std(mean_potential_history, std, labels, save = save, folder = folder, file_name = setup)
    
    if not save:
        plt.show(block = False)
        plt.pause(60)
        plt.close()
    
    game_t.set_initial_action_profile(np.array([1,4]))
    game_tatarenko.set_initial_action_profile(np.array([1,4]))
    
    beta_t = game_t.compute_beta(1e-1)
    
    for i in range(10):
        game_t.play(beta = beta_t)
        game_tatarenko.play()
        potentials_history_t[i] = np.transpose(game_t.potentials_history).copy()
        potentials_history_tatarenko[i] = np.transpose(game_tatarenko.potentials_history).copy()

    mean_potential_history[0] = np.mean(potentials_history_t, 0)
    mean_potential_history[1] = np.mean(potentials_history_tatarenko, 0)
    
    std[0] = np.std(potentials_history_t, 0)
    std[1] = np.std(potentials_history_tatarenko, 0)  
    
    setup = 'Comparison_log_linear_t_two_plateau_trench'
    plot_lines_with_std(mean_potential_history, std, labels, save = save, folder = folder, file_name = setup)
    
    if not save:
        plt.show(block = False)
        plt.pause(60)
        plt.close()
  
def main_simulation_experiment():
    # action_space = [0, 1, 2, 3]
    # no_players = 2
    # firstNE = np.array([1,1])
    # secondNE = np.array([3,3])
    # initial_action_profile = secondNE
    
    # delta = 0.25
    
    # gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta)
    # game = Game(gameSetup, mu=mu)
    # game.set_initial_action_profile(initial_action_profile)

    # beta_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential beta (3,3) random _')
    
    # initial_action_profile = np.array([1,3])
    # game.set_initial_action_profile(initial_action_profile)

    # beta_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential beta (1,3) random _')

    # initial_action_profile = np.array([0,2])
    # game.set_initial_action_profile(initial_action_profile)

    # beta_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential beta (0,2) random _')
    
    action_space = [0, 1, 2, 3]
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    initial_action_profile = secondNE
    
    delta = 0.25
    
    save = False
    folder = 'WEEK 5'
    
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta)
    mu_matrix = np.zeros([1, 16])
    mu_matrix[0, 15] = 1
    game = Game(gameSetup, algorithm = "log_linear", max_iter = 1e6, mu=mu)
    
    game.set_mu_matrix(mu_matrix)
    game.set_initial_action_profile(initial_action_profile)

    delta_experiments(game, save = save, folder = folder, file_name = 'Average potential delta (3,3) random _')
    
    # initial_action_profile = np.array([1,3])
    # game.set_initial_action_profile(initial_action_profile)

    # delta_experiments(game, save = save, folder = folder, file_name = 'Average potential delta (1,3) random _')

    # initial_action_profile = np.array([0,2])
    # game.set_initial_action_profile(initial_action_profile)

    # delta_experiments(game, save = save, folder = folder, file_name = 'Average potential delta (0,2) random _')

    # action_space = [0, 1, 2, 3]
    # no_players = 2
    # firstNE = np.array([1,1])
    # secondNE = np.array([3,3])
    # initial_action_profile = secondNE
    
    # delta = 0.25
    
    # gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta)
    # game = Game(gameSetup, mu=mu)
    # game.set_initial_action_profile(initial_action_profile)

    # epsilon_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential epsilon (3,3) random _')
    
    # initial_action_profile = np.array([1,3])
    # game.set_initial_action_profile(initial_action_profile)

    # epsilon_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential epsilon (1,3) random _')

    # initial_action_profile = np.array([0,2])
    # game.set_initial_action_profile(initial_action_profile)

    # epsilon_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential epsilon (0,2) random _')

    # delta_experiments()
    
    # epsilon_experiments()
    
    # action_space = [0, 1, 2, 3]
    # no_players = 2
    # firstNE = np.array([1,1])
    # secondNE = np.array([3,3])
    
    # save = False 
    # folder = 'WEEK 4'
    # title = 'Average potential'
    # n_exp = 10
    
    # # mean_potential_history = np.zeros((1, game.max_iter))
        
    # gameSetup = RandomIdenticalInterestGame(action_space, no_players, firstNE, secondNE, 0.25)
    # # game = Game(gameSetup, algorithm = "log_linear_t", mu=mu)
    # # game.set_initial_action_profile(secondNE)
    
    # game = Game(gameSetup, algorithm = "best_response", mu=mu)
    # game.set_initial_action_profile(secondNE)
    # plot_payoff(game.gameSetup.payoff_player_1)
    
    # for _ in range(n_exp):
                
    #     potentials_history = np.zeros((n_exp, game.max_iter))
    #     for i in range(0, n_exp):
    #         game.play()
    #         potentials_history[i] = np.transpose(game.potentials_history).copy()

            
    # mean_potential_history = np.mean(potentials_history, 0)
    
            
    # print(game.action_profile)
    # plot_potential(mean_potential_history)
    # plt.show(block = False)
    # plt.pause(20)
    # plt.close()

def custom_game_experiments(delta):
    
    # action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # firstNE = np.array([2,2])
    # secondNE = np.array([7,7])
    
    action_space = [0, 1, 2, 3, 4, 5]
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    trench = 0.25
    
    payoff_matrix = generate_one_plateau_payoff_matrix(delta, no_actions = no_actions, trench = trench)
    # payoff_matrix = generate_two_plateau_payoff_matrix(delta)

    # payoff_matrix = np.zeros([6,6])
    # payoff_matrix[0,0] = 1
    # payoff_matrix[1,1] = 1
    
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    
    mu_matrix = np.ones([1, no_actions**2])
    mu_matrix /= np.sum(mu_matrix)
    
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 3 * 1e3, mu=mu)
    game.set_mu_matrix(mu_matrix)
    
    folder = 'WEEK 6'
    save = False
    
    beta_experiments_fast(game, save = save, folder = folder, file_name = "betas_experiment_fast_uniform_one_plateau")
    
    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 28] = 1
    game.set_mu_matrix(mu_matrix)

    beta_experiments_fast(game, save = save, folder = folder, file_name = "betas_experiment_fast_trench_one_plateau")

    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 14] = 1
    game.set_mu_matrix(mu_matrix)
    
    beta_experiments_fast(game, save = save, folder = folder, file_name = "betas_experiment_fast_secondNE_one_plateau")

    mu_matrix = np.ones([1, no_actions**2])
    mu_matrix /= np.sum(mu_matrix)
    game.set_mu_matrix(mu_matrix)
    
    epsilon_experiments_fast(game, save = save, folder = folder, scale_factor = 5, file_name = "eps_experiment_fast_scale_factor_5_uniform_one_plateau")
    
    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 28] = 1
    game.set_mu_matrix(mu_matrix)
    
    epsilon_experiments_fast(game, save = save, folder = folder, scale_factor = 5, file_name = "eps_experiment_fast_scale_factor_5_trench_one_plateau")

    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 14] = 1
    game.set_mu_matrix(mu_matrix)
    
    epsilon_experiments_fast(game, save = save, folder = folder, scale_factor = 5, file_name = "eps_experiment_fast_scale_factor_5_secondNE_one_plateau")

    mu_matrix = np.ones([1, no_actions**2])
    mu_matrix /= np.sum(mu_matrix)
    game.set_mu_matrix(mu_matrix)
    
    delta_experiments_fast(game, trench = trench, save = save, folder = folder, file_name = "delta_experiment_fast_faster_unifrom_one_plateau")
    
    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 28] = 1
    game.set_mu_matrix(mu_matrix)
    
    delta_experiments_fast(game, trench = trench, save = save, folder = folder, file_name = "delta_experiment_fast_faster_trench_one_plateau")
    
    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 14] = 1
    game.set_mu_matrix(mu_matrix)
    
    delta_experiments_fast(game, trench = trench, save = save, folder = folder, file_name = "delta_experiment_fast_faster_secondNE_one_plateau")
        
    print(game.stationary)
    print(np.sum(game.gameSetup.P[0,:]))
    stationary = np.reshape(game.stationary,(-1, game.gameSetup.no_actions))
    plot_payoff(stationary)
    plot_payoff(game.gameSetup.P)
    plot_potential(game.expected_value)
    plot_potential(game.potentials_history)
    plot_payoff(game.gameSetup.payoff_player_1)
    plt.show(block = False)
    plt.pause(20)
    plt.close()
    # plt.show()

def custom_game_no_actions_experiments(k = [6, 8, 12, 18], delta = 0.25, eps = 1e-1, max_iter = 1000000):
    
    no_players = 2
    action_spaces = []
    payoff_matrices = []
    gameSetups = []
    games = []
    scale_factor =  5000 #100
    expected_values = np.zeros((len(k)+1, max_iter))
    # UNIFORM
    save = True
    folder = 'WEEK 6'
    
    for idx, no_actions in enumerate(k):
        action_spaces.append(np.arange(0, no_actions, 1))
        payoff_matrices.append(generate_two_plateau_payoff_matrix(delta = delta, no_actions = len(action_spaces[idx])))
        gameSetups.append(IdenticalInterestGame(action_spaces[idx], no_players, np.array([1,1]), np.array([no_actions-2, no_actions-2]), delta = delta, payoff_matrix = payoff_matrices[idx]))        
        plot_payoff(payoff_matrices[idx], save = save, folder = folder, file_name = "Payoff matrix no_actions_" + str(no_actions) + "_experiments_two_plateau")
        
        mu_matrix = np.ones([1, no_actions**2])
        mu_matrix /= np.sum(mu_matrix)
        games.append(Game(gameSetups[idx], algorithm = "log_linear_fast", max_iter = max_iter, mu=mu))
        games[idx].set_mu_matrix(mu_matrix)
        
        beta = games[idx].compute_beta(eps)
        print(beta)
        print(games[idx].compute_t(eps))
        
        epsilon_experiments_fast(games[idx], save = save, folder = folder, scale_factor = scale_factor, file_name = "comparison_no_actions_" +  str(no_actions) + "_eps_experiment_fast_faster_unifrom_real_scale_50_two_plateau")

        # games[idx].play(beta = beta, scale_factor = scale_factor)
            
        # expected_values[idx] = np.transpose(games[idx].expected_value)
        
        # plot_potential(expected_values[idx])
                        
    
    # expected_values[len(k)] = (1 - eps) * np.ones((1, max_iter))
    # labels = [r'k = 6', r'k = 8', r'k = 12', r'k = 18',  r'$\Phi(a^*) - \epsilon$']

    # plot_lines(expected_values, labels, plot_e_efficient = True, title = 'Expected potential value', save = save, folder = folder, file_name = 'comparison_no_actions_one_plateau_uniform')
    
    # # SECOND NE
    # for idx, no_actions in enumerate(k):
       
    #     mu_matrix = np.zeros([1, no_actions**2])
    #     mu_matrix[0, (no_actions - 2)*no_actions + no_actions - 2] = 1
    #     # mu_matrix /= np.sum(mu_matrix)
    #     games[idx].set_mu_matrix(mu_matrix)
        
    #     beta = games[idx].compute_beta(eps)
    #     print(beta)
    #     print(games[idx].compute_t(eps))
    #     games[idx].play(beta = beta, scale_factor = scale_factor)
            
    #     expected_values[idx] = np.transpose(games[idx].expected_value)
        
    # plot_lines(expected_values, labels, plot_e_efficient = True, title = 'Expected potential value', save = save, folder = folder, file_name = 'comprison_no_actions_one_plateau_secondNE')
    
    # # trench
    # for idx, no_actions in enumerate(k):
       
    #     mu_matrix = np.zeros([1, no_actions**2])
    #     if no_actions == 6:
    #         mu_matrix[0, 6*2 + 3] = 1
    #     else:
    #         mu_matrix[0, no_actions*(int(no_actions/2) - 1) + (int(no_actions/2) - 1) ] = 1
    #         print("correct element? ")
    #         print(payoff_matrices[idx][4,4])
    #     # mu_matrix /= np.sum(mu_matrix)
    #     games[idx].set_mu_matrix(mu_matrix)
        
    #     beta = games[idx].compute_beta(eps)
    #     print(beta)
    #     print(games[idx].compute_t(eps))
    #     games[idx].play(beta = beta, scale_factor = scale_factor)
            
    #     expected_values[idx] = np.transpose(games[idx].expected_value)
        
    # plot_lines(expected_values, labels, plot_e_efficient = True, title = 'Expected potential value', save = save, folder = folder, file_name = 'comprison_no_actions_one_plateau_trench')
    
    # plt.show()
 
def custom_game_no_players_experiments(N = [4, 6, 8], delta = 0.25, eps = 1e-1, max_iter = 10000):
    
    action_space = np.arange(0, 4)
    no_actions = len(action_space)
    no_players = N
    payoff_matrices = []
    gameSetups = []
    games = []
    expected_values = np.zeros((len(N)+1, max_iter))
    # UNIFORM
    save = False
    folder = 'WEEK 7'
    game_type = "Symmetrical"
    
    for idx, no_players in enumerate(N):
        payoff_matrices.append(generate_two_plateau_payoff_matrix_multi(delta = delta, no_actions = len(action_space), no_players = no_players))
        gameSetups.append(IdenticalInterestGame(action_space, no_players, np.array([1,1]), np.array([no_actions-2, no_actions-2]), type = game_type, delta = delta, payoff_matrix = payoff_matrices[idx]))        
        # plot_payoff(payoff_matrices[idx], save = save, folder = folder, file_name = "Payoff matrix no_players_" + str(no_actions) + "_experiments_two_plateau_asymmetrical")
        
        mu_matrix = np.ones([1, no_actions**no_players])
        mu_matrix /= np.sum(mu_matrix)
        games.append(Game(gameSetups[idx], algorithm = "log_linear_fast", max_iter = max_iter, mu=mu))
        games[idx].set_mu_matrix(mu_matrix)
        
        beta = games[idx].compute_beta(eps)
        print(no_players)
        print(beta)
        print(games[idx].compute_t(eps))
        
        # epsilon_experiments_fast(games[idx], save = save, folder = folder, scale_factor = scale_factor, file_name = "comparison_no_players_" +  str(no_actions) + "_eps_experiment_fast_faster_unifrom_real_scale_50_two_plateau")

        games[idx].play(beta = beta)
            
        expected_values[idx] = np.transpose(games[idx].expected_value) #potentials_history) 
        
        # plot_potential(expected_values[idx])
                        
    
    expected_values[len(N)] = (1 - eps) * np.ones((1, max_iter))
    labels = [r'N = 2', r'N = 4', r'N = 6',  r'$\Phi(a^*) - \epsilon$']

    plot_lines(expected_values, labels, plot_e_efficient = True, title = 'Expected potential value', save = save, folder = folder, file_name = 'comparison_no_players_two_plateau_uniform_' + game_type)
    
    # # SECOND NE
    # for idx, no_actions in enumerate(k):
       
    #     mu_matrix = np.zeros([1, no_actions**2])
    #     mu_matrix[0, (no_actions - 2)*no_actions + no_actions - 2] = 1
    #     # mu_matrix /= np.sum(mu_matrix)
    #     games[idx].set_mu_matrix(mu_matrix)
        
    #     beta = games[idx].compute_beta(eps)
    #     print(beta)
    #     print(games[idx].compute_t(eps))
    #     games[idx].play(beta = beta, scale_factor = scale_factor)
            
    #     expected_values[idx] = np.transpose(games[idx].expected_value)
        
    # plot_lines(expected_values, labels, plot_e_efficient = True, title = 'Expected potential value', save = save, folder = folder, file_name = 'comprison_no_actions_one_plateau_secondNE')
    
    # # trench
    # for idx, no_actions in enumerate(k):
       
    #     mu_matrix = np.zeros([1, no_actions**2])
    #     if no_actions == 6:
    #         mu_matrix[0, 6*2 + 3] = 1
    #     else:
    #         mu_matrix[0, no_actions*(int(no_actions/2) - 1) + (int(no_actions/2) - 1) ] = 1
    #         print("correct element? ")
    #         print(payoff_matrices[idx][4,4])
    #     # mu_matrix /= np.sum(mu_matrix)
    #     games[idx].set_mu_matrix(mu_matrix)
        
    #     beta = games[idx].compute_beta(eps)
    #     print(beta)
    #     print(games[idx].compute_t(eps))
    #     games[idx].play(beta = beta, scale_factor = scale_factor)
            
    #     expected_values[idx] = np.transpose(games[idx].expected_value)
        
    # plot_lines(expected_values, labels, plot_e_efficient = True, title = 'Expected potential value', save = save, folder = folder, file_name = 'comprison_no_actions_one_plateau_trench')
    
    plt.show()
    
def epsilon_experiments(delta):
     
    action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # action_space = [0, 1, 2, 3, 4, 5]
    no_actions = len(action_space)

    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])
    
    payoff_matrix = generate_two_plateau_payoff_matrix(no_actions= no_actions, delta=delta)

    # payoff_matrix = np.zeros([6,6])
    # payoff_matrix[0,0] = 1
    # payoff_matrix[1,1] = 1
    
    save = False
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    
    mu_matrix = np.ones([1, len(action_space)**2])
    mu_matrix /= np.sum(mu_matrix)
    
    # mu_matrix = np.zeros([1, len(action_space)**2])
    # # mu_matrix[0, 66] = 1
    # mu_matrix[0, 28] = 1
    
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e6, mu=mu)
    game.set_mu_matrix(mu_matrix)

    epsilon_experiments_fast(game, save = save, folder = "WEEK 6", file_name = "eps_experiment_fast_faster_unifrom_real_scale_1")
    epsilon_experiments_fast(game, save = save, folder = "WEEK 6", scale_factor = 50, file_name = "eps_experiment_fast_faster_unifrom_real_scale_50")
    epsilon_experiments_fast(game, save = save, folder = "WEEK 6", scale_factor = 5000, file_name = "eps_experiment_fast_faster_unifrom_real_scale_5000")
    epsilon_experiments_fast(game, save = save, folder = "WEEK 6", scale_factor = 1000000, file_name = "eps_experiment_fast_faster_unifrom_real_scale_mil")

