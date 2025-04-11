import numpy as np
from lib.games.gamebase import *
from lib.games.identinterest import *
from lib.games.trafficrouting import *
from lib.aux_functions.plot import *
from lib.games.coverage import *
import sparse 

plot = False

def mu(action_profile):
    return 1.0/16.0

def mu_congestion(profile):
    if (profile == np.zeros(len(profile))).all():
        return 1
    return 0

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
            
    if plot and not save:
        plot_payoff(game.gameSetup.payoff_player_1, folder = folder, save = save, file_name = "Payoff matrix " + str(file_name))
    betas = np.arange(beta_t/2, 3*beta_t/2 + beta_t/8.0, beta_t/8.0)
    potential_history = np.zeros((len(betas)+1, game.max_iter))

    
    for idx, beta in enumerate(betas):
        print(beta)
        game.play(beta=beta)
        potential_history[idx] = np.transpose(game.expected_value)

            # game.gameSetup.reset_payoff_matrix()
            # game.reset_game()
            

    potential_history[len(betas)] = (1-eps) * np.ones((1, game.max_iter))
    # labels = [ r'$\frac{\beta_T}{2}$', r'$\frac{5\beta_T}{8}$', r'$\frac{6\beta_T}{8}$', r'$\frac{7\beta_T}{8}$', r'$\beta_T$', r'$\Phi(a^*) - \epsilon$']
   
    labels = [ r'$\frac{\beta_T}{2}$', r'$\frac{5\beta_T}{8}$', r'$\frac{6\beta_T}{8}$', r'$\frac{7\beta_T}{8}$', r'$\beta_T$', r'$\frac{9\beta_T}{8}$', r'$\frac{10\beta_T}{8}$', r'$\frac{11\beta_T}{8}$', r'$\frac{3\beta_T}{2}$', r'$\Phi(a^*) - \epsilon$']
    
    plot_lines(potential_history, labels, True, title, file_name = file_name, save = save, folder = folder)

    if not save and plot:
        # plt.show()
        plt.show(block = False)
        plt.pause(20)
        plt.close()
    
    return potential_history
 
def epsilon_experiments_fast(game, epsilons = [0.2, 0.1, 0.05, 0.01, 0.001], scale_factor = 1, save = False, folder = None, file_name = None, title = 'Expected potential value'):
    
    potentials_history = np.zeros((10, game.max_iter))
    
    if not save and plot:
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

    if not save and plot:
        plt.show()
        # plt.show(block = False)
        # plt.pause(20)
        # plt.close()
    
    return potentials_history

def delta_experiments_fast(game, deltas = [0.9, 0.75, 0.5, 0.25, 0.1], trench = None, eps = 0.1, save = False, folder = None, file_name = None, title = 'Average potential'):
       
    beta_t = game.compute_beta(eps)
    plot_payoff(game.gameSetup.payoff_player_1)
    potential_history = np.zeros((6, game.max_iter))
    
    for idx, delta in enumerate(deltas):
        
        print(delta)
        if trench is None:
            # payoff_matrix = generate_one_plateau_payoff_matrix(delta = delta, no_actions = game.gameSetup.no_actions)
            payoff_matrix = generate_two_plateau_payoff_matrix(delta)
        else:
            # payoff_matrix = generate_one_plateau_payoff_matrix(delta = delta, no_actions = game.gameSetup.no_actions, trench = trench)
            payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = game.gameSetup.no_actions, trench = trench)


        game.gameSetup.set_payoff_matrix(delta, payoff_matrix)
        game.reset_game()
        if plot and not save:
            plot_payoff(game.gameSetup.payoff_player_1, save = save, folder = folder, file_name = "Payoff matrix " + str(file_name) + str(delta) )
        beta_t = game.compute_beta(eps)
        print(beta_t)
        
        game.play(beta=beta_t)
            
            
        potential_history[idx] = np.transpose(game.expected_value)
    
    potential_history[5] = (1-eps) * np.ones((1, game.max_iter))
    labels = [ r'$\Delta = 0.9$', r'$\Delta = 0.75$', r'$\Delta = 0.5$', r'$\Delta = 0.25$', r'$\Delta = 0.1$', r'$\Phi(a^*) - \epsilon$']
   
    plot_lines(potential_history, labels, True, title = title, folder = folder, save = save, file_name = file_name)

    if not save and plot:
        # plt.show()
        plt.show(block = False)
        plt.pause(20)
        plt.close()
    
    return potential_history

def generate_two_value_payoff_matrix(delta = 0.25, no_actions = 6, no_players = 2):
    
    firstNE = np.array([1]*no_players)
    payoff = np.ones(shape = [no_actions] * no_players) * (1-delta)
    
    payoff[tuple(firstNE)] = 1
    
    return payoff

def generate_two_plateau_payoff_matrix(delta = 0.25, no_actions = 6):
        
    A = no_actions
    firstNE = np.array([1, 1])
    secondNE = np.array([A - 2, A - 2])

    b = 1 - delta 
       
    payoff = (rng.random(size=np.array([no_actions, no_actions])) * 0.275 + 0.625 - delta) 

    payoff_firstNE= (rng.random(size=np.array([3, 3]))*0.15 + 0.85 - delta) 
    
    payoff_secondNE = (rng.random(size=np.array([3, 3]))*0.15 + 0.85 - delta)
    
    payoff[0:3,0:3] = payoff_firstNE
    
    payoff[-3::, -3::] = payoff_secondNE
    
    payoff[firstNE[0], firstNE[1]] = 1
    payoff[secondNE[0], secondNE[1]] = 1 - delta
    
    return payoff

def load_game(folder, no_game):
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "games", folder)
    
    game_path = os.path.join(root, f"game_{no_game}.pckl")
    
    print("Loading game: " + str(no_game) + " in folder " + folder)

    with open(game_path, 'rb') as f:
        payoff_matrix = pickle.load(f)
    
    return payoff_matrix
    
def generate_two_plateau_diagonal_payoff_matrix(delta = 0.25, no_actions = 6, trench = 0):
    
    no_players = 2
    firstNE = np.array([1]*no_players)
    
    payoff = np.ones(shape = [no_actions] * no_players) * trench

    np.fill_diagonal(payoff, 1 - delta)

    # payoff = np.kron(np.eye(no_actions//2,dtype=int), np.ones((2, 2)) * (1-delta - trench)) + np.ones(shape = [no_actions] * no_players) * trench
    
    payoff[tuple(firstNE)] = 1
    
    # payoff[tuple(4*firstNE)] = 1 - delta - 0.1
    
    return payoff 

def generate_two_plateau_payoff_matrix_multi(delta = 0.25, no_actions = 4, no_players = 2):
        
    firstNE = tuple(np.zeros(no_players, dtype=int))     
    secondNE = tuple((3 * np.ones(no_players)).astype(int))  

    b = 1 - delta 
    
    # shape = [no_actions] * no_players
    # dtype = np.float32  # Choose appropriate data type (float64 or float32)
    # payoff = np.memmap("payoff_matrix.npy", dtype=dtype, mode='write', shape=shape)
    
    # Generate the initial matrix values
    payoff = (rng.random(size=[no_actions] * no_players) * 0.25 + 0.75) * 0.6 * (1 - delta)
    
       
    # payoff = (rng.random(size = [no_actions] * no_players) * 0.25 + 0.75) * 0.6 * (1-delta) # 0.25

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

def generate_two_plateau_diagonal_payoff_matrix_multi(delta = 0.25, no_actions = 4, no_players = 2):
    
    coords = [np.arange(0, no_actions)]*no_players
    payoffs = (1-delta)*np.ones(no_actions)
    payoffs[1] = 1
    
    payoff = sparse.COO(coords, payoffs, shape = [no_actions]*no_players)
    
    return payoff

def generate_two_plateau_hard_payoff_matrix(delta = 0.25, trench = 0.1):
    
    no_players = 2
    no_actions = 6
    
    firstNE = np.array([1]*no_players)
    
    payoff = np.ones((no_actions, no_actions)) * trench
    payoff[1,1] = 1
    payoff[0,3] = 1 - delta
    payoff[2,5] = 1 - delta
    payoff[3,2] = 1 - delta - 0.1
    payoff[4,4] = 1 - delta 
    payoff[5,0] = 1 - delta
    
    return payoff
    
def generate_one_plateau_payoff_matrix(delta = 0.25, no_actions = 6, trench = None):
        
    firstNE = np.array([1,1])
    secondNE = np.array([firstNE[0]+1, firstNE[1]+1])

    b = 1 - delta 
    if trench is None:
        trench = 0.5 * (1-delta)
        
    payoff = (rng.random(size=np.array([no_actions, no_actions])) *0.5 + 0.5) * trench

    payoff_firstNE= (rng.random(size=np.array([3, 3]))*0.2 + 0.8) * b
    
    payoff[0:3,0:3] = payoff_firstNE
    
    payoff[firstNE[0], firstNE[1]] = 1
    payoff[firstNE[0]+1, firstNE[1]+1] = 1 - delta
    
    return payoff
           
def custom_game_alg_experiments(delta = 0.25, eps = 0.1, n_exp = 20, max_iter = 5000):
    
    action_space = np.arange(0, 6)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    # secondNE = np.array([2,2])
    
    # initial_action_profile = np.array([2,2])
    
    # payoff_matrix = generate_one_plateau_payoff_matrix(delta, no_actions = no_actions, trench = 0.1)
    trench = 0.0
    
    initial_action_profile = secondNE
    # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)
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
    folder = 'Report/Identical Interest Games/Experiments'
    setup = 'Comparison_two_plateau_secondNE_delta_' + str(delta) + '_maxiter_' + str(max_iter) + '_no_actions_' + str(no_actions) #+ '_trench_' + str(trench) + '_0'
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

def compare_log_linear_t(delta = 0.25, n_exp = 100, max_iter = 200000):
    action_space = [0, 1, 2, 3, 4, 5]
    no_players = 2 
    
    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    trench = 0.1
    
    # payoff_matrix = generate_two_plateau_payoff_matrix(delta = delta, no_actions = len(action_space))
    payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta = delta, no_actions = len(action_space), trench = trench)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta, payoff_matrix = payoff_matrix)
    
    game_t = Game(gameSetup, algorithm = "log_linear_t",  max_iter = max_iter, mu=mu)
    game_t.set_initial_action_profile(secondNE)
    
    game_tatarenko = Game(gameSetup, algorithm = "log_linear_tatarenko",  max_iter = max_iter, mu=mu)
    game_tatarenko.set_initial_action_profile(secondNE)
    
    game = Game(gameSetup, algorithm = "log_linear",  max_iter = max_iter, mu=mu)
    game.set_initial_action_profile(secondNE)
    
    potentials_history = np.zeros((n_exp, game.max_iter))
    potentials_history_t = np.zeros((n_exp, game_t.max_iter))
    potentials_history_tatarenko = np.zeros((n_exp, game_tatarenko.max_iter))

    beta_t = game_t.compute_beta(1e-1)
    
    for i in range(n_exp):
        game_t.play(beta = beta_t)
        game.play(beta = beta_t)
        game_tatarenko.play()
        potentials_history[i] = np.transpose(game.potentials_history).copy()
        potentials_history_t[i] = np.transpose(game_t.potentials_history).copy()
        potentials_history_tatarenko[i] = np.transpose(game_tatarenko.potentials_history).copy()

    mean_potential_history = np.zeros((3, max_iter))
    mean_potential_history[0] = np.mean(potentials_history_t, 0)
    mean_potential_history[1] = np.mean(potentials_history_tatarenko, 0)
    mean_potential_history[2] = np.mean(potentials_history, 0)
    
    std = np.zeros((3, max_iter))
    std[0] = np.std(potentials_history_t, 0)
    std[1] = np.std(potentials_history_tatarenko, 0)  
    std[2] = np.std(potentials_history, )
    labels = [ r'$\beta(t) = \beta_T\cdot\frac{a\cdot\log(t+1)}{1+a\cdot\log(t+1)}$',  r'$\beta(t) = (t+1)^n$', r'$\beta = \beta_T$']
    
    save = False
    folder = 'WEEK 6'
    setup = 'Comparison_log_linear_t_two_plateau_secondNE'
    plot_payoff(game_t.gameSetup.payoff_player_1, save = save, folder = folder, file_name = 'Payoff matrix' + setup)
    plot_lines_with_std(mean_potential_history, std, labels, save = save, folder = folder, file_name = setup)
        
    # game_t.set_initial_action_profile(np.array([1,4]))
    # game_tatarenko.set_initial_action_profile(np.array([1,4]))
    
    # beta_t = game_t.compute_beta(1e-1)
    
    # for i in range(10):
    #     game_t.play(beta = beta_t)
    #     game_tatarenko.play()
    #     potentials_history_t[i] = np.transpose(game_t.potentials_history).copy()
    #     potentials_history_tatarenko[i] = np.transpose(game_tatarenko.potentials_history).copy()

    # mean_potential_history[0] = np.mean(potentials_history_t, 0)
    # mean_potential_history[1] = np.mean(potentials_history_tatarenko, 0)
    
    # std[0] = np.std(potentials_history_t, 0)
    # std[1] = np.std(potentials_history_tatarenko, 0)  
    
    # setup = 'Comparison_log_linear_t_two_plateau_trench'
    # plot_lines_with_std(mean_potential_history, std, labels, save = save, folder = folder, file_name = setup)
    
    # if not save:
    #     plt.show(block = False)
    #     plt.pause(60)
    #     plt.close()
        
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "Diagonal", "comparison", "time_varying")
    
    log_linear = os.path.join(root, "log_linear_2.pckl")
    log_linear_t = os.path.join(root, "log_linear_t_2.pckl")
    log_linear_tatarenko = os.path.join(root, "log_linear_tatarenko_2.pckl")
    
    with open(log_linear, 'wb') as f:
        pickle.dump(potentials_history, f, pickle.HIGHEST_PROTOCOL)
    with open(log_linear_t, 'wb') as f:
        pickle.dump(potentials_history_t, f, pickle.HIGHEST_PROTOCOL)
    with open(log_linear_tatarenko, 'wb') as f:
        pickle.dump(potentials_history_tatarenko, f, pickle.HIGHEST_PROTOCOL)
        
    if not save:
        plt.show()
        # plt.pause(60)
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

def custom_game_experiments(delta = 0.25):
    
    # action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # firstNE = np.array([2,2])
    # secondNE = np.array([7,7])
    
    action_space = [0, 1, 2, 3, 4, 5]
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    trench = 0.2
    
    # payoff_matrix = generate_one_plateau_payoff_matrix(delta, no_actions = no_actions, trench = trench)
    # payoff_matrix = generate_two_plateau_payoff_matrix(delta)
    payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)

    # payoff_matrix = np.zeros([6,6])
    # payoff_matrix[0,0] = 1
    # payoff_matrix[1,1] = 1
    
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    
    mu_matrix = np.ones([1, no_actions**2])
    mu_matrix /= np.sum(mu_matrix)
    
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e6, mu=mu)
    game.set_mu_matrix(mu_matrix)
    
    folder = 'WEEK 7'
    save = True
    
    setup = "_diagonal_blocks_trench_" + str(trench)
    
    beta_experiments_fast(game, save = save, folder = folder, file_name = "betas_experiment_fast_uniform" + setup)
    
    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 28] = 1
    game.set_mu_matrix(mu_matrix)

    beta_experiments_fast(game, save = save, folder = folder, file_name = "betas_experiment_fast_trench" + setup)

    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 14] = 1
    game.set_mu_matrix(mu_matrix)
    
    beta_experiments_fast(game, save = save, folder = folder, file_name = "betas_experiment_fast_secondNE" + setup)

    mu_matrix = np.ones([1, no_actions**2])
    mu_matrix /= np.sum(mu_matrix)
    game.set_mu_matrix(mu_matrix)
    
    epsilon_experiments_fast(game, save = save, folder = folder, scale_factor = 5, file_name = "eps_experiment_fast_scale_factor_5_uniform" + setup)
    
    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 28] = 1
    game.set_mu_matrix(mu_matrix)
    
    epsilon_experiments_fast(game, save = save, folder = folder, scale_factor = 5, file_name = "eps_experiment_fast_scale_factor_5_trench" + setup)

    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 14] = 1
    game.set_mu_matrix(mu_matrix)
    
    epsilon_experiments_fast(game, save = save, folder = folder, scale_factor = 5, file_name = "eps_experiment_fast_scale_factor_5_secondNE" + setup)

    mu_matrix = np.ones([1, no_actions**2])
    mu_matrix /= np.sum(mu_matrix)
    game.set_mu_matrix(mu_matrix)
    
    delta_experiments_fast(game, trench = trench, save = save, folder = folder, file_name = "delta_experiment_fast_faster_unifrom" + setup)
    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 28] = 1
    game.set_mu_matrix(mu_matrix)
    
    delta_experiments_fast(game, trench = trench, save = save, folder = folder, file_name = "delta_experiment_fast_faster_trench" + setup)
    
    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 14] = 1
    game.set_mu_matrix(mu_matrix)
    
    delta_experiments_fast(game, trench = trench, save = save, folder = folder, file_name = "delta_experiment_fast_faster_secondNE_diagonal_trench_" + str(trench) )
        
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
 
def custom_game_no_players_experiments(N = [2, 4, 6], delta = 0.25, eps = 1e-1, max_iter = 100000):
    
    action_space = np.arange(0, 10)
    no_actions = len(action_space)
    no_players = N
    payoff_matrices = []
    gameSetups = []
    games = []
    expected_values = np.zeros((len(N)+1, max_iter))
    # UNIFORM
    save = False
    folder = 'WEEK 8'
    game_type = "Asymmetrical"
    
    for idx, no_players in enumerate(N):
        payoff_matrices.append(generate_two_plateau_payoff_matrix_multi(delta = delta, no_actions = len(action_space), no_players = no_players))
        gameSetups.append(IdenticalInterestGame(action_space, no_players, np.array([1,1]), np.array([no_actions-2, no_actions-2]), type = game_type, delta = delta, payoff_matrix = payoff_matrices[idx]))        
        # plot_payoff(payoff_matrices[idx], save = save, folder = folder, file_name = "Payoff matrix no_players_" + str(no_actions) + "_experiments_two_plateau_asymmetrical")
        
        mu_matrix = np.ones([1, no_actions**no_players])
        mu_matrix /= np.sum(mu_matrix)
        games.append(Game(gameSetups[idx], algorithm = "log_linear", max_iter = max_iter, mu=mu))
        games[idx].set_mu_matrix(mu_matrix)
        games[idx].set_initial_action_profile(np.array([no_actions-2]*no_players))
        
        beta = games[idx].compute_beta(eps)
        print(no_players)
        print(beta)
        print(games[idx].compute_t(eps))
        
        # epsilon_experiments_fast(games[idx], save = save, folder = folder, scale_factor = scale_factor, file_name = "comparison_no_players_" +  str(no_actions) + "_eps_experiment_fast_faster_unifrom_real_scale_50_two_plateau")

        games[idx].play(beta = beta)
            
        expected_values[idx] = np.transpose(games[idx].potentials_history) 
        
        # plot_potential(expected_values[idx])                    
    
    expected_values[len(N)] = (1 - eps) * np.ones((1, max_iter))
    labels = [r'N = 4', r'N = 6', r'N = 8',  r'$\Phi(a^*) - \epsilon$']
    # labels = [r'N = 8',  r'$\Phi(a^*) - \epsilon$']

    plot_lines(expected_values, labels, plot_e_efficient = True, title = 'Expected potential value', save = save, folder = folder, file_name = 'comparison_no_players_two_plateau_uniform_' + game_type + "_real")
    
    # game_type = "Symmetrical"
    # for idx, no_players in enumerate(N):
        
    #     gameSetups[idx].type = game_type
    #     gameSetups[idx].set_payoff_matrix(payoff_matrices[idx])
        
    #     games[idx].gameSetup = gameSetups[idx]
    #     games[idx].reset_game()
        
    #     beta = games[idx].compute_beta(eps)
    #     print(no_players)
    #     print(beta)
    #     print(games[idx].compute_t(eps))
        
    #     # epsilon_experiments_fast(games[idx], save = save, folder = folder, scale_factor = scale_factor, file_name = "comparison_no_players_" +  str(no_actions) + "_eps_experiment_fast_faster_unifrom_real_scale_50_two_plateau")

    #     games[idx].play(beta = beta)
            
    #     expected_values[idx] = np.transpose(games[idx].expected_value) #potentials_history) 
 
    # expected_values[len(N)] = (1 - eps) * np.ones((1, max_iter))

    # plot_lines(expected_values, labels, plot_e_efficient = True, title = 'Expected potential value', save = save, folder = folder, file_name = 'comparison_no_players_two_plateau_uniform_' + game_type + "_real")
    
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
    
    potentials_history = []
    
    potentials_history.append(epsilon_experiments_fast(game, save = save, folder = "WEEK 6", file_name = "eps_experiment_fast_faster_unifrom_real_scale_1"))
    potentials_history.append(epsilon_experiments_fast(game, save = save, folder = "WEEK 6", scale_factor = 50, file_name = "eps_experiment_fast_faster_unifrom_real_scale_50"))
    potentials_history.append(epsilon_experiments_fast(game, save = save, folder = "WEEK 6", scale_factor = 5000, file_name = "eps_experiment_fast_faster_unifrom_real_scale_5000"))
    potentials_history.append(epsilon_experiments_fast(game, save = save, folder = "WEEK 6", scale_factor = 1000000, file_name = "eps_experiment_fast_faster_unifrom_real_scale_mil"))
    
    return potentials_history

def epsilon_experiments_base(game):
     
    potentials_history = []
    
    potentials_history.append(epsilon_experiments_fast(game))
    potentials_history.append(epsilon_experiments_fast(game, scale_factor = 50))
    potentials_history.append(epsilon_experiments_fast(game, scale_factor = 5000))
    potentials_history.append(epsilon_experiments_fast(game, scale_factor = 1000000))
    
    return potentials_history

def custom_game_no_players_sim_experiments(N = [6, 8, 10], delta = 0.25, eps = 1e-1, n_exp = 1, max_iter = 100):
    
    action_space = np.arange(0, 4)
    no_actions = len(action_space)
    no_players = N
    payoff_matrices = []
    gameSetups = []
    games = []
    potentials_history = np.zeros((len(N)+1, max_iter))
    # UNIFORM
    save = True
    folder = 'WEEK 8'
    game_type = "Asymmetrical"
    
    for idx, no_players in enumerate(N):
        payoff_matrices.append(generate_two_plateau_payoff_matrix_multi(delta = delta, no_actions = len(action_space), no_players = no_players))
        gameSetups.append(IdenticalInterestGame(action_space, no_players, np.array([1,1]), np.array([no_actions-2, no_actions-2]), type = game_type, delta = delta, payoff_matrix = payoff_matrices[idx]))        
        # plot_payoff(payoff_matrices[idx], save = save, folder = folder, file_name = "Payoff matrix no_players_" + str(no_actions) + "_experiments_two_plateau_asymmetrical")
        
        mu_matrix = np.ones([1, no_actions**no_players])
        mu_matrix /= np.sum(mu_matrix)
        games.append(Game(gameSetups[idx], algorithm = "log_linear", max_iter = max_iter, mu=mu))
        games[idx].set_mu_matrix(mu_matrix)
        games[idx].set_initial_action_profile(np.array([no_actions-2]*no_players))
        
        beta = games[idx].compute_beta(eps)
        print(no_players)
        print(beta)
        print(games[idx].compute_t(eps))
        
        # epsilon_experiments_fast(games[idx], save = save, folder = folder, scale_factor = scale_factor, file_name = "comparison_no_players_" +  str(no_actions) + "_eps_experiment_fast_faster_unifrom_real_scale_50_two_plateau")
        
        for _ in range(n_exp):
            games[idx].play(beta = beta)
            potentials_history[idx] += np.transpose(games[idx].potentials_history)[0]/n_exp
        
        # plot_potential(expected_values[idx])                    
    
    potentials_history[len(N)] = (1 - eps) * np.ones((1, max_iter))
    labels = [r'N = 6', r'N = 8', r'N = 10',  r'$\Phi(a^*) - \epsilon$']
    # labels = [r'N = 8',  r'$\Phi(a^*) - \epsilon$']

    plot_lines(potentials_history, labels, plot_e_efficient = True, title = 'Expected potential value', save = save, folder = folder, file_name = 'comparison_no_players_two_plateau_uniform_' + game_type + "_real")
    
    # potentials_history = np.zeros((len(N)+1, max_iter))

    # game_type = "Symmetrical"
    # for idx, no_players in enumerate(N):
        
    #     gameSetups[idx].type = game_type
    #     gameSetups[idx].set_payoff_matrix(payoff_matrices[idx])
        
    #     games[idx].gameSetup = gameSetups[idx]
    #     games[idx].reset_game()
        
    #     beta = games[idx].compute_beta(eps)
    #     print(no_players)
    #     print(beta)
    #     print(games[idx].compute_t(eps))
        
    #     # epsilon_experiments_fast(games[idx], save = save, folder = folder, scale_factor = scale_factor, file_name = "comparison_no_players_" +  str(no_actions) + "_eps_experiment_fast_faster_unifrom_real_scale_50_two_plateau")

    #     for _ in range(n_exp):
    #         games[idx].play(beta = beta)
            
    #         potentials_history[idx] += np.transpose(games[idx].potentials_history)[0]/n_exp
 
    # potentials_history[len(N)] = (1 - eps) * np.ones((1, max_iter))

    # plot_lines(potentials_history, labels, plot_e_efficient = True, title = 'Expected potential value', save = save, folder = folder, file_name = 'comparison_no_players_two_plateau_uniform_' + game_type + "_real")
    
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

def traffic_routing_experiments(eps = 0.05, n_exp = 10, max_iter = 10000):
    
    # gameSetup = CongestionGame()
    gameSetup = CongestionGame("SiouxFallsSymmetric", 10, modified = True, modified_no_players=70)

    
    game = Game(gameSetup, algorithm = "log_linear", max_iter = max_iter, mu = mu_congestion)
    
    print(game.gameSetup.delta)
    beta_t = game.compute_beta(eps)
    t = game.compute_t(eps)
    print(beta_t)
    print(t)
    initial_action_profile = np.array([9]*game.gameSetup.no_players) # [0 0 0 0 0 0 2 0 0 0 0 0 0 3 0 0 0 4 0 0 1 0 0 0 0 1 0 0 0 0 3 1 1 0 0 1 3 
    #                           1 2 2 0 0 0 0 0 0 1 1 2 0 0 0 0 0 2 3 0 4 0 1 0 1 0 0 0 4 0 0 0 0 0 0 3 1
    #                           0 4 4 4 0 0 2 1 0 0 1 0 0 0 3 0 0 0 0 0 0 2 0 0 3 2 0 0 0 0 0 0 1 0 0 0 0 
    #                           0 1 0 0 0 0 3 0 0 0 2 0 0 0 0 1 0 0 4 3 3 0 0 1 3 3 0 4 0 1 0 0 0 1 0 0 0 
    #                           0 2 0 0 2 1 0 0 0 0 2 3 0 0 1 3 0 2 0 0 0 0 0 2 0 0 3 0 0 0 0 1 0 0 0 1 1
    #                           2 0 0 1 1 1 2 0 0 2 3 0 1 0 0 0 1 3 2 0 0 1 0 0 0 1 0 1 1 0 0 0 1 2 0 2 1 
    #                           0 0 0 1 3 0 0 0 0 0 1 0 1 2 2 3 0 2 0 0 0 0 0 0 0 2 0 3 1 0 0 0 1 3 0 1 0
    #                           0 0 0 0 0 0 0 0 0 0 0 2 0 0 1 4 0 0 0 3 0 0 2 4 0 0 1 0 0 0 3 0 3 0 0 4 2
    #                           2 1 0 0 1 0 4 0 4 0 3 2 1 0 0 3 4 3 0 0 4 1 3 0 0 0 3 1 0 2 3 1 0 0 0 0 0
    #                           0 1 1 1 0 0 1 0 0 2 0 0 2 2 2 3 0 0 0 0 1 1 1 1 0 4 4 3 2 1 0 1 1 1 1 1 0
    #                           0 0 0 0 3 0 3 0 0 2 0 4 0 0 0 1 0 1 1 2 0 1 0 0 1 0 0 0 0 3 2 3 2 2 1 2 1
    #                           1 0 3 0 0 0 1 0 0 0 1 0 1 2 2 0 0 3 0 0 0 4 2 1 0 0 2 0 0 0 0 0 0 0 0 0 0
    #                           0 0 0 0 3 0 0 4 0 0 0 0 1 1 0 0 0 0 1 0 1 0 0 0 3 2 0 0 0 0 2 0 0 0 0 1 0
    #                           0 0 0 0 0 0 0 0 1 0 3 0 2 1 0 3 0 0 0 0 2 1 1 1 1 0 0 0 0 0 0 0 2 0 3 0 id
    #                           id 0 2 2 0 1 0 0 0 0] #np.array([0]*game.gameSetup.no_players)
    print(game.gameSetup.potential_function(initial_action_profile))
    potentials_history = np.zeros((n_exp, max_iter))
    
    for i in range(n_exp):   
        initial_action_profile = np.random.randint(0, 5, game.gameSetup.no_players)
        game.play(initial_action_profile = initial_action_profile, beta = beta_t)
        potentials_history[i] = np.transpose(game.potentials_history)
    
    mean_potential = np.mean(potentials_history, 0)
    
    std = np.std(potentials_history, 0)
    
    plot_potential(mean_potential)
    plot_potential_with_std(mean_potential, std)
    plt.show()
    
def traffic_routing_alg_comparison_experiments(eps = 0.05, n_exp = 10, max_iter = 4000):
    
    save = True
    folder = 'WEEK 12'
    
    network = "SiouxFalls"
    gameSetup = CongestionGame()
    # gameSetup = CongestionGame("SiouxFallsSymmetric", 4, modified = True, modified_no_players=4)
        
    game_log_linear = Game(gameSetup, algorithm = "log_linear", max_iter = max_iter, mu = mu_congestion)
    game_mwu = Game(gameSetup, algorithm = "multiplicative_weight", max_iter = max_iter, mu = mu_congestion)
    game_alpha_best = Game(gameSetup, algorithm = "alpha_best_response", max_iter = max_iter, mu = mu_congestion)
    
    print(game_log_linear.gameSetup.delta)
    beta_t = game_log_linear.compute_beta(eps)
    t = game_log_linear.compute_t(eps)
    
    initial_action_profile = np.array([3]*game_log_linear.gameSetup.no_players) 
    #np.array([4]*gameSetup.no_players)
    
    potentials_history_log_linear = np.zeros((n_exp, max_iter))
    objectives_history_log_lineaer = np.zeros((n_exp, max_iter))
    potentials_history_mwu = np.zeros((n_exp, max_iter))
    objectives_history_mwu = np.zeros((n_exp, max_iter))
    
    for i in range(n_exp):   
        print("Experiment no: " + str(i+1))
        initial_action_profile = np.random.randint(0, 5, game_mwu.gameSetup.no_players)
        game_mwu.play(initial_action_profile = initial_action_profile, beta = beta_t)
        potentials_history_mwu[i] = np.transpose(game_mwu.potentials_history)
        objectives_history_mwu[i] = np.transpose(game_mwu.objectives_history)
        game_log_linear.play(initial_action_profile = initial_action_profile, beta = beta_t)
        potentials_history_log_linear[i] = np.transpose(game_log_linear.potentials_history)
        objectives_history_log_lineaer[i] = np.transpose(game_log_linear.objectives_history)
        
    game_alpha_best.play(initial_action_profile = initial_action_profile, beta = beta_t)
    potentials_history_alpha_best = np.transpose(game_alpha_best.potentials_history)
    objectives_history_alpha_best = np.transpose(game_alpha_best.objectives_history)
    
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
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", network, "random_initial")
    
    log_linear_potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    mwu_potentials_path = os.path.join(root, "mwu_potentials.pckl")
    alpha_potentials_path = os.path.join(root, "alpha_best_potentials.pckl")
    log_linear_objective_path = os.path.join(root, "log_linear_objective.pckl")
    mwu_objective_path = os.path.join(root, "mwu_objective.pckl")
    alpha_objective_path = os.path.join(root, "alpha_best_objective.pckl")

    with open(log_linear_potentials_path, 'wb') as f:
        pickle.dump(potentials_history_log_linear, f, pickle.HIGHEST_PROTOCOL)
    with open(mwu_potentials_path, 'wb') as f:
        pickle.dump(potentials_history_mwu, f, pickle.HIGHEST_PROTOCOL)
    with open(alpha_potentials_path, 'wb') as f:
        pickle.dump(potentials_history_alpha_best, f, pickle.HIGHEST_PROTOCOL)
        
    with open(log_linear_objective_path, 'wb') as f:
        pickle.dump(objectives_history_log_lineaer, f, pickle.HIGHEST_PROTOCOL)
    with open(mwu_objective_path, 'wb') as f:
        pickle.dump(objectives_history_mwu, f, pickle.HIGHEST_PROTOCOL)
    with open(alpha_objective_path, 'wb') as f:
        pickle.dump(objectives_history_alpha_best, f, pickle.HIGHEST_PROTOCOL)

    plot_lines_with_std(mean_potential, std, labels, plot_e_efficient = True, save = save, folder = folder, file_name="Comparison_4_actions")
    
    plt.show()
    plt.plot(objectives_history_alpha_best)
    plt.show()

def log_linear_binary_experiment(eps = 1e-1, n_exp = 5, max_iter = 3000):
    
    save = False
    folder = 'WEEK 14'
    
    network = "SiouxFalls"
    gameSetup = CongestionGame()
    
    game = Game(gameSetup, algorithm = "log_linear", max_iter = max_iter, mu = mu_congestion)
    game_binary= Game(gameSetup, algorithm = "log_linear_binary", max_iter = max_iter, mu = mu_congestion)
    
    print(game.gameSetup.delta)
    beta_t = game.compute_beta(eps)
    t = game.compute_t(eps)
    
    initial_action_profile = np.array([3]*game.gameSetup.no_players) 
    # np.array([4]*gameSetup.no_players)
    
    potentials_history = np.zeros((n_exp, max_iter))
    objectives_history = np.zeros((n_exp, max_iter))
    potentials_history_binary = np.zeros((n_exp, max_iter))
    objectives_history_binary = np.zeros((n_exp, max_iter))
    
    for i in range(n_exp):   
        print("Experiment no: " + str(i+1))
        game.play(initial_action_profile = initial_action_profile, beta = beta_t)
        potentials_history[i] = np.transpose(game.potentials_history)
        objectives_history[i] = np.transpose(game.objectives_history)
        game_binary.play(initial_action_profile = initial_action_profile, beta = beta_t)
        potentials_history_binary[i] = np.transpose(game_binary.potentials_history)
        objectives_history_binary[i] = np.transpose(game_binary.objectives_history)
    
    mean_potential = np.zeros((3, max_iter))
    mean_potential[0] = np.mean(potentials_history, 0)
    mean_potential[1] = np.mean(potentials_history_binary, 0)
    mean_potential[2] = (1-eps) * np.ones((1, max_iter))
    
    std = np.zeros((2,max_iter))
    std[0] = np.std(potentials_history, 0)
    std[1] = np.std(potentials_history_binary, 0)
    
    labels = ['Log linear learning', 'Log linear binary learning',  r'$\Phi(a^*) - \epsilon$']
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", network, "binary")
    
    log_linear_potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    log_linear_binary_potentials_path = os.path.join(root, "log_linear_binary_potentials.pckl")
    log_linear_objective_path = os.path.join(root, "log_linear_objective.pckl")
    log_linear_binary_objective_path = os.path.join(root, "log_linear_binary_objective.pckl")

    with open(log_linear_potentials_path, 'wb') as f:
        pickle.dump(potentials_history, f, pickle.HIGHEST_PROTOCOL)
    with open(log_linear_binary_potentials_path, 'wb') as f:
        pickle.dump(potentials_history_binary, f, pickle.HIGHEST_PROTOCOL)
        
    with open(log_linear_objective_path, 'wb') as f:
        pickle.dump(objectives_history, f, pickle.HIGHEST_PROTOCOL)
    with open(log_linear_binary_objective_path, 'wb') as f:
        pickle.dump(objectives_history_binary, f, pickle.HIGHEST_PROTOCOL)

    plot_lines_with_std(mean_potential, std, labels, plot_e_efficient = True, save = save, folder = folder, file_name="Comparison_binary")
    
    plt.show()
    plt.plot(objectives_history_binary)
    plt.show()

def exp3p_experiment(eps = 1e-1, n_exp = 10, max_iter = 10000):
    
    save = True
    folder = 'WEEK 12'
    
    network = "SiouxFalls"
    gameSetup = CongestionGame()
    
    game = Game(gameSetup, algorithm = "exp3p", max_iter = max_iter, mu = mu_congestion)
    game_binary = Game(gameSetup, algorithm = "log_linear_binary", max_iter = max_iter, mu = mu_congestion)
    game_ewa = Game(gameSetup, algorithm = "exponential_weight_annealing", max_iter = max_iter, mu = mu_congestion)

    print(game.gameSetup.delta)
    beta_t = game.compute_beta(eps)
    t = game.compute_t(eps)
    
    initial_action_profile = np.array([3]*game.gameSetup.no_players) 
    # np.array([4]*gameSetup.no_players)
    
    potentials_history = np.zeros((n_exp, max_iter))
    objectives_history = np.zeros((n_exp, max_iter))
    potentials_history_binary = np.zeros((n_exp, max_iter))
    objectives_history_binary = np.zeros((n_exp, max_iter))
    potentials_history_ewa = np.zeros((n_exp, max_iter))
    objectives_history_ewa = np.zeros((n_exp, max_iter))
    
    for i in range(n_exp):   
        print("Experiment no: " + str(i+1))
        game.play(initial_action_profile = initial_action_profile, beta = beta_t)
        potentials_history[i] = np.transpose(game.potentials_history)
        objectives_history[i] = np.transpose(game.objectives_history)
        game_binary.play(initial_action_profile = initial_action_profile, beta = beta_t)
        potentials_history_binary[i] = np.transpose(game_binary.potentials_history)
        objectives_history_binary[i] = np.transpose(game_binary.objectives_history)
        game_ewa.play(initial_action_profile = initial_action_profile, beta = beta_t)
        potentials_history_ewa[i] = np.transpose(game_ewa.potentials_history)
        objectives_history_ewa[i] = np.transpose(game_ewa.objectives_history)
    
    mean_potential = np.zeros((4, max_iter))
    mean_potential[0] = np.mean(potentials_history_binary, 0)
    mean_potential[1] = np.mean(potentials_history, 0)
    mean_potential[2] = np.mean(potentials_history_ewa, 0)
    mean_potential[3] = (1-eps) * np.ones((1, max_iter))
    
    std = np.zeros((3,max_iter))
    std[0] = np.std(potentials_history_binary, 0)
    std[1] = np.std(potentials_history, 0)
    std[2] = np.std(potentials_history_ewa)
    
    labels = ['Log linear binary learning',  'EXP3.P', 'EWA', r'$\Phi(a^*) - \epsilon$']
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", network, "binary", "comparison")
    
    log_linear_potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    log_linear_binary_potentials_path = os.path.join(root, "log_linear_binary_potentials.pckl")
    log_linear_ewa_potentials_path = os.path.join(root, "log_linear_ewa_potentials.pckl")
    log_linear_objective_path = os.path.join(root, "log_linear_objective.pckl")
    log_linear_binary_objective_path = os.path.join(root, "log_linear_binary_objective.pckl")
    log_linear_ewa_objective_path = os.path.join(root, "log_linear_ewa_objective.pckl")

    with open(log_linear_potentials_path, 'wb') as f:
        pickle.dump(potentials_history, f, pickle.HIGHEST_PROTOCOL)
    with open(log_linear_binary_potentials_path, 'wb') as f:
        pickle.dump(potentials_history_binary, f, pickle.HIGHEST_PROTOCOL)
    with open(log_linear_ewa_potentials_path, 'wb') as f:
        pickle.dump(potentials_history_ewa, f, pickle.HIGHEST_PROTOCOL)
       
    with open(log_linear_objective_path, 'wb') as f:
        pickle.dump(objectives_history, f, pickle.HIGHEST_PROTOCOL)
    with open(log_linear_binary_objective_path, 'wb') as f:
        pickle.dump(objectives_history_binary, f, pickle.HIGHEST_PROTOCOL)
    with open(log_linear_ewa_objective_path, 'wb') as f:
        pickle.dump(objectives_history_ewa, f, pickle.HIGHEST_PROTOCOL)

    plot_lines_with_std(mean_potential, std, labels, plot_e_efficient = True, save = save, folder = folder, file_name="Comparison_binary")
    
    plt.show()
    plt.plot(objectives_history_binary)
    plt.show()
    
def comparison_identical_interest_games(delta = 0.25, eps = 0.1, n_exp = 50, max_iter = 15000):
    
    action_space = np.arange(0, 20)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    # secondNE = np.array([2,2])
    
    # initial_action_profile = np.array([2,2])
    
    # payoff_matrix = generate_one_plateau_payoff_matrix(delta, no_actions = no_actions, trench = 0.1)
    trench = 0.1
    
    initial_action_profile = secondNE
    # initial_action_profile = np.array([10,10])
    # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)
    payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, noisy_utility = True, delta = delta, payoff_matrix = payoff_matrix)

    game_log_linear = Game(gameSetup, algorithm = "log_linear", max_iter = max_iter, mu=mu)
    game_log_linear.set_initial_action_profile(initial_action_profile)

    game_mwu = Game(gameSetup, algorithm = "multiplicative_weight", max_iter = max_iter, mu=mu)
    game_mwu.set_initial_action_profile(initial_action_profile)
    
    game_alpha_best = Game(gameSetup, algorithm = "alpha_best_response", max_iter = max_iter, mu=mu)
    game_alpha_best.set_initial_action_profile(initial_action_profile)

    potentials_history_log_linear = np.zeros((n_exp, max_iter))
    potentials_history_mwu = np.zeros((n_exp*10, max_iter))
    potentials_history_alpha_best = np.zeros((n_exp, max_iter))
    
    beta_t = game_log_linear.compute_beta(1e-1)
    
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, initial_action_profile[0]*game_log_linear.gameSetup.no_actions + initial_action_profile[1]] = 1
    game_log_linear.set_mu_matrix(mu_matrix)
    
    # game_log_linear.play(beta = beta_t)
    # game_alpha_best.play()
    
    # potentials_history_log_linear[0] = np.transpose(game_log_linear.expected_value)
    # potentials_history_alpha_best[0] = np.transpose(game_alpha_best.potentials_history)  

    for i in range(n_exp):
        
        payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)
        # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)

        game_log_linear.reset_game(delta, payoff_matrix)
        game_alpha_best.reset_game(delta, payoff_matrix)
        game_mwu.reset_game(delta, payoff_matrix)
        
        game_log_linear.play(beta = beta_t)
        game_alpha_best.play()
        
        potentials_history_log_linear[i] = np.transpose(game_log_linear.potentials_history)
        potentials_history_alpha_best[i] = np.transpose(game_alpha_best.potentials_history)
        for j in range(10):
            
            # game_log_linear.play(beta = beta_t)
            
            # potentials_history_log_linear[i*10+j] = np.transpose(game_log_linear.potentials_history)

                
            game_mwu.play()
            
            potentials_history_mwu[i*10+j] = np.transpose(game_mwu.potentials_history)
        
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
        
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "secondNE", "comparison")
    
    log_linear_potentials_path = os.path.join(root, "log_linear_potentials_noisy.pckl")
    mwu_potentials_path = os.path.join(root, "mwu_potentials_noisy.pckl")
    alpha_potentials_path = os.path.join(root, "alpha_best_potentials_noisy.pckl")

    with open(log_linear_potentials_path, 'wb') as f:
        pickle.dump(potentials_history_log_linear, f, pickle.HIGHEST_PROTOCOL)
    with open(mwu_potentials_path, 'wb') as f:
        pickle.dump(potentials_history_mwu, f, pickle.HIGHEST_PROTOCOL)
    with open(alpha_potentials_path, 'wb') as f:
        pickle.dump(potentials_history_alpha_best, f, pickle.HIGHEST_PROTOCOL)
        
    save = True
    folder = 'Report/Identical Interest Games/Experiments'
    setup = 'Comparison_two_plateau_secondNE_delta_' + str(delta) + '_maxiter_' + str(max_iter) + '_no_actions_' + str(no_actions) #+ '_trench_' + str(trench) + '_0'
    plot_payoff(payoff_matrix, save = save, folder = folder, file_name = 'Payoff_matrix_' + setup)
    plot_lines_with_std(mean_potential, std, labels, plot_e_efficient = True, save = save, folder = folder, file_name = setup)
    
    if not save:
        plt.show(block = False)
        plt.pause(60)
        plt.close()
    
def beta_experiments_fast_with_std(delta = 0.25, eps = 0.1, n_exp = 50, max_iter = 8000): 
    
    action_space = np.arange(0, 6)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    trench = 0.2
    
    initial_action_profile = secondNE
    # initial_action_profile = np.array([0,5])
    # payoff_matrix = generate_one_plateau_payoff_matrix(delta, no_actions = no_actions)
    payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)
    # payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = max_iter, mu=mu)
    game.set_initial_action_profile(initial_action_profile)
    
    # mu_matrix = np.zeros([1, len(action_space)**no_players])
    # mu_matrix[0, initial_action_profile[0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
    
    mu_matrix = np.ones([1, len(action_space)**no_players])/len(action_space)**no_players
    
    game.set_mu_matrix(mu_matrix)
    
    potential_history = []
    for i in range(n_exp):
        # payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)
        # payoff_matrix = generate_one_plateau_payoff_matrix(delta, no_actions = no_actions)
        
        payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)
        game.reset_game(payoff_matrix)
        potential_history.append(beta_experiments_fast(game, eps))
        plt.close('all')

    
    # mean_potential = np.mean(potential_history, 0)
    
    # std = np.zeros((len(mean_potential), max_iter))
    # for i in range(len(mean_potential)):
    #     std[i] = np.std(potential_history[i], 0)
    
    # labels = [ r'$\frac{\beta_T}{2}$', r'$\frac{5\beta_T}{8}$', r'$\frac{6\beta_T}{8}$', r'$\frac{7\beta_T}{8}$', r'$\beta_T$', r'$\frac{9\beta_T}{8}$', r'$\frac{10\beta_T}{8}$', r'$\frac{11\beta_T}{8}$', r'$\frac{3\beta_T}{2}$', r'$\Phi(a^*) - \epsilon$']

    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "Diagonal", "uniform", "betas")
    
    potentials_path = os.path.join(root, "log_linear_potentials.pckl")

    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)

    # plot_lines_with_std(mean_potential, std, labels)
    # plt.show()

def delta_experiments_fast_with_std(eps = 0.1, n_exp = 5, max_iter = 300000): 
    
    action_space = np.arange(0, 6)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    # secondNE = np.array([2,2])
    
    # initial_action_profile = np.array([2,2])
    
    # payoff_matrix = generate_one_plateau_payoff_matrix(delta, no_actions = no_actions, trench = 0.1)
    trench = 0.2
    
    initial_action_profile = secondNE
    # initial_action_profile = np.array([0,5])

    # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)
    # payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)
    # payoff_matrix = generate_one_plateau_payoff_matrix(delta, no_actions = no_actions)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = 0.25)

    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = max_iter, mu=mu)
    game.set_initial_action_profile(initial_action_profile)
    
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, initial_action_profile[0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
    
    # mu_matrix = np.ones([1, len(action_space)**no_players])/len(action_space)**no_players

    game.set_mu_matrix(mu_matrix)
    
    potential_history = []
    for i in range(n_exp):
        potential_history.append(delta_experiments_fast(game, eps=eps))
        plt.close('all')
    
    # mean_potential = np.mean(potential_history, 0)
    
    # std = np.zeros((len(mean_potential), max_iter))
    # for i in range(len(mean_potential)):
    #     std[i] = np.std(potential_history[0], 0)
    
    # labels = [ r'$\Delta = 0.9$', r'$\Delta = 0.75$', r'$\Delta = 0.5$', r'$\Delta = 0.25$', r'$\Delta = 0.1$', r'$\Phi(a^*) - \epsilon$']

    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "secondNE", "deltas")
    
    potentials_path = os.path.join(root, "log_linear_potentials.pckl")

    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)

    # plot_lines_with_std(mean_potential, std, labels)
    # plt.show()
    
def eps_experiments_fast_with_std(delta = 0.25, n_exp = 50, max_iter = 8000): 
    
    action_space = np.arange(0, 6)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])
    
    # initial_action_profile = secondNE
    initial_action_profile = np.array([0,5])

    # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)
    payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)
    # payoff_matrix = generate_one_plateau_payoff_matrix(delta, no_actions = no_actions)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = 0.25)

    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = max_iter, mu=mu)
    game.set_initial_action_profile(initial_action_profile)
    
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, initial_action_profile[0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
    
    # mu_matrix = np.ones([1, len(action_space)**no_players])/len(action_space)**no_players

    game.set_mu_matrix(mu_matrix)
    
    potential_history = []
    for i in range(n_exp):
        payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)
        # payoff_matrix = generate_one_plateau_payoff_matrix(delta, no_actions = no_actions)
        game.reset_game(payoff_matrix)
        potential_history.append(epsilon_experiments_base(game))
        plt.close('all')
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "trench", "eps")
    
    potentials_path = os.path.join(root, "log_linear_potentials.pckl")

    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)

    # plot_lines_with_std(mean_potential, std, labels)
    # plt.show()
 
def no_actions_experiments_fast_with_std(delta = 0.25, eps = 0.1, k = [35, 50, 75], n_exp = 50, max_iter = 40000):
        
    no_players = 2
    action_spaces = []
    payoff_matrices = []
    gameSetups = []
    games = []
    expected_values = np.zeros((len(k)+1, n_exp, max_iter))
    # UNIFORM
    
    save = False
    folder = 'WEEK 6'
    
    for idx, no_actions in enumerate(k):
        action_spaces.append(np.arange(0, no_actions, 1))
        payoff_matrices.append(generate_two_plateau_payoff_matrix(delta = delta, no_actions = len(action_spaces[idx])))
        gameSetups.append(IdenticalInterestGame(action_spaces[idx], no_players, np.array([1,1]), np.array([no_actions-2, no_actions-2]), delta = delta, payoff_matrix = payoff_matrices[idx]))        
        plot_payoff(payoff_matrices[idx], save = save, folder = folder, file_name = "Payoff matrix no_actions_" + str(no_actions) + "_experiments_two_plateau")
        
        # mu_matrix = np.ones([1, no_actions**2])
        # mu_matrix /= np.sum(mu_matrix)
        initial_action_profile = np.array([no_actions-2, no_actions-2])
        mu_matrix = np.zeros([1, no_actions**no_players])
        mu_matrix[0, initial_action_profile[0]*no_actions + initial_action_profile[1]] = 1
        
        # games.append(Game(gameSetups[idx], algorithm = "log_linear_fast", max_iter = max_iter, mu=mu))
        games.append(Game(gameSetups[idx], algorithm = "log_linear_binary", max_iter = max_iter, mu=mu))

        games[idx].set_mu_matrix(mu_matrix)
        games[idx].set_initial_action_profile(initial_action_profile)
        
        beta = games[idx].compute_beta(eps)
        print(beta)
        print(games[idx].compute_t(eps))
        
        for i in range(n_exp):
            payoff_matrix = generate_two_plateau_payoff_matrix(delta = delta, no_actions = len(action_spaces[idx]))
            games[idx].reset_game(payoff_matrix = payoff_matrix)
            games[idx].play(beta = beta)
            # expected_values[idx][i] = np.transpose(games[idx].expected_value)
            expected_values[idx][i] = np.transpose(games[idx].potentials_history)

    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "secondNE", "A")
    
    # potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    potentials_path = os.path.join(root, "log_linear_potentials_binary_2.pckl")


    with open(potentials_path, 'wb') as f:
        pickle.dump(expected_values, f, pickle.HIGHEST_PROTOCOL)
    
def no_players_experiments_fast_with_std(N = [2, 4, 6, 8], n_exp = 5, delta = 0.25, eps = 1e-1, max_iter = 15000):
    
    action_space = np.arange(0, 4)
    no_actions = len(action_space)
    no_players = N
    payoff_matrices = []
    gameSetups = []
    games = []
    expected_values = np.zeros((len(N)+1, n_exp, max_iter))
    # UNIFORM
    save = False
    folder = 'WEEK 8'
    game_type = "Asymmetrical"
    
    for idx, no_players in enumerate(N):
        payoff_matrices.append(generate_two_plateau_payoff_matrix_multi(delta = delta, no_actions = len(action_space), no_players = no_players))
        gameSetups.append(IdenticalInterestGame(action_space, no_players, np.array([1,1]), np.array([no_actions-2, no_actions-2]), type = game_type, delta = delta, payoff_matrix = payoff_matrices[idx]))        
        # plot_payoff(payoff_matrices[idx], save = save, folder = folder, file_name = "Payoff matrix no_players_" + str(no_actions) + "_experiments_two_plateau_asymmetrical")
        
        mu_matrix = np.ones([1, no_actions**no_players])
        mu_matrix /= np.sum(mu_matrix)
        # indices = np.arange(no_players)  # Indices: [0, 1, 2, ..., no_players-1]
        # total_idx = sum(3 * (no_actions ** i) for i in indices)
        # mu_matrix = np.zeros([1, no_actions**no_players])
        # mu_matrix[0, total_idx] = 1
        games.append(Game(gameSetups[idx], algorithm = "log_linear_fast", max_iter = max_iter, mu=mu))
        games[idx].set_mu_matrix(mu_matrix)
        games[idx].set_initial_action_profile(np.array([no_actions-2]*no_players))
        
        beta = games[idx].compute_beta(eps)
        print(no_players)
        print(beta)
        print(games[idx].compute_t(eps))
        
        # epsilon_experiments_fast(games[idx], save = save, folder = folder, scale_factor = scale_factor, file_name = "comparison_no_players_" +  str(no_actions) + "_eps_experiment_fast_faster_unifrom_real_scale_50_two_plateau")

        for i in range(n_exp):
            print("Experiment no: " + str(i+1))
            payoff_matrix = generate_two_plateau_payoff_matrix_multi(delta = delta, no_actions = len(action_space), no_players = no_players)
            games[idx].reset_game(payoff_matrix = payoff_matrix)
            games[idx].play(beta = beta)
            expected_values[idx][i] = np.transpose(games[idx].expected_value)
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "uniform", "N")
    
    potentials_path = os.path.join(root, "log_linear_potentials.pckl")

    with open(potentials_path, 'wb') as f:
        pickle.dump(expected_values, f, pickle.HIGHEST_PROTOCOL)
        
def trenches_experiments_with_std(trenches = [0, 0.1, 0.2, 0.3, 0.4], n_exp = 50, delta = 0.25, eps = 1e-1, max_iter = 200000):
    
    action_space = np.arange(0, 6)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])
    
    initial_action_profile = secondNE
    # initial_action_profile = np.array([0,5])

    # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = 0.25)

    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = max_iter, mu=mu)
    game.set_initial_action_profile(initial_action_profile)
    
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, initial_action_profile[0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
    
    # mu_matrix = np.ones([1, len(action_space)**no_players])/len(action_space)**no_players

    game.set_mu_matrix(mu_matrix)
    
    # potential_history = []
    beta_t = game.compute_beta(eps)
    
    expected_values = np.zeros((len(trenches), n_exp, max_iter))

    for idx, trench in enumerate(trenches):
        
        payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta=delta, no_actions = no_actions, trench = trench)
        plot_payoff(payoff_matrix)
        plt.show()
        game.reset_game(payoff_matrix = payoff_matrix)
        
        for i in range(n_exp):
            game.play(beta = beta_t)
            expected_values[idx][i] = np.transpose(game.expected_value)
            plt.close('all')
            print(game.expected_value[20000])
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "Diagonal", "secondNE", "trenches")
    
    potentials_path = os.path.join(root, "log_linear_potentials.pckl")

    with open(potentials_path, 'wb') as f:
        pickle.dump(expected_values, f, pickle.HIGHEST_PROTOCOL)

def identical_interest_binary_experiment(delta = 0.25, eps = 1e-1, n_exp = 100, max_iter = 5000):
    
    action_space = np.arange(0, 6)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    trench = 0.2
    
    initial_action_profile = secondNE
    payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    game = Game(gameSetup, algorithm = "log_linear", max_iter = max_iter, mu=mu)
    game_binary = Game(gameSetup, algorithm = "log_linear_binary", max_iter = max_iter, mu = mu)
    
    game.set_initial_action_profile(initial_action_profile)
    game_binary.set_initial_action_profile(initial_action_profile)
        
    potential_history = np.zeros((n_exp, max_iter))
    potential_history_binary = np.zeros((n_exp, max_iter))
    
    beta_t = game.compute_beta(eps)
    
    for i in range(n_exp):
        payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)

        game.reset_game(payoff_matrix = payoff_matrix)
        game_binary.reset_game(payoff_matrix = payoff_matrix)
        
        game.play(beta = beta_t)
        game_binary.play(beta = beta_t)
        potential_history[i] = np.transpose(game.potentials_history)
        potential_history_binary[i] = np.transpose(game_binary.potentials_history)
        
        plt.close('all')

    plot_potential(np.mean(potential_history, 0))
    plt.show()
    plot_potential(np.mean(potential_history_binary, 0))
    plt.show()

    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "secondNE", "comparison", "binary")
    
    potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    potentials_binary_path = os.path.join(root, "log_linear_binary_potentials.pckl")

    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)
    
    with open(potentials_binary_path, 'wb') as f:
        pickle.dump(potential_history_binary, f, pickle.HIGHEST_PROTOCOL)
        
    plot_potential(np.mean(potential_history, 0))
    plt.show()  
    plot_potential(np.mean(potential_history_binary, 0))
    plt.show()

def identical_interest_binary_experiment(delta = 0.25, eps = 1e-1, n_exp = 100, max_iter = 5000):
    
    action_space = np.arange(0, 6)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    trench = 0.2
    
    initial_action_profile = secondNE
    payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    game = Game(gameSetup, algorithm = "log_linear", max_iter = max_iter, mu=mu)
    game_binary = Game(gameSetup, algorithm = "log_linear_binary", max_iter = max_iter, mu = mu)
    
    game.set_initial_action_profile(initial_action_profile)
    game_binary.set_initial_action_profile(initial_action_profile)
        
    potential_history = np.zeros((n_exp, max_iter))
    potential_history_binary = np.zeros((n_exp, max_iter))
    
    beta_t = game.compute_beta(eps)
    
    for i in range(n_exp):
        payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)

        game.reset_game(payoff_matrix = payoff_matrix)
        game_binary.reset_game(payoff_matrix = payoff_matrix)
        
        game.play(beta = beta_t)
        game_binary.play(beta = beta_t)
        potential_history[i] = np.transpose(game.potentials_history)
        potential_history_binary[i] = np.transpose(game_binary.potentials_history)
        
        plt.close('all')

    plot_potential(np.mean(potential_history, 0))
    plt.show()
    plot_potential(np.mean(potential_history_binary, 0))
    plt.show()

    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "secondNE", "comparison", "binary")
    
    potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    potentials_binary_path = os.path.join(root, "log_linear_binary_potentials.pckl")

    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)
    
    with open(potentials_binary_path, 'wb') as f:
        pickle.dump(potential_history_binary, f, pickle.HIGHEST_PROTOCOL)
        
    plot_potential(np.mean(potential_history, 0))
    plt.show()  
    plot_potential(np.mean(potential_history_binary, 0))
    plt.show()

def identical_interest_exp3p_experiment(delta = 0.25, eps = 1e-1, n_exp = 50, max_iter = 200000):
    
    action_space = np.arange(0, 6)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    trench = 0.2
    
    initial_action_profile = secondNE
    # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    
    game = Game(gameSetup, algorithm = "exp3p", max_iter = max_iter, mu = mu_congestion)
    game_binary = Game(gameSetup, algorithm = "log_linear_binary", max_iter = max_iter, mu = mu_congestion)
    game_ewa = Game(gameSetup, algorithm = "exponential_weight_annealing", max_iter = max_iter, mu = mu_congestion)

    print(game.gameSetup.delta)
    beta_t = game.compute_beta(eps)
    t = game.compute_t(eps)
    
    initial_action_profile = np.array([3]*game.gameSetup.no_players) 
    # np.array([4]*gameSetup.no_players)
    
    potentials_history = np.zeros((n_exp, max_iter))
    potentials_history_binary = np.zeros((n_exp, max_iter))
    potentials_history_ewa = np.zeros((n_exp, max_iter))
    
    for i in range(n_exp):   
        print("Experiment no: " + str(i+1))
        
        payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)

        game.reset_game(payoff_matrix = payoff_matrix)
        game_binary.reset_game(payoff_matrix = payoff_matrix)
        game_ewa.reset_game(payoff_matrix = payoff_matrix)
        
        game.play(initial_action_profile = initial_action_profile, beta = beta_t)
        potentials_history[i] = np.transpose(game.potentials_history)
        game_binary.play(initial_action_profile = initial_action_profile, beta = beta_t)
        potentials_history_binary[i] = np.transpose(game_binary.potentials_history)
        game_ewa.play(initial_action_profile = initial_action_profile, beta = beta_t)
        potentials_history_ewa[i] = np.transpose(game_ewa.potentials_history)
    
    mean_potential = np.zeros((4, max_iter))
    mean_potential[0] = np.mean(potentials_history_binary, 0)
    mean_potential[1] = np.mean(potentials_history, 0)
    mean_potential[2] = np.mean(potentials_history_ewa, 0)
    mean_potential[3] = (1-eps) * np.ones((1, max_iter))
    
    std = np.zeros((3,max_iter))
    std[0] = np.std(potentials_history_binary, 0)
    std[1] = np.std(potentials_history, 0)
    std[2] = np.std(potentials_history_ewa, 0)
    
    labels = ['Log linear binary learning',  'EXP3.P', 'EWA', r'$\Phi(a^*) - \epsilon$']
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "secondNE", "comparison", "binary", "exp3p_ewa")
    
    log_linear_potentials_path = os.path.join(root, "log_linear_potentials.pckl")
    log_linear_binary_potentials_path = os.path.join(root, "log_linear_binary_potentials.pckl")
    log_linear_ewa_potentials_path = os.path.join(root, "log_linear_ewa_potentials.pckl")

    with open(log_linear_potentials_path, 'wb') as f:
        pickle.dump(potentials_history, f, pickle.HIGHEST_PROTOCOL)
    with open(log_linear_binary_potentials_path, 'wb') as f:
        pickle.dump(potentials_history_binary, f, pickle.HIGHEST_PROTOCOL)
    with open(log_linear_ewa_potentials_path, 'wb') as f:
        pickle.dump(potentials_history_ewa, f, pickle.HIGHEST_PROTOCOL)

    plot_lines_with_std(mean_potential, std, labels, plot_e_efficient = True)
    
    plt.show()

def identical_interest_noisy_utility_experiment(delta = 0.15, eps = 1e-1, n_exp = 100, max_iter = 100000):
    
    action_space = np.arange(0, 6)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    trench = 0.2
    
    initial_action_profile = secondNE
    payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    gameSetup_noisy = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, noisy_utility = True, delta = delta, payoff_matrix = payoff_matrix)
    # gameSetup_noisy = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    game = Game(gameSetup, algorithm = "log_linear", max_iter = max_iter, mu=mu)
    game_noisy = Game(gameSetup_noisy, algorithm = "log_linear", max_iter = max_iter, mu = mu)
    
    game.set_initial_action_profile(initial_action_profile)
    game_noisy.set_initial_action_profile(initial_action_profile)
        
    potential_history = np.zeros((n_exp, max_iter))
    potential_history_noisy = np.zeros((n_exp, max_iter))
    
    beta_t = game.compute_beta(eps)
    
    for i in range(n_exp):
        payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)
        # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)

        game.reset_game(payoff_matrix = payoff_matrix)
        game_noisy.reset_game(payoff_matrix = payoff_matrix)
        
        game.play(beta = beta_t)
        game_noisy.play(beta = beta_t)
        potential_history[i] = np.transpose(game.potentials_history)
        potential_history_noisy[i] = np.transpose(game_noisy.potentials_history)
        
        plt.close('all')

    plot_potential(np.mean(potential_history, 0))
    plt.show()
    plot_potential(np.mean(potential_history_noisy, 0))
    plt.show()

    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "SecondNE", "comparison", "noisy_beta")
    
    potentials_path = os.path.join(root, "log_linear_potentials_01.pckl")
    potentials_noisy_path = os.path.join(root, "log_linear_noisy_potentials_01.pckl")

    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)
    
    with open(potentials_noisy_path, 'wb') as f:
        pickle.dump(potential_history_noisy, f, pickle.HIGHEST_PROTOCOL)
        
    # plot_potential(np.mean(potential_history, 0))
    # plt.show()  
    # plot_potential(np.mean(potential_history_binary, 0))
    # plt.show()

def identical_interest_noisy_beta_experiment(gammas = [0.2, 0.1, 0.05, 0.01, 0.001, 0.0001], delta = 0.25, eps = 1e-1, n_exp = 100, max_iter = 10000):
    
    action_space = np.arange(0, 6)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    trench = 0.2
    
    initial_action_profile = secondNE
    payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    game = Game(gameSetup, algorithm = "log_linear", max_iter = max_iter, mu = mu)
    
    game.set_initial_action_profile(initial_action_profile)
        
    potential_history = np.zeros((len(gammas), n_exp, max_iter))
    
    beta_t = game.compute_beta(eps)
    
    for idx, gamma in enumerate(gammas):
        for i in range(n_exp):
            payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = no_actions)
            # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, no_actions = no_actions, trench = trench)

            game.reset_game(payoff_matrix = payoff_matrix)  
        
            game.play(beta = beta_t, gamma = gamma)
            potential_history[idx][i] = np.transpose(game.potentials_history)
        
            plt.close('all')

    plot_potential(np.mean(potential_history[4], 0))
    plt.show()

    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "TwoPlateau", "SecondNE", "comparison", "noisy_beta")
    
    potentials_path = os.path.join(root, "log_linear_potentials_gammas.pckl")

    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)
    
    # plot_potential(np.mean(potential_history, 0))
    # plt.show()  
    # plot_potential(np.mean(potential_history_binary, 0))
    # plt.show()
    
def deltaExperiments(algorithm = "log_linear_fast", no_actions = 10, no_players = 2, deltas = [0.15, 0.1, 0.075], noisy_utility = False, eps = 0.05, max_iter = 1000000, n_exp = 30): 
    
    print("Delta experiments for algorithm: " + algorithm)
    
    action_space = np.arange(0, no_actions)
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, noisy_utility = noisy_utility, delta = deltas[0])
    game = Game(gameSetup, algorithm = algorithm, max_iter = max_iter)
    
    initial_action_profile = secondNE
    game.set_initial_action_profile(initial_action_profile)
    
    # indicator initial joint action profile distribution
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, initial_action_profile[0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
    
    # uniform initial joint action profile distribution
    # mu_matrix = np.ones([1, no_actions**no_players])/no_actions**no_players
    
    payoff_matrix = []
    
    if "fast" in algorithm:
        game.set_mu_matrix(mu_matrix)
            
        potential_history = np.zeros((len(deltas), n_exp, max_iter))
    else:
        potentials = np.zeros((10, max_iter))
        potential_history = np.zeros((len(deltas), n_exp, max_iter))
        
    mean_potential_history = np.zeros((len(deltas), max_iter))
    std = np.zeros((len(deltas), max_iter))
    
    if noisy_utility:
        betas = np.zeros(len(deltas))
        for i, delta in enumerate(deltas):
            game.reset_game(delta = delta)
            betas[i] = game.compute_beta(eps)
        
        eta = 1/2.0/np.max(betas)
        game.gameSetup.eta = eta
    
    for idx, delta in enumerate(deltas):
        
        print("Currently testing delta: " + str(delta))
        
        folder = "delta_" + str(int(delta*1000)).zfill(4)
        for i in range(n_exp):
            if i % 10 == 0:
                print("     Experiment No. " + str(i+1))
            try:
                payoff_matrix = load_game(folder, i) 
            except:
                payoff_matrix = save_two_player_game(no_actions, delta, i)
            # generate_two_plateau_payoff_matrix(delta = delta, no_actions = no_actions)
            # game.gameSetup.set_payoff_matrix(delta, payoff_matrix)
            game.reset_game(delta = delta, payoff_matrix = payoff_matrix)
            
            if "fast" in algorithm:

                beta = game.compute_beta(eps)
                game.play(beta = beta)
                
                potential_history[idx][i] = np.transpose(game.expected_value) 
            else:
                for j in range(10):
                    beta = game.compute_beta(eps)
                    game.play(beta = beta)
                
                    potentials[j] = np.transpose(game.potentials_history) 
                
                potential_history[idx][i] = np.mean(potentials, 0) 

    
        mean_potential_history[idx] = np.mean(potential_history[idx], 0)
        
        index = np.argwhere(mean_potential_history[idx] > 1 - eps)
        print(index)
        std[idx] = np.std(potential_history[idx], 0)
    
    plot_lines_with_std(mean_potential_history, std, ["150", "100", "075"])
    plt.show(block = False)
    
    # potential_history = []
    # for i in range(n_exp):
    #     potential_history.append(delta_experiments_fast(game, eps=eps))
    #     plt.close('all')
    
    # # mean_potential = np.mean(potential_history, 0)
    
    # # std = np.zeros((len(mean_potential), max_iter))
    # # for i in range(len(mean_potential)):
    # #     std[i] = np.std(potential_history[0], 0)
    
    if noisy_utility:
        algorithm = algorithm + "_noisy"
    
    # # labels = [ r'$\Delta = 0.9$', r'$\Delta = 0.75$', r'$\Delta = 0.5$', r'$\Delta = 0.25$', r'$\Delta = 0.1$', r'$\Phi(a^*) - \epsilon$']

    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "deltaExperiment")
    
    potentials_path = os.path.join(root, algorithm + "_potentials.pckl")
    
    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)


def epsExperiments(algorithm = "log_linear_fast", no_actions = 10, no_players = 2, delta = 0.1, epsilons = [0.1, 0.05, 0.025, 0.01], noisy_utility = False, max_iter = 3000000, n_exp = 30): 
    
    print("Epsilon experiments for algorithm: " + algorithm)
    action_space = np.arange(0, no_actions)
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, noisy_utility = noisy_utility, delta = delta)
    game = Game(gameSetup, algorithm = algorithm, max_iter = max_iter)
    
    initial_action_profile = secondNE
    game.set_initial_action_profile(initial_action_profile)
    
    # indicator initial joint action profile distribution
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, initial_action_profile[0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
    
    # uniform initial joint action profile distribution
    # mu_matrix = np.ones([1, no_actions**no_players])/no_actions**no_players
    
    if "fast" in algorithm:
        game.set_mu_matrix(mu_matrix)
        potential_history = np.zeros((len(epsilons), n_exp, max_iter))
    else:
        potentials = np.zeros((10, max_iter))
        potential_history = np.zeros((len(epsilons), n_exp, max_iter))

    mean_potential_history = np.zeros((len(epsilons), max_iter))
    std = np.zeros((len(epsilons), max_iter))
    
    folder = "delta_" + str(int(delta*1000)).zfill(4)
    
    if noisy_utility:
        betas = np.zeros(len(epsilons))
        for i, eps in enumerate(epsilons):
            betas[i] = game.compute_beta(eps)
        
        eta = 1/2.0/np.max(betas)
        game.gameSetup.eta = eta

    for i in range(n_exp):
        if i % 10 == 0:
            print("     Experiment No. " + str(i+1))
        try:
            payoff_matrix = load_game(folder, i) 
        except:
            payoff_matrix = save_two_player_game(no_actions, delta, i)
        # game.gameSetup.set_payoff_matrix(delta, payoff_matrix)
        game.reset_game(delta = delta, payoff_matrix = payoff_matrix)

        for idx, eps in enumerate(epsilons):
            
            print("Currently testing eps: " + str(eps))
            
            if "fast" in algorithm:

                beta = game.compute_beta(eps)
                game.play(beta = beta)
                
                potential_history[idx][i] = np.transpose(game.expected_value) 
            else:
                for j in range(10):
                    beta = game.compute_beta(eps)
                    game.play(beta = beta)
                
                    potentials[j] = np.transpose(game.potentials_history) 
                
                potential_history[idx][i] = np.mean(potentials, 0) 
                    
        
    for idx, eps in enumerate(epsilons):
        
        mean_potential_history[idx] = np.mean(potential_history[idx], 0)
        index = np.argwhere(mean_potential_history[idx] > 1 - eps)
        print(index)
        std[idx] = np.std(potential_history[idx], 0)
    
    # plot_lines_with_std(mean_potential_history, std, ["01", "005", "0025", '0001'])
    # plt.show(block = False)
    
    # potential_history = []
    # for i in range(n_exp):
    #     potential_history.append(delta_experiments_fast(game, eps=eps))
    #     plt.close('all')
    
    # # mean_potential = np.mean(potential_history, 0)
    
    # # std = np.zeros((len(mean_potential), max_iter))
    # # for i in range(len(mean_potential)):
    # #     std[i] = np.std(potential_history[0], 0)
    
    # # labels = [ r'$\Delta = 0.9$', r'$\Delta = 0.75$', r'$\Delta = 0.5$', r'$\Delta = 0.25$', r'$\Delta = 0.1$', r'$\Phi(a^*) - \epsilon$']

    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "epsExperiment")
    
    if noisy_utility:
        algorithm = algorithm + "_noisy"
        
    potentials_path = os.path.join(root, algorithm + "_potentials.pckl")

    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)
        

def coverageExperiments(no_players = [100, 200, 300, 400, 500], eps = 0.05, max_iter = 10000, n_exp = 100): # [50, 100, 200, 300, 400, 500]
    
    potential_history_ll = np.zeros((len(no_players), n_exp, max_iter))
    potential_history_mll = np.zeros((len(no_players), n_exp, max_iter))

    for j, n in enumerate(no_players):
        gameSetup = CoverageGame(no_resources = 10, no_players = n,  resource_values = [0.05, 0.15, 0.14, 0.1, 0.01, 0.1, 0.11, 0.2, 0.09, 0.05])#[1, 0.7, 0.5, 0.2, 0.2])#rng.uniform(0, 1, 5))
        gameSetup_m = CoverageGame(no_resources = 10, no_players = n,  resource_values = [0.05, 0.15, 0.14, 0.1, 0.01, 0.1, 0.11, 0.2, 0.09, 0.05], symmetric = True)#[1, 0.7, 0.5, 0.2, 0.2])#rng.uniform(0, 1, 5))
        game_ll = Game(gameSetup, algorithm = "log_linear", max_iter = max_iter)    
        game_mll = Game(gameSetup_m, algorithm = "modified_log_linear", max_iter = max_iter)    

        initial_action_profile =  np.array([0]*n) #rng.integers(0, 5, size = game.gameSetup.no_players)
        beta_t = game_ll.compute_beta(eps)
        
        for i in range(n_exp):
            game_ll.reset_game()
            game_ll.play(initial_action_profile = initial_action_profile, beta = beta_t, gamma = 0)

            game_mll.reset_game()
            game_mll.play(initial_action_profile = initial_action_profile, beta = beta_t, gamma = 0)

            potential_history_ll[j][i] = np.transpose(game_ll.potentials_history)
            potential_history_mll[j][i] = np.transpose(game_mll.potentials_history)

    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "CoverageProblem")
        
    potentials_ll_path = os.path.join(root, game_ll.algorithm + "_potentials.pckl")
    potentials_mll_path = os.path.join(root, game_mll.algorithm + "_potentials.pckl")
    
    print(potential_history_ll)
    
    with open(potentials_ll_path, 'wb') as f:
        pickle.dump(potential_history_ll, f, pickle.HIGHEST_PROTOCOL)
    with open(potentials_mll_path, 'wb') as f:
        pickle.dump(potential_history_mll, f, pickle.HIGHEST_PROTOCOL)

def temp(no_players = [100, 200, 300, 400, 500], eps = 0.05, max_iter = 10000, n_exp = 100): # [50, 100, 200, 300, 400, 500]
    
    potential_history_ll = np.zeros((len(no_players), n_exp, max_iter))
    potential_history_mll = np.zeros((len(no_players), n_exp, max_iter))

    for j, n in enumerate(no_players):
        gameSetup = CoverageGame(no_resources = 10, no_players = n,  resource_values = [0.05, 0.15, 0.14, 0.1, 0.01, 0.1, 0.11, 0.2, 0.09, 0.05])#[1, 0.7, 0.5, 0.2, 0.2])#rng.uniform(0, 1, 5))
        gameSetup_m = CoverageGame(no_resources = 10, no_players = n,  resource_values = [0.05, 0.15, 0.14, 0.1, 0.01, 0.1, 0.11, 0.2, 0.09, 0.05], symmetric = True)#[1, 0.7, 0.5, 0.2, 0.2])#rng.uniform(0, 1, 5))
        game_ll = Game(gameSetup, algorithm = "log_linear", max_iter = max_iter)    
        game_mll = Game(gameSetup, algorithm = "modified_log_linear", max_iter = max_iter) 
        
        print("Asymmetric" + str(n))
        print(game_ll.gameSetup.max_potential)
        print(game_ll.gameSetup.min_potential)
        print("Symmetric" + str(n))
        print(game_mll.gameSetup.max_potential)
        print(game_mll.gameSetup.min_potential)
        
        
        
def save_two_player_game(no_actions, delta, no_game):
    
    payoff_matrix = generate_two_plateau_payoff_matrix(delta = delta, no_actions = no_actions)
    
    root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "IdenticalInterest", "games", "delta_" + str(int(delta*1000)).zfill(4))
        
    game_path = os.path.join(root, "game_" + str(no_game) + ".pckl")
    
    print(game_path)
    
    with open(game_path, 'wb') as f:
        pickle.dump(payoff_matrix, f, pickle.HIGHEST_PROTOCOL)

    return payoff_matrix


def runExperiments():
    
    # deltaExperiments(max_iter = 2000000)
    
    # epsExperiments(max_iter = 3000000)
    
    # deltaExperiments(algorithm = "log_linear_binary_fast", max_iter = 2000000)    
    
    # epsExperiments(algorithm = "log_linear_binary_fast", max_iter = 3000000)
    
    # deltaExperiments(algorithm = "log_linear", noisy_utility = True, max_iter = 1000000)
    
    epsExperiments(algorithm = "log_linear", noisy_utility = True, max_iter = 2000000, n_exp = 5)    

    # coverageExperiments(n_exp = 10)