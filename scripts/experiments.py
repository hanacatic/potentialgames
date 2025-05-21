import numpy as np
from src.mechanism.game_engine import *
from src.mechanism.game_setup.identical_interest import *
from src.utils.plot import *
from src.mechanism.game_setup.coverage import *
import sparse 

# TODO clean up
# TODO streamline the saving of figures
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
    
    # epsExperiments(max_iter = 100)
    
    # deltaExperiments(algorithm = "log_linear_binary_fast", max_iter = 2000000)    
    
    # epsExperiments(algorithm = "log_linear_binary_fast", max_iter = 3000000)
    
    # deltaExperiments(algorithm = "log_linear", noisy_utility = True, max_iter = 1000000)
    
    # epsExperiments(algorithm = "log_linear", noisy_utility = True, max_iter = 2000000)    

    coverageExperiments(n_exp = 100)