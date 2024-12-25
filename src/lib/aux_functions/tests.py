from lib.aux_functions.experiments import *
from lib.games import *
from lib.player import *
from lib.games.trafficrouting import CongestionGame
import matplotlib.pyplot as plt

def test_custom_game():
    
    action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    no_players = 2
    
    firstNE = np.array([2,2])
    secondNE = np.array([7,7])
    
    delta = 0.25
    b = 1 - delta
    c = 1 - delta
    l = 0.25
    
    payoff_firstNE = np.array([[l, 0.75*b, 0.75*b, 0.75*b, l], [0.75*b, 0.9*b, b, 0.9*b, 0.75*b], [0.75*b, b, 1, b, 0.75*b], [0.75*b, 0.9*b, b, 0.9*b, 0.75*b], [l, 0.75*b, 0.75*b, 0.75*b, l]])
    payoff_secondNE = np.array([[l, 0.5*c, 0.5*c, 0.5*c, l], [0.5*c, 0.75*c, 0.9*c, 0.75*c, 0.5*c], [0.5*c, 0.9*c, c, 0.9*c, 0.5*c], [0.5*c, 0.75*c, 0.9*c, 0.75*c, 0.5*c], [l, 0.5*c, 0.5*c, 0.5*c, l]])
    payoff = l * np.ones([len(action_space), len(action_space)])
    payoff[0:5,0:5] = payoff_firstNE
    payoff[-5::,-5::] = payoff_secondNE
    
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff)

    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 66] = 1
    # mu_matrix = np.ones([1, len(action_space)**2])
    # mu_matrix /= np.sum(mu_matrix)7
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e5, mu=mu)
    beta_t = game.compute_beta(0.1)
    
    game.set_mu_matrix(mu_matrix)
    game.play(beta = beta_t)
    print(game.stationary)
    
    stationary = np.reshape(game.stationary,(-1, game.gameSetup.no_actions))
    plot_payoff(stationary)
    plot_potential(game.expected_value)
    
    plot_payoff(game.gameSetup.payoff_player_1)
    plt.show(block = False)
    plt.pause(20)
    plt.close()
     
    beta_experiments_fast(game, save = False, folder = "WEEK 5", file_name = "betas_experiment_fast_uniform_2")
    
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
    
def test_log_linear():

    action_space = [0, 1, 2, 3, 4, 5]
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    
    delta = 0.25
    # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta = delta, no_actions = len(action_space), trench = 0.1)
    payoff_matrix = generate_two_plateau_payoff_matrix(delta = delta, no_actions = len(action_space))

            
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta, payoff_matrix = payoff_matrix)
    
    game = Game(gameSetup, algorithm = "log_linear",  max_iter = 2000, mu=mu)
    game.set_initial_action_profile(secondNE)

    # game.set_initial_action_profile(np.array([1,4]))

    potentials_history = np.zeros((1, game.max_iter))
    beta_t = game.compute_beta(1e-1)
    
    game.play(beta = beta_t)
    potentials_history = game.potentials_history
    
    plot_potential(potentials_history)
    plt.show()

def test_log_linear_t(delta = 0.25):
    action_space = [0, 1, 2, 3, 4, 5]
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    
    payoff_matrix = generate_two_plateau_payoff_matrix(delta = delta, no_actions = len(action_space))
            
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta, payoff_matrix = payoff_matrix)
    
    game = Game(gameSetup, algorithm = "log_linear_t",  max_iter = 50000, mu=mu)
    game.set_initial_action_profile(secondNE)

    game.set_initial_action_profile(np.array([1,4]))

    potentials_history = np.zeros((10, game.max_iter))
    beta_t = game.compute_beta(1e-1)
    
    for i in range(10):
        game.play(beta = beta_t)
        potentials_history[i] = np.transpose(game.potentials_history).copy()
       
    mean_potential_history = np.mean(potentials_history, 0)
    
    std = np.std(potentials_history, 0)
    plot_payoff(game.gameSetup.payoff_player_1)
    plot_potential(mean_potential_history)
    plot_potential_with_std(mean_potential_history, std)

    plt.show(block = False)
    plt.pause(60)
    plt.close()

def test_mwu():
    
    action_space = np.arange(0,6)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([5,5])
    
    delta = 0.25
    payoff_matrix = generate_two_plateau_hard_payoff_matrix(delta = delta, trench = 0.0)
            
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta, payoff_matrix = payoff_matrix)
    
    game = Game(gameSetup, algorithm = "multiplicative_weight",  max_iter = 10000, mu=mu)
    game.set_initial_action_profile(secondNE)

    game.set_initial_action_profile(np.array([1,4]))

    potentials_history = np.zeros((1, game.max_iter))
    beta_t = game.compute_beta(1e-1)
    
    game.play(beta = beta_t)
    potentials_history = game.potentials_history
    
    plot_potential(potentials_history)
    plt.show()
    
def test_best_response(delta = 0.25):
    
    action_space = [0, 1, 2, 3]
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
            
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta)
    
    game = Game(gameSetup, algorithm = "best_response", max_iter = 5000, mu=mu)
    game.set_initial_action_profile(secondNE)
    game.set_initial_action_profile(np.array([1,3]))

    potentials_history = np.zeros((10, game.max_iter))
    
    for i in range(10):
        game.play()
        potentials_history[i] = np.transpose(game.potentials_history).copy()

    mean_potential_history = np.mean(potentials_history, 0)
    std = np.std(potentials_history, 0)

    plot_payoff(game.gameSetup.payoff_player_1)
    plot_potential(mean_potential_history)
    plot_potential_with_std(mean_potential_history, std)
    
    print(game.action_profile)
    plt.show(block = False)
    plt.pause(20)
    plt.close()
         
def test_alpha_best_response(initial_action_profile):
    action_space = [0, 1, 2, 3, 4, 5]
    no_players = 2
    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    
    delta = 0.25
    payoff_matrix = generate_two_plateau_payoff_matrix(delta)
    
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    
    game = Game(gameSetup, algorithm = "alpha_best_response", max_iter = 1e4, mu=mu)
    game.set_initial_action_profile(initial_action_profile)

    game.play()
    
    plot_payoff(game.gameSetup.payoff_player_1)
    plot_potential(game.potentials_history)
    
    plt.show(block = False)
    plt.pause(60)
    plt.close()

def test_epsilon(initial_action_profile):
    
    action_space = [0, 1, 2, 3, 4, 5]
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    delta = 0.05
    payoff_matrix = generate_two_plateau_payoff_matrix(delta)
    plot_payoff(payoff_matrix)
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    # mu_matrix = np.zeros([1, len(action_space)**2])
    # mu_matrix[0, 15] = 1
    
    mu_matrix = np.ones([1, len(action_space)**2])
    mu_matrix /= np.sum(mu_matrix)
    
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e7, mu=mu)
    beta_t = game.compute_beta(0.1)
    
    game.set_mu_matrix(mu_matrix)
    game.play(beta = beta_t)
    print(beta_t)

    print("Stationary distribution: ")
    print(game.stationary)
    
    stationary = np.reshape(game.stationary,(-1, game.gameSetup.no_actions))
    
    plot_payoff(game.gameSetup.P, title = "Transition matrix")
    plot_payoff(stationary, title = "Stationary distribution")
    plot_potential(game.expected_value)
    
    plt.show(block = False)
    plt.pause(60)
    plt.close()

def test_two_value_game():
    action_space = [0, 1, 2, 3, 4, 5]
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    delta = 0.25
    payoff_matrix = generate_two_value_payoff_matrix(delta)
        
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 15] = 1
    
    # mu_matrix = np.ones([1, len(action_space)**2])
    # mu_matrix /= np.sum(mu_matrix)
    
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1000, mu=mu)
    beta_t = game.compute_beta(0.1)
    
    game.set_mu_matrix(mu_matrix)
    game.play(beta = beta_t)
    
    print("Stationary distribution: ")
    print(game.stationary)
    
    stationary = np.reshape(game.stationary,(-1, game.gameSetup.no_actions))
    
    plot_payoff(game.gameSetup.P.todense(), title = "Transition matrix")
    plot_payoff(stationary, title = "Stationary distribution")
    plot_potential(game.expected_value)
    
    plt.show(block = False)
    plt.pause(60)
    plt.close()

def test_two_plateau_diagonal_game():
    
    action_space = [0, 1, 2, 3, 4, 5, 6, 7]
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    delta = 0.3
    trench = 0.001
    scale_factor = 1000
    
    payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, len(action_space), trench)
    # plot_payoff(payoff_matrix)
    # plt.show()
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    # mu_matrix = np.zeros([1, len(action_space)**2])
    # mu_matrix[0, 15] = 1
    
    mu_matrix = np.ones([1, len(action_space)**2])
    mu_matrix /= np.sum(mu_matrix)
    
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e6, mu=mu)
    beta_t = game.compute_beta(0.1)
    
    game.set_mu_matrix(mu_matrix)
    game.play(beta = beta_t, scale_factor = scale_factor)
    
    print(beta_t)
    print(game.compute_t(0.1))
    print("Stationary distribution: ")
    print(game.stationary)
    
    stationary = np.reshape(game.stationary,(-1, game.gameSetup.no_actions))
    save = True
    folder = "WEEK 7"
    setup = "two_plateau_diagonal_no_actions_" + str(len(action_space)) + "delta_" + str(delta) + "_trench_" + str(trench) + "_scaling_factor_" + str(scale_factor)
    plot_payoff(payoff_matrix, save = save, folder = folder, file_name = "Payoff_matrix_" + setup) 
    plot_payoff(game.gameSetup.P, title = "Transition matrix")
    plot_payoff(stationary, title = "Stationary distribution")
    plot_potential(game.expected_value, save = save, folder = folder, title = "Expected potential value", file_name = "Expected_pot_" + setup)
    
    plt.show(block = False)
    plt.pause(60)
    plt.close() 

def test_transition_matrix():
    
    action_space = [0, 1, 2, 3]
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    delta = 0.25
    payoff_matrix = generate_two_plateau_payoff_matrix_multi(delta, no_actions, no_players)
    
    print("payoff_matrix_generated")    
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 15] = 1
    
    # mu_matrix = np.ones([1, len(action_space)**2])
    # mu_matrix /= np.sum(mu_matrix)
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e6, mu=mu)
    beta_t = game.compute_beta(0.1)
    
    transition_matrix_sparse = gameSetup.formulate_transition_matrix_sparse(beta_t)
    print("transition_matrix_formulated")
    # game.set_mu_matrix(mu_matrix)
    # game.play(beta = beta_t)
    
    # print("Stationary distribution: ")
    # print(game.stationary)
    
    # stationary = np.reshape(game.stationary,(-1, game.gameSetup.no_actions))
    
    # plot_payoff(game.gameSetup.P, title = "Transition matrix")
    # plot_payoff(stationary, title = "Stationary distribution")
    # plot_potential(game.expected_value)
    
    # plt.show(block = False)
    # plt.pause(60)
    # plt.close()
   
def test_generalisation():
    action_space = [0, 1, 2, 3, 4, 5]
    # action_space = [0, 1]
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1, 1])
    secondNE = np.array([4, 4])

    delta = 0.25
    
    payoff_matrix = generate_two_plateau_payoff_matrix(delta, no_actions = len(action_space))
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    
    mu_matrix = np.ones([1, len(action_space)**2])
    mu_matrix /= np.sum(mu_matrix)
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e5, mu=mu)
    beta_t = game.compute_beta(0.1)
    
    game.set_mu_matrix(mu_matrix)
    P = gameSetup.formulate_transition_matrix(beta_t)
    plot_payoff(P)
    plot_payoff(gameSetup.P)
    game.play(beta = beta_t)
    print(game.stationary)
    
    stationary = np.reshape(game.stationary,(-1, game.gameSetup.no_actions))
    plot_payoff(stationary)
    plot_potential(game.expected_value)
    
    # plot_payoff(game.gameSetup.payoff_player_1)
    # plt.show(block = False)
    # plt.pause(20)
    # plt.close()
    plt.show()

def test_symmetric_payoff():
    action_space = np.arange(0, 4)
    
    no_players = 3

    delta = 0.25

    payoff_matrix = generate_two_plateau_payoff_matrix_multi(delta, len(action_space), no_players)

    sym = make_symmetric_nd(payoff_matrix)
    test1 = np.array([1, 1, 1])
    test2 = np.array([2, 1, 1])
    test3 = np.array([1, 2, 1])
    test4 = np.array([1, 1, 2])
    
    print(sym[tuple(test1)])
    print(sym[tuple(test2)])
    print(sym[tuple(test3)])
    print(sym[tuple(test4)])
    
def test_two_value_payoff_matrix():
       
    action_space = np.arange(0, 6)
    
    no_players = 2

    delta = 0.25

    payoff_matrix = generate_two_value_payoff_matrix(delta, len(action_space), no_players)
    
    plot_payoff(payoff_matrix)
    
    plt.show()
    
def test_two_plateau_diagonal_payoff_matrix():
       
    action_space = np.arange(0, 35)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])
    initial_action_profile = secondNE
    
    delta = 0.3
    max_iter = 250000
    payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta, len(action_space), trench = 0.55)
    
    plot_payoff(payoff_matrix)
    
    gameSetup = IdenticalInterestGame(action_space, no_players , firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = max_iter, mu=mu)
    game.set_initial_action_profile(initial_action_profile)

    potentials_history = np.zeros((1, max_iter))
    beta_t = game.compute_beta(1e-1)
    
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, [0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
    game.set_mu_matrix(mu_matrix)
    
    game.play(beta = beta_t, scale_factor = 1)
    
    potentials_history = game.expected_value
    
    plot_potential(potentials_history)
    
    plt.show()

def test_two_plateau_hard_payoff_matrix():
          
    action_space = np.arange(0, 6)
    no_actions = len(action_space)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])
    initial_action_profile = secondNE
    
    delta = 0.25
    max_iter = 10000
    payoff_matrix = generate_two_plateau_hard_payoff_matrix(delta, trench = 0.1)
    
    plot_payoff(payoff_matrix)
    
    gameSetup = IdenticalInterestGame(action_space, no_players , firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = max_iter, mu=mu)
    game.set_initial_action_profile(initial_action_profile)

    potentials_history = np.zeros((1, max_iter))
    beta_t = game.compute_beta(1e-1)
    
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, [0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
    game.set_mu_matrix(mu_matrix)
    
    game.play(beta = beta_t, scale_factor = 10000)
    
    potentials_history = game.expected_value
    
    plot_potential(potentials_history)
    
    plt.show()

def test_multipleplayers():
    action_space = np.arange(0, 4)
    # action_space = [0, 1]
    no_actions = len(action_space)
    no_players = 12
    
    # firstNE = np.array([0, 0, 0, 0, 0])
    # secondNE = np.array([1, 1, 1, 1, 1])
    
    # firstNE = np.array([0, 0, 0, 0, 0, 0])
    # secondNE = np.array([1, 1, 1, 1, 1, 1])
    
    firstNE = [1]*no_players
    secondNE = [no_actions - 2]*no_players

    delta = 0.25
    
    # payoff_matrix = (1-delta)*np.ones([no_actions] * no_players)
    # payoff_matrix[tuple(secondNE)] = 1
    payoff_matrix = generate_two_plateau_payoff_matrix_multi(delta, len(action_space), no_players)
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    
    mu_matrix = np.ones([1, len(action_space)**no_players])
    mu_matrix /= np.sum(mu_matrix)
    
    game = Game(gameSetup, algorithm = "log_linear", max_iter = 2*1e5, mu=mu)
    beta_t = game.compute_beta(0.1)
    game.set_initial_action_profile(np.array(secondNE))
    # mu_matrix = np.zeros([1, len(action_space)**no_players])
    # mu_matrix[0, secondNE[0]*game.gameSetup.no_actions + secondNE[1]] = 1
    
    game.set_mu_matrix(mu_matrix)
    
    # P = gameSetup.formulate_transition_matrix(beta_t)
    # test = np.linalg.matrix_power(game.gameSetup.P.todense(), 10)
    
    print(game.compute_t(0.1))
    # plot_payoff((P @ P).todense())
    # plot_payoff(test)

    game.play(beta = beta_t)
        
    # print(game.stationary.shape)
    # plot_payoff(game.stationary)
    plot_potential(game.potentials_history)
    
    # # plot_payoff(game.gameSetup.payoff_player_1)
    plt.show(block = False)
    plt.pause(20)
    plt.close()
    # plt.show()
 
def mu_congestion(profile):
    if (profile == np.zeros(len(profile))).all():
        return 1
    return 0

def test_congestion_game():
    
    # gameSetup = CongestionGame("SiouxFallsSymmetric", 4, modified = True, modified_no_players=4)
    gameSetup = CongestionGame()

    
    gameSetup.travel_time(0, 0, np.zeros((gameSetup.no_players - 1)).astype(int))
    
    # plot_network(gameSetup.network)
    
    # plt.show(block = False)
    # plt.pause(1)
    # plt.close()
        
    # game = Game(gameSetup, algorithm = "log_linear", max_iter = 2000, mu = mu_congestion)    
    game = Game(gameSetup, algorithm = "exponential_weight_annealing", max_iter = 10000, mu = mu_congestion)

    print(game.gameSetup.delta)
    beta_t = game.compute_beta(0.1)
    print(beta_t)
    initial_action_profile =  np.array([3]*game.gameSetup.no_players) #rng.integers(0, 5, size = game.gameSetup.no_players)
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
    #                           0 0 0 0 0 0 0 0 1 0 3 0 2 1 0 3 0 0 0 0 2 1 1 1 1 0 0 0 0 0 0 0 2 0 3 0 0
    #                           0 0 2 2 0 1 0 0 0 0] #np.array([0]*game.gameSetup.no_players)
    print(game.gameSetup.potential_function(initial_action_profile))
    game.play(initial_action_profile = initial_action_profile, beta = beta_t)
    
    plot_potential(game.potentials_history)
    
    print(game.action_profile)
    # print(game.gameSetup.objective(game.action_profile))
    plt.grid()
    plt.show()
    plt.close()
    
    plt.plot(game.objectives_history)
    plt.grid()
    plt.show()
    plt.close()
        
def test_modified_log_linear():
    gameSetup = CongestionGame("SiouxFallsSymmetric", 10, modified = True, modified_no_players=200)
    
    gameSetup.travel_time(0, 0, np.zeros((gameSetup.no_players - 1)).astype(int))
    
    # plot_network(gameSetup.network)
    
    # plt.show(block = False)
    # plt.pause(1)
    # plt.close()
        
    game = Game(gameSetup, algorithm = "log_linear", max_iter = 5000, mu = mu_congestion)
    
    print(game.gameSetup.delta)
    beta_t = game.compute_beta(0.1)
    print(beta_t)
    initial_action_profile = rng.integers(0, 10, size = game.gameSetup.no_players) #np.array([9]*game.gameSetup.no_players) 
    # initial_action_profile[game.gameSetup.no_actions - 1] = game.gameSetup.no_players
    
    print(initial_action_profile)
    
    # print(game.gameSetup.potential_function(initial_action_profile))
    game.play(initial_action_profile = initial_action_profile, beta = beta_t)
    
    plot_potential(game.potentials_history)
    
    print(game.action_profile)
    # print(game.gameSetup.objective(game.action_profile))
    plt.grid()
    plt.show()
    plt.close()
    
    
    game = Game(gameSetup, algorithm = "modified_log_linear", max_iter = 5000, mu = mu_congestion)
    
    print(game.gameSetup.delta)
    beta_t = game.compute_beta(0.1)
    print(beta_t)
    
    print(initial_action_profile)
    
    # print(game.gameSetup.potential_function(initial_action_profile))
    game.play(initial_action_profile = initial_action_profile, beta = beta_t)
    
    plot_potential(game.potentials_history)
    
    print(game.action_profile)
    # print(game.gameSetup.objective(game.action_profile))
    plt.grid()
    plt.show()
    plt.close()

def test_exponential_weight_annealing():

    action_space = [0, 1, 2, 3, 4, 5]
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    
    delta = 0.1
    # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta = delta, no_actions = len(action_space), trench = 0.1)
    payoff_matrix = generate_two_plateau_payoff_matrix(delta = delta, no_actions = len(action_space))

            
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta, payoff_matrix = payoff_matrix)
    
    game = Game(gameSetup, algorithm = "exponential_weight_annealing",  max_iter = 100000, mu=mu)
    game.set_initial_action_profile(secondNE)

    game.set_initial_action_profile(np.array([1,4]))

    potentials_history = np.zeros((1, game.max_iter))
    beta_t = game.compute_beta(1e-1)
    
    game.play(beta = beta_t)
    potentials_history = game.potentials_history
    
    plot_potential(potentials_history)
    plt.show()

def test_exp3p():

    # action_space = [0, 1, 2, 3, 4, 5 ]#, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    action_space = np.arange(0,6)
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    initial_action_profile = np.array([4, 5])
    delta = 0.25
    # payoff_matrix = generate_two_plateau_diagonal_payoff_matrix(delta = delta, no_actions = len(action_space), trench = 0.4)
    payoff_matrix = generate_two_plateau_payoff_matrix(delta = delta, no_actions = len(action_space))

            
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta, payoff_matrix = payoff_matrix)
    
    game = Game(gameSetup, algorithm = "exp3p",  max_iter = 100000, mu=mu)

    game.set_initial_action_profile(initial_action_profile)

    potentials_history = np.zeros((1, game.max_iter))
    beta_t = game.compute_beta(1e-1)
    
    for i in range(1):
        game.play(beta = beta_t)
        potentials_history[i] = np.transpose(game.potentials_history)
    
    mean_pot = np.mean(potentials_history, 0)
    plot_payoff(payoff_matrix)
    plot_potential(mean_pot)
    plt.show()