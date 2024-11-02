from aux_functions.experiments import *
from game import *
from player import *

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
    # epsilon_experiments_fast(game, save = True, folder = "WEEK 5", file_name = "eps_experiment_fast_uniform_")
    
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
    
def test_transition_matrix():
    
    action_space = [0, 1, 2, 3, 4, 5]
    no_players = 2
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    delta = 0.25
    payoff_matrix = generate_two_plateau_payoff_matrix(delta)
        
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 15] = 1
    
    # mu_matrix = np.ones([1, len(action_space)**2])
    # mu_matrix /= np.sum(mu_matrix)
    
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e6, mu=mu)
    beta_t = game.compute_beta(0.1)
    
    game.set_mu_matrix(mu_matrix)
    game.play(beta = beta_t)
    
    print("Stationary distribution: ")
    print(game.stationary)
    
    stationary = np.reshape(game.stationary,(-1, game.gameSetup.no_actions))
    
    plot_payoff(game.gameSetup.P, title = "Transition matrix")
    plot_payoff(stationary, title = "Stationary distribution")
    plot_potential(game.expected_value)
    
    plt.show(block = False)
    plt.pause(60)
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
    
    print(sym[tuple(test1)])
    print(sym[tuple(test2)])
    print(sym[tuple(test3)])
    
def test_multipleplayers():
    action_space = np.arange(0, 4)
    # action_space = [0, 1]
    no_actions = len(action_space)
    no_players = 4
    
    # firstNE = np.array([0, 0, 0, 0, 0])
    # secondNE = np.array([1, 1, 1, 1, 1])
    
    # firstNE = np.array([0, 0, 0, 0, 0, 0])
    # secondNE = np.array([1, 1, 1, 1, 1, 1])
    
    firstNE = np.array([1, 1])
    secondNE = np.array([no_actions - 2, no_actions - 2, no_actions - 2])

    delta = 0.25
    
    # payoff_matrix = (1-delta)*np.ones([no_actions] * no_players)
    # payoff_matrix[tuple(secondNE)] = 1
    payoff_matrix = generate_two_plateau_payoff_matrix_multi(delta, len(action_space), no_players)
    gameSetup = IdenticalInterestGame(action_space, no_players, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    
    mu_matrix = np.ones([1, len(action_space)**no_players])
    mu_matrix /= np.sum(mu_matrix)
    
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e4, mu=mu)
    beta_t = game.compute_beta(0.1)
    print(payoff_matrix)
    # mu_matrix = np.zeros([1, len(action_space)**no_players])
    # mu_matrix[0, secondNE[0]*game.gameSetup.no_actions + secondNE[1]] = 1
    
    game.set_mu_matrix(mu_matrix)
    
    P = gameSetup.formulate_transition_matrix(beta_t)
    # test = np.linalg.matrix_power(game.gameSetup.P.todense(), 10)
    
    print(game.compute_t(0.1))
    # plot_payoff((P @ P).todense())
    # plot_payoff(test)

    game.play(beta = beta_t, scale_factor = 100)
        
    print(game.stationary.shape)
    # plot_payoff(game.stationary)
    plot_potential(game.expected_value)
    
    # plot_payoff(game.gameSetup.payoff_player_1)
    # plt.show(block = False)
    # plt.pause(20)
    # plt.close()
    plt.show()
 