import numpy as np
from game import Game,IdenticalInterestGame, rng
from aux_functions.plot import *

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

def delta_experiments_fast(game, deltas = [0.9, 0.75, 0.5, 0.25, 0.1], eps = 0.1, save = False, folder = None, file_name = None, title = 'Average potential'):
       
    beta_t = game.compute_beta(eps)
    plot_payoff(game.gameSetup.payoff_player_1)
    potential_history = np.zeros((6, game.max_iter))
    
    for idx, delta in enumerate(deltas):
        
        print(delta)
        payoff_matrix = generate_exp_payoff_matrix(delta)
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

def generate_exp_payoff_matrix(delta = 0.1):
    
    # no_actions = 10
    
    # firstNE = np.array([2,2])
    # secondNE = np.array([7,7])

    # b = 1 - delta 
       
    # payoff = rng.random(size=np.array([10, 10])) * 0.2 * (1-delta)

    # payoff_firstNE = (rng.random(size=np.array([5, 5]))*0.6 + 0.4) * 0.75 * b
    # payoff_firstNE[1:4,1:4] = (rng.random(size=np.array([3, 3]))*0.2 + 0.8) * 0.9 * b
    
    # payoff_secondNE = (rng.random(size=np.array([5, 5]))*0.65 + 0.35) * 0.5 * (1 - delta)
    # payoff_secondNE[1:4,1:4] = (rng.random(size=np.array([3, 3]))*0.15 + 0.85) * 0.9 * (1-delta)

    # payoff[0:5,0:5] = payoff_firstNE
    # payoff[-5::,-5::] = payoff_secondNE
    
    # payoff[firstNE[0], firstNE[1]] = 1
    # payoff[secondNE[0], secondNE[1]] = 1 - delta
    
    
    no_actions = 6
    
    firstNE = np.array([1,1])
    secondNE = np.array([4,4])

    b = 1 - delta 
       
    payoff = rng.random(size=np.array([6, 6])) * 0.25 * (1-delta)

    # payoff_firstNE = (rng.random(size=np.array([5, 5]))*0.6 + 0.4) * 0.75 * b
    payoff_firstNE= (rng.random(size=np.array([3, 3]))*0.3 + 0.7) * b
    
    # payoff_secondNE = (rng.random(size=np.array([5, 5]))*0.65 + 0.35) * 0.5 * (1 - delta)
    payoff_secondNE = (rng.random(size=np.array([3, 3]))*0.4 + 0.6) * (1-delta)

    payoff[0:3,0:3] = payoff_firstNE
    payoff[-3::,-3::] = payoff_secondNE
    
    payoff[firstNE[0], firstNE[1]] = 1
    payoff[secondNE[0], secondNE[1]] = 1 - delta
    
    return payoff

def test_custom_game():
    
    action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
    
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta = delta, payoff_matrix = payoff)

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
    action_space = [0, 1, 2, 3]

    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
            
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta)
    
    game = Game(gameSetup, algorithm = "log_linear_t",  max_iter = 5000, mu=mu)
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

    plt.show(block = False)
    plt.pause(60)
    plt.close()

def test_best_response(delta = 0.25):
    
    action_space = [0, 1, 2, 3]

    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
            
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta)
    
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

    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    delta = 0.25
    payoff_matrix = generate_exp_payoff_matrix(delta)
        
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

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

    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    
    delta = 0.25
    payoff_matrix = generate_exp_payoff_matrix(delta)
    
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    
    game = Game(gameSetup, algorithm = "alpha_best_response", max_iter = 1e4, mu=mu)
    game.set_initial_action_profile(initial_action_profile)

    game.play()
    
    plot_payoff(game.gameSetup.payoff_player_1)
    plot_potential(game.potentials_history)
    
    plt.show(block = False)
    plt.pause(60)
    plt.close()

def custom_game_alg_experiments():
    print("YES")
    
def main_simulation_experiment():
    # action_space = [0, 1, 2, 3]
    
    # firstNE = np.array([1,1])
    # secondNE = np.array([3,3])
    # initial_action_profile = secondNE
    
    # delta = 0.25
    
    # gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta)
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
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    initial_action_profile = secondNE
    
    delta = 0.25
    
    save = False
    folder = 'WEEK 5'
    
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta)
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
    
    # firstNE = np.array([1,1])
    # secondNE = np.array([3,3])
    # initial_action_profile = secondNE
    
    # delta = 0.25
    
    # gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta)
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
    
    # firstNE = np.array([1,1])
    # secondNE = np.array([3,3])
    
    # save = False 
    # folder = 'WEEK 4'
    # title = 'Average potential'
    # n_exp = 10
    
    # # mean_potential_history = np.zeros((1, game.max_iter))
        
    # gameSetup = RandomIdenticalInterestGame(action_space, firstNE, secondNE, 0.25)
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

    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    
    payoff_matrix = generate_exp_payoff_matrix(0.25)
    # payoff_matrix = np.zeros([6,6])
    # payoff_matrix[0,0] = 1
    # payoff_matrix[1,1] = 1
    
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    
    mu_matrix = np.ones([1, len(action_space)**2])
    mu_matrix /= np.sum(mu_matrix)
    
    mu_matrix = np.zeros([1, len(action_space)**2])
    # mu_matrix[0, 66] = 1
    mu_matrix[0, 28] = 1
    
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e6, mu=mu)
    game.set_mu_matrix(mu_matrix)

    # beta_t = game.compute_beta(0.1)
    # game.play(beta = beta_t)
    
    # beta_experiments_fast(game, save = True, folder = "WEEK 5", file_name = "betas_experiment_fast_secondNE_real_symmetrical_2")
    
    
    # mu_matrix = np.zeros([1, len(action_space)**2])
    # # mu_matrix[0, 66] = 1
    # mu_matrix[0, 28] = 1
    # game.set_mu_matrix(mu_matrix)

    # beta_experiments_fast(game, save = True, folder = "WEEK 5", file_name = "betas_experiment_fast_second_NE_real_2")

    # mu_matrix = np.zeros([1, len(action_space)**2])
    # # mu_matrix[0, 66] = 1
    # mu_matrix[0, 5] = 1
    # game.set_mu_matrix(mu_matrix)
    
    # # beta_experiments_fast(game, save = True, folder = "WEEK 5", file_name = "betas_experiment_fast_valley_real_2")

    # payoff_matrix = generate_exp_payoff_matrix(0.25)
    # gameSetup.set_payoff_matrix(payoff_matrix)
    # game.reset_game()
    
    # # mu_matrix = np.zeros([1, len(action_space)**2])
    # # # mu_matrix[0, 66] = 1
    # # mu_matrix[0, 5] = 1
    # # game.set_mu_matrix(mu_matrix)
    epsilon_experiments_fast(game, save = True, folder = "WEEK 6", file_name = "eps_experiment_fast_faster_unifrom_real_less_zoom")
    
    # # mu_matrix = np.zeros([1, len(action_space)**2])
    # # # mu_matrix[0, 66] = 1
    # # mu_matrix[0, 28] = 1
    # # game.set_mu_matrix(mu_matrix)
    
    # # epsilon_experiments_fast(game, save = True, folder = "WEEK 5", file_name = "eps_experiment_fast_second_NE_real_4")

    # mu_matrix = np.ones([1, len(action_space)**2])
    # mu_matrix /= np.sum(mu_matrix)
    # game.set_mu_matrix(mu_matrix)
    
    # epsilon_experiments_fast(game, save = True, folder = "WEEK 5", file_name = "eps_experiment_fast_uniform_real_symmetrical")

    
    # delta_experiments_fast(game, save = True, folder = "WEEK 6", file_name = "delta_experiment_fast_faster_unifrom_real_4")
    
    
    # # mu_matrix = np.zeros([1, len(action_space)**2])
    # # # mu_matrix[0, 66] = 1
    # # mu_matrix[0, 5] = 1
    # # game.set_mu_matrix(mu_matrix)
    
    # # delta_experiments_fast(game, save = True, folder = "WEEK 5", file_name = "delta_experiment_fast_valley_real_3")
    
    # # mu_matrix = np.zeros([1, len(action_space)**2])
    # # # mu_matrix[0, 66] = 1
    # # mu_matrix[0, 28] = 1
    # # game.set_mu_matrix(mu_matrix)
    
    # # delta_experiments_fast(game, save = True, folder = "WEEK 5", file_name = "delta_experiment_fast_secondNE_real_3")
    
    # # mu_matrix = np.ones([1, len(action_space)**2])
    # # mu_matrix /= np.sum(mu_matrix)
    # # game.set_mu_matrix(mu_matrix)
    
    # delta_experiments_fast(game, save = True, folder = "WEEK 5", file_name = "delta_experiment_fast_uniform_real_symmetrical")
    
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

def epsilon_experiments(delta):
     
    # action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # firstNE = np.array([2,2])
    # secondNE = np.array([7,7])
    
    action_space = [0, 1, 2, 3, 4, 5]

    firstNE = np.array([1,1])
    secondNE = np.array([4,4])
    
    payoff_matrix = generate_exp_payoff_matrix(0.25)
    # payoff_matrix = np.zeros([6,6])
    # payoff_matrix[0,0] = 1
    # payoff_matrix[1,1] = 1
    
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)
    
    mu_matrix = np.ones([1, len(action_space)**2])
    mu_matrix /= np.sum(mu_matrix)
    
    mu_matrix = np.zeros([1, len(action_space)**2])
    # mu_matrix[0, 66] = 1
    mu_matrix[0, 28] = 1
    
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e6, mu=mu)
    game.set_mu_matrix(mu_matrix)

    epsilon_experiments_fast(game, save = True, folder = "WEEK 6", file_name = "eps_experiment_fast_faster_unifrom_real_scale_1")
    epsilon_experiments_fast(game, save = True, folder = "WEEK 6", scale_factor = 50, file_name = "eps_experiment_fast_faster_unifrom_real_scale_50")
    epsilon_experiments_fast(game, save = True, folder = "WEEK 6", scale_factor = 5000, file_name = "eps_experiment_fast_faster_unifrom_real_scale_5000")
    epsilon_experiments_fast(game, save = True, folder = "WEEK 6", scale_factor = 1000000, file_name = "eps_experiment_fast_faster_unifrom_real_scale_mil")

def test_epsilon(initial_action_profile):
    
    action_space = [0, 1, 2, 3, 4, 5]

    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    delta = 0.05
    payoff_matrix = generate_exp_payoff_matrix(delta)
    plot_payoff(payoff_matrix)
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, delta = delta, payoff_matrix = payoff_matrix)

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