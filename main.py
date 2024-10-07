import numpy as np
from game import Game, AsymmetricalIdenticalInterestGame 
from plot import *

import cProfile

RATIONALITY = 100
EPS = 0.5e-1
    
def mu(action_profile):
    return 1.0/16.0

def beta_experiments(game, n_exp = 10, save = False, folder = None, file_name = None, title = 'Average potential'): 
    
    beta_t = game.compute_beta(EPS)
    # game.set_max_iter(EPS)
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
            
            game.gameSetup.reset_payoff_matrix()
            game.reset_game()
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
    game = Game(game.gameSetup, algorithm = "log_linear_t", mu=mu)
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
        # plt.show()
        plt.show(block = False)
        plt.pause(20)
        plt.close()
        
def delta_experiments(game, deltas = [0.9, 0.75, 0.5, 0.25, 0.1],  n_exp = 10, save = False, folder = None, file_name = None, title = 'Average potential'):
       
    beta_t = game.compute_beta(EPS)
    plot_payoff(game.gameSetup.payoff_player_1)
    mean_potential_history = np.zeros((6, game.max_iter))
    
    for idx, delta in enumerate(deltas):
        
        print(delta)
        game.gameSetup.reset_payoff_matrix(delta)
        game.reset_game()
        plot_payoff(game.gameSetup.payoff_player_1)
        beta_t = game.compute_beta(EPS)
        print(beta_t)
        
        potentials_history = np.zeros((n_exp, game.max_iter))
        for i in range(0, n_exp):
            potentials_history[i] = np.transpose(game.potentials_history).copy()
            game.play(beta=beta_t)
            
            game.gameSetup.reset_payoff_matrix(delta)
            game.reset_game()
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
    mean_potential_history[5] = (1-EPS) * np.ones((1, game.max_iter))
    labels = [ r'$\Delta = 0.9$', r'$\Delta = 0.75$', r'$\Delta = 0.5$', r'$\Delta = 0.25$', r'$\Delta = 0.1$', r'$\Phi(a^*) - \epsilon$']
   
    plot_lines(mean_potential_history, labels, True, title = title, folder = folder, save = save, file_name = file_name)

    if not save:
        # plt.show()
        plt.show(block = False)
        plt.pause(20)
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
            
            game.gameSetup.reset_payoff_matrix()
            game.reset_game()
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
        mean_potential_history[idx + 5] = (1 - eps) * np.ones((1, game.max_iter))
    labels = [ r'$\epsilon = 0.2$', r'$\epsilon = 0.1$', r'$\epsilon = 0.05$', r'$\epsilon = 0.01$', r'$\epsilon = 0.001$']
   
    plot_lines_eps_exp(mean_potential_history, labels, True, title = title, folder = folder, save = save, file_name = file_name)

    if not save:
        plt.show()
        # plt.show(block = False)
        # plt.pause(20)
        # plt.close()

def main():
    action_space = [0, 1, 2, 3]
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    initial_action_profile = secondNE
    
    delta = 0.25
    
    gameSetup = AsymmetricalIdenticalInterestGame(action_space, firstNE, secondNE, delta)
    game = Game(gameSetup, mu=mu)
    game.set_initial_action_profile(initial_action_profile)

    beta_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential beta (3,3) random _')
    
    initial_action_profile = np.array([1,3])
    game.set_initial_action_profile(initial_action_profile)

    beta_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential beta (1,3) random _')

    initial_action_profile = np.array([0,2])
    game.set_initial_action_profile(initial_action_profile)

    beta_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential beta (0,2) random _')
    
    action_space = [0, 1, 2, 3]
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    initial_action_profile = secondNE
    
    delta = 0.25
    
    gameSetup = AsymmetricalIdenticalInterestGame(action_space, firstNE, secondNE, delta)
    game = Game(gameSetup, mu=mu)
    game.set_initial_action_profile(initial_action_profile)

    delta_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential delta (3,3) random _')
    
    initial_action_profile = np.array([1,3])
    game.set_initial_action_profile(initial_action_profile)

    delta_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential delta (1,3) random _')

    initial_action_profile = np.array([0,2])
    game.set_initial_action_profile(initial_action_profile)

    delta_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential delta (0,2) random _')

    action_space = [0, 1, 2, 3]
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    initial_action_profile = secondNE
    
    delta = 0.25
    
    gameSetup = AsymmetricalIdenticalInterestGame(action_space, firstNE, secondNE, delta)
    game = Game(gameSetup, mu=mu)
    game.set_initial_action_profile(initial_action_profile)

    epsilon_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential epsilon (3,3) random _')
    
    initial_action_profile = np.array([1,3])
    game.set_initial_action_profile(initial_action_profile)

    epsilon_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential epsilon (1,3) random _')

    initial_action_profile = np.array([0,2])
    game.set_initial_action_profile(initial_action_profile)

    epsilon_experiments(game, save = True, folder = 'WEEK 4', file_name = 'Average potential epsilon (0,2) random _')

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

def test():
    
    action_space = [0, 1, 2, 3]

    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    # mean_potential_history = np.zeros((1, game.max_iter))
        
    gameSetup = AsymmetricalIdenticalInterestGame(action_space, firstNE, secondNE, 0.25)
    # game = Game(gameSetup, algorithm = "log_linear_t", mu=mu)
    # game.set_initial_action_profile(secondNE)
    
    game = Game(gameSetup, algorithm = "best_response", max_iter = 5000, mu=mu)
    game.set_initial_action_profile(secondNE)
    game.set_initial_action_profile(np.array([1,3]))

    potentials_history = np.zeros((10, game.max_iter))

    
    for i in range(10):
        game.play()
        potentials_history[i] = np.transpose(game.potentials_history).copy()

            
    mean_potential_history = np.mean(potentials_history, 0)
    plot_payoff(game.gameSetup.payoff_player_1)
    plot_potential(mean_potential_history)
    
    print(game.action_profile)
    plt.show(block = False)
    plt.pause(20)
    plt.close()


if __name__ == '__main__':
    
    test()
    # cProfile.run('main()')