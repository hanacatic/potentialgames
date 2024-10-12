import numpy as np
from game import Game, IdenticalInterestGame 
from plot import *
from helpers import beta_experiments, delta_experiments, epsilon_experiments
import cProfile

RATIONALITY = 100
EPS = 0.5e-1
    
def mu(action_profile):
    return 1.0/16.0

def main():
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
    game = Game(gameSetup, algorithm = "log_linear", max_iter = 1e7, mu=mu)
    
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

def test():
    
    action_space = [0, 1, 2, 3]

    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    # mean_potential_history = np.zeros((1, game.max_iter))
        
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, 0.25, "Symmetrical")
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

def test_transition_matrix():
    
    action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    # mean_potential_history = np.zeros((1, game.max_iter))
        
    gameSetup = IdenticalInterestGame(action_space, firstNE, secondNE, 0.25)
    # game = Game(gameSetup, algorithm = "log_linear_t", mu=mu)
    # game.set_initial_action_profile(secondNE)
    
    mu_matrix = np.zeros([1, len(action_space)**2])
    mu_matrix[0, 15] = 1
    # mu_matrix = np.ones([1, len(action_space)**2])
    # mu_matrix /= np.sum(mu_matrix)
    game = Game(gameSetup, algorithm = "log_linear_fast", max_iter = 1e4, mu=mu)
    beta_t = game.compute_beta(0.01)
    
    game.set_mu_matrix(mu_matrix)
    game.play(beta = beta_t)
    print(game.stationary)
    
    stationary = np.reshape(game.stationary,(-1, game.gameSetup.no_actions))
    plot_payoff(stationary)
    plot_potential(game.expected_value)
    
    print(game.action_profile)
    plt.show(block = False)
    plt.pause(20)
    plt.close()
 
def test_custom_game():
    action_space = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    firstNE = np.array([2,2])
    secondNE = np.array([7,7])
    
    delta = 0.1
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
    beta_t = game.compute_beta(0.01)
    
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

if __name__ == '__main__':
    
    test_custom_game()
    # cProfile.run('main()')