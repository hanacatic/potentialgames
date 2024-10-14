import numpy as np
from game import Game, IdenticalInterestGame, rng 
from plot import *
from experiments import *
import cProfile
import sys

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

        
def custom_game_alg_experiments():
    print("YES")
    

if __name__ == '__main__':
    
    np.set_printoptions(threshold=sys.maxsize)

    # test_custom_game()
    test_transition_matrix()
    # test_alpha_best_response(np.array([2,2]))
    # custom_game_experiments(0.25)
    # cProfile.run('main()')