import numpy as np
from lib.aux_functions.plot import *
from lib.aux_functions.experiments import *
from lib.aux_functions.tests import *
import cProfile
import sys

if __name__ == '__main__':
    
    np.set_printoptions(threshold=sys.maxsize)
    
    # test()
    # test_alpha_best_response( np.array([4,4]))
    # custom_game_experiments(0.1)
    # custom_game_alg_experiments()
    # test_log_linear_t()
    # compare_log_linear_t()
    # epsilon_experiments_fast
    # test_custom_game()
    # test_two_value_payoff_matrix()
    # test_two_plateau_diagonal_payoff_matrix()
    # test_two_plateau_diagonal_game()
    # test_two_value_game()
    # epsilon_experiments(0.25)
    # custom_game_no_actions_experiments()
    # custom_game_no_players_experiments()
    # test_symmetric_payoff()
    # test_multipleplayers()
    # cProfile.run('test_transition_matrix()')

    # cProfile.run('test_multipleplayers()')
    # custom_game_experiments()
    # test()
    # test_transition_matrix()
    
    # payoff = generate_two_plateau_diagonal_payoff_matrix_multi(0.25, 6, 10)
    # print(payoff.nbytes)
    # cProfile.run('test_transition_matrix()', sort='ncalls')
    # cProfile.run('test_multipleplayers()', sort='cumtime')
    # cProfile.run('test_log_linear()', sort='cumtime')
    # cProfile.run('test_mwu()', sort='cumtime')
    # cProfile.run('custom_game_alg_experiments()', sort='cumtime')
    # test_two_plateau_hard_payoff_matrix()
    
    # test_mwu()
    
    # custom_game_no_players_sim_experiments()
    test_congestion_game()