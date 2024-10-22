import numpy as np
from game import Game, IdenticalInterestGame, rng 
from aux_functions.plot import *
from aux_functions.experiments import *
import cProfile
import sys

if __name__ == '__main__':
    
    np.set_printoptions(threshold=sys.maxsize)
    
    # custom_game_experiments(0.1)
    # custom_game_alg_experiments()
    # test_log_linear_t()
    # compare_log_linear_t()
    # epsilon_experiments_fast
    # test_custom_game()
    # epsilon_experiments(0.25)
    custom_game_no_actions_experiments()
    # cProfile.run('main_simulation_experiment()')