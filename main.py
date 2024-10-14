import numpy as np
from game import Game, IdenticalInterestGame, rng 
from aux_functions.plot import *
from aux_functions.experiments import *
import cProfile
import sys

if __name__ == '__main__':
    
    np.set_printoptions(threshold=sys.maxsize)
    
    test_log_linear_t()
    # test_custom_game()
    # custom_game_experiments(0.25)
    # cProfile.run('main_simulation_experiment()')