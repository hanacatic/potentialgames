import numpy as np
from lib.aux_functions.plot import *
from lib.aux_functions.experiments import *
from lib.aux_functions.tests import *
from lib.aux_functions.visualise_data import *
import cProfile
import sys

if __name__ == '__main__':
    
    # temp()
    runExperiments()
    # test_log_linear()
    
    # test_success_probability()
    
    # test_coverage_game()
    
    # custom_game_no_players_experiments()
    
    # visualise_coverage("CoverageProblem", "log_linear_potentials.pckl", "modified_log_linear_potentials.pckl", True, "Paper/Experiments", "coverage", "modifiedcoverage")
    # coverageExperiments(n_exp = 10)
    # visualise_deltas_data("deltaExperiment", "log_linear_binary_fast_potentials.pckl", 550000, True, "Paper/Experiments", "deltaExperimentBinaryFinal")
    # visualise_deltas_data("deltaExperiment", "log_linear_fast_potentials.pckl", 550000, True, "Paper/Experiments", "deltaExperimentFinal")
    # visualise_deltas_data("deltaExperiment", "log_linear_noisy_potentials.pckl", 550000, True, "Paper/Experiments", "deltaExperimentNoisyFinal")

    # visualise_eps_data("epsExperiment", "log_linear_fast_potentials.pckl", 2000000, True, "Paper/Experiments", "epsExperimentFinal")
    # visualise_eps_data("epsExperiment", "log_linear_binary_fast_potentials.pckl", 2000000, True, "Paper/Experiments", "epsExperimentBinaryFinal")
    # visualise_eps_data("epsExperiment", "log_linear_noisy_potentials.pckl", 2000000, True, "Paper/Experiments", "epsExperimentNoisyFinal")
    # test_log_linear_fast()
    # save_two_player_game(10, 0.075, 30)
    # save_two_player_game(10, 0.1, 30)
    # save_two_player_game(10, 0.15, 30)
    