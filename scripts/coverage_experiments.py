import numpy as np

import pickle 

from pathlib import Path

from potentialgames.mechanism.game_setup import CoverageSetup
from potentialgames.mechanism import GameEngine
from potentialgames.utils import logger, compare_lines

"""
Run coverage game experiments for varying numbers of players and save potential histories.
For each specified number of players, this function:
- Sets up asymmetric and symmetric coverage games.
- Loads or computes potential values for the game setup.
- Runs multiple experiments for each configuration, simulating the game for an N iterations.
- Saves the results to disk for later analysis.
- Optionally, computes and plots the mean and standard deviation of the potential histories.
Args:
    no_players (list of int): List of player counts to run experiments for.
    eps (float): Epsilon parameter for computing beta in the log-linear algorithm.
    max_iter (int): Maximum number of iterations per experiment.
    n_exp (int): Number of experiments (runs) per player count.
    verbose (bool): If True, computes and plots statistics of the results.
Saves:
    Pickled numpy arrays of potential histories for both asymmetric and symmetric games.
"""

def coverage_experiment(no_players = [100, 200, 300, 400, 500], eps = 0.05, max_iter = 10000, n_exp = 100, verbose = True):
        
    potential_history_asym = np.zeros((len(no_players), n_exp, max_iter))
    potential_history_sym= np.zeros((len(no_players), n_exp, max_iter))

    repo_root  = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "Coverage"

    for j, n in enumerate(no_players):
        
        gameSetup_asym = CoverageSetup(no_resources = 10, no_players = n, resource_values = [0.05, 0.15, 0.14, 0.1, 0.01, 0.1, 0.11, 0.2, 0.09, 0.05])
        gameSetup_sym = CoverageSetup(no_resources = 10, no_players = n, resource_values = [0.05, 0.15, 0.14, 0.1, 0.01, 0.1, 0.11, 0.2, 0.09, 0.05], symmetric = True)
    
        try:
            potentials_path_asym = data_path / f"coverage_potentials_{n}_asymmetric.pckl"
            delta_path_asym = data_path / f"coverage_delta_{n}_asymmetric.pckl"
            gameSetup_asym.load_data(potentials_path_asym, delta_path_asym)
        except FileNotFoundError:
            logger.info(f"Game setup for {n} players not found, computing potentials.")
            gameSetup_asym.compute_potentials()
            gameSetup_asym.save_data(potentials_path_asym, delta_path_asym)
        try:
            potentials_path_sym = data_path / f"coverage_potentials_{n}_symmetric.pckl"
            delta_path_sym = data_path / f"coverage_delta_{n}_symmetric.pckl"
            gameSetup_sym.load_data(potentials_path_sym, delta_path_sym)
        except FileNotFoundError:
            logger.info(f"Game setup for {n} players not found, computing potentials.")
            gameSetup_sym.compute_potentials()
            gameSetup_sym.save_data(potentials_path_sym, delta_path_sym)
        
        logger.info(f"Running coverage experiment for {n} players")

        game_asym = GameEngine(gameSetup_asym, algorithm = "log_linear", max_iter = max_iter)    
        game_sym= GameEngine(gameSetup_sym, algorithm = "modified_log_linear", max_iter = max_iter)    

        initial_action_profile =  np.array([0]*n) #rng.integers(0, 5, size = game.gameSetup.no_players)
        beta = game_asym.compute_beta(eps)
        
        for i in range(n_exp):
            
            game_asym.reset_game()
            game_asym.play(initial_action_profile = initial_action_profile, beta = beta, gamma = 0)

            game_sym.reset_game()
            game_sym.play(initial_action_profile = initial_action_profile, beta = beta, gamma = 0)

            potential_history_asym[j][i] = np.transpose(game_asym.potentials_history)
            potential_history_sym[j][i] = np.transpose(game_sym.potentials_history)
        
        gameSetup_asym.save_data(potentials_path_asym, delta_path_asym)
        gameSetup_sym.save_data(potentials_path_sym, delta_path_sym)
    
    potentials_asym_path = repo_root / "data" / "experiments" / "coverage" / f"{game_asym.algorithm}_potentials.pckl"
    potentials_asym_path.parent.mkdir(parents=True, exist_ok=True)
    
    potentials_sym_path = repo_root / "data" / "experiments" / "coverage" / f"{game_sym.algorithm}_potentials.pckl"
    potentials_sym_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(potentials_asym_path, 'wb') as f:
        pickle.dump(potential_history_asym, f, pickle.HIGHEST_PROTOCOL)
    with open(potentials_sym_path, 'wb') as f:
        pickle.dump(potential_history_sym, f, pickle.HIGHEST_PROTOCOL)
   
    if verbose:
        
        mean_potential_history_asym = np.mean(potential_history_asym, axis=1)
        std_asym = np.std(potential_history_asym, axis=1)
        
        mean_potential_history_sym = np.mean(potential_history_sym, axis=1)
        std_sym = np.std(potential_history_sym, axis=1)

        compare_lines(mean_potential_history_asym, no_players, std_asym)
        compare_lines(mean_potential_history_sym, no_players, std_sym)             
   