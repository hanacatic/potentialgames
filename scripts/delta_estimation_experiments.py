import numpy as np
import pickle

from pathlib import Path
import matplotlib.pyplot as plt

from potentialgames.mechanism.game_setup import PayoffMatrix, IdenticalInterestSetup
from potentialgames.mechanism.game_engine import GameEngine
from potentialgames.utils import compute_beta, logger, compare_lines


def delta_estimation_experiment(no_actions=10, no_players=2, delta=0.1, est=[1, 1.25, 1.5, 2, 3, 4], symmetric=False, eps=0.05, max_iter=1000000, n_exp=30, load_games=True, verbose=True):

    logger.info("Delta estimation experiments")

    action_space = np.arange(0, no_actions)
    
    firstNE = np.array([1, 1])
    secondNE = np.array([no_actions - 2, no_actions - 2])
    
    payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric=symmetric)
    
    gameSetup = IdenticalInterestSetup(action_space, payoff_matrix=payoff_matrix)
    game = GameEngine(gameSetup, algorithm="fast_log_linear", max_iter=max_iter)
    
    initial_action_profile = secondNE
    game.set_initial_action_profile(initial_action_profile)
    
    # indicator initial joint action profile distribution
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, initial_action_profile[0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
    
    # uniform initial joint action profile distribution
    # mu_matrix = np.ones([1, no_actions**no_players])/no_actions**no_players
    
    game.set_mu_matrix(mu_matrix)
    potential_history = np.zeros((len(est), n_exp, max_iter))   
    
    if verbose:
        mean_potential_history = np.zeros((len(est), max_iter))
        std = np.zeros((len(est), max_iter))
        conv_markers = np.zeros((len(est), 2))
        
    repo_root = Path(__file__).resolve().parents[1]  # Adjust .parent levels if needed
    data_path = repo_root / "data" / "IdenticalInterest" / "games"
    
    folder = "delta_" + str(int(delta * 1000)).zfill(4)
    
    for idx, e in enumerate(est):
        
        for i in range(n_exp):
            
            file_path = data_path / folder / f"game_{i}.pckl"
            
            if load_games:
                try:
                    payoff_matrix = PayoffMatrix.load(file_path)
                    logger.info("       Loaded game from file: " + str(file_path))
                except Exception as e:
                    logger.error(f"     Error loading game from file {file_path}: {e}")
                    exit()
            else:
                logger.info("       Generating new game matrix for delta: " + str(delta) + " and no_actions: " + str(no_actions))
                payoff_matrix.regenerate(method="generate_plateau_matrix", delta = delta, no_actions = no_actions, symmetric = symmetric)
                payoff_matrix.save(file_path)
                logger.info("       Saved game to file: " + str(file_path))
            
            game.reset_game(payoff_matrix=payoff_matrix)
            
            beta = compute_beta(game.no_actions, game.no_players, e*delta, eps, game.symmetric, game.use_noisy_utility) 
            game.play(beta=beta)
            potential_history[idx][i] = np.transpose(game.expected_value)
        
        if verbose:
            mean_potential_history[idx] = np.mean(potential_history[idx], axis=0)
            std[idx] = np.std(potential_history[idx], axis=0)
            index = np.argwhere(mean_potential_history[idx] > 1 - eps)
            conv_markers[idx, 0] = index[0][0] if len(index) > 0 else max_iter
            conv_markers[idx, 1] = mean_potential_history[idx][int(conv_markers[idx, 0])] if len(index) > 0 else None
        
    potentials_path = repo_root / "data" / "experiments" / "delta_est" / f"potential_history.pckl"
    potentials_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f)
    
    if verbose:
        compare_lines(mean_potential_history, [str(e*100)+"%" for e in est], std=std, markers=conv_markers)