import numpy as np

import pickle

from pathlib import Path

from potentialgames.mechanism.game_setup import PayoffMatrix, IdenticalInterestSetup
from potentialgames.mechanism import GameEngine
from potentialgames.utils import logger, compare_lines


def full_feedback_comparison(no_actions=10, no_players=2, delta=0.1, symmetric = False, epsilon = 0.05, max_iter = 1000000, n_exp = 30, load_games=True, verbose=True):
    
    logger.info("Comparison of Log Linear Learning and HEDGE")
    
    action_space = np.arange(0, no_actions)
    
    firstNE = np.array([1, 1])
    secondNE = np.array([no_actions - 2, no_actions - 2])
    
    payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric=symmetric)
    
    gameSetup = IdenticalInterestSetup(action_space, payoff_matrix=payoff_matrix)
    
    game_lll = GameEngine(gameSetup, algorithm="fast_log_linear", max_iter=max_iter)
    game_hedge = GameEngine(gameSetup, algorithm="hedge", max_iter=max_iter)
    
    initial_action_profile = secondNE
    game_lll.set_initial_action_profile(initial_action_profile)
    game_hedge.set_initial_action_profile(initial_action_profile)
    
    # indicator initial joint action profile distribution
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, initial_action_profile[0]*game_lll.gameSetup.no_actions + initial_action_profile[1]] = 1
    
    game_lll.set_mu_matrix(mu_matrix)
    game_hedge.set_mu_matrix(mu_matrix)  
                     
    repo_root = Path(__file__).resolve().parents[1]  # Adjust .parent levels if needed
    data_path = repo_root / "data" / "IdenticalInterest" / "games"
    
    folder = "delta_" + str(int(delta * 1000)).zfill(4)
    
    potential_history_lll = np.zeros((1, n_exp, max_iter))
    
    potentials_hedge = np.zeros((10, max_iter))
    potential_history_hedge = np.zeros((1, n_exp, max_iter))
    
    if verbose:
        mean_potential_history_lll = np.zeros((1, max_iter))
        std_lll = np.zeros((1, max_iter))
        conv_markers_lll = np.zeros((1, 2))
        
        mean_potential_history_hedge = np.zeros((1, max_iter))
        std_hedge = np.zeros((1, max_iter))
        conv_markers_hedge = np.zeros((1, 2))

    for i in range(n_exp):
        
        file_path = data_path / folder / f"game_{i}.pckl"
        
        if load_games:
            try:
                payoff_matrix = PayoffMatrix.load(file_path)
                logger.info("       Loaded game from file: " + str(file_path))
            except Exception as e:
                logger.error(f"Error loading game from {file_path}: {e}")
                exit()
        else:
            logger.info("       Generating new game matrix for delta: " + str(delta) + " and no_actions: " + str(no_actions))
            payoff_matrix.regenerate(method="generate_plateau_matrix", delta = delta, no_actions = no_actions, symmetric = symmetric)                
            payoff_matrix.save(file_path)
            logger.info("       Saved game to file: " + str(file_path))
        
        game_lll.reset_game(payoff_matrix=payoff_matrix)
        game_hedge.reset_game(payoff_matrix=payoff_matrix)
        
        logger.info("      Currently testing game " + str(i+1) + " with delta: " + str(delta))
        
        beta = game_lll.compute_beta(epsilon)
        game_lll.play(beta=beta)
        
        potential_history_lll[0, i, :] = np.transpose(game_lll.expected_value)
        
        for j in range(10):
            
            game_hedge.play()
            game_hedge.reset_game()
            
            potentials_hedge[j] = np.transpose(game_hedge.potentials_history)

        potential_history_hedge[0, i, :] = np.mean(potentials_hedge, 0)

    if verbose:
        mean_potential_history_lll[0] = np.mean(potential_history_lll[0], 0)
        index = np.argwhere(mean_potential_history_lll[0] > 1 - epsilon)
        conv_markers_lll[0, 0] = index[0][0] if len(index) > 0 else None
        conv_markers_lll[0, 1] = mean_potential_history_lll[0][int(conv_markers_lll[0, 0])] if len(index) > 0 else None
        std_lll[0] = np.std(potential_history_lll[0], 0)
        
        mean_potential_history_hedge[0] = np.mean(potential_history_hedge[0], 0)
        index = np.argwhere(mean_potential_history_hedge[0] > 1 - epsilon)
        conv_markers_hedge[0, 0] = index[0][0] if len(index) > 0 else None
        conv_markers_hedge[0, 1] = mean_potential_history_hedge[0][int(conv_markers_hedge[0, 0])] if len(index) > 0 else None
        std_hedge[0] = np.std(potential_history_hedge[0], 0)

    potentials_lll_path = repo_root / "data" / "experiments" / "full_feedback_comparison" / "potentials_lll.pckl"
    potentials_hedge_path = repo_root / "data" / "experiments" / "full_feedback_comparison" / "potentials_hedge.pckl"
    
    potentials_lll_path.parent.mkdir(parents=True, exist_ok=True)
    potentials_hedge_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(potentials_lll_path, "wb") as f:
        pickle.dump(potential_history_lll, f)
    with open(potentials_hedge_path, "wb") as f:
        pickle.dump(potential_history_hedge, f)

    if verbose:
        compare_lines(
            [mean_potential_history_lll[0], mean_potential_history_hedge[0]],
            [str(epsilon), str(epsilon)],
            [std_lll[0], std_hedge[0]], 
            markers=[conv_markers_lll[0], conv_markers_hedge[0]]
        )