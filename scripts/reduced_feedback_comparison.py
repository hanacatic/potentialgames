import numpy as np

import pickle

from pathlib import Path

from potentialgames.mechanism.game_setup import PayoffMatrix, IdenticalInterestSetup
from potentialgames.mechanism import GameEngine
from potentialgames.utils import logger, compare_lines


def reduced_feedback_comparison(no_actions=10, no_players=2, delta=0.1, symmetric=False, epsilon=0.05, max_iter=1000000, n_exp=30, load_games=True, verbose=True):
    
    logger.info("Comparison of Binary Log Linear Learning with EXP3p and Exponential Weight with Annealing")
    
    action_space = np.arange(0, no_actions)
    
    firstNE = np.array([1, 1])
    secondNE = np.array([no_actions - 2, no_actions - 2])
    
    payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric=symmetric)
    
    gameSetup = IdenticalInterestSetup(action_space, payoff_matrix=payoff_matrix)
    
    game_bll = GameEngine(gameSetup, algorithm="fast_binary_log_linear", max_iter=max_iter)
    game_exp3p = GameEngine(gameSetup, algorithm="exp3p", max_iter=max_iter)
    game_ewa = GameEngine(gameSetup, algorithm="exponential_weight_with_annealing", max_iter=max_iter)
    
    initial_action_profile = secondNE
    game_bll.set_initial_action_profile(initial_action_profile) 
    game_exp3p.set_initial_action_profile(initial_action_profile)
    game_ewa.set_initial_action_profile(initial_action_profile)
    
    # indicator initial joint action profile distribution
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, initial_action_profile[0]*game_bll.gameSetup.no_actions + initial_action_profile[1]] = 1
    
    game_bll.set_mu_matrix(mu_matrix)
    game_exp3p.set_mu_matrix(mu_matrix)
    game_ewa.set_mu_matrix(mu_matrix)
    
    repo_root = Path(__file__).resolve().parents[1]  # Adjust .parent levels if needed
    data_path = repo_root / "data" / "IdenticalInterest" / "games"
    
    folder = "delta_" + str(int(delta * 1000)).zfill(4)
    
    potential_history_bll = np.zeros((1, n_exp, max_iter))
    
    potentials_exp3p = np.zeros((10, max_iter))
    potential_history_exp3p = np.zeros((1, n_exp, max_iter))
    
    potentials_ewa = np.zeros((10, max_iter))
    potential_history_ewa = np.zeros((1, n_exp, max_iter))
    
    if verbose:
        mean_potential_history_bll = np.zeros((1, max_iter))
        std_bll = np.zeros((1, max_iter))
        conv_markers_bll = np.zeros((1, 2))
        
        mean_potential_history_exp3p = np.zeros((1, max_iter))
        std_exp3p = np.zeros((1, max_iter))
        conv_markers_exp3p = np.zeros((1, 2))
        
        mean_potential_history_ewa = np.zeros((1, max_iter))
        std_ewa = np.zeros((1, max_iter))
        conv_markers_ewa = np.zeros((1, 2))
        
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

        game_bll.reset_game(payoff_matrix=payoff_matrix)
        game_exp3p.reset_game(payoff_matrix=payoff_matrix)
        game_ewa.reset_game(payoff_matrix=payoff_matrix)
        
        logger.info("      Currently testing game " + str(i+1) + " with delta: " + str(delta))

        beta = game_bll.compute_beta(epsilon)
        game_bll.play(beta=beta)
        
        potential_history_bll[0, i, :] = np.transpose(game_bll.expected_value)
        
        for j in range(10):
        
            game_exp3p.play()
            game_exp3p.reset_game()
            
            potentials_exp3p[j] = np.transpose(game_exp3p.potentials_history)
            
            game_ewa.play()
            game_ewa.reset_game()
            
            potentials_ewa[j] = np.transpose(game_ewa.potentials_history)

        potential_history_exp3p[0, i, :] = np.mean(potentials_exp3p, 0)
        potential_history_ewa[0, i, :] = np.mean(potentials_ewa, 0)
        
        
    if verbose:
        mean_potential_history_bll[0] = np.mean(potential_history_bll[0], 0)
        index = np.argwhere(mean_potential_history_bll[0] > 1 - epsilon)
        conv_markers_bll[0, 0] = index[0][0] if len(index) > 0 else None
        conv_markers_bll[0, 1] = mean_potential_history_bll[0][int(conv_markers_bll[0, 0])] if len(index) > 0 else None
        std_bll[0] = np.std(potential_history_bll[0], 0)

        mean_potential_history_exp3p[0] = np.mean(potential_history_exp3p[0], 0)
        index = np.argwhere(mean_potential_history_exp3p[0] > 1 - epsilon)
        conv_markers_exp3p[0, 0] = index[0][0] if len(index) > 0 else None
        conv_markers_exp3p[0, 1] = mean_potential_history_exp3p[0][int(conv_markers_exp3p[0, 0])] if len(index) > 0 else None
        std_exp3p[0] = np.std(potential_history_exp3p[0], 0)

        mean_potential_history_ewa[0] = np.mean(potential_history_ewa[0], 0)
        index = np.argwhere(mean_potential_history_ewa[0] > 1 - epsilon)
        conv_markers_ewa[0, 0] = index[0][0] if len(index) > 0 else None
        conv_markers_ewa[0, 1] = mean_potential_history_ewa[0][int(conv_markers_ewa[0, 0])] if len(index) > 0 else None
        std_ewa[0] = np.std(potential_history_ewa[0], 0)
    
    potentials_bll_path = repo_root / "data" / "experiments" / "reduced_feedback_comparison" / "potentials_bll.pckl"
    potentials_exp3p_path = repo_root / "data" / "experiments" / "reduced_feedback_comparison" / "potentials_exp3p.pckl"
    potentials_ewa_path = repo_root / "data" / "experiments" / "reduced_feedback_comparison" / "potentials_ewa.pckl"

    potentials_bll_path.parent.mkdir(parents=True, exist_ok=True)
    potentials_exp3p_path.parent.mkdir(parents=True, exist_ok=True)
    potentials_ewa_path.parent.mkdir(parents=True, exist_ok=True)

    with open(potentials_bll_path, "wb") as f:
        pickle.dump(potential_history_bll, f)
    with open(potentials_exp3p_path, "wb") as f:
        pickle.dump(potential_history_exp3p, f)
    with open(potentials_ewa_path, "wb") as f:
        pickle.dump(potential_history_ewa, f)
        
    if verbose:
        compare_lines(
            [mean_potential_history_bll[0], mean_potential_history_exp3p[0], mean_potential_history_ewa[0]],
            [str(epsilon), str(epsilon), str(epsilon)],
            [std_bll[0], std_exp3p[0], std_ewa[0]], 
            markers=[conv_markers_bll[0], conv_markers_exp3p[0], conv_markers_ewa[0]]
        )