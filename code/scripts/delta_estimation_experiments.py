import numpy as np
import pickle

from pathlib import Path
import multiprocessing as mp

from utils import REPO_ROOT
from potentialgames.mechanism.game_setup import PayoffMatrix, IdenticalInterestSetup
from potentialgames.mechanism.game_engine import GameEngine
from potentialgames.utils import compute_beta, logger, compare_lines
"""
    Runs a series of experiments to evaluate the effect of different delta estimation errors on the convergence of a specified game-theoretic algorithm.
    
    Parameters:
        algorithm (str): The learning algorithm to use (default: "fast_log_linear").
        no_actions (int): Number of possible actions for each player (default: 10).
        no_players (int): Number of players in the game (default: 2).
        delta (float): The true suboptimality gap for the payoff matrix (default: 0.1).
        est (list[float]): List of estimation error multipliers to test, which affect the delta used in the algorithm (default: [1, 1.5, 2., 2.5, 3., 3.5]).
        symmetric (bool): Whether to use a symmetric payoff matrix (default: False).
        epsilon (float): Convergence threshold for the potential function (default: 0.05).
        max_iter (int): Maximum number of iterations for each experiment (default: 1,000,000).
        n_exp (int): Number of experiments to run for each estimation error multiplier (default: 30).
        load_games (bool): Whether to load pre-generated games from disk or generate new ones (default: True).
        easy_game (bool): Whether to generate easier games with larger basins of attraction for the NE (default: False).
        verbose (bool): Whether to print progress and plot results (default: True).
        n_processes (int or None): Number of parallel processes to use for running experiments. If None, uses all available CPU cores.

    Saves:
        The potential history of experiments to a pickle file in the data/experiments/delta_est directory.
"""
def run_single_experiment(delta_est, game_idx, algorithm, no_actions, no_players, delta, epsilon, symmetric, max_iter, payoff_matrix):
    """
    Runs a single experiment for a given delta estimation and game index and preloaded payoff matrix.

    Args:
        delta_est (float): The estimated delta value (multiplier of the true delta).
        game_idx (int): The index of the game being run.
        algorithm (str): The learning algorithm to use.
        no_actions (int): Number of possible actions for each player.
        no_players (int): Number of players in the game.
        delta (float): The true suboptimality gap for the payoff matrix.
        epsilon (float): The tolerance for convergence.
        symmetric (bool): Whether the payoff matrix is symmetric.
        max_iter (int): Maximum number of iterations for the experiment.
        payoff_matrix (PayoffMatrix): The preloaded payoff matrix for the game.

    Returns:
        tuple: (delta_est, game_idx, potential_history) where potential_history is the recorded potential values over iterations, or None if an error occurred.
    """
    try:
        # Create fresh game instances for this process
        action_space = np.arange(0, no_actions)
        gameSetup = IdenticalInterestSetup(action_space, payoff_matrix=payoff_matrix)
    
        game = GameEngine(gameSetup, algorithm=algorithm, max_iter=max_iter)
    
        initial_action_profile = payoff_matrix.secondNE
        game.set_initial_action_profile(initial_action_profile)
        
        # Set up mu_matrix for fast algorithms
        if "fast" in algorithm:
            mu_matrix = np.zeros([1, len(action_space)**no_players])
            mu_matrix[0, initial_action_profile[0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
            game.set_mu_matrix(mu_matrix)
        
        # Run the experiment based on algorithm type
        if "fast" in algorithm:
            beta = compute_beta(game.no_actions, game.no_players, delta_est, epsilon, game.symmetric, game.use_noisy_utility) 
            game.play(beta=beta)
            potential_history = np.transpose(game.expected_value)
        else:
            # For non-fast algorithms, run multiple times and average
            potentials = np.zeros((5, max_iter))
            beta = compute_beta(game.no_actions, game.no_players, delta_est, epsilon, game.symmetric, game.use_noisy_utility) 
            for j in range(5):
                game.play(beta=beta)
                potentials[j] = np.transpose(game.potentials_history)
            potential_history = np.mean(potentials, 0)

        return (delta_est, game_idx, potential_history)
    
    except Exception as e:
        logger.error(f"Error in experiment {game_idx}: {e}")
        return (delta_est, game_idx, None)
  
def delta_estimation_experiment(
    algorithm="fast_log_linear",
    no_actions=10,
    no_players=2,
    delta=0.1,
    est=[1, 1.5, 2., 2.5, 3., 3.5],
    symmetric=False,
    epsilon=0.05,
    max_iter=1000000,
    n_exp=30,
    load_games=True,
    easy_game=False,
    verbose=True,
    n_processes=None,
    ):
    """
    Runs a series of experiments to evaluate the effect of different delta estimation errors on the convergence of a specified game-theoretic algorithm.
    """
    logger.info("Comparison of Log Linear Learning and HEDGE")
    
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    logger.info(f"Using {n_processes} processes")
        
    firstNE = np.array([1, 1])
    secondNE = np.array([2, 2])  
                     
    repo_root = REPO_ROOT
    data_path = repo_root / "data" / "IdenticalInterest" / "games"
    
    folder = "delta_" + str(int(delta * 1000)).zfill(4)
    
    # PRE-LOAD ALL GAMES INTO MEMORY
    logger.info("Pre-loading all game matrices...")
    all_payoff_matrices = {}
    
    for game_idx in range(n_exp):

        game_name = f"game_{game_idx}"
        if easy_game:
            game_name = "easy_" + game_name
    
        file_path = data_path / folder / f"{game_name}.pckl"
        
        if load_games:
            try:
                all_payoff_matrices[game_idx] = PayoffMatrix.load(file_path)
                if game_idx % 10 == 0:  # Log every 10th game
                    logger.info(f"Loaded game {game_idx}/{n_exp}")
            except Exception as e:
                logger.error(f"Error loading game from file {file_path}: {e}")
                # Create a new game matrix if loading fails
                payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric=symmetric)
                if easy_game:
                    payoff_matrix.regenerate(method="generate_easy_matrix", delta=delta, no_actions=no_actions, symmetric=symmetric)
                else:
                    payoff_matrix.regenerate(method="generate_plateau_matrix", delta=delta, no_actions=no_actions, symmetric=symmetric)
                payoff_matrix.save(file_path)
                all_payoff_matrices[game_idx] = payoff_matrix
                logger.info(f"Generated and saved new game {game_idx}")
        else:
            logger.info(f"Generating new game matrix {game_idx} for delta: {delta} and no_actions: {no_actions}")
            payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric=symmetric)
            if easy_game:
                payoff_matrix.regenerate(method="generate_easy_matrix", delta=delta, no_actions=no_actions, symmetric=symmetric)
            else:
                payoff_matrix.regenerate(method="generate_plateau_matrix", delta=delta, no_actions=no_actions, symmetric=symmetric)            
            payoff_matrix.save(file_path)
            all_payoff_matrices[game_idx] = payoff_matrix
            logger.info(f"Generated and saved new game {game_idx}")

    
    logger.info(f"Pre-loaded {len(all_payoff_matrices)} games")
    
    experiment_args = []
    
    delta_est = [delta*e for e in est]
    for e in delta_est:
        for game_idx in range(n_exp):
            payoff_matrix = all_payoff_matrices[game_idx]
            args_tuple = (e, game_idx, algorithm, no_actions, no_players, delta,
                          epsilon, symmetric, max_iter, payoff_matrix)
            experiment_args.append(args_tuple)

    potential_history = np.zeros((len(est), n_exp, max_iter))
    
    
    # Run experiments in parallel using starmap with preloaded data
    with mp.Pool(processes=n_processes) as pool:
        total_experiments = len(experiment_args)
        logger.info(f"Starting {total_experiments} experiments with preloaded matrices...")
        
        results = pool.starmap(run_single_experiment, experiment_args)
    
    successful_experiments = 0
    for e, game_idx, result_data in results:
        if result_data is not None:
            delta_est_idx = delta_est.index(e)
            potential_history[delta_est_idx][game_idx] = result_data
            successful_experiments += 1
        else:
            logger.warning(f"Failed experiment: est_err={est[delta_est_idx]}, game_idx={game_idx}")
            
    if verbose:
        mean_potential_history = np.zeros((len(est), max_iter))
        std = np.zeros((len(est), max_iter))
        conv_markers = np.zeros((len(est), 2))
        
        for idx, e in enumerate(est):
            mean_potential_history[idx] = np.mean(potential_history[idx], axis=0)
            std[idx] = np.std(potential_history[idx], axis=0)
            index = np.argwhere(mean_potential_history[idx] > 1 - epsilon)
            conv_markers[idx, 0] = index[0][0] if len(index) > 0 else None
            conv_markers[idx, 1] = mean_potential_history[idx][int(conv_markers[idx, 0])] if len(index) > 0 else None
    
    file_name = algorithm
    if easy_game:
        file_name = "easy_" + file_name
    
    potentials_path = repo_root / "data" / "experiments" / "delta_est" / f"{file_name}_potentials.pckl"
    potentials_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Results saved to {potentials_path}")
    
    
    if verbose:
        compare_lines(mean_potential_history, None, [str(e*100)+"%" for e in est], std=std, markers=conv_markers)
