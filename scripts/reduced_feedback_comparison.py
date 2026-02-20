import numpy as np
import pickle

from pathlib import Path
import multiprocessing as mp

from potentialgames.mechanism.game_setup import PayoffMatrix, IdenticalInterestSetup
from potentialgames.mechanism import GameEngine
from potentialgames.utils import logger, compare_lines


"""
    Runs parallel experiments to compare reduced-feedback log-linearlearning performance
    with state-of-the-art EXP3P and Exponential Weights with Annealing across multiple potential games.
    
    
    Parameters:
        algorithm (str): Learning algorithm to evaluate (default: "fast_binary_log_linear").
        no_actions (int): Number of actions per player.
        no_players (int): Number of players.
        delta (float): The suboptimality gap for the payoff matrix.
        symmetric (bool): Whether payoff matrices are symmetric.
        epsilon (float): Convergence threshold and exploration parameter.
        max_iter (int): Maximum iterations per experiment.
        n_exp (int): Number of independent game experiments.
        load_games (bool): Whether to load existing payoff matrices from disk.
        verbose (bool): Whether to compute statistics and generate plots.
    Saves:
        The potential history of experiments to a pickle file in the data/experiments/reduced_feedback_comparison/ directory.   
"""
def run_single_experiment(game_idx, algorithm, no_actions, no_players, delta, epsilon, symmetric, max_iter, payoff_matrix):
    """
    Runs a single experiment for a given game index and preloaded payoff matrix.

    Args:
        game_idx (int): Index of the game experiment.
        algorithm (str): Learning algorithm to evaluate.
        no_actions (int): Number of actions per player.
        no_players (int): Number of players.
        delta (float): The suboptimality gap for the payoff matrix.
        epsilon (float): The tolerance for convergence.
        symmetric (bool): Whether payoff matrices are symmetric.
        max_iter (int): Maximum iterations for the experiment.
        payoff_matrix (PayoffMatrix): Preloaded payoff matrix for the game.

    Returns:
        tuple: (game_idx, potential_history) where potential_history is the recorded potential values over iterations, or None if an error occurred.
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
            beta = game.compute_beta(epsilon)
            game.play(beta=beta)
            potential_history = np.transpose(game.expected_value)
        else:
            # For non-fast algorithms, run multiple times and average
            potentials = np.zeros((5, max_iter))
            for j in range(5):
                beta = game.compute_beta(epsilon)
                game.play(beta=beta)
                potentials[j] = np.transpose(game.potentials_history)
            potential_history = np.mean(potentials, 0)

        return (game_idx, potential_history)
    
    except Exception as e:
        logger.error(f"Error in experiment {game_idx}: {e}")
        return (game_idx, None)

def reduced_feedback_comparison(
    algorithm = "fast_binary_log_linear",
    no_actions=10, 
    no_players=2, 
    delta=0.1, 
    symmetric = False, 
    epsilon = 0.05, 
    max_iter = 1000000, 
    n_exp = 30, 
    load_games=True, 
    verbose=True,
    n_processes=None
    ):
    """
    Runs a series of experiments to compare the performance of reduced-feedback log-linear learning with state-of-the-art algorithms across multiple potential games.
    """
    logger.info("Comparison of Binary Log Linear Learning, EXP3P and Exponential Weights with Annealing.")
    
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    logger.info(f"Using {n_processes} processes")
        
    firstNE = np.array([1, 1])
    secondNE = np.array([no_actions - 2, no_actions - 2])  
                     
    repo_root = Path(__file__).resolve().parents[1]  # Adjust .parent levels if needed
    data_path = repo_root / "data" / "IdenticalInterest" / "games"
    
    folder = "delta_" + str(int(delta * 1000)).zfill(4)
    
    # PRE-LOAD ALL GAMES INTO MEMORY
    logger.info("Pre-loading all game matrices...")
    all_payoff_matrices = {}
    
    for game_idx in range(n_exp):
        file_path = data_path / folder / f"game_{game_idx}.pckl"
        
        if load_games:
            try:
                all_payoff_matrices[game_idx] = PayoffMatrix.load(file_path)
                if game_idx % 10 == 0:  # Log every 10th game
                    logger.info(f"Loaded game {game_idx}/{n_exp}")
            except Exception as e:
                logger.error(f"Error loading game from file {file_path}: {e}")
                # Create a new game matrix if loading fails
                payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric=symmetric)
                payoff_matrix.regenerate(method="generate_plateau_matrix", delta=delta, no_actions=no_actions, symmetric=symmetric)
                payoff_matrix.save(file_path)
                all_payoff_matrices[game_idx] = payoff_matrix
                logger.info(f"Generated and saved new game {game_idx}")
        else:
            logger.info(f"Generating new game matrix {game_idx} for delta: {delta} and no_actions: {no_actions}")
            payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric=symmetric)
            payoff_matrix.regenerate(method="generate_plateau_matrix", delta=delta, no_actions=no_actions, symmetric=symmetric)
            payoff_matrix.save(file_path)
            all_payoff_matrices[game_idx] = payoff_matrix
    
    logger.info(f"Pre-loaded {len(all_payoff_matrices)} games")
    
    # Prepare arguments for all experiments with preloaded matrices
    experiment_args = []
    for game_idx in range(n_exp):
        payoff_matrix = all_payoff_matrices[game_idx]
        args_tuple = (game_idx, algorithm, no_actions, no_players, delta, 
                      epsilon, symmetric, max_iter, payoff_matrix)
        
        experiment_args.append(args_tuple)
    
    potential_history = np.zeros((1, n_exp, max_iter))
    
    # Run experiments in parallel using starmap with preloaded data
    with mp.Pool(processes=n_processes) as pool:
        total_experiments = len(experiment_args)
        logger.info(f"Starting {total_experiments} experiments with preloaded matrices...")
        
        results = pool.starmap(run_single_experiment, experiment_args)
   
    successful_experiments = 0
    for game_idx, result_data in results:
        if result_data is not None:
            potential_history[0][game_idx] = result_data
            successful_experiments += 1
        else:
            logger.warning(f"Failed experiment: game_idx={game_idx}.")

    logger.info(f"Completed {successful_experiments}/{total_experiments} experiments successfully")

    if verbose:
        mean_potential_history = np.zeros((1, max_iter))
        std = np.zeros((1, max_iter))
        conv_markers = np.zeros((1, 2))

        mean_potential_history[0] = np.mean(potential_history[0], 0)
        index = np.argwhere(mean_potential_history[0] > 1 - epsilon)
        conv_markers[0, 0] = index[0][0] if len(index) > 0 else None
        conv_markers[0, 1] = mean_potential_history[0][int(conv_markers[0, 0])] if len(index) > 0 else None
        std[0] = np.std(potential_history[0], 0)

    potentials_path = repo_root / "data" / "experiments" / "reduced_feedback_comparison" / f"{algorithm}_potentials.pckl"
    potentials_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Results saved to {potentials_path}")
    
    # Plot results if verbose
    if verbose:
        compare_lines(mean_potential_history, None, [algorithm], 
                     std, markers=conv_markers)