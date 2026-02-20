import numpy as np
import pickle 

from pathlib import Path
import multiprocessing as mp

from potentialgames.mechanism.game_setup import PayoffMatrix, IdenticalInterestSetup
from potentialgames.mechanism.game_engine import GameEngine
from potentialgames.utils import compute_beta, logger, compare_lines

"""
    Runs a series of experiments to evaluate the effect of different epsilon values on the convergence of a specified game-theoretic algorithm.
    The function simulates repeated plays of a potential game for various epsilon values, using either pre-generated or newly generated payoff matrices.
    It supports both deterministic and noisy utility settings.
    
Parameters:
    algorithm (str): The learning algorithm to use (default: "fast_log_linear").
    no_actions (int): Number of possible actions for each player (default: 10).
    no_players (int): Number of players in the game (default: 2).
    delta (float): Parameter controlling the payoff matrix structure (default: 0.1).
    epsilons (list of float): List of epsilon values to test (default: [0.1, 0.05, 0.025, 0.01]).
    use_noisy_utility (bool): Whether to use noisy utilities in the game (default: False).
    symmetric (bool): Whether to use a symmetric payoff matrix (default: False).
    eps (float): Epsilon value for convergence threshold (default: 0.05).
    max_iter (int): Maximum number of iterations for each experiment (default: 1,000,000).
    n_exp (int): Number of experiments (games) to run for each epsilon (default: 30).
    load_games (bool): Whether to load existing games from disk or generate new ones (default: True).
    verbose (bool): Whether to print progress and plot results (default: True).
Saves:
    The potential history of experiments to a pickle file in the data/experiments/delta directory.
 """

def run_single_experiment(epsilon, game_idx, algorithm, no_actions, no_players, delta, symmetric, use_noisy_utility, max_iter, payoff_matrix, eta_noise):
    """
    Run a single experiment for one epsilon value and one game iteration with preloaded payoff matrix.
    
    Args:
        epsilon (float): Epsilon value for this experiment
        game_idx (int): Game index (0 to n_exp-1)
        algorithm (str): Learning algorithm to use
        no_actions (int): Number of actions per player
        no_players (int): Number of players
        delta (float): Delta parameter for payoff matrix
        symmetric (bool): Whether to use symmetric payoff matrix
        use_noisy_utility (bool): Whether to use noisy utilities
        max_iter (int): Maximum iterations for the game
        payoff_matrix (PayoffMatrix): Pre-loaded payoff matrix
        eta_noise (float): Noise level if using noisy utility
        
    Returns:
        tuple: (epsilon, game_idx, potential_history_for_this_experiment)
    """
    try:

        # Create fresh game instances for this process
        action_space = np.arange(0, no_actions)
        gameSetup = IdenticalInterestSetup(action_space, use_noisy_utility=use_noisy_utility, payoff_matrix=payoff_matrix)
        game = GameEngine(gameSetup, algorithm=algorithm, max_iter=max_iter)
        
        initial_action_profile = payoff_matrix.secondNE
        game.set_initial_action_profile(initial_action_profile)
        
        # Set up mu_matrix for fast algorithms
        if "fast" in algorithm:
            mu_matrix = np.zeros([1, len(action_space)**no_players])
            mu_matrix[0, initial_action_profile[0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
            game.set_mu_matrix(mu_matrix)
        
        # Set noise if needed
        if use_noisy_utility:
            game.eta_noise = eta_noise
                
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

        return (epsilon, game_idx, potential_history)
        
    except Exception as e:
        print(f"Error in experiment epsilon={epsilon}, game_idx={game_idx}: {e}")
        return (epsilon, game_idx, None)


def epsilon_experiment(
    algorithm="fast_log_linear", 
    no_actions=10, 
    no_players=2, 
    delta=0.1, 
    epsilons=[0.1, 0.05, 0.025, 0.01], 
    use_noisy_utility=False, 
    symmetric=False, 
    eps=0.05, 
    max_iter=1000000, 
    n_exp=30, 
    load_games=True, 
    verbose=True,
    n_processes=None
):
    """
    Runs a series of experiments to evaluate the effect of different epsilon values on the convergence of a specified game-theoretic algorithm.
    """
    logger.info("Epsilon experiments for algorithm: " + algorithm)
    
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    logger.info(f"Using {n_processes} processes")
    
    # Setup paths and parameters
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / "IdenticalInterest" / "games"
    folder = "delta_" + str(int(delta*1000)).zfill(4)
    
    firstNE = np.array([1, 1])
    secondNE = np.array([no_actions - 2, no_actions - 2])
    
    # Calculate eta_noise if using noisy utility
    eta_noise = None
    if use_noisy_utility:
        # To ensure comparability of the results, all games should have the same noise level
        betas = np.zeros(len(epsilons))
        for i, epsilon in enumerate(epsilons):
            betas[i] = compute_beta(no_actions, no_players, delta, epsilon, symmetric, use_noisy_utility)

        eta_noise = 1/2.0/np.max(betas)
    
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
    for epsilon in epsilons:
        for game_idx in range(n_exp):
            payoff_matrix = all_payoff_matrices[game_idx]
            args_tuple = (epsilon, game_idx, algorithm, no_actions, no_players, delta, 
                         symmetric, use_noisy_utility, max_iter, payoff_matrix,
                         eta_noise)
            experiment_args.append(args_tuple)
    
    # Initialize result storage
    potential_history = np.zeros((len(epsilons), n_exp, max_iter))
    
    # Run experiments in parallel using starmap with preloaded data
    with mp.Pool(processes=n_processes) as pool:
        total_experiments = len(experiment_args)
        logger.info(f"Starting {total_experiments} experiments with preloaded matrices...")
        
        results = pool.starmap(run_single_experiment, experiment_args)
    
    # Process results
    successful_experiments = 0
    for epsilon, game_idx, result_data in results:
        if result_data is not None:
            epsilon_idx = epsilons.index(epsilon)
            potential_history[epsilon_idx][game_idx] = result_data
            successful_experiments += 1
        else:
            logger.warning(f"Failed experiment: epsilon={epsilon}, game_idx={game_idx}")
    
    logger.info(f"Completed {successful_experiments}/{total_experiments} experiments successfully")
    
    # Calculate statistics if verbose
    if verbose:
        mean_potential_history = np.zeros((len(epsilons), max_iter))
        std = np.zeros((len(epsilons), max_iter))
        conv_markers = np.zeros((len(epsilons), 2))
        
        for idx, epsilon in enumerate(epsilons):
            mean_potential_history[idx] = np.mean(potential_history[idx], 0)
            index = np.argwhere(mean_potential_history[idx] > 1 - epsilon)
            conv_markers[idx, 0] = index[0][0] if len(index) > 0 else None
            conv_markers[idx, 1] = mean_potential_history[idx][int(conv_markers[idx, 0])] if len(index) > 0 else None
            std[idx] = np.std(potential_history[idx], 0)
    
    # Save results to pickle file
    algorithm_name = algorithm
    if use_noisy_utility:
        algorithm_name = algorithm + "_noisy"
        
    potentials_path = repo_root / "data" / "experiments" / "epsilon" / f"{algorithm_name}_potentials.pckl"
    potentials_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Results saved to {potentials_path}")

    # Plot results if verbose
    if verbose:
        compare_lines(mean_potential_history, None, [str(epsilon) for epsilon in epsilons], 
                     std, markers=conv_markers)