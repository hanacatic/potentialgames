import numpy as np
import pickle 

from pathlib import Path
import matplotlib.pyplot as plt

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
def epsilon_experiment(algorithm = "fast_log_linear", no_actions = 10, no_players = 2, delta = 0.1, epsilons = [0.1, 0.05, 0.025, 0.01], use_noisy_utility = False, symmetric = False, eps = 0.05, max_iter = 1000000, n_exp = 30, load_games=True, verbose=True): 

    logger.info("Epsilon experiments for algorithm: " + algorithm)
    action_space = np.arange(0, no_actions)
    
    firstNE = np.array([1,1])
    secondNE = np.array([no_actions - 2, no_actions - 2])

    payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric = symmetric)

    gameSetup = IdenticalInterestSetup(action_space, use_noisy_utility = use_noisy_utility, payoff_matrix = payoff_matrix)
    game = GameEngine(gameSetup, algorithm = algorithm, max_iter = max_iter)
    
    initial_action_profile = secondNE
    game.set_initial_action_profile(initial_action_profile)
    
    # indicator initial joint action profile distribution
    mu_matrix = np.zeros([1, len(action_space)**no_players])
    mu_matrix[0, initial_action_profile[0]*game.gameSetup.no_actions + initial_action_profile[1]] = 1
    
    # uniform initial joint action profile distribution
    # mu_matrix = np.ones([1, no_actions**no_players])/no_actions**no_players

    if "fast" in algorithm:
        game.set_mu_matrix(mu_matrix)
        potential_history = np.zeros((len(epsilons), n_exp, max_iter))
    else:
        potentials = np.zeros((10, max_iter))
        potential_history = np.zeros((len(epsilons), n_exp, max_iter))

    if use_noisy_utility:
        # To ensure comparability of the results, all games should have the same noise level
        betas = np.zeros(len(epsilons))
        for i, epsilon in enumerate(epsilons):
            betas[i] = game.compute_beta(epsilon)
        
        eta_noise = 1/2.0/np.max(epsilons)
        game.eta_noise = eta_noise
    
    if verbose:
        mean_potential_history = np.zeros((len(epsilons), max_iter))
        std = np.zeros((len(epsilons), max_iter))
        conv_markers = np.zeros((len(epsilons), 2))
        
    repo_root  = Path(__file__).resolve().parents[1]  # Adjust .parent levels if needed
    data_path = repo_root / "data" / "IdenticalInterest" / "games"
    
    folder = "delta_" + str(int(delta*1000)).zfill(4)

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
            
        game.reset_game(payoff_matrix = payoff_matrix)
            
        for idx, epsilon in enumerate(epsilons):
            
            logger.info("       Currently testing epsilon: " + str(epsilon))
            
            if "fast" in algorithm:

                beta = game.compute_beta(epsilon)
                game.play(beta = beta)
                
                potential_history[idx][i] = np.transpose(game.expected_value) 
            else:
                for j in range(10):
                    beta = game.compute_beta(epsilon)
                    game.play(beta = beta)
                
                    potentials[j] = np.transpose(game.potentials_history) 
                
                potential_history[idx][i] = np.mean(potentials, 0) 
                    
    if verbose:
        for idx, epsilon in enumerate(epsilons):
            mean_potential_history[idx] = np.mean(potential_history[idx], 0)
            index = np.argwhere(mean_potential_history[idx] > 1 - epsilon)
            conv_markers[idx, 0] = index[0][0] if len(index) > 0 else None
            conv_markers[idx, 1] = mean_potential_history[idx][int(conv_markers[idx, 0])] if len(index) > 0 else None
            std[idx] = np.std(potential_history[idx], 0)  
        
    if use_noisy_utility:
        algorithm = algorithm + "_noisy"
        
    potentials_path = repo_root / "data" / "experiments" / "epsilon" / f"{algorithm}_potentials.pckl"
    potentials_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(potentials_path, 'wb') as f:
        pickle.dump(potential_history, f, pickle.HIGHEST_PROTOCOL)
        
    if verbose:
        compare_lines(mean_potential_history, [str(epsilon) for epsilon in epsilons], std, markers=conv_markers)
    