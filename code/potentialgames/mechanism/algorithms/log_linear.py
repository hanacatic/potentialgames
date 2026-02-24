import numpy as np

from ...mechanism.algorithms.abstract_algorithm import LearningAlgorithm
from ...utils.logger import logger
from ...utils.math import rng

class LogLinearAlgorithm(LearningAlgorithm):
    """
    Log-linear learning algorithm.
    """

    @classmethod
    def run(cls, game: "GameEngine", beta: float, gamma: float = 0) -> None:
        """
            Log-linear learning algorithm.
        Args:
            beta (double): Player rationality.
            gamma (int, optional): Exploration factor. Defaults to 0.
        Raises:
            Exception: Missing arguments.
        """
        if beta is None:
            raise ValueError("Rationality parameter 'beta' must be provided.")

        logger.info("Log-linear learning")
        
        if game.use_noisy_utility and game.eta_noise is None:
            game.eta_noise = 1/2.0/beta
       
        for i in range(0, game.max_iter): 
            
            if i % 50000 == 0:
                logger.info(f"{i}th iteration")
            
            player_id = rng.integers(0, len(game.players), 1)[0] # randomly choose a player
            player = game.players[player_id]
            
            opponents_actions = game.action_profile[game.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
            
            game.action_profile[player_id] = cls.update_player(player, beta, opponents_actions, game.eta_noise, gamma) # update the players action
                
            game.potentials_history[i] = game.gameSetup.potential_function(game.action_profile) # compute the value of the potential function

    @classmethod
    def update_player(cls, player, beta, opponents_actions, eta_noise, gamma=0):
        """
            Update player's strategy and sample new action from the strategy - based on log-linear learning.
        Args:
            beta (float): Player's rationality
            opponents_actions (np.ndarray): Joint action profile of the opponents.
            eta (float): Noise.
            gamma (float, optional): Exploration factor. Defaults to 0.

        Raises:
            Exception: Noise bound not provided.

        Returns:
            int: index of the chosen action
        """

        if player.utilities is None or not np.array_equal(player.previous_opponents_actions, opponents_actions):
            
            player.compute_utilities(opponents_actions, eta_noise)
            
            # compute strategy        
            exp_values = np.exp(beta * (player.utilities - np.max(player.utilities)))
            player.probabilities = exp_values/np.sum(exp_values)
            
            player.probabilities = gamma/player.n_actions + (1-gamma)*player.probabilities
            player.previous_opponents_actions = opponents_actions
            
        return player.sample_action()