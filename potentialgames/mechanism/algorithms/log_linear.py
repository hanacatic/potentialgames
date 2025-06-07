import numpy as np

from potentialgames.mechanism.algorithms.abstract_algorithm import LearningAlgorithm
from potentialgames.utils.logger import logger
from potentialgames.utils.helpers import rng

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
        
        if game.use_noisy_utility and game.gameSetup.eta is None:
            game.gameSetup.eta = 1/2.0/beta
       
        for i in range(0, game.max_iter): 
            
            if i % 50000 == 0:
                logger.info(f"{i}th iteration")
            
            player_id = rng.integers(0, len(game.players), 1)[0] # randomly choose a player
            player = game.players[player_id]
            
            opponents_actions = game.action_profile[game.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
            
            game.action_profile[player_id] = player.update_log_linear(beta, opponents_actions, game.gameSetup.eta, gamma) # update the players action
                
            game.potentials_history[i] = game.gameSetup.potential_function(game.action_profile) # compute the value of the potential function

