import numpy as np

from potentialgames.mechanism.algorithms.abstract_algorithm import LearningAlgorithm
from potentialgames.utils.logger import logger
from potentialgames.utils.helpers import rng


class BinaryLogLinearAlgorithm(LearningAlgorithm):
    """
    Log-linear learning with two-point feedback.
    """

    @classmethod
    def run(cls, game: "GameEngine", beta: float) -> None:
        """
            Log-linear learning with two-point feedback.
        Args:
            beta (double): Player rationality.
        Raises:
            Exception: Missing arguments.
        """
            
        if beta is None:
            raise ValueError("Rationality parameter 'beta' must be provided.")
        
        logger.info("Binary log-linear learning")

        for i in range(game.max_iter):
            
            if i % 100 == 0:
                logger.info(f"{i}th iteration")
            
            player_id = rng.integers(0, len(game.players), 1)[0] # randomly choose a player 
            player = game.players[player_id]
            
            opponents_actions = game.action_profile[game.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
                
            game.action_profile[player_id] = player.update_log_linear_binary(beta, opponents_actions) # update the players action
                
            game.potentials_history[i] = game.gameSetup.potential_function(game.action_profile) # compute the value of the potential function
