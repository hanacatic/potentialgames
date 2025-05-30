import numpy as np

from src.mechanism.algorithms.abstract_algorithm import LearningAlgorithm
from src.utils.logger import logger
from src.utils.helpers import rng


class BinaryLogLinearAlgorithm(LearningAlgorithm):
    """
    Log-linear learning with two-point feedback.
    """
    
    def __init__(self):
        super().__init__()

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

        for i in range(self.max_iter):
            
            if i % 100 == 0:
                logger.info(f"{i}th iteration")
            
            player_id = rng.integers(0, len(self.players), 1)[0] # randomly choose a player 
            player = self.players[player_id]
            
            opponents_actions = self.action_profile[self.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
                
            self.action_profile[player_id] = player.update_log_linear_binary(beta, opponents_actions) # update the players action
                
            self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
