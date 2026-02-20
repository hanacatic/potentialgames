import numpy as np

from ...mechanism.algorithms.abstract_algorithm import LearningAlgorithm
from ...utils.logger import logger
from ...utils.helpers import rng


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
            
            print(f"Player {player_id} is updating their action", end="\n")  # Player update indicator
            opponents_actions = game.action_profile[game.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
                
            game.action_profile[player_id] = cls.update_player(player, beta, opponents_actions) # update the players action
                
            game.potentials_history[i] = game.gameSetup.potential_function(game.action_profile) # compute the value of the potential function

    @classmethod
    def update_player(
        cls,
        player,
        beta: float,
        opponents_actions: np.ndarray
    ) -> int:
        """
            Update player's strategy and sample new action from the strategy - based on log-linear learning with two-point feedback.

        Args:
            beta (float): Player's rationality
            opponents_actions (np.ndarray): Joint action profiles of the opponents.

        Returns:
            int: Chosen action.
        """
        
        # sample trial action
        new_action = rng.integers(0, player.n_actions, 1).astype(int)[0]
        new_utility = player.utility(new_action, opponents_actions)
        
        if player.utilities is None or not np.array_equal(player.previous_opponents_actions, opponents_actions):
            player.utilities = player.utility(player.previous_action, opponents_actions)
        
        player.action_space = [[player.previous_action, new_action]]
        # compute utilities
        utilities = np.array([player.utilities, new_utility])
        # compute strategy
        exp_values = np.exp(beta * (utilities - np.max(utilities)))
        probabilities = exp_values/np.sum(exp_values)
        player.probabilities = probabilities.T

        return player.sample_action()
    