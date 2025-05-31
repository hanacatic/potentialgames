import numpy as np

from src.mechanism.algorithms.abstract_algorithm import LearningAlgorithm
from src.utils.logger import logger
from src.utils.helpers import rng

class ModifiedLogLinearAlgorithm(LearningAlgorithm):
    """
    Modified log-linear learning algorithm.
    """

    @classmethod
    def run(cls, game: "GameEngine", beta: float) -> None:
        """
            Modified log-linear learning leveraging symmetry of the game.

        Args:
            beta (double): Player rationality.
        """
        [game.players[i].set_modified_utility(game.gameSetup.modified_utility_functions[i]) for i in game.player_idx_map]
        
        game.phi = np.zeros(game.gameSetup.no_actions) # histogram over the action space
        
        for a in game.action_profile:
            game.phi[a] += 1
        
        for i in range(game.max_iter):
            
            # determine the probability that the player is going to change change his action based on the action they played in the previous round
            # implemented based on section 3.1 in Dynamics in Congestion Games, D. Shah and J. Shin
            player_clock = [1/game.gameSetup.no_players*game.phi[game.action_profile[i]] for i in range(game.gameSetup.no_players)]
            player_clock = player_clock/np.sum(player_clock)
                    
            player_id = rng.choice(game.player_idx_map, size=1, p=player_clock)[0]
            
            player = game.players[player_id]
            
            game.phi[game.action_profile[player_id]] -= 1
            
            game.action_profile[player_id] = player.update_modified_log_linear(beta, game.phi) # update chosen players strategy
            
            game.potentials_history[i] = game.gameSetup.potential_function(game.action_profile) # compute the value of the potential function
            
            game.phi[game.action_profile[player_id]] += 1 
