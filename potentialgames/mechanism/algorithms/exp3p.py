import numpy as np

from ...mechanism.algorithms.abstract_algorithm import LearningAlgorithm
from ...mechanism.game_setup.coverage import CoverageSetup
from ...utils.logger import logger
from ...utils.helpers import rng

class EXP3PAlgorithm(LearningAlgorithm):
    
    @classmethod
    def run(cls, game: "GameEngine"):

        logger.info("EXP3P")

        A = game.no_actions
        N = game.no_players
        max_iter = game.max_iter
        
        # Parameters
        beta = np.sqrt(np.log(A) / (max_iter * A))
        gamma = 1.05 * np.sqrt(np.log(A) * A / max_iter)
        eta = 0.95 * np.sqrt(np.log(A) /(max_iter * A))

        for player_id in range(N):
            player = game.players[player_id]
            player.reset()
            
        past_exp_potential = 0
        
        # Main loop
        for i in range(max_iter):

            if i % 500 == 0:
                logger.info(f"{i}th iteration")

            player_id = rng.integers(0, len(game.players), 1)[0] # randomly choose a player

            player = game.players[player_id]
            
            game.action_profile[player_id] = player.sample_action() # sample an action from the mixed strategy

            opponents_actions = game.action_profile[game.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
            action = game.action_profile[player_id]

            cls.update_player(player, action, opponents_actions, gamma, beta, eta) # update players strategy

            game.potentials_history[i] = (past_exp_potential * i + game.gameSetup.potential_function(game.action_profile))/(i+1) # empirical expected value

            past_exp_potential = game.potentials_history[i]

            if isinstance(game.gameSetup, CoverageSetup):
                game.objectives_history[i] = game.gameSetup.objective(game.action_profile)
                
    @classmethod
    def update_player(cls, player, action, opponents_actions, gamma, beta, eta):
        
        v = player.zeros_vector.copy()
        v[action] = player.utility(action, opponents_actions)

        if player.min_payoff is not None:
            v[action] = (v[action] - player.min_payoff)/(player.max_payoff - player.min_payoff)

        v[action] = v[action]/player.probabilities[0][action]

        player.rewards_estimate = player.rewards_estimate + beta * np.divide(player.ones_vector, player.probabilities) + v

        temp = np.multiply(eta, player.rewards_estimate)
        player.weights = np.exp(temp - np.max(temp))
        player.weights = player.weights / np.sum(player.weights)
        player.probabilities = (1 - gamma) * player.weights + gamma / player.n_actions * player.ones_vector

        player.probabilities = player.probabilities / np.sum(player.probabilities)
