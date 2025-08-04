import numpy as np

from ...mechanism.algorithms.abstract_algorithm import LearningAlgorithm
from ...mechanism.game_setup.coverage import CoverageSetup
from ...utils.logger import logger
from ...utils.helpers import rng


class ExponentialWeightWithAnnealingAlgorithm(LearningAlgorithm):
    
    @classmethod
    def run(cls, game: "GameEngine"):
        
        logger.info("Exponential weight with annealing")
        
        gamma_n = 1
        eps_n = 1

        past_exp_potential = 0

        for player_id in range(game.no_players):
            player = game.players[player_id]
            player.reset()

        # Main loop
        for i in range(game.max_iter):

            gamma_n = 1/(i+1)**0.6  # Proposed step size sequence
            eps_n = 1/(1+(i+1)**0.25*np.log(i+2)**0.5)  # Proposed step vanishing factor

            if i % 500 == 0:
                logger.info(f"{i}th iteration")
            
            player_id = rng.integers(0, len(game.players), 1)[0] # randomly choose a player

            player = game.players[player_id]

            game.action_profile[player_id] = player.sample_action() # sample an action from the mixed strategy
            
            opponents_actions = game.action_profile[game.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
            action = game.action_profile[player_id]

            cls.update_player(player, action, opponents_actions, gamma_n = gamma_n, eps_n = eps_n) # update the mixed strategy

            game.potentials_history[i] =  (past_exp_potential*i + game.gameSetup.potential_function(game.action_profile))/(i+1) #rho@self.gameSetup.potential #self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function

            past_exp_potential = game.potentials_history[i]

            if isinstance(game.gameSetup, CoverageSetup):
                game.objectives_history[i] = game.gameSetup.objective(game.action_profile)
                
    @classmethod
    def update_player(cls, player, action, opponents_actions, gamma_n=1, eps_n=1):
        
        v = player.zeros_vector.copy()
        v[action] = player.utility(action, opponents_actions)

        if player.min_payoff is not None:
            v[action] = (v[action] - player.min_payoff)/(player.max_payoff - player.min_payoff)

        v[action] /= player.probabilities[0][action]
        player.scores += gamma_n * v

        exp_values = np.exp((player.scores - np.max(player.scores)))
        lambda_scores = exp_values/np.sum(exp_values)

        player.probabilities = eps_n*player.ones_vector/player.n_actions + (1-eps_n)*lambda_scores
        player.probabilities = player.probabilities / np.sum(player.probabilities)