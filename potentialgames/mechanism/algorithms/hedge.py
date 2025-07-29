from ...mechanism.algorithms.abstract_algorithm import LearningAlgorithm

class HedgeAlgorithm(LearningAlgorithm):
    """
        Multiplicative weight update Hedge implementation.
        Y. Freund and R. E. Schapire, ‘A decision-theoretic generalization of on-line learning and an application to boosting’
    """
    @classmethod
    def run(cls, game: "GameEngine", beta: float, scale_factor: int) -> None:

        gamma_t = np.sqrt(8*np.log(game.no_actions)/game.max_iter) # exploration factor

        for player_id in range(game.no_players):
            player = game.players[player_id]
            player.reset()
        past_exp_potential = 0

        # Main loop
        for i in range(game.max_iter):

            if i % 20 == 0:
                print(str(i) + "th iteration")
                        
            player_id = rng.integers(0, len(game.players), 1)[0] # randomly choose a player

            player = game.players[player_id]

            game.action_profile[player_id] = player.sample_action()

            opponents_actions = game.action_profile[game.opponents_idx_map[player_id]] # extract the opponents actions from the action profile

            cls.update_mw(player, gamma_t, opponents_actions) # update players mixed strategy

            game.potentials_history[i] =  (past_exp_potential*i + game.potential_function(game.action_profile))/(i+1) # compute the value of the potential function

            past_exp_potential = game.potentials_history[i]

            if isinstance(game.gameSetup, CongestionGame):
                game.objectives_history[i] = game.gameSetup.objective(game.action_profile)
                
     @classmethod
    def update_player(cls, player, gamma_t, opponents_actions):
        
        if player.utilities is None or not np.array_equal(player.previous_opponents_actions, opponents_actions):
            
            player.compute_utilities(opponents_actions)

        losses = player.ones_vector - player.utilities

        player.probabilities = np.multiply(player.probabilities, np.exp(np.multiply(gamma_t, -losses))) # update straetgy

        player.probabilities = player.probabilities / np.sum(player.probabilities)