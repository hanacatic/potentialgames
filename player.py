import numpy as np

rng = np.random.default_rng()

class Player:
    
    def __init__(self, player_id, no_actions, utility):
        
        self.id = player_id
        self.no_actions = no_actions # size of the actions space
        self.utility = utility # utility function
        self.past_action = None
        self.action_space = np.arange(self.no_actions).reshape(1, self.no_actions)
    
    def update_log_linear(self, beta, idx_opponents_actions): # choose a new action only based on the opponents action, in this case they will be the same as the actions in the previous step

        utilities = np.array([self.utility(i, idx_opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)
        
        exp_values = np.exp(beta * utilities)
        
        self.p = exp_values/np.sum(exp_values)
        
        idx_a = rng.choice(self.action_space[0], size=1, p=self.p[0])

        self.past_action = idx_a
        return idx_a
    
    def best_response(self, idx_opponents_actions):
        
        utilities = np.array([self.utility(i, idx_opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)

        idx_a = np.argmax(utilities)
        
        self.past_action = idx_a
        
        return idx_a
        
    def reset_player(self, no_actions, utility):

        self.no_actions = no_actions # size of the actions space
        self.utility = utility # utility function
