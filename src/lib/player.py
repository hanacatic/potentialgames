import numpy as np

rng = np.random.default_rng()

class Player:
    
    def __init__(self, player_id, action_space, utility):
        
        self.id = player_id
        self.no_actions = len(action_space) # size of the actions space
        self.utility = utility # utility function
        self.past_action = None
        self.action_space = np.arange(self.no_actions).reshape(1, self.no_actions)
        self.prob = 1/self.no_actions*np.ones([1, self.no_actions])
        self.initial_action = np.array([0])
        self.ones = np.ones(self.no_actions)
        self.past_opponents_actions = None
        self.utilities = None
        
    def update_log_linear(self, beta, opponents_actions): # choose a new action only based on the opponents action, in this case they will be the same as the actions in the previous step
        
        if self.utilities is None or all(self.past_opponents_actions != opponents_actions):
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)
        
            exp_values = np.exp(beta * self.utilities)
        
            self.prob = exp_values/np.sum(exp_values)
            self.past_opponents_actions = opponents_actions
        
        idx_a = rng.choice(self.action_space[0], size=1, p=self.prob[0])

        self.past_action = idx_a
        
        return idx_a
    
    def best_response(self, opponents_actions):
        
        if self.utilities is None or self.past_opponents_actions != opponents_actions:
        
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)

            idx_a = np.argmax(self.utilities)
        
            self.past_action = idx_a
            self.past_opponents_actions = opponents_actions
        
        return self.past_action
    
    def mixed_strategy(self):
        
        return self.prob
        
    def update_mw(self, opponents_actions, gamma_t = 0.5):
        
        if self.utilities is None or self.past_opponents_actions != opponents_actions:
            
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)
            self.past_opponents_actions = opponents_actions

        losses = self.ones - self.utilities
        
        self.prob = np.multiply( self.prob, 1 + gamma_t * (-losses))
        
        self.prob = self.prob / np.sum(self.prob)
        
    def reset_player(self, no_actions, utility):

        self.no_actions = no_actions # size of the actions space
        self.utility = utility # utility function