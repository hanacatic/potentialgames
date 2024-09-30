import numpy as np

class Player:
    
    def __init__(self, player_id, no_actions, utility):
        
        self.id = player_id
        self.no_actions = no_actions # size of the actions space
        self.utility = utility # utility function
        self.past_action = None
        self.changed = True
        self.converged = False
    
    def update_log_linear(self, beta, idx_opponents_actions): # choose a new action only based on the opponents action, in this case they will be the same as the actions in the previous step
        
        p = np.zeros((self.no_actions, 1))
        
        for i in range(0, self.no_actions):

            p[i] = np.exp(beta * self.utility(i, idx_opponents_actions)) # compute the probability of chosing any of the actions in the action space
        
        p /= np.sum(p)

        idx_a = np.random.choice(range(0, self.no_actions), 1, True, np.transpose(p)[0]) # randomly chose an action based on the computed probability distribution
        
        self.converged = any(True for x in p if x > 0.999)
        self.converged = sum(1 for x in p if x < 1e-3) == (self.no_actions - 1)

        if idx_a == self.past_action:
            self.changed = False
        else:
            self.changed = True
            
        self.past_action = idx_a
        return idx_a
