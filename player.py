import numpy as np

class Player:
    
    def __init__(self, player_id, rationality, no_actions, utility):
        
        self.id = player_id
        self.beta = rationality
        self.no_actions = no_actions # size of the actions space
        self.utility = utility # utility function
    
    def update(self, idx_opponents_actions): # choose a new action only based on the opponents action, in this case they will be the same as the actions in the previous step
        
        p = np.zeros((self.no_actions,1))
        for i in range(0, self.no_actions):
            p[i] = np.exp(self.beta * self.utility(i, idx_opponents_actions))
        
        p /= np.sum(p)
        
        idx_a = np.random.choice(range(0, self.no_actions), 1, p)
        
        return idx_a
