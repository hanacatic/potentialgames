import numpy as np
from player import Player
from helpers import rejection_sampling

# constants 
RATIONALITY = 0 
EPS = 1e-3

class Game:
    max_iter = 1
    
    def __init__(self, no_players, action_space, utility_functions, mu): # mu - initial distribution
        
        self.players = np.array([ Player(i, RATIONALITY, len(action_space), utility_functions[i]) for i in range(0, no_players)], dtype = object)
        
        self.action_profile = np.random.randint(0, len(action_space), no_players)
        self.action_profile = rejection_sampling(mu, self.action_profile, len(action_space))
        
    def play(self):
        
        for i in range(0, self.max_iter):
            
            player_id = np.random.randint(0, len(self.players), 1)
            player = self.players[player_id][0] 
                       
            self.action_profile[player_id] = player.update(self.action_profile)
    

