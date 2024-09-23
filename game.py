import numpy as np
from player import Player
from helpers import rejection_sampling


class Game:
    
    max_iter = 20000   
    
    def __init__(self, no_players, rationality, action_space, utility_functions, potential_function, mu): # mu - initial distribution
        
        self.action_space = action_space
        self.no_players = no_players
        self.players = np.array([ Player(i, rationality, len(self.action_space), utility_functions[i]) for i in range(0, self.no_players)], dtype = object)
        
        self.action_profile = np.random.randint(0, len(self.action_space), no_players) # discrete uniform distribution
        self.action_profile = self.sample_initial_action_profile(mu)
        
        self.action_profile_history = np.zeros((self.max_iter, self.no_players))
        
        self.potential_function = potential_function
        self.potentials = np.zeros((self.max_iter, 1))
        
    def sample_initial_action_profile(self, mu):
        
        self.initial_action_profile = rejection_sampling(mu, self.action_profile, len(self.action_space))
        
        return self.initial_action_profile
        
    def play(self):
        
        self.action_profile = self.initial_action_profile.copy()
        
        for i in range(0, self.max_iter): 

            self.action_profile_history[i] = self.action_profile.copy()
            
            player_id = np.random.randint(0, len(self.players), 1) # randomly choose a player

            player = self.players[player_id][0] 
            
            opponents_actions = self.action_profile.copy() # extract the opponents actions from the action profile
            opponents_actions = np.delete(opponents_actions, player_id)
            
            self.action_profile[player_id] = player.update(opponents_actions) # update the players action
            
            self.potentials[i] = self.potential_function(self.action_profile) # compute the value of the potential function
            