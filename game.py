import numpy as np
from player import Player
from helpers import rejection_sampling


class Game:
    
    max_iter = 10000
    
    def __init__(self, no_players, rationality, action_space, utility_functions, potential_function, mu): # mu - initial distribution
        
        self.players = np.array([ Player(i, rationality, len(action_space), utility_functions[i]) for i in range(0, no_players)], dtype = object)
        
        self.action_profile = np.random.randint(0, len(action_space), no_players)
        self.action_profile = rejection_sampling(mu, self.action_profile, len(action_space))
        # self.action_profile = np.array([0, 0])
        self.initial_action_profile = self.action_profile.copy()

        self.potential_function = potential_function
        self.potentials = np.zeros((self.max_iter, 1))
        
    def play(self):
        
        self.initial_action_profile = self.action_profile.copy()
        
        for i in range(0, self.max_iter): 
            
            player_id = np.random.randint(0, len(self.players), 1) # randomly choose a player

            player = self.players[player_id][0] 
            
            opponents_actions = self.action_profile.copy() # extract the opponents actions from the action profile
            opponents_actions = np.delete(opponents_actions, player_id)
            
            # print('player_id: ', player_id)
            # print('opponnents_actions: ', opponents_actions)
            
            self.action_profile[player_id] = player.update(opponents_actions) # update the players action
            
            self.potentials[i] = self.potential_function(player.id, self.action_profile[player_id], opponents_actions) # compute the value of the potential function
            