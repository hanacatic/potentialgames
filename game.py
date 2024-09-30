import numpy as np
from player import Player
from helpers import rejection_sampling


class Game:
    
    max_iter = 1000   
    
    def __init__(self, no_players, rationality, action_space, utility_functions, potential_function, mu): # mu - initial distribution
        
        self.action_space = action_space
        self.no_players = no_players
        self.players = np.array([ Player(i, rationality, len(self.action_space), utility_functions[i]) for i in range(0, self.no_players)], dtype = object)
        
        self.action_profile = np.random.randint(0, len(self.action_space), no_players) # discrete uniform distribution
        self.action_profile = self.sample_initial_action_profile(mu)
        
        self.action_profile_history = np.zeros((self.max_iter, self.no_players))
        self.player_id_history = np.zeros((self.max_iter, 1))
        
        self.potential_function = potential_function
        self.potentials_history = np.zeros((self.max_iter, 1))
        
        self.last_change = 0
        self.player_converged_history = np.zeros((self.max_iter, 1))
        
        self.converged = False
        self.converged_iteration = self.max_iter - 1
        
    def sample_initial_action_profile(self, mu):
        
        self.initial_action_profile = rejection_sampling(mu, self.action_profile, len(self.action_space))
        
        return self.initial_action_profile
    
    def set_initial_action_profile(self, initial_action_profile):
        
        self.initial_action_profile = initial_action_profile
        
    def play(self, initial_action_profile = None):
        
        if initial_action_profile == None:
            self.action_profile = self.initial_action_profile.copy()
        else:
            self.action_profile = initial_action_profile
        
        for i in range(0, self.max_iter): 

            self.action_profile_history[i] = self.action_profile.copy()
            
            player_id = np.random.randint(0, len(self.players), 1) # randomly choose a player
            
            self.player_id_history[i] = player_id

            player = self.players[player_id][0] 
            
            opponents_actions = self.action_profile.copy() # extract the opponents actions from the action profile
            opponents_actions = np.delete(opponents_actions, player_id)
            
            self.action_profile[player_id] = player.update(opponents_actions) # update the players action
            
            self.potentials_history[i] = self.potential_function(self.action_profile) # compute the value of the potential function
            self.player_converged_history[i] = player.converged

            if player.changed*player.converged:
                self.last_change = i
                self.converged = False
            elif (not self.converged) and len(np.unique(self.player_id_history[self.last_change:i])) == self.no_players and all(self.player_converged_history[self.last_change:i]):
                self.converged = True
                self.converged_iteration = i

class RandomIdenticalInterestGame:
    
    def __init__(self, action_space, firstNE, secondNE, delta): 
        
        self.no_players = 2
        self.no_actions = len(action_space)
        self.action_space = action_space
        self.firstNE = firstNE
        self.secondNE = secondNE
        self.delta = delta
        self.payoff = self.generate_payoff_matrix()
    
    def generate_payoff_matrix(self):
        
        payoff = np.random.uniform(0.0, 1 - self.delta, size = (self.no_actions, self.no_actions))
        
        payoff[self.firstNE[0], self.firstNE[1]] = 1
        payoff[self.secondNE[0], self.secondNE[1]] = 1 - self.delta
        
        return payoff

    def utility_function_player_1(self, player_action, opponents_action):

        return self.payoff[player_action, opponents_action]

    def utility_function_player_2(self, player_action, opponents_action):
        
        return np.transpose(self.payoff)[player_action, opponents_action]

    def potential_function(self, action_profile):
        
        return self.payoff[action_profile[0], action_profile[1]]