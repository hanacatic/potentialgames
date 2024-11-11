import numpy as np
from lib.aux_functions.helpers import *
from scipy.sparse import lil_matrix, csc_matrix
from functools import partial

class IdenticalInterestGame:
    
    def __init__(self, action_space, no_players, firstNE, secondNE, delta, type = "Asymmetrical", payoff_matrix = None): 
        
        self.no_players = no_players
        self.no_actions = len(action_space)
        self.no_action_profiles = self.no_actions**self.no_players
        self.action_space = action_space
        self.firstNE = firstNE
        self.secondNE = secondNE
        self.delta = delta
        self.type = type
        self.action_profile_template = [0]*self.no_players
        self.opponents_idx_map = [ np.delete(np.arange(self.no_players), player_id) for player_id in range(self.no_players) ]

        if payoff_matrix is None:
            self.generate_payoff_matrix()
        else:
            self.set_payoff_matrix(payoff_matrix)

        self.utility_functions = []
        
        for i in range(0, self.no_players):
            self.utility_functions.append(partial(self.utility_function, i))
            # self.utility_functions.append(lambda player_action, opponents_action: self.utility_function(i, player_action, opponents_action))
    
    def formulate_transition_matrix(self, beta): 
                 
        self.potential = np.zeros((self.no_action_profiles, 1))
        
        P = np.zeros([self.no_action_profiles, self.no_action_profiles])     

        for idx in range(self.no_action_profiles):
            
            profile = np.unravel_index(idx, (self.no_actions,)*(self.no_players))

            self.potential[idx] = self.potential_function(profile)
                        
            for player_id in range(self.no_players):
                
                opponents_actions = profile[self.opponents_idx_map[player_id][0]] # extract the opponents actions from the action profile
                
                utilities = np.array([self.utility_functions[player_id](i, opponents_actions) for i in range(self.no_actions)])
                exp_values = np.exp(beta * utilities)
        
                p = exp_values/np.sum(exp_values)
                
                i = idx - profile[player_id]*self.no_actions**(self.no_players - 1 - player_id)
                stride = self.no_actions ** (self.no_players - 1 - player_id)
                
                P[idx, i: i + self.no_actions**(self.no_players - player_id) : stride] += 1/self.no_players*p
                     
        self.P = P
                
        return self.P
         
    def formulate_transition_matrix_sparse(self, beta):
            
        P_row, P_col, P_data = [], [], []

        i = 0

        strides = self.no_actions ** (self.no_players - 1 - np.arange(0, self.no_players))
        action_space = np.arange(0, self.no_actions)
       
        while i < self.no_actions**(self.no_players - 1):
            
            opponents_actions = np.unravel_index(i, (self.no_actions,)*(self.no_players - 1))

            for player_id in range(self.no_players):
                
                utilities = np.array([self.utility_functions[player_id](i, opponents_actions) for i in range(self.no_actions)])
                
                exp_values = np.exp(beta * (utilities - np.max(utilities)))
        
                p = exp_values/np.sum(exp_values)
                
                stride = strides[player_id]
                idx = opponents_actions @ strides[self.opponents_idx_map[player_id]]
                
                for j, prob in enumerate(p):
                    P_row.extend(idx + action_space*stride)
                    P_col.extend([idx + j * stride]*self.no_actions)
                    P_data.extend([prob / self.no_players]*self.no_actions)   
                      
            i += 1 
            
        P = csc_matrix((P_data, (P_row, P_col)), shape=(self.no_action_profiles, self.no_action_profiles))
        P.sum_duplicates()
        self.P = P

        self.formulate_potential_vec()
        return self.P
           
    def generate_payoff_matrix(self):
        
        self.payoff_player_1 = np.random.uniform(0.0, 1 - self.delta, size = [self.no_actions] * self.no_players)
        
        self.payoff_player_1[tuple(self.firstNE)] = 1
        self.payoff_player_1[tuple(self.secondNE)] = 1 - self.delta
        
        if self.type == "Symmetrical": 
            self.payoff_player_1 = make_symmetric_nd(self.payoff_player_1)
    
    def reset_payoff_matrix(self, delta = None):
        
        if delta:
            self.delta = delta
            
        self.generate_payoff_matrix()
     
    def set_payoff_matrix(self, payoff):
        
        self.payoff_player_1 = payoff
        
        if self.type == "Symmetrical":
            self.payoff_player_1 = make_symmetric_nd(self.payoff_player_1)

    def potential_function(self, action_profile):
        
        return self.payoff_player_1[tuple(action_profile)]
    
    def utility_function(self, player_id, player_action, opponents_action):

        if self.no_players > 2:
            self.action_profile_template[:player_id] = opponents_action[:player_id]
            self.action_profile_template[player_id + 1:] = opponents_action[player_id:]
            self.action_profile_template[player_id] = player_action
        else:
            self.action_profile_template[player_id] = player_action
            self.action_profile_template[not player_id] = opponents_action
        
        return self.potential_function(self.action_profile_template)
        
    def formulate_potential_vec(self):
        
        self.potential = lil_matrix((self.no_action_profiles, 1))
        
        for idx in np.arange(self.no_action_profiles):
            element = np.unravel_index(idx, (self.no_actions,)*(self.no_players))
            self.potential[idx] = self.potential_function(element)