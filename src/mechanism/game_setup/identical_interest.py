import numpy as np
from src.utils.helpers import * 
from src.mechanism.game_setup import AbstractGameSetup
from scipy.sparse import lil_matrix, csr_matrix
from functools import partial

rng = np.random.default_rng()

class IdenticalInterestSetup(AbstractGameSetup):
    """
        Class representing the Identical Interest Matrix game.
        
        Includes information on the number of players and actions, utility functions and potential function.
    """
    
    def __init__(self, action_space, no_players, firstNE, secondNE, delta, type = "Asymmetrical", noisy_utility = False, eta = None, payoff_matrix = None): 
        """
            Identical Interest game constructor. 
        Args:
            action_space (np.array): action space, assumed to be identical for all players
            no_players (_type_): number of players in the game.
            firstNE (_type_): coordinates of the largest Nash equilibrium.
            secondNE (_type_): coordinates of the second largest Nash equilibrium.
            delta (_type_): the difference of potential between the first NE and second NE.
            type (str, optional): type of game. Defaults to "Asymmetrical".
            noisy_utility (bool, optional): Use noisy utilities. Defaults to False.
            eta (double, optional): Noise range. Defaults to None.
            payoff_matrix (_type_, optional): payoff matrix. Defaults to None.
        """
        
        self.no_players = no_players
        self.no_actions = len(action_space)
        self.no_action_profiles = self.no_actions**self.no_players
        self.action_space = [action_space]*self.no_players
        self.firstNE = firstNE
        self.secondNE = secondNE
        self.delta = delta
        self.type = type
        self.noisy_utility = noisy_utility
        
        self.symmetric = False
        
        if self.noisy_utility:
            self.eta = eta
        elif eta is not None:
            raise Exception("Sorry, the eta is not null, but the noisy utility mode is not enabled!")
        else:
            self.eta = 0
        
        self.action_profile_template = [0]*self.no_players
        self.potential_vec = np.zeros((self.no_action_profiles, 1))

        self.opponents_idx_map = [ np.delete(np.arange(self.no_players), player_id) for player_id in range(self.no_players) ]

        # If payoff matrix has not been defined then generate a random payoff matrix
        if payoff_matrix is None:
            self.generate_payoff_matrix()
        else:
            self.set_payoff_matrix(delta, payoff_matrix)

        # Array of player utility functions, formulated based on a general utility function
        self.utility_functions = []
        for i in range(0, self.no_players):
            self.utility_functions.append(partial(self.utility_function, i))
    
    def formulate_transition_matrix(self, beta): 
        """
        Generate transition matrix for the game with given rationality.

        Args:
            beta (double): player rationality. Assumed that all players have the same rationality.

        Returns:
            np.array(A^N x A^N): transition matrix.
        """
                         
        P = np.zeros([self.no_action_profiles, self.no_action_profiles])     

        for idx in range(self.no_action_profiles):
            
            profile = np.unravel_index(idx, (self.no_actions,)*(self.no_players))

            self.potential_vec[idx] = self.potential_function(profile)
                        
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
        """
            Generate a sparse transition matrix for the game with given rationality.
        Args:
            beta (double): player rationality. Assumed that all players have the same rationality.

        Returns:
            csr_matrix: sprase transition matrix
        """
            
        P_row, P_col, P_data = [], [], []

        i = 0

        strides = self.no_actions ** (self.no_players - 1 - np.arange(0, self.no_players))
        action_space = np.arange(0, self.no_actions).tolist()
        while i < self.no_actions**(self.no_players - 1):
            
            opponents_actions = np.unravel_index(i, (self.no_actions,)*(self.no_players - 1))

            for player_id in range(self.no_players):
                
                utilities = np.array([self.utility_functions[player_id](i, opponents_actions) for i in range(self.no_actions)])
                
                exp_values = np.exp(beta * (utilities - np.max(utilities)))
        
                p = exp_values/np.sum(exp_values)
                
                stride = strides[player_id]
                idx = opponents_actions @ strides[self.opponents_idx_map[player_id]]
                                
                for j, prob in enumerate(p):
                    P_row.extend(idx + np.multiply(action_space, stride))
                    P_col.extend([idx + j * stride]*self.no_actions)
                    P_data.extend([prob / self.no_players]*self.no_actions)
                      
            i += 1 
        P_data = P_data 
        
        P = csr_matrix((P_data, (P_row, P_col)), shape=(self.no_action_profiles, self.no_action_profiles))
        P.sum_duplicates()
        self.P = P

        self.formulate_potential_vec()
        return self.P

    def formulate_transition_matrix_binary(self, beta): 
        """
        Generate transition matrix for the binary log linear game with given rationality.

        Args:
            beta (double): player rationality. Assumed that all players have the same rationality.

        Returns:
            np.array(A^N x A^N): transition matrix.
        """
                         
        P = np.zeros([self.no_action_profiles, self.no_action_profiles])     

        for idx in range(self.no_action_profiles):
            
            profile = np.unravel_index(idx, (self.no_actions,)*(self.no_players))

            self.potential_vec[idx] = self.potential_function(profile)
                        
            for player_id in range(self.no_players):
                
                opponents_actions = profile[self.opponents_idx_map[player_id][0]] # extract the opponents actions from the action profile
                player_action = profile[player_id]
                
                player_utility = self.utility_functions[player_id](player_action, opponents_actions)
                
                i = idx

                for a in range(self.no_actions):
                    
                    a_utility = self.utility_functions[player_id](a, opponents_actions)
                    utilities = np.array([player_utility, a_utility])
                    
                    exp_values = np.exp(beta * utilities)
                    
                    p = exp_values/np.sum(exp_values)
                    
                    p = 1/self.no_players/self.no_actions*p
                
                    j = idx + (a - player_action)*self.no_actions**(self.no_players - 1 - player_id)
                    
                    P[idx, i] += p[0]
                    P[idx, j] += p[1]
                    
        self.P = P
                
        return self.P
           
    def generate_payoff_matrix(self):
        """
            Generate random payoff matrix.
        """
        
        self.payoff_player_1 = np.random.uniform(0.0, 1 - self.delta, size = [self.no_actions] * self.no_players)
        
        self.payoff_player_1[tuple(self.firstNE)] = 1
        self.payoff_player_1[tuple(self.secondNE)] = 1 - self.delta
        
        if self.type == "Symmetrical": 
            self.payoff_player_1 = make_symmetric_nd(self.payoff_player_1)
    
    def reset_payoff_matrix(self, delta = None):
        """
            Generate a new random payoff matrix for the game.
        Args:
            delta (_type_, optional): the difference of potential between the first NE and second NE. Defaults to None.
        """

        if delta:
            self.delta = delta
            
        self.generate_payoff_matrix()
     
    def set_payoff_matrix(self, delta, payoff):
        """
            Set new payoff matrix.
        Args:
            payoff(AxAx...A): payoff matrix.
        """
        
        # Get dimensions of the payoff matrix and update N, A and action space accordingly

        self.payoff_player_1 = payoff.copy()
        self.delta = delta
        
        # Make the payoff matrix symmetric
        if self.type == "Symmetrical":
            self.payoff_player_1 = make_symmetric_nd(self.payoff_player_1)

    def potential_function(self, action_profile):
        """
            Compute value of potential function at a given joint action profile.
        Args:
            action_profile (Nd np.array): joint action profile.

        Returns:
            double: value of the potential function.
        """
        return self.payoff_player_1[tuple(action_profile)]
    
    def utility_function(self, player_id, player_action, opponents_action):
        """
            Compute utility of an action given opponents actions.
        Args:
            player_id (int): player id.
            player_action (int): player action.
            opponents_action (N-1d np.array): opponents actions.
            eta (double): noise bound
        Returns:
            double: utility of action given opponents actions.
        """

        if self.no_players > 2:
            self.action_profile_template[:player_id] = opponents_action[:player_id]
            self.action_profile_template[player_id + 1:] = opponents_action[player_id:]
            self.action_profile_template[player_id] = player_action
        else:
            self.action_profile_template[player_id] = player_action
            self.action_profile_template[not player_id] = opponents_action
    
        return self.potential_function(self.action_profile_template)
    
    def formulate_potential_vec(self):
        """
            Formulate sparse vector of potential function values for all joint action profiles.
            Feasible when the total number of joint action profiles is reasonable.
        """
        
        self.potential_vec = lil_matrix((self.no_action_profiles, 1))
        
        for idx in np.arange(self.no_action_profiles):
            element = np.unravel_index(idx, (self.no_actions,)*(self.no_players))
            self.potential_vec[idx] = self.potential_function(element)