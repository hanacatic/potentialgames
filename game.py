import numpy as np
from player import Player
from helpers import rejection_sampling

rng = np.random.default_rng()

class Game:
    
    def __init__(self, gameSetup, algorithm = 'log_linear',  max_iter = 200000, mu = None): # mu - initial distribution
        
        self.gameSetup = gameSetup # all the game rules and game data
        self.algorithm = algorithm
        self.max_iter = int(max_iter)
        
        self.players = np.array([ Player(i, self.gameSetup.no_actions, gameSetup.utility_functions[i]) for i in range(0, self.gameSetup.no_players)], dtype = object)
        
        self.action_profile = np.random.randint(0, self.gameSetup.no_players, self.gameSetup.no_players) # discrete uniform distribution
        self.action_profile = self.sample_initial_action_profile(mu)
           
        self.potentials_history = np.zeros((self.max_iter, 1))
        self.expected_value = None
               
    def sample_initial_action_profile(self, mu):
        
        self.initial_action_profile = rejection_sampling(mu, self.action_profile, self.gameSetup.no_actions)
        
        return self.initial_action_profile
    
    def set_initial_action_profile(self, initial_action_profile):
        
        self.initial_action_profile = initial_action_profile
    
    def set_mu_matrix(self, mu_matrix):
        
        self.mu_matrix = mu_matrix

    def play(self, initial_action_profile = None, beta = None):
        
        if initial_action_profile == None:
            self.action_profile = self.initial_action_profile.copy()
        else:
            self.action_profile = initial_action_profile
        
        match self.algorithm:
            case "log_linear":
                self.log_linear(beta)
            case "log_linear_t":
                self.log_linear_t()
            case "log_linear_fast":
                # self.set_mu_matrix(mu_matrix)
                self.log_linear_fast(beta)
            case "best_response":
                self.best_response()
            case "multiplicative_weight":
                self.multiplicative_weight()
            case _:
                self.log_linear(beta)
        
    def log_linear(self, beta):
        
        for i in range(0, self.max_iter): 
            
            self.log_linear_iteration(i, beta)
    
    def log_linear_t(self):
        
        for i in range(self.max_iter): 

            # beta = np.log(i+1)*(self.gameSetup.no_players**2+1)/self.gameSetup.no_players
            
            # beta = np.log(i+1)
            
            beta = min((i+1), 500)
            
            self.log_linear_iteration(i, beta)
    
    def log_linear_fast(self, beta):
        
        P = self.gameSetup.formulate_transition_matrix(beta)
        self.gameSetup.formulate_potential_vec()
        mu0 = self.mu_matrix.copy()
        
        self.expected_value = np.zeros((int(self.max_iter), 1))
        
        for i in range(self.max_iter):
            
            mu = mu0 @ P
            
            mu0 = mu
            
            self.expected_value[i] = mu @ self.gameSetup.potential

        
        self.stationary = mu
                           
    def log_linear_iteration(self, i, beta):
        
        player_id = rng.integers(0, len(self.players), 1) # randomly choose a player
            
        player = self.players[player_id][0] 
            
        mask = np.arange(len(self.action_profile)) != player_id
        opponents_actions = self.action_profile[mask] # extract the opponents actions from the action profile
            
        self.action_profile[player_id] = player.update_log_linear(beta, opponents_actions) # update the players action
            
        self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
         
    def best_response(self):
        
        for i in range(self.max_iter):
            
            player_id = rng.integers(0, len(self.players), 1) # randomly choose a player
            
            player = self.players[player_id][0] 
            
            mask = np.arange(len(self.action_profile)) != player_id
            opponents_actions = self.action_profile[mask] # extract the opponents actions from the action profile
        
            self.action_profile[player_id] = player.best_response(opponents_actions) # update the players action

            self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
    
    def multiplicative_weight(self):
        
        gamma_t = np.sqrt(np.log(self.gameSetup.no_players)/self.max_iter) 
              
        for i in range(self.max_iter):
            
            mixed_strategies = np.zeros([self.gameSetup.no_players, self.gameSetup.no_actions])
                        
            for player_id in range(self.gameSetup.no_players):
                
                player = self.players[player_id]
                
                mixed_strategies[player_id] = player.mixed_strategy()
                
                self.action_profile[player_id] = rng.choice(self.gameSetup.action_space, 1, p = mixed_strategies[player_id])
            
            self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function

            for player_id in range(self.gameSetup.no_players):
                
                mask = np.arange(len(self.action_profile)) != player_id
                opponents_actions = self.action_profile[mask] # extract the opponents actions from the action profile
                
                player = self.players[player_id]
                
                player.update_mw(opponents_actions, gamma_t = gamma_t)
                                      
    def compute_beta(self, epsilon):
        
        A = self.gameSetup.no_actions
        N = self.gameSetup.no_players
        delta = self.gameSetup.delta
        
        # return 1/max(epsilon/2, delta)*np.log(A**N*(1-epsilon/2)*(4/(epsilon*A**N*(epsilon/2)) - 1/(A**N*(epsilon/2))))
        
        return 1/max(epsilon, delta)*np.log(A**N/epsilon)

    def compute_t(self, epsilon):
        
        A = self.gameSetup.no_actions
        N = self.gameSetup.no_players
        delta = self.gameSetup.delta
        beta = self.compute_beta(epsilon)
        
        # return 25*N**2*A**5*np.exp(4*beta)/16/np.pi**2*(np.log(np.log(A**N)) + np.log(beta) + 2*np.log(4/epsilon))
        
        return N**2*A**5*(A**N/epsilon)**(1/max(epsilon, delta))
        # return self.game.no_players*(self.game.no_players**self.game.no_actions/epsilon)**(1/max(epsilon, self.game.delta))

    def set_max_iter(self, epsilon):
        self.max_iter = int(min(1e5, self.compute_t(epsilon)))
        
        self.action_profile_history = np.zeros((self.max_iter, self.gameSetup.no_players))
        self.player_id_history = np.zeros((self.max_iter, 1))
        
        self.potentials_history = np.zeros((self.max_iter, 1))
        self.player_converged_history = np.zeros((self.max_iter, 1))
    
    def reset_game(self):
        
        # self.gameSetup = gameSetup
        [self.players[i].reset_player(self.gameSetup.no_actions, self.gameSetup.utility_functions[i]) for i in range(0, self.gameSetup.no_players)]

class IdenticalInterestGame:
    
    def __init__(self, action_space, firstNE, secondNE, delta, type = "Asymmetrical", payoff_matrix = None): 
        
        self.no_players = 2
        self.no_actions = len(action_space)
        self.no_action_profiles = self.no_actions**self.no_players
        self.action_space = action_space
        self.firstNE = firstNE
        self.secondNE = secondNE
        self.delta = delta
        self.type = type
        
        if payoff_matrix.all() == None:
            self.generate_payoff_matrix()
        else:
            self.set_payoff_matrix(payoff_matrix)

        self.utility_functions = [self.utility_function_player_1, self.utility_function_player_2]
        
    def formulate_transition_matrix(self, beta):
    
        self.P = np.zeros([self.no_action_profiles, self.no_action_profiles])
        
        for k in range(self.no_actions):
            for j in range(self.no_actions):
                
                utilities = np.array([self.utility_functions[0](i, j) for i in range(self.no_actions)])
                exp_values = np.exp(beta * utilities)
        
                p = exp_values/np.sum(exp_values)
                
                self.P[k*self.no_actions+j, j*self.no_actions:(j+1)*self.no_actions] += 1/self.no_players*p
        
        for j in range(self.no_actions):
            for k in range(self.no_actions):
                
                utilities = np.array([self.utility_functions[1](i, k) for i in range(self.no_actions)])
                exp_values = np.exp(beta * utilities)
        
                p = exp_values/np.sum(exp_values)
                
                self.P[k*self.no_actions+j, k::self.no_actions] += 1/self.no_players*p
        
        return self.P
           
    def generate_payoff_matrix(self):
        
        self.payoff_player_1 = np.random.uniform(0.0, 1 - self.delta, size = (self.no_actions, self.no_actions))
        
        self.payoff_player_1[self.firstNE[0], self.firstNE[1]] = 1
        self.payoff_player_1[self.secondNE[0], self.secondNE[1]] = 1 - self.delta
        
        if self.type == "Asymmetrical": 
            self.payoff_player_2 = np.transpose(self.payoff_player_1)
        else: 
            self.payoff_player_1 = (self.payoff_player_1 + np.transpose(self.payoff_player_1)) / 2
            self.payoff_player_2 = self.payoff_player_1
    
    def reset_payoff_matrix(self, delta = None):
        
        if delta:
            self.delta = delta
            
        self.generate_payoff_matrix()
     
    def set_payoff_matrix(self, payoff):
        
        self.payoff_player_1 = payoff
        
        if self.type == "Asymmetrical":
            self.payoff_player_2 = np.transpose(payoff)  
        else:
            self.payoff_player_1 = (self.payoff_player_1 + np.transpose(self.payoff_player_1)) / 2
            self.payoff_player_2 = self.payoff_player_1 
        
    def utility_function_player_1(self, player_action, opponents_action):

        return self.payoff_player_1[player_action, opponents_action]

    def utility_function_player_2(self, player_action, opponents_action):
        
        return self.payoff_player_2[player_action, opponents_action]

    def potential_function(self, action_profile):
        
        return self.payoff_player_1[action_profile[0], action_profile[1]]
    
    def formulate_potential_vec(self):
        
        self.potential = np.zeros([self.no_action_profiles, 1])
        
        for i in range(self.no_actions):
            for j in range(self.no_actions):
                self.potential[i * self.no_actions + j, 0] = self.potential_function(np.array([i, j]))