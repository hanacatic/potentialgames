import numpy as np
from lib.player import Player
from lib.aux_functions.helpers import *
from scipy.sparse import csr_matrix, csc_array

rng = np.random.default_rng()

class Game:
    
    def __init__(self, gameSetup, algorithm = 'log_linear',  max_iter = 200000, mu = None): # mu - initial distribution
        
        self.gameSetup = gameSetup # all the game rules and game data
        self.algorithm = algorithm
        self.max_iter = int(max_iter)
        
        self.players = np.array([ Player(i, self.gameSetup.no_actions, gameSetup.utility_functions[i]) for i in range(0, self.gameSetup.no_players)], dtype = object)
        
        self.action_profile = np.random.randint(0, self.gameSetup.no_actions, self.gameSetup.no_players) # discrete uniform distribution
        self.action_profile = self.sample_initial_action_profile(mu)
           
        self.potentials_history = np.zeros((self.max_iter, 1))
        self.expected_value = None
        
        self.opponents_idx_map = [ np.delete(np.arange(self.gameSetup.no_players), player_id) for player_id in range(self.gameSetup.no_players) ]
               
    def sample_initial_action_profile(self, mu):
        
        self.initial_action_profile = rejection_sampling(mu, self.action_profile, self.gameSetup.no_actions)
        
        return self.initial_action_profile
    
    def set_initial_action_profile(self, initial_action_profile):
        
        self.initial_action_profile = initial_action_profile
    
    def set_mu_matrix(self, mu_matrix):
        
        self.mu_matrix = mu_matrix

    def play(self, initial_action_profile = None, beta = None, scale_factor = 1):
        
        if initial_action_profile == None:
            self.action_profile = self.initial_action_profile.copy()
        else:
            self.action_profile = initial_action_profile
        
        match self.algorithm:
            case "log_linear":
                print(beta)
                self.log_linear(beta)
            case "log_linear_t":
                self.log_linear_t(beta)
            case "log_linear_tatarenko":
                self.log_linear_tatarenko()
            case "log_linear_fast":
                # self.set_mu_matrix(mu_matrix)
                if self.gameSetup.no_players == 2:
                    self.log_linear_fast(beta, scale_factor)
                else:
                    self.log_linear_fast_sparse(beta, scale_factor)
            case "best_response":
                self.best_response()
            case "alpha_best_response":
                self.alpha_best_response()
            case "multiplicative_weight":
                self.multiplicative_weight()
            case _:
                self.log_linear(beta)
        
    def log_linear(self, beta):
               
        for i in range(0, self.max_iter): 
            
            self.log_linear_iteration(i, beta)
    
    def log_linear_t(self, beta_t):

        for i in range(self.max_iter): 
            
            beta = beta_t*(1/self.gameSetup.no_players *np.log(1+i)/(1+ 1/self.gameSetup.no_players * np.log(i+1)))
            
            self.log_linear_iteration(i, beta)
            
    def log_linear_tatarenko(self):

        for i in range(self.max_iter): 

            # beta = np.log(i+1)*(self.gameSetup.no_players**2+1)/self.gameSetup.no_players
                        
            beta = min((i+1), 500) # Tatarenko

  
            self.log_linear_iteration(i, beta)
    
    def log_linear_fast(self, beta, scale_factor):
        
        P = self.gameSetup.formulate_transition_matrix(beta)
        mu0 = self.mu_matrix.copy()
        
        self.expected_value = np.zeros((int(self.max_iter), 1))
        
        P = np.linalg.matrix_power(P, scale_factor)
        
        for i in range(self.max_iter):
            
            mu = mu0 @ P
            
            mu0 = mu
            
            self.expected_value[i] = mu @ self.gameSetup.potential
        
        self.expected_value = self.expected_value
        self.stationary = mu
                
    def log_linear_fast_sparse(self, beta, scale_factor):
        
        P = self.gameSetup.formulate_transition_matrix_sparse(beta)
        mu0 = csc_array(self.mu_matrix)
        
        self.expected_value = np.zeros((int(self.max_iter), 1))
        self.expected_value = csr_matrix(self.expected_value)

        for i in range(self.max_iter):
            
            mu = mu0 @ P
                        
            mu0 = mu
            
            self.expected_value[i] = mu @ self.gameSetup.potential
        
        self.expected_value = self.expected_value.todense()
        self.stationary = mu.todense()
                           
    def log_linear_iteration(self, i, beta):
        
        player_id = rng.integers(0, len(self.players), 1) # randomly choose a player
            
        player = self.players[player_id][0] 
        
        opponents_actions = self.action_profile[self.opponents_idx_map[player_id[0]]] # extract the opponents actions from the action profile
            
        self.action_profile[player_id] = player.update_log_linear(beta, opponents_actions) # update the players action
            
        self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
         
    def best_response(self):
        
        for i in range(self.max_iter):
            
            player_id = rng.integers(0, len(self.players), 1) # randomly choose a player
            
            player = self.players[player_id][0] 
            
            opponents_actions = self.action_profile[self.opponents_idx_map[player_id[0]]] # extract the opponents actions from the action profile
    
            self.action_profile[player_id] = player.best_response(opponents_actions) # update the players action

            self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function

    def alpha_best_response(self):
        
        improvement = True
        for i in range(self.max_iter):
            
            chosen_player = 0
            chosen_player_action = self.action_profile[chosen_player]
            best_improvement = 0
            
            if improvement == False:
                self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile)
                continue
            else:
                improvement = False
                
            for player_id in range(0, self.gameSetup.no_players):
                          
                player = self.players[player_id]
            
                opponents_actions = self.action_profile[self.opponents_idx_map[player_id]] # extract the opponents actions from the action profile

                current_payoff = player.utility(self.action_profile[player_id], opponents_actions)
                best_action = player.best_response(opponents_actions) # update the players action
                
                best_payoff  = player.utility(best_action, opponents_actions)

                if best_payoff - current_payoff > best_improvement:
                    
                    chosen_player = player_id
                    chosen_player_action = best_action
                    best_improvement = best_payoff - current_payoff
                    
                    improvement = True
                         
            self.action_profile[chosen_player] = chosen_player_action 

            self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
    
    def multiplicative_weight(self):
        
        gamma_t = np.sqrt(np.log(self.gameSetup.no_players)/self.max_iter) 
        mixed_strategies = np.zeros([self.gameSetup.no_players, self.gameSetup.no_actions])
                      
        for i in range(self.max_iter):            
                        
            for player_id in range(self.gameSetup.no_players):
                
                player = self.players[player_id]
                
                mixed_strategies[player_id] = player.mixed_strategy()
                
                self.action_profile[player_id] = rng.choice(self.gameSetup.action_space, 1, p = mixed_strategies[player_id])
            
            self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function

            for player_id in range(self.gameSetup.no_players):
                
                opponents_actions = self.action_profile[self.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
                
                player = self.players[player_id]
                
                player.update_mw(opponents_actions, gamma_t = gamma_t)
                                      
    def compute_beta(self, epsilon):
        
        A = self.gameSetup.no_actions
        N = self.gameSetup.no_players
        delta = self.gameSetup.delta
        
        # return 1/max(epsilon/2, delta)*np.log(A**N*(1-epsilon/2)*(4/(epsilon*A**N*(epsilon/2)) - 1/(A**N*(epsilon/2))))
        # if self.gameSetup.type == "Asymmetrical":
        #     return 1/max(epsilon, delta)*np.log(A**N/epsilon)
        # print("Symmetrical in compute beta")
        return  1/max(epsilon, delta)*np.log(N**A/epsilon)

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
    
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        
    def reset_game(self):

        [self.players[i].reset_player(self.gameSetup.no_actions, self.gameSetup.utility_functions[i]) for i in range(0, self.gameSetup.no_players)]

