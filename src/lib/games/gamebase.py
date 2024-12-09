import numpy as np
from lib.player import Player
from lib.games.trafficrouting import CongestionGame
from lib.aux_functions.helpers import *
from scipy.sparse import csr_matrix, csc_array

rng = np.random.default_rng()

class Game:
    
    def __init__(self, gameSetup, algorithm = 'log_linear',  max_iter = 200000, mu = None): # mu - initial distribution
        
        self.gameSetup = gameSetup # all the game rules and game data
        self.algorithm = algorithm
        self.max_iter = int(max_iter)
        
        self.players = np.array([ Player(i, self.gameSetup.action_space[i], gameSetup.utility_functions[i]) for i in range(0, self.gameSetup.no_players)], dtype = object)
        self.action_space = [np.arange(len(self.gameSetup.action_space[player_id])) for player_id in range(self.gameSetup.no_players)]

        self.action_profile = [0] * self.gameSetup.no_players # np.random.randint(0, self.gameSetup.no_actions, self.gameSetup.no_players) # discrete uniform distribution
        self.action_profile = self.sample_initial_action_profile(mu)
           
        self.potentials_history = np.zeros((self.max_iter, 1))
        self.expected_value = None
        
        if isinstance(self.gameSetup, CongestionGame):
            self.objectives_history = np.zeros((self.max_iter, 1))
            for i in range(self.gameSetup.no_players):
                self.players[i].min_payoff = self.gameSetup.min_travel_times[i]
                self.players[i].max_payoff = self.gameSetup.max_travel_times[i]
        
        self.opponents_idx_map = [ np.delete(np.arange(self.gameSetup.no_players), player_id) for player_id in range(self.gameSetup.no_players) ]
        self.player_idx_map = np.arange(0, self.gameSetup.no_players)  
                    
    def sample_initial_action_profile(self, mu):
        
        self.initial_action_profile = rejection_sampling(mu, self.action_profile, self.action_space, M = 0.5, iterations = 1000)
        
        return self.initial_action_profile
    
    def set_initial_action_profile(self, initial_action_profile):
        
        self.initial_action_profile = initial_action_profile
    
    def set_mu_matrix(self, mu_matrix):
        
        self.mu_matrix = mu_matrix

    def play(self, initial_action_profile = None, beta = None, scale_factor = 1):
        
        if initial_action_profile is None:
            self.action_profile = self.initial_action_profile.copy()
        else:
            self.initial_action_profile = initial_action_profile.copy()
            self.action_profile = initial_action_profile.copy()
        
        for player_id in range(self.gameSetup.no_players):
            
            player = self.players[player_id]
            player.past_action = self.action_profile[player_id].copy()            
        
        if self.algorithm == "log_linear":
            self.log_linear(beta)
        elif self.algorithm == "log_linear_t":
            print(beta)
            self.log_linear_t(beta)
        elif self.algorithm == "log_linear_tatarenko":
                self.log_linear_tatarenko()
        elif self.algorithm == "log_linear_fast":
            # self.set_mu_matrix(mu_matrix)
            if self.gameSetup.no_players == 2:
                self.log_linear_fast(beta, scale_factor)
            else:
                self.log_linear_fast_sparse(beta, scale_factor)
        elif self.algorithm == "log_linear_binary":
            self.log_linear_binary(beta)
        elif self.algorithm == "modified_log_linear":
            for i in self.player_idx_map:
                self.players[i].set_modified_utility(self.gameSetup.modified_utility_functions[i])
            self.modified_log_linear(beta = beta)
        elif self.algorithm == "exponential_weight_annealing":
            self.exponential_weight_annealing()
        elif self.algorithm == "best_response":
            self.best_response()
        elif self.algorithm == "alpha_best_response":
            self.alpha_best_response()
        elif self.algorithm == "multiplicative_weight":
            self.multiplicative_weight()
        
    def log_linear(self, beta):
               
        print("Log linear learning")
        for i in range(0, self.max_iter): 
            
            if i % 100 == 0:
                print(str(i) + "th iteration")
            
            self.log_linear_iteration(i, beta)
    
    def log_linear_t(self, beta_t):

        for i in range(self.max_iter): 
            
            # beta = beta_t*(1/self.gameSetup.no_players *np.log(1+i)/(1 + 1/self.gameSetup.no_players * np.log(i+1)))
            beta = beta_t*(np.log(1/self.gameSetup.no_players)*np.log(1+i)/(1 + np.log(1/self.gameSetup.no_players)* np.log(i+1)))

            self.log_linear_iteration(i, beta)
            
    def log_linear_tatarenko(self):

        for i in range(self.max_iter): 

            # beta = np.log(i+1)*(self.gameSetup.no_players**2+1)/self.gameSetup.no_players
                        
            beta = min((i+1), 5000) # Tatarenko
  
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
        
        if isinstance(self.gameSetup, CongestionGame):
            self.objectives_history[i] = self.gameSetup.objective(self.action_profile)
    
    def log_linear_binary(self, beta):
        
        print("Log linear binary")
        
        for i in range(self.max_iter):
            
            if i % 100 == 0:
                print(str(i) + "th iteration")
            
            self.log_linear_binary_iteration(i, beta)
    
    def log_linear_binary_iteration(self, i, beta):
        
        player_id = rng.integers(0, len(self.players), 1) # randomly choose a player
            
        player = self.players[player_id][0] 
        
        opponents_actions = self.action_profile[self.opponents_idx_map[player_id[0]]] # extract the opponents actions from the action profile
            
        self.action_profile[player_id] = player.update_log_linear_binary(beta, opponents_actions) # update the players action
            
        self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
        
        if isinstance(self.gameSetup, CongestionGame):
            self.objectives_history[i] = self.gameSetup.objective(self.action_profile)
              
    def modified_log_linear(self, beta, alpha = 0.2):
        
        self.time_played = np.zeros(self.gameSetup.no_players)
        self.phi = np.zeros(self.gameSetup.no_actions)
        
        for a in self.action_profile:
            self.phi[a] += 1
            
        for i in range(self.max_iter):
            time = i
            self.modified_log_linear_iteration(time, beta, alpha)
    
    def modified_log_linear_iteration(self, i, beta, alpha):
        
        player_clock = [1/self.gameSetup.no_players*self.phi[self.action_profile[i]] for i in range(self.gameSetup.no_players)]
        
        # print("phi: ")
        # print(self.phi)
        # print("player clock: ")
        # print(player_clock)
        
        player_clock = player_clock/np.sum(player_clock)
        
        # print(player_clock)
        
        player_id = rng.choice(self.player_idx_map, size=1, p=player_clock)
        
        player = self.players[player_id][0] 
        
        self.phi[self.action_profile[player_id]] -= 1
        
        self.action_profile[player_id] = player.update_modified_log_linear(beta, self.phi)
        
        self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
        
        self.phi[self.action_profile[player_id]] += 1 
        # for player_id in range(self.gameSetup.no_players):
        #     player_clock = alpha * self.phi[self.action_profile[player_id]]
            
        #     if time - self.time_played[player_id] < player_clock:
        #         continue
            
        #     self.players[player_id]
            
        #     self.time_played[player_id] = time
              
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
            
            if i % 20 == 0:
                print(str(i) + "th iteration")
                        
            chosen_player = 0
            chosen_player_action = self.action_profile[chosen_player]
            best_improvement = 0
            
            if improvement == False:
                self.potentials_history[i] = self.potentials_history[i-1]
                
                if isinstance(self.gameSetup, CongestionGame):
                    self.objectives_history[i] = self.objectives_history[i-1]
                     
                continue

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

            if isinstance(self.gameSetup, CongestionGame):
                self.objectives_history[i] = self.gameSetup.objective(self.action_profile)      
    
    def multiplicative_weight(self):
        
        print("Multiplicative weight update")
        
        gamma_t = np.sqrt(8*np.log(self.gameSetup.no_actions)/self.max_iter) 
        mixed_strategies = np.zeros([self.gameSetup.no_players, self.gameSetup.no_actions])
                      
        for i in range(self.max_iter):            
            
            if i % 20 == 0:
                print(str(i) + "th iteration")
                        
            for player_id in range(self.gameSetup.no_players):
                
                player = self.players[player_id]
                
                mixed_strategies[player_id] = player.mixed_strategy()
                
                self.action_profile[player_id] = rng.choice(self.action_space[player_id], 1, p = mixed_strategies[player_id])
            
            self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
            
            if isinstance(self.gameSetup, CongestionGame):
                self.objectives_history[i] = self.gameSetup.objective(self.action_profile)
                
            for player_id in range(self.gameSetup.no_players):
                
                opponents_actions = self.action_profile[self.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
                
                player = self.players[player_id]
                
                player.update_mw(opponents_actions, gamma_t = gamma_t)                    
    
    def exponential_weight_annealing(self, b = 0.6, a = 0.5, p = 1): #a = 0.5
        
        print("Exponential weight with annealing")
        
        gamma_n = 1
        eps_n = 1 
        mixed_strategies = np.zeros([self.gameSetup.no_players, self.gameSetup.no_actions])
        
        for i in range(self.max_iter):
            
            # gamma_n = 1/np.log(self.gameSetup.no_actions)/np.log(i+2)**b
            gamma_n = np.sqrt(100*np.log(self.gameSetup.no_actions)/(i+2)**(2*b)) 

            eps_n = 1/(np.log(self.gameSetup.no_actions)**a*(i+2)**a*np.log(i+2)**p)

            if i % 20 == 0:
                print(str(i) + "th iteration")
                print(self.players[0].scores)
                print(self.players[0].prob)
                        
            for player_id in range(self.gameSetup.no_players):
                
                player = self.players[player_id]
                
                mixed_strategies[player_id] = player.mixed_strategy()
                
                self.action_profile[player_id] = rng.choice(self.action_space[player_id], 1, p = mixed_strategies[player_id])
            
  
            for player_id in range(self.gameSetup.no_players):
                
                opponents_actions = self.action_profile[self.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
                action = self.action_profile[player_id]
                
                player = self.players[player_id]
                
                player.update_ewa(action, opponents_actions, gamma_n = gamma_n, eps_n = eps_n)
            
            self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
            
            if isinstance(self.gameSetup, CongestionGame):
                self.objectives_history[i] = self.gameSetup.objective(self.action_profile)
                
    def compute_beta(self, epsilon):
        
        A = self.gameSetup.no_actions
        N = self.gameSetup.no_players
        delta = self.gameSetup.delta
        
        # return 1/max(epsilon/2, delta)*np.log(A**N*(1-epsilon/2)*(4/(epsilon*A**N*(epsilon/2)) - 1/(A**N*(epsilon/2))))
        # if self.gameSetup.type == "Asymmetrical":
        # return 1/max(epsilon, delta)*np.log(A**N/epsilon)

        return 1/max(epsilon, delta)*(N*np.log(A) - np.log(epsilon))
        # print("Symmetrical in compute beta")
        # return  1/max(epsilon, delta)*np.log(N**A/epsilon)

    def compute_t(self, epsilon):
        
        A = self.gameSetup.no_actions
        N = self.gameSetup.no_players
        delta = self.gameSetup.delta
        beta = self.compute_beta(epsilon)
        
        # return 25*N**2*A**5*np.exp(4*beta)/16/np.pi**2*(np.log(np.log(A**N)) + np.log(beta) + 2*np.log(4/epsilon))
        
        # return N**2*A**5*(A**N/epsilon)**(1/max(epsilon, delta))
        
        return np.log(N**2*A**5) + (1/max(epsilon, delta))*N*np.log(A/epsilon)

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
