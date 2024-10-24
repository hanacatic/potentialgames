import numpy as np
from itertools import product
from player import Player
from aux_functions.helpers import rejection_sampling

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
                self.log_linear_fast(beta, scale_factor)
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
        print(beta_t)
        for i in range(self.max_iter): 
            
            beta = beta_t*(1/self.gameSetup.no_players *np.log(1+i)/(1+ 1/self.gameSetup.no_players * np.log(i+1)))
            # beta = beta_t*(np.log(1+i)/(1 + np.log(i+1)))

            
            self.log_linear_iteration(i, beta)
            
    def log_linear_tatarenko(self):

        for i in range(self.max_iter): 

            # beta = np.log(i+1)*(self.gameSetup.no_players**2+1)/self.gameSetup.no_players
            
            # beta = np.log(i+1)
            
            beta = min((i+1), 500) # Tatarenko

  
            self.log_linear_iteration(i, beta)
            
    def log_linear_fast(self, beta, scale_factor):
        
        P = self.gameSetup.formulate_transition_matrix(beta)
        self.gameSetup.formulate_potential_vec()
        mu0 = self.mu_matrix.copy()
        
        self.expected_value = np.zeros((int(self.max_iter), 1))
        
        P = np.linalg.matrix_power(P, scale_factor)
        
        # P = np.linalg.matrix_power(P, 10)
        
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
            
                mask = np.arange(len(self.action_profile)) != player_id
                opponents_actions = self.action_profile[mask] # extract the opponents actions from the action profile
                
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
    
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        
    def reset_game(self):
        
        # self.gameSetup = gameSetup
        [self.players[i].reset_player(self.gameSetup.no_actions, self.gameSetup.utility_functions[i]) for i in range(0, self.gameSetup.no_players)]

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
        
        if payoff_matrix is None:
            self.generate_payoff_matrix()
        else:
            self.set_payoff_matrix(payoff_matrix)

        self.utility_functions = []
        
        for i in range(0, self.no_players):
            self.utility_functions.append(lambda player_action, opponents_action: self.utility_function(i, player_action, opponents_action))
        
    def formulate_transition_matrix(self, beta):
    
        self.P = np.zeros([self.no_action_profiles, self.no_action_profiles])     
        
        # # # for j in range(self.no_actions):
        # # #     for k in range(self.no_actions):
                
        # # #         utilities = np.array([self.utility_functions[0](i, k) for i in range(self.no_actions)])
        # # #         exp_values = np.exp(beta * utilities)
        
        # # #         p = exp_values/np.sum(exp_values)
                
        # # #         self.P[j*self.no_actions+k, k::self.no_actions] += 1/self.no_players*p

        # # for j in range(self.no_actions):
        # #     for k in range(self.no_actions):
                
        # #         utilities = np.array([self.utility_functions[1](i, j) for i in range(self.no_actions)])
        # #         exp_values = np.exp(beta * utilities)
        
        # #         p = exp_values/np.sum(exp_values)
                
        # #         self.P[j*self.no_actions+k, j*self.no_actions:(j+1)*self.no_actions] += 1/self.no_players*p
        
        # P = np.zeros([self.no_action_profiles, self.no_action_profiles])     

        # for player_id in range(self.no_players):
        #     for action_id in range(self.no_actions):
        #         for idx, opponents_actions in enumerate(list(product(np.arange(self.no_actions), repeat = self.no_players - 1))):
        #             utilities = np.array([self.utility_functions[player_id](i, opponents_actions) for i in range(self.no_actions)])
        #             exp_values = np.exp(beta * utilities)
        
        #             p = exp_values/np.sum(exp_values)
        #             print("player_id: ")
        #             print(player_id)
        #             print("action_id:")
        #             print(action_id)
        #             print("action_profile_id: ")
        #             print(idx)
        #             print("start: ")
        #             print(idx*self.no_actions**player_id)
        #             print("stop: ")
        #             print((idx*self.no_actions**player_id + self.no_actions**(self.no_players - 1 - player_id)*(self.no_actions)))
        #             print("step: ")
        #             print(self.no_actions**(self.no_players - 1 - player_id))
        #             print(np.arange(idx*self.no_actions**player_id,(idx*self.no_actions**player_id + self.no_actions**(self.no_players - 1 - player_id)*(self.no_actions)),self.no_actions**(self.no_players - 1 - player_id)))
        #             print("row: ")
        #             print(idx*self.no_actions**player_id + action_id*self.no_actions**(self.no_players - 1 - player_id))
        #             # P[action_id*self.no_actions**(self.no_players - player_id) + idx, idx*(self.no_actions**player_id):((idx + self.no_actions**(self.no_players - 1 - player_id))*(self.no_actions**player_id)):self.no_actions**(self.no_players - 1 - player_id)] += 1/self.no_players*p
        #             # P[action_id*self.no_actions**(self.no_players - 1) + idx, idx*self.no_actions**player_id:(idx*self.no_actions**player_id + self.no_actions**(self.no_players - 1 - player_id)*(self.no_actions)):self.no_actions**(self.no_players - 1 - player_id)] += 1/self.no_players*p
        #             P[idx*self.no_actions**player_id + action_id*self.no_actions**(self.no_players - 1 - player_id), idx*self.no_actions**player_id:(idx*self.no_actions**player_id + self.no_actions**(self.no_players - 1 - player_id)*(self.no_actions)):self.no_actions**(self.no_players - 1 - player_id)] += 1/self.no_players*p
            
        # self.P = P
        
        P = np.zeros([self.no_action_profiles, self.no_action_profiles])     

        for idx, profile in enumerate(np.array(list(product(np.arange(self.no_actions), repeat = self.no_players)))):
            for player_id in range(self.no_players):
                
                mask = np.arange(len(profile)) != player_id
                opponents_actions = profile[mask] # extract the opponents actions from the action profile
                
                utilities = np.array([self.utility_functions[player_id](i, opponents_actions) for i in range(self.no_actions)])
                exp_values = np.exp(beta * utilities)
        
                p = exp_values/np.sum(exp_values)
                
                i = idx - profile[player_id]*self.no_actions**(self.no_players - 1 - player_id)
                P[idx, i: i + self.no_actions**(self.no_players - player_id) :self.no_actions**(self.no_players - 1 - player_id)] += 1/self.no_players*p
        self.P = P
        return self.P
           
    def generate_payoff_matrix(self):
        
        self.payoff_player_1 = np.random.uniform(0.0, 1 - self.delta, size = [self.no_actions] * self.no_players)
        
        
        self.payoff_player_1[self.firstNE[0], self.firstNE[1]] = 1
        self.payoff_player_1[self.secondNE[0], self.secondNE[1]] = 1 - self.delta
        
        if self.type == "Symmetrical": 
            self.payoff_player_1 = (self.payoff_player_1 + np.transpose(self.payoff_player_1)) / 2
    
    def reset_payoff_matrix(self, delta = None):
        
        if delta:
            self.delta = delta
            
        self.generate_payoff_matrix()
     
    def set_payoff_matrix(self, payoff):
        
        self.payoff_player_1 = payoff
        
        if self.type == "Symmetrical":
            self.payoff_player_1 = (self.payoff_player_1 + np.transpose(self.payoff_player_1)) / 2

    def potential_function(self, action_profile):
        
        return self.payoff_player_1[tuple(action_profile)]
    
    def utility_function(self, player_id, player_action, opponents_action):
        
        action_profile = np.insert(opponents_action, player_id, player_action)
        
        return self.potential_function(action_profile)
        
    def formulate_potential_vec(self):
        
        self.potential = np.zeros([self.no_action_profiles, 1])
        
        for idx, element in enumerate(product(np.arange(self.no_actions), repeat = self.no_players)):
            self.potential[idx] = self.potential_function(element)
