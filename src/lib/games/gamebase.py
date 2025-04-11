import numpy as np
from lib.player import Player
from lib.games.trafficrouting import CongestionGame
from lib.aux_functions.helpers import *
from scipy.sparse import csr_matrix, csc_array

rng = np.random.default_rng(seed = 2)

class Game:
    """
        Class representing the Game base.
        
        Includes information on the game setup and the algorithms.
    """
    
    def __init__(self, gameSetup, algorithm = 'log_linear',  max_iter = 200000, mu = None): # mu - initial distribution
        """_summary_

        Args:
            gameSetup (IdentInterest/TrafficRouting): Game setup, defines the type of game, the properties of the game, including the number of players and the number of actions
            algorithm (str, optional): Type of algorithm to solve the game. Defaults to 'log_linear'.
            max_iter (int, optional): Maximum number of iterations available to the algorithm. Defaults to 200000.
            mu (function, optional): Initial joint action profile distribution. Defaults to None.
        """
        
        self.gameSetup = gameSetup # all the game rules and game data
        self.algorithm = algorithm
        self.max_iter = int(max_iter)
        
        # define the players
        self.players = np.array([ Player(i, self.gameSetup.action_space[i], gameSetup.utility_functions[i], gameSetup.noisy_utility) for i in range(0, self.gameSetup.no_players)], dtype = object)
        # define the action space
        self.action_space = [np.arange(len(self.gameSetup.action_space[player_id])) for player_id in range(self.gameSetup.no_players)]

        # initial joint action profile
        self.action_profile = [0] * self.gameSetup.no_players
        
        if mu is not None: 
            self.action_profile = self.sample_initial_action_profile(mu)
        
        # initialise properties required by the algorithms
        self.expected_value = None
        # maps for caching data reused by the algorithm for speed up
        self.opponents_idx_map = [ np.delete(np.arange(self.gameSetup.no_players), player_id) for player_id in range(self.gameSetup.no_players) ]
        self.player_idx_map = np.arange(0, self.gameSetup.no_players) 
        
        self.potentials_history = np.zeros((self.max_iter, 1)) 
        
        if isinstance(self.gameSetup, CongestionGame):
            self.objectives_history = np.zeros((self.max_iter, 1))
            for i in range(self.gameSetup.no_players):
                self.players[i].min_payoff = self.gameSetup.min_travel_times[i]
                self.players[i].max_payoff = self.gameSetup.max_travel_times[i]                            

    def sample_initial_action_profile(self, mu):
        """
            Samples a joint action profile from the given distribution.

        Args:
            mu (function): Joint action profile distribution.

        Returns:
            np.array(N): Joint action profile.
        """
        
        # TODO test rejection sampling in multiplayer setting
        self.initial_action_profile = rejection_sampling(mu, self.action_profile, self.action_space, M = 0.5, iterations = 1000)
        
        return self.initial_action_profile
    
    def set_initial_action_profile(self, initial_action_profile):
        """
            Sets the initial joint action profile to the given joint action profile.

        Args:
            initial_action_profile (np.array(N)): Joint action profile.
        """
        
        self.initial_action_profile = initial_action_profile
    
    def set_mu_matrix(self, mu_matrix):
        """
            Sets the initial joint action profile matrix distribution to the given distribution.

        Args:
            mu_matrix (np.array(AxA....A (Nd))): Initial joint action profile matrix distribution.
        """
        
        self.mu_matrix = mu_matrix

    def play(self, initial_action_profile = None, beta = None, scale_factor = 1, gamma = 0):
        """
            Base function calls the required algorithm.

        Args:
            initial_action_profile (np.array(N), optional): Initial joint action profile. Defaults to None.
            beta (double, optional): Player rationality. Defaults to None.
            scale_factor (int, optional): Scale factor. Defaults to 1.
            gamma (int, optional): Exploration factor. Defaults to 0.
        """
        
        # Set initial joint action profile
        if initial_action_profile is None:
            self.action_profile = self.initial_action_profile.copy()
        else:
            self.initial_action_profile = initial_action_profile.copy()
            self.action_profile = initial_action_profile.copy()
        
        # Reset players (gameBase function play can have successive calls)
        for player_id in range(self.gameSetup.no_players):
            
            player = self.players[player_id]
            player.past_action = self.action_profile[player_id].copy()            
        
        # Choose algorithm
        if self.algorithm == "log_linear":
            self.log_linear(beta, gamma)
        elif self.algorithm == "log_linear_t":
            print(beta)
            self.log_linear_t(beta)
        elif self.algorithm == "log_linear_tatarenko":
            self.log_linear_tatarenko()
        elif self.algorithm == "log_linear_fast":
            if self.gameSetup.no_players == 2:
                self.log_linear_fast(beta, scale_factor)
            else:
                self.log_linear_fast_sparse(beta, scale_factor)
        elif self.algorithm == "log_linear_binary":
            self.log_linear_binary(beta)
        elif self.algorithm == "log_linear_binary_fast":
            self.log_linear_binary_fast(beta, scale_factor)
        elif self.algorithm == "modified_log_linear":
            for i in self.player_idx_map:
                self.players[i].set_modified_utility(self.gameSetup.modified_utility_functions[i])
            self.modified_log_linear(beta = beta)
        elif self.algorithm == "exponential_weight_annealing":
            self.exponential_weight_annealing()
        elif self.algorithm == "exp3p":
            self.exp3p()
        elif self.algorithm == "best_response":
            self.best_response()
        elif self.algorithm == "alpha_best_response":
            self.alpha_best_response()
        elif self.algorithm == "multiplicative_weight":
            self.multiplicative_weight()
        
    def log_linear(self, beta, gamma = 0):
        """
            Log-linear learning algorithm.

        Args:
            beta (double): Player rationality.
            gamma (int, optional): Exploration factor. Defaults to 0.

        Raises:
            Exception: Missing arguments.
        """
        
        if beta is None:
            raise Exception("Sorry, you have not provided rationality!")

        print("Log linear learning")
        
        # Noisy utility 
        if self.gameSetup.noisy_utility and self.gameSetup.eta is None:
            self.gameSetup.eta = 1/2.0/beta
                    
        print(self.gameSetup.eta)
        
        for i in range(0, self.max_iter): 
            
            if i % 50000 == 0:
                print(str(i) + "th iteration")
            
            self.log_linear_iteration(i, beta, gamma)
    
    def log_linear_t(self, beta_t):
        """
            Log-linear learning with time-varying rationality as proposed in the project.

        Args:
            beta_t (double): Player rationality.
        """
        
        # Noisy utility 
        if self.gameSetup.noisy_utility and self.gameSetup.eta is None:
                self.gameSetup.eta = 1/2.0/beta

        for i in range(self.max_iter): 
            
            # Proposed time-varying rationality
            beta = beta_t*(1/self.gameSetup.no_actions *np.log(i + self.gameSetup.no_actions)/(1 + 1/self.gameSetup.no_actions * np.log(i+self.gameSetup.no_actions)))

            self.log_linear_iteration(i, beta)
            
    def log_linear_tatarenko(self):
        """
            Log-linear learning with time-varyin rationality as proposed in T. Tatarenko, Game-theoretic learning and distributed optimization in memoryless multi-agent systems.
        """

        for i in range(self.max_iter): 

            beta = np.log(i+1)*(self.gameSetup.no_players**2+1)/self.gameSetup.no_players #  T. Tatarenko, Game-theoretic learning and distributed optimization in memoryless multi-agent systems.  (Theorem 3.5.2)
                        
            # beta = min((i+1), 5000) #  T. Tatarenko, Game-theoretic learning and distributed optimization in memoryless multi-agent systems. (Theorem 3.5.3)
  
            self.log_linear_iteration(i, beta, 0)
    
    def log_linear_fast(self, beta, scale_factor):
        """
            Log-linear learning utilising the Markov Chain approach.
            
        Args:
            beta (double): Player rationality.
            scale_factor (int): Scaling factor.
        """
        
        # Transition matrix of the Markov chain induced by log-linear learning for the game.
        P = self.gameSetup.formulate_transition_matrix(beta)
        mu0 = self.mu_matrix.copy()
        
        self.expected_value = np.zeros((int(self.max_iter), 1))
        
        # Transition matrix models scale_factor steps at once
        P = np.linalg.matrix_power(P, scale_factor)
        
        # Main loop
        for i in range(self.max_iter):
            
            mu = mu0 @ P
            mu0 = mu
            
            self.expected_value[i] = mu @ self.gameSetup.potential_vec
        
        self.expected_value = self.expected_value
        self.stationary = mu
                
    def log_linear_fast_sparse(self, beta, scale_factor):
        """_summary_

        Args:
            beta (double): Player rationality.
            scale_factor (int): Scale factor.
        """
        
        # Transition matrix of the Markov chain induced by log-linear learning for the game.
        P = self.gameSetup.formulate_transition_matrix_sparse(beta)
        mu0 = csc_array(self.mu_matrix)
        
        self.expected_value = np.zeros((int(self.max_iter), 1))
        self.expected_value = csr_matrix(self.expected_value)
        
        # Transition matrix modeling multiple transition steps at once has reduced sparsity and has negative impact on the computation time, thus it is not utilised.
        if scale_factor != 1:
            raise Exception("Sorry, for sparse implementation the scale factor must be 1!")

        # Main loop
        for i in range(self.max_iter):
            
            mu = mu0 @ P          
            mu0 = mu
            
            self.expected_value[i] = mu @ self.gameSetup.potential_vec

        self.expected_value = self.expected_value.todense()
        self.stationary = mu.todense()
                           
    def log_linear_iteration(self, i, beta, gamma = 0):
        """
            A single iteration of log-linear learning.
            
        Args:
            i (int): No. iteration.
            beta (double): Player rationality.
            gamma (int, optional): Exploration factor. Defaults to 0.
        """
        
        player_id = rng.integers(0, len(self.players), 1) # randomly choose a player
            
        player = self.players[player_id][0] 
        
        opponents_actions = self.action_profile[self.opponents_idx_map[player_id[0]]] # extract the opponents actions from the action profile
            
        self.action_profile[player_id] = player.update_log_linear(beta, opponents_actions, self.gameSetup.eta, gamma) # update the players action
            
        self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
        
        if isinstance(self.gameSetup, CongestionGame):
            self.objectives_history[i] = self.gameSetup.objective(self.action_profile)
    
    def log_linear_binary(self, beta):
        """
            Log-linear learning with two-point feedback.

        Args:
            beta (double): Player rationality.
        """
        
        print("Log linear binary")
        
        for i in range(self.max_iter):
            
            if i % 100 == 0:
                print(str(i) + "th iteration")
            
            self.log_linear_binary_iteration(i, beta)
    
    def log_linear_binary_iteration(self, i, beta):
        """
            A single iteration of log-linear learning with two-point feedback.

        Args:
            i (int): No. iteration.
            beta (double): Player rationality.
        """
        
        player_id = rng.integers(0, len(self.players), 1) # randomly choose a player
            
        player = self.players[player_id][0] 
        
        opponents_actions = self.action_profile[self.opponents_idx_map[player_id[0]]] # extract the opponents actions from the action profile
            
        self.action_profile[player_id] = player.update_log_linear_binary(beta, opponents_actions) # update the players action
            
        self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
        
        if isinstance(self.gameSetup, CongestionGame):
            self.objectives_history[i] = self.gameSetup.objective(self.action_profile)
            
    def log_linear_binary_fast(self, beta, scale_factor):
        """
            Binary log-linear learning utilising the Markov Chain approach.
            
        Args:
            beta (double): Player rationality.
            scale_factor (int): Scaling factor.
        """
        
        # Transition matrix of the Markov chain induced by log-linear learning for the game.
        P = self.gameSetup.formulate_transition_matrix_binary(beta)
        mu0 = self.mu_matrix.copy()
        
        self.expected_value = np.zeros((int(self.max_iter), 1))
        
        # Transition matrix models scale_factor steps at once
        P = np.linalg.matrix_power(P, scale_factor)
        
        # Main loop
        for i in range(self.max_iter):
            
            mu = mu0 @ P
            mu0 = mu
            
            self.expected_value[i] = mu @ self.gameSetup.potential_vec
        
        self.expected_value = self.expected_value
        self.stationary = mu
                
    def modified_log_linear(self, beta):
        """
            Modified log-linear learning leveraging symmetry of the game.

        Args:
            beta (double): Player rationality.
        """
        
        self.phi = np.zeros(self.gameSetup.no_actions) # histogram over the action space
        
        for a in self.action_profile:
            self.phi[a] += 1
        
        # Main loop
        for i in range(self.max_iter):
            time = i
            self.modified_log_linear_iteration(time, beta)
    
    def modified_log_linear_iteration(self, i, beta):
        """
            A single iteration of modified log-linear learning.

        Args:
            i (int): No. iteration.
            beta (double): Player rationality.
        """
        
        # determine the probability that the player is going to change change his action based on the action they played in the previous round
        # implemented based on section 3.1 in Dynamics in Congestion Games, D. Shah and J. Shin
        player_clock = [1/self.gameSetup.no_players*self.phi[self.action_profile[i]] for i in range(self.gameSetup.no_players)]
        player_clock = player_clock/np.sum(player_clock)
                
        player_id = rng.choice(self.player_idx_map, size=1, p=player_clock) # chose the next player from the player_clock probability distribution
        
        player = self.players[player_id][0] 
        
        self.phi[self.action_profile[player_id]] -= 1
        
        self.action_profile[player_id] = player.update_modified_log_linear(beta, self.phi) # update chosen players strategy
        
        self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
        
        self.phi[self.action_profile[player_id]] += 1 
  
    def best_response(self):
        """
            Iterative best response algorithm.
        """
        # Main loop
        for i in range(self.max_iter):
            
            player_id = rng.integers(0, len(self.players), 1) # randomly choose a player
            
            player = self.players[player_id][0] 
            
            opponents_actions = self.action_profile[self.opponents_idx_map[player_id[0]]] # extract the opponents actions from the action profile
    
            self.action_profile[player_id] = player.best_response(opponents_actions) # update the players action

            self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function

    def alpha_best_response(self):
        """
            Alpha best response algorithm with largest gain dynamics.
             S. Chien and A. Sinclair, ‘Convergence to approximate nash equilibria in congestion games’
        """
        
        improvement = True # keeping track if there is a possible improvement
        
        for player_id in range(self.gameSetup.no_players):
            player = self.players[player_id]
            player.reset()
        
        # Main loop
        for i in range(self.max_iter):
            
            if i % 20 == 0:
                print(str(i) + "th iteration")
                        
            chosen_player = 0
            chosen_player_action = self.action_profile[chosen_player]
            best_improvement = 0
            
            # For caching 
            if improvement == False:
                self.potentials_history[i] = self.potentials_history[i-1]
                
                if isinstance(self.gameSetup, CongestionGame):
                    self.objectives_history[i] = self.objectives_history[i-1]  
                continue

            improvement = False
            
            # Find the player with largest possible gain
            for player_id in range(0, self.gameSetup.no_players):
                          
                player = self.players[player_id]
            
                opponents_actions = self.action_profile[self.opponents_idx_map[player_id]] # extract the opponents actions from the action profile

                current_payoff = player.utility(self.action_profile[player_id], opponents_actions)
                
                best_action = player.best_response(opponents_actions) # update the players action
                best_payoff  = player.utility(best_action, opponents_actions) # compute the utility

                # Check if largest gain/best improvement
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
        """
            Multiplicative weight update Hedge implementation.
            Y. Freund and R. E. Schapire, ‘A decision-theoretic generalization of on-line learning and an application to boosting’
        """
        
        print("Multiplicative weight update")
        
        gamma_t = np.sqrt(8*np.log(self.gameSetup.no_actions)/self.max_iter) # exploration factor
         
        mixed_strategies = np.zeros([self.gameSetup.no_players, self.gameSetup.no_actions])
        
        for player_id in range(self.gameSetup.no_players):
            player = self.players[player_id]
            player.reset()
            
        past_exp_potential = 0   
        
        # Main loop         
        for i in range(self.max_iter):  
                      
            if i % 20 == 0:
                print(str(i) + "th iteration")
                        
            player_id = rng.integers(0, len(self.players), 1)[0] # randomly choose a player
            
            player = self.players[player_id] 
            
            mixed_strategies[player_id] = player.mixed_strategy() # obtain players mixed strategy
            
            self.action_profile[player_id] = rng.choice(self.action_space[player_id], 1, p = mixed_strategies[player_id])              
    
            opponents_actions = self.action_profile[self.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
            action = self.action_profile[player_id]
            
            player.update_mw(opponents_actions, gamma_t = gamma_t) # update players mixed strategy
            
            self.potentials_history[i] =  (past_exp_potential*i + self.gameSetup.potential_function(self.action_profile))/(i+1) # compute the value of the potential function

            past_exp_potential = self.potentials_history[i]
            
            if isinstance(self.gameSetup, CongestionGame):
                self.objectives_history[i] = self.gameSetup.objective(self.action_profile)
    
    def sample_from_mixed_strategy(self, mixed_strategies):
        """
            Samples an action for each player for its respective probability distribution.

        Args:
            mixed_strategies (np.array(NxA)): Array of mixed stategies for each player.
        Returns:
            Exception: If there is a NaN value in strategies.
        """
        
        for player_id in range(self.gameSetup.no_players):
                
            player = self.players[player_id]
            
            mixed_strategies[player_id] = player.mixed_strategy()
            
            if np.isnan(mixed_strategies[player_id]).any():
                return Exception("There seems to be a NaN value in the strategy of one of the players!")
                
            self.action_profile[player_id] = rng.choice(self.action_space[player_id], 1, p = mixed_strategies[player_id])              
    
    def exponential_weight_annealing(self, b = 0.6, a = 0.25, p = 0.5): 
        """
            Exponential weight with annealing algorithm.
            A. Heliou, J. Cohen and P. Mertikopoulos, ‘Learning with bandit feedback in potential games’        

        Args:
            b (float, optional): Parameter that defines the step size. Defaults to 0.6.
            a (float, optional): Parameter that defines the vanishing factor. Defaults to 0.25.
            p (float, optional): Parameter that defines the vanishing factor. Defaults to 0.5.
        """
        
        print("Exponential weight with annealing")
        
        gamma_n = 1
        eps_n = 1 
        mixed_strategies = np.zeros([self.gameSetup.no_players, self.gameSetup.no_actions])
                
        past_exp_potential = 0
        
        for player_id in range(self.gameSetup.no_players):
            player = self.players[player_id]
            player.reset()

        # Main loop
        for i in range(self.max_iter):
            
            gamma_n = 1/(i+1)**b  # Proposed step size sequence
            eps_n = 1/(1+(i+1)**a*np.log(i+2)**p)  # Proposed step vanishing factor

            if i % 500 == 0:
                print(str(i) + "th iteration")
            
            player_id = rng.integers(0, len(self.players), 1)[0] # randomly choose a player
            
            player = self.players[player_id] 
            
            mixed_strategies[player_id] = player.mixed_strategy() # obtain the players mixed strategy
            
            self.action_profile[player_id] = rng.choice(self.action_space[player_id], 1, p = mixed_strategies[player_id]) # sample action from the strategy             
    
            opponents_actions = self.action_profile[self.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
            action = self.action_profile[player_id]
              
            player.update_ewa(action, opponents_actions, gamma_n = gamma_n, eps_n = eps_n) # update the mixed strategy
            
            self.potentials_history[i] =  (past_exp_potential*i + self.gameSetup.potential_function(self.action_profile))/(i+1) #rho@self.gameSetup.potential #self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function

            past_exp_potential = self.potentials_history[i]

            if isinstance(self.gameSetup, CongestionGame):
                self.objectives_history[i] = self.gameSetup.objective(self.action_profile)
    
    def exp3p(self):
        """
            EXP3P.
            Algorithm  P. Auer, N. Cesa-Bianchi, Y. Freund and R. E. Schapire, ‘The nonstochastic multiarmed bandit problem’
            Parameters  S. Bubeck, N. Cesa-Bianchi et al., ‘Regret analysis of stochastic and nonstochastic multi-armed bandit problems’ (Yheorem 3.3)
        """
        
        print("EXP3P")  
          
        A = self.gameSetup.no_actions

        # Parameters
        beta = np.sqrt(np.log(A)/(self.max_iter*A))
        gamma = 1.05*np.sqrt(np.log(A)*A/self.max_iter)
        eta = 0.95*np.sqrt(np.log(A)/(self.max_iter*A))
        
        mixed_strategies = np.zeros([self.gameSetup.no_players, self.gameSetup.no_actions])

        for player_id in range(self.gameSetup.no_players):
            player = self.players[player_id]
            player.reset()
            
        past_exp_potential = 0
        
        # Main loop
        for i in range(self.max_iter):

            if i % 500 == 0:
                print(str(i) + "th iteration")
            
            player_id = rng.integers(0, len(self.players), 1)[0] # randomly choose a player
            
            player = self.players[player_id] 
            
            mixed_strategies[player_id] = player.mixed_strategy() # obtain the players strategy
            
            self.action_profile[player_id] = rng.choice(self.action_space[player_id], 1, p = mixed_strategies[player_id]) # sample action from players strategy              
    
            opponents_actions = self.action_profile[self.opponents_idx_map[player_id]] # extract the opponents actions from the action profile
            action = self.action_profile[player_id]
                                
            player.update_exp3p(action, opponents_actions, gamma, beta, eta) # update players strategy
            
            self.potentials_history[i] = (past_exp_potential * i + self.gameSetup.potential_function(self.action_profile))/(i+1) # empirical expected value

            past_exp_potential = self.potentials_history[i]

            if isinstance(self.gameSetup, CongestionGame):
                self.objectives_history[i] = self.gameSetup.objective(self.action_profile)   
        
    def compute_beta(self, epsilon):
        """
            Computes the lower bound on rationality that guarantees finite time convergence.
            A. Maddux, R. Ouhamma and M. Kamgarpour, ‘Finite-time convergence to an ϵ-efficient nash equilibrium in potential games'

        Args:
            epsilon (double): Desired precision.

        Returns:
            double: Lower bound on players rationality.
        """
        
        A = self.gameSetup.no_actions
        N = self.gameSetup.no_players
        delta = self.gameSetup.delta
        
        return 1/max(epsilon, delta)*(N*np.log(A) - np.log(epsilon))

    def compute_t(self, epsilon):
        """
            Computes the maximum time until convergence.
            A. Maddux, R. Ouhamma and M. Kamgarpour, ‘Finite-time convergence to an ϵ-efficient nash equilibrium in potential games'

        Args:
            epsilon (double): Desired precision

        Returns:
            double: Maximum time until convergence
        """
        
        A = self.gameSetup.no_actions
        N = self.gameSetup.no_players
        delta = self.gameSetup.delta
        beta = self.compute_beta(epsilon)
        
        if self.gameSetup.noisy_utility:
            return np.log(N**1.5*A**3) + N + beta*(1+1/beta)*(N+3) * np.log(-2*np.log(epsilon))
        
        return np.log(N**2*A**5) + (1/max(epsilon, delta))*N*np.log(A/epsilon)

    def set_max_iter(self, epsilon):
        """
            Set the number of iterations based the convergence guarantee for the desired precision of the game.

        Args:
            epsilon (double): Precision.
        """
        
        self.max_iter = int(min(1e5, self.compute_t(epsilon)))
        
        self.action_profile_history = np.zeros((self.max_iter, self.gameSetup.no_players))
        self.player_id_history = np.zeros((self.max_iter, 1))
        
        self.potentials_history = np.zeros((self.max_iter, 1))
        self.player_converged_history = np.zeros((self.max_iter, 1))
    
    def set_algorithm(self, algorithm):
        """
            Change the learning algorithm.

        Args:
            algorithm (str): Algorithm.
        """
        
        self.algorithm = algorithm
        
    def reset_game(self, delta = None, payoff_matrix = None):
        """
            Reset the game.

        Args:
            delta (double, optional): Suboptimality gap of the game. Defaults to None.
            payoff_matrix (np.array(AxAxA...(Nd)), optional): Payoff matrix of the game. Defaults to None.
        """
        
        if delta is None:
            delta = self.gameSetup.delta
        if payoff_matrix is not None:
            self.gameSetup.set_payoff_matrix(delta, payoff_matrix)

        [self.players[i].reset_player(self.gameSetup.no_actions, self.gameSetup.utility_functions[i]) for i in range(0, self.gameSetup.no_players)]
