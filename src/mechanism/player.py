import numpy as np

rng = np.random.default_rng()

class Player:
    """
        Class representing the Player.
        
        Includes information on the action spcae of the player, past actions, strategies etc.
    """
    
    def __init__(self, player_id, action_space, utility, noisy_utility = False, fixed_share = False):
        """
        

        Args:
            player_id (int): Player id.
            action_space (np.array(A)): Player's action space.
            utility (function): Utility function.
            noisy_utility (bool, optional): Use noisy utilities. Defaults to False.
            fixed_share (bool, optional): Use fixed-share log linear learning. Defaults to False.
        """
        
        self.id = player_id
        self.no_actions = len(action_space) # size of the actions space
        self.utility = utility # utility function
        self.noisy_utility = noisy_utility
        self.fixed_share = fixed_share
        self.past_action = None
        self.action_space = np.arange(self.no_actions).reshape(1, self.no_actions)
        self.prob = 1/self.no_actions*np.ones([1, self.no_actions])
        self.weights = 1/self.no_actions*np.ones([1, self.no_actions])
        self.scores = 1/self.no_actions*np.ones([1, self.no_actions])
        self.initial_action = np.array([0])
        self.ones = np.ones(self.no_actions)
        self.rewards_estimate = np.zeros(self.no_actions)
        self.min_payoff = None
        self.max_payoff = None
        self.past_opponents_actions = None
        self.utilities = None
        
    def update_log_linear(self, beta, opponents_actions, eta, gamma = 0): # choose a new action only based on the opponents action, in this case they will be the same as the actions in the previous step
        """
            Update player's strategy and sample new action from the strategy - based on log-linear learning.
        Args:
            beta (double): Player's rationality
            opponents_actions (np.array(N-1)): Joint action profile of the opponents.
            eta (double): Noise.
            gamma (double, optional): Exploration factor. Defaults to 0.

        Raises:
            Exception: Noise bound not provided.

        Returns:
            int: index of the chosen action
        """
        if eta <= 0 and self.noisy_utility:
            raise Exception("Sorry, you are using noisy utility, but haven't provided the bounds!")

        if self.utilities is None or (self.past_opponents_actions != opponents_actions).any(): # caching
            # compute utilities
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)
            
            if self.min_payoff is not None:
                self.utilities = (self.utilities - self.min_payoff)/(self.max_payoff - self.min_payoff)
            
            self.utilities +=  rng.uniform(-eta, eta, self.no_actions)

            # print(self.utilities)
            # print(self.utilities - np.max(self.utilities))        
            # print(beta)    
            # compute strategy        
            exp_values = np.exp(beta * (self.utilities - np.max(self.utilities)))
            self.prob = exp_values/np.sum(exp_values)
            
            # print(self.prob)
            self.prob = gamma/self.no_actions + (1-gamma)*self.prob
            self.past_opponents_actions = opponents_actions
            
            # print(self.prob)

        idx_a = rng.choice(self.action_space[0], size=1, p=self.prob[0]) # sample action from strategy
        
        # print(idx_a)
        # print("----------------------------")
        self.past_action = idx_a
        
        return idx_a
        
    def update_log_linear_binary(self, beta, opponents_actions):
        """
            Update player's strategy and sample new action from the strategy - based on log-linear learning with two-point feedback.

        Args:
            beta (double): Player's rationality
            opponents_actions (np.array(N-1)): Joint action profiles of the opponents.

        Returns:
            int: Chosen action.
        """
        
        # sample trial action
        new_action = rng.integers(0, self.no_actions, 1).astype(int)[0]
        new_utility = self.utility(new_action, opponents_actions)
        
        if self.utilities is None or (self.past_opponents_actions != opponents_actions).any():
            self.utilities = self.utility(self.past_action, opponents_actions)
        
        actions = [self.past_action, new_action]
        # compute utilities
        utilities = np.array([self.utilities, new_utility])
        # compute strategy
        exp_values = np.exp(beta * (utilities - np.max(utilities)))
        self.prob = exp_values/np.sum(exp_values)
        
        idx_a = rng.choice(actions, size=1, p=self.prob.T[0]) # sample action
        self.past_action = idx_a[0]
        
        return self.past_action
    
    def update_modified_log_linear(self, beta, opponents_actions):
        """
            Update player's strategy and sample new action from the strategy - based on modified log-linear learning.

        Args:
            beta (double): Player's rationality
            opponents_actions (np.array(N-1)): Joint action profiles of the opponents.

        Returns:
            int: Chosen action.
        """
        
        if self.utilities is None or (self.past_opponents_actions != opponents_actions).any(): # caching
            # Compute utilities
            self.utilities = np.array([self.utility_modified(i, opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)
            # Compute strategy
            exp_values = np.exp(beta * (self.utilities - np.max(self.utilities)))
            self.prob = exp_values/np.sum(exp_values)
            self.past_opponents_actions = opponents_actions
            
        idx_a = rng.choice(self.action_space[0], size=1, p=self.prob[0]) # sample action

        self.past_action = idx_a
        
        return idx_a
    
    def best_response(self, opponents_actions):
        """
            Compute the best response of the player given other players' actions.

        Args:
            opponents_actions (np.array(N-1)): Joint action profile of the opponents.

        Returns:
            int: Action
        """
        
        # compute utilities
        if self.utilities is None or (self.past_opponents_actions != opponents_actions).any():
        
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)

            idx_a = np.argmax(self.utilities) # obtain the utility maximiser
        
            self.past_action = idx_a
            self.past_opponents_actions = opponents_actions
        
        return self.past_action
    
    def mixed_strategy(self):
        """
        
        Returns:
            np.array(A): Strategy.
        """
        
        return self.prob
        
    def update_mw(self, opponents_actions, gamma_t = 0.5):
        """
            Update the player's strategy based on Hedge given other player's actions.

        Args:
            opponents_actions (np.array(N-1)): Joint action profile of the opponents.
            gamma_t (float, optional): Exploration factor. Defaults to 0.5.
        """
                
        # compute utilities 
        if self.utilities is None or (self.past_opponents_actions != opponents_actions).any(): # caching
            
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)
            self.past_opponents_actions = opponents_actions
            
            if self.min_payoff is not None:
                self.utilities = (self.utilities - self.min_payoff)/(self.max_payoff - self.min_payoff)
        
        losses = self.ones - self.utilities
        
        self.prob = np.multiply(self.prob, np.exp(np.multiply(gamma_t, -losses))) # update straetgy
        
        self.prob = self.prob / np.sum(self.prob)
    
    def update_ewa(self, action, opponents_actions, gamma_n, eps_n):
        """
            Update the player's strategy based on Exponential weight with annealing given the past action and given the other players actions.
        Args:
            action (int): Action.
            opponents_actions (np.array(N-1)): Joint action profile of the opponents.
            gamma_n (double): Step size.
            eps_n (double): vanishing factor.
        """
        
        v = np.zeros(self.no_actions)
        v[action] = self.utility(action, opponents_actions)
        
        if self.min_payoff is not None:
            v[action] = (v[action] - self.min_payoff)/(self.max_payoff - self.min_payoff)
        
        v[action] /= self.prob[0][action]
        self.scores += gamma_n * v

        exp_values = np.exp((self.scores - np.max(self.scores)))
        lambda_scores = exp_values/np.sum(exp_values)
        
        self.prob = eps_n*self.ones/self.no_actions + (1-eps_n)*lambda_scores
        self.prob = self.prob / np.sum(self.prob)
        
    def update_exp3p(self, action, opponents_actions, gamma, beta, eta):
        """
            Update the player's strategy based on EXP3P algorithm.

        Args:
            action (int): Action.
            opponents_actions (np.array(N-1)): Joint action profile of the opponents.
            gamma (double): Parameter of EXP3P
            beta (double): Parameter of EXP3P
            eta (double): Parameter of EXP3P
        """
        
        v = np.zeros(self.no_actions)
        v[action] = self.utility(action, opponents_actions)
                
        if self.min_payoff is not None:
            v[action] = (v[action] - self.min_payoff)/(self.max_payoff - self.min_payoff)

        v[action] = v[action]/self.prob[0][action]
                
        self.rewards_estimate = self.rewards_estimate + beta*np.divide(self.ones, self.prob) + v
       
        temp = np.multiply(eta, self.rewards_estimate)
        self.weights  = np.exp(temp - np.max(temp))
        self.weights  = self.weights/np.sum(self.weights)
        self.prob = (1-gamma)*self.weights + gamma/self.no_actions*self.ones
        
        self.prob = self.prob/np.sum(self.prob)
        
    def set_modified_utility(self, utility_modified):
        """
            Set utility function in the modified game case.
        Args:
            utility_modified (function): Utility function.
        """
        
        self.utility_modified = utility_modified
    
    def reset(self):
        """
            Reset initial values of important attributes.
        """
        
        self.past_action = None
        self.prob = 1/self.no_actions*np.ones([1, self.no_actions])
        self.weights = 1/self.no_actions*np.ones([1, self.no_actions])
        self.scores = 1/self.no_actions*np.ones([1, self.no_actions])
        self.initial_action = np.array([0])
        self.ones = np.ones(self.no_actions)
        self.rewards_estimate = np.zeros(self.no_actions)
        self.past_opponents_actions = None
        self.utilities = None
        
    def reset_player(self, no_actions, utility):
        """
            Reset the utility function of a player.

        Args:
            no_actions (int): Number of actions.
            utility (function): Utility function
        """

        self.no_actions = no_actions # size of the actions space
        self.utility = utility # utility function