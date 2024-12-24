import numpy as np

rng = np.random.default_rng()

class Player:
    
    def __init__(self, player_id, action_space, utility):
        
        self.id = player_id
        self.no_actions = len(action_space) # size of the actions space
        self.utility = utility # utility function
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
        
    def update_log_linear(self, beta, opponents_actions): # choose a new action only based on the opponents action, in this case they will be the same as the actions in the previous step
        
        if self.utilities is None or all(self.past_opponents_actions != opponents_actions):
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)
            exp_values = np.exp(beta * (self.utilities - np.max(self.utilities)))
            self.prob = exp_values/np.sum(exp_values)
            self.past_opponents_actions = opponents_actions
        
        idx_a = rng.choice(self.action_space[0], size=1, p=self.prob[0])

        self.past_action = idx_a
        
        return idx_a
    
    def update_log_linear_binary(self, beta, opponents_actions):
        
        new_action = rng.integers(0, self.no_actions, 1).astype(int)[0]
        new_utility = self.utility(new_action, opponents_actions)
        
        if self.utilities is None or all(self.past_opponents_actions != opponents_actions):
            self.utilities = self.utility(self.past_action, opponents_actions)
        
        actions = [self.past_action, new_action]
        utilities = np.array([self.utilities, new_utility])
        exp_values = np.exp(beta * (utilities - np.max(utilities)))
        self.prob = exp_values/np.sum(exp_values)
        
        idx_a = rng.choice(actions, size=1, p=self.prob.T[0])
        self.past_action = idx_a[0]
        
        return self.past_action
    
    def update_modified_log_linear(self, beta, opponents_actions):
        
        if self.utilities is None or (self.past_opponents_actions != opponents_actions).any():
            self.utilities = np.array([self.utility_modified(i, opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)
            exp_values = np.exp(beta * (self.utilities - np.max(self.utilities)))
            self.prob = exp_values/np.sum(exp_values)
            self.past_opponents_actions = opponents_actions
            
        idx_a = rng.choice(self.action_space[0], size=1, p=self.prob[0])

        self.past_action = idx_a
        
        return idx_a
    
    def best_response(self, opponents_actions):
        
        if self.utilities is None or (self.past_opponents_actions != opponents_actions).any():
        
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)

            idx_a = np.argmax(self.utilities)
        
            self.past_action = idx_a
            self.past_opponents_actions = opponents_actions
        
        return self.past_action
    
    def mixed_strategy(self):
        
        return self.prob
        
    def update_mw(self, opponents_actions, gamma_t = 0.5):
                
        if self.utilities is None or (self.past_opponents_actions != opponents_actions).any():
            
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.no_actions)]).reshape(1, self.no_actions)
            self.past_opponents_actions = opponents_actions
            
            if self.min_payoff is not None:
                self.utilities = (self.utilities - self.min_payoff)/(self.max_payoff - self.min_payoff)
        
        losses = self.ones - self.utilities
        
        # self.prob = np.multiply( self.prob, 1 + gamma_t * (-losses))
        self.prob = np.multiply(self.prob, np.exp(np.multiply(gamma_t, -losses)))
        
        self.prob = self.prob / np.sum(self.prob)
    
    def update_ewa(self, action, opponents_actions, gamma_n, eps_n):
        
        v = np.zeros(self.no_actions)
        v[action] = self.utility(action, opponents_actions)
        
        if self.min_payoff is not None:
            self.v[action] = (self.v[action] - self.min_payoff)/(self.max_payoff - self.min_payoff)
        
        v[action] /= self.prob[0][action]
        self.scores += gamma_n * v

        exp_values = np.exp((self.scores - np.max(self.scores)))
        lambda_scores = exp_values/np.sum(exp_values)
        
        self.prob = eps_n*self.ones/self.no_actions + (1-eps_n)*lambda_scores
        self.prob = self.prob / np.sum(self.prob)
        
    def update_exp3p(self, action, opponents_actions, gamma, beta, eta):
        
        # implemented exp4 from paper (Auer et al. 2002)
        v = np.zeros(self.no_actions)
        v[action] = self.utility(action, opponents_actions)
        
        # print(v[action])
        
        if self.min_payoff is not None:
            self.v[action] = (self.v[action] - self.min_payoff)/(self.max_payoff - self.min_payoff)

        # v[action] = (1-v[action]) / self.prob[0][action]
        # print(self.prob)
        # print(self.prob[0][action])
        v[action] = v[action]/self.prob[0][action]

        # print(v)
        
        # print(self.rewards_estimate)
        
        self.rewards_estimate = self.rewards_estimate + beta*np.divide(self.ones, self.prob) + v

        # self.rewards_estimate = self.rewards_estimate + beta*np.divide(self.ones, self.prob)*(v>0) + v
       
        temp = np.multiply(eta, self.rewards_estimate)
        self.weights  = np.exp(temp - np.max(temp))
        self.weights  = self.weights/np.sum(self.weights)
        self.prob = (1-gamma)*self.weights + gamma/self.no_actions*self.ones
        
        self.prob = self.prob/np.sum(self.prob)
        # print(self.prob)
        
        # self.rewards_estimate = self.rewards_estimate + eta*v
        # exp_values = np.exp(self.rewards_estimate-np.max(self.rewards_estimate))
        # print(exp_values)
        # self.prob = exp_values/np.sum(exp_values)
        # print()
        
    def set_modified_utility(self, utility_modified):
        
        self.utility_modified = utility_modified
        
    def reset_player(self, no_actions, utility):

        self.no_actions = no_actions # size of the actions space
        self.utility = utility # utility function