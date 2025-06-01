import numpy as np

rng = np.random.default_rng()

class Player:
    """
    Class representing the Player.

    Includes information on the action space of the player, past actions, strategies etc.
    """
       
    def __init__(
        self,
        player_id: int,
        action_space: np.ndarray,
        utility: callable,
        noisy_utility: bool = False,
        fixed_share: bool = False
    ) -> None:
        """
        Args:
            player_id (int): Player id.
            action_space (np.array(A)): Player's action space.
            utility (function): Utility function.
            noisy_utility (bool, optional): Use noisy utilities. Defaults to False.
            fixed_share (bool, optional): Use fixed-share log linear learning. Defaults to False.
        """
        
        self.id = player_id
        self.update_player(action_space, utility, noisy_utility, fixed_share)

    def _initialize_player_attributes(self):
        """
        Initialize or reset player attributes to their default values.
        """
        self.previous_action = None
        self.action_space = np.arange(self.n_actions).reshape(1, self.n_actions)
        self.probabilities = 1/self.n_actions*np.ones([1, self.n_actions])
        self.weights = 1/self.n_actions*np.ones([1, self.n_actions])
        self.scores = 1/self.n_actions*np.ones([1, self.n_actions])
        self.initial_action = np.array([0])
        self.ones_vector = np.ones(self.n_actions)
        self.rewards_estimate = np.zeros(self.n_actions)
        self.min_payoff = None
        self.max_payoff = None
        self.previous_opponents_actions = None
        self.utilities = None
        
    def update_log_linear(
        self,
        beta: float,
        opponents_actions: np.ndarray,
        eta: float,
        gamma: float = 0
    ) -> int:
        """
            Update player's strategy and sample new action from the strategy - based on log-linear learning.
        Args:
            beta (float): Player's rationality
            opponents_actions (np.ndarray): Joint action profile of the opponents.
            eta (float): Noise.
            gamma (float, optional): Exploration factor. Defaults to 0.

        Raises:
            Exception: Noise bound not provided.

        Returns:
            int: index of the chosen action
        """
        if self.noisy_utility and eta <= 0:
            raise ValueError("Noisy utility requires a positive noise bound (eta).")

        if self.utilities is None or not np.array_equal(self.previous_opponents_actions, opponents_actions):
            # compute utilities
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.n_actions)]).reshape(1, self.n_actions)
            
            if self.min_payoff is not None:
                self.utilities = (self.utilities - self.min_payoff)/(self.max_payoff - self.min_payoff)
            
            self.utilities +=  rng.uniform(-eta, eta, self.n_actions)

            # compute strategy        
            exp_values = np.exp(beta * (self.utilities - np.max(self.utilities)))
            self.probabilities = exp_values/np.sum(exp_values)
            
            self.probabilities = gamma/self.n_actions + (1-gamma)*self.probabilities
            self.previous_opponents_actions = opponents_actions
            
        chosen_action = rng.choice(self.action_space[0], size=1, p=self.probabilities[0])

        self.previous_action = chosen_action
        
        return chosen_action
        
    def update_log_linear_binary(
        self,
        beta: float,
        opponents_actions: np.ndarray
    ) -> int:
        """
            Update player's strategy and sample new action from the strategy - based on log-linear learning with two-point feedback.

        Args:
            beta (float): Player's rationality
            opponents_actions (np.ndarray): Joint action profiles of the opponents.

        Returns:
            int: Chosen action.
        """
        
        # sample trial action
        new_action = rng.integers(0, self.n_actions, 1).astype(int)[0]
        new_utility = self.utility(new_action, opponents_actions)
        
        if self.utilities is None or not np.array_equal(self.previous_opponents_actions, opponents_actions):
            self.utilities = self.utility(self.previous_action, opponents_actions)
        
        actions = [self.previous_action, new_action]
        # compute utilities
        utilities = np.array([self.utilities, new_utility])
        # compute strategy
        exp_values = np.exp(beta * (utilities - np.max(utilities)))
        probabilities = exp_values/np.sum(exp_values)
        
        chosen_action = rng.choice(actions, size=1, p=probabilities.T[0])
        self.previous_action = chosen_action[0]
        
        return chosen_action
    
    def update_modified_log_linear(
        self,
        beta: float,
        opponents_actions: np.ndarray
    ) -> int:
        """
            Update player's strategy and sample new action from the strategy - based on modified log-linear learning.

        Args:
            beta (float): Player's rationality
            opponents_actions (np.ndarray): Joint action profiles of the opponents.

        Returns:
            int: Chosen action.
        """
        
        if self.utilities is None or not np.array_equal(self.previous_opponents_actions, opponents_actions):
            # Compute utilities
            self.utilities = np.array([self.utility_modified(i, opponents_actions) for i in range(self.n_actions)]).reshape(1, self.n_actions)
            # Compute strategy
            exp_values = np.exp(beta * (self.utilities - np.max(self.utilities)))
            self.probabilities = exp_values/np.sum(exp_values)
            self.previous_opponents_actions = opponents_actions
            
        chosen_action = rng.choice(self.action_space[0], size=1, p=self.probabilities[0])

        self.previous_action = chosen_action
        
        return chosen_action
    
    def best_response(self, opponents_actions: np.ndarray) -> int:
        """
            Compute the best response of the player given other players' actions.

        Args:
            opponents_actions (np.array(N-1)): Joint action profile of the opponents.

        Returns:
            int: Action
        """
        
        # compute utilities
        if self.utilities is None or not np.array_equal(self.previous_opponents_actions, opponents_actions):
                    
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.n_actions)]).reshape(1, self.n_actions)

            chosen_action = np.argmax(self.utilities)
        
            self.previous_action = chosen_action
            self.previous_opponents_actions = opponents_actions
        
        return self.previous_action
    
    def mixed_strategy(self) -> np.ndarray:
        """
        Returns:
            np.array(A): Strategy.
        """
        return self.probabilities
        
    def update_mw(
        self,
        opponents_actions: np.ndarray,
        gamma_t: float = 0.5
    ) -> None:
        """
            Update the player's strategy based on Hedge given other player's actions.

        Args:
            opponents_actions (np.array(N-1)): Joint action profile of the opponents.
            gamma_t (float, optional): Exploration factor. Defaults to 0.5.
        """
                
        # compute utilities 
        # Efficiently cache utilities only if opponents' actions have changed
        if self.utilities is None or not np.array_equal(self.previous_opponents_actions, opponents_actions):
            
            self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.n_actions)]).reshape(1, self.n_actions)
            self.previous_opponents_actions = opponents_actions
            
            if self.min_payoff is not None:
                self.utilities = (self.utilities - self.min_payoff)/(self.max_payoff - self.min_payoff)
        
        losses = self.ones_vector - self.utilities
        
        self.probabilities = np.multiply(self.probabilities, np.exp(np.multiply(gamma_t, -losses)))
        
        self.probabilities = self.probabilities / np.sum(self.probabilities)
    
    def update_ewa(
        self,
        action: int,
        opponents_actions: np.ndarray,
        gamma_n: float,
        eps_n: float
    ) -> None:
        """
            Update the player's strategy based on Exponential weight with annealing given the past action and given the other players actions.
        Args:
            action (int): Action.
            opponents_actions (np.array(N-1)): Joint action profile of the opponents.
            gamma_n (float): Step size.
            eps_n (float): Vanishing factor.
        """

        v = np.zeros(self.n_actions)
        v[action] = self.utility(action, opponents_actions)
        
        if self.min_payoff is not None:
            v[action] = (v[action] - self.min_payoff)/(self.max_payoff - self.min_payoff)
        
        v[action] /= self.probabilities[0][action]
        self.scores += gamma_n * v

        exp_values = np.exp((self.scores - np.max(self.scores)))
        lambda_scores = exp_values/np.sum(exp_values)
        
        self.probabilities = eps_n*self.ones_vector/self.n_actions + (1-eps_n)*lambda_scores
        self.probabilities = self.probabilities / np.sum(self.probabilities)
        
    def update_exp3p(
        self,
        action: int,
        opponents_actions: np.ndarray,
        gamma: float,
        beta: float,
        eta: float
    ) -> None:
        """
            Update the player's strategy based on EXP3P algorithm.

        Args:
            action (int): Action.
            opponents_actions (np.array(N-1)): Joint action profile of the opponents.
            gamma (float): Parameter of EXP3P
            beta (float): Parameter of EXP3P
            eta (float): Parameter of EXP3P
        """
        
        v = np.zeros(self.n_actions)
        v[action] = self.utility(action, opponents_actions)
                
        if self.min_payoff is not None:
            v[action] = (v[action] - self.min_payoff)/(self.max_payoff - self.min_payoff)

        v[action] = v[action]/self.probabilities[0][action]
                
        self.rewards_estimate = self.rewards_estimate + beta*np.divide(self.ones_vector, self.probabilities) + v
       
        temp = np.multiply(eta, self.rewards_estimate)
        self.weights  = np.exp(temp - np.max(temp))
        self.weights  = self.weights/np.sum(self.weights)
        self.probabilities = (1-gamma)*self.weights + gamma/self.n_actions*self.ones_vector
        
        self.probabilities = self.probabilities/np.sum(self.probabilities)
        
    def set_modified_utility(self, utility_modified: callable) -> None:
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
        self._initialize_player_attributes()
        
    def update_player(self, action_space: np.ndarray, utility: callable, noisy_utility: bool = False, fixed_share: bool = False) -> None:
        """
            Update the player's action space, utility function, and optional parameters.

        Args:
            action_space (np.ndarray): Player's action space.
            utility (function): Utility function.
            noisy_utility (bool, optional): Use noisy utilities. Defaults to False.
            fixed_share (bool, optional): Use fixed-share log linear learning. Defaults to False.
        """
        self.n_actions = len(action_space)
        self.action_space = np.arange(self.n_actions).reshape(1, self.n_actions)
        self.utility = utility
        self.noisy_utility = noisy_utility
        self.fixed_share = fixed_share
        self._initialize_player_attributes()
