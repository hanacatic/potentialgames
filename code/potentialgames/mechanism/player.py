import numpy as np

from ..utils import logger, rng


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
        self.update(action_space, utility, noisy_utility, fixed_share)

    def _initialize_player_attributes(self):
        """
        Initialize or reset player attributes to their default values.
        """
        self.previous_action = None
        self.action_space = np.arange(self.n_actions).reshape(1, self.n_actions)
        self.probabilities = np.ascontiguousarray(1/self.n_actions*np.ones([1, self.n_actions]))
        self.weights = np.ascontiguousarray(1/self.n_actions*np.ones([1, self.n_actions]))
        self.scores = 1/self.n_actions*np.ones([1, self.n_actions])
        self.initial_action = np.array([0])
        self.ones_vector = np.ascontiguousarray(np.ones(self.n_actions))
        self.zeros_vector = np.zeros(self.n_actions)
        self.rewards_estimate = np.ascontiguousarray(np.zeros(self.n_actions))
        self.min_payoff = None
        self.max_payoff = None
        self.previous_opponents_actions = None
        self.utilities = None
        
    def compute_utilities(self, opponents_actions=None, eta_noise=0.0):
        """
        Compute the utilities for the player based on the current action space and utility function.

        Args:
            opponents_actions: Actions taken by opponents (if needed by utility function).
            eta_noise (float): Magnitude of uniform noise to add to utilities.
        """
        if self.noisy_utility and eta_noise <= 0:
            raise ValueError("Noisy utility requires a positive noise bound (eta).")

        self.utilities = np.array([self.utility(i, opponents_actions) for i in range(self.n_actions)]).reshape(1, self.n_actions)
        self.utilities = np.ascontiguousarray(self.utilities)

        if self.min_payoff is not None and self.max_payoff is not None and self.max_payoff != self.min_payoff:
            self.utilities = (self.utilities - self.min_payoff) / (self.max_payoff - self.min_payoff)

        if eta_noise > 0.0:
            self.utilities += rng.uniform(-eta_noise, eta_noise, self.n_actions)

    def sample_action(self) -> int:
        """
        Sample an action from the player's action space based on the current probabilities.

        Returns:
            int: Chosen action.
        """
        if self.probabilities is None:
            raise ValueError("Probabilities are not set. Please update the player first.")
        
        chosen_action = rng.choice(self.action_space[0], size=1, p=self.probabilities[0])[0]
        self.previous_action = chosen_action
        return chosen_action
    
    def reset(self):
        """
            Reset initial values of important attributes.
        """
        self._initialize_player_attributes()
        
    def update(self, action_space: np.ndarray, utility: callable, noisy_utility: bool = False, fixed_share: bool = False) -> None:
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
