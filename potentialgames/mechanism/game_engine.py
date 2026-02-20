import numpy as np
from typing import Callable, Optional

from ..mechanism import Player
from ..utils.helpers import *
from ..mechanism.game_setup.abstract_setup import AbstractGameSetup
from ..mechanism.algorithms import LogLinearAlgorithm, BinaryLogLinearAlgorithm, FastLogLinearAlgorithm, FastBinaryLogLinearAlgorithm, ModifiedLogLinearAlgorithm, HedgeAlgorithm, EXP3PAlgorithm, ExponentialWeightWithAnnealingAlgorithm
from ..utils import logger, plot_line, compute_t, compute_beta


class GameEngine:
    """
        Class representing the GameEngine base.
        
        Includes information on the game setup and the algorithms.
    """

    algorithm_registry = {}

    @classmethod
    def register_algorithm(cls, name: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            cls.algorithm_registry[name] = func
            return func
        return decorator

    def __init__(self, gameSetup: AbstractGameSetup, algorithm: str = 'log_linear',  max_iter: int = 200000, mu: Optional[Callable] = None): # mu - initial distribution
        """
        Args:
            gameSetup (IdentInterest/TrafficRouting): Game setup, defines the type of game, the properties of the game, including the number of players and the number of actions
            algorithm (str, optional): Type of algorithm to solve the game. Defaults to 'log_linear'.
            max_iter (int, optional): Maximum number of iterations available to the algorithm. Defaults to 200000.
            mu (function, optional): Initial joint action profile distribution. Defaults to None.
        """
        
        self.gameSetup = gameSetup # all the game rules and game data
        self.algorithm = algorithm
        self.max_iter = int(max_iter)
        
        self.players = np.array([ Player(i, self.gameSetup.action_space[i], gameSetup.utility_functions[i], self.use_noisy_utility) for i in range(0, self.no_players)], dtype = object)
        self.action_space = [np.arange(len(self.gameSetup.action_space[player_id])) for player_id in range(self.no_players)]

        # initial joint action profile
        self.action_profile = [0] * self.no_players
        
        if mu is not None: 
            self.action_profile = self.sample_initial_action_profile(mu)
        
        self.expected_value = None

        self.opponents_idx_map = [ np.delete(np.arange(self.no_players), player_id) for player_id in range(self.no_players) ]
        self.player_idx_map = np.arange(0, self.no_players) 
        
        self.potentials_history = np.zeros((self.max_iter, 1))                           

    @property
    def no_players(self) -> int:
        """
            Returns the number of players in the game.
        """
        return self.gameSetup.no_players
    @property
    def no_actions(self) -> int:
        """
            Returns the number of actions in the game.
        """
        return self.gameSetup.no_actions
    @property
    def delta(self) -> float:
        """
            Returns the suboptimality gap of the game.
        """
        return self.gameSetup.delta
    @property
    def symmetric(self) -> bool:
        """
            Returns whether the game is symmetric or not.
        """
        return self.gameSetup.symmetric
    @property
    def use_noisy_utility(self) -> bool:
        """
            Returns whether the game uses noisy utility or not.
        """
        return self.gameSetup.use_noisy_utility
    
    @property
    def eta_noise(self):
        """
            Returns the noise level used in the game.
        """
        return self.gameSetup.eta if self.use_noisy_utility else 0
    
    @eta_noise.setter
    def eta_noise(self, eta):
        
        self.gameSetup.eta = eta
    
    def sample_initial_action_profile(self, mu: Callable) -> np.ndarray:
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
    
    def set_initial_action_profile(self, initial_action_profile: np.ndarray) -> None:
        """
            Sets the initial joint action profile to the given joint action profile.
        Args:
            initial_action_profile (np.array(N)): Joint action profile.
        """
        self.initial_action_profile = initial_action_profile
        
    def set_mu_matrix(self, mu_matrix: np.ndarray) -> None:
        """
            Sets the initial joint action profile matrix distribution to the given distribution.
        Args:
            mu_matrix (np.array(AxA....A (Nd))): Initial joint action profile matrix distribution.
        """
        self.mu_matrix = mu_matrix

    def set_uniform_mu_matrix(self) -> None:
        """
            Sets the initial joint action profile matrix distribution to a uniform distribution.
        """
        self.mu_matrix = self.gameSetup.get_uniform_mu_matrix()
    
    def play(
        self,
        initial_action_profile: Optional[np.ndarray] = None,
        beta: Optional[float] = None,
        scale_factor: int = 1,
        gamma: float = 0,
        max_iter: Optional[int] = None
    ) -> None:
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
        
        # Reset players (game_engine function play can have successive calls)
        for player_id in range(self.no_players):
            player = self.players[player_id]
            player.previous_action = self.action_profile[player_id].copy()            

        if max_iter is not None:
            self.set_max_iter(max_iter)
            
        # Use the algorithm registry
        if self.algorithm in self.algorithm_registry:
            self.algorithm_registry[self.algorithm](self, beta, scale_factor, gamma)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def compute_beta(self, epsilon: float) -> float:
        
        A = self.no_actions
        N = self.no_players
        delta = self.delta
        symmetric = self.symmetric
        use_noisy_utility = self.use_noisy_utility
        
        return compute_beta(A, N, delta, epsilon, symmetric, use_noisy_utility)

    def compute_t(self, epsilon: float) -> int:
        
        A = self.no_actions
        N = self.no_players
        delta = self.delta
        symmetric = self.symmetric
        use_noisy_utility = self.use_noisy_utility
        
        beta = self.compute_beta(epsilon)
        
        return compute_t(A, N, delta, epsilon, beta, symmetric, use_noisy_utility)
    
    def set_max_iter(self, max_iter) -> None:
        """
            Set the number of iterations based the convergence guarantee for the desired precision of the game.
        Args:
            epsilon (double): Precision.
        """
        self.max_iter = int(max_iter)
                
        self.action_profile_history = np.zeros((self.max_iter, self.no_players))
        self.player_id_history = np.zeros((self.max_iter, 1))
        
        self.potentials_history = np.zeros((self.max_iter, 1))
        self.player_converged_history = np.zeros((self.max_iter, 1))
    
    def set_algorithm(self, algorithm: str) -> None:
        """
            Change the learning algorithm.
        Args:
            algorithm (str): Algorithm.
        """
        self.algorithm = algorithm
        
    def reset_game(self, payoff_matrix: Optional[np.ndarray] = None) -> None:
        """
            Reset the game to its initial state.
        Args:
            delta (double, optional): Suboptimality gap of the game. Defaults to None.
            payoff_matrix (np.array(AxAxA...(Nd)), optional): Payoff matrix of the game. Defaults to None.
        """
        
        if payoff_matrix is not None:
            self.gameSetup.set_payoff_matrix(payoff_matrix)

        [self.players[i].update(self.gameSetup.action_space[i], self.gameSetup.utility_functions[i]) for i in range(0, self.no_players)]
        
    def plot(self):
        """
            Plots the potentials history.
        """
        if "fast" in self.algorithm:
            if self.expected_value is not None:
                plot_line(self.expected_value, title="Expected Value", ylabel="Expected Value", xlabel="T")
            else:
                logger.warning("Expected value not available to plot.")
        else:
            if self.potentials_history is not None:
                plot_line(self.potentials_history, title="Potentials History", ylabel="Potential", xlabel="T")
            else:
                logger.warning("Potentials history not available to plot.")

# --- Algorithm registration ---

@GameEngine.register_algorithm("log_linear")
def _register_log_linear(game: 'GameEngine', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    LogLinearAlgorithm.run(game, beta, gamma)

@GameEngine.register_algorithm("fast_log_linear")
def _register_log_linear_fast(game: 'GameEngine', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    FastLogLinearAlgorithm.run(game, beta, scale_factor)

@GameEngine.register_algorithm("binary_log_linear")
def _register_log_linear_binary(game: 'GameEngine', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    BinaryLogLinearAlgorithm.run(game, beta)

@GameEngine.register_algorithm("fast_binary_log_linear")
def _register_log_linear_binary_fast(game: 'GameEngine', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    FastBinaryLogLinearAlgorithm.run(game, beta, scale_factor)

@GameEngine.register_algorithm("modified_log_linear")
def _register_modified_log_linear(game: 'GameEngine', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    for i in game.player_idx_map:
        game.players[i].utility = game.gameSetup.modified_utility_functions[i]
    ModifiedLogLinearAlgorithm.run(game, beta)

@GameEngine.register_algorithm("hedge")
def _register_hedge(game: 'GameEngine', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    HedgeAlgorithm.run(game)

@GameEngine.register_algorithm("exp3p")
def _register_exp3p(game: 'GameEngine', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    EXP3PAlgorithm.run(game)
    
@GameEngine.register_algorithm("exponential_weight_with_annealing")
def _register_exponential_weight_with_annealing(game: 'GameEngine', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    ExponentialWeightWithAnnealingAlgorithm.run(game)