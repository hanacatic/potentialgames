import numpy as np
from src.mechanism import Player
from src.utils.helpers import *
from scipy.sparse import csr_matrix, csc_array
from typing import Callable, Optional
from src.mechanism.game_setup.abstract_setup import AbstractGameSetup
from src.mechanism.algorithms import LogLinearAlgorithm, BinaryLogLinearAlgorithm, FastLogLinearAlgorithm, FastBinaryLogLinearAlgorithm
from src.utils.logger import logger

rng = np.random.default_rng(seed = 2)

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

    def play(
        self,
        initial_action_profile: Optional[np.ndarray] = None,
        beta: Optional[float] = None,
        scale_factor: int = 1,
        gamma: float = 0
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
        
        # Reset players (gameBase function play can have successive calls)
        for player_id in range(self.gameSetup.no_players):
            player = self.players[player_id]
            player.previous_action = self.action_profile[player_id].copy()            

        # Use the algorithm registry
        if self.algorithm in self.algorithm_registry:
            self.algorithm_registry[self.algorithm](self, beta, scale_factor, gamma)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
    def _modified_log_linear(self, beta: float) -> None:
        """
            Modified log-linear learning leveraging symmetry of the game.

        Args:
            beta (double): Player rationality.
        """
        [self.players[i].set_modified_utility(self.gameSetup.modified_utility_functions[i]) for i in self.player_idx_map]
        self._modified_log_linear_impl(beta=beta)

    def _modified_log_linear_impl(self, beta: float) -> None:
        self.phi = np.zeros(self.gameSetup.no_actions) # histogram over the action space
        
        for a in self.action_profile:
            self.phi[a] += 1
        
        # Main loop
        for i in range(self.max_iter):
            time = i
            self._modified_log_linear_iteration(time, beta)
    
    def _modified_log_linear_iteration(self, i: int, beta: float) -> None:
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
                
        player_id = rng.choice(self.player_idx_map, size=1, p=player_clock)[0]
        
        player = self.players[player_id]
        
        self.phi[self.action_profile[player_id]] -= 1
        
        self.action_profile[player_id] = player.update_modified_log_linear(beta, self.phi) # update chosen players strategy
        
        self.potentials_history[i] = self.gameSetup.potential_function(self.action_profile) # compute the value of the potential function
        
        self.phi[self.action_profile[player_id]] += 1 

    def compute_beta(self, epsilon: float) -> float:
        """
            Computes the lower bound on rationality that guarantees finite time convergence.
        Args:
            epsilon (double): Desired precision.

        Returns:
            double: Lower bound on players rationality.
        """
        
        A = self.gameSetup.no_actions
        N = self.gameSetup.no_players
        delta = self.gameSetup.delta
        
        if self.gameSetup.symmetric is True:
            1/max(epsilon, delta)*(A*np.log(N) - np.log(epsilon))
        
        return 1/max(epsilon, delta)*(N*np.log(A) - np.log(epsilon))

    def compute_t(self, epsilon: float) -> float:
        """
            Computes the maximum time until convergence.
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

    def set_max_iter(self, epsilon: float) -> None:
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
    
    def set_algorithm(self, algorithm: str) -> None:
        """
            Change the learning algorithm.
        Args:
            algorithm (str): Algorithm.
        """
        self.algorithm = algorithm
        
    def reset_game(self, delta: Optional[float] = None, payoff_matrix: Optional[np.ndarray] = None) -> None:
        """
            Reset the game to its initial state.
        Args:
            delta (double, optional): Suboptimality gap of the game. Defaults to None.
            payoff_matrix (np.array(AxAxA...(Nd)), optional): Payoff matrix of the game. Defaults to None.
        """
        
        if delta is None:
            delta = self.gameSetup.delta
        if payoff_matrix is not None:
            self.gameSetup.set_payoff_matrix(delta, payoff_matrix)

        [self.players[i].reset_player(self.gameSetup.no_actions, self.gameSetup.utility_functions[i]) for i in range(0, self.gameSetup.no_players)]

# --- Algorithm registration ---

@GameEngine.register_algorithm("log_linear")
def _register_log_linear(game: 'Game', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    LogLinearAlgorithm.run(game, beta, gamma)

@GameEngine.register_algorithm("log_linear_fast")
def _register_log_linear_fast(game: 'Game', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    FastLogLinearAlgorithm.run(game, beta, scale_factor)

@GameEngine.register_algorithm("log_linear_binary")
def _register_log_linear_binary(game: 'Game', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    BinaryLogLinearAlgorithm.run(game, beta)

@GameEngine.register_algorithm("log_linear_binary_fast")
def _register_log_linear_binary_fast(game: 'Game', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    FastBinaryLogLinearAlgorithm(game, beta, scale_factor)

@GameEngine.register_algorithm("modified_log_linear")
def _register_modified_log_linear(game: 'Game', beta: Optional[float], scale_factor: int, gamma: float) -> None:
    [game.players[i].set_modified_utility(game.gameSetup.modified_utility_functions[i]) for i in game.player_idx_map]
    game._modified_log_linear(beta=beta)
