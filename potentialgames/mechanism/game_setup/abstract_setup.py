from abc import ABC, abstractmethod
import numpy as np

class AbstractGameSetup(ABC):
    """
    Abstract base class for game setups (e.g., CoverageGame, IdenticalInterestGame).
    Defines the structure and common methods required by the Game class.
    """

    def __init__(self, no_players, action_space, noisy_utility=False, eta=None):
        """
        Initialize the game setup with players, action space, and optional noise settings.

        :param no_players: Number of players in the game.
        :param action_space: List or array defining the action space for each player.
        :param noisy_utility: Boolean indicating if noisy utilities are used.
        :param eta: Noise range (if noisy utilities are enabled).
        """
        self.no_players = no_players
        self.action_space = action_space
        self.noisy_utility = noisy_utility
        self.eta = eta if noisy_utility else 0
        self.utility_functions = []
        self.modified_utility_functions = []
        self.delta = None
        self.symmetric = False
        self.potential_vec = None

    @abstractmethod
    def utility_function(self, player_id, player_action, opponents_actions):
        """
        Compute the utility for a given player based on their action and opponents' actions.

        :param player_id: ID of the player.
        :param player_action: Action taken by the player.
        :param opponents_actions: Actions taken by the opponents.
        :return: Utility value for the player.
        """
        pass

    @abstractmethod
    def potential_function(self, action_profile):
        """
        Compute the potential function value for a given action profile.

        :param action_profile: List or array of actions taken by all players.
        :return: Value of the potential function.
        """
        pass

    @abstractmethod
    def formulate_transition_matrix(self, beta):
        """
        Formulate the transition matrix for the game based on player rationality.

        :param beta: Rationality parameter for the players.
        :return: Transition matrix.
        """
        pass

    @abstractmethod
    def formulate_transition_matrix_sparse(self, beta):
        """
        Formulate a sparse transition matrix for the game based on player rationality.

        :param beta: Rationality parameter for the players.
        :return: Sparse transition matrix.
        """
        pass

    @abstractmethod
    def formulate_binary_transition_matrix(self, beta):
        """
        Formulate a binary transition matrix for the game based on player rationality.

        :param beta: Rationality parameter for the players.
        :return: Binary transition matrix.
        """
        pass