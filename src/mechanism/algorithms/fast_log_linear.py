from mechanism.algorithms.abstract_algorithm import LearningAlgorithm
import numpy as np
from scipy.sparse import csr_matrix, csc_array

class FastLogLinear(LearningAlgorithm):
    """
        Log-linear learning utilising the Markov Chain approach.
    """
    
    @classmethod
    def run(cls, game: "GameEngine", beta: float, scale_factor: int) -> None:
        """
            Log-linear learning utilising the Markov Chain approach.
        Args:
            beta (double): Player rationality.
            scale_factor (int): Scaling factor.
        """
        if game.gameSetup.no_players == 2:
            cls.log_linear_fast_impl(game, beta, scale_factor)
        else:
            cls.log_linear_fast_sparse(game, beta, scale_factor)

    @classmethod
    def log_linear_fast_impl(cls, game: "GameEngine", beta: float, scale_factor: int) -> None:
        """
        Computes the expected value trajectory and stationary distribution of the Markov chain
        induced by log-linear learning for the game, using a fast implementation with matrix powers.
        Args:
            beta (float): The inverse temperature parameter controlling the randomness of the log-linear learning.
            scale_factor (int): The number of steps to model at once by raising the transition matrix to this power.
        """
        # Transition matrix of the Markov chain induced by log-linear learning for the game.
        P = game.gameSetup.formulate_transition_matrix(beta)
        mu0 = game.mu_matrix.copy()
        
        game.expected_value = np.zeros((int(game.max_iter), 1))
        
        # Transition matrix models scale_factor steps at once
        P = np.linalg.matrix_power(P, scale_factor)
        
        # Main loop
        for i in range(game.max_iter):
            mu = mu0 @ P
            mu0 = mu
            game.expected_value[i] = mu @ game.gameSetup.potential_vec
        
        game.expected_value = game.expected_value
        game.stationary = mu

    @classmethod
    def log_linear_fast_sparse(cls, game: "GameEngine", beta: float, scale_factor: int) -> None:
        """
        Computes the expected value trajectory and stationary distribution of the Markov chain
        induced by log-linear learning for the game, using a sparse matrix implementation.
        Args:
            beta (float): Player rationality.
            scale_factor (int): Scale factor.
        """
        # Transition matrix of the Markov chain induced by log-linear learning for the game.
        P = game.gameSetup.formulate_transition_matrix_sparse(beta)
        mu0 = csc_array(game.mu_matrix)
        
        game.expected_value = np.zeros((int(game.max_iter), 1))
        game.expected_value = csr_matrix(game.expected_value)
        
        # Transition matrix modeling multiple transition steps at once has reduced sparsity and has negative impact on the computation time, thus it is not utilised.
        if scale_factor != 1:
            raise ValueError("For sparse implementation, the scale factor must be 1.")

        # Main loop
        for i in range(game.max_iter):
            mu = mu0 @ P          
            mu0 = mu
            game.expected_value[i] = mu @ game.gameSetup.potential_vec

        game.expected_value = game.expected_value.todense()
        game.stationary = mu.todense()

