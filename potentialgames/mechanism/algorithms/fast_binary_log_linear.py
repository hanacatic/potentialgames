import numpy as np
from scipy.sparse import csr_matrix, csc_array

from ...mechanism.algorithms.abstract_algorithm import LearningAlgorithm
from ...utils import logger


class FastBinaryLogLinearAlgorithm(LearningAlgorithm):
    """
        Binary log-linear learning utilising the Markov Chain approach.
    """
    
    @classmethod
    def run(cls, game: "GameEngine", beta: float, scale_factor: int) -> None:
        """
            Binary log-linear learning utilising the Markov Chain approach.
            
        Args:
            beta (double): Player rationality.
            scale_factor (int): Scaling factor.
        """
        
        if not hasattr(game, "mu_matrix") or game.mu_matrix is None:
            game.mu_matrix = game.gameSetup.get_uniform_mu_matrix()
            logger.warning("Initial distribution 'mu_matrix' not set. Using default uniform distribution.")
        
        if beta is None:
            raise ValueError("Rationality parameter 'beta' must be provided.")

        if scale_factor <= 0:
            raise ValueError("Scale factor must be a positive number.")
        
        # Transition matrix of the Markov chain induced by log-linear learning for the game.
        P = game.gameSetup.formulate_binary_transition_matrix(beta)
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
        game.stationary_distribution = mu