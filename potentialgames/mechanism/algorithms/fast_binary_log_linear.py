import numpy as np
from scipy.sparse import csr_matrix, csc_array

from potentialgames.mechanism.algorithms.abstract_algorithm import LearningAlgorithm


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