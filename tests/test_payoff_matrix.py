import numpy as np
import pytest

from potentialgames.mechanism.game_setup.payoff_matrix import PayoffMatrix  


def test_generate_random_matrix():
    
    no_players = 2
    no_actions = 6
    
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    symmetric = False
    
    payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric)
    payoff_matrix.generate_random_matrix()
    
    payoff_matrix.plot()
    
    assert payoff_matrix.matrix[1,1] == 1
    assert payoff_matrix.matrix[4,4] == 1 - delta
    assert payoff_matrix.matrix[0,0] < 1 - delta
    assert payoff_matrix.matrix[2,2] < 1 - delta

def test_generate_plateau_matrix():
    
    no_players = 2
    no_actions = 6
    
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    symmetric = False
    
    payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric)
    
    payoff_matrix.generate_plateau_matrix()
    
    payoff_matrix.plot()
    
    assert payoff_matrix.matrix[1,1] == 1
    assert payoff_matrix.matrix[4,4] == 1 - delta
    assert np.all(payoff_matrix.matrix[3:6, 3:6] <= 1 - delta)

def test_generate_easy_matrix():
    
    no_players = 2
    no_actions = 6
    
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    symmetric = False
    
    payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric)
    
    payoff_matrix.generate_easy_matrix()
    
    payoff_matrix.plot()
    
    assert payoff_matrix.matrix[1,1] == 1
    assert payoff_matrix.matrix[4,4] == 1 - delta
    assert np.all(payoff_matrix.matrix[3:6, 3:6] <= 1 - delta)
    
def test_regenerate_method_not_implemented():
    """
    Test that the regenerate method raises a NotImplementedError.
    """
    no_players = 2
    no_actions = 6
    
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    symmetric = False
    
    payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric)
    
    with pytest.raises(ValueError, match="Method not found in PayoffMatrix."):
        payoff_matrix.regenerate(method="non_existent_method")
        
def test_regenerate_with_callable():
    """
    Test that the regenerate method can accept a callable.
    """
    no_players = 2
    no_actions = 6
    
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    symmetric = False
    
    payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric)
    
    def custom_method(matrix):
        matrix.generate_random_matrix()
    
    payoff_matrix.regenerate(method=custom_method)
    
    payoff_matrix.plot()

def test_regenerate_with_string_method():
    """
    Test that the regenerate method can accept a string naming a method.
    """
    no_players = 2
    no_actions = 6
    
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    symmetric = False
    
    payoff_matrix = PayoffMatrix(no_players, no_actions, firstNE, secondNE, delta, symmetric)
    
    payoff_matrix.regenerate(method="generate_plateau_matrix", no_players=no_players, no_actions=no_actions*2, firstNE=firstNE, secondNE=secondNE, delta=delta, symmetric=symmetric, plateau_size=2)
    
    payoff_matrix.plot()