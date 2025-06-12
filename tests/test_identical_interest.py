import numpy as np
import pytest

from potentialgames.mechanism.game_setup.identical_interest import IdenticalInterestSetup
from potentialgames.utils.plot import *


def test_initialization():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    
    gameSetup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta)
    
    assert gameSetup.payoff.no_players == no_players
    assert gameSetup.payoff.no_actions == len(action_space)
    assert gameSetup.payoff.firstNE == firstNE
    assert gameSetup.payoff.secondNE == secondNE
    assert gameSetup.payoff.delta == delta
    assert gameSetup.payoff.matrix.shape == (6, 6)
    
def test_payoff_matrix():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    
    gameSetup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta)
    
    gameSetup.payoff.plot()
    
def test_potential_function():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    
    gameSetup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta)

    assert np.isclose(gameSetup.potential_function(firstNE), 1.0)
    assert np.isclose(gameSetup.potential_function(secondNE), 1.0 - delta)

    random_profile = [0, 1]
    if random_profile != firstNE and random_profile != secondNE:
        val = gameSetup.potential_function(random_profile)
        assert 0 <= val < 1 - delta, f"Potential value {val} out of expected range for random profile {random_profile}"
        
def test_utility_function():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    
    gameSetup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta)

    for i in action_space:
        for j in action_space:
            
            u_0 = gameSetup.utility_function(0, i, [j])            
            u_1 = gameSetup.utility_function(1, j, [i])
            
            potential_value = gameSetup.potential_function([i, j])
            
            assert np.isclose(u_0, potential_value), f"Utility for player 0 with action {i} and opponents' action {j} does not match potential value {potential_value}"
            assert np.isclose(u_1, potential_value), f"Utility for player 1 with action {j} and opponents' action {i} does not match potential value {potential_value}"

def test_utility_function_noisy():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    
    gameSetup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta, use_noisy_utility=True)

    for i in action_space:
        for j in action_space:
            
            u_0 = gameSetup.utility_function(0, i, [j])
            u_1 = gameSetup.utility_function(1, j, [i])
            
            potential_value = gameSetup.potential_function([i, j])
            
            assert np.isclose(u_0, potential_value), f"Utility for player 0 with action {i} and opponents' action {j} does not match potential value {potential_value}"
            assert np.isclose(u_1, potential_value), f"Utility for player 1 with action {j} and opponents' action {i} does not match potential value {potential_value}"
            
def test_utility_function_noisy_eta():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    
    eta = 0.5
    
    gameSetup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta, use_noisy_utility=True, eta=eta)

    for i in action_space:
        for j in action_space:
            
            u_0 = gameSetup.utility_function(0, i, [j])
            u_1 = gameSetup.utility_function(1, j, [i])
            
            potential_value = gameSetup.potential_function([i, j])
            
            assert np.isclose(u_0, potential_value), f"Utility for player 0 with action {i} and opponents' action {j} does not match potential value {potential_value}"
            assert np.isclose(u_1, potential_value), f"Utility for player 1 with action {j} and opponents' action {i} does not match potential value {potential_value}"
            
def test_utility_function_eta_nonzero():
        
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
        
    eta = 0.5
                
    with pytest.raises(ValueError, match="Sorry, the eta is not null, but the noisy utility mode is not enabled!"):
        IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta, eta=eta)
            

def test_symmetric_setup():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    
    gameSetup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta, symmetric=True)
    
    assert gameSetup.symmetric is True
    
    for i in action_space:
        for j in action_space:
            
            phi_ij = gameSetup.potential_function([i, j])
            phi_ji = gameSetup.potential_function([j, i])
            
            assert np.isclose(phi_ij, phi_ji), f"Potential function is not symmetric for actions {i} and {j}: {phi_ij} != {phi_ji}"

def test_transition_matrix():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    
    delta = 0.1
    beta = 10
    
    gameSetup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta, symmetric=True)
   
    transition_matrix = gameSetup.formulate_transition_matrix(10)
    
    plot_matrix(transition_matrix, xlabel="Player 2", ylabel="Player 1", title="Transition Matrix")
    
    assert (transition_matrix[7, 7] > 0.5), "Transition matrix diagonal element is not greater than 0.5"
    assert np.allclose(np.sum(transition_matrix, axis=1), 1.0), "Each row of the transition matrix should sum to 1"

def test_sparse_transition_matrix():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    
    delta = 0.1
    beta = 10
    
    gameSetup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta, symmetric=True)
   
    transition_matrix = gameSetup.formulate_transition_matrix(beta)
    sparse_transition_matrix = gameSetup.formulate_transition_matrix_sparse(beta).toarray()
    
    assert np.allclose(transition_matrix, sparse_transition_matrix), "Sparse and dense transition matrices differ"

def test_binary_transition_matrix():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    
    delta = 0.1
    beta = 10
    
    gameSetup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta, symmetric=True)
   
    binary_transition_matrix = gameSetup.formulate_binary_transition_matrix(beta)
        
    plot_matrix(binary_transition_matrix, xlabel="Player 2", ylabel="Player 1", title="Transition Matrix")
    
    assert (binary_transition_matrix[7, 7] > 0.5), "Transition matrix diagonal element is not greater than 0.5"
    assert np.allclose(np.sum(binary_transition_matrix, axis=1), 1.0), "Each row of the transition matrix should sum to 1"
