import pytest
import numpy as np

from potentialgames.mechanism.game_setup import IdenticalInterestSetup
from potentialgames.mechanism.game_engine import GameEngine
from potentialgames.utils import compare_lines

def make_simple_game_fast():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    
    delta = 0.1
    
    setup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta)
    game = GameEngine(setup, algorithm="fast_log_linear", max_iter=100)
    game.set_initial_action_profile(np.array([0, 0]))

    # Using default unifrom mu matrix
    
    return game

def test_fast_log_linear_algorithm():
    
    game = make_simple_game_fast()
    
    game.play(beta=0.5, scale_factor=1)
    
    mean_potential_history = np.zeros((2, game.max_iter))
    
    mean_potential_history[0] = game.expected_value.T

    assert len(game.expected_value) == game.max_iter

    eps = 0.1
    beta = game.compute_beta(eps)
    
    game.play(beta=beta, scale_factor=1)    
    
    mean_potential_history[1] = game.expected_value.T    
    
    one_minus_eps = [1 - eps] * game.max_iter
    plot = compare_lines(
        mean_potential_history,
        lines_legend=[r"$\beta=0.5$", r"$\beta=\beta$"],
        special_lines = [one_minus_eps],
        special_lines_legend = [r"$\phi(a^*)-\epsilon$"],
        title="Expected Value Comparison",
        xlabel="T",
        save=False
    )
    
def test_stationary_distribution():
    
    game = make_simple_game_fast()
    
    eps = 0.1
    beta = game.compute_beta(eps)
    game.play(beta=beta, scale_factor=1)    
    
    stationary_distribution = game.stationary_distribution
    assert stationary_distribution is not None, "Stationary distribution should not be None"
    
    # Check if the stationary distribution sums to 1
    assert np.isclose(np.sum(stationary_distribution), 1), "Stationary distribution does not sum to 1"
    
    
def test_missing_params():
    
    game = make_simple_game_fast()
    
    with pytest.raises(ValueError, match="Rationality parameter 'beta' must be provided."):
        game.play(beta=None, scale_factor=1)
        
    with pytest.raises(ValueError, match="Scale factor must be a positive number."):
        game.play(beta=0.5, scale_factor=-1)

def test_scale_factor():
    
    game = make_simple_game_fast()
    
    eps = 0.1
    beta = game.compute_beta(eps)
    
    max_iter = 10000
    results = []

    for scale_factor in [1, 2, 5]:
        
        game.reset_game()
        game.set_uniform_mu_matrix()
        game.play(beta=beta, scale_factor=scale_factor, max_iter = max_iter//scale_factor)
        
        results.append(game.expected_value[-1].copy())
    
    assert all(np.isclose(results[0], r) for r in results), "Expected values for different scale factors should be the same"


