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
    
    mean_1 = game.expected_value.T
    assert len(game.expected_value) == game.max_iter

    eps = 0.1
    beta = game.compute_beta(eps)
    
    game.play(beta=beta, scale_factor=1)    
    
    mean_2 = game.expected_value.T
    
    one_minus_eps = [1 - eps] * game.max_iter
    plot = compare_lines(
        [mean_1.T, mean_2.T],
        [r"$\beta=0.5$", r"$\beta=\beta$"],
        [one_minus_eps],
        [r"$\phi(a^*)-\epsilon$"],
        title="Expected Value Comparison",
        xlabel="T",
        save=False
    )

def test_fast_log_linear_algorithm():
    
    game = make_simple_game_fast()
    
    game.play(beta=0.5, scale_factor=1)
    
    mean_1 = game.expected_value.T
    assert len(game.expected_value) == game.max_iter

    eps = 0.1
    beta = game.compute_beta(eps)
    
    game.play(beta=beta, scale_factor=1)    
    
    mean_2 = game.expected_value.T
    
    one_minus_eps = [1 - eps] * game.max_iter
    plot = compare_lines(
        [mean_1.T, mean_2.T],
        [r"$\beta=0.5$", r"$\beta=\beta$"],
        [one_minus_eps],
        [r"$\phi(a^*)-\epsilon$"],
        title="Expected Value Comparison",
        xlabel="T",
        save=False
    )
