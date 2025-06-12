import pytest
import numpy as np

from potentialgames.mechanism.game_setup import IdenticalInterestSetup 
from potentialgames.mechanism.game_engine import GameEngine

def make_simple_game():
    
    action_space = np.arange(6)
    no_players = 2
    firstNE = [1, 1]
    secondNE = [4, 4]
    delta = 0.1
    
    setup = IdenticalInterestSetup(action_space, no_players, firstNE, secondNE, delta)
    
    game = GameEngine(setup, algorithm="log_linear", max_iter=1e4)
    game.set_initial_action_profile(np.array([0, 0]))
    
    return game

def test_log_linear_algorithm():
    
    game = make_simple_game()
    
    game.play(beta=0.5, gamma=0)
    game.plot()
    
    mean_1 = np.mean(game.potentials_history)
    assert len(game.potentials_history) == game.max_iter
    
    eps = 0.1
    
    beta = game.compute_beta(eps)
    game.play(beta=beta, gamma=0)
    game.plot()
    
    mean_2 = np.mean(game.potentials_history)
    assert mean_2 > mean_1, f"Expected mean {mean_2} to be greater than {mean_1} after playing with beta={beta}"
    
def test_missing_beta():
    
    game = make_simple_game()
    
    with pytest.raises(ValueError, match="Rationality parameter 'beta' must be provided."):
        game.play(beta=None, gamma=0)
    

def test_log_linear_algorithm_with_gamma():
    
    game1 = make_simple_game()
    game2 = make_simple_game()
    
    eps = 0.1
    beta1 = game1.compute_beta(eps)
    beta2 = game2.compute_beta(eps)
    game1.play(beta=beta1, gamma=0.0)
    game2.play(beta=beta2, gamma=0.1)
    
    game1.plot()
    game2.plot()
    # The histories should differ if gamma has an effect
    assert not np.allclose(game1.potentials_history, game2.potentials_history), \
        "Changing gamma should affect the potential history"