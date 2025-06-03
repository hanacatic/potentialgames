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
    
    game = GameEngine(setup, algorithm="log_linear", max_iter=10)
    game.set_initial_action_profile(np.array([0, 0]))
    
    return game

def test_log_linear_algorithm():
    game = make_simple_game()
    
    game.play(beta=0.5, gamma=0)
    
    assert len(game.potentials_history) == game.max_iter