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
    
    game = GameEngine(setup, algorithm="hedge", max_iter=1e4)
    game.set_initial_action_profile(np.array([0, 0]))
    
    return game

def test_hedge_algorithm():
    
    game = make_simple_game()
    
    game.play()
    game.plot()
    
    assert len(game.potentials_history) == game.max_iter
    
def test_hedge_algorithm_convergence():
    
    game = make_simple_game()
    
    game.max_iter = 10000
    game.play()
    
    potentials = game.potentials_history

    first_half_var = np.var(potentials[:len(potentials)//2])
    second_half_var = np.var(potentials[len(potentials)//2:])
    
    assert first_half_var > second_half_var, f"Expected variance to decrease, got {first_half_var} and {second_half_var}"
    