import pytest
import numpy as np

from pathlib import Path

from potentialgames.mechanism.game_setup import CoverageSetup 
from potentialgames.mechanism.game_engine import GameEngine


def make_simple_game():
    
    no_players = 200
    
    setup = CoverageSetup(no_resources = 10, no_players = no_players, resource_values = [0.05, 0.15, 0.14, 0.1, 0.01, 0.1, 0.11, 0.2, 0.09, 0.05], symmetric = True)
    
    try:
        repo_root  = Path(__file__).resolve().parents[1] 
        data_path = repo_root / "data" / "Coverage"

        potentials_path = data_path / f"coverage_potentials_{no_players}_asymmetric.pckl"
        delta_path = data_path / f"coverage_delta_{no_players}_asymmetric.pckl"
        
        setup.load_data(potentials_path, delta_path)
    except FileNotFoundError:
        logger.info(f"Game setup for {no_players} players not found, computing potentials.")
        setup.compute_potentials()
        setup.save_data(potentials_path, delta_path)

            
    game = GameEngine(setup, algorithm="modified_log_linear", max_iter=1e4)
    
    return game

def test_modified_log_linear_algorithm():
        
    game = make_simple_game()
    
    initial_action_profile =  np.array([0]*game.no_players) #rng.integers(0, 5, size = game.gameSetup.no_players)

    game.play(initial_action_profile = initial_action_profile, beta=0.5)

    game.plot()
    
    mean_1 = np.mean(game.potentials_history)
    assert len(game.potentials_history) == game.max_iter
    
    eps = 0.1
    beta = game.compute_beta(eps)
    
    game.play(initial_action_profile = initial_action_profile, beta=beta)
    game.plot()
    
    mean_2 = np.mean(game.potentials_history)
    assert mean_2 > mean_1, f"Expected mean {mean_2} to be greater than {mean_1} after playing with beta={beta}"
    
    
def test_missing_beta():
    
    game = make_simple_game()
    
    initial_action_profile =  np.array([0]*game.no_players) #rng.integers(0, 5, size = game.gameSetup.no_players)

    with pytest.raises(ValueError, match="Rationality parameter 'beta' must be provided."):
        game.play(initial_action_profile=initial_action_profile, beta=None, gamma=0)
