import numpy as np
import pytest

from potentialgames.mechanism.game_setup.coverage import CoverageSetup


def test_initialization():
    
    setup = CoverageSetup(no_resources=3, no_players=2)
    assert setup.no_resources == 3
    assert setup.no_players == 2
    assert setup.no_actions == 3
    assert setup.symmetric in [True, False]

def test_success_probability():
    
    setup = CoverageSetup(no_resources=3, no_players=2)
    
    assert setup.success_probability(None, 0, 1) == 0
    assert setup.success_probability(None, 1, 1) > 0
    assert setup.success_probability(0, 1, 1) > 0
    
def test_eta():
    
    setup = CoverageSetup(no_resources=3, no_players=2, use_noisy_utility=True, eta=0.5)
    
    assert setup.eta == 0.5
    with pytest.raises(ValueError):
        CoverageSetup(no_resources=3, no_players=2, use_noisy_utility=False, eta=0.5)

