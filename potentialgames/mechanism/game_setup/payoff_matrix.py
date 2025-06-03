import numpy as np

from potentialgames.utils import make_symmetric_nd, plot_matrix

class PayoffMatrix:
    
    def __init__(self, no_players, no_actions, firstNE, secondNE, delta, symmetric):
        
        self.no_players = no_players
        self.no_actions = no_actions
        self.firstNE = firstNE
        self.secondNE = secondNE
        self.delta = delta 
        self.symmetric = symmetric
        
    @property
    def matrix(self):
        return self._payoff_player_1

    def generate_random_matrix(self, no_players=None, no_actions=None, firstNE=None, secondNE=None, delta=None, symmetric=None):
        
        if no_players is not None:
            self.no_players = no_players
        if no_actions is not None:
            self.no_actions = no_actions
        if firstNE is not None:
            self.firstNE = firstNE
        if secondNE is not None:
            self.secondNE = secondNE
        if delta is not None:
            self.delta = delta
        if symmetric is not None:
            self.symmetric = symmetric
            
        self._payoff_player_1 = np.random.uniform(0.0, 1 - self.delta, size = [self.no_actions] * self.no_players)
        
        self._payoff_player_1[tuple(self.firstNE)] = 1
        self._payoff_player_1[tuple(self.secondNE)] = 1 - self.delta
        
        if self.symmetric: 
            self._payoff_player_1 = make_symmetric_nd(self._payoff_player_1)
            
    def regenerate(self, method=None, **kwargs):
        """
        Regenerate the matrix, possibly with new properties.
        """
        if method is None:
            self.generate_random_matrix(**kwargs)
        elif callable(method):
            method(self, **kwargs)
        elif isinstance(method, str):
            if hasattr(self, method):
                getattr(self, method)(**kwargs)
            else:
                raise ValueError(f"Method {method} not found in PayoffMatrix.")
        else:
            raise ValueError("method must be None, a callable, or a string naming a method.")
        
    def plot(self):
        """
        Plot the payoff matrix.
        """
        plot_matrix(self.matrix, xlabel="Player 2", ylabel="Player 1", title="Payoff Matrix")