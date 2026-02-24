import numpy as np
from pathlib import Path
import pickle

from ...utils import make_symmetric_nd, plot_matrix


class PayoffMatrix:
    
    """A class to represent a payoff matrix for N-player games with specific properties."""
    
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
        """
        Generate a random payoff matrix for N players with specified properties.
        Args:
            no_players (int, optional): Number of players in the game. If not provided, uses the instance's value.
            no_actions (int, optional): Number of actions available to each player. If not provided, uses the instance's value.
            firstNE (list or tuple, optional): Indices representing the first Nash Equilibrium (NE) in the payoff matrix.
            secondNE (list or tuple, optional): Indices representing the second Nash Equilibrium (NE) in the payoff matrix.
            delta (float, optional): Difference in payoff between the first and second NE. Used to set the payoff for the second NE.
            symmetric (bool, optional): If True, ensures the generated payoff matrix is symmetric.
        Returns:
            None: The method updates the instance's payoff matrix in place.
     
        """
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

    def generate_plateau_matrix(self, no_players=None, no_actions=None, firstNE=None, secondNE=None, delta=None, symmetric=None, plateau_size=3):
        """
        Generate a two-plateau payoff matrix for N players.
        Args:
            no_players (int, optional): Number of players in the game. If not provided, uses the instance's value.
            no_actions (int, optional): Number of actions available to each player. If not provided, uses the instance's value.
            firstNE (list or tuple, optional): Indices specifying the first Nash Equilibrium. If not provided, uses the instance's value.
            secondNE (list or tuple, optional): Indices specifying the second Nash Equilibrium. If not provided, uses the instance's value.
            delta (float, optional): Value to decrease payoffs for the second NE and plateaus. If not provided, uses the instance's value.
            symmetric (bool, optional): Whether to make the payoff matrix symmetric. If not provided, uses the instance's value.
            plateau_size (int, optional): Size of each plateau (must be between 1 and half the number of actions). Default is 3.
        Raises:
            ValueError: If plateau_size is less than 1 or greater than half the number of actions.
        """
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

        if plateau_size < 1 or plateau_size > self.no_actions // 2:
            raise ValueError("The size of the plateau must be between 1 and half the number of actions.")
        
        shape = [self.no_actions] * self.no_players
        payoff = np.random.random(size=shape) * 0.275 + 0.625 - self.delta

        first_slice = tuple(slice(0, plateau_size) for _ in range(self.no_players))
        second_slice = tuple(slice(self.no_actions - plateau_size, self.no_actions) for _ in range(self.no_players))

        payoff[first_slice] = np.random.random(size=[plateau_size]*self.no_players) * 0.15 + 0.85 - self.delta
        payoff[second_slice] = np.random.random(size=[plateau_size]*self.no_players) * 0.15 + 0.85 - self.delta
        
        payoff[tuple(self.firstNE)] = 1
        payoff[tuple(self.secondNE)] = 1 - self.delta
        
        if self.symmetric:
            payoff = make_symmetric_nd(payoff)
            
        self._payoff_player_1 = payoff
        
    def generate_easy_matrix(self, no_players=None, no_actions=None, firstNE=None, secondNE=None, delta=None, symmetric=None, plateau_size=3):
        """
        Generate a two-plateau payoff matrix for N players.
        Args:
            no_players (int, optional): Number of players in the game. If not provided, uses the instance's value.
            no_actions (int, optional): Number of actions available to each player. If not provided, uses the instance's value.
            firstNE (list or tuple, optional): Indices specifying the first Nash Equilibrium. If not provided, uses the instance's value.
            secondNE (list or tuple, optional): Indices specifying the second Nash Equilibrium. If not provided, uses the instance's value.
            delta (float, optional): Value to decrease payoffs for the second NE and plateaus. If not provided, uses the instance's value.
            symmetric (bool, optional): Whether to make the payoff matrix symmetric. If not provided, uses the instance's value.
            plateau_size (int, optional): Size of each plateau (must be between 1 and half the number of actions). Default is 3.
        Raises:
            ValueError: If plateau_size is less than 1 or greater than half the number of actions.
        """
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
             
        shape = [self.no_actions] * self.no_players
        payoff = np.random.random(size=shape) * 0.275 + 0.625 - self.delta

        first_slice = tuple(slice(0, plateau_size) for _ in range(self.no_players))

        payoff[first_slice] = np.random.random(size=[plateau_size]*self.no_players) * 0.15 + 0.85 - self.delta
        
        payoff[tuple(self.firstNE)] = 1
        payoff[tuple(self.secondNE)] = 1 - self.delta
        
        if self.symmetric:
            payoff = make_symmetric_nd(payoff)
            
        self._payoff_player_1 = payoff
        
    def regenerate(self, method=None, **kwargs):
        """
        Regenerate the matrix, possibly with new properties.
        Args:
            method (callable or str, optional): A method to call for regeneration. If None, generates a random matrix.
            **kwargs: Additional keyword arguments to pass to the method.
        """
        if method is None:
            self.generate_random_matrix(**kwargs)
        elif callable(method):
            method(self, **kwargs)
        elif isinstance(method, str):
            if hasattr(self, method):
                getattr(self, method)(**kwargs)
            else:
                raise ValueError("Method not found in PayoffMatrix.")
        else:
            raise ValueError("Method must be None, a callable, or a string naming a method.")
    
    def save(self, file_path):  
        """
        Save the payoff matrix to a file using pickle.
        Args:
            file_path (str or Path): The path where the matrix will be saved.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)      
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, file_path):
        """
        Load a payoff matrix from a file using pickle.
        Args:
            file_path (str or Path): The path from which the matrix will be loaded.
        Returns:
            PayoffMatrix: An instance of PayoffMatrix loaded from the file.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)       
         
    def plot(self):
        """
        Plot the payoff matrix.
        """
        plot_matrix(self.matrix, xlabel="Player 2", ylabel="Player 1", title="Payoff Matrix")