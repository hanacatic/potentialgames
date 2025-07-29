import os
import pickle
import numpy as np
from functools import partial

from ...mechanism.game_setup import AbstractGameSetup
from ...utils import rng, logger


# TODO docs

class CoverageSetup(AbstractGameSetup):
    
    def __init__(self, no_resources, no_players, resource_values = None, success_weights = None, symmetric = False, use_noisy_utility= False, eta=None):
        """
        Initializes the CoverageGame class.

        Args:
            no_resources (int): Number of resources in the game.
            no_players (int): Number of players in the game.
            resource_values (list or None): Values of the resources. If None, all resources are initialized with equal values.
            noisy_utility (bool): Whether to use noisy utility functions. Defaults to False.
            symmetric (bool): Whether the game is symmetric. Defaults to False.
        """
        self.no_resources = no_resources
        self.no_players = no_players
        self.no_actions = self.no_resources
        self.use_noisy_utility = use_noisy_utility
        self.symmetric = symmetric
        
        if self.use_noisy_utility:
            self.eta = eta
        elif eta is not None:
            raise ValueError("Sorry, the eta is not null, but the noisy utility mode is not enabled!")
        else:
            self.eta = 0
            
        self.action_space = [np.arange(self.no_actions)]*self.no_players
        self.opponents_idx_map = [ np.delete(np.arange(self.no_players), player_id) for player_id in range(self.no_players) ]       
                        
        if (resource_values is None) or not (len(resource_values) == self.no_resources):
            self.resource_values = np.ones(self.no_resources)
        else:
            self.resource_values = resource_values
            
        self.resource_values = self.resource_values/np.sum(self.resource_values)
        
        if success_weights is None or len(success_weights) != self.no_resources:
            self.success_weights = np.arange(1, self.no_resources+1)/500 
        else:
            self.success_weights = np.array(success_weights)
            
        self.utility_functions = []
        self.modified_utility_functions = []
        
        for i in range(0, self.no_players):
            self.utility_functions.append(partial(self.utility_function, i))
            self.modified_utility_functions.append(partial(self.modified_utility_function, i))
        
        self.max_potential = 0
        self.min_potential = None
        self.second_min_potential = None
        
        self.improve_max_potential = None
        self.imporve_min_potential = None
        self.improve_second_min_potential = None
            
    def load_data(self, potential_file_path, delta_file_path):
        """
        Load the potential interval and delta from a files using pickle.
        
        Args:
            potential_file_path (str or Path): Path to the file containing potential values.
            delta_file_path (str or Path): Path to the file containing the delta value.
        """

        with open(potential_file_path, 'rb') as f:
            f.seek(0)
            potential = pickle.load(f)
            self.max_potential = potential[0]
            self.min_potential = potential[1]   
        with open(delta_file_path, 'rb') as f:
            f.seek(0)
            self.delta = pickle.load(f)/(self.max_potential - self.min_potential)
        
        self.improve_max_potential = self.max_potential
        self.improve_min_potential = self.min_potential
        self.improve_second_min_potential = self.min_potential+self.delta*(self.max_potential - self.min_potential)

    def save_data(self, potential_file_path, delta_file_path):
        """
        Saves the potential interval and delta to files using pickle.
        
        Args:
            potential_file_path (Path): The file path where the potential interval (max and min potential) will be saved.
            delta_file_path (Path): The file path where the delta value (difference between second minimum and minimum potential) will be saved.
        """
        
        try:
            potential_file_path.parent.mkdir(parents=True, exist_ok=True)      
            delta_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(delta_file_path, 'wb') as f:
                pickle.dump(self.improve_second_min_potential - self.improve_min_potential, f, pickle.HIGHEST_PROTOCOL)
            with open(potential_file_path, 'wb') as f:
                pickle.dump(np.array([self.improve_max_potential, self.improve_min_potential]), f, pickle.HIGHEST_PROTOCOL) 
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            
    def compute_potentials(self):
        """
        Estimates the potential function values for a game setup by iterating 
        through random action profiles. Computes the maximum potential, minimum 
        potential, and the second minimum potential, and calculates the delta 
        between the two smallest potentials. Saves the delta and potential values 
        to specified file paths.
        Args:
            None
        Returns:
            None
        """
        
        self.max_potential = 0
        self.min_potential = None
        self.second_min_potential = None
        
        logger.info("Starting Potential Function Estimation.")
        
        for i in range(10000):
            
            if i % 100 == 0:
                logger.info("   Currently on " + str(i) + "th iteration")
            
            action_profile = np.array([rng.integers(0, len(self.action_space[agent_id]), 1)[0] for agent_id in range(self.no_players)])
            
            potential = self.potential_function_est(action_profile)
            
            if potential > self.max_potential:
                self.max_potential = potential
            if self.min_potential is None:
               self.min_potential = potential 
               self.second_min_potential = potential    
            elif potential < self.min_potential:
                self.second_min_potential = self.min_potential
                self.min_potential = potential
            elif potential < self.second_min_potential:
                self.second_min_potential = potential

        self.delta = self.second_min_potential - self.min_potential
        
        self.improve_max_potential = self.max_potential
        self.improve_min_potential = self.min_potential
        self.improve_second_min_potential = self.second_min_potential
        
        logger.info("     Done.")


    def success_probability(self, agent_id, resource, action):
        """
        Calculate the success probability of an agent performing an action on a resource.
        Args:
            agent_id (int or None): The ID of the agent. If None, a default weight-based probability is returned.
            resource (int): The resource being acted upon.
            action (int): The action performed by the agent.
        Returns:
            float: The success probability of the action on the resource.
        """
        if not action == resource:
            return 0

        if agent_id is None:
            return self.success_weights[resource]
        else:
            return self.success_weights[resource] * (agent_id + 1) * 2 / (self.no_players - 1)
        
    def utility_function(self, agent_id, action, opponents_actions):
        """
        Computes the utility value for a given agent based on their action and the actions of their opponents.
        Args:
            agent_id (int): The ID of the agent for whom the utility is being calculated.
            action (int): The action taken by the agent.
            opponents_actions (list): A list of actions taken by the opponents.
        Returns:
            float: The computed utility value for the agent.
        """
        utility = 0.0
        
        for t in range(self.no_resources):
                                    
            if not action == t:
                continue
            
            prob = 1
            for i in range(self.no_players-1):
                prob = prob*(1-self.success_probability(self.opponents_idx_map[agent_id][i], t, opponents_actions[i]))
            
            utility = utility + prob*self.resource_values[t]*self.success_probability(agent_id, t, action)
        
        return utility
    
    def modified_utility_function(self, agent_id, action, opponents_actions):
        """
        Computes the modified utility for a given agent based on their action, 
        the actions of opponents, and the resource values.
        Args:
            agent_id (int): The ID of the agent for whom the utility is being calculated.
            action (int): The action taken by the agent.
            opponents_actions (list[int]): A list representing the actions of the opponents.
        Returns:
            float: The computed utility value for the given agent and action.
        """
        
        utility = 0.0
        
        for t in range(self.no_resources):
            
            if not action == t:
                continue
            
            prob = 1
            for i in range(self.no_actions):
                prob = prob*(1-self.success_probability(agent_id = None, resource = t, action = i))**opponents_actions[i]
            
            utility = utility + prob*self.resource_values[t]*self.success_probability(agent_id = None, resource=t, action = action)
        
        return utility
            
    def potential_function_est(self, profile):
        """
        Estimates the potential function value for a given strategy profile.
        Args:
            profile (list): A list representing the strategy profile of all players.
        Returns:
            float: The computed potential value for the given profile.
        """
        
        potential = 0.0
        
        for t in range(self.no_resources):
            
            prob = 1
            for j in range(self.no_players):
                prob = prob * (1-self.success_probability(j, t, profile[j]))
            
            potential = potential + (1-prob)*self.resource_values[t]
            
        if self.improve_max_potential is not None:
            if self.improve_max_potential < potential:
                self.improve_max_potential = potential
            elif self.improve_min_potential > potential:
                self.improve_second_min_potential = self.improve_min_potential
                self.improve_min_potential = potential
            elif self.improve_second_min_potential > potential:
                self.improve_second_min_potential = potential
            
        return potential
    
    def potential_function(self, profile):
        """
        Computes the normalized potential value for a given profile.
        Args:
            profile: The input profile for which the potential is calculated.
        Returns:
            float: The normalized potential value, scaled between 0 and 1.
        """
        
        potential =  self.potential_function_est(profile)
        return (potential - self.min_potential)/(self.max_potential-self.min_potential)
    
    def formulate_transition_matrix(self, beta): 
        """
        Generate transition matrix for the game with given rationality.
        """
        raise NotImplementedError("Transition matrix formulation is not implemented for CoverageSetup.")
    
    def formulate_transition_matrix_sparse(self, beta):
        """
        Generate a sparse transition matrix for the game with given rationality.
        """
        raise NotImplementedError("Sparse transition matrix formulation is not implemented for CoverageSetup.")
    
    def formulate_binary_transition_matrix(self, beta):
        """
        Generate a binary transition matrix for the game with given rationality.
        """
        raise NotImplementedError("Binary transition matrix formulation is not implemented for CoverageSetup.")