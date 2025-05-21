import os
import pickle
import numpy as np
from functools import partial

from src.mechanism.game_setup import AbstractGameSetup


rng = np.random.default_rng()

# TODO clean up
# TODO docs


class CoverageSetup(AbstractGameSetup):
    
    def __init__(self, no_resources, no_players, resource_values = None, noisy_utility = False, symmetric = False):
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
        self.noisy_utility = noisy_utility
        self.eta = 0.0
        self.symmetric = symmetric
        
        self.opponents_idx_map = [ np.delete(np.arange(self.no_players), player_id) for player_id in range(self.no_players) ]

        self.utility_functions = []
        self.modified_utility_functions = []
        
        # self.action_sets = np.zeros((self.no_actions))
        self.action_space = [np.arange(self.no_actions)]*self.no_players
                        
        # self.weights = resource_values * 2 # np.arange(1, self.no_resources+1) / self.no_resources /2.5 # rng.uniform(0,0.4,self.no_resources)
        # self.weights = self.weights/np.sum(self.weights)
        # self.weights = np.zeros((self.no_actions, self.no_resources))
        
        
        # for i, a in enumerate(self.action_space[0]): # range(self.no_actions):
        #     print(i)
        #     temp = [int(d) for d in [*str(bin(a)[2:].zfill(self.no_resources))]]
            
        #     self.action_sets[i] = np.sum(temp)
            
        #     if i == 0:
        #         continue
        #     print(weights)
        #     print(temp)
        #     self.weights[i] = weights*temp
        #     self.weights[i] = self.weights[i]/np.sum(self.weights[i])
            
        # self.resources = np.zeros((self.no_resources), dtype=int)

        # for i in range(self.no_resources):
        #     self.resources[i] = 2**i
        
        if (resource_values is None) or not (len(resource_values) == self.no_resources):
            self.resource_values = np.ones(self.no_resources)
        else:
            self.resource_values = resource_values
            
        self.resource_values = self.resource_values/np.sum(self.resource_values)
        self.weights = np.arange(1, self.no_resources+1)/500 #self.no_players #np.flip(self.resource_values)
        # print(self.resource_values)
        for i in range(0, self.no_players):
            self.utility_functions.append(partial(self.utility_function, i))
            self.modified_utility_functions.append(partial(self.modified_utility_function, i))
        
        root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", "CoverageProblem")

        self.file_path_delta = os.path.join(root, "estimated_delta" + str(self.no_players) + "_" + str(self.no_actions) + "_" + str(self.symmetric) + ".pckl")
        self.file_path_potential = os.path.join(root, "estimated_potential" + str(self.no_players) + "_" + str(self.no_actions) + str(self.symmetric) + ".pckl")
        
        if os.path.exists(self.file_path_potential):
            with open(self.file_path_potential, 'rb') as f:
                potential = pickle.load(f)
                self.max_potential = potential[0]
                self.min_potential = potential[1]
            with open(self.file_path_delta, 'rb') as f:
                self.delta = pickle.load(f)/(self.max_potential - self.min_potential)

            self.improve_max_potential = self.max_potential
            self.improve_min_potential = self.min_potential
            self.improve_second_min_potential = self.min_potential+self.delta*(self.max_potential - self.min_potential)
        
        else:
            self.improve_max_potential = None
            self.imporve_min_potential = None
            self.improve_second_min_potential = None
            
            self.compute_potentials()

            
        print("Potential")
        print(self.max_potential)
        print(self.min_potential)
        print(self.delta)
        
    def __del__(self):
        try:
            print(self.file_path_delta)
            # Only update if improved
            if self.improve_max_potential is not None and self.improve_second_min_potential - self.improve_min_potential > 0:
                with open(self.file_path_delta, 'wb') as f:
                    pickle.dump(self.improve_second_min_potential - self.improve_min_potential, f, pickle.HIGHEST_PROTOCOL)
            
            print(self.file_path_potential)
            if self.improve_max_potential is not None:
                with open(self.file_path_potential, 'wb') as f:
                    pickle.dump(np.array([self.improve_max_potential, self.improve_min_potential]), f, pickle.HIGHEST_PROTOCOL)
        except:
            print("Error occurred")
            
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
        
        print("Starting Potential Function Estimation.")
        
        for i in range(10000):
            
            if i % 100 == 0:
                print("Currently on " + str(i) + "th iteration")
            
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
        
        with open(self.file_path_delta, 'wb') as f:
            pickle.dump(self.delta, f, pickle.HIGHEST_PROTOCOL)
            
        with open(self.file_path_potential, 'wb') as f:
            pickle.dump(np.array([self.max_potential, self.min_potential]), f, pickle.HIGHEST_PROTOCOL)

        print("     Done.")


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
            return self.weights[resource]
        else:
            return self.weights[resource] * (agent_id + 1) * 2 / (self.no_players - 1)
        if not action == resource:
            return 0
        
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
                # print(self.success_probability(self.opponents_idx_map[agent_id][i], t, opponents_actions[i]))
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