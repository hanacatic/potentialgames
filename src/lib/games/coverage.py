import os
import pickle
import numpy as np
from functools import partial

rng = np.random.default_rng()

# TODO clean up
# TODO docs

class CoverageGame:
    
    def __init__(self, no_resources, no_players, resource_values = None, noisy_utility = False, symmetric = False):
        
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
                
        if not action == resource:
            return 0
        
        # return 1/(np.mod(self.action_sets[action], 7)**(2*np.mod(resource, 2))+3)
        
        # return 1/(action*self.action_sets[action]+1)
        
        # return 1/(np.sin(self.action_sets[action]/10)+2)
    
        # prob = 1/(0.1*action**2 - action*resource + 0.2*resource**2 + 2) #* agent_id / self.no_players


        # prob = 0.8*action/self.no_actions - 0.5*1 / (resource+2) - 0.1*1/(self.action_sets[action] + 2)

        # prob = 1  - 1/(self.action_sets[action] + 2) - 1 / (resource+2) - 0.1*action / (self.no_actions)#* agent_id / self.no_players
        # print(prob)
        # prob = prob*agent_id*2/self.no_players/(self.no_players-1)
        # return 0.5#np.clip(prob * self.resource_values[resource], 0, 1)
        # return self.weights[action][resource]/(self.action_sets[action] + 1)#*(agent_id+1)*2/(self.no_players-1)
        # return self.resource_values[resource]/(self.action_sets[action] + 1)#*(agent_id+1)*2/(self.no_players-1)
        if agent_id is None:
            return self.weights[resource] 
        else:
            return self.weights[resource]*(agent_id+1)*2/(self.no_players-1)
    
    def utility_function(self, agent_id, action, opponents_actions):
                
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
        potential =  self.potential_function_est(profile)
        return (potential - self.min_potential)/(self.max_potential-self.min_potential)