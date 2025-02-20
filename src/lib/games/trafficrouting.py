import os
import numpy as np 
import pandas as pd
import pickle
import networkx as nx
from itertools import islice
from functools import partial

rng = np.random.default_rng()

class CongestionGame:
    """
        Class representing the Congestion game.
        
        Includes information on the number of players and actions, utility functions and potential function.
    """
    
    def __init__(self, network = "SiouxFalls", no_actions = 5, noisy_utility = False, modified = False, modified_no_players = None):
        """
        

        Args:
            network (str, optional): Network name. Defaults to "SiouxFalls".
            no_actions (int, optional): Number of actions. Defaults to 5.
            noisy_utility (bool, optional): Use noisy utilities. Defaults to False.
            modified (bool, optional): Use modified network. Defaults to False.
            modified_no_players (_type_, optional): Number of players if modified network. Defaults to None.
        """
        self.no_actions = no_actions
        self.modified = modified
        self.modified_no_players = modified_no_players
        self.noisy_utility = noisy_utility
        
        self.load_from_tntp(network)
        self.build_network()
    
        self.opponents_idx_map = [ np.delete(np.arange(self.no_players), player_id) for player_id in range(self.no_players) ]
        
        self.load_precomputed_data(network)
                        
        self.utility_functions = []
        self.modified_utility_functions = []
        
        for i in range(0, self.no_players):
            self.utility_functions.append(partial(self.utility_function, i))
            self.modified_utility_functions.append(partial(self.modified_utility_function, i))

    def __del__(self):
        """
            Congestion game destructor. Saves the improved empirical information on minimum and maximum potential.
        """
        
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
            
            
    def load_from_tntp(self, network):
        """
            Load network.

        Args:
            network (str): Name of network.
        """
        
        # Based on _scripts https://github.com/bstabler/TransportationNetworks
        
        root = os.path.dirname(os.path.abspath('.'))
        netfile = os.path.join(root, "potentialgames_ws", "TransportationNetworks", network, network + '_net.tntp')
        net = pd.read_csv(netfile, skiprows=8, sep='\t')
        trimmed= [s.strip().lower() for s in net.columns]
        net.columns = trimmed

        # And drop the silly first and last columns
        net.drop(['~', ';'], axis=1, inplace=True)
        
        self.net = net.to_numpy() # initial node, terminal node, capacity, length, free flow time, b, power, speed, toll, link type
        
        initial_nodes = self.net[:, 0] - 1
        terminal_nodes = self.net[:, 1] - 1
        
        self.nodes = (np.unique(np.stack([initial_nodes, terminal_nodes], 0))).astype(int)
        self.edges = np.array([initial_nodes, terminal_nodes]).astype(int)
        self.capacities = self.net[:, 2] / 100 
        self.lengths = self.net[:, 3]
        self.free_flows = self.net[:, 4]
        self.b = self.net[:, 5]
        self.powers = self.net[:, 6]
        
        tripsfile = os.path.join(root, "potentialgames_ws", "TransportationNetworks", network, network + '_trips.tntp')
        
        f = open(tripsfile, 'r')
        all_rows = f.read()
        blocks = all_rows.split('Origin')[1:]
        matrix = {}
        
        for k in range(len(blocks)): 
            orig = blocks[k].split('\n')
            dests = orig[1:]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            orig=int(orig[0])

            d = [eval('{'+a.replace(';',',').replace(' ','') +'}') for a in dests]
            destinations = {}
            for i in d:
                destinations = {**destinations, **i}
            matrix[orig] = destinations
        
        zones = max(matrix.keys())
        mat = np.zeros((zones, zones))
        for i in range(zones):
            for j in range(zones):
                # We map values to a index i-1, as Numpy is base 0
                mat[i, j] = matrix.get(i+1,{}).get(j+1,0)
        
        coordsfile = os.path.join(root, "potentialgames_ws", "TransportationNetworks", network, network + '_node.tntp')
        coords = pd.read_csv(coordsfile, sep='\t')
        trimmed= [s.strip().lower() for s in coords.columns]

        coords.columns = trimmed
        
        # And drop the silly first and last columns
        coords.drop([';'], axis=1, inplace=True)
        self.coords = coords.to_numpy() #
        self.coords = np.stack([self.coords[:, 1], self.coords[:, 2]], 1)        
    
        if not self.modified:
            self.agents = np.array(np.nonzero(mat))
            self.no_players = len(self.agents[0])
            self.demand = np.zeros((self.no_players, 1))
            print("agents")
            print(self.agents)
            
            for i in range(self.no_players):
                self.demand[i] = mat[self.agents[0, i], self.agents[1, i]] / 100
        else:
            self.agents = np.array( np.nonzero(mat))
            self.agents = np.repeat(self.agents, self.modified_no_players, axis = 1)
            self.no_players = self.modified_no_players
            self.demand = np.zeros((self.no_players, 1))
            
            for i in range(self.no_players):

                self.demand[i] = mat[self.agents[0, i], self.agents[1, i]] / 100 / self.no_players
        
    def load_precomputed_data(self, network):
        """
            Load precomputed data. Includes the empirically estimated min and max travel times, and strategies.

        Args:
            network (str): Name of network.
        """
        
        self.improve_max_potential = None
        
        root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", network)
        self.file_path_strategies = os.path.join(root, "precomputed_strategies.pckl")
        
        if os.path.exists(self.file_path_strategies):
            with open(self.file_path_strategies, 'rb') as f:
                self.action_space = pickle.load(f)
        else:
            self.compute_strategies()
            
        self.file_path_travel_times = os.path.join(root, "estimated_travel_times.pckl")
        self.file_path_delta = os.path.join(root, "estimated_delta.pckl")
        self.file_path_potential = os.path.join(root, "estimated_potential.pckl")
        
        if os.path.exists(self.file_path_travel_times):
            with open(self.file_path_travel_times, 'rb') as f:
                travel_times = pickle.load(f)
                self.max_travel_times = travel_times[0]
                self.min_travel_times = travel_times[1]
            with open(self.file_path_potential, 'rb') as f:
                potential = pickle.load(f)
                self.max_potential = potential[0]
                self.min_potential = potential[1]
            with open(self.file_path_delta, 'rb') as f:
                self.delta = pickle.load(f)/(self.max_potential - self.min_potential)
        else:
            self.estimate_travel_times()
        
        self.improve_max_potential = self.max_potential
        self.improve_min_potential = self.min_potential
        self.improve_second_min_potential = self.min_potential+self.delta*(self.max_potential - self.min_potential)
        
    def build_network(self):
        """
            Build network of type DiGraph.
        """
        
        # based on https://github.com/sessap/noregretgames/blob/master/Traffic_Routing/Network_functions.py
        
        self.network = nx.DiGraph()
        
        for node in self.nodes: 
            self.network.add_node(str(node), pos = (self.coords[node, 0], self.coords[node, 1]))
            
        for i in range(len(self.edges[0])):
            self.network.add_edge(str(self.edges[0, i]), str(self.edges[1, i]), weight = self.lengths[i])        
    
    def find_edge_idx(self, node1, node2):
        """
            Find the index of the edge connecting the given nodes.

        Args:
            node1 (int): idx of first node
            node2 (int): idx of second node

        Returns:
            int: index of the edge 
        """
        for e in range(len(self.edges[0])):
            if self.edges[0][e] == node1 and self.edges[1][e] == node2:
                return e
       
    def compute_strategies(self):
        """
            Compute k shortest paths between each two nodes in the network.
            
            Based on https://github.com/sessap/noregretgames/blob/master/Traffic_Routing/Network_functions.py
        """
        
        i = 0
        self.action_space = []
                
        # Main loop
        for i in range(self.no_players):
            
            actions = self.k_shortest_paths(str(self.agents[0,i]), str(self.agents[1,i]), self.no_actions, weight="weight") # Compute k shortest path
            self.action_space_i = []                    
            
            # Find the edges belonging to the path
            for j in range(len(actions)):

                edges_vec = np.zeros((len(self.edges[0]), 1))

                for node in range(len(actions[j]) - 1):
                    idx = self.find_edge_idx(int(actions[j][node]), int(actions[j][node + 1]))
                    edges_vec[idx] = 1
                    
                demand_vec = np.multiply(edges_vec, self.demand[i]) # Determine the demand vector
                
                self.action_space_i.append(demand_vec)
                    
            self.action_space.append(self.action_space_i)
            
        with open(self.file_path_strategies, 'wb') as f:
            pickle.dump(self.action_space, f, pickle.HIGHEST_PROTOCOL)
                    
    def k_shortest_paths(self, source, target, k, weight=None):
        """
            Find k shortest paths in the network between a source node and a target node in the network.
            
        Args:
            source (str): Number of the node.
            target (str): Number of the node.
            k (int): Desired number of shortest paths.
            weight (str, optional): Type of shortest path. Defaults to None.

        Returns:
            list: List of k shortest paths.
        """
        
        return list(
            islice(nx.shortest_simple_paths(self.network, source, target, weight=weight), k)
        )
    
    def estimate_travel_times(self):
        """
            Estimate travel times for each agent in the network.
        """
        
        self.max_travel_times = np.zeros(self.no_players)
        self.min_travel_times = np.ones(self.no_players) * np.inf
        self.max_potential = 0
        self.min_potential = None
        self.second_min_potential = None
        
        print("Starting Travel Times estimation.")
        
        # Main loop
        for i in range(10000):
            
            if i % 100 == 0:
                print("Currently on " + str(i) + "th iteration")
            
            action_profile = np.array([rng.integers(0, len(self.action_space[agent_id]), 1)[0] for agent_id in range(self.no_players)])
            
            # Compute max and min travel times for each agent for given joint action profile
            self.max_travel_times = [max(self.max_travel_times[agent_id], self.travel_time(agent_id, action_profile[agent_id], action_profile[self.opponents_idx_map[agent_id]])) for agent_id in range(self.no_players)]
            self.min_travel_times = [min(self.min_travel_times[agent_id], self.travel_time(agent_id, action_profile[agent_id], action_profile[self.opponents_idx_map[agent_id]])) for agent_id in range(self.no_players)]
            
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
        
        with open(self.file_path_travel_times, 'wb') as f:
            pickle.dump([self.max_travel_times, self.min_travel_times], f, pickle.HIGHEST_PROTOCOL)
        
        with open(self.file_path_delta, 'wb') as f:
            pickle.dump(self.delta, f, pickle.HIGHEST_PROTOCOL)
        
        with open(self.file_path_potential, 'wb') as f:
            pickle.dump(np.array([self.max_potential, self.min_potential]), f, pickle.HIGHEST_PROTOCOL)

        print("     Done.")
        
    def congestion(self, x):
        """
            Compute the congestion over each edge based on the demand, capacity and free flow of the edge.

        Args:
            x (np.array(E)): Demand over each edge.

        Returns:
            np.array(E): Congestion over each edge.
        """
        
        congestions = np.multiply(self.free_flows,(1 + np.multiply(self.b, np.power(np.divide(x.T, self.capacities)[0].T, self.powers))))
        
        return congestions
                
    def travel_time(self, agent_id, action, opponents_actions):
        """
            Computes the total travel time for the given agent and their plan given other players chosen paths.

        Args:
            agent_id (int): Agent id
            action (int): Action
            opponents_actions (np.array(N-1)): Joint action profile of the opponents

        Returns:
            double: Total travel time.
        """
        
        ones = self.action_space[agent_id][action] > 0
                
        phi = np.sum([np.multiply(ones, self.action_space[self.opponents_idx_map[agent_id][i]][opponents_actions[i]]) for i in range(self.no_players - 1)], axis = 0) # compute demand of opponents on the common edges
        
        x =  self.action_space[agent_id][action] + phi

        congestions = self.congestion(x) # Compute the congestion
                
        return (self.action_space[agent_id][action]).T @ congestions
    
    def travel_time_modified(self, agent_id, action, opponents_actions):
        """
            Computes the total travel time for the given agent and their plan given other players chosen paths for the modified game.

        Args:
            agent_id (int): Agent id
            action (int): Action
            opponents_actions (np.array(N-1)): Joint action profile of the opponents

        Returns:
            double: Total travel time.
        """
        
        ones = self.action_space[agent_id][action] > 0
                
        phi = np.sum([np.multiply(ones*opponents_actions[i], self.action_space[0][i]) for i in range(self.no_actions)], axis = 0)
         
        x =  self.action_space[agent_id][action] + phi
        
        congestions = self.congestion(x)
                
        return (self.action_space[agent_id][action]>0).T @ congestions
    
    def utility_function(self, agent_id, action, opponents_actions):
        """
            Compute utility of an action for an agent given other players' actions.
            Utility is negative cost. Cost is the total travel time.    

        Args:
            agent_id (int): Agent id
            action (int): Action
            opponents_actions (np.array(N-1)): Joint action profile of the opponents.

        Returns:
            double: Utility.
        """

        total_travel_time = self.travel_time(agent_id, action, opponents_actions)
        
        return -total_travel_time
        
    def modified_utility_function(self, agent_id, action, opponents_actions):
        """
            Compute utility of an action for an agent given other players' actions for the modified game.
            Utility is negative cost. Cost is the total travel time for the modified game.  

        Args:
            agent_id (int): Agent id
            action (int): Action
            opponents_actions (np.array(N-1)): Joint action profile of the opponents.

        Returns:
            double: Utility.
        """
        
        total_travel_time = self.travel_time_modified(agent_id, action, opponents_actions)
        
        return ((-total_travel_time + self.max_potential)/(self.max_potential - self.min_potential))
        
    def potential_function_est(self, profile):
        """
            Estimate value of the potential function for the given joint action profile.

        Args:
            profile (np.array(N)): Joint action profile.

        Returns:
            double: Potential value
        """
        
        x = np.zeros((len(self.edges[0]), 1))
        potentials = 0
        
        # Main loop
        for i in range(self.no_players):
            
            x = x + self.action_space[i][profile[i]]
            congestions = self.congestion(x)
    
            potentials = potentials + (congestions * (self.action_space[i][profile[i]] > 0))
        
        potential = np.sum(potentials)
        
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
            Compute normalised potential value.

        Args:
            profile (np.array(N)): Joint action profile.

        Returns:
            double: Potential value.
        """
    
        potential = self.potential_function_est(profile)
        
        return (-potential + self.max_potential)/(self.max_potential - self.min_potential)
    
    def objective(self, profile):
        """
            Compute the value of the objective function - the average congestion over an edge in the network.

        Args:
            profile (np.array(N)): Joint action profile.

        Returns:
            double: Average congestion.
        """
        
        ones = np.ones(len(self.edges[0]))
        
        x = np.sum([np.multiply(ones, self.action_space[player_id][profile[player_id]]) for player_id in range(self.no_players)], axis = 0)
        
        congestion = np.mean(0.15* np.power(np.divide(x.T, self.capacities)[0].T, self.powers))
        
        return congestion