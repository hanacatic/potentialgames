import os
import numpy as np 
import pandas as pd
import pickle
import networkx as nx
from itertools import islice
from functools import partial

rng = np.random.default_rng()

class CongestionGame:
    
    def __init__(self, network = "SiouxFalls", no_actions = 5):
        
        self.no_actions = no_actions
        
        self.load_from_tntp(network)
        self.build_network()
    
        self.opponents_idx_map = [ np.delete(np.arange(self.no_players), player_id) for player_id in range(self.no_players) ]
        
        self.load_precomputed_data(network)
        
        self.delta = 1e-6
                
        self.utility_functions = []
        
        for i in range(0, self.no_players):
            self.utility_functions.append(partial(self.utility_function, i))
    
    def load_from_tntp(self, network):
        
        # Based on _scripts https://github.com/bstabler/TransportationNetworks
        
        root = os.path.dirname(os.path.abspath('.'))
        netfile = os.path.join(root, "potentialgames_ws", "TransportationNetworks", network, network + '_net.tntp')
        net = pd.read_csv(netfile, skiprows=8, sep='\t')
        trimmed= [s.strip().lower() for s in net.columns]
        net.columns = trimmed

        # And drop the silly first andlast columns
        net.drop(['~', ';'], axis=1, inplace=True)
        
        self.net = net.to_numpy() # initial node, terminal node, capacity, length, free flow time, b, power, speed, toll, link type
        
        initial_nodes = self.net[:, 0] - 1
        terminal_nodes = self.net[:, 1] - 1
        
        self.nodes = (np.unique(np.stack([initial_nodes, terminal_nodes], 0))).astype(int)
        self.edges = np.array([initial_nodes, terminal_nodes]).astype(int)
        self.capacities = self.net[:, 2] / 100 
        self.lengths = self.net[:, 3]
        self.free_flows = self.net[:, 4] #/ 100
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
        
        self.agents = np.array(np.nonzero(mat))
        self.no_players = len(self.agents[0])
        self.demand = np.zeros((self.no_players, 1))
        for i in range(self.no_players):
            self.demand[i] = mat[self.agents[0, i], self.agents[1, i]] / 100
    
    def load_precomputed_data(self, network):
        
        root = os.path.join(os.path.dirname(os.path.abspath('.')),  "potentialgames_ws", "potentialgames", "src", "lib", "games", "data", network)
        self.file_path_strategies = os.path.join(root, "precomputed_strategies.pckl")
        
        if os.path.exists(self.file_path_strategies):
            with open(self.file_path_strategies, 'rb') as f:
                self.action_space = pickle.load(f)
        else:
            self.compute_strategies()
            
        self.file_path_travel_times = os.path.join(root, "estimated_travel_times.pckl")
        # self.file_path_delta = os.path.join(root, "estimated_delta.pckl")
        
        if os.path.exists(self.file_path_travel_times):
            with open(self.file_path_travel_times, 'rb') as f:
                self.action_space = pickle.load(f)
        else:
            self.estimate_travel_times()
        
    def build_network(self):
        
        # based on https://github.com/sessap/noregretgames/blob/master/Traffic_Routing/Network_functions.py
        
        self.network = nx.DiGraph()
        
        for node in self.nodes: 
            self.network.add_node(str(node), pos = (self.coords[node, 0], self.coords[node, 1]))
            
        for i in range(len(self.edges[0])):
            self.network.add_edge(str(self.edges[0, i]), str(self.edges[1, i]), weight = self.lengths[i])        
    
    def find_edge_idx(self, node1, node2):
        for e in range(len(self.edges[0])):
            if self.edges[0][e] == node1 and self.edges[1][e] == node2:
                return e
       
    def compute_strategies(self):
        
        # based on https://github.com/sessap/noregretgames/blob/master/Traffic_Routing/Network_functions.py

        i = 0
        self.action_space = []
                
        for i in range(self.no_players):
        
            actions = self.k_shortest_paths(str(self.agents[0,i]), str(self.agents[1,i]), self.no_actions, weight="weight")
            self.action_space_i = []                    
            for j in range(len(actions)):

                edges_vec = np.zeros((len(self.edges[0]), 1))

                for node in range(len(actions[j]) - 1):
                    idx = self.find_edge_idx(int(actions[j][node]), int(actions[j][node + 1]))
                    edges_vec[idx] = 1
                    
                demand_vec = np.multiply(edges_vec, self.demand[i])
                
                self.action_space_i.append(demand_vec)
                    
            self.action_space.append(self.action_space_i)
            
        with open(self.file_path_strategies, 'wb') as f:
            pickle.dump(self.action_space, f, pickle.HIGHEST_PROTOCOL)
                    
    def k_shortest_paths(self, source, target, k, weight=None):
        return list(
            islice(nx.shortest_simple_paths(self.network, source, target, weight=weight), k)
        )
    
    def estimate_travel_times(self):
        
        self.max_travel_times = np.zeros(self.no_players)
        
        for i in range(10000):
            if i % 100 == 0:
                print("Currently on " + str(i) + "th iteration")
            
            action_profile = np.array([rng.integers(0, len(self.action_space[agent_id]), 1)[0] for agent_id in range(self.no_players)])
            
            self.max_travel_times = [max(self.max_travel_times[agent_id], self.travel_time(agent_id, action_profile[agent_id], action_profile[self.opponents_idx_map[agent_id]])) for agent_id in range(self.no_players)]
            
                
    def travel_time(self, agent_id, action, opponents_actions):
        
        ones = self.action_space[agent_id][action] > 0
                
        phi = np.sum([np.multiply(ones, self.action_space[self.opponents_idx_map[agent_id][i]][opponents_actions[i]]) for i in range(self.no_players - 1)], axis = 0)
        
        x =  self.action_space[agent_id][action] + phi
        
        congestions = np.multiply(np.multiply(ones.T, self.free_flows),(1 + np.multiply(self.b, np.power(np.divide(x.T, self.capacities)[0].T, self.powers))))
                
        return self.action_space[agent_id][action].T @ congestions[0]
        
    def utility_function(self, agent_id, action, opponents_actions):

        total_travel_time = self.travel_time(agent_id, action, opponents_actions)
        
        utility = 1-0.00025*(total_travel_time)

        return  max(utility, [0.0]) #-total_travel_time
        
    def potential_function(self, profile):
        total_time = 0
        # for i in range(self.no_players):
        #     total_time = total_time + self.travel_time(i, profile[i], profile[self.opponents_idx_map[i]])
        return self.objective(profile)
    
    def objective(self, profile):
        ones = np.ones(len(self.edges[0]))
        
        x = np.sum([np.multiply(ones, self.action_space[player_id][profile[player_id]]) for player_id in range(self.no_players)], axis = 0)
        
        congestion = np.mean(0.15* np.power(np.divide(x.T, self.capacities)[0].T, self.powers))
        
        return congestion