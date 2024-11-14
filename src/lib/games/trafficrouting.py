import os
import numpy as np 
import pandas as pd
import networkx as nx
from itertools import islice

class CongestionGame:
    
    def __init__(self):
        
        self.load_from_tntp()
        self.build_network()
        self.compute_strategies()
         
    def load_from_tntp(self, network = "SiouxFalls"):
        
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
        
        self.agents = np.array(np.nonzero(mat))
        self.no_players = len(self.agents[0])
        self.demand = np.zeros((self.no_players, 1))
        for i in range(self.no_players):
            self.demand[i] = mat[self.agents[0, i], self.agents[1, i]] / 100
    
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
        
            actions = self.k_shortest_paths(str(self.agents[0,i]), str(self.agents[1,i]), 5, weight="weight")
            self.action_space_i = []                    
            for j in range(len(actions)):

                edges_vec = np.zeros((len(self.edges[0]), 1))

                for node in range(len(actions[j]) - 1):
                    idx = self.find_edge_idx(int(actions[j][node]), int(actions[j][node + 1]))
                    edges_vec[idx] = 1
                    
                demand_vec = np.multiply(edges_vec, self.demand[i])
                if j == 0:
                    self.action_space_i.append(demand_vec)
                elif np.dot(demand_vec.T, self.free_flows) < 3 * np.dot(self.action_space_i[0].T, self.free_flows):
                    self.action_space_i.append(demand_vec)
                    
            self.action_space.append(self.action_space_i)
            
    def k_shortest_paths(self, source, target, k, weight=None):
        return list(
            islice(nx.shortest_simple_paths(self.network, source, target, weight=weight), k)
        )