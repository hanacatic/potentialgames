import os
import numpy as np 
import pandas as pd
import openmatrix as omx

class CongestionGame:
    
    def __init__(self):
        self.load_from_tntp()
        
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

        self.demand = mat
