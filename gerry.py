### -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:42:12 2022
@author: Vedika Vishweshwar
"""

"""
In this document, I load districting data on Georgia into Python and check that it has loaded correctly. I load data on Georgia's precincts and Georgia's census block groups respectively. This data is in the form of dual graphs, where each node represents either a census block group or a precinct. Edges between nodes reflect geographic adjacency. Each node in the precincts graph contains ACS 2019 data by precinct. Each node in the block group graph contains ACS 2019 data by census block group. Loading these graphs is the first step of running short bursts. 
"""

# Here we import the relevant libraries to load the data. Gerrrychain is a Python library used to build ensembles of districting plans using Markov chain Monte Carlo methods. It can be installed through this website: https://gerrychain.readthedocs.io/en/latest/
import matplotlib.pyplot as plt
import pickle
import numpy as np
import networkx as nx
import geopandas as gpd
import gerrychain

# Import attributes of gerrychain to check that the nodes of the graphs have information saved:
from gerrychain import Graph, Partition, Election

# Import the Georgia precincts graph:
ga_precincts = Graph.from_json("georgia.json")
print("ga_precincts:", ga_precincts)

# Look at the data stored in one node of the precincts graph: 
print("ga_precincts.nodes[0]:", ga_precincts.nodes[0])
    
# Look at the BVAP of this node:
print("ga_precincts.nodes[2][BVAP]:", ga_precincts.nodes[2]['BVAP'])

# Count how many precincts there are: 
print("len(ga_precincts.nodes):", len(ga_precincts.nodes))

# Check whether the precincts dual graph is conencted: 
print(nx.is_connected(ga_precincts))

# Import the Georgia census block groups graph: 
ga_bg = pickle.load(open("GA_blockgroup_graph.p", "rb"))

# Look at data stored in one node of the block groups graph:
print(ga_bg.nodes[0])

# Access the BVAP of this node (from 2010 census): 
print(ga_bg.nodes[0]['BVAP10'])

# Count how many census block groups there are:  
print(len(ga_bg.nodes))

# Check whether the block group dual graph is connected:
print(nx.is_connected(ga_bg))

# To check what attributes the gerrychain package has: 
print(dir(gerrychain))
print(dir(gerrychain.graph))