### -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 14:42:12 2022
@author: Vedika Vishweshwar
"""

"""
"This document runs short bursts on Georgia's precincts data. Since these bursts ran slower than those on block groups data, 
this document is not used in data-gathering. However, it is still included in the Github folder for reference in case readers wish to experiment with Georgia's precincts data, or apply this code to precincts data from other states. 


This document proceeds as follows: first, the ideal population of each district is calculated, after which the initial partition is created. This is the seed plan from which each short burst starts. Then, the short bursts are run according to the specified parametres (burst length and number of bursts), recording the maximum number of majority-minority districts observed in each burst and the ensemble of bursts. 
"""


""" Here, num_dist specifies the number of districts in each plan visited in the short bursts. Georgia has 
180 legislative house districts, 56 legislative senate districts, and 14 congressional districts. This document runs short bursts 
on all three types of plans. """
num_dist = 14

# Import the relevant libraries to run short bursts: 
import matplotlib.pyplot as plt
import random 
import gerrychain
import numpy as np
import networkx as nx
import geopandas as gpd
import pickle
import zipfile

"""This is to make the code more replicable. For example, if we want the same result twice, we can run the code with exactly the
same random seed. Here, the random seed is 48."""
random.seed(48)

# Import attributes of gerrychain that we need to create the initial partition: 
from gerrychain import Graph, Partition, Election, proposals, updaters, constraints, accept 
from gerrychain.updaters import cut_edges, Tally
from gerrychain.tree import recursive_tree_part

# Import the Georgia precincts graph. Ensure that the precincts data is in the same directory as this file.
ga_precincts = Graph.from_json("georgia.json")

# Import attributes of gerrychain that we need to create the initial partition: 
from gerrychain import Graph, Partition, Election, proposals, updaters, constraints, accept 
from gerrychain.updaters import cut_edges, Tally
from gerrychain.tree import recursive_tree_part

# Ideal population calculation: 
""" Here, the ideal population is calculated for each district. Conceptually, this is the total population of Georgia divided by 
the number of districts. The total population is calculated by summing the population of all the nodes in the block group dual 
graph. The number of districts depends on which plan we are running short bursts on.
"""
pop = 0
for v in range(len(ga_precincts.nodes())):
    totpop = ga_precincts.nodes[v]["POPULATION"]
    pop = pop + totpop 
ideal_pop = pop/num_dist
print("Ideal Pop:", ideal_pop)

# Creating the initial partition: 
"""
Here, the initial partition is created for the short bursts. The seed plan is grown using Gerrychain's Partition class. Partition 
takes three arguments: a graph, an assignment of nodes to districts, and a dictionary of updaters. 

The relevant graph is the Georgia precincts graph graph. The recursive_tree_part function is used for the assignment argument, 
which partitions a tree into range(num_dist) parts of a population that are within epsilon = 2% of the ideal population. 

Then, we extract information from each district in the partition through the updaters. Specifically, we extract the number of 
cut edges, whether or not it is connected, its BVAP, VAP, and population. 
"""

# Creating the initial partition
initial_partition = Partition(ga_precincts, 
assignment = recursive_tree_part(ga_precincts, range(num_dist), ideal_pop, "POPULATION", 0.02, 10), 
updaters={
    "cut edges": cut_edges, 
    "connectedness": (nx.is_connected(ga_precincts)), 
    "population": Tally("POPULATION", alias = "population"), 
    "BVAP": Tally("BVAP"), 
    "VAP": Tally("VAP")
})

## Check that each partition has information saved:
for district, BVAP in initial_partition["BVAP"].items():
    print("District{} BVAP: {}". format(district, BVAP))
for district, VAP in initial_partition["VAP"].items():
    print("District{} VAP: {}". format(district, VAP))


# Short Bursts:
"""
This is the code for setting up and running the short bursts. First, this section specifies the parameters of the short bursts 
and defines the proposal function for each new plan. Then, it defines the constraints for proposed plans, which ensure 
that they are valid. Lastly, inside a 'for' loop, the short bursts are run using the MarkovChain function, and the highest 
number of majority-minority districts observed per burst are recorded. 
"""
 
## Import additional attributes from Gerrychain required to run the short bursts: 
from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous, contiguous
from gerrychain.proposals import propose_random_flip, recom
from functools import partial
from gerrychain.accept import always_accept

## Specify the parameters of the short bursts: 
burst_length = 5  # length of each burst
num_bursts = 10  # number of bursts in the run
total_steps = 5000  # this is equal to num_bursts * burst_length.
initial_state = initial_partition   # initial_state specifies the seed plan the short bursts start from


## Defining the constraints: 
""" 
1) Compactness constraint: To keep the districts as compact as the original plan, we bound the number of cut edges at 2 
times the number of cut edges as the initial plan.

2) Population constraint: To ensure that the chain only generates partitions that are within epsilon of the ideal population. 
the gerrychain.constraints.within_percent_of_ideal_population function accomplishes exactly this. 
"""

# 1) Compactness constraint:
compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]), 
    2*len(initial_partition["cut_edges"])
)

# 2) Population constraint: 
pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)

## Defining the proposal function: 
""" The proposal function used is the Recombination Markov Chain, also known as ReCom. For a given plan, ReCom randomly selects 
two adjacent districts, merges them into a single district, and then randomly re-partitions them in a manner that 
maintains the population balance. """
proposal1 = partial(recom, pop_col = "POPULATION", pop_target = ideal_pop, epsilon = 0.02, node_repeats = 1)

# Running the short bursts: 
"""
This is the code to run the short bursts for range(num_bursts) number of times. For every transition in each burst, the number of 
majority minority districts are counted and recorded it in list_of_mm. Then, the most recent plan with the highest number 
of majority minority districts is identified and the next burst restarts from this plan. list_of_mm keeps track of the 
plans with the maximum number of majority minority districts per burst, which is reflected in max1[1] and max1[0] respectively.
"""

list_of_max = []

for i in range(num_bursts):
    print("i:", i)
    list_of_mm = [] 
    Chain3 = MarkovChain(
        proposal = proposal1, 
        constraints = [compactness_bound, pop_constraint],
        # A every proposed plan that meets the compactness and population constraints
        accept = always_accept, 
        initial_state = initial_partition, 
        total_steps = burst_length) 
        # For loop for each transition in a single burst. This equals burst length.
    for part in Chain3:  
        # Here we calculate the number of majority minority districts: 
        # First, list the BVAP and VAP for every node in the plan
        BVAP4 = list((part["BVAP"].items())) 
        VAP4 = list((part["VAP"].items())) 
        # Set the number of majority minority districts to 0 initially
        maj_min_districts1 = 0 
        # For loop for every district in the plan
        for i, j in zip(BVAP4, VAP4):
            # Add 1 to the number of majority minority districts if we find a district where BVAP/VAP > 0.5
            maj_min_perc1 = (i[1]/j[1])  
            if maj_min_perc1 >= 0.5:
                maj_min_districts1 += 1 
                # Record the tally of majority minority districts per transition
                mm = [maj_min_districts1] 
                # Append the plan to this tally
                mm.append(part) 
        # Create master list of the number of majority minority districts per transition per burst. Recall that 
        # list_of_mm is already an empty list defined above the for loop. 
        list_of_mm += [mm] 
    
    print("list_of_mm:", list_of_mm) 
    max1 = [0] 
    max1_a = 0
    # Iterating through the master list of the number of majority minority districts per burst 
    for a in range(len(list_of_mm)): 
        # Identifies the burst step
        print("a", a) 
        # Print the number of majority_minority districts for that step
        print("list_of_mm[a][0]:", list_of_mm[a][0]) 
        # Print the maximum found across the burst
        print("max1[0]:", max1[0])  
        # Find the plan and the position of the maximum of list_of_mm
        if list_of_mm[a][0] >= max1[0]: 
            max1_a = a 
            max1 = list_of_mm[a]  
    # Prints the burst step that yielded the maximum number of majority_minority districts 
    print('max_a:', max1_a) 
    # Set the initial partition of the next burst to the most recent plan with the highest number of majority minority districts 
    # in the previous plan
    initial_partition = max1[1] 





 

    


    
            
                
            





