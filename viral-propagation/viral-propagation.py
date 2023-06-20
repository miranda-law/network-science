# ==============================================================================
# Author: Miranda Law
# Course: COSC-355 Network Science
# Assignment: HW5 Viral Propagation
# Last updated: 12/01/2022
# Desc: We will simulate a toy version of a "market crash" caused by the 
#       collapse of the stock price of a company. Given that stock prices are
#       interdependent, we will assume that if one company's stock value 
#       collapses, then there is a chance that the stock value of other 
#       companies may also collapse. The higher the interdependence between two
#       companies, the higher the chances of this happening.
# ==============================================================================

# ==============================================================================
# Imports
# ==============================================================================
import gzip
import csv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from tabulate import tabulate

# ==============================================================================
# 1. Load node labels and edges
# ==============================================================================

# Initialize empty dict for id: company name
id2name = {}

# Read in company names
with open("stocks_62_names.txt") as f:
    reader = csv.reader(f, delimiter=' ')
    for line in reader:
        id = line[0]
        name = line[1]
        id2name[id] = name

# Initialize list for the Pearson correlation b/w the prices of these stocks
# [[id1, id2, Pearson correlation b/w the prices of these stocks]]
edges = []

# Read in edges
with open("stocks_62_pearson.net.txt") as f:
    reader = csv.reader(f, delimiter=' ')
    for line in reader:
        edges.append(line)

# ==============================================================================
# 2. Create and visualize this graph
# ==============================================================================

# Create graph
graph = nx.Graph()

for e in edges:
    graph.add_edge(id2name[e[0]], id2name[e[1]], weight=float(e[2]))


"""
draw_weighted_graph(): visualize a weighted graph using networkx
    input:
        g - weighted networkx graph
        node_colors - a list of node colors
        label_color - color used for node labels
        pos (optional) - list of node positions
    output:
        dictionary of positions computed by spring_layout 
"""

def draw_weighted_graph(g, node_colors, label_color, pos={}):
    positions = pos
    if len(positions) == 0:
        positions = nx.spring_layout(g)

    nodes = nx.draw_networkx_nodes(g, pos=positions, node_color=node_colors)
    labels = nx.draw_networkx_labels(g, pos=positions, font_color=label_color)
    weights = [graph[u][v]['weight'] for u,v in graph.edges()]
    edges = nx.draw_networkx_edges(g, pos=positions, width=weights)

    plt.show()
    return positions


p = draw_weighted_graph(graph, node_colors=['green' for u in graph.nodes()], label_color='white')


# ==============================================================================
# 3. Simulate one propagation using the independent cascade model
# ==============================================================================

# Initialize an infected dictionary with every node having value False
infected = dict([(node, False) for node in graph.nodes()])

def simulate_independent_cascade(graph, starting_node, weight_attr_name, weight_multiplier, infected):
    
    # Mark a starting node u as infected with value True
    infected[starting_node] = True

    # Keep track of infected neighbors
    infectedNeighbors = []

    # infect all uninfected neighbors of the starting node
    for n in graph.neighbors(starting_node):

        # if a neighbor is not infected
        if infected[n] == False:
            r = random.random()
            pTransmission = graph[starting_node][n]["weight"] * weight_multiplier

            # infect node
            if r < pTransmission:
                infected[n] = True
                infectedNeighbors.append(n)
    
    # recurse over infected neighbors
    for n in infectedNeighbors:
        simulate_independent_cascade(graph, n, weight_attr_name, weight_multiplier, infected)

    return infected

infected = simulate_independent_cascade(graph, starting_node='NSC', weight_attr_name='weight', weight_multiplier=0.1, infected=infected)

infected_count = sum([1 for is_infected in infected.values() if is_infected])
print("Infected nodes: %d" % infected_count)


p = draw_weighted_graph(graph, pos=p, node_colors=[('red' if infected[u] else 'green') for u in graph.nodes()], label_color='white')

# ==============================================================================
# 4. Compute the average size of the infection starting from a node
# ==============================================================================


"""
Create a function that simulates a series of infection (received as a num_trials parameter) from a starting node (received as a parameter start) and returns the average number of nodes that were infected over the trials. Use 0.1 as weight_multiplier, but you can also experiment with other values.

Use this function to perform a simulation with 1,000 (or better, 10,000) trials, from each of the nodes (i.e., 1,000 trials per node).
"""

def infectionSim(graph, num_trials, starting_node):
    avgInfections = 0

    for i in range(0, num_trials):
        # Initialize an infected dictionary with every node having value False
        infected = dict([(node, False) for node in graph.nodes()])
        # simulate infection cascade
        infected = simulate_independent_cascade(graph, starting_node=starting_node, weight_attr_name='weight', weight_multiplier=0.1, infected=infected)
        # count infections
        infected_count = sum([1 for is_infected in infected.values() if is_infected])
        
        avgInfections += infected_count
    return avgInfections/num_trials

infections = {'AA': 41.5092, 'AEP': 37.3161, 'AGC': 40.5815, 'AIG': 43.1038, 'AIT': 42.9706, 'ARC': 35.7447, 'AVP': 34.9996, 'AXP': 44.332, 'BAC': 42.2972, 'BA': 41.9883, 'BAX': 40.4744, 'BC': 42.3927, 'BEL': 39.6272, 'BHI': 39.4215, 'BMY': 38.3863, 'CHA': 37.6502, 'CI': 38.1488, 'CL': 41.7246, 'COL': 43.1573, 'CPB': 38.8259, 'CSC': 39.0753, 'DAL': 36.4931, 'DD': 44.0299, 'DIS': 43.6066, 'DOW': 42.4996, 'ETR': 34.9259, 'FDX': 42.5246, 'FLR': 44.1762, 'F': 41.1176, 'GD': 43.4098, 'GE': 44.2487, 'HAL': 40.9693, 'HON': 45.2854, 'HRS': 41.8192, 'IBM': 39.195, 'IFF': 44.1562, 'IP': 42.8175, 'JNJ': 42.8687, 'JPM': 43.1812, 'KO': 40.3128, 'MCD': 38.6994, 'MMM': 45.1378, 'MO': 39.8073, 'MRK': 37.7826, 'NSC': 42.8431, 'ORCL': 40.9911, 'OXY': 41.7513, 'PEP': 39.3918, 'PG': 38.8311, 'ROK': 43.5629, 'SLB': 42.9527, 'SO': 34.3268, 'S': 28.2537, 'T': 39.4639, 'UIS': 35.2503, 'USB': 43.8457, 'UTX': 44.8699, 'WFC': 44.2681, 'WMB': 39.8825, 'WMT': 35.1847, 'WY': 42.4036, 'XRX': 41.7984}
""" for node in graph.nodes():
    infections[node] = infectionSim(graph, 10000, node) """

#print(infections)
# {'AA': 41.5092, 'AEP': 37.3161, 'AGC': 40.5815, 'AIG': 43.1038, 'AIT': 42.9706, 'ARC': 35.7447, 'AVP': 34.9996, 'AXP': 44.332, 'BAC': 42.2972, 'BA': 41.9883, 'BAX': 40.4744, 'BC': 42.3927, 'BEL': 39.6272, 'BHI': 39.4215, 'BMY': 38.3863, 'CHA': 37.6502, 'CI': 38.1488, 'CL': 41.7246, 'COL': 43.1573, 'CPB': 38.8259, 'CSC': 39.0753, 'DAL': 36.4931, 'DD': 44.0299, 'DIS': 43.6066, 'DOW': 42.4996, 'ETR': 34.9259, 'FDX': 42.5246, 'FLR': 44.1762, 'F': 41.1176, 'GD': 43.4098, 'GE': 44.2487, 'HAL': 40.9693, 'HON': 45.2854, 'HRS': 41.8192, 'IBM': 39.195, 'IFF': 44.1562, 'IP': 42.8175, 'JNJ': 42.8687, 'JPM': 43.1812, 'KO': 40.3128, 'MCD': 38.6994, 'MMM': 45.1378, 'MO': 39.8073, 'MRK': 37.7826, 'NSC': 42.8431, 'ORCL': 40.9911, 'OXY': 41.7513, 'PEP': 39.3918, 'PG': 38.8311, 'ROK': 43.5629, 'SLB': 42.9527, 'SO': 34.3268, 'S': 28.2537, 'T': 39.4639, 'UIS': 35.2503, 'USB': 43.8457, 'UTX': 44.8699, 'WFC': 44.2681, 'WMB': 39.8825, 'WMT': 35.1847, 'WY': 42.4036, 'XRX': 41.7984}

sortedAvgs = sorted(infections.items(), key = lambda x: x[1], reverse = True)
print(sortedAvgs)
# [('HON', 45.2854), ('MMM', 45.1378), ('UTX', 44.8699), ('AXP', 44.332), ('WFC', 44.2681), ('GE', 44.2487), ('FLR', 44.1762), ('IFF', 44.1562), ('DD', 44.0299), ('USB', 43.8457), ('DIS', 43.6066), ('ROK', 43.5629), ('GD', 43.4098), ('JPM', 43.1812), ('COL', 43.1573), ('AIG', 43.1038), ('AIT', 42.9706), ('SLB', 42.9527), ('JNJ', 42.8687), ('NSC', 42.8431), ('IP', 42.8175), ('FDX', 42.5246), ('DOW', 42.4996), ('WY', 42.4036), ('BC', 42.3927), ('BAC', 42.2972), ('BA', 41.9883), ('HRS', 41.8192), ('XRX', 41.7984), ('OXY', 41.7513), ('CL', 41.7246), ('AA', 41.5092), ('F', 41.1176), ('ORCL', 40.9911), ('HAL', 40.9693), ('AGC', 40.5815), ('BAX', 40.4744), ('KO', 40.3128), ('WMB', 39.8825), ('MO', 39.8073), ('BEL', 39.6272), ('T', 39.4639), ('BHI', 39.4215), ('PEP', 39.3918), ('IBM', 39.195), ('CSC', 39.0753), ('PG', 38.8311), ('CPB', 38.8259), ('MCD', 38.6994), ('BMY', 38.3863), ('CI', 38.1488), ('MRK', 37.7826), ('CHA', 37.6502), ('AEP', 37.3161), ('DAL', 36.4931), ('ARC', 35.7447), ('UIS', 35.2503), ('WMT', 35.1847), ('AVP', 34.9996), ('ETR', 34.9259), ('SO', 34.3268), ('S', 28.2537)]

"""print("Company       AvgCascadeSize")
for ele1,ele2 in sortedAvgs:
    print("{:<14}{}".format(ele1,ele2))"""

print(tabulate(sortedAvgs, headers = ["Company", "Cascade Size (avg)"], tablefmt="grid"))