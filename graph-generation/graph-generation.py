# ===========================================================================================
# Author: Miranda Law
# Course: COSC-355 Network Science
# Assignment: HW2 Graph Generation
# Last updated: 10/18/2022
# Desc: Model ER and BA graphs using the networkx python library
#       Run the code at the bottom of this file by calling the functions as specified.
# ===========================================================================================

# ===========================================================================================
# Imports
# ===========================================================================================
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# ===========================================================================================
# Functions
# ===========================================================================================

"""
flip_coin() performs Bernoulli trials given a probability p 
    input: p - probability (probabilities must be in [0,1])
    output: True with probability p, and False with probability 1-p
"""
def flip_coin(p):
    # Check if input is a valid probability
    if p >= 0 and p <= 1:
        return np.random.random() <= p
    raise ValueError('Probabilities must be in [0,1]')
"""
generate_random_graph(): creates a graph of N nodes with edges between each pair of (u, v)
    nodes of probability p
    input: N - number of nodes, p - probability
    output: nx graph
"""
def generate_random_graph(N, p):
    # initialize graph
    g = nx.Graph()

    # create empty graph with N nodes
    for i in range(0, N):
        g.add_node(i)
    
    # add edges with probability p
    for u in range(0, N):
        for v in range(u + 1, N):
            if(flip_coin(p)):
                g.add_edge(u, v)
    return g

"""
print_er_statistics(g,p) given an ER graph g and a probability p prints its actual average
    degree <k> and its expected average degree p(N-1)
    input: g - nx graph, p - probability
    output: N/A
"""
def print_er_statistics(g, p):
    # number of nodes of the graph g
    N = len(g.nodes())
    actualAvg = 0.0

    # compute expected average degree
    expectedAvg = p * (N - 1)

    # compute actual average degree
    for i in range(N):
        actualAvg += g.degree(i)
    actualAvg /= N

    # print
    print("Actual average degree <k>: " + str(actualAvg))
    print("Expected average degree: " + str(expectedAvg))

"""
select_targets() selects m target nodes in a graph g, with probabilities proportional to 
    the degrees of the nodes
    input: g - nx graph, m - number of target nodes
    output: length m array of nodes from g
"""
def select_targets(g, m):

    # Check if feasible
    if len(g.nodes()) < m:
        raise ValueError('Graph has less than m nodes')

    # Compute sum of degree
    sum_degree = 0

    # YOUR CODE HERE: COMPUTE SUM OF DEGREE OF NODES
    for i in range(0, len(g.nodes())):
        sum_degree += g.degree(i)

    if sum_degree == 0:
        raise ValueError('Graph as no edges')

    # Compute probabilities
    probabilities = []
    for u in g.nodes():
        # YOUR CODE HERE: COMPUTE PROBABILITY OF SELECTING NODE u
        # THEN APPEND IT TO probabilities USING probabilities.append(...)
        prob = g.degree(u)/sum_degree
        probabilities.append(prob)

    # Normalize the probabilities by dividing them by their sum if the sum is close enough to 1
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()

    # Sample with replacement
    selected = np.random.choice(g.nodes(), size=m, replace=False, p=probabilities)

    return selected

"""
generate_preferential_attachment_graph(): creates a graph of N nodes modeling after the BA model
    input: N - number of nodes, m0 - initial nodes forming a star graph, m - number of target nodes
    output: nx graph
"""
def generate_preferential_attachment_graph(N, m0, m):
    # initialize graph
    g = nx.Graph()

    # create empty graph with N nodes
    for i in range(0, m0):
        g.add_node(i)
    
    # add edges from node 0 to nodes 1, 2, 3, ..., m0 - 1
    for i in range(1, m0):
        g.add_edge(0, i)

    for u in range(m0, N):
        targets = select_targets(g, m)
        # add node u numbered from m0 to N - 1
        g.add_node(u)

        # link each node u to each of the m targets
        for node in targets:
            g.add_edge(u, node)
    return g


"""
draw_degree_dist() draws a log-log plot of the degree distribution of a graph g
    input: g - nx graph
    output: N/A
"""
def draw_degree_dist(g):
    # list of frequencies of the degrees in the network
    degree_freq = nx.degree_histogram(g)
    # possible degrees
    degrees = range(len(degree_freq))

    # draw plot
    plt.loglog(degrees, degree_freq) 
    plt.title("Degree Distribution")

    plt.show()

# ===========================================================================================
# Running the code
# ===========================================================================================

# 2 preferential attachment (BA) graphs with N=2100, m0=5, m=1 and N=2000, m0=2, m=2.
# ER: N=800 and p=0.0005, 0.001, 0.002, 0.005, 0.01

# Create graph
#g = generate_random_graph(800, 0.0005)
g = generate_preferential_attachment_graph(2100, 5, 1)

# Print ER graph stats
#print_er_statistics(g, 0.0005)

# Draw graph
nx.draw_networkx(g, node_size = 10, with_labels = False)
plt.show()
# Plot degree distribution of graph
draw_degree_dist(g)