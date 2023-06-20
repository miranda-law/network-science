# ==============================================================================
# Author: Miranda Law
# Course: COSC-355 Network Science
# Assignment: HW3 PageRank
# Last updated: 11/01/2022
# Desc: Implement PageRank on the WEBSPAM-UK2007 collection, a collection of
#       hosts in the UK that has been used for multiple studies on the effect
#       of web spam.
# ==============================================================================

# ==============================================================================
# Imports
# ==============================================================================
import gzip
import csv
import numpy as np  
import math

# ==============================================================================
# 1. Read the host names
# ==============================================================================

# Initialize empty dict for id: host name
id2name = {}

# Read in host names
with gzip.open("uk-2007-05.hostnames.txt.gz", "rt", encoding="utf-8") as input_file:
    reader = csv.reader(input_file, delimiter=' ', quotechar='"')
    for record in reader:
        # Append to dict with id: host name
        id2name[int(record[0])] = record[1]

# ==============================================================================
# 2. Implement PageRank
# ==============================================================================

"""
pagerank() reads in a compressed graph file and computes scores for 
    each node
    input:
        compressedGraphFile - compressed graph file name
        iterations - integer for number of iterations
        alpha - float
    output:
        score vector
"""
def pagerank(compressedGraphFile, iterations, alpha):

    # Get number of nodes
    N = 0
    with gzip.open(compressedGraphFile, "rt", encoding="utf-8") as input_file:
        reader = csv.reader(input_file, delimiter=' ')
        # Read first line only
        N = int(next(reader)[0])
    
    # Initialize score and aux vectors to 1/N and 0 respectively for N entries
    score = []
    aux = []
    for i in range(0, N):
        score.append(1/N)
        aux.append(0)

    # Run for `iterations` iterations
    for i in range(0, iterations):
        # Open compressed file
        with gzip.open(compressedGraphFile, "rt", encoding="utf-8") as input_file:
            reader = csv.reader(input_file, delimiter=' ')
            # Skip first line
            next(reader)
            # Go through each row
            # source (row number), destinations ([destination:weight])
            for source, destinations in enumerate(reader):
                for d in destinations:
                    # current destination, drop weight
                    destination = int(d.split(':', 1)[0])
                    aux[destination] += score[source]/len(destinations)
        # For each node
        for v in range(0, N):
            score[v] = alpha * aux[v] + (1 - alpha) * (1/N)
            aux[v] = 0
    return score

# ==============================================================================
# 3. Find the nodes with higher PageRank
# ==============================================================================

# Run page rank
ranked_nodes = pagerank("uk-2007-05.hostgraph_weighted.graph-txt.gz", 20, .85)

# Obtain list of hosts sorted by PageRank score [(id, score)]
ranked_nodes = list(enumerate(ranked_nodes))

# Sort list by scores, greatest to least
ranked_nodes = sorted(ranked_nodes, key=lambda host: host[1], reverse=True)

# Filter list by hostnames containing '.co.uk'
co_uk_hosts = [n for n in ranked_nodes if '.co.uk' in id2name[n[0]]]

# Filter list by hostnames containing '.co.uk'
gov_uk_hosts = [n for n in ranked_nodes if '.gov.uk' in id2name[n[0]]]

# ==============================================================================
# 4. Read a list of spam hosts
# ==============================================================================

# Initialize empty dict for id: boolean (is spam or not)
is_spam = {}

# Read set 1 labels
with open('WEBSPAM-UK2007-SET1-labels.txt') as f:
    lines = f.readlines()
    for line in lines:
        split_line = line.split()
        # if spam, add ID to dict
        if split_line[1] == 'spam':
            is_spam[int(split_line[0])] = True

# Read set 2 labels
with open('WEBSPAM-UK2007-SET2-labels.txt') as f:
    lines = f.readlines()
    for line in lines:
        split_line = line.split()
        # if spam, add ID to dict
        if split_line[1] == 'spam':
            is_spam[int(split_line[0])] = True

# ==============================================================================
# 5. Run non-spam PageRank (still need to fix degrees)
# ==============================================================================

"""
nonspam_pagerank() reads in a compressed graph file and computes scores for 
    each node if neither the source nor destination is a spam host
    input:
        compressedGraphFile - compressed graph file name
        iterations - integer for number of iterations
        alpha - float
        spamList - dict of spam IDs {id: True}
    output:
        score vector
"""

def nonspam_pagerank(compressedGraphFile, iterations, alpha, spamList):

    # Get number of nodes
    N = 0
    with gzip.open(compressedGraphFile, "rt", encoding="utf-8") as input_file:
        reader = csv.reader(input_file, delimiter=' ')
        # Read first line only
        N = int(next(reader)[0])
    
    # Initialize score and aux vectors to 1/N and 0 respectively for N entries
    score = []
    aux = []
    for i in range(0, N):
        score.append(1/N)
        aux.append(0)

    # Run for `iterations` iterations
    for i in range(0, iterations):
        # Open compressed file
        with gzip.open(compressedGraphFile, "rt", encoding="utf-8") as input_file:
            reader = csv.reader(input_file, delimiter=' ')
            # Skip first line
            next(reader)
            # Go through each row
            # source (row number), destinations ([destination:weight])
            for source, destinations in enumerate(reader):

                # Ignore link if source is spam
                if source in spamList.keys():
                    continue

                outDegree = 0
                realDestinations = []

                for d in destinations:
                    # current destination, drop weight
                    destination = int(d.split(':', 1)[0])
                    
                    # Ignore link if destination is spam
                    if destination in spamList.keys():
                        continue
                    
                    # Count out degree for source
                    outDegree += 1
                    realDestinations.append(destination)
                
                for d in realDestinations:
                    aux[d] += score[source]/outDegree

        # For each node
        for v in range(0, N):
            score[v] = alpha * aux[v] + (1 - alpha) * (1/N)
            aux[v] = 0

    return score

# Run page rank for nonspam
ranked_nonspam = nonspam_pagerank("uk-2007-05.hostgraph_weighted.graph-txt.gz", 20, .85, is_spam)

# Obtain list of nonspam hosts sorted by PageRank score [(id, score)]
ranked_nonspam = list(enumerate(ranked_nonspam))

# Sort list by scores, greatest to least
ranked_nonspam = sorted(ranked_nonspam, key=lambda host: host[1], reverse=True)

# Filter list by hostnames containing '.co.uk'
co_uk_nonspam = [n for n in ranked_nonspam if '.co.uk' in id2name[n[0]]]

# Filter list by hostnames containing '.co.uk'
gov_uk_nonspam = [n for n in ranked_nonspam if '.gov.uk' in id2name[n[0]]]

# ==============================================================================
# 6. Compute spam gain
# ==============================================================================

# Initialize empty dict for {id: gain}
gain = {}

# Compute spam gain
for i in range(len(ranked_nodes)):
    currHost = ranked_nodes[i][0]
    normalRank = ranked_nodes[i][1]
    noSpamRank = ranked_nonspam[i][1]

    gain[currHost] = normalRank/noSpamRank

# Sort gain into list of tuples [(id, gain)]
sorted_gain = sorted(gain.items(), key=lambda x: x[1], reverse=True)

# ==============================================================================
# 7. Compute one variant of PageRank
# ==============================================================================

"""
pagerank_variant() reads in a compressed graph file and computes scores for 
    each node of a certain domain
    input:
        compressedGraphFile - compressed graph file name
        iterations - integer for number of iterations
        alpha - float
        ids - id dict {id: host name}
        domain - string (i.e. '.co.uk', '.gov.uk')
    output:
        score vector
"""
def pagerank_variant(compressedGraphFile, iterations, alpha, ids, domain):

    # Get number of nodes
    N = 0
    # Number of nodes in induced subgraph
    induced_N = 0

    with gzip.open(compressedGraphFile, "rt", encoding="utf-8") as input_file:
        reader = csv.reader(input_file, delimiter=' ')
        # Set total number of nodes
        N = int(next(reader)[0])

        # Count nodes with domain
        for source, destinations in enumerate(reader):
            if domain in ids[source]:
                induced_N += 1
                
    # Initialize score vector to 1/induced_N for nodes we care about and 0 otherwise
    # Initialize aux vector to 0 for all entries
    score = []
    aux = []
    for i in range(0, N):
        aux.append(0)
        if domain in ids[i]:
            score.append(1/induced_N)
        else:
            score.append(0)

    # Run for `iterations` iterations
    for i in range(0, iterations):
        # Open compressed file
        with gzip.open(compressedGraphFile, "rt", encoding="utf-8") as input_file:
            reader = csv.reader(input_file, delimiter=' ')
            # Skip first line
            next(reader)
            # Go through each row
            # source (row number), destinations ([destination:weight])
            for source, destinations in enumerate(reader):

                # if the source is part of domain
                if domain in ids[source]:

                    # keep track of destinations we care about
                    realDestinations = []

                    # iterate through all possible destinations
                    for d in destinations:
                        # current destination, drop weight
                        destination = int(d.split(':', 1)[0])

                        # if destination is part of domain
                        if domain in ids[destination]:
                            realDestinations.append(destination)
                    
                    for d in realDestinations:
                        aux[d] += score[source]/len(realDestinations)
        # For each node
        for v in range(0, N):
            if domain in ids[v]:
                score[v] = alpha * aux[v] + (1 - alpha) * (1/induced_N)
            aux[v] = 0

    
    return score

# Run page rank variant
ranked_variant = pagerank_variant("uk-2007-05.hostgraph_weighted.graph-txt.gz", 20, .85, id2name, '.co.uk')

# Obtain list of hosts sorted by PageRank score [(id, score)]
ranked_variant = list(enumerate(ranked_variant))

# Sort list by scores, greatest to least
ranked_variant = sorted(ranked_variant, key=lambda host: host[1], reverse=True)

# ==============================================================================
# Function library
# ==============================================================================

"""
print_ranking() prints a list of hosts and score in the format 
    1. Host site [score]
    ...
    numHost. Host site [score]

    input:
        numHost - list length
        scoreList - sorted list of scores
        ids - dict of spam IDs {id: host site}
    output:
        N/As
"""
def print_ranking(numHost, scoreList, ids):
    for i in range(0, numHost):
        print(str(i+1) + ". " + ids[scoreList[i][0]] + " [" + str(scoreList[i][1]) + "]")

# ==============================================================================
# Testing
# ==============================================================================

print("With Spam")
print_ranking(20, ranked_nodes, id2name)

print("\nNonspam")
print_ranking(20, ranked_nonspam, id2name)

print("\nWith Spam co.uk")
print_ranking(20, co_uk_hosts, id2name)

print("\nNonspam co.uk")
print_ranking(20, co_uk_nonspam, id2name)

print("\nWith Spam gov.uk")
print_ranking(20, gov_uk_hosts, id2name)

print("\nNonspam gov.uk")
print_ranking(20, gov_uk_nonspam, id2name)

print("\nSpam gain")
print_ranking(20, sorted_gain, id2name)

print("\nVariant")
print_ranking(20, ranked_variant, id2name)