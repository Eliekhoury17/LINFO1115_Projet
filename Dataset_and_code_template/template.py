import pandas as pd
import numpy as np
import sys 
from template_utils import *
from collections import defaultdict, deque

sys.setrecursionlimit(6000)

# Undirected graph
# Task 1: Average degree, number of bridges, number of local bridges
def Q1(dataframe):
    # Convert DataFrame to adjacency list representation
    adjacency_list = {}
    for index, row in dataframe.iterrows():
        src, dst = row['Src'], row['Dst']
        if src not in adjacency_list:
            adjacency_list[src] = []
        if dst not in adjacency_list:
            adjacency_list[dst] = []
        adjacency_list[src].append(dst)
        adjacency_list[dst].append(src)  # For undirected graph

    # Calculate average degree
    all_nodes_degrees = [len(adjacency_list[node]) for node in adjacency_list]
    average_degree = sum(all_nodes_degrees) / len(all_nodes_degrees)

    # Count bridges
    bridges_count = count_bridges(adjacency_list)

    # Count local bridges
    local_bridges_count = count_local_bridges(adjacency_list)

    return average_degree, bridges_count, local_bridges_count


# Undirected graph
# Task 2: Average similarity score between neighbors
def Q2(dataframe):
    # Construct an adjacency list from the DataFrame
    adjacency_list = {}
    for index, row in dataframe.iterrows():
        src, dst = row['Src'], row['Dst']
        if src not in adjacency_list:
            adjacency_list[src] = set()
        if dst not in adjacency_list:
            adjacency_list[dst] = set()
        adjacency_list[src].add(dst)
        adjacency_list[dst].add(src)

    similarity_scores = []
    for src, neighbors in adjacency_list.items():
        for dst in neighbors:
            # To ensure each pair is processed only once
            if src < dst:
                neighbors_u = set(adjacency_list[src])
                neighbors_v = set(adjacency_list[dst])
                common_neighbors = neighbors_u.intersection(neighbors_v)
                all_neighbors = neighbors_u.union(neighbors_v)
                
                if len(all_neighbors) > 0:  # Avoid division by zero
                    similarity_score = len(common_neighbors) / len(all_neighbors)
                    similarity_scores.append(similarity_score)
    
    # Calculate the average similarity score
    average_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    return average_similarity_score



# Directed graph
# Task 3: PageRank
def Q3(dataframe):
    # Initialize structures
    adjacency_list = {}
    pagerank_scores = {}
    incoming_links = {}
    damping_factor = 0.85
    tolerance = 1e-6
    num_nodes = 0

    # Construct adjacency list and initialize PageRank scores
    for index, row in dataframe.iterrows():
        src, dst = row['Src'], row['Dst']
        if src not in adjacency_list:
            adjacency_list[src] = set()
            pagerank_scores[src] = 1.0
            num_nodes += 1
        if dst not in adjacency_list:
            adjacency_list[dst] = set()
            pagerank_scores[dst] = 1.0
            num_nodes += 1
        adjacency_list[src].add(dst)
        if dst not in incoming_links:
            incoming_links[dst] = set()
        incoming_links[dst].add(src)

    # Iterate until convergence
    change = 1
    while change > tolerance:
        new_pagerank_scores = {}
        change = 0
        for node in pagerank_scores:
            rank_sum = 0
            if node in incoming_links:
                for incoming_node in incoming_links[node]:
                    rank_sum += pagerank_scores[incoming_node] / len(adjacency_list[incoming_node])
            new_rank = (1 - damping_factor) / num_nodes + damping_factor * rank_sum
            change += abs(new_rank - pagerank_scores[node])
            new_pagerank_scores[node] = new_rank
        pagerank_scores = new_pagerank_scores

    # Find the node with the highest PageRank score
    node_with_highest_pagerank = max(pagerank_scores, key=pagerank_scores.get)
    highest_pagerank_value = pagerank_scores[node_with_highest_pagerank]

    return [node_with_highest_pagerank, highest_pagerank_value]

# Undirected graph
# Task 4: Small-world phenomenon
def Q4(dataframe):
    # Construct adjacency list
    adjacency_list = {}
    for index, row in dataframe.iterrows():
        src, dst = row['Src'], row['Dst']
        if src not in adjacency_list:
            adjacency_list[src] = set()
        if dst not in adjacency_list:
            adjacency_list[dst] = set()
        adjacency_list[src].add(dst)
        adjacency_list[dst].add(src)
    
    # Calculate all pairs shortest path lengths
    all_pairs_shortest_paths = {}
    for node in adjacency_list:
        all_pairs_shortest_paths[node] = bfs_shortest_path(adjacency_list, node)
    
    # Extract lengths and calculate diameter
    lengths = [length for target_dict in all_pairs_shortest_paths.values() for length in target_dict.values() if length > 0]
    diameter = max(lengths)
    
    # Count occurrences of each path length
    result = [lengths.count(i) for i in range(1, diameter + 1)]
    
    return result

# Undirected graph
# Task 5: Betweenness centrality
def Q5(dataframe):
    # Construct adjacency list
    adjacency_list = {}
    for _, row in dataframe.iterrows():
        src, dst = row['Src'], row['Dst']
        if src not in adjacency_list:
            adjacency_list[src] = set()
        if dst not in adjacency_list:
            adjacency_list[dst] = set()
        adjacency_list[src].add(dst)
        adjacency_list[dst].add(src)

    # Compute betweenness centrality
    betweenness = compute_betweenness(adjacency_list)

    # Find the node with the highest betweenness centrality
    node_with_highest_betweenness = max(betweenness, key=betweenness.get)
    highest_betweenness_value = betweenness[node_with_highest_betweenness]

    return [node_with_highest_betweenness, highest_betweenness_value]




# you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

df = pd.read_csv('powergrid.csv')
print("Q1", Q1(df))
print("Q2", Q2(df))
print("Q3", Q3(df))
print("Q4", Q4(df))
print("Q5", Q5(df))
