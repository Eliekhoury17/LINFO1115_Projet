# If needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

# First, import the libraries needed for your helper functions
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import math

# Then write the classes and/or functions you wishes to use in the exercises
def count_bridges(adjacency_list):
    visited = set()
    low = {}
    disc = {}
    parent = {}
    bridges_count = 0
    time = [0]  # Use a list for mutable integer

    def dfs(u):
        nonlocal bridges_count
        visited.add(u)
        disc[u] = low[u] = time[0] # Init discovery time and low value
        time[0] += 1

        for v in adjacency_list[u]:
            if v not in visited:  # Tree edge
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v]) # Check if the subtree starting from v has done new connection to ancestor of u
                if low[v] > disc[u]: # In that case, all subnodes generated have never been discovered and aren't reachable without passing by (u, v)
                    bridges_count += 1  # (u, v) is a bridge
            elif v != parent.get(u): # Back edge
                low[u] = min(low[u], disc[v]) # If v has been discovered before, it means that by an other path u should be discovered sooner too

    for node in adjacency_list:
        if node not in visited:
            dfs(node)

    return bridges_count

def count_local_bridges(adjacency_list):
    local_bridges_count = 0
    for src in adjacency_list:
        for dst in adjacency_list[src]:
            mutual_neighbors = set(adjacency_list[src]) & set(adjacency_list[dst])
            # Remove the direct connection to consider only mutual neighbors
            mutual_neighbors.discard(src)
            mutual_neighbors.discard(dst)
            if len(mutual_neighbors) == 0: # No other mutual neighbors so distance between src and dst > 2
                local_bridges_count += 1
    # Divide by 2 because each undirected edge is counted twice
    return local_bridges_count // 2

def bfs_shortest_path(adjacency_list, start):
    visited = set([start])
    queue = deque([(start, 0)])
    distances = {}
    
    while queue:
        current_node, current_distance = queue.popleft()
        distances[current_node] = current_distance
        
        for neighbor in adjacency_list[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_distance + 1))
    
    return distances

def compute_betweenness(adjacency_list):
    betweenness = {node: 0 for node in adjacency_list}
    
    for start_node in adjacency_list:
        # Initialize path counts and predecessors
        path_counts = {node: 0 for node in adjacency_list}
        path_counts[start_node] = 1
        predecessors = {node: [] for node in adjacency_list}
        
        # Breadth-first search from start_node
        queue = deque([start_node])
        levels = {start_node: 0}
        while queue:
            current_node = queue.popleft()
            for neighbor in adjacency_list[current_node]:
                # If first time seeing neighbor, discover it and set its level
                if neighbor not in levels:
                    queue.append(neighbor)
                    levels[neighbor] = levels[current_node] + 1
                # If this edge leads to the next level in the BFS tree
                if levels[neighbor] == levels[current_node] + 1:
                    path_counts[neighbor] += path_counts[current_node]
                    predecessors[neighbor].append(current_node)

        # Accumulate betweenness
        node_contributions = {node: 1 for node in adjacency_list}
        for node in sorted(levels, key=levels.get, reverse=True):
            for pred in predecessors[node]:
                if path_counts[pred] > 0:  # Check to avoid division by zero
                    contribution = node_contributions[node] * path_counts[pred] / path_counts[node]
                    node_contributions[pred] += contribution
                    betweenness[pred] += contribution

    # Normalize the betweenness scores (dividing by 2 for undirected graphs) and convert to integer
    for node in betweenness:
        betweenness[node] = round(betweenness[node] / 2)  # Use round, int, or math.ceil here as needed
    
    return betweenness
