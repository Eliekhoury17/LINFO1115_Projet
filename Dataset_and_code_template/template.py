import pandas as pd
import numpy as np
import sys 
from template_utils import *

sys.setrecursionlimit(6000)

# Undirected graph
# Task 1: Average degree, number of bridges, number of local bridges
def Q1(dataframe):
    # Your code here
    return [0, 0, 0] # [average degree, nb bridges, nb local bridges]

# Undirected graph
# Task 2: Average similarity score between neighbors
def Q2(dataframe):
    # Your code here
    return 0.0 # the average similarity score between neighbors

# Directed graph
# Task 3: PageRank
def Q3(dataframe):
    # Your code here
    return [0, 0.0] # the id of the node with the highest pagerank score, the associated pagerank value.
    # Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-6)

# Undirected graph
# Task 4: Small-world phenomenon
def Q4(dataframe):
    # Your code here
    return [0, 0, 0, 0, 0] # at index 0 the number of shortest paths of lenght 0, at index 1 the number of shortest paths of length 1, ...
    # Note that we will ignore the value at index 0 as it can be set to 0 or the number of nodes in the graph

# Undirected graph
# Task 5: Betweenness centrality
def Q5(dataframe):
    # Your code here
    return [0, 0.0] # the id of the node with the highest betweenness centrality, the associated betweenness centrality value.

# you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

df = pd.read_csv('powergrid.csv')
print("Q1", Q1(df))
print("Q2", Q2(df))
print("Q3", Q3(df))
print("Q4", Q4(df))
print("Q5", Q5(df))
