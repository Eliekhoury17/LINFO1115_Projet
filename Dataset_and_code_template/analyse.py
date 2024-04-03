import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Task 1
def Q1_modified(dataframe):
    adjacency_list = {}
    for index, row in dataframe.iterrows():
        src, dst = row['Src'], row['Dst']
        if src not in adjacency_list:
            adjacency_list[src] = []
        if dst not in adjacency_list:
            adjacency_list[dst] = []
        adjacency_list[src].append(dst)
        adjacency_list[dst].append(src)  # Pour un graphe non dirigé

    all_nodes_degrees = [len(adjacency_list[node]) for node in adjacency_list]
    average_degree = sum(all_nodes_degrees) / len(all_nodes_degrees)
    
    return [average_degree,all_nodes_degrees]

def plot_degree_distribution(all_nodes_degrees):
    plt.figure(figsize=(10, 6))
    plt.hist(all_nodes_degrees, bins=range(min(all_nodes_degrees), max(all_nodes_degrees) + 1, 1), alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution des degrés dans le réseau électrique')
    plt.xlabel('Degré')
    plt.ylabel('Nombre de nœuds')
    plt.xticks(range(min(all_nodes_degrees), max(all_nodes_degrees) + 1, 1))  # Ajustez selon la distribution
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('histogram-of-the-degree-distribution.pdf')
    plt.show()

# Lire le fichier CSV
df = pd.read_csv('powergrid.csv')

# Appeler Q1_modified pour obtenir les résultats et la liste des degrés
_, all_nodes_degrees = Q1_modified(df)

# Tracer l'histogramme
plot_degree_distribution(all_nodes_degrees)


#Task 2

def plot_cumulative_similarity(similarity_scores):
    # Trier les scores de similarité
    sorted_scores = sorted(similarity_scores)
    
    # Calculer la distribution cumulative
    cumulative_distribution = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_scores, cumulative_distribution, linestyle='-', color='blue')  # Utiliser une ligne solide
    plt.title('Distribution Cumulative des Scores de Similarité')
    plt.xlabel('Score de Similarité')
    plt.ylabel('Distribution Cumulative (%)')
    plt.grid(True)
    plt.savefig('Distribution-Cumulative-des-Scores-de-Similarité.pdf')
    plt.show()

def Q2_modified(dataframe):
    # Construire une liste d'adjacence à partir du DataFrame
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
            # Pour s'assurer que chaque paire est traitée une seule fois
            if src < dst:
                neighbors_u = set(adjacency_list[src])
                neighbors_v = set(adjacency_list[dst])
                common_neighbors = neighbors_u.intersection(neighbors_v)
                all_neighbors = neighbors_u.union(neighbors_v)
                
                if len(all_neighbors) > 0:  # Éviter la division par zéro
                    similarity_score = len(common_neighbors) / len(all_neighbors)
                    similarity_scores.append(similarity_score)
    
    # Calculer le score de similarité moyen
    average_similarity_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
    
    return similarity_scores

# Générer et afficher le graphique
plot_cumulative_similarity(Q2_modified(df))



# Task 4:
import matplotlib.pyplot as plt

# Les résultats fournis pour la Tâche 4, représentant le nombre de chemins par longueur de chemin
path_counts = [13188, 32070, 60992, 104216, 161518, 231116, 317050, 417178, 527538, 643300, 760572, 876378, 993332, 1106938, 1212646, 1303336, 1364872, 1387570, 1388020, 1371436, 1333408, 1280458, 1222186, 1151852, 1063390, 944232, 800454, 648234, 499750, 366986, 260126, 179052, 121462, 84140, 59208, 42164, 30202, 20678, 12908, 7356, 4008, 1918, 738, 260, 88, 16]

# Générer le graphique
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(path_counts) + 1), path_counts, marker='o', linestyle='-', color='blue')
plt.title('Distribution des Longueurs des Chemins les Plus Courts dans le Réseau')
plt.xlabel('Longueur du Chemin')
plt.ylabel('Nombre de Chemins')
plt.grid(True)
plt.xticks(range(1, len(path_counts) + 1))  # Assurez-vous que chaque longueur de chemin est bien marquée
plt.tight_layout()  # Ajuste automatiquement les paramètres de la subplot pour qu'elle rentre dans la figure
plt.savefig('Distribution-des-Longueurs-des-Chemins-les-Plus-Courts-dans-le-Réseau.pdf')
plt.show()
