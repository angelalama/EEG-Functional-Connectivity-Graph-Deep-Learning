import numpy as np
import pandas as pd
import networkx as nx

# ==========================================
# LOAD MATRICES
# ==========================================

open_matrices = np.load("open_matrices.npy")
closed_matrices = np.load("closed_matrices.npy")

print("Open:", open_matrices.shape)
print("Closed:", closed_matrices.shape)

# ==========================================
# GRAPH FEATURE EXTRACTION
# ==========================================

def extract_graph_metrics(matrix):

    threshold = 0.6

    G = nx.Graph()

    n_nodes = matrix.shape[0]

    for i in range(n_nodes):
        G.add_node(i)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):

            if abs(matrix[i, j]) > threshold:

                G.add_edge(
                    i,
                    j,
                    weight=float(matrix[i, j])
                )

    degree_dict = dict(G.degree())
    clustering_dict = nx.clustering(G)
    betweenness_dict = nx.betweenness_centrality(G)

    mean_degree = np.mean(
        list(degree_dict.values())
    )

    mean_clustering = np.mean(
        list(clustering_dict.values())
    )

    mean_betweenness = np.mean(
        list(betweenness_dict.values())
    )

    density = nx.density(G)

    global_efficiency = nx.global_efficiency(G)

    if nx.number_of_edges(G) > 0:

        largest_component = max(
            nx.connected_components(G),
            key=len
        )

        subgraph = G.subgraph(
            largest_component
        )

        if subgraph.number_of_nodes() > 1:

            avg_path_length = nx.average_shortest_path_length(
                subgraph
            )

        else:

            avg_path_length = np.nan

    else:

        avg_path_length = np.nan

    try:

        assortativity = nx.degree_assortativity_coefficient(
            G
        )

    except:

        assortativity = np.nan
# Return one feature vector per subject
    return [

        mean_degree,
        mean_clustering,
        mean_betweenness,
        density,
        global_efficiency,
        avg_path_length,
        assortativity

    ]

# ==========================================
# BUILD DATASET
# ==========================================

rows = []

# OPEN = 0

for matrix in open_matrices:

    features = extract_graph_metrics(matrix)

    rows.append(

        features + [0]

    )

# CLOSED = 1

for matrix in closed_matrices:

    features = extract_graph_metrics(matrix)

    rows.append(

        features + [1]

    )

# ==========================================
# DATAFRAME
# ==========================================

columns = [

    "MeanDegree",
    "MeanClustering",
    "MeanBetweenness",
    "Density",
    "GlobalEfficiency",
    "AveragePathLength",
    "Assortativity",
    "Condition"

]

df = pd.DataFrame(
    rows,
    columns=columns
)

print(df.head())

print("\nDataset shape:")
print(df.shape)

# ==========================================
# SAVE
# ==========================================

df.to_csv(
    "ml_dataset_full_metrics.csv",
    index=False
)

print("\nSaved:")
print("ml_dataset_full_metrics.csv")