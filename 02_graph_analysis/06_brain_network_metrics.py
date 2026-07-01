"""
Compute graph theory metrics from average EEG functional
connectivity networks for Eyes Open and Eyes Closed conditions.
"""

import os
import numpy as np
import networkx as nx
import pandas as pd

# ======================================================
# PROJECT PATHS
# Update these paths before running the script.
# ======================================================

project_folder = r"C:\Users\angel\Documents\GHIBLI\ENIAC 2026"

data_folder = os.path.join(
    project_folder,
    "connectivity_matrices"
)

electrodes_path = os.path.join(
    project_folder,
    "electrodes.tsv"
)

# ======================================================
# LOAD CONNECTIVITY MATRICES
# ======================================================

open_matrices = []
closed_matrices = []

for file in os.listdir(data_folder):

    if not file.endswith(".npy"):
        continue

    matrix = np.load(
        os.path.join(data_folder, file)
    )

    if "open" in file:
        open_matrices.append(matrix)

    elif "closed" in file:
        closed_matrices.append(matrix)

open_matrices = np.array(open_matrices)
closed_matrices = np.array(closed_matrices)

print("Open matrices:", open_matrices.shape)
print("Closed matrices:", closed_matrices.shape)

# ======================================================
# LOAD ELECTRODE LABELS
# ======================================================

electrodes = pd.read_csv(
    electrodes_path,
    sep="\t"
)

node_names = electrodes.iloc[:, 0].tolist()

n_nodes = open_matrices.shape[1]

# ======================================================
# COMPUTE MEAN CONNECTIVITY MATRICES
# ======================================================

open_mean = np.mean(
    open_matrices,
    axis=0
)

closed_mean = np.mean(
    closed_matrices,
    axis=0
)

# ======================================================
# BUILD GRAPH USING THE STRONGEST CONNECTIONS
# ======================================================

def build_graph(matrix, threshold_percent=95):

    threshold = np.percentile(
        matrix,
        threshold_percent
    )

    G = nx.Graph()

    for i in range(n_nodes):

        G.add_node(
            i,
            label=node_names[i]
        )

    for i in range(n_nodes):

        for j in range(i + 1, n_nodes):

            if matrix[i, j] >= threshold:

                G.add_edge(
                    i,
                    j,
                    weight=matrix[i, j]
                )

    return G

G_open = build_graph(open_mean)
G_closed = build_graph(closed_mean)

# ======================================================
# COMPUTE GRAPH METRICS
# ======================================================

def compute_metrics(G, condition):

    print(f"\n==================== {condition} ====================")

    degree = dict(G.degree())

    clustering = nx.clustering(G)

    betweenness = nx.betweenness_centrality(G)

    avg_degree = np.mean(
        list(degree.values())
    )

    avg_clustering = np.mean(
        list(clustering.values())
    )

    avg_betweenness = np.mean(
        list(betweenness.values())
    )

    print("Average degree:", avg_degree)
    print("Average clustering:", avg_clustering)
    print("Average betweenness:", avg_betweenness)

    print("\nTop hubs:")

    top_hubs = sorted(
        degree.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for node, deg in top_hubs:

        print(
            f"{node_names[node]}: degree {deg}"
        )

    return {

        "average_degree": avg_degree,

        "average_clustering": avg_clustering,

        "average_betweenness": avg_betweenness

    }

# ======================================================
# ANALYZE BOTH CONDITIONS
# ======================================================

metrics_open = compute_metrics(
    G_open,
    "OPEN EYES"
)

metrics_closed = compute_metrics(
    G_closed,
    "CLOSED EYES"
)

# ======================================================
# SAVE SUMMARY TABLE
# ======================================================

summary = pd.DataFrame([

    {
        "Condition": "Eyes Open",
        **metrics_open
    },

    {
        "Condition": "Eyes Closed",
        **metrics_closed
    }

])

summary.to_csv(
    "brain_network_metrics.csv",
    index=False
)

print("\n====================")
print("SUMMARY")
print("====================")

print(summary)

print("\nSaved: brain_network_metrics.csv")