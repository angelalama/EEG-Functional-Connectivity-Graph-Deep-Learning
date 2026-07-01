import os
import numpy as np
import torch
import pandas as pd

from torch_geometric.data import Data

"""
Create PyTorch Geometric graph objects from EEG functional connectivity matrices.

Before running this script:

1. Generate the connectivity matrices using the preprocessing pipeline.
2. Update the paths below to match your local project directory.
"""

# ======================================================
# PROJECT PATHS
#
# Update these paths to match your local project
# directory before running the script.
#
# project_folder = r"C:\Users\angel\Documents\GHIBLI\ENIAC 2026"
# data_folder = os.path.join(project_folder, "connectivity_matrices")
# electrodes_path = os.path.join(project_folder, "electrodes.tsv")
# ======================================================


# ============================================
# FOLDER
# ============================================

data_folder = r"C:\Users\angel\Documents\GHIBLI\ENIAC 2026\connectivity_matrices"

# ============================================
# LOAD ELECTRODES (TSV)
# ============================================

electrodes_path = r"C:\Users\angel\Documents\GHIBLI\ENIAC 2026\electrodes.tsv"

electrodes = pd.read_csv(electrodes_path, sep="\t")

labels = electrodes["name"].tolist()

# ============================================
# GRAPH CONSTRUCTION PARAMETERS
# ============================================
# Correlation threshold used to define graph edges
threshold = 0.5

# ============================================
# GRAPH LIST
# ============================================

graph_dataset = []

# ============================================
# LOOP OVER SUBJECTS
# ============================================

for file in os.listdir(data_folder):

    if not file.endswith(".npy"):
        continue

    matrix_path = os.path.join(data_folder, file)

    connectivity = np.load(matrix_path)

    n_nodes = connectivity.shape[0]

    # ========================================
    # NODE FEATURES
    # ========================================

    x = torch.eye(n_nodes, dtype=torch.float)

    # ========================================
    # EDGES
    # ========================================

    adj = np.abs(connectivity) > threshold
    np.fill_diagonal(adj, False)

    rows, cols = np.where(adj)

    edge_index = torch.tensor(
        np.array([rows, cols]),
        dtype=torch.long
    )

    edge_weight = torch.tensor(
        connectivity[rows, cols],
        dtype=torch.float
    )

    # ========================================
    # ASSIGN GRAPH LABEL
    # ========================================

    if "open" in file:
        label = 0
    else:
        label = 1

    y = torch.tensor([label], dtype=torch.long)

    # ========================================
    # GRAPH OBJECT
    # ========================================

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weight,
        y=y
    )

    # STORE EEG ELECTRODE NAMES
    graph.node_names = labels

    graph_dataset.append(graph)

# ============================================
# SAVE DATASET
# ============================================

torch.save(
    graph_dataset,
    "eeg_graph_dataset.pt"
)

print("\n================================")
print("Dataset created WITH EEG labels!")
print("================================")

print("Number of graphs:", len(graph_dataset))
print("Nodes per graph:", graph_dataset[0].num_nodes)
print("Label example:", graph_dataset[0].y.item())

print("\nExample node mapping:")
for i in range(5):
    print(i, "→", graph_dataset[0].node_names[i])