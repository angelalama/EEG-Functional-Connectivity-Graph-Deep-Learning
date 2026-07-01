
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# LOAD DATA
# ==========================================

sig_matrix = np.load(
    "fdr_significant_matrix.npy"
)

electrodes = pd.read_csv(
    "electrodes.tsv",
    sep="\t",
    header=0
)

# ==========================================
# NAMES + COORDS
# ==========================================

node_names = electrodes.iloc[:,0].tolist()

x_coords = pd.to_numeric(
    electrodes.iloc[:,1],
    errors="coerce"
).values

y_coords = pd.to_numeric(
    electrodes.iloc[:,2],
    errors="coerce"
).values

# ==========================================
# BUILD GRAPH
# ==========================================

G = nx.Graph()

for i in range(len(node_names)):
    G.add_node(i)

for i in range(sig_matrix.shape[0]):

    for j in range(i+1, sig_matrix.shape[1]):

        if sig_matrix[i,j] == 1:

            G.add_edge(i,j)

# ==========================================
# REAL EEG POSITIONS
# ==========================================

pos = {}

for i in range(len(node_names)):

    pos[i] = (
        x_coords[i],
        y_coords[i]
    )

# ==========================================
# HUBS
# ==========================================

degree_dict = dict(G.degree())

node_sizes = [

    50 + degree_dict[node]*20

    for node in G.nodes()

]

# ==========================================
# PLOT
# ==========================================

plt.figure(figsize=(12,10))

nx.draw_networkx_edges(
    G,
    pos,
    alpha=0.4,
    width=0.5
)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes
)

labels = {
    i: node_names[i]
    for i in range(len(node_names))
}

nx.draw_networkx_labels(
    G,
    pos,
    labels,
    font_size=7
)

plt.title(
    "Significant EEG Functional Network"
)

plt.axis("equal")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "real_eeg_network.png",
    dpi=300
)

plt.show()

print("\nSaved: real_eeg_network.png")