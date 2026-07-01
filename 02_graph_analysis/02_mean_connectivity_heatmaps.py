"""
Visualize the average functional connectivity matrices
for Eyes Open and Eyes Closed conditions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# FOLDER
# Update this path before running the script
# =====================================

data_folder = r"C:\Users\angel\Documents\GHIBLI\ENIAC 2026\connectivity_matrices"

# =====================================
# LOAD ALL MATRICES
# =====================================

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

# =====================================
# CONVERT TO ARRAYS
# =====================================

open_matrices = np.array(open_matrices)
closed_matrices = np.array(closed_matrices)

print("Open matrices:", open_matrices.shape)
print("Closed matrices:", closed_matrices.shape)

# =====================================
# MEAN MATRICES
# =====================================

mean_open = np.mean(
    open_matrices,
    axis=0
)

mean_closed = np.mean(
    closed_matrices,
    axis=0
)

difference = (
    mean_closed
    - mean_open
)

# =====================================
# FIGURE
# =====================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5),
    constrained_layout=True
)

# =====================================
# OPEN
# =====================================

im1 = axes[0].imshow(
    mean_open,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

axes[0].set_title(
    "Mean Eyes Open (n=60)"
)

axes[0].set_xlabel(
    "EEG Channels"
)

axes[0].set_ylabel(
    "EEG Channels"
)

# =====================================
# CLOSED
# =====================================

im2 = axes[1].imshow(
    mean_closed,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

axes[1].set_title(
    "Mean Eyes Closed (n=60)"
)

axes[1].set_xlabel(
    "EEG Channels"
)

axes[1].set_ylabel(
    "EEG Channels"
)

# =====================================
# DIFFERENCE
# =====================================

im3 = axes[2].imshow(
    difference,
    cmap="bwr",
    vmin=-0.3,
    vmax=0.3
)

axes[2].set_title(
    "Difference (Closed - Open)"
)

axes[2].set_xlabel(
    "EEG Channels"
)

axes[2].set_ylabel(
    "EEG Channels"
)

# =====================================
# COLORBARS
# =====================================

cbar1 = fig.colorbar(
    im2,
    ax=axes[:2],
    shrink=0.8
)

cbar1.set_label(
    "Pearson Correlation"
)

cbar2 = fig.colorbar(
    im3,
    ax=axes[2],
    shrink=0.8
)

cbar2.set_label(
    "Difference"
)

# =====================================
# SAVE
# =====================================

plt.savefig(
    "mean_connectivity_heatmaps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================
# SUMMARY
# =====================================

print("\n==============================")
print("AVERAGE CONNECTIVITY ANALYSIS")
print("==============================")

print(
    "Mean Open Connectivity:",
    np.mean(mean_open)
)

print(
    "Mean Closed Connectivity:",
    np.mean(mean_closed)
)

print(
    "Mean Difference:",
    np.mean(difference)
)

print(
    "\nFigure saved:"
)

print(
    "mean_connectivity_heatmaps.png"
)