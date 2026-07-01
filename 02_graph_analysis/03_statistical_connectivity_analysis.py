"""
Perform paired t-tests on each functional connection
between Eyes Open and Eyes Closed conditions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel

# =====================================
# FOLDER
# Update this path before running the script
# =====================================

data_folder = r"C:\Users\angel\Documents\GHIBLI\ENIAC 2026\connectivity_matrices"

# =====================================
# LOAD MATRICES
# =====================================

open_matrices = []
closed_matrices = []

for subject in range(1, 61):

    subject_name = f"sub-{subject:02d}"

    open_file = os.path.join(
        data_folder,
        f"{subject_name}_open.npy"
    )

    closed_file = os.path.join(
        data_folder,
        f"{subject_name}_closed.npy"
    )

    open_matrices.append(
        np.load(open_file)
    )

    closed_matrices.append(
        np.load(closed_file)
    )

open_matrices = np.array(open_matrices)
closed_matrices = np.array(closed_matrices)

print("Open shape:", open_matrices.shape)
print("Closed shape:", closed_matrices.shape)

# =====================================
# T-TEST MATRIX
# =====================================

n_channels = open_matrices.shape[1]

t_matrix = np.zeros(
    (n_channels, n_channels)
)

p_matrix = np.ones(
    (n_channels, n_channels)
)

# =====================================
# TEST EACH CONNECTION
# =====================================

for i in range(n_channels):

    for j in range(n_channels):

        open_values = open_matrices[:, i, j]

        closed_values = closed_matrices[:, i, j]

        t_stat, p_val = ttest_rel(

            closed_values,

            open_values

        )

        t_matrix[i, j] = t_stat

        p_matrix[i, j] = p_val

# =====================================
# SIGNIFICANCE MASK
# =====================================

significant = p_matrix < 0.05

# =====================================
# COUNT SIGNIFICANT CONNECTIONS
# =====================================

n_significant = np.sum(significant)

print("\n========================")
print("STATISTICAL RESULTS")
print("========================")

print(
    "Significant connections:",
    n_significant
)

# =====================================
# FIGURE
# =====================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18,5),
    constrained_layout=True
)

# -------------------------------------
# T VALUES
# -------------------------------------

im1 = axes[0].imshow(
    t_matrix,
    cmap="bwr"
)

axes[0].set_title(
    "T Statistics"
)

# -------------------------------------
# P VALUES
# -------------------------------------

im2 = axes[1].imshow(
    p_matrix,
    cmap="viridis"
)

axes[1].set_title(
    "P Values"
)

# -------------------------------------
# SIGNIFICANT
# -------------------------------------

im3 = axes[2].imshow(
    significant,
    cmap="gray_r"
)

axes[2].set_title(
    "P < 0.05"
)

# =====================================
# COLORBARS
# =====================================

fig.colorbar(
    im1,
    ax=axes[0]
)

fig.colorbar(
    im2,
    ax=axes[1]
)

# =====================================
# SAVE
# =====================================

plt.savefig(
    "statistical_connectivity_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================
# SAVE MATRICES
# =====================================

np.save(
    "t_matrix.npy",
    t_matrix
)

np.save(
    "p_matrix.npy",
    p_matrix
)

print("\nSaved:")
print("statistical_connectivity_analysis.png")
print("t_matrix.npy")
print("p_matrix.npy")