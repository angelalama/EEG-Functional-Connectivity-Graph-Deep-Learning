import numpy as np

from statsmodels.stats.multitest import multipletests

# ==========================================
# LOAD P-VALUES
# ==========================================

p_matrix = np.load(
    "p_matrix.npy"
)

print("Matrix shape:",
      p_matrix.shape)

# ==========================================
# EXTRACT UPPER TRIANGLE
# ==========================================

n_channels = p_matrix.shape[0]

upper_idx = np.triu_indices(
    n_channels,
    k=1
)

p_values = p_matrix[
    upper_idx
]

print(
    "Number of tests:",
    len(p_values)
)

# ==========================================
# FDR CORRECTION
# ==========================================

rejected, p_corrected, _, _ = multipletests(

    p_values,

    alpha=0.05,

    method="fdr_bh"

)

# ==========================================
# CREATE SIGNIFICANCE MATRIX
# ==========================================

significance_matrix = np.zeros(
    (n_channels, n_channels),
    dtype=int
)

significance_matrix[
    upper_idx
] = rejected.astype(int)

# Hacer la matriz simétrica

significance_matrix = (

    significance_matrix
    +
    significance_matrix.T

)

# ==========================================
# SAVE
# ==========================================

np.save(

    "fdr_significant_matrix.npy",

    significance_matrix

)

# ==========================================
# RESULTS
# ==========================================

n_significant = np.sum(
    rejected
)

print("\n========================")
print("FDR CORRECTION FINISHED")
print("========================")

print(
    "Connections surviving FDR:",
    n_significant
)

print(
    f"Percentage: {(100*n_significant/len(p_values)):.2f}%"
)
