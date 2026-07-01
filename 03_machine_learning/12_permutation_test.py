import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ==========================================
# LOAD DATA
# ==========================================
# ==========================================
# UPDATE THIS PATH BEFORE RUNNING
# folder = r"C:\Users\angel\Documents\GHIBLI\ENIAC 2026\connectivity_matrices"
# ==========================================


folder = r"C:\Users\angel\Documents\GHIBLI\ENIAC 2026\connectivity_matrices"

X = []
y = []

for file in os.listdir(folder):

    if file.endswith(".npy"):

        matrix = np.load(os.path.join(folder, file))

        features = matrix[np.triu_indices_from(matrix, k=1)]
        X.append(features)

        if "open" in file:
            y.append(0)
        elif "closed" in file:
            y.append(1)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# ==========================================
# RANDOM FOREST CLASSIFIER
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,   # ↓ reducido
    random_state=42,
    n_jobs=-1
)

# ==========================================
# REAL ACCURACY
# ==========================================

real_scores = cross_val_score(model, X, y, cv=3, n_jobs=-1)  # ↓ cv=3
real_acc = real_scores.mean()

print("\nREAL ACCURACY:", real_acc)

# ==========================================
# PERMUTATION TEST 
# ==========================================

print("\nRUNNING PERMUTATION TEST...")

n_perm = 200   # ↓ 1000 → 200 (mucho más rápido)
perm_accs = []

for i in range(n_perm):

    if i % 20 == 0:
        print("Permutation:", i)

    y_perm = np.random.permutation(y)

    scores = cross_val_score(model, X, y_perm, cv=3, n_jobs=-1)

    perm_accs.append(scores.mean())

perm_accs = np.array(perm_accs)

# ==========================================
# P-VALUE
# ==========================================

p_value = np.mean(perm_accs >= real_acc)

print("\n========================")
print("PERMUTATION TEST RESULT")
print("========================")
print("Real Accuracy:", real_acc)
print("Permutation Mean:", perm_accs.mean())
print("P-value:", p_value)

# ==========================================
# SAVE
# ==========================================

np.save("permutation_scores.npy", perm_accs)