"""
Train and evaluate a Support Vector Machine (SVM) classifier
using graph-theoretical features extracted from EEG
functional connectivity networks.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "ml_dataset_full_metrics.csv"
)

# ==========================================
# SELECT FEATURES
# ==========================================

X = df[[
    "MeanDegree",
    "MeanClustering",
    "MeanBetweenness",
    "Density",
    "GlobalEfficiency",
    "AveragePathLength",
    "Assortativity"
]]

# ==========================================
# TARGET VARIABLE
# ==========================================

y = df["Condition"]

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.30,

    stratify=y,

    random_state=42
)

# ==========================================
# FEATURE STANDARDIZATION
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# SUPPORT VECTOR MACHINE MODEL
# ==========================================

model = SVC(

    kernel="rbf",

    C=1.0,

    gamma="scale",

    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(
    X_train,
    y_train
)

# ==========================================
# MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

# ==========================================
# MODEL EVALUATION
# ==========================================

print("\n==============================")
print("SUPPORT VECTOR MACHINE RESULTS")
print("==============================")

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred
))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred
))