import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATASET
# ==========================================
# ==========================================
# UPDATE THIS PATH BEFORE RUNNING
# folder = r"C:\Users\angel\Documents\GHIBLI\ENIAC 2026\connectivity_matrices"
# ==========================================

df = pd.read_csv(
    "ml_dataset_full_metrics.csv"
)

print(df.head())

# ==========================================
# FEATURES
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
# TARGET
# ==========================================

y = df["Condition"]

# ==========================================
# SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.30,

    stratify=y,

    random_state=42

)

print("\nTraining:", len(X_train))
print("Testing :", len(X_test))

# ==========================================
# MODEL
# ==========================================

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42

)

model.fit(
    X_train,
    y_train
)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

# ==========================================
# RESULTS
# ==========================================

print("\n==============================")
print("RANDOM FOREST RESULTS")
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

print("\nFeature Importance Ranking:")

for feature, importance in zip(
    X.columns,
    model.feature_importances_
):

    print(
        feature,
        round(importance, 4)
    )