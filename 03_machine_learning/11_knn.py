import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "ml_dataset_full_metrics.csv"
)

X = df.drop(
    columns=["Condition"]
)

y = df["Condition"]

# ==========================================
# SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=42,

    stratify=y

)

# ==========================================
# SCALE
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ==========================================
# MODEL
# ==========================================

model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(
    X_train,
    y_train
)

# ==========================================
# PREDICT
# ==========================================

y_pred = model.predict(
    X_test
)

# ==========================================
# RESULTS
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("KNN RESULTS")
print("==============================")

print("\nAccuracy:")
print(round(accuracy,4))

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