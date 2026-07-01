"""
Evaluate the Graph Convolutional Network (GCN)
using stratified 5-fold cross-validation
on EEG functional connectivity graphs.
"""

import torch
import numpy as np
import random
import pandas as pd

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold

# =====================================
# REPRODUCIBILITY
# =====================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# =====================================
# LOAD DATASET
# =====================================

dataset = torch.load(
    "eeg_graph_dataset.pt",
    weights_only=False
)

print("Graphs:", len(dataset))

# =====================================
# LOAD ELECTRODE NAMES
# =====================================

electrodes = pd.read_csv(
    "electrodes.tsv",
    sep="\t",
    header=0
)

node_names = electrodes.iloc[:, 0].tolist()

print("Electrodes loaded:", len(node_names))
print("Example nodes:", node_names[:10])

# Assign electrode labels to each graph
for g in dataset:
    g.node_names = node_names

# =====================================
# LABELS
# =====================================

labels = np.array([
    g.y.item() for g in dataset
])

# =====================================
# GCN MODEL
# =====================================

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(61, 32)
        self.conv2 = GCNConv(32, 16)
        self.fc = torch.nn.Linear(16, 2)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x

# =====================================
# TRAINING FUNCTION
# =====================================

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

# =====================================
# EVALUATION FUNCTION
# =====================================

def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred = model(batch).argmax(dim=1)

            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total

# =====================================
# DEVICE
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# CROSS VALIDATION
# =====================================

kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_accuracies = []

# =====================================
# CROSS-VALIDATION LOOP
# =====================================

for fold, (train_idx, test_idx) in enumerate(
    kfold.split(np.arange(len(dataset)), labels)
):

    print("\n========================")
    print(f"FOLD {fold + 1}")
    print("========================")

    train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    model = GCN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # TRAIN
    for epoch in range(50):
        train(model, train_loader, optimizer, criterion, device)

    # EVALUATE
    accuracy = evaluate(model, test_loader, device)

    fold_accuracies.append(accuracy)

    print(f"Fold Accuracy: {accuracy:.4f}")

# =====================================
# FINAL RESULTS
# =====================================

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print("\n===================================")
print("5-FOLD CROSS VALIDATION RESULTS")
print("===================================")

print(f"Mean Accuracy: {mean_acc:.4f}")
print(f"Std Accuracy : {std_acc:.4f}")
print(f"{mean_acc*100:.2f} ± {std_acc*100:.2f}%")