"""
Train a Graph Convolutional Network (GCN)
to classify EEG functional connectivity graphs
into Eyes Open and Eyes Closed conditions.
"""

import torch
import numpy as np
import random

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import train_test_split

# =====================================
# REPRODUCIBILITY
# =====================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
# VERIFY ELECTRODE LABELS
# =====================================

print("\nExample EEG node names:")
print(dataset[0].node_names[:5])

# =====================================
# TRAIN TEST SPLIT
# =====================================

indices = np.arange(len(dataset))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=[g.y.item() for g in dataset]
)

train_dataset = [dataset[i] for i in train_idx]
test_dataset = [dataset[i] for i in test_idx]

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16
)

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
# SETUP
# =====================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GCN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

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

    return total_loss / len(loader)

# =====================================
# EVALUATION FUNCTION
# =====================================

def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred = model(batch)
            pred = pred.argmax(dim=1)

            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total

# =====================================
# TRAINING LOOP
# =====================================

epochs = 50

print("\n====================")
print("TRAINING GCN MODEL")
print("====================")

for epoch in range(epochs):

    loss = train(model, train_loader, optimizer, criterion, device)
    train_acc = test(model, train_loader, device)
    test_acc = test(model, test_loader, device)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Loss: {loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Test Acc: {test_acc:.4f}"
    )

# =====================================
# FINAL TEST
# =====================================

final_test_accuracy = test(model, test_loader, device)

print("\n====================")
print("TEST ACCURACY")
print("====================")

print(f"{final_test_accuracy:.4f}")

# =====================================
# EXAMPLE NODE MAPPING
# =====================================

print("\nEEG mapping available (example):")
print(dataset[0].node_names[:10])