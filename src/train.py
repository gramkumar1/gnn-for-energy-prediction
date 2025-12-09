import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from src.gnn_model import GNNModel 
from src.data_loader import load_qm9_data_loaders

hidden_channels = 64
num_node_features = 11
num_targets = 1
learning_rate = 0.001
epochs = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss()
target_index = [7]

train_loader, val_loader, test_loader = load_qm9_data_loaders(
    root_dir='./data/QM9',
    batch_size=32,
    random_seed=67
)

model = GNNModel(
    num_node_features=num_node_features, 
    hidden_channels=hidden_channels, 
    num_targets=num_targets
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        num_atoms_batch = (data.ptr[1:] - data.ptr[:-1]).float().view(-1, 1)

        out = model(data, num_atoms=num_atoms_batch)

        num_original_targets = 19
        target_y = data.y.view(data.num_graphs, num_original_targets)[:, target_index]

        loss = criterion(out, target_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def validate(loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            num_atoms_batch = (data.ptr[1:] - data.ptr[:-1]).float().view(-1, 1)

            out = model(data, num_atoms=num_atoms_batch)

            num_original_targets = 19
            target_y = data.y.view(data.num_graphs, num_original_targets)[:, target_index]

            loss = criterion(out, target_y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


best_val_loss = float('inf')

print("Starting training...")
for epoch in range(1, epochs + 1):
    train_loss = train()
    val_loss = validate(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"Saved new best model at Epoch {epoch}")

    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print(f"Training finished. Best validation loss achieved: {best_val_loss:.4f}")
