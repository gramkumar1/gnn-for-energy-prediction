import torch
import numpy as np
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

def load_qm9_data_loaders(root_dir='./data/QM9', batch_size=32, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=67):
    dataset = QM9(root=root_dir)
    for data in dataset:
        num_atoms = data.x.size(0)
        data.num_atoms = torch.tensor([[num_atoms]], dtype=torch.float)

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader