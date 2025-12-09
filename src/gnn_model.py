import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_targets):
        super(GNNModel, self).__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.3) 
        input_size_mlp = hidden_channels + 1
        self.layer3 = nn.Linear(input_size_mlp, 32) 
        self.layer4 = nn.Linear(32, num_targets)

    def forward(self, data, num_atoms):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch) 
        x = torch.cat([x, num_atoms], dim=1) 
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        
        return x
