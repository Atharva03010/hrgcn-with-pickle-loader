import torch
from torch import nn
from torch_geometric.nn import SAGEConv, to_hetero

class GNN(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = self.sigmoid(x)
        return x

class HRGCN(nn.Module):
    def __init__(self, graph, hidden_channels, out_channels, **kwargs):
        super().__init__()
        self.graph = graph
        self.model = GNN(hidden_channels, out_channels)
        self.model = to_hetero(self.model, self.graph.metadata(), aggr='sum')
        self.svdd_center = None

    def forward(self, x_dict, edge_index_dict):
        return self.model(x_dict, edge_index_dict)

    def set_svdd_center(self, center):
        self.svdd_center = center

    def get_svdd_center(self):
        return self.svdd_center