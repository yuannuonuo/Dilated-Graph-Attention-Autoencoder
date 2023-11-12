import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv

# 3.定义GAT网络
class GATNet(nn.Module):
    def __init__(self, num_node_features, hidden,num_classes,heads):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels=num_node_features,out_channels=hidden,heads=heads)
        self.conv2 = GATConv(in_channels=hidden*heads,out_channels=num_classes,heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)