import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv

# 3.定义GAT网络
class SAGEConvNet(nn.Module):
    def __init__(self, num_node_features, hidden,num_classes):
        super(SAGEConvNet, self).__init__()
        self.conv1 = SAGEConv(num_node_features,hidden)
        self.conv2 = SAGEConv(hidden,num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)