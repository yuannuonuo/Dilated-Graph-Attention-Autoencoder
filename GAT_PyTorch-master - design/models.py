import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT_MultiLayer(nn.Module):
    def __init__(self, nfeat, nhid, nhid_list, nclass, dropout, alpha, nheads, residual_status, freeze_status):
        """Dense version of GAT."""
        super(GAT_MultiLayer, self).__init__()
        self.residual = residual_status
        if self.residual==True:

            self.dropout = dropout

            #Structure Learning
            self.attentions_1 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
            for i, attention in enumerate(self.attentions_1):
                if freeze_status == 1:  # freeze structure learning layer
                    self.add_module('attention_1_{}'.format(i), attention.requires_grad_(False))
                else:
                    self.add_module('attention_1_{}'.format(i), attention)



            # Nodes Learning
            self.attentionList=[]
            for i,layer_nhid in enumerate(nhid_list):
                currentAttentions=None
                if i>=1:
                    currentAttentions = [GraphAttentionLayer(nhid_list[i-1]* nheads, layer_nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
                    self.attentionList.append(currentAttentions)
                    for counter, attention in enumerate(currentAttentions):
                        if freeze_status == 2:  # freeze nodes learning layer
                           self.add_module("attention_"+str(i+2)+"_{}".format(counter), attention.requires_grad_(False))
                        else:
                           self.add_module("attention_" + str(i + 2) + "_{}".format(counter), attention)

            self.out_att = GraphAttentionLayer((nhid_list[0]+nhid_list[-1])*nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        else:
            self.dropout = dropout

            self.attentions_1 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
            for i, attention in enumerate(self.attentions_1):
                if freeze_status == 2:  # freeze structure learning layer
                    self.add_module('attention_1_{}'.format(i), attention)
                else:
                    self.add_module('attention_1_{}'.format(i), attention)

            self.attentionList = []
            for i, layer_nhid in enumerate(nhid_list):
                currentAttentions = None
                if i == 1:
                    currentAttentions = [GraphAttentionLayer(nhid * nheads, layer_nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
                    self.attentionList.append(currentAttentions)
                    for counter, attention in enumerate(currentAttentions):
                        if freeze_status == 2:  # freeze nodes learning layer
                            self.add_module("attention_" + str(i + 2) + "_{}".format(counter),
                                            attention.requires_grad_(False))
                        else:
                            self.add_module("attention_" + str(i + 2) + "_{}".format(counter), attention)
                elif i > 1:
                    currentAttentions = [GraphAttentionLayer(nhid_list[i - 1] * nheads, layer_nhid, dropout=dropout, alpha=alpha,concat=True) for _ in range(nheads)]
                    self.attentionList.append(currentAttentions)
                    for counter, attention in enumerate(currentAttentions):
                        if freeze_status == 2:  # freeze nodes learning layer
                            self.add_module("attention_" + str(i + 2) + "_{}".format(counter),attention.requires_grad_(False))
                        else:
                            self.add_module("attention_" + str(i + 2) + "_{}".format(counter), attention)

            self.out_att = GraphAttentionLayer(nhid_list[-1] * nheads, nclass, dropout=dropout,alpha=alpha, concat=False)

    def dot_product_decode(self,Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    def freeze(self,layers):
        for layer in layers:
            for child in layer.children():
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, x, adjs):
        if self.residual==True:
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adjs[0]) for att in self.attentions_1], dim=1)
            x_pre = x
            x = F.dropout(x, self.dropout, training=self.training)
            for currentAttentionsCounter in range(len(self.attentionList)):
                currentAttentions=self.attentionList[currentAttentionsCounter]
                x = torch.cat([att(x, adjs[currentAttentionsCounter+1]) for att in currentAttentions], dim=1)
                x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([x, x_pre], dim=1)
            x = F.elu(self.out_att(x, adjs[0]))
            adj_pred = self.dot_product_decode(x)
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adjs[0]) for att in self.attentions_1], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
            for currentAttentionsCounter in range(len(self.attentionList)):
                currentAttentions = self.attentionList[currentAttentionsCounter]
                x = torch.cat([att(x, adjs[currentAttentionsCounter + 1]) for att in currentAttentions], dim=1)
                x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adjs[0]))
            adj_pred = self.dot_product_decode(x)
        return F.log_softmax(x, dim=1), adj_pred

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adjs):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adjs[0]) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adjs[0]))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

