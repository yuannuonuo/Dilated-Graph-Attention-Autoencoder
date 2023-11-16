import torch
from tqdm import tqdm
from torch_sparse import SparseTensor
from torch import Tensor,LongTensor
import torch_geometric
import scipy.sparse as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader,NeighborSampler
from torch_geometric.utils import to_networkx
from torch_geometric.nn import MessagePassing

# def draw(graph):
#     nids = graph.n_id
#     graph = to_networkx(graph)
#     for i, nid in enumerate(nids):
#         graph.nodes[i]['txt'] = str(nid.item())
#     node_labels = nx.get_node_attributes(graph, 'txt')
#     print(node_labels)
#     nx.draw_networkx(graph, labels=node_labels, node_color='#00BFFF')
#     plt.axis("off")
#     plt.show()
dataset = Planetoid(root="./Cora",name="Cora") #将数据保存在当前目录下
dilationRateList=[1,3]
maxDilationRate=max(dilationRateList)
neighborLayerSizes=[-1 for i in range(maxDilationRate)]
coraData=dataset[0]
nodesNum=coraData.x.shape[0]
adj=torch.FloatTensor(nodesNum,nodesNum).fill_(0)
loader=NeighborSampler(coraData.edge_index, sizes=neighborLayerSizes,batch_size=1,shuffle=True,num_workers=4)
# print(loader.sample([0])[1])
# print(loader.sample([0])[2][-1-1].edge_index)
# print(loader.sample([0])[1][loader.sample([0])[2][-1-1].edge_index])
# for counter in range(nodesNum):
for counter in tqdm(range(nodesNum)):
    processedNodeList=[counter]
    noProcessedNodeList=[]
    user_index=[counter]
    samples=loader.sample(user_index)
    sampleNids=samples[1]
    samplesAdjs=samples[2]
    for dilationRate in range(1,maxDilationRate+1):
        currentSampleAdj = None
        if maxDilationRate == 1:
            currentSampleAdj = sampleNids[samplesAdjs.edge_index]  # mapping n_id to Adj
        else:
            currentSampleAdj = sampleNids[
                samplesAdjs[-1 - (dilationRate - 1)].edge_index]  # mapping n_id to Adj
        sourceNodesList = currentSampleAdj[0]
        targetNodesList = currentSampleAdj[1]
        nodesLength = int(sourceNodesList.shape[0])
        if dilationRate in dilationRateList:
            for nodeCounter in range(nodesLength):
                sourceNode=int(sourceNodesList[nodeCounter])
                targetNode=int(targetNodesList[nodeCounter])
                if (sourceNode not in processedNodeList) and (sourceNode not in noProcessedNodeList):
                    adj[sourceNode, counter] = 1/dilationRate
                    adj[counter, sourceNode] = 1/dilationRate
                    processedNodeList.append(sourceNode)
        else:
            for nodeCounter in range(nodesLength):
                sourceNode = int(sourceNodesList[nodeCounter])
                targetNode = int(targetNodesList[nodeCounter])
                if sourceNode not in processedNodeList:
                    noProcessedNodeList.append(sourceNode)
adj=adj+torch.FloatTensor(sp.eye(adj.shape[0]).todense())



# for counter in range(coraData.train_mask.shape):
#     print(coraData[counter])
# coraData.n_id = torch.arange(coraData.num_nodes)
# adj = torch_geometric.utils.to_scipy_sparse_matrix(coraData.edge_index)
# num_neighbors = [2,2]
# input_nodes = torch.tensor([0])
# directed = True
# train_index=Tensor([1,2,3,4,5,6,7,8,9,10])

# loader=NeighborSampler(coraData.edge_index, node_idx=train_index,sizes=[-1],batch_size=1,shuffle=True,num_workers=4)
# for i in range(1,len(train_index)+1):
#     user_index=[i]
#     print(loader.sample(user_index))

# loader=NeighborLoader(coraData, num_neighbors=num_neighbors,input_nodes=None,directed=True,batch_size=1)
# for batch in loader:
#     print(batch.n_id)
#     edge_index=batch.edge_index
#     print(edge_index)
#     adj_t = SparseTensor(row=edge_index[1], col=edge_index[0])
#     print(adj_t)
# sampled_data = next(iter(loader))
# print(sampled_data)
# edge_2 = np.array(sampled_data.edge_index).T
# edge_2 = edge_2.tolist()
# edge_2 = list(tuple(line) for line in edge_2)
# G_2 = nx.Graph()
# G_2.add_edges_from(edge_2)
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# option = {'font_family':'serif', 'font_size':'15', 'font_weight':'semibold'}
# nx.draw_networkx(G_2, node_size=400, **option)
# plt.show()
# input_nodes = torch.tensor([124])
# loader=NeighborLoader(coraData, num_neighbors=num_neighbors,input_nodes=input_nodes,directed=True)
# sampled_data1 = next(iter(loader))
# print(sampled_data1.n_id)

# from torch_geometric.datasets import OGB_MAG
# from torch_geometric.loader import NeighborLoader
#
# hetero_data = OGB_MAG("./OGB_MAG")[0]
# loader = NeighborLoader(
#         hetero_data,
#          # Sample 30 neighbors for each node and edge type for 2 iterations
#         num_neighbors={key: [30] * 2 for key in hetero_data.edge_types},
#         # Use a batch size of 128 for sampling training nodes of type paper
#         batch_size=128,
#         input_nodes=('paper', hetero_data['paper'].train_mask),
# )
#
# sampled_hetero_data = next(iter(loader))
# print(sampled_hetero_data['paper'].batch_size)


# dataset = Planetoid(root="./Cora",name="Cora") #将数据保存在当前目录下
# coraData=dataset[0]
# print(coraData)
# edge_index=coraData.edge_index
# train_loader = NeighborSampler(edge_index, node_idx=LongTensor([0]),sizes=[10], shuffle=True,num_workers=4)
# print(train_loader.sample(batch=LongTensor([8])))