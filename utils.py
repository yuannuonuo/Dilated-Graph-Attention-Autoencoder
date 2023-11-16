import gc
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from torch import Tensor,LongTensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader,NeighborSampler


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def generateAdj(data,adj,dilationRateList):
    print("Generate "+str(dilationRateList)+"'s Adj.")
    maxDilationRate = max(dilationRateList)
    neighborLayerSizes = [-1 for i in range(maxDilationRate)]
    nodesNum = data.x.shape[0]
    adj1 = torch.FloatTensor(nodesNum, nodesNum).fill_(0)
    loader = NeighborSampler(data.edge_index, sizes=neighborLayerSizes, shuffle=True, num_workers=4)
    for counter in tqdm(range(nodesNum)):
        processedNodeList = [counter]
        noProcessedNodeList = []
        user_index = [counter]
        samples = loader.sample(user_index)
        sampleNids = samples[1]
        samplesAdjs = samples[2]
        for dilationRate in range(1, maxDilationRate + 1):
            currentSampleAdj = None
            if maxDilationRate == 1:
                currentSampleAdj = sampleNids[samplesAdjs.edge_index]  # mapping n_id to Adj
            else:
                currentSampleAdj = sampleNids[
                    samplesAdjs[-1 - (dilationRate - 1)].edge_index]  # mapping n_id to Adj
            sourceNodesList = currentSampleAdj[0]
            nodesLength = int(sourceNodesList.shape[0])
            if dilationRate in dilationRateList:
                for nodeCounter in range(nodesLength):
                    sourceNode = int(sourceNodesList[nodeCounter])
                    if (sourceNode not in processedNodeList) and (sourceNode not in noProcessedNodeList):
                        adj1[sourceNode, counter] = 1 / dilationRate
                        adj1[counter, sourceNode] = 1 / dilationRate
                        processedNodeList.append(sourceNode)
            else:
                for nodeCounter in range(nodesLength):
                    sourceNode = int(sourceNodesList[nodeCounter])
                    if sourceNode not in processedNodeList:
                        noProcessedNodeList.append(sourceNode)
    adj1 = adj1 + torch.FloatTensor(sp.eye(adj.shape[0]).todense())
    return adj1

def generateAdjWithBatch(data,maxDilation,dilationRateList,batchNodes):
    maxDilationRate = max(dilationRateList)
    neighborLayerSizes = [-1 for i in range(maxDilation)]
    loader = NeighborSampler(data.edge_index, sizes=neighborLayerSizes, shuffle=True, num_workers=4)
    allSamples = loader.sample(batchNodes)
    allSampleNids = allSamples[1]
    adjSize=allSampleNids.shape[0]
    allSampleNidsList=list(map(int,allSampleNids.tolist()))
    adj1 = torch.FloatTensor(adjSize, adjSize).fill_(0)
    for counter in range(len(allSampleNidsList)):
        processedNodeList = [counter]
        noProcessedNodeList = []
        user_index = [allSampleNidsList[counter]]
        samples = loader.sample(user_index)
        sampleNids = samples[1]
        samplesAdjs = samples[2]
        for dilationRate in range(1, maxDilationRate + 1):
            currentSampleAdj = None
            if samplesAdjs.__class__!=[].__class__:
                currentSampleAdj = sampleNids[samplesAdjs.edge_index]
            else:
                currentSampleAdj = sampleNids[samplesAdjs[-1 - (dilationRate - 1)].edge_index]
            sourceNodesList = currentSampleAdj[0]
            nodesLength = int(sourceNodesList.shape[0])
            if dilationRate in dilationRateList:
                for nodeCounter in range(nodesLength):
                    sourceNode = int(sourceNodesList[nodeCounter])
                    if sourceNode in allSampleNidsList:
                        sourceNodeIndex = allSampleNidsList.index(sourceNode)
                        if (sourceNodeIndex not in processedNodeList) and (sourceNodeIndex not in noProcessedNodeList):
                            adj1[sourceNodeIndex, counter] = 1 / dilationRate
                            adj1[counter, sourceNodeIndex] = 1 / dilationRate
                            processedNodeList.append(sourceNodeIndex)
            else:
                for nodeCounter in range(nodesLength):
                    sourceNode = int(sourceNodesList[nodeCounter])
                    if sourceNode in allSampleNidsList:
                        sourceNodeIndex = allSampleNidsList.index(sourceNode)
                        if sourceNodeIndex not in processedNodeList:
                            noProcessedNodeList.append(sourceNodeIndex)
    adj1 = adj1 + torch.FloatTensor(sp.eye(adj1.shape[0]).todense())
    return adj1,allSampleNids


def load_data_batch(path="./data/cora/", dataset="cora",dilationRateSequence=[[1]],batchNodes=None,featuresList=None,LabelList=None,edgeList=None,trainIndexList=None):
    """Load citation network dataset (cora only for now)"""

    # idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)#ndarray float32
    features = sp.csr_matrix(featuresList, dtype=np.float32)  # ndarray float32
    labels = encode_onehot(LabelList)  # ndarray str
    # labels = encode_onehot(idx_features_labels[:, -1])#ndarray str

    # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx = np.array(trainIndexList, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)#ndarray int32
    edges_unordered = np.array(edgeList, dtype=np.int32)  # ndarray int32
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    batchNodes=[idx_map[node] for node in batchNodes]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = adj + sp.eye(adj.shape[0])
    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    sList = []
    tList = []
    for item in edges:
        sList.append(item[0])
        tList.append(item[1])
        sList.append(item[1])
        tList.append(item[0])
    coraData = Data(x=features, edge_index=LongTensor([sList, tList]),y=labels)
    maxDilation=max(max(dilationRateSequence))
    allSampleNids=None
    adj_batch_list = []
    for dilationRateListCounter in range(len(dilationRateSequence)):
        adj_batch, allSampleNidsList = generateAdjWithBatch(coraData, maxDilation, dilationRateSequence[dilationRateListCounter],batchNodes)
        if dilationRateListCounter==0:
            allSampleNids=allSampleNidsList
        adj_batch_list.append(adj_batch)
    real_adj = torch.FloatTensor(allSampleNids.shape[0], allSampleNids.shape[0]).fill_(0)
    for i in range(allSampleNids.shape[0]):
        for j in range(allSampleNids.shape[0]):
            real_adj[i][j] = adj[int(allSampleNids[i])][int(allSampleNids[j])]
    features_batch = features[allSampleNids]
    labels_batch = labels[allSampleNids]
    return real_adj, features_batch, labels_batch, adj_batch_list


def load_data(path="./data/cora/", dataset="cora",dilationRateSequence=[[1]]):
    """Load citation network dataset (cora only for now)"""

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0],dtype=np.float32), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = adj + sp.eye(adj.shape[0])
    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    sList = []
    tList = []
    for item in edges:
        sList.append(item[0])
        tList.append(item[1])
        sList.append(item[1])
        tList.append(item[0])
    coraData = Data(x=features, edge_index=LongTensor([sList, tList]),y=labels)
    adjList=[]
    for dilationRateList in dilationRateSequence:
        adjList.append(generateAdj(coraData, adj, dilationRateList))

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, adjList


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

