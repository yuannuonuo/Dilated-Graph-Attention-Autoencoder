import gc
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
from torch import LongTensor


def encode_onehot(labels):                                   # 把标签转换成onehot
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def normalize(mx):                                          # 归一化
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def  load_data(path="./data/cora/", dataset="cora"):
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    dict = {int(element):i for i,element in enumerate(idx_features_labels[:, 0:1].reshape(-1))}
    labels = encode_onehot(idx_features_labels[:, -1])
    e = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = []
    for i, x in enumerate(e):
        edges.append([dict[e[i][0]], dict[e[i][1]]])
        edges.append([dict[e[i][1]], dict[e[i][0]]])
    sList = []
    tList = []
    del idx_features_labels
    del dict
    del e
    gc.collect()
    for item in edges:
        sList.append(item[0])
        tList.append(item[1])
        sList.append(item[1])
        tList.append(item[0])
    features = normalize(features)
    features = torch.tensor(np.array(features.todense()), dtype=torch.float32)
    labels = torch.LongTensor(np.where(labels)[1])
    data = Data(x=features, edge_index=LongTensor([sList, tList]), y=labels)
    edges = torch.tensor(edges, dtype=torch.int64).T
    return features, edges, labels, data
