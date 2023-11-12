from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import scipy.sparse as sp
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader,NeighborSampler

from utils import load_data, accuracy, load_data_batch
from models import GAT, SpGAT, GAT_MultiLayer

datasetName="twitter15"

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--multilayer', action='store_true', default=True, help='Usa multilayer or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_list', type=list, default=[100,50], help='Number of hidden units.')
parser.add_argument('--dilation_list', type=list, default=[[1],[3]], help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience.')
parser.add_argument('--batch', default=True, help='Usa Batch or not.')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size when using batch.')
parser.add_argument('--residual', default=True, help='Use Residual or not.')
parser.add_argument('--freeze',  type=int, default=1, help='Use Freeze or not.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

idx_batch = torch.LongTensor(range(args.batch_size))
def loadDataWithBatch_Random(rangeTuple):
    currentBatch=random.sample(rangeTuple,args.batch_size)
    real_adj, features, labels, adjList = load_data_batch(path="./data/cora/", dataset="cora",dilationRateSequence=args.dilation_list,batchNodes=currentBatch)
    if args.cuda:
        features = features.cuda()
        labels = labels.cuda()
        features_batch, labels_batch = Variable(features), Variable(labels)
        adjs_batch = []
        if args.cuda:
            for i in range(len(adjList)):
                adjs_batch.append(Variable(adjList[i].cuda()))
    return real_adj,features_batch,labels_batch,adjs_batch

trainIndexFile = open("./data/"+datasetName+"/train.txt","r")
testIndexFile = open("./data/"+datasetName+"/test.txt","r")
valIndexFile = open("./data/"+datasetName+"/val.txt","r")
edgeFile = open("./data/"+datasetName+"/"+datasetName+".cites","r")
nodesFile = open("./data/"+datasetName+"/"+datasetName+".content","r")
trainIndexFileLines = trainIndexFile.readlines()
testIndexFileLines = testIndexFile.readlines()
valIndexFileLines = valIndexFile.readlines()
edgeFileLines=edgeFile.readlines()
nodesFileLines=nodesFile.readlines()
nodesFeatureDict={}
nodesLabelDict={}
labelType=[]
for line in nodesFileLines:
    no=line.split("\t")[0]
    features=line.split("\t")[1:-1]
    label = line.split("\t")[-1].replace("\n","")
    nodesFeatureDict[no]=np.array(features)
    nodesLabelDict[no]=label
    if label not in labelType:
        labelType.append(label)
def generateBatch(indexFileLines,edgeFileLines,nodesLabelDict,nodesFeatureDict,trainStatus,num):
    indexFileLinesLength = len(indexFileLines)
    featuresList = []
    LabelList = []
    LabelDict = {}
    edgeList = []
    indexList = []
    indexs = []
    if trainStatus:
        while(True):
            batchIndexList = random.sample(range(indexFileLinesLength), args.batch_size)
            featuresList = []
            LabelList = []
            LabelTypeList = []
            edgeList = []
            indexList = []
            indexs = []
            for index in batchIndexList:
                currentIndex = indexFileLines[index].split("\t")[0]
                indexList.append(int(currentIndex))
                indexs.append(int(currentIndex))
                Label = indexFileLines[index].split("\t")[1].replace("\n","")
                LabelList.append(Label)
                if Label not in LabelTypeList:
                    LabelTypeList.append(Label)
            if len(LabelTypeList)>=len(labelType):
                break
            else:
                print("Less than "+str(len(labelType))+" types of labels, regenerating......")
        for line in edgeFileLines:
            id_1 = line.split("\t")[0]
            id_2 = line.split("\t")[1].replace("\n","")
            if int(id_1) in indexList:
                if int(id_2) not in indexList:
                    indexList.append(int(id_2))
                    LabelList.append(nodesLabelDict[id_2])
                if [int(id_1),int(id_2)] not in edgeList:
                    edgeList.append([int(id_1),int(id_2)])
            if int(id_2) in indexList:
                if int(id_1) not in indexList:
                    indexList.append(int(id_1))
                    LabelList.append(nodesLabelDict[id_1])
                if [int(id_1), int(id_2)] not in edgeList:
                    edgeList.append([int(id_1), int(id_2)])
        for index in indexList:
            featuresList.append(nodesFeatureDict[str(index)])
        featuresList=np.array(featuresList,dtype=np.float32)
        LabelList=np.array(LabelList)
        edgeList=np.array(edgeList,dtype=np.int32)
        real_adj, features_batch, labels_batch, adj_batch_list=load_data_batch(path="./data/twitter15/", dataset="twitter15",dilationRateSequence=args.dilation_list,batchNodes=indexs,featuresList=featuresList,LabelList=LabelList,edgeList=edgeList,trainIndexList=indexList)
        return real_adj, features_batch, labels_batch, adj_batch_list
    else:
        batchIndexList = range(num*args.batch_size,(num+1)*args.batch_size)
        featuresList = []
        LabelList = []
        edgeList = []
        indexList = []
        indexs = []
        for index in batchIndexList:
            if index<indexFileLinesLength:
                currentIndex = indexFileLines[index].split("\t")[0]
                indexList.append(int(currentIndex))
                indexs.append(int(currentIndex))
                Label = indexFileLines[index].split("\t")[1].replace("\n", "")
                LabelList.append(Label)
        for line in edgeFileLines:
            id_1 = line.split("\t")[0]
            id_2 = line.split("\t")[1].replace("\n","")
            if int(id_1) in indexList:
                if int(id_2) not in indexList:
                    indexList.append(int(id_2))
                    LabelList.append(nodesLabelDict[id_2])
                if [int(id_1),int(id_2)] not in edgeList:
                    edgeList.append([int(id_1),int(id_2)])
            if int(id_2) in indexList:
                if int(id_1) not in indexList:
                    indexList.append(int(id_1))
                    LabelList.append(nodesLabelDict[id_1])
                if [int(id_1), int(id_2)] not in edgeList:
                    edgeList.append([int(id_1), int(id_2)])
        for index in indexList:
            featuresList.append(nodesFeatureDict[str(index)])
        featuresList=np.array(featuresList,dtype=np.float32)
        LabelList=np.array(LabelList)
        edgeList=np.array(edgeList,dtype=np.int32)
        real_adj, features_batch, labels_batch, adj_batch_list=load_data_batch(path="./data/twitter15/", dataset="twitter15",dilationRateSequence=args.dilation_list,batchNodes=indexs,featuresList=featuresList,LabelList=LabelList,edgeList=edgeList,trainIndexList=indexList)
        return real_adj, features_batch, labels_batch, adj_batch_list
testBatches=[]
print("Processing Test Batch")
for i in tqdm(range(len(testIndexFileLines)//args.batch_size)):
    real_adj, features_batch, labels_batch, adj_batch_list=generateBatch(testIndexFileLines,edgeFileLines,nodesLabelDict,nodesFeatureDict,False,i)
    testBatches.append([real_adj, features_batch, labels_batch, adj_batch_list])
print("TestBatch process completed")

# Load all data
# print("load all data.")
# realadj, features, labels, idx_train, idx_val, idx_test, adjList = load_data(path="./data/twitter15/", dataset="twitter15",dilationRateSequence=args.dilation_list)



# Model and optimizer
if args.multilayer:
    # model = SpGAT(nfeat=features.shape[1],
    #             nhid=args.hidden,
    #             nclass=int(labels.max()) + 1,
    #             dropout=args.dropout,
    #             nheads=args.nb_heads,
    #             alpha=args.alpha)
    model = GAT_MultiLayer(nfeat=testBatches[0][1].shape[1],
                nhid=args.hidden_list[0],
                nhid_list=args.hidden_list,
                nclass=int(len(labelType)),
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha,
                residual_status=args.residual,
                freeze_status=args.freeze)
# else:
#     model = GAT(nfeat=features.shape[1],
#                 nhid=args.hidden_list[0],
#                 nclass=int(len(labelType)),
#                 dropout=args.dropout,
#                 nheads=args.nb_heads,
#                 alpha=args.alpha)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

# if args.cuda:
model.cuda()
#     features = features.cuda()
#     # labels = labels.cuda()
#     # realadj=realadj.cuda()
#     # idx_train = idx_train.cuda()
#     # idx_val = idx_val.cuda()
#     # idx_test = idx_test.cuda()
idx_batch = idx_batch.cuda()
#
# features, labels = Variable(features),  Variable(labels)
# adjs=[]
# if args.cuda:
#    for i in range(len(adjList)):
#        adjs.append(Variable(adjList[i].cuda()))

# def train_structure(epoch):
#     t = time.time()
#     model.train()
#     optimizer.zero_grad()
#     output,adj_pred = model(features, adjs)
#     pos_weight = float(realadj.shape[0] * realadj.shape[0] - realadj.sum()) / realadj.sum()
#     weight_mask = realadj.to_dense().view(-1) == 1
#     weight_tensor = torch.ones(weight_mask.size(0))
#     weight_tensor[weight_mask] = pos_weight
#     weight_tensor=weight_tensor.cuda()
#     loss_train = F.binary_cross_entropy(adj_pred.view(-1), realadj.to_dense().view(-1), weight=weight_tensor)
#     loss_train.backward()
#     optimizer.step()
#
#     if not args.fastmode:
#         # Evaluate validation set performance separately,
#         # deactivates dropout during validation run.
#         model.eval()
#         output,adj_pred = model(features, adjs)
#
#     loss_val = F.binary_cross_entropy(adj_pred.view(-1), realadj.to_dense().view(-1), weight=weight_tensor)
#     print('Structure training. Epoch: {:04d}'.format(epoch+1),
#           'loss_train: {:.4f}'.format(loss_train.data.item()),
#           'loss_val: {:.4f}'.format(loss_val.data.item()),
#           'time: {:.4f}s'.format(time.time() - t))
#
#     return loss_val.data.item()
#
# def train_nodes(epoch):
#     t = time.time()
#     model.train()
#     optimizer.zero_grad()
#     output,adj_pred = model(features, adjs)
#     loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#     acc_train = accuracy(output[idx_train], labels[idx_train])
#     loss_train.backward()
#     optimizer.step()
#
#     if not args.fastmode:
#         # Evaluate validation set performance separately,
#         # deactivates dropout during validation run.
#         model.eval()
#         output,adj_pred = model(features, adjs)
#
#     loss_val = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_val = accuracy(output[idx_test], labels[idx_test])
#     print('Nodes training. Epoch: {:04d}'.format(epoch+1),
#           'loss_train: {:.4f}'.format(loss_train.data.item()),
#           'acc_train: {:.4f}'.format(acc_train.data.item()),
#           'loss_val: {:.4f}'.format(loss_val.data.item()),
#           'acc_val: {:.4f}'.format(acc_val.data.item()),
#           'time: {:.4f}s'.format(time.time() - t))
#
#     return acc_val.data.item()

def train_batch_structure(epoch):
    real_adj,features_batch,labels_batch,adjs_batch=generateBatch(trainIndexFileLines,edgeFileLines,nodesLabelDict,nodesFeatureDict,True,None)
    features_batch=features_batch.cuda()
    labels_batch=labels_batch.cuda()
    real_adj = real_adj.cuda()
    features_batch, labels_batch = Variable(features_batch),  Variable(labels_batch)
    t = time.time()
    model.train()
    optimizer.zero_grad()
    adjs=[]
    for i in range(len(adjs_batch)):
        adjs.append(Variable(adjs_batch[i].cuda()))
    output,adj_pred = model(features_batch, adjs)
    pos_weight = float(real_adj.shape[0] * real_adj.shape[0] - real_adj.sum()) / real_adj.sum()
    weight_mask = real_adj.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    weight_tensor = weight_tensor.cuda()
    loss_train = F.binary_cross_entropy(adj_pred.view(-1), real_adj.to_dense().view(-1), weight=weight_tensor)
    loss_train.backward()
    optimizer.step()

    loss_val_list=[]
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        for testBatch in testBatches:
            real_adj_batch = testBatch[0].cuda()
            features_batch = testBatch[1].cuda()
            adjs_batch = testBatch[3]
            adjs_batches = []
            for i in range(len(adjs_batch)):
                adjs_batches.append(Variable(adjs_batch[i].cuda()))
            output,adj_pred = model(features_batch, adjs_batches)
            pos_weight = float(real_adj_batch.shape[0] * real_adj_batch.shape[0] - real_adj_batch.sum()) / real_adj_batch.sum()
            weight_mask = real_adj_batch.to_dense().view(-1) == 1
            weight_tensor = torch.ones(weight_mask.size(0))
            weight_tensor[weight_mask] = pos_weight
            weight_tensor = weight_tensor.cuda()
            loss_val_list.append(F.binary_cross_entropy(adj_pred.view(-1), real_adj_batch.to_dense().view(-1), weight=weight_tensor).data.item())
        loss_val=sum(loss_val_list)/len(loss_val_list)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val

def train_batch_nodes(epoch):
    _,features_batch,labels_batch,adjs_batch=generateBatch(trainIndexFileLines,edgeFileLines,nodesLabelDict,nodesFeatureDict,True,None)
    t = time.time()
    features_batch=features_batch.cuda()
    labels_batch=labels_batch.cuda()
    features_batch, labels_batch = Variable(features_batch), Variable(labels_batch)
    adjs_batches = []
    for i in range(len(adjs_batch)):
        adjs_batches.append(Variable(adjs_batch[i].cuda()))
    model.train()
    optimizer.zero_grad()
    output,adj_pred = model(features_batch, adjs_batches)
    loss_train= F.nll_loss(output[idx_batch], labels_batch[idx_batch])
    acc_train = accuracy(output[idx_batch], labels_batch[idx_batch])
    loss_train.backward()
    optimizer.step()

    test_output_total=[]
    test_label_total = []
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        for testBatch in testBatches:
            features_batch=testBatch[1].cuda()
            labels_batch = testBatch[2].cuda()
            adjs_batch = testBatch[3]
            adjs= []
            for i in range(len(adjs_batch)):
                adjs.append(Variable(adjs_batch[i].cuda()))
            test_output,test_adj_pred = model(features_batch, adjs)
            for output in test_output[idx_batch].tolist():
                test_output_total.append(output)
            for label in labels_batch[idx_batch].tolist():
                test_label_total.append(int(label))
        test_output_total=torch.Tensor(test_output_total).cuda()
        test_label_total = torch.LongTensor(test_label_total).cuda()
    testBatchSize = torch.LongTensor(range(len(testIndexFileLines))).cuda()
    loss_val = F.nll_loss(test_output_total[testBatchSize], test_label_total[testBatchSize])
    acc_val = accuracy(test_output_total[testBatchSize], test_label_total[testBatchSize])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return acc_val.data.item()

def train_nodes_structure(epoch):
    real_adj, features_batch, labels_batch, adjs_batch = generateBatch(trainIndexFileLines, edgeFileLines, nodesLabelDict,nodesFeatureDict, True, None)
    t = time.time()
    features_batch = features_batch.cuda()
    labels_batch = labels_batch.cuda()
    real_adj = real_adj.cuda()
    features_batch, labels_batch = Variable(features_batch), Variable(labels_batch)
    adjs_batches = []
    for i in range(len(adjs_batch)):
        adjs_batches.append(Variable(adjs_batch[i].cuda()))
    model.train()
    optimizer.zero_grad()
    output, adj_pred = model(features_batch, adjs_batches)
    pos_weight = float(real_adj.shape[0] * real_adj.shape[0] - real_adj.sum()) / real_adj.sum()
    weight_mask = real_adj.to_dense().view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    weight_tensor = weight_tensor.cuda()
    loss_train_nodes = 0.5*F.nll_loss(output[idx_batch], labels_batch[idx_batch])
    loss_train_structure = 0.5*F.binary_cross_entropy(adj_pred.view(-1), real_adj.to_dense().view(-1), weight=weight_tensor)
    loss_train=loss_train_structure+loss_train_nodes
    acc_train = accuracy(output[idx_batch], labels_batch[idx_batch])
    loss_train.backward()
    optimizer.step()

    test_output_total = []
    test_label_total = []
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        for testBatch in testBatches:
            features_batch = testBatch[1].cuda()
            labels_batch = testBatch[2].cuda()
            adjs_batch = testBatch[3]
            adjs = []
            for i in range(len(adjs_batch)):
                adjs.append(Variable(adjs_batch[i].cuda()))
            test_output, test_adj_pred = model(features_batch, adjs)
            for output in test_output[idx_batch].tolist():
                test_output_total.append(output)
            for label in labels_batch[idx_batch].tolist():
                test_label_total.append(int(label))
        test_output_total = torch.Tensor(test_output_total).cuda()
        test_label_total = torch.LongTensor(test_label_total).cuda()
    testBatchSize = torch.LongTensor(range(len(testIndexFileLines))).cuda()
    loss_val = F.nll_loss(test_output_total[testBatchSize], test_label_total[testBatchSize])
    acc_val = accuracy(test_output_total[testBatchSize], test_label_total[testBatchSize])
    print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()),
              'time: {:.4f}s'.format(time.time() - t))

    return acc_val.data.item()


def compute_test_nodes():
    test_output_total = []
    test_label_total = []
    model.eval()
    # testFile=open("./data/"+datasetName+"/test_new.txt", "a+")
    # testFile1 = open("./data/"+datasetName+"/test_new_1.txt", "a+")
    for testBatchCounter in range(len(testBatches)):
        testBatch=testBatches[testBatchCounter]
        features_batch = testBatch[1].cuda()
        labels_batch = testBatch[2].cuda()
        adjs_batch = testBatch[3]
        features_batch, labels_batch = Variable(features_batch), Variable(labels_batch)
        adjs_batches = []
        for i in range(len(adjs_batch)):
            adjs_batches.append(Variable(adjs_batch[i].cuda()))
        outputs,_= model(features_batch, adjs_batches)
        outputsList=outputs[idx_batch].tolist()
        labelsList=labels_batch[idx_batch].tolist()
        # for outputCounter in range(len(outputsList)):
        #     output=np.array(outputsList[outputCounter])
        #     maxIndex=np.argmax(output)
        #     if str(maxIndex)==str(labelsList[outputCounter]):
        #         no = testIndexFileLines[testBatchCounter*args.batch_size+outputCounter].split("\t")[0]
        #         label = testIndexFileLines[testBatchCounter * args.batch_size + outputCounter].split("\t")[1]
        #         testFile.write(no+"\t"+label)
        #     else:
        #         no = testIndexFileLines[testBatchCounter * args.batch_size + outputCounter].split("\t")[0]
        #         label = testIndexFileLines[testBatchCounter * args.batch_size + outputCounter].split("\t")[1]
        #         testFile1.write(no + "\t" + label)
        for output in outputsList:
            test_output_total.append(output)
        for label in labelsList:
            test_label_total.append(int(label))
    test_output_total = torch.Tensor(test_output_total).cuda()
    test_label_total = torch.LongTensor(test_label_total).cuda()
    testBatchSize = torch.LongTensor(range(len(test_output_total))).cuda()
    loss_test = F.nll_loss(test_output_total[testBatchSize], test_label_total[testBatchSize])
    acc_test = accuracy(test_output_total[testBatchSize], test_label_total[testBatchSize])
    print("Nodes classification test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))

# def compute_test_structure():
#     model.eval()
#     output,adj_pred = model(features, adjs)
#     pos_weight = float(realadj.shape[0] * realadj.shape[0] - realadj.sum()) / realadj.sum()
#     weight_mask = realadj.to_dense().view(-1) == 1
#     weight_tensor = torch.ones(weight_mask.size(0))
#     weight_tensor[weight_mask] = pos_weight
#     weight_tensor = weight_tensor.cuda()
#     loss_test = F.binary_cross_entropy(adj_pred.view(-1), realadj.to_dense().view(-1), weight=weight_tensor)
#     print("Structure rebuild test set results:",
#           "loss= {:.4f}".format(loss_test.data.item()))


# Train model
if args.batch==False:
    print("debug zone")
    # t_total = time.time()
    # values = []
    # bad_counter = 0
    # best=None
    # best_epoch = 0
    # for name, parameter in model.named_parameters():
    #     print(name)
    #     print(parameter)
    # if args.freeze == 1:
    #     # Restore best model
    #     best = -1*(args.epochs + 1)
    #     for epoch in range(args.epochs):
    #         acc_val = train_nodes(epoch)
    #         values.append(acc_val)
    #         if values[-1] > best:
    #             print("current best acc_val! saving......")
    #             torch.save(model.state_dict(), '{}.pkl'.format(epoch+1))
    #             best = values[-1]
    #             best_epoch = epoch+1
    #             bad_counter = 0
    #         else:
    #             bad_counter += 1
    #         # if bad_counter == args.patience:
    #         #     break
    #         files = glob.glob('*.pkl')
    #         for file in files:
    #             if int(file.split(".pkl")[0]) < best_epoch:
    #                 os.remove(file)
            # Testing
            # compute_test_nodes()
    # elif args.freeze == 2:
    #     print('Loading strcture training results')
    #     model.load_state_dict(torch.load('./models/best_nodes.pkl'))
    #     best = args.epochs + 1
    #     for epoch in range(args.epochs):
    #         loss_val = train_structure(epoch)
    #         values.append(loss_val)
    #         if values[-1] < best:
    #             print("current best loss_val! saving......")
    #             torch.save(model.state_dict(), '{}.pkl'.format(epoch+1))
    #             best = values[-1]
    #             best_epoch = epoch+1
    #             bad_counter = 0
    #         else:
    #             bad_counter += 1
    #         # if bad_counter == args.patience:
    #         #     break
    #         files = glob.glob('*.pkl')
    #         for file in files:
    #             if int(file.split(".pkl")[0]) < best_epoch:
    #                 os.remove(file)
    #         # Testing
    #         # compute_test_structure()
    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
else:
    # Train model batch
    t_total = time.time()
    values = []
    bad_counter = 0
    best = None
    best_epoch = 0
    if args.freeze == 1:
        # Restore best model
        model.load_state_dict(torch.load("./models/best structure.pkl"))
        for name, parameter in model.named_parameters():
            print(name)
            print(parameter)
        best = -1 * (args.epochs + 1)
        for epoch in range(args.epochs):
            acc_val = train_batch_nodes(epoch)
            values.append(acc_val)
            if values[-1] > best:
                print("current best acc_val! saving......")
                torch.save(model.state_dict(), '{}.pkl'.format(epoch + 1))
                best = values[-1]
                best_epoch = epoch + 1
                bad_counter = 0
            else:
                bad_counter += 1
            files = glob.glob('*.pkl')
            for file in files:
                if int(file.split(".pkl")[0]) < best_epoch:
                    os.remove(file)
            # Testing
            # compute_test_nodes()
    elif args.freeze == 2:
        # print('Loading structure training results')
        # model.load_state_dict(torch.load("./models/"+datasetName+"/3layers_dilation_1_12_13/best_nodes.pkl"))
        for name, parameter in model.named_parameters():
            print(name)
            print(parameter)
        best = -1 * (args.epochs + 1)
        best = args.epochs + 1
        for epoch in range(args.epochs):
            loss_val = train_batch_structure(epoch)
            values.append(loss_val)
            if values[-1] < best:
                print("current best loss_val! saving......")
                torch.save(model.state_dict(), '{}.pkl'.format(epoch + 1))
                best = values[-1]
                best_epoch = epoch + 1
                bad_counter = 0
            else:
                bad_counter += 1
            # if bad_counter == args.patience:
            #     break
            files = glob.glob('*.pkl')
            for file in files:
                if int(file.split(".pkl")[0]) < best_epoch:
                    os.remove(file)
            # Testing
            # compute_test_structure()
    if args.freeze == 3:
        # Restore best model
        # model.load_state_dict(torch.load("./models/"+datasetName+"/3layers_dilation_1_12_13/without autoencoder.pkl"))
        for name, parameter in model.named_parameters():
            print(name)
            print(parameter)
        best = -1 * (args.epochs + 1)
        for epoch in range(args.epochs):
            acc_val = train_nodes_structure(epoch)
            values.append(acc_val)
            if values[-1] > best:
                print("current best acc_val! saving......")
                torch.save(model.state_dict(), '{}.pkl'.format(epoch + 1))
                best = values[-1]
                best_epoch = epoch + 1
                bad_counter = 0
            else:
                bad_counter += 1
            files = glob.glob('*.pkl')
            for file in files:
                if int(file.split(".pkl")[0]) < best_epoch:
                    os.remove(file)
            # Testing
            # compute_test_nodes()
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# model.load_state_dict(torch.load("./models/"+datasetName+"/3layers_dilation_1_12_13/best_nodes.pkl"))
# compute_test_nodes()




