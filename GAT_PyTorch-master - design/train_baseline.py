import torch
import torch.nn.functional as F
from GCN import GCN
from GATConv import GATNet
from SAGEConv import SAGEConvNet
from utils_baseline import load_data

dataset="pheme"
features, edges, labels,data = load_data(path="./data/"+dataset+"/", dataset=dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model=SAGEConvNet(2000, 100, 4)
model = GATNet(num_node_features=2000, hidden=100, num_classes=4, heads=8)
# model = GCN(2000, 100, 4)
trainList=[]
trainFileLines=open("./data/"+dataset+"/train.txt","r").readlines()
for line in trainFileLines:
    lineNum=int(line.split("\t")[0])-10000
    trainList.append(lineNum)
testList=[]
testFileLines=open("./data/"+dataset+"/test.txt","r").readlines()
for line in testFileLines:
    lineNum=int(line.split("\t")[0])-10000
    testList.append(lineNum)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)               # 梯度优化算法
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    # out = model(features, edges)
    out = model(data)
    train_output=[]
    label_real = []
    for outCounter in range(len(out.tolist())):
        if outCounter in trainList:
            ouput=out[outCounter]
            train_output.append(ouput)
            label=labels.tolist()[outCounter]
            label_real.append(label)
    train_output=tuple(train_output)
    train_output=torch.stack(train_output,0)
    train_output = torch.Tensor(train_output).cuda()
    label_real = torch.LongTensor(label_real).cuda()
    loss = F.nll_loss(train_output, label_real)        # 损失函数
    loss.backward()
    optimizer.step()
    model.eval()
    testOutputs = []
    realLabels = []
    # _, pred = model(features, edges).max(dim=1)
    _, pred = model(data).max(dim=1)
    for predCounter in range(len(pred.tolist())):
        if predCounter in testList:
            testOutputs.append(int(pred[predCounter]))
            label = labels.tolist()[predCounter]
            realLabels.append(label)
    corretNum=0
    for testOutputCounter in range(len(testOutputs)):
        if testOutputs[testOutputCounter]==realLabels[testOutputCounter]:
            corretNum+=1
    acc = int(corretNum) / int(len(realLabels))  # 计算正确率
    print(f"epoch:{epoch + 1}, loss:{loss.item()}, acc:{acc}")
