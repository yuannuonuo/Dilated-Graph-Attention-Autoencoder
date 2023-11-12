import random

DatasetName="pheme"
DataPath="./data/"+DatasetName+"/label.txt"
Train_ratio=1000
Test_ratio=2000
Val_ratio=726

labelFile=open(DataPath,"r")
labelFileLines=labelFile.readlines()
labelDict={}
labelDict_Reverse={}
for line in labelFileLines:
    label=line.split(":")[0]
    no=line.split(":")[1].replace("\n","")
    if label not in labelDict.keys():
        labelDict[label]=[no]
    else:
        labelDict[label].append(no)
    labelDict_Reverse[no]=label


splitDict={}
for key in labelDict.keys():
    splitDict["train"]=[]
    splitDict["val"] = []
    splitDict["test"] = []
labelNum=len(list(labelDict.keys()))
TrainList=[Train_ratio-((Train_ratio//labelNum)*(labelNum-1)) if counter==labelNum-1 else Train_ratio//labelNum for counter in range(labelNum)]
TestList=[Test_ratio-((Test_ratio//labelNum)*(labelNum-1)) if counter==labelNum-1 else Test_ratio//labelNum for counter in range(labelNum)]
ValList=[len(labelDict[list(labelDict.keys())[counter]])-TrainList[counter]-TestList[counter] for counter in range(labelNum)]
for keyCounter in range(labelNum):
    key=list(labelDict.keys())[keyCounter]
    noList=labelDict[key]
    noListRange=range(len(noList))
    trainIndexList=random.sample(noListRange, TrainList[keyCounter])
    currentTrainList=[]
    for index in trainIndexList:
        currentTrainList.append(noList[index])
        splitDict["train"].append(noList[index])
    for currentIndex in currentTrainList:
        noList.remove(currentIndex)
    noListRange = range(len(noList))
    testIndexList = random.sample(noListRange, TestList[keyCounter])
    currentTestList = []
    for index in testIndexList:
        currentTestList.append(noList[index])
        splitDict["test"].append(noList[index])
    for currentIndex in currentTestList:
        noList.remove(currentIndex)
    for item in noList:
        splitDict["val"].append(item)
    print(key,len(splitDict["train"]),len(splitDict["test"]),len(splitDict["val"]))
for key in splitDict.keys():
    print(key,len(splitDict[key]))
    file=open("./data/"+DatasetName+"/"+key+"_1.txt","a+")
    currentList=splitDict[key]
    random.shuffle(currentList)
    for no in currentList:
        currentLabel=labelDict_Reverse[no]
        file.write(no+"\t"+currentLabel+"\n")




