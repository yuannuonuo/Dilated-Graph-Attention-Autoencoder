datasetName="pheme"
testFile_Wrong=open("./data/"+datasetName+"/test_new_1.txt","r")
testFile_Right=open("./data/"+datasetName+"/test_new.txt","r")
testFile_RightLines=testFile_Right.readlines()
testFile_WrongLines=testFile_Wrong.readlines()
labelTypeDict={}
for line in testFile_RightLines:
    label=str(line.split("\t")[1].replace("\n",""))
    if label not in labelTypeDict.keys():
        labelTypeDict[label] = 1
    else:
        labelTypeDict[label] += 1
modifiedList=[]
testFile_Right=open("./data/"+datasetName+"/test_new_3.txt","a+")
for key in labelTypeDict.keys():
    currentRest=800/len(list(labelTypeDict.keys()))-labelTypeDict[key]
    currentList=[]
    for line in testFile_WrongLines:
        label=str(line.split("\t")[1].replace("\n",""))
        if label==key and len(currentList)<currentRest:
            currentList.append(line)
    modifiedList+=currentList
    for line in currentList:
        testFile_Right.write(line)
testFile_Wrong_2=open("./data/"+datasetName+"/test_new_2.txt","a+")
for line in testFile_WrongLines:
    if line not in modifiedList:
        testFile_Wrong_2.write(line)


