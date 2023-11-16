datasetName="pheme"
twitterNodeFile=open("./data/"+datasetName+"/"+datasetName+".content","r")
newTwitterNodeFile=open("./data/"+datasetName+"/"+datasetName+"_new.content","a+")
twitterEdgeFile=open("./data/"+datasetName+"/"+datasetName+".cites","r")
newtwitterEdgeFile=open("./data/"+datasetName+"/"+datasetName+"_new.cites","a+")
twitterLabelFile=open("./data/"+datasetName+"/label.txt","r")
newtwitterLabelFile=open("./data/"+datasetName+"/label_new.txt","a+")
twitterNodeFileLines=twitterNodeFile.readlines()
twitterEdgeFileLines=twitterEdgeFile.readlines()
twitterLabelFileLines=twitterLabelFile.readlines()
newDict={}
counter=10000
newNode=[]
nodeNumber=0

for line in twitterNodeFileLines:
    id=line.split("\t")[0]
    rest="\t".join(line.split("\t")[1:]).replace("\n","")
    if id not in newDict.keys():
        newDict[id]=str(counter)
        counter+=1
        nodeNumber+=1
        newTwitterNodeFile.write(newDict[id]+"\t"+rest+"\n")
newEdge=[]
for line in twitterEdgeFileLines:
    id_1=line.split("\t")[0]
    id_2=line.split("\t")[1].replace("\n","")
    if id_1 not in newNode:
        newNode.append(id_1)
    if id_2 not in newNode:
        newNode.append(id_2)
    if id_1!=id_2 and newDict[id_1]+"\t"+newDict[id_2] not in newEdge:
        newtwitterEdgeFile.write(newDict[id_1]+"\t"+newDict[id_2]+"\n")
        newEdge.append(newDict[id_1]+"\t"+newDict[id_2])
for line in twitterLabelFileLines:
    label=line.split("\t")[1].replace("\n","")
    no=line.split("\t")[0]
    if no in newDict.keys():
        newtwitterLabelFile.write(label+":"+newDict[no]+"\n")
