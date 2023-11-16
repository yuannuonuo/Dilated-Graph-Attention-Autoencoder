import os

dirList=["charliehebdo","ferguson","germanwings-crash","ottawashooting","sydneysiege","source-tweet","reactions","non-rumours","rumours"]
labelFile=open("./data/pheme/label.txt","a+")
for root,dirs,files in os.walk("./data/pheme"):
    for dir in dirs:
        if dir not in dirList:
            fullPath=os.path.join(root,dir)
            if "non" in fullPath:
                labelFile.write(dir+"\t"+"non-rumours\n")
            else:
                labelFile.write(dir+"\t"+"rumours\n")