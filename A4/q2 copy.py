import numpy as np
import matplotlib.pyplot as plt


data=np.load('mnist.npz')
trainX= data['x_train']
trainy=data['y_train']
testX=data['x_test']
testy=data['y_test']
trainX=np.reshape(trainX,(trainX.shape[0],trainX.shape[1]*trainX.shape[2])) #Vectorizing the train data
testX=np.reshape(testX,(testX.shape[0],testX.shape[1]*testX.shape[2]))
dataSet=[]
valDataSet=[]
valMappedY=[]
mappedY=[]
for i in range(trainy.shape[0]):
    if trainy[i]==0 or trainy[i]==1:
        if(len(valDataSet)!=2000):
            if(valMappedY.count(trainy[i])<1000):
                valDataSet.append(trainX[i])
                valMappedY.append(trainy[i])
            else:
                dataSet.append(trainX[i])
                mappedY.append(trainy[i])
        else:
            dataSet.append(trainX[i])
            mappedY.append(trainy[i])
temptestX=[]
temptesty=[]
for i in range(testy.shape[0]):
    if testy[i]==0 or testy[i]==1:
        temptestX.append(testX[i])
        temptesty.append(testy[i])
testy=np.array(temptesty)
testX=np.array(temptestX)
dataSet=np.array(dataSet)
mappedY=np.array(mappedY)
valDataSet=np.array(valDataSet)
valMappedY=np.array(valMappedY)

mappedY=list(mappedY)
for i in range(len(mappedY)):
    if mappedY[i]==0:
        mappedY[i]=-1
mappedY=np.array(mappedY)

testy=list(testy)
for i in range(len(testy)):
    if testy[i]==0:
        testy[i]=-1
testy=np.array(testy)

valMappedY=list(valMappedY)
for i in range(len(valMappedY)):
    if valMappedY[i]==0:
        valMappedY[i]=-1
valMappedY=np.array(valMappedY)

# print(dataSet.shape)
# print(mappedY.shape)
# print(valDataSet.shape)
# print(valMappedY.shape)

dataSet=np.transpose(dataSet)
testX=np.transpose(testX)
valDataSet=np.transpose(valDataSet)
mean=np.mean(dataSet,axis=1,keepdims=True)
dataSet=dataSet-mean
testX=testX-mean
valDataSet=valDataSet-mean
testX=np.transpose(testX)
valDataSet=np.transpose(valDataSet)
print("Mean done")
np.set_printoptions(suppress = True)
cov=np.dot(dataSet,dataSet.T)/(dataSet.shape[0]-1)
print("Cov done")
eigenValues,eigenVectors=np.linalg.eigh(cov)
idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
U=eigenVectors
p=5
Up=U[:,:p]
Yp=np.dot(Up.T,dataSet)
Yp=Yp.T
Yp=Yp.T
# print(Yp.shape)





def makeStump(Yp,mappedY,valDataSet,valMappedY,Up,listTrees):
    TreeList={"SSR":float("inf")}
    for i in range(Yp.shape[0]):
        uniqueVals=list(set(Yp[i]))
        midPoints=[]
        for j in range(Yp.shape[1]-1):
            midPoints.append((Yp[i][j]+Yp[i][j+1])/2)
        uniqueVals=midPoints
        indices=np.random.choice(len(uniqueVals),size=2,replace=False)
        uniqueVals=list(set(np.array(uniqueVals)[indices]))
        uniqueVals.sort()
        dimensionTreeList={"SSR":float("inf")}
        for j in range(len(uniqueVals)-1):
            midPoint=(uniqueVals[j]+uniqueVals[j+1])/2
            countLeft=0
            leftLabelSummation=0
            countRight=0
            rightLabelSummation=0
            for k in range(Yp.shape[1]):
                if(Yp[i][k]<midPoint):
                    countLeft+=1
                    leftLabelSummation+=mappedY[k]
                else:
                    countRight+=1
                    rightLabelSummation+=mappedY[k]
            countLeftClass=leftLabelSummation/countLeft
            countRightClass=rightLabelSummation/countRight
            SSR=0
            for k in range(Yp.shape[1]):
                if(Yp[i][k]<midPoint):
                    SSR+=(1/2)*(mappedY[k]-countLeftClass)**2
                else:
                    SSR+=(1/2)*(mappedY[k]-countRightClass)**2

            if(SSR<dimensionTreeList["SSR"]):
                dimensionTreeList["SSR"]=SSR
                dimensionTreeList["midPoint"]=midPoint
                dimensionTreeList["countLeftClass"]=countLeftClass
                dimensionTreeList["countRightClass"]=countRightClass
        
        if(dimensionTreeList["SSR"]<TreeList["SSR"]):
            TreeList["SSR"]=dimensionTreeList["SSR"]
            TreeList["midPoint"]=dimensionTreeList["midPoint"]
            TreeList["countLeftClass"]=dimensionTreeList["countLeftClass"]
            TreeList["countRightClass"]=dimensionTreeList["countRightClass"]
            TreeList["dimension"]=i
    # print(TreeList)
    for k in range(Yp.shape[1]):
        if(Yp[TreeList["dimension"]][k]<TreeList["midPoint"]):
            mappedY[k]=(mappedY[k]-(0.01*TreeList["countLeftClass"]))
        else:
            mappedY[k]=(mappedY[k]-(0.01*TreeList["countRightClass"]))

    MSE=0
    for i in range(valDataSet.shape[0]):
        ip=np.dot(Up.T,valDataSet[i])
        prediction=0
        for j in listTrees:
            if ip[j["dimension"]]<j["midPoint"]:
                prediction+=((0.01)*j["countLeftClass"])
            else:
                prediction+=((0.01)*j["countRightClass"])
        if ip[TreeList["dimension"]]<TreeList["midPoint"]:
            prediction+=((0.01)*TreeList["countLeftClass"])
        else:
            prediction+=((0.01)*TreeList["countRightClass"])
        MSE+=(valMappedY[i]-prediction)**2
    MSE/=valDataSet.shape[0]
    returnTree={"dimension":TreeList["dimension"],"midPoint":TreeList["midPoint"],"countLeftClass":TreeList["countLeftClass"],"countRightClass":TreeList["countRightClass"],"MSE":MSE}
    return returnTree,mappedY

listTrees=[]
mappedYInput=list(mappedY.copy())
file=open("Tree2Regression","w+")
for i in range(300):
    returnTree,mappedYInput=makeStump(Yp,mappedYInput,valDataSet,valMappedY,Up,listTrees)
    print(returnTree)
    listTrees.append(returnTree)
    file.write(str(returnTree))
    file.write(str("\n"))
    file.flush()



# Iteration 1000

MSE=[]
listTrees=[]
with open("Tree1000Regression","r") as file:
    totalFile=file.readlines()
    for tree in totalFile:
        newTree=(eval(tree.strip()))
        newMSE=newTree['MSE'] 
        listTrees.append(newTree)
        MSE.append(newMSE)
fig = plt.figure(figsize=(20, 2))
ax = fig.add_subplot(111)
ax.plot(MSE)
plt.show()

MSE=float("inf")
index=-1
for j in range(len(listTrees)):
    if listTrees[j]["MSE"]<=MSE:
        index=j
MSE=0
for i in range(testX.shape[0]):
    ip=np.dot(Up.T,testX[i])
    prediction=0
    for j in range(index):
        if ip[listTrees[j]["dimension"]]<listTrees[j]["midPoint"]:
            prediction+=((0.01)*listTrees[j]["countLeftClass"])
        else:
            prediction+=((0.01)*listTrees[j]["countRightClass"])
    MSE+=(testy[i]-prediction)**2
MSE/=testX.shape[0]
print("Total MSE is:",MSE)



# Iteration 100

# MSE=[]
# listTrees=[]
# with open("Tree100Regression","r") as file:
#     totalFile=file.readlines()
#     for tree in totalFile:
#         newTree=(eval(tree.strip()))
#         newMSE=newTree['MSE'] 
#         listTrees.append(newTree)
#         MSE.append(newMSE)
# fig = plt.figure(figsize=(20, 2))
# ax = fig.add_subplot(111)
# ax.plot(MSE)
# plt.show()

# MSE=float("inf")
# index=-1
# for j in range(len(listTrees)):
#     if listTrees[j]["MSE"]<=MSE:
#         index=j
# MSE=0
# for i in range(testX.shape[0]):
#     ip=np.dot(Up.T,testX[i])
#     prediction=0
#     for j in range(index):
#         if ip[listTrees[j]["dimension"]]<listTrees[j]["midPoint"]:
#             prediction+=((0.01)*listTrees[j]["countLeftClass"])
#         else:
#             prediction+=((0.01)*listTrees[j]["countRightClass"])
#     MSE+=(testy[i]-prediction)**2
# MSE/=testX.shape[0]
# print("Total MSE is:",MSE)



# Iteration 10

# MSE=[]
# listTrees=[]
# with open("Tree10Regression","r") as file:
#     totalFile=file.readlines()
#     for tree in totalFile:
#         newTree=(eval(tree.strip()))
#         newMSE=newTree['MSE'] 
#         listTrees.append(newTree)
#         MSE.append(newMSE)
# fig = plt.figure(figsize=(20, 2))
# ax = fig.add_subplot(111)
# ax.plot(MSE)
# plt.show()

# MSE=float("inf")
# index=-1
# for j in range(len(listTrees)):
#     if listTrees[j]["MSE"]<=MSE:
#         index=j
# MSE=0
# for i in range(testX.shape[0]):
#     ip=np.dot(Up.T,testX[i])
#     prediction=0
#     for j in range(index):
#         if ip[listTrees[j]["dimension"]]<listTrees[j]["midPoint"]:
#             prediction+=((0.01)*listTrees[j]["countLeftClass"])
#         else:
#             prediction+=((0.01)*listTrees[j]["countRightClass"])
#     MSE+=(testy[i]-prediction)**2
# MSE/=testX.shape[0]
# print("Total MSE is:",MSE)




# Iteration 2

# MSE=[]
# listTrees=[]
# with open("Tree2Regression","r") as file:
#     totalFile=file.readlines()
#     for tree in totalFile:
#         newTree=(eval(tree.strip()))
#         newMSE=newTree['MSE'] 
#         listTrees.append(newTree)
#         MSE.append(newMSE)
# fig = plt.figure(figsize=(20, 2))
# ax = fig.add_subplot(111)
# ax.plot(MSE)
# plt.show()

# MSE=float("inf")
# index=-1
# for j in range(len(listTrees)):
#     if listTrees[j]["MSE"]<=MSE:
#         index=j
# MSE=0
# for i in range(testX.shape[0]):
#     ip=np.dot(Up.T,testX[i])
#     prediction=0
#     for j in range(index):
#         if ip[listTrees[j]["dimension"]]<listTrees[j]["midPoint"]:
#             prediction+=((0.01)*listTrees[j]["countLeftClass"])
#         else:
#             prediction+=((0.01)*listTrees[j]["countRightClass"])
#     MSE+=(testy[i]-prediction)**2
# MSE/=testX.shape[0]
# print("Total MSE is:",MSE)