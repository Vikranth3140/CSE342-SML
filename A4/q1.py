import numpy as np

import matplotlib.pyplot as plt

# Load the MNIST dataset
data=np.load('mnist.npz')
trainX= data['x_train']
trainy=data['y_train']
testX=data['x_test']
testy=data['y_test']

# Vectorize the train and test data
trainX=np.reshape(trainX,(trainX.shape[0],trainX.shape[1]*trainX.shape[2]))
testX=np.reshape(testX,(testX.shape[0],testX.shape[1]*testX.shape[2]))

# Prepare the training and validation datasets
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

# Convert class labels to -1 and 1
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

# Preprocess the datasets
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

def makeStump(Yp,mappedY,weights,valDataSet,valMappedY,Up,listTrees):
    TreeList={"loss":float("inf")}
    for i in range(Yp.shape[0]):
        uniqueVals=list(set(Yp[i]))
        midPoints=[]
        for j in range(Yp.shape[1]-1):
            midPoints.append((Yp[i][j]+Yp[i][j+1])/2)
        uniqueVals=midPoints
        indices=np.random.choice(len(uniqueVals),size=2,replace=False)
        uniqueVals=list(set(np.array(uniqueVals)[indices]))
        uniqueVals.sort()
        dimensionTreeList={"loss":float("inf")}
        for j in range(len(uniqueVals)-1):
            midPoint=(uniqueVals[j]+uniqueVals[j+1])/2
            countLeft={-1:0,1:0}
            countRight={-1:0,1:0}
            for k in range(Yp.shape[1]):
                if(Yp[i][k]<midPoint):
                    countLeft[mappedY[k]]+=1
                else:
                    countRight[mappedY[k]]+=1
            
            if countLeft[-1]>countLeft[1]:
                countLeftClass=-1
                countRightClass=1
            else:
                countLeftClass=1
                countRightClass=-1
            loss=0
            lossIndexes=[]
            for k in range(Yp.shape[1]):
                if(Yp[i][k]<midPoint):
                    if(mappedY[k]!=countLeftClass):
                        loss+=weights[k]
                        lossIndexes.append(k)
                else:
                    if(mappedY[k]!=countRightClass):
                        loss+=weights[k]
                        lossIndexes.append(k)
            loss=loss/sum(weights)

            if(loss<dimensionTreeList["loss"]):
                dimensionTreeList["loss"]=loss
                dimensionTreeList["lossIndexes"]=lossIndexes
                dimensionTreeList["midPoint"]=midPoint
                dimensionTreeList["countLeftClass"]=countLeftClass
                dimensionTreeList["countRightClass"]=countRightClass
        
        if(dimensionTreeList["loss"]<TreeList["loss"]):
            TreeList["loss"]=dimensionTreeList["loss"]
            TreeList["lossIndexes"]=dimensionTreeList["lossIndexes"]
            TreeList["midPoint"]=dimensionTreeList["midPoint"]
            TreeList["countLeftClass"]=dimensionTreeList["countLeftClass"]
            TreeList["countRightClass"]=dimensionTreeList["countRightClass"]
            TreeList["dimension"]=i
            TreeList["alpha"]=np.log((1-TreeList["loss"])/TreeList["loss"])
    # print(TreeList)
    for i in TreeList["lossIndexes"]:
        weights[i]*=((1-TreeList["loss"])/TreeList["loss"])
    accuracy=0
    for i in range(valDataSet.shape[0]):
        summation=0
        ip=np.dot(Up.T,valDataSet[i])
        for j in listTrees:
            if ip[j["dimension"]]<j["midPoint"]:
                summation+=j["alpha"]*j["countLeftClass"]
            else:
                summation+=j["alpha"]*j["countRightClass"]
        if ip[TreeList["dimension"]]<TreeList["midPoint"]:
            summation+=TreeList["alpha"]*TreeList["countLeftClass"]
        else:
            summation+=TreeList["alpha"]*TreeList["countRightClass"]
        if summation<0:
            if valMappedY[i]==-1:
                accuracy+=1
        else:
            if valMappedY[i]==1:
                accuracy+=1
    accuracy/=valDataSet.shape[0]
    returnTree={"dimension":TreeList["dimension"],"midPoint":TreeList["midPoint"],"countLeftClass":TreeList["countLeftClass"],"countRightClass":TreeList["countRightClass"],"accuracy":accuracy,"alpha":TreeList["alpha"]}
    return returnTree,weights

listTrees=[]            
weights=[1/(mappedY.shape[0]) for i in range(mappedY.shape[0])]
file=open("Tree2","w+")
for i in range(300):
    returnTree,weights=makeStump(Yp,mappedY,weights,valDataSet,valMappedY,Up,listTrees)
    print(returnTree)
    listTrees.append(returnTree)
    file.write(str(returnTree))
    file.write(str("\n"))
    file.flush()





# Iteration 1000

accuracy=[]
listTrees=[]
with open("Tree1000","r") as file:
    totalFile=file.readlines()
    for tree in totalFile:
        newTree=(eval(tree.strip()))
        newAccuracy=newTree['accuracy']
        listTrees.append(newTree)
        accuracy.append(newAccuracy)
fig = plt.figure(figsize=(20, 2))
ax = fig.add_subplot(111)
ax.plot(accuracy)
plt.show()

accuracy=0
index=-1
for j in range(len(listTrees)):
       if listTrees[j]["accuracy"]>=accuracy:
             index=j

predictedClassAccuracy={-1:0,1:0}
predictedClassCount={-1:0,1:0}
for i in range(testX.shape[0]):
    summation=0
    ip=np.dot(Up.T,testX[i])
    for j in range(index):
        if ip[listTrees[j]["dimension"]]<listTrees[j]["midPoint"]:
            summation+=listTrees[j]["alpha"]*listTrees[j]["countLeftClass"]
        else:
            summation+=listTrees[j]["alpha"]*listTrees[j]["countRightClass"]
    if summation<0:
        if testy[i]==-1:
            predictedClassAccuracy[-1]+=1
    else:
        if testy[i]==1:
            predictedClassAccuracy[1]+=1
    predictedClassCount[testy[i]]+=1
# print(predictedClassAccuracy,predictedClassCount)
print("Total Accuracy is:",(predictedClassAccuracy[-1]+predictedClassAccuracy[1])/(predictedClassCount[-1]+predictedClassCount[1])*100)
predictedClassAccuracy[-1]=(predictedClassAccuracy[-1]/predictedClassCount[-1])*100
predictedClassAccuracy[1]=(predictedClassAccuracy[1]/predictedClassCount[1])*100
print("Class Wise Accuracy is:",predictedClassAccuracy)





# IIteration 100

# accuracy=[]
# listTrees=[]
# with open("Tree100","r") as file:
#     totalFile=file.readlines()
#     for tree in totalFile:
#         newTree=(eval(tree.strip()))
#         newAccuracy=newTree['accuracy']
#         listTrees.append(newTree)
#         accuracy.append(newAccuracy)
# fig = plt.figure(figsize=(20, 2))
# ax = fig.add_subplot(111)
# ax.plot(accuracy)
# plt.show()

# accuracy=0
# index=-1
# for j in range(len(listTrees)):
#        if listTrees[j]["accuracy"]>=accuracy:
#              index=j

# predictedClassAccuracy={-1:0,1:0}
# predictedClassCount={-1:0,1:0}
# for i in range(testX.shape[0]):
#     summation=0
#     ip=np.dot(Up.T,testX[i])
#     for j in range(index):
#         if ip[listTrees[j]["dimension"]]<listTrees[j]["midPoint"]:
#             summation+=listTrees[j]["alpha"]*listTrees[j]["countLeftClass"]
#         else:
#             summation+=listTrees[j]["alpha"]*listTrees[j]["countRightClass"]
#     if summation<0:
#         if testy[i]==-1:
#             predictedClassAccuracy[-1]+=1
#     else:
#         if testy[i]==1:
#             predictedClassAccuracy[1]+=1
#     predictedClassCount[testy[i]]+=1
# # print(predictedClassAccuracy,predictedClassCount)
# print("Total Accuracy is:",(predictedClassAccuracy[-1]+predictedClassAccuracy[1])/(predictedClassCount[-1]+predictedClassCount[1])*100)
# predictedClassAccuracy[-1]=(predictedClassAccuracy[-1]/predictedClassCount[-1])*100
# predictedClassAccuracy[1]=(predictedClassAccuracy[1]/predictedClassCount[1])*100
# print("Class Wise Accuracy is:",predictedClassAccuracy)





# Iteration 10

# accuracy=[]
# listTrees=[]
# with open("Tree10","r") as file:
#     totalFile=file.readlines()
#     for tree in totalFile:
#         newTree=(eval(tree.strip()))
#         newAccuracy=newTree['accuracy']
#         listTrees.append(newTree)
#         accuracy.append(newAccuracy)
# fig = plt.figure(figsize=(20, 2))
# ax = fig.add_subplot(111)
# ax.plot(accuracy)
# plt.show()

# accuracy=0
# index=-1
# for j in range(len(listTrees)):
#        if listTrees[j]["accuracy"]>=accuracy:
#              index=j

# predictedClassAccuracy={-1:0,1:0}
# predictedClassCount={-1:0,1:0}
# for i in range(testX.shape[0]):
#     summation=0
#     ip=np.dot(Up.T,testX[i])
#     for j in range(index):
#         if ip[listTrees[j]["dimension"]]<listTrees[j]["midPoint"]:
#             summation+=listTrees[j]["alpha"]*listTrees[j]["countLeftClass"]
#         else:
#             summation+=listTrees[j]["alpha"]*listTrees[j]["countRightClass"]
#     if summation<0:
#         if testy[i]==-1:
#             predictedClassAccuracy[-1]+=1
#     else:
#         if testy[i]==1:
#             predictedClassAccuracy[1]+=1
#     predictedClassCount[testy[i]]+=1
# # print(predictedClassAccuracy,predictedClassCount)
# print("Total Accuracy is:",(predictedClassAccuracy[-1]+predictedClassAccuracy[1])/(predictedClassCount[-1]+predictedClassCount[1])*100)
# predictedClassAccuracy[-1]=(predictedClassAccuracy[-1]/predictedClassCount[-1])*100
# predictedClassAccuracy[1]=(predictedClassAccuracy[1]/predictedClassCount[1])*100
# print("Class Wise Accuracy is:",predictedClassAccuracy)






# Iteration 2

# accuracy=[]
# listTrees=[]
# with open("Tree2","r") as file:
#     totalFile=file.readlines()
#     for tree in totalFile:
#         newTree=(eval(tree.strip()))
#         newAccuracy=newTree['accuracy']
#         listTrees.append(newTree)
#         accuracy.append(newAccuracy)
# fig = plt.figure(figsize=(20, 2))
# ax = fig.add_subplot(111)
# ax.plot(accuracy)
# plt.show()

# accuracy=0
# index=-1
# for j in range(len(listTrees)):
#        if listTrees[j]["accuracy"]>=accuracy:
#              index=j

# predictedClassAccuracy={-1:0,1:0}
# predictedClassCount={-1:0,1:0}
# for i in range(testX.shape[0]):
#     summation=0
#     ip=np.dot(Up.T,testX[i])
#     for j in range(index):
#         if ip[listTrees[j]["dimension"]]<listTrees[j]["midPoint"]:
#             summation+=listTrees[j]["alpha"]*listTrees[j]["countLeftClass"]
#         else:
#             summation+=listTrees[j]["alpha"]*listTrees[j]["countRightClass"]
#     if summation<0:
#         if testy[i]==-1:
#             predictedClassAccuracy[-1]+=1
#     else:
#         if testy[i]==1:
#             predictedClassAccuracy[1]+=1
#     predictedClassCount[testy[i]]+=1
# # print(predictedClassAccuracy,predictedClassCount)
# print("Total Accuracy is:",(predictedClassAccuracy[-1]+predictedClassAccuracy[1])/(predictedClassCount[-1]+predictedClassCount[1])*100)
# predictedClassAccuracy[-1]=(predictedClassAccuracy[-1]/predictedClassCount[-1])*100
# predictedClassAccuracy[1]=(predictedClassAccuracy[1]/predictedClassCount[1])*100
# print("Class Wise Accuracy is:",predictedClassAccuracy)