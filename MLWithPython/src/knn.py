from numpy import tile
from numpy import zeros
from numpy import shape
import operator
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #Equlidian Distance Calculation 
    diffMat = tile(inX,(dataSetSize,1)) -dataSet
    sqDiffMat  = diffMat*2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistInstances = distances.agrsort()
    classCount = ()
    #voting with Lowest K-distances 
    for i in range(k):
        voteIlabel = labels[sortedDistInstances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    sortedClassCount = sorted(classCount.itemItems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount
#File2matrix generates tab sep files where three are present in tab seperated in vector 
def file2Matrix(filename):
    fr = open(filename)
    #Get number OF Lines in file 
    numberOfLines =len(fr.readLine())
    #create defualt numpy matrix
    returnMat = zeros((numberOfLines,3))
    classLabelVector =[]
    fr = open(filename)
    index =0
    for line in fr.readLine():
        #parse line to list
        line =line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index = index+1
    return returnMat.classLabelVector

#function is used to normalize dataset 
def autoNorm(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataset = zeros(shape(dataset))
    m = dataset.shape[0]
    #Elementwise Division
    normDataset = dataset - tile(minVals,(m,1))
    normDataset = normDataset/tile(ranges,(m,1))
    return normDataset,ranges,minVals

def datingClassTest():
    hoRatio = 0,10
    datingDataMat,datingLabels = file2Matrix('datingTestSet.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount =0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifer came back with :%d, the real answer is :%d" %(classifierResult,datingLabels[i])
        if(classifierResult!=datingLabels[i]):
            errorRate=+0.1
    print "the total error rate is: %f"%(errorRate/float(numTestVecs))


