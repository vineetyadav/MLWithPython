from numpy import zeros
import os
from knn import classify0
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileSet = os.listdir('trainingDigits')
    m = len(trainingFileSet)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileSet[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,] = img2vector('trainingDigits/%s'%fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount =0.0
    mTest = len(testFileList[i])
    fileStr = fileNameStr.split('.')[0]
    classNumStr = fileNameStr.split('_')[0]
    vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
    classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
    print "the classifier came back with:%d, the real answer is:%d"%(classifierResult,classNumStr)
    if(classifierResult!=classNumStr):
        errorCount+=1.0
    print "\nThe total number of error is:%d"%errorCount
    print "\nthe Total error rate is:%f"(errorCount/float(mTest))
    
        