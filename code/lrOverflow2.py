from __future__ import print_function
import sys
import copy
import numpy as np
import math

def readFile(path):
    with open(path, "rt") as f:
        return f.read()

def txtToDict(text):
    text = text.splitlines()
    # list of lines here

    dictionary = dict()
    #make lines into dict
    for i in text:
        l = i.split()
        dictionary[l[0]] = l[-1]

    return dictionary

def tsvToDict(input):
    input = input.splitlines()
    Y = []
    X = []
    for i in input:
        i.split("\t")
        Y.append(i[0])
        feature = i[1:]
        feature = feature.split("\t")
        feature = feature[1:]


        dic = dict()
        for j in feature:
            word = j[:int(j.find(":"))]
            dic[int(word)] = 1
        X.append(dic)

    return X, Y



#apply SGD
def train(input, numEpoch=40):

    X,Y  = tsvToDict(input)
    print(len(Y))
    theta = {}
    learningRate = 0.1

    theta = runSGD(theta, X, Y, learningRate, numEpoch)

    return theta

def getMax(xi):
    max = 0.0
    for j in xi:
        if j > max:
            max = j
    
    return max

def runSGD(theta, X, Y, learningRate, numEpoch):

    if numEpoch <= 0:
        return theta
    else:

        for i in range(len(Y)):
            #print("###########################start at ith")
            #get a max Xi for cutting off the overflow
            xMax = getMax(X[i])
            for j in X[i]:
                #j is a single feature of X
                thetaJ = theta.get(j, 0)


                thetaJ = singleSGD(thetaJ, theta, xMax, learningRate, X[i], Y[i], j)
                theta[j] = thetaJ
        print("Finish the last %s epoch" %numEpoch)
        return runSGD(theta, X, Y, learningRate, int(numEpoch)-1)

#update the theta J for single time
#theta J is a number
def singleSGD(thetaJ, theta, xMax, learningRate, xi, yi, j):
    #print("singleSGD start")
    thetaJ = thetaJ + learningRate*partialDerivativeSGD(theta, xMax, xi, yi, j)
    #print("singleSGD end\n")
    return thetaJ

def partialDerivativeSGD(theta, xMax, xi, yi, j):
    u = 0.0
    dot = thetaDotX(theta, xi, xMax)
    
    u = math.pow(math.e, dot)

    result = xi[j] * (int(yi) - u / (1+u))

    return result


def negetiveConditionalLogLikelihood(theta, X, Y):
    nCLL = 0.0
    for i in range(len(Y)):
        dot = thetaDotX(theta, X[i])
        u = np.complex64(1.0)
        u = np.exp(dot)
        u = np.float64(u)
        result = - float(Y[i]) * dot + np.log(1+u)
        nCLL += result



    return nCLL

def possibility(theta, yi, xi):
    productSum = 1.0


    #u = math.pow(math.e, thetaDotX(theta, xi))
    u = np.complex64(1.0)
    #u = "%.16f" % (np.power(math.e, thetaDotX(theta, xi)))
    #u = np.power(math.e, thetaDotX(theta, xi))
    dot = thetaDotX(theta, xi)

    u = np.exp(thetaDotX(theta, xi))
    """
    try:
        u = math.exp(thetaDotX(theta, xi))
    except OverflowError:
        u = float('inf')
    """
    p = np.power(u, yi)/(1+u)
    p = np.float64(p)
    return p



def possibility2(theta, y, x):
    productSum = 1.0
    for i in range(len(x)):
        u = 0.0
        u = math.pow(math.e, thetaDotX(theta, x[i]))
        productSum = productSum * ((math.pow(u, y[i]))/(1+u))
    return productSum



def predictLabels(input, theta):
    labels = []
    X, Y = tsvToDict(input)
    for i in range(len(X)):

        p = possibility(theta, 1, X[i])
        if p > 0.5:
            labels.append(1)
        else:
            labels.append(0)
    return labels

#theta: [123:0.123, 23:3442, 45:12312424, ...]
#xi: {123:1,23:1,45:1,24:1} less than M in sum
#x: [{x,x,x}first data, {x,x,x}second data..., {x,x,x}ith data] N in sum]
def thetaDotX(theta, xi, xMax):
    return dot(theta, xi, xMax)

"""
def dot(T, X, xMax):
    product = 0.0
    for i,t in T.items():
        if i in X:
            product += t * X.get(i,0)
    return product

"""
def dot(T, X, xMax):
    #print("dot start")
    product = 0.0

    for i in X:
        if i in T:
            product += T.get(i, 0) * (X[i] - xMax)

    #print("dot end\n")
    return product

def listToFile(outfile, list):
    outTxt = open(outfile, "wb")


    for i in list:
        i = str(i)+"\n"
        outTxt.write(i)

def metrics(trainLabels, testLabels, trainInput, testInput):
    trainX, trainY = tsvToDict(trainInput)
    testX, testY = tsvToDict(testInput)
    trainError = 0.0
    trainRightCount = 0
    testError = 0.0
    testErrorCount = 0


    for i in range(len(trainLabels)):

        if int(trainLabels[i]) == int(trainY[i]):

            trainRightCount += 1

    trainError = 1.0 - (trainRightCount / float(len(trainLabels)))


    for i in range(len(testLabels)):
        if int(testLabels[i]) != int(testY[i]):
            testErrorCount += 1
    testError = testErrorCount / float(len(testLabels))


    output=[]
    output.append("error(train): %f" %trainError)
    output.append("error(test): %f" %testError)
    return output



if __name__ == '__main__':
    #command line
    #formatted input
    trainInput = sys.argv[1] #train input
    trainInput = readFile(trainInput)

    validationInput = sys.argv[2]
    validationInput = readFile(validationInput)


    testInput = sys.argv[3]
    testInput = readFile(testInput)

    dictInput = sys.argv[4]
    dictInput = readFile(dictInput)

    trainOut = sys.argv[5]

    testOut = sys.argv[6]

    metricsOut = sys.argv[7]

    numEpoch = sys.argv[8]

    #run

    dictionary = txtToDict(dictInput)

    theta = {}
    theta = train(trainInput, numEpoch)


    trainLabels = predictLabels(trainInput, theta)
    testLabels = predictLabels(testInput, theta)
    listToFile(trainOut, trainLabels)
    listToFile(testOut, testLabels)
    X, Y = tsvToDict(trainInput)
    print(len(Y))
    print(negetiveConditionalLogLikelihood(theta, X, Y)/float(len(Y)))

    output = metrics(trainLabels, testLabels, trainInput, testInput)
    listToFile(metricsOut, output)





