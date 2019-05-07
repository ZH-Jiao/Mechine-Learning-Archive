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


        dic = list()
        for j in feature:
            word = j[:int(j.find(":"))]
            dic.append(int(word))
        dic.append(-1)
        X.append(dic)
    # print(X)
    # print("\n")
    return X, Y



#apply SGD
def train(input, dictionary, numEpoch=40):
    print("start train")
    X,Y  = tsvToDict(input)

    print(len(Y))
    theta=[]
    learningRate = 0.1
    for i in dictionary:
        theta.append(0.0)
    theta.append(1.0)

    theta = runSGD(theta, X, Y, learningRate, numEpoch)

    return theta



def runSGD(theta, X, Y, learningRate, numEpoch):
    print("start runSGD")

    if numEpoch <= 0:
        return theta
    else:

        for i in range(len(Y)):

            #print("###########################start at ith")
            #get a max Xi for cutting off the overflow
            # xMax = getMax(X[i])

            #calculating the gradident
            u = 0.0
            dot = thetaDotX(theta, X[i])

            u = np.exp(dot)

            # result = xi[j] * (int(yi) - u / (1+u))
            gradient = (float(Y[i]) - u / (1.0 + u))

            for j in X[i]:
                #j is a single feature of X
                # if j in theta:
                #     thetaJ = theta[j]
                # else:
                #     thetaJ = 0.0
                #     theta[j] = 0.0
                thetaJ = theta[j]
                # dot = thetaDotX(theta, X[i])
                thetaJ = singleSGD(thetaJ, theta, learningRate, gradient, j)
                theta[j] = thetaJ
        print("Finish the last %s epoch" %numEpoch)
        return runSGD(theta, X, Y, learningRate, int(numEpoch)-1)

#update the theta J for single time
#theta J is a number
def singleSGD(thetaJ, theta, learningRate, gradient, j):
    """
    u = 0.0
    dot = thetaDotX(theta, xi)

    u = np.exp(dot)

    # result = xi[j] * (int(yi) - u / (1+u))
    gradient = (float(yi) - u / (1.0 + u))
    """
    #print("singleSGD start")
    thetaJ = thetaJ + learningRate*gradient
    #thetaJ = thetaJ - learningRate * partialDerivativeSGD(theta, xi, yi, j)
    #print("singleSGD end\n")
    return thetaJ

def thetaDotX(T, X):
    #print("dot start")

    product = 0.0


    for i in X:
        # if i in T:
            # product += T.get(i, 0) * (X[i] - xMax)
        product += T[i]
    #print("dot end\n")
    return product


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

    dot = thetaDotX(theta, xi)

    u = np.exp(thetaDotX(theta, xi))

    # p = np.power(u, yi)/(1+u)
    p = float(u/(1+u))
    # p = np.float64(p)
    return p



def predictLabels(input, theta):
    labels = []
    X, Y = tsvToDict(input)
    for i in range(len(X)):

        p = possibility(theta, 1, X[i])
        if p >= 0.5:
            labels.append(1)
        else:
            labels.append(0)
    return labels


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

    theta = train(trainInput, dictionary, numEpoch)


    trainLabels = predictLabels(trainInput, theta)
    testLabels = predictLabels(testInput, theta)
    listToFile(trainOut, trainLabels)
    listToFile(testOut, testLabels)
    X, Y = tsvToDict(trainInput)
    print(len(Y))
    print(negetiveConditionalLogLikelihood(theta, X, Y)/float(len(Y)))

    output = metrics(trainLabels, testLabels, trainInput, testInput)
    listToFile(metricsOut, output)





