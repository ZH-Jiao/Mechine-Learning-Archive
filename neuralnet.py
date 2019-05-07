from __future__ import print_function
import sys
import copy
import numpy as np
import math




# x = [ , , , , ] T
def linearForward(x, alpha):
    #alpha: n x m, x: mX1




    a = np.dot(alpha,x)
    #print("A", a, a.shape)
    # a: nX1
    return a

def sigmoidForward(a):
    z = 1.0/(1.0+np.exp(-a))
    return z

def softmaxForward(b):
    exp = np.exp(b)
    sum = 0.0
    for i in exp:
        sum += i
    return exp / sum

def crossEntropyForward(y,yHat):
    return y * np.log(yHat)

class Quan():
    def __init__(self, x, a, z, b, yHat, J):
        self.x = x
        self.a = a
        self.z = z
        self.b = b
        self.yHat = yHat
        self.J = J



def linearBackward(z,beta,gb):
    print("LBackward", gb.shape, z.shape, beta.shape)
    gbeta = np.dot(gb.reshape(len(gb),1).T, z.reshape(1,len(z)).T)
    #print("Beta in LB", beta.shape)
    betaStar = copy.deepcopy(beta)
    betaStar = np.delete(betaStar, 0, axis=1)
    #print("beta star",betaStar.shape)
    #print("gB in LB", gb.shape)
    gz = np.dot(betaStar.T, gb)
    return gbeta, gz

def linearBackward2(x,alpha,ga):
    galpha = np.dot(ga.reshape(len(ga), 1), x.reshape(1, len(x)))

    ga = np.dot(alpha.T, ga)
    #print("galpha", galpha.shape, ga.shape)
    return galpha, ga

def gSigmoid(z):
    a = np.exp(-z)
    return a/((1+a)**2)

def sigmoidBackward(a, z, gz):
    z = np.delete(z,0)
    return np.multiply(np.multiply(gz,z), 1-z)

def NNForward(x, y, alpha, beta):

    x = x.reshape(1,-1).T
    y = y.reshape(1,-1)
    print("NNforward", x.shape, alpha.shape)
    a = linearForward(x, alpha)
    z = sigmoidForward(a)

    z = np.insert(z, 1, 0, axis=0)
    b = linearForward(z, beta)

    yHat = softmaxForward(b)
    J = crossEntropyForward(y,yHat)

    o = Quan(x, a, z, b, yHat, J)
    return o


def NNBackward(x, y, alpha, beta, o):
    gJ = 1
    x = x.reshape(1, -1).T
    y = y.reshape(1, -1)
    #gyHat = crossEntropyBackward()
    #gb = softmaxBackward(b, y, yHat, J, gJ)
    print(o.yHat.shape, y.shape)
    gb = gJ * (o.yHat.T - y)
    print("GB",gb.shape)
    #print("z", o.z.shape)
    gbeta, gz = linearBackward(o.z,beta,gb)
    #print("gBeta",gbeta.shape,"gZ", gz.shape)
    #print("Z",o.z.shape)
    ga = sigmoidBackward(o.a, o.z, gz)
    galpha, gx = linearBackward2(x, o.a, ga)

    return galpha, gbeta

def endForward(x, y, alpha, beta):

    MCE = 0.0
    error = 0.0
    label = []

    for i in range(len(x)):
        #print("xi",x[i].shape)
        o = NNForward(x[i], y[i], alpha, beta)
        # print("SHAPE", y[i].shape,o.yHat.shape)
        MCE += -np.sum(np.dot(y[i].reshape(1,-1), np.log(o.yHat).reshape(-1,1)))
        yHatId = np.argmax(o.yHat.reshape(-1,1))
        #print("yhat",o.yHat)
        #print("ID", yHatId)
        yHat = np.zeros(len(y[0]))
        yHat[yHatId] = 1

        label.append(yHatId)
        #print("yHatId", yHat, y[i])

        if np.argmax(yHat) != np.argmax(y[i]):
            error += 1.0

    print(MCE)
    MCE = MCE / len(y)
    return MCE, error / float(len(y)), label

def oneHotY(y):
    result = np.zeros((len(y),10))
    for i in range(len(y)):
        result[i][y[i]] = 1
    return result


def SGD(x, y, testX, testY, units, numepoch):
    print("start SGD")
    print(x, testX)
    # dimension of Y
    k = 10

    metric = []

    if int(initFlag) == 1:
        alpha = np.random.rand(units, len(x[0])) / 10

        beta = np.random.rand(k, units) / 10

    elif int(initFlag) == 2:

        # initialize alpha
        alpha = np.zeros((units, len(x[0])))
        # initialize beta

        beta = np.zeros((k, units))

    print("SGD1,x, testx", x.shape, testX.shape)

    alpha = np.insert(alpha, 0, 0, axis=1)
    beta = np.insert(beta, 0, 0, axis=1)
    x = np.insert(x, 0, 1, axis=1)
    testX = np.insert(testX, 0, 1, axis=1)

    y = oneHotY(y)
    testY = oneHotY(testY)

    trainLabel = []
    testLabel = []

    print("alpha, X", alpha.shape, x.shape, testX.shape)
    for j in range(numepoch):
        for i in range(len(x)):
            #print("X", x.shape)
            #print("Xi", x[i].shape)
            o = NNForward(x[i], y[i], alpha, beta)

            galpha, gbeta = NNBackward(x[i], y[i], alpha, beta, o)

            alpha = alpha - learningRate * galpha
            beta = beta - learningRate * gbeta

        #print("alpha, X[-1]", alpha.shape, x[-1].shape, testX.shape)

        trainMCE, errorTrain, trainLabel = endForward(x, y, alpha, beta)
        testMCE, errorTest, testLabel = endForward(testX, testY, alpha, beta)


        print("epoch=" + str(j+1) + " crossentropy(train): " + str(trainMCE))
        print("epoch=" + str(j+1) + " crossentropy(test): " + str(testMCE) + "\n")
        metric.append("epoch=" + str(j+1) + " crossentropy(train): " + str(trainMCE))
        metric.append("epoch=" + str(j+1) + " crossentropy(test): " + str(testMCE))

##########################################################################################################################
    #errorTrain, errorTest = error(alpha, beta, x, y, testX, testY)
    metric.append("error(train): " + str(errorTrain))
    metric.append("error(test): " + str(errorTest))


    return alpha, beta, metric, trainLabel, testLabel


"""
def error(alpha, beta, x, y, testX, testY):
    oTrain = NNForward(x, y, alpha, beta)
    oTest = NNForward(testX, testY, alpha, beta)
    trainRight = 0
    testRight = 0

    ndarray.max(axis=None, out=None, keepdims=False)
    for i in range(len(np.nditer(oTrain.yHat))):

        if  -0.5 < np.nditer(oTrain.yHat)[i] - np.nditer(y)[i] < 0.5:
            trainRight += 1
    for i in range(len(np.nditer(oTest.yHat))):

        if -0.5 < np.nditer(oTest.yHat)[i] - np.nditer(yTest)[i] < 0.5:
           testright += 1


    trainError =1 - trainRight/len(np.nditer(oTrain.yHat))
    testError = 1- testRight/len(np.nditer(oTest.yHat))
    return  trainError, testError
"""

def readFile(path):
    with open(path,"rt") as f:
        return f.read()

def textToList(text):
    text = text.splitlines()
    #print(type(text))
    #print(text)
    print("test to list start")
    # y is the output label
    y = np.array([])
    yList = []

    x = []
    xList = []

    for i in text:

        i = i.split(",")
        #y = np.append(y, i[0])
        yList.append(int(i[0]))
        xList.append(i[1:])
    y = np.array(yList)
    x = np.array(xList)
    x = x.astype(np.float64)

    """
    for i in text:
        i = i.split(",")
        print(i)
        i = i[1:].insert(0, 1)
        print("iP", i)
        #lst = np.array(i)
        #np.append(x, lst)
        xList.append(i)
    x = np.array(xList)
    
    """
    return x,y

def listToFile(outfile, list):
    outTxt = open(outfile, "wb")

    for i in list:
        i = str(i)+"\n"
        outTxt.write(i)

def getLabel(o):
    result = []
    for i in o.yHat:
        if i >= 0.5:
            result.append(1)
        else:
            result.append(0)
    return result

if __name__ == '__main__':
    print("code start")
    #command line
    #formatted input
    trainInput = sys.argv[1] #train input
    trainInput = readFile(trainInput)


    testInput = sys.argv[2]
    testInput = readFile(testInput)

    trainOut = sys.argv[3]
    testOut = sys.argv[4]


    metricsOut = sys.argv[5]

    numEpoch = sys.argv[6]
    numEpoch = int(numEpoch)

    hiddenUnits = sys.argv[7]
    hiddenUnits = int(hiddenUnits)

    #1 for RANDOM, 2 for ZERO initialization
    initFlag = sys.argv[8]

    learningRate = sys.argv[9]
    learningRate = float(learningRate)
    print("input finish")




    #run
    print("start")

    x, y = textToList(trainInput)
    testX, testY = textToList(testInput)
    print("START", x.shape, testX.shape)
    print("finish XY")
    #alpha, beta = initialize(alpha, beta, initFlag)

    alpha, beta, metric, trainLabel, testLabel = SGD(x, y, testX, testY, hiddenUnits, numEpoch)
    print("finish SGD")
    #newO = NNForward(x, y, alpha, beta)
    #trainLabel = getLabel(newO)

    #newO = NNForward(testX, testY, alpha, beta)
    #testLabel = getLabel(newO)

    listToFile(trainOut, trainLabel)
    listToFile(testOut, testLabel)
    #print(metric)
    listToFile(metricsOut, metric)
    #outTxt = open(metricsOut, "wb")
    #outTxt.write(metric)


