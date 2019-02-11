from __future__ import print_function
import sys
import copy
import numpy as np



def readFile(path):
    with open(path, "rt") as f:
        return f.read()

#labels should be a list of labels
def largerProbability(labels):
    labelsValue = set()
    for i in labels:
        labelsValue.add(i)

    p = []
    for value in labelsValue:
        #proberbility of binary
        p0=0.0

        for j in labels:
            if j == value:
                p0 += 1
        p0 = p0/len(labels)
        p.append(p0)

    return max(p)

def majorityVote(singleData):
    labelsValue = dict()

    for i in singleData:
        labelsValue.setdefault(i, [])
        labelsValue[i].append(0)
    guess = None
    count = 0
    for i in labelsValue:
        if len(labelsValue[i]) > count:
            count = len(labelsValue[i])
            guess = i

    return guess


#calculate H(X)
def entropy(labels):
    labelsValue = set()
    for i in labels:
        labelsValue.add(i)

    entro = 0.0
    for value in labelsValue:
        #proberbility of binary
        p0=0.0

        for j in labels:
            if j == value:
                p0 += 1
        p0 = p0/len(labels)

        logp = np.log2(p0)
        #entropy to be add up
        entro -= p0*logp

    return entro

#return error rate of the label
def errorRate(labels):
    return 1-largerProbability(labels)

def infoGain(labels, attribute):
    attributeValue = set()
    labelsValue = set()

    for i in labels:
        labelsValue.add(i)
    for j in attribute:
        attributeValue.add(j)


    #H(Y|X)
    hYX = 0.0
    for x in attributeValue:
        #H(Y|X=v)
        hYXV = 0.0
        pX = 0.0

        for y in labelsValue:

            pXY = 0.0

            xCount = 0.0
            xyCount = 0.0

            for i in range(len(attribute)):
                if attribute[i] == x:
                    xCount += 1.0
                    if labels[i] == y:
                        xyCount += 1.0
            #print("xcount", xCount)
            pXY = xyCount/float(len(attribute))
            pX = xCount/float(len(attribute))
            #print("pX",pX)
            #P(Y=i | X=v) >>>> P()logP()
            pXYpX = pXY/pX
            #print("PXYPX", pXYpX)
            if pXYpX != 0:
                pLogP =  (pXYpX) * np.log2(pXYpX)
                #
                hYXV -= pLogP

        hYX -= -hYXV * pX
        #print("hyx", hYX)
    #H(Y)
    hY = entropy(labels)
    #print(hYX)
    return hY - hYX

label = [0,1,1,1]
att = [0,2,0,2]







#################################
#build tree
def biggestMutualInformation(labels,data):
    data = data[:-1]
    biggestMutualInfo = 0
    attributeToSplitID = 0
    for i in range(len(data)):
        #m is current info gain
        m = infoGain(labels, data[i])

        if m > biggestMutualInfo:
            biggestMutualInfo = m
            #ID of attributeToSplit
            attributeToSplitID = i


    return attributeToSplitID

class Node:
    def __init__(self,key,attributeID):
        self.left = None
        self.right = None
        self.val = key
        self.attribute = attributeID
        #False means not a leaf node
        self.isleaf = False

        #at the super stage
        self.labelClass = None
        self.superAttribute = None
        #self.labelVal is { attributeValue: predict labelValue}
        self.labelVal = dict()
        self.splitValue = None
        self.depth = 0
        #superSplit is the split option
        self.superSplit = None
        # node.attValue is like {y:[2,4,7], n:[1,3]}
    def attributeValue(self, attValue):
        self.attValue = attValue

    def callAttributeValue(self):
        return self.attValue
    def saveDepth(self, depth):
        self.depth = depth

#Print the Tree
# A function to do inorder tree traversal def printInorder(root):
def printPreorder(root):
    if root:


        # First print the data of node
        print(root.val, root.attribute, "depth", root.depth, root.labelVal.values(), "\t",end="")
        # Then recur on left child
        printPreorder(root.left)

        print("turn right")
        # Finally recur on right child
        printPreorder(root.right)

#print tree by DFS

def printTree(root, trainFile, title, count=0):

    if count == 0:
        data, labels = dataCleanReverse(trainFile)
        labelsValue = dict()

        for i in labels:
            labelsValue.setdefault(i, [])
            labelsValue[i].append(0)

        keys = labelsValue.keys()

        print("[", end="")
        print(len(labelsValue[keys[0]]), keys[0], "/", end="")
        print(len(labelsValue[keys[1]]), keys[1], end="")
        print("]")


    if root:

        if count != 0:
            for d in range(int(root.depth-1)):
                print("| ", end="")
            #print(title, root.superAttribute)

            print(title[root.superAttribute], "= ", end="")
            print(root.superSplit, end="")
            print(":", "[",end = "")
            print(root.labelClass.positiveNum, root.labelClass.positiveName, "/",end="")
            print(root.labelClass.negativeNum, root.labelClass.negativeName, end="")
            print("]")



        # First print the data of node
        #print(root.val, root.attribute, "depth", root.depth, root.labelVal.values(), "\t",end="")
        # Then recur on left child
        printTree(root.left, trainFile, title, count+1)

        #print("turn right")
        # Finally recur on right child
        printTree(root.right, trainFile, title, count+1)

class AllLabel():
    def __init__(self,labels):
        self.positiveNameAll = None
        self.negativeNameAll = None
        positiveName = None
        negativeName = None
        for i in labels:
            if positiveName == None:
                positiveName = i
            if positiveName != i and negativeName == None:
                negativeName = i



        self.positiveNameAll = positiveName
        self.negativeNameAll = negativeName

#categorize the labels
class Label(AllLabel):
    def __init__(self,labels):
        global allLabel
        self.labels = labels
        positiveName = None
        negativeName = None
        self.positiveNum = 0
        self.negativeNum = 0
        #print("labelsclass",labels)
        for i in labels:


            if allLabel.positiveNameAll == i:
                self.positiveNum += 1
            elif allLabel.negativeNameAll == i:
                self.negativeNum += 1

        self.positiveName = allLabel.positiveNameAll
        self.negativeName = allLabel.negativeNameAll

"""
#categorize the labels
class Label(AllLabel):
    def __init__(self,labels):
        self.labels = labels
        positiveName = None
        negativeName = None
        self.positiveNum = 0
        self.negativeNum = 0
        print("labelsclass",labels)
        for i in labels:
            if positiveName == None:
                positiveName = i
            if positiveName != i and negativeName == None:
                negativeName = i

            if positiveName == i:
                self.positiveNum += 1
            elif negativeName == i:
                self.negativeNum += 1

        self.positiveName = positiveName
        self.negativeName = negativeName
"""

#data should be like [[x,x,x],[x,x,x],[y,y,y]]
def buildTree(data, labelsResult, nodeID, branchDataID = None):
    #data = branchData, labelsResult = branchData[-1]

    #print("111LLL",labelsResult)
    attID = biggestMutualInformation(labelsResult, data)
    node = Node(nodeID, attID)

    #get the element in selected attribute
    #key: att value, value: attribute row ID
    attValue = dict()
    if branchDataID == None:
        for a in range(len(data[0])):
            if data[attID][a] in attValue:
                attValue[data[attID][a]].append(a)
            else:
                attValue[data[attID][a]] = [a]
    else:
        for a in range(len(data[0])):
            if data[attID][a] in attValue:
                attValue[data[attID][a]].append(branchDataID[a])
            else:
                attValue[data[attID][a]] = [branchDataID[a]]
    '''
    for value in range(len(attValue)):
        for i in attValue[value]:
            if i ==
    '''
    node.attributeValue(attValue)
    #categorize the labels

    labelClass = Label(labelsResult)
    node.labelClass = labelClass


    ##Calculate the label in this split (Majority vote)
    #node.attValue is like {y:[2,4,7], n:[1,3]}


    positiveVote = 0
    negativeVote = 0
    for split in node.attValue:
        for id in range(len(node.attValue[split])):
            if labelsResult[id] == labelClass.positiveName:
                positiveVote += 1
            elif labelsResult[id] == labelClass.negativeName:
                negativeVote += 1
        #split is current attribute value,

        if positiveVote >= negativeVote:
            if positiveVote > 0:

                node.labelVal[split] = labelClass.positiveName
                node.splitValue = split
        else:
            if negativeVote > 0:
                node.labelVal[split] = labelClass.negativeName
                node.splitValue = split

    return node




#recursion
#data is in the format of [[att1,x,x,x,x,x],[att2,,x,x,x,x,x],[label,y,y,y,y,y]]
def trainTree(data, currNode, depth = 1, maxDepth = 2):
    #print("DDDATA",data)

    currNode.saveDepth(depth)
    #data = data[:-1]
    if currNode == None:
        return None


    if len(currNode.labelVal) <= 1 or depth > maxDepth:
        currNode.isLeaf = True
        return None

    else:

        attValue = currNode.callAttributeValue()

        count = 0
        for n in attValue:
            count +=1
            #attValue[n] is a list of rowID in the attribute
            branchDataID = attValue[n]
            branchData = []


            #create a structure of branchData
            for col in range(len(data)):
                branchData.append([])

            for col in range(len(data)):
                for row in branchDataID:

                    #branchData[col].extend(data[col][branchDataID[row]])
                    branchData[col].append(data[col][row])

            #branchData for the X=n case
            """
            print('col', col)
            print('row', row)
            print('branchData', branchData)
            print('branchID', branchDataID)
            print('data', data)
            """
            global nodeID

            if count == 1:
                nodeID += 1
                currNode.left = buildTree(branchData, branchData[-1], nodeID, branchDataID)
                currNode.left.superSplit = n
                currNode.left.superAttribute = currNode.attribute

                result = trainTree(data, currNode.left, depth + 1, maxDepth)
            elif count == 2:
                nodeID += 1
                currNode.right = buildTree(branchData, branchData[-1], nodeID, branchDataID)
                currNode.right.superSplit = n
                currNode.right.superAttribute = currNode.attribute
                result = trainTree(data, currNode.right, depth + 1, maxDepth)

            #trainTree(data, currNode.right, depth+1)

    return currNode

#reverse list of data to
def reverse(data):

    newData = []
    for col in range(len(data[0])):
        newData.append([])
    for row in range(len(data)):
        for col in range(len(data[row])):
            newData[col].append(data[row][col])
    return newData

#Apply data to trained tree and get predict labels
def generateLabels(tree, data):
    predictLabelList = []
    for i in range(len(data)):
        #data[i][tree.attribute] is the value of the attribute in this row of data
        #tree.attribute is the No. of all the attribute (colID)
        predictLabel = recursiveLabels(tree, data[i])
        #print("=================generateLabels in TEST =====================")
        #print(predictLabel)
        predictLabelList.append(predictLabel)

    return predictLabelList

#return the Final predict Label for the given one piece of data
def recursiveLabels(node, oneData):
        #print("=================recursiveLabels in TEST =====================")
        #print(oneData)
        #print(node.labelVal)

        result = None

        if node.left == None and node.right == None:
            result = majorityVote(node.labelClass.labels)

            return result

        if len(node.labelVal) <= 1:
            #print("labelVal in else")
            #print(node.labelVal)
            for key in node.labelVal:
                result = node.labelVal[key]

            return result

        else:

            if oneData[node.attribute] == node.left.superSplit:
                #print("left", node.left.labelVal)
                result = recursiveLabels(node.left, oneData)
            elif oneData[node.attribute] == node.right.superSplit:
                #print("right", node.right.labelVal)

                result = recursiveLabels(node.right, oneData)
            #result = majorityVote(node.labelClass.labels)

            return result
            """
            if node.left != None:
                print("left", node.left.labelVal)
                if oneData[node.left.attribute] == node.left.superSplit:
                result = recursiveLabels(node.left, oneData)
            elif node.right != None:
                print("right", node.right.labelVal)
                if oneData[node.right.attribute] == node.right.superSplit:
                result = recursiveLabels(node.right, oneData)
                """
            """
            else:
                print("labelVal in else", node.labelVal)
                for key in node.labelVal:
                    result = node.labelVal[key]
            """



#transform the raw text to a usable data
#data is in the format of [[att1,x,x,x,x,x],[att2,,x,x,x,x,x],[label,y,y,y,y,y]]
def dataCleanReverse(text):

    text = text.splitlines()

    text = text[1:]
    labels = []

    #print("text", text)

    # make text a 2D list
    for i in text:
        labels.append(i.split(",")[-1])

    newText = []
    for i in text:
        newText += [i.split(",")]


    data = reverse(newText)
    #data = data[:-1]

    return data, labels

def dataClean(text):
    text = text.splitlines()

    text = text[1:]
    labels = []

    #print("text", text)

    # make text a 2D list
    for i in text:
        labels.append(i.split(",")[-1])

    newText = []
    for i in text:
        newText += [i.split(",")]

    #print("text2", newText)



    return newText

#run train function, return root of the Tree
def runTrain(trainFile, maxDep):

    t = trainFile
    data, labels = dataCleanReverse(t)


    # training
    labelsCopy = copy.deepcopy(labels)

    global allLabel
    allLabel = AllLabel(labelsCopy)
    #print("input data=======================================")
    #print(data)
    root = None
    if maxDep == 0:
        root = None
    elif maxDep >= 1:
        root = buildTree(data, labelsCopy, 0)
        #if maxDep >=2:
        trainTree(data, root, maxDepth = maxDep )

    return root

#apply tree to the given data(with labels), return predicted labels
def applyTree(tree, inputFile):
    test = inputFile
    testClean = dataClean(test)
    if tree != None:

        testLabels = generateLabels(tree, testClean)
    else:

        data, labels = dataCleanReverse(inputFile)
        betterLabel = majorityVote(labels)
        testLabels = []
        for i in range(len(labels)):
            testLabels.append(betterLabel)
    return testLabels

#return error rate from the predictedLabels to the data(file)
def metricsCalculator(data, predictedLabels):
    data, labels = dataCleanReverse(data)

    correctCount = 0.0
    for i in range(len(labels)):
        if labels[i] == predictedLabels[i]:
            correctCount += 1

    correctRate = correctCount/len(labels)
    return (1 - correctRate)


#Global varible nodeID!!
nodeID = 0
allLabel = None

"""
#Test Code
t = readFile("C:/Users/Zhiheng/OneDrive/CMU/CourseFile/10601 Machine Learning/HW2/handout (2)/handout/small_train.csv")

t = t.splitlines()

text = t[1:]
labels = []

print("text",text)

#make text a 2D list
for i in text:
    labels.append(i.split(",")[-1])

newText = []
for i in text:
    newText += [i.split(",")]


print("text2",newText)
data = reverse(newText)
data = data[:-1]


data, labels = dataCleanReverse(t)

print("labels111",labels)
#training

labelsCopy = copy.deepcopy(labels)
print("input data=======================================")
print(data)
root = buildTree(data, labelsCopy, 0)

trainTree(data,root)
print("print Tree ===============================================================================")
#printPreorder(root)
#test data
print("\nTEST=====================================================================================")
test = readFile("C:/Users/Zhiheng/OneDrive/CMU/CourseFile/10601 Machine Learning/HW2/handout (2)/handout/small_train.csv")
testClean = dataClean(test)
testLabels = generateLabels(root, testClean)

print(entropy(labels))
print(errorRate(labels))
print("test label output", testLabels)
print("xxxx xxxxx xxxxxx", labels)
print(type(t))
print(type(root))
print(type(root.left))
print(type(root.left.left))
print(root.left.left.left)

print(t)
train = copy.copy(test)


print(train)
title = train.splitlines()[0].split(",")
print("title", title)
printTree(root, t, title)
"""
"""
print('root', root.callAttributeValue())
print('root left', root.left.callAttributeValue())
print('root left left', root.left.left.callAttributeValue())
"""

def listToFile(outfile, list):
    outTxt = open(outfile, "wb")


    for i in list:
        i = i+"\n"
        outTxt.write(i)
"""
#new test
infile = readFile("C:/Users/Zhiheng/OneDrive/CMU/CourseFile/10601 Machine Learning/HW2/handout (2)/handout/politicians_train.csv")
testInput = readFile("C:/Users/Zhiheng/OneDrive/CMU/CourseFile/10601 Machine Learning/HW2/handout (2)/handout/politicians_test.csv")
maxDepth = 7

#Run train

tree = runTrain(infile, maxDepth)

#apply trainData to the tree
trainLabels = applyTree(tree, infile)

#apply testData to the tree
testLabels =  applyTree(tree, testInput)

#metrics calculate
trainError = metricsCalculator(infile, trainLabels)
testError = metricsCalculator(testInput, testLabels)

# write file

listToFile(trainOut, trainLabels)
listToFile(testOut, testLabels)

outTxt = open(metricsOut, "wb")
outTxt.write("error(train): &f" %trainError)
outTxt.write("error(test): &f" %testError)

train = infile

print("test label output", trainLabels)
print("xxxx xxxxx xxxxxx", dataCleanReverse(testInput)[1])

title = train.splitlines()[0].split(",")
print("\n\n", "===================Trained Tree=====================", "\n")
printTree(tree, infile, title)

print("\n\nerror(train)",trainError,"\nerror(test)", testError)
"""


if __name__ == '__main__':
    infile = sys.argv[1] #train input
    infile = readFile(infile)


    testInput = sys.argv[2]
    testInput = readFile(testInput)
    #op = open(testInput, "rb")
    #testInput = op.readlines()
    #outfile = sys.argv[2]
    maxDepth = int(sys.argv[3])
    trainOut = sys.argv[4]
    testOut = sys.argv[5]
    metricsOut = sys.argv[6]




    # main part


    #Run train
    tree = runTrain(infile, maxDepth)

    #apply trainData to the tree
    trainLabels = applyTree(tree, infile)

    #apply testData to the tree
    testLabels =  applyTree(tree, testInput)

    #metrics calculate
    trainError = metricsCalculator(infile, trainLabels)
    testError = metricsCalculator(testInput, testLabels)

    # write file

    listToFile(trainOut, trainLabels)
    listToFile(testOut, testLabels)

    outTxt = open(metricsOut, "wb")
    outTxt.write("error(train): %f\n" %trainError)
    outTxt.write("error(test): %f\n" %testError)

    train = infile
    train = train.splitlines()
    title = train[0].split(",")

    printTree(tree, infile, title)


    print("error(train)", trainError, end="")
    print("\n")
    print("error(test", testError, end="")
    print("\n")
