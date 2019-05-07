from __future__ import print_function
import sys
import copy
import numpy as np
import math



def readFile(path):
    with open(path, "rt") as f:
        return f.read()


def textToList(text):
    text = text.splitlines()
    # print(type(text))
    # print(text)
    print("test to list start")
    # y is the output label
    output = []
    for i in text:
        i = i.split(" ")
        output.append(i)

    return output

#transform [<word>_<tag>] to [word],[tag]
def inputListToList(sample):
    result = []
    for row in sample:
        result.append([])
        for col in row:
            col = col.split("_")
            #col = [[word][tag]]
            result[-1].append(col)

    #result = [[word][tag],  ]
    return result

def wordToIndex(indexToWord, input):
    #indexToword is splitlined
    #print(indexToWord)
    result =[]
    for row in range(len(input)):
        result.append([])
        for col in range(len(input[row])):
            wordIndex = indexToWord.index(input[row][col][0])
            result[-1].append(wordIndex)
    return result

def tagToIndex(indexToTag, input):
    # indexToword is splitlined
    result = []
    for row in range(len(input)):
        result.append([])
        for col in range(len(input[row])):
            tagIndex = indexToTag.index(input[row][col][1])
            result[-1].append(tagIndex)

    return result

def listToArray(lst):
    x = len(lst)
    #y = len(lst[0])
    #find max length of sentence
    y = 0
    for i in range(len(lst)):
        if len(lst[i]) > y:
            y = len(lst[i])

    result=np.full((x,y),-1)
    #print(result.shape)
    #print(len(lst))
    for row in range(len(lst)):
        #print(len(lst[row]))
        for col in range(len(lst[row])):
            result[row][col] = lst[row][col]
    return result

##############################
#start
def tagToPrior(indexToTag, tagIndex):
    #indexToTag is splitline: [A,B]
    #tagIndex is training sample of tags
    #count # of the relationships
    PICount = np.zeros(len(indexToTag))
    for i in range(len(tagIndex)):

        PICount[tagIndex[i][0]] += 1.0
    PI = np.zeros(PICount.shape)

    # calculate the Prob.
    for i in range((len(PI))):
        PI[i] = ( PICount[i]+1.0 ) / ( PICount.sum() + len(PICount) )

    return PI



def tagToA(indexToTag, tagIndex):
    ACount = np.zeros((len(indexToTag), len(indexToTag)))
    # count # of the relationships
    for row in range((len(tagIndex))):
        for col in range((len(tagIndex[row])-1)):
            if tagIndex[row][col] != -1 and tagIndex[row][col+1] != -1:
                ACount[tagIndex[row][col]][tagIndex[row][col+1]] += 1.0
    A = np.zeros(ACount.shape)
    #print(ACount)

    # calculate the Prob.
    for row in range((len(ACount))):
        for col in range((len(ACount[row]))):
            A[row][col] = ( ACount[row][col]+1.0 ) / ( ACount[row].sum() + len(ACount[row]) )

    return A

def tagToB(indexToword, wordIndex, indexToTag, tagIndex):
    BCount = np.zeros((len(indexToTag),len(indexToWord)))
    # count # of the relationships
    for row in range((len(tagIndex))):
        for col in range((len(tagIndex[row]))):
            if tagIndex[row][col] != -1 and wordIndex[row][col] != -1:
                BCount[tagIndex[row][col]][wordIndex[row][col]] += 1.0
    B = np.zeros(BCount.shape)
    #print("BCOUNT",BCount)
    # calculate the Prob.
    for row in range((len(BCount))):
        for col in range((len(BCount[row]))):
            B[row][col] = (BCount[row][col] + 1.0) / (BCount[row].sum() + len(BCount[row]))

    return B













def oneDListToFile(outfile, list):
    outTxt = open(outfile, "wb")

    for i in list:
        i = str(i) + "\n"
        outTxt.write(i)

def twoDListToFile(outfile, list):
    outTxt = open(outfile, "wb")

    for i in list:
        lst = ""
        for j in i:
            lst += str(j) + " "
        lst = lst[:-1] + "\n"
        outTxt.write(lst)


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
    # command line
    # formatted input
    trainInput = sys.argv[1]  # train input
    trainInput = readFile(trainInput)

    indexToWord = sys.argv[2]
    indexToWord = readFile(indexToWord)

    indexToTag = sys.argv[3]
    indexToTag = readFile(indexToTag)


    hmmprior = sys.argv[4]

    hmmemit = sys.argv[5]

    hmmtrans = sys.argv[6]

    print("input finish")

    # run
    print("start")
    trainInput = textToList(trainInput)

    trainInput = inputListToList(trainInput)


    indexToWord = indexToWord.splitlines()
    indexToTag = indexToTag.splitlines()

    wordIndex = wordToIndex(indexToWord,trainInput)
    tagIndex = tagToIndex(indexToTag,trainInput)


    #print(wordIndex)

    #print(tagIndex)

    Pi = tagToPrior(indexToTag,tagIndex)
    A = tagToA(indexToTag,tagIndex)
    B = tagToB(indexToWord,wordIndex,indexToTag,tagIndex)

    print("pi",Pi)
    print("A",A)
    print("B",B)

    oneDListToFile(hmmprior, Pi)
    twoDListToFile(hmmtrans, A)
    twoDListToFile(hmmemit, B)







    """

    listToFile(trainOut, trainLabel)
    listToFile(testOut, testLabel)
    # print(metric)
    listToFile(metricsOut, metric)
    # outTxt = open(metricsOut, "wb")
    # outTxt.write(metric)
    """

