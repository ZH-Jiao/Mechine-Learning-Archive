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
    #still string
    return output


#Matrix alpha
def alpha(sentence, PI, A, B):
    #alpha is row: J, col: T
    #Length A is the number of tags
    Alpha = np.zeros((len(sentence),len(A)))
    for row in range(len(Alpha)):
        for col in range(len(Alpha[row])):
            #print("Alpha")
            #print(Alpha)
            Alpha[row][col] = singleAlpha(sentence, Alpha, row, col , PI, A, B)

    return Alpha

#single alpha t,j
def singleAlpha(sentence, Alpha, t, j, PI, A, B):

    if t == 0:
        # sentence[t] is xt(index)
        result = PI[j] * B[j][sentence[t]]
        #print("t=0",t,j,result)
        return result

    elif t > 0:
        sum = 0.0
        result = 0.0
        #HERE################################################################
        #sum among all possible tags
        for k in range(len(A)):
            #dynamic programming
            if Alpha[t-1, k] != 0:
                sum = Alpha[t-1, k] * A[k][j]
            else:
                sum = singleAlpha(sentence, Alpha, t-1, k, PI, A, B) * A[k][j]
            #print("sum",sum)
            result += sum
        result = B[j][sentence[t]] * result
        #print("t>0",t, j, result)

        return result


#Matrix beta
def beta(sentence, PI, A, B):
    #beta is row:J, col:T
    Beta = np.zeros((len(sentence), len(A)))
    #reversed
    for row in range(len(Beta))[::-1]:
        for col in range(len(Beta[row])):
            Beta[row][col] = singleBeta(sentence, Beta, row, col, PI, A, B)

    return Beta

#single beta t,j
def singleBeta(sentence, Beta, t, j, PI, A, B):
    T = len(B[0])-1
    if t == T:
        result = 1.0
        #print("t=0", t, j, result)
        return result

    elif t < T:
        sum = 0.0
        result = 0.0
        for k in range(len(A)):
            #DP
            if Beta[t+1][k] != 0:
                sum = Beta[t+1][k] * A[j][k] * B[k][sentence[t+1]]
            else:
                sum = singleBeta(sentence, Beta, t + 1, k, PI, A, B) * A[j][k] * B[k][sentence[t+1]]
            #print("sum", sum)
            result += sum

        #print("t>0", t, j, result)

        return result

#alpha * beta


###########################################################################
#Shared with learnhmm
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

"""
def wordToIndex(indexToWord, input):
    #indexToword is splitlined
    #print(input)
    #print(indexToWord)
    result =[]
    for row in range(len(input)):
        result.append([])
        for col in range(len(input[row])):
            wordIndex = indexToWord.index(input[row][col][0])
            result[-1].append(wordIndex)
    return listToArray(result)
"""
# make wordIndex in different length, list
def wordToIndex(indexToWord, input):
    #indexToword is splitlined
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
######################################################################################
# Main
def predict(wordIndex, PI, A, B):
    tagIndex = np.full((wordIndex.shape), -1)

    # for each sentence (example)
    for row in range(len(wordIndex)):
        # NOW EACH WORDINDEX ROW HAS DIFFERENT LENGTH
        Alpha = alpha(wordIndex[row], PI, A, B)
        Beta = beta(wordIndex[row], PI, A, B)


        for col in range(len(wordIndex[row])):
            #print(wordIndex)
            #print(Alpha)
            #print(Beta)
            conditionalProb = Alpha[wordIndex[row][col]] * Beta[wordIndex[row][col]]
            #print("CondiProb")
            #print(conditionalProb)
            tagIndex[row][col] = np.argmax(conditionalProb)
    # empty word is occupied by unknown value
    print(tagIndex)
    return tagIndex

#######################################################################################
#File IO
def threeDListToFile(outfile, list):
    outTxt = open(outfile, "wb")

    for i in list:
        lst = ""
        for j in i:
            lst += str(j[0]) + "_" + str(j[1]) + " "

        lst = lst[:-1] + "\n"
        outTxt.write(lst)


def predictToFile(predictTag, testInput, indexToTag):
    result = copy.deepcopy(testInput)
    for row in range(len(testInput)):
        for col in range(len(testInput[row])):

            result[row][col][1] = indexToTag[predictTag[row][col]]
    #result[][] is [[word,tag]]
    return result





if __name__ == '__main__':
    print("code start")
    # command line
    # formatted input
    testInput = sys.argv[1]
    testInput = readFile(testInput)

    indexToWord = sys.argv[2]
    indexToWord = readFile(indexToWord)

    indexToTag = sys.argv[3]
    indexToTag = readFile(indexToTag)


    hmmprior = sys.argv[4]
    hmmprior = readFile(hmmprior)

    hmmemit = sys.argv[5]
    hmmemit = readFile(hmmemit)

    hmmtrans = sys.argv[6]
    hmmtrans = readFile(hmmtrans)

    predictedFile = sys.argv[7]

    metricFile = sys.argv[8]

    print("input finish")

    # run
    print("start")
    testInput = textToList(testInput)
    testInput = inputListToList(testInput)

    indexToWord = indexToWord.splitlines()
    indexToTag = indexToTag.splitlines()

    wordIndex = wordToIndex(indexToWord,testInput)
    #tagIndex not useful here
    tagIndex = tagToIndex(indexToTag,testInput)




    Pi = np.array(textToList(hmmprior)).astype(np.float)
    A = np.array(textToList(hmmtrans)).astype(np.float)
    B = np.array(textToList(hmmemit)).astype(np.float)

    print("pi",Pi)
    print("A",A)
    print("B",B)

    predictTag = predict(wordIndex, Pi, A, B)

    output = predictToFile(predictTag, testInput, indexToTag)
    threeDListToFile(predictedFile, output)

    #oneDListToFile(hmmprior, Pi)
    #twoDListToFile(hmmtrans, A)
    #twoDListToFile(hmmemit, B)


