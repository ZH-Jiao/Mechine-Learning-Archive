from __future__ import print_function
import sys
import copy
import numpy as np

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


def featureEngineering(input, dictionary, flag):
    if int(flag) == 1:
        return featureModel1(input, dictionary)
    elif int(flag) == 2:
        return featureModel2(input, dictionary)

def featureModel1(input, dictionary):

    input = input.splitlines()




    output = list()
    #a list of review
    #output = [[1, {2103:1, 112:1, ...}], [0,{...}], ...]

    for line in input:

        line = line.split("\t")

        review = [0] * 2

        review[1] = dict()

        review[0] = line[0]


        text = []
        text = line[1].split()

        for word in text:
            if word in dictionary:
                review[1][dictionary[word]] = 1

        output.append(review)

    return output

def featureModel2(input, dictionary, threshold = 4):
    input = input.splitlines()




    output = list()
    #a list of review
    #output = [[1, {2103:1, 112:1, ...}], [0,{...}], ...]

    for line in input:

        line = line.split("\t")

        review = [0] * 2

        review[1] = dict()

        review[0] = line[0]


        text = []
        text = line[1].split()

        for word in text:
            if word in dictionary:
                review[1][dictionary[word]] = review[1].get(dictionary[word],0) + 1

        delList = []
        for word in review[1]:
            if review[1][word] >= 4:
                delList.append(word)
            else:
                review[1][word] = 1
        for i in delList:
            del review[1][i]


        output.append(review)


    return output

def dictToTsv(outFile, dictionary):
    #dict = [[1, {2103: 1, 112: 1, ...}], [0, {...}], ...]
    outTxt = open(outFile, "wb")

    for line in dictionary:
        outTxt.write(line[0])
        for j in line[1]:
            text = "\t" + str(j) + ":" + str(line[1][j])
            outTxt.write(text)
        outTxt.write("\n")


def listToFile(outfile, list):
    outTxt = open(outfile, "wb")


    for i in list:
        i = i+"\n"
        outTxt.write(i)


if __name__ == '__main__':
    #command line
    trainInput = sys.argv[1] #train input
    trainInput = readFile(trainInput)

    validationInput = sys.argv[2]
    validationInput = readFile(validationInput)


    testInput = sys.argv[3]
    testInput = readFile(testInput)

    dictInput = sys.argv[4]
    dictInput = readFile(dictInput)

    formattedTrainOut = sys.argv[5]

    formattedValidationOut = sys.argv[6]

    formattedTestOut = sys.argv[7]

    featureFlag = sys.argv[8]

    #run

    dictionary = txtToDict(dictInput)

    trainVector = featureEngineering(trainInput, dictionary, featureFlag)
    testVector = featureEngineering(testInput, dictionary, featureFlag)
    validationVector = featureEngineering(validationInput, dictionary, featureFlag)


    dictToTsv(formattedTrainOut, trainVector)
    dictToTsv(formattedTestOut, testVector)
    dictToTsv(formattedValidationOut, validationVector)

