from __future__ import print_function
import sys
import sys
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


"""
#Test Code
t = readFile("C:/Users/Zhiheng/OneDrive/CMU/CourseFile/10601 Machine Learning/HW2/handout (2)/handout/small_train.csv")
t = t.splitlines()



text = t[1:]
labels = []
for i in text:
    labels.append(i.split(",")[-1])

print(entropy(labels))
print(errorRate(labels))

print(labels)
print(type(t))
"""

if __name__ == '__main__':
    infile = sys.argv[1]
    op = open(infile, "rb")
    text = op.readlines()

    # main part
    text = text[1:]
    labels = []
    for i in text:
        labels.append(i.split(",")[-1])

    # write file
    outfile = sys.argv[2]
    outTxt = open(outfile, "wb")

    # entropy: 0.996316519559
    # error: 0.464285714286

    outTxt.write("entropy: %f" % entropy(labels))
    outTxt.write("\n")
    outTxt.write("error: %f" % errorRate(labels))

    print('The input file is: %s' % (infile))
    print('The output file is: %s' % (outfile))

