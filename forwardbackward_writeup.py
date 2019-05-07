from __future__ import print_function
import sys
import copy
import numpy as np
import math


a = np.array([[0.25, 0.75],[0.6, 0.4]])
b = np.array([[1.0/6, 2.0/3, 1.0/6], [4.0/8, 1.0/8, 3.0/8]])
pi = np.array([[0.4],[0.6]])






def alpha(t, j):

    if t == 0:
        result = pi[j] * b[j][t]
        print("t=0",t,j,result)
        return result

    elif t > 0:
        sum = 0.0
        result = 0.0
        for k in range(len(b)):
            sum = alpha(t-1, k) * a[k][j]
            print("sum",sum)
            result += sum
        result = b[j][t] * result
        print("t>0",t, j, result)

        return result


def beta(t, j):
    T = len(b[0])-1
    if t == T:
        result = 1.0
        print("t=0", t, j, result)
        return result

    elif t < T:
        sum = 0.0
        result = 0.0
        for k in range(len(b)):
            sum = beta(t + 1, k) * a[j][k] *b[k][t+1]
            print("sum", sum)
            result += sum

        print("t>0", t, j, result)

        return result

def logLikelihoodInSequence(T):
    sum = 0.0
    for j in range(len(b)):
        sum += alpha(T,j)
    result = np.log(sum)

    return result



print(a)
print(b)
#print(alpha(2,0))
#print("beta",beta(2,1))
#print("alpha", alpha(2,1))

print(logLikelihoodInSequence(2))

