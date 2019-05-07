from environment import MountainCar
import numpy as np
import random
import sys
import pdb
import copy

def main(args):
    pass

def updateQ(Q ,learningRate, gamma, reward,   ):
    firstPart = (1-learningRate)*Q[s][a]
    secondPart = learningRate * (reward + gamma * maxQ(s,a) )
    result = firstPart + secondPart

    Q[s][a] = result
    return None

#has state in the dict
def qsaw(state,action,w,bias):
    result = 0.0
    for i in state:
        if state[i] != 0:
            result += state[i]* w[i][action]
    result += bias

    return result

def maxQsaw(state,w,bias):
    result = qsaw(state,0,w,bias)
    for ac in [0,1,2]:
        guess = qsaw(state,ac,w,bias)
        if guess > result:
            result = guess
    return result

def updataW(w, learningRate, state,b, gamma, reward, nextState, mode):
    if mode=="raw":
        q = qsaw(state, w, b)
        q2 = qsaw(nextState,w,b)

    elif mode=="tile":
        for i in state:

            q = qsaw(state[i],w[i],b)
            q2 = maxQsaw(nextState,w[i],b)
    firstPart = learningRate*(q - (reward+ gamma*q2))
    secondPart = state
    result = w - firstPart*secondPart

    return result


def pickAction(epsilon, state, w, bias):
    action = None
    r = random.random()
    if r < epsilon:
        action = random.randint(0,2)


    else:
        max = qsaw(state, 0, w, bias)
        action = 0
        for i in [0,1,2]:
            guess = qsaw(state, i, w, bias)

            if guess > max:
                max = guess
                action = i
    return action

def oneDListToFile(outfile, list):
    outTxt = open(outfile, "wb")

    for i in list:
        i = str(i) + "\n"
        outTxt.write(i)

def twoDListToFile(outfile, list, bias):
    outTxt = open(outfile, "wb")
    t=bias
    t = str(t) + "\n"
    outTxt.write(t)
    for i in list:

        for j in i:
            j = str(j) + "\n"
            outTxt.write(j)

if __name__ == "__main__":
    #main(sys.argv)
    print("code start")
    # command line
    # formatted input

    modeInput = sys.argv[1]  # #raw or tile
    #trainInput = readFile(trainInput)

    weightOut = sys.argv[2]
    #indexToWord = readFile(indexToWord)

    returnsOut = sys.argv[3]
    #indexToTag = readFile(indexToTag)


    episodes = sys.argv[4]
    #maximum of the length of an episode
    maxIterations = sys.argv[5]
    #value of epsilon-greedy
    epsilon = sys.argv[6]
    #discount factor
    gamma = sys.argv[7]
    #alpha
    learningRate = sys.argv[8]

    #######################################################################################
    #main

    MC = MountainCar(mode=modeInput)
    totalReward = []
    w = np.zeros((MC.state_space, 3))
    bias = 0
    gamma = float(gamma)
    learningRate = float(learningRate)

    for i in range(int(episodes)):
        sumReward = 0
        state = MC.reset()
        done = False
        iteration = 1
        while done==False and iteration <= int(maxIterations):
            action = pickAction(float(epsilon), state, w, bias)
            #print("action",action)

            nextState, reward, done = MC.step(action)


            #w = updataW(w, learningRate, MC.state, action, gamma, reward, nextState, modeInput, bias)
            # Update Weight
            q = 0.0
            q2 = 0.0
            #print(state,nextState)
            if modeInput == "raw":
                q = qsaw(state, action, w, bias)
                q2 = maxQsaw(nextState, w, bias)

            elif modeInput == "tile":

                q += qsaw(state,action, w, bias)
                q2 += maxQsaw(nextState, w, bias)

            #print(q,q2)
            firstPart = learningRate * (q - (reward + gamma * q2))
            #print(firstPart)
            gradient = 0.0

            #print(gradient)
            for i in state:
                gradient=state[i]*firstPart
                w[i][action] = w[i][action] - gradient
            bias = bias - firstPart



            #for next run
            state = copy.deepcopy(nextState)
            iteration += 1






            sumReward += reward

        totalReward.append(sumReward)

    #print(w)




    oneDListToFile(returnsOut, totalReward)
    twoDListToFile(weightOut, w, bias)








